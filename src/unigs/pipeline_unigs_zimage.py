# Copyright 2024 The HuggingFace Team and UniGS authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UniGS pipeline on Z-Image Turbo (native omni context-token concat)."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.utils import logging

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
except ImportError:
    from diffusers import DiffusionPipeline

from .backbones import ZIMAGE_CHECKPOINT, default_dit_guidance, default_dit_max_sequence_length
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .dit import encode_zimage_prompt, forward_zimage_omni
from .pipeline_unigs import PipelineImageInput, UniGSPipelineOutput, _as_mask_tensor, _as_pil_rgb, retrieve_timesteps
from .pipeline_unigs_common import (
    decode_dual_outputs,
    prefix_task_prompt,
    prepare_control_and_mask_latents,
    prepare_noisy_dual_latents,
    prepare_task_condition,
)
from .transformer import adapt_unigs_transformer, calculate_shift


logger = logging.get_logger(__name__)

DEFAULT_ZIMAGE_CHECKPOINT = ZIMAGE_CHECKPOINT


def _import_zimage():
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    except ImportError as err:
        raise ImportError(
            "Z-Image UniGS requires a recent Diffusers with `ZImageTransformer2DModel` "
            "(install from source if your wheel is older than the Z-Image release)."
        ) from err
    return ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler


class UniGSZImagePipeline(DiffusionPipeline):
    """UniGS on [`ZImageTransformer2DModel`] (Z-Image Turbo).

    Control + coarse mask are clean omni context streams. Image and colormap
    are stacked on height as the noisy target (omni unpatchify returns the last
    stream only). Transformer output is negated like `ZImagePipeline`.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae: AutoencoderKL,
        transformer,
        tokenizer,
        text_encoder,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker, unigs_dit_family="z_image")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", None) or 16)
        vae_scale = self.vae_scale_factor * 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale, vae_latent_channels=self.latent_channels)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale,
            vae_latent_channels=self.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.default_sample_size = 64
        self.palette = LocationAwarePalette()
        self.pdm = ProgressiveDichotomyModule()

    @classmethod
    def from_zimage(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        scheduler=None,
        **kwargs,
    ) -> "UniGSZImagePipeline":
        """Load Z-Image Turbo and adapt it to UniGS omni context-token concat."""
        ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler = _import_zimage()
        pretrained_model_name_or_path = pretrained_model_name_or_path or DEFAULT_ZIMAGE_CHECKPOINT
        load_kw = dict(revision=revision, variant=variant, torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", **load_kw)
        transformer = ZImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", **load_kw
        )
        transformer = adapt_unigs_transformer(transformer, family="z_image")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
        )
        try:
            from transformers import Qwen2Model

            text_encoder = Qwen2Model.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", **load_kw
            )
        except Exception:
            text_encoder = AutoModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", **load_kw
            )
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        return cls(
            vae=vae,
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            **kwargs,
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
    ):
        device = device or self._execution_device
        max_sequence_length = max_sequence_length or default_dit_max_sequence_length("z_image")
        prompt_embeds = encode_zimage_prompt(
            self.text_encoder, self.tokenizer, prompt, device, max_sequence_length=max_sequence_length
        )
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
        negative = None
        if do_classifier_free_guidance:
            negative = encode_zimage_prompt(
                self.text_encoder,
                self.tokenizer,
                negative_prompt or "",
                device,
                max_sequence_length=max_sequence_length,
            )
            if num_images_per_prompt > 1:
                negative = [pe for pe in negative for _ in range(num_images_per_prompt)]
        return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative}

    def _latent_hw(self, height: int, width: int) -> tuple[int, int]:
        return (
            2 * (int(height) // (self.vae_scale_factor * 2)),
            2 * (int(width) // (self.vae_scale_factor * 2)),
        )

    def _predict(self, image, colormap, control, mask, timestep, encoded, guidance_scale: float):
        do_cfg = encoded.get("negative_prompt_embeds") is not None and guidance_scale is not None and guidance_scale > 0
        if do_cfg:
            image = torch.cat([image, image], dim=0)
            colormap = torch.cat([colormap, colormap], dim=0)
            control = torch.cat([control, control], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            caps = list(encoded["prompt_embeds"]) + list(encoded["negative_prompt_embeds"])
            timestep = timestep.repeat(2)
            pred = forward_zimage_omni(
                self.transformer, image, colormap, control, mask, timestep, caps, negate=True
            )[0]
            cond, uncond = pred.chunk(2)
            return uncond + guidance_scale * (cond - uncond)
        return forward_zimage_omni(
            self.transformer, image, colormap, control, mask, timestep, encoded["prompt_embeds"], negate=True
        )[0]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[PipelineImageInput] = None,
        colormap: Optional[PipelineImageInput] = None,
        task: str = "inpainting",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        max_sequence_length: Optional[int] = None,
        decode_masks: bool = True,
        pdm_delta: Optional[float] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Union[UniGSPipelineOutput, tuple]:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if guidance_scale is None:
            guidance_scale = default_dit_guidance("z_image")
        if prompt is None:
            raise ValueError("Provide `prompt`.")
        prompt = prefix_task_prompt(prompt, task)
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 0
        encoded = self.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
        )
        dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else next(self.transformer.parameters()).dtype
        init_image = (
            self.image_processor.preprocess(_as_pil_rgb(image), height=height, width=width).to(device=device, dtype=dtype)
            if image is not None
            else None
        )
        colormap_tensor = (
            self.image_processor.preprocess(_as_pil_rgb(colormap), height=height, width=width).to(
                device=device, dtype=dtype
            )
            if colormap is not None
            else None
        )
        coarse_mask = (
            _as_mask_tensor(mask_image, height, width, device, dtype)
            if mask_image is not None
            else torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        )
        control_image, coarse_mask = prepare_task_condition(task, init_image, colormap_tensor, coarse_mask)
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()
        latent_h, latent_w = self._latent_hw(height, width)
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        try:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu
            )
        except TypeError:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )
        image_latents, colormap_latents = prepare_noisy_dual_latents(
            batch_size * num_images_per_prompt,
            self.latent_channels,
            latent_h,
            latent_w,
            dtype,
            device,
            generator,
            latents,
        )
        control_latents, mask_latents = prepare_control_and_mask_latents(
            self.vae,
            control_image,
            coarse_mask,
            batch_size * num_images_per_prompt,
            latent_h,
            latent_w,
            self.latent_channels,
            dtype,
            device,
            generator,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = (1000 - t.expand(image_latents.shape[0])) / 1000
                noise_pred = self._predict(
                    image_latents, colormap_latents, control_latents, mask_latents, timestep, encoded, guidance_scale
                )
                target = torch.cat([image_latents, colormap_latents], dim=1)
                target = self.scheduler.step(noise_pred, t, target, return_dict=False)[0]
                image_latents, colormap_latents = target.chunk(2, dim=1)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, target)
        result = decode_dual_outputs(
            self.vae,
            self.image_processor,
            image_latents,
            colormap_latents,
            output_type,
            decode_masks,
            self.pdm,
            task,
            pdm_delta,
            return_dict,
        )
        self.maybe_free_model_hooks()
        return result

    def inpaint(self, prompt, image, mask_image, **kwargs):
        return self(prompt=prompt, image=image, mask_image=mask_image, task="inpainting", **kwargs)

    def synthesize(self, prompt, colormap, **kwargs):
        return self(prompt=prompt, colormap=colormap, task="synthesis", **kwargs)

    def referring(self, prompt, image, mask_image=None, **kwargs):
        return self(prompt=prompt, image=image, mask_image=mask_image, task="referring", **kwargs)

    def segment(self, image, prompt: str = "panoptic: all entities.", **kwargs):
        return self(prompt=prompt, image=image, task="entity", **kwargs)
