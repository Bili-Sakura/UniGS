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

"""UniGS pipeline on Stable Diffusion 3.5 Medium (context-token concat)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.utils import logging

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
except ImportError:
    from diffusers import DiffusionPipeline

from .backbones import SD3_CHECKPOINT, default_dit_guidance, default_dit_max_sequence_length
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .dit import encode_sd3_prompt, forward_sd3_token_concat
from .pipeline_unigs import PipelineImageInput, UniGSPipelineOutput, _as_mask_tensor, _as_pil_rgb, retrieve_timesteps
from .pipeline_unigs_common import (
    decode_dual_outputs,
    prefix_task_prompt,
    prepare_control_and_mask_latents,
    prepare_noisy_dual_latents,
    prepare_task_condition,
)
from .transformer import adapt_unigs_transformer


logger = logging.get_logger(__name__)

DEFAULT_SD3_CHECKPOINT = SD3_CHECKPOINT


def _import_sd3():
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
        from transformers import CLIPTextModelWithProjection, T5EncoderModel
    except ImportError as err:
        raise ImportError(
            "SD 3.5 UniGS requires Diffusers with `SD3Transformer2DModel` and "
            "transformers `CLIPTextModelWithProjection` / `T5EncoderModel`."
        ) from err
    return SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, CLIPTextModelWithProjection, T5EncoderModel


class UniGSSD3Pipeline(DiffusionPipeline):
    """UniGS on [`SD3Transformer2DModel`] (Stable Diffusion 3.5 Medium).

    Image and colormap latents are denoised jointly. The coarse mask and control
    latent are extra context tokens concatenated after patch embed.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = ["tokenizer_3", "text_encoder_3", "safety_checker", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae: AutoencoderKL,
        transformer,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        tokenizer_3=None,
        text_encoder_3=None,
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
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            tokenizer_3=tokenizer_3,
            text_encoder_3=text_encoder_3,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker, unigs_dit_family="sd3")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", None) or 16)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.default_sample_size = 64
        self.palette = LocationAwarePalette()
        self.pdm = ProgressiveDichotomyModule()

    @classmethod
    def from_sd3(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        scheduler=None,
        **kwargs,
    ) -> "UniGSSD3Pipeline":
        """Load SD 3.5 Medium and adapt it to UniGS context-token concat."""
        SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, CLIPTextModelWithProjection, T5EncoderModel = _import_sd3()
        pretrained_model_name_or_path = pretrained_model_name_or_path or DEFAULT_SD3_CHECKPOINT
        load_kw = dict(revision=revision, variant=variant, torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", **load_kw)
        transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer", **load_kw
        )
        transformer = adapt_unigs_transformer(transformer, family="sd3")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2", revision=revision
        )
        try:
            tokenizer_3 = T5TokenizerFast.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer_3", revision=revision
            )
        except Exception:
            tokenizer_3 = None
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", **load_kw
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2", **load_kw
        )
        try:
            text_encoder_3 = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder_3", **load_kw
            )
        except Exception:
            text_encoder_3 = None
            tokenizer_3 = None
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        return cls(
            vae=vae,
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            tokenizer_3=tokenizer_3,
            text_encoder_3=text_encoder_3,
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
        max_sequence_length = max_sequence_length or default_dit_max_sequence_length("sd3")
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(prompt, list) and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        joint_dim = int(getattr(self.transformer.config, "joint_attention_dim", 4096))
        prompt_embeds, pooled = encode_sd3_prompt(
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
            self.tokenizer,
            self.tokenizer_2,
            self.tokenizer_3,
            prompt,
            device,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            joint_attention_dim=joint_dim,
        )
        negative = negative_pooled = None
        if do_classifier_free_guidance:
            negative, negative_pooled = encode_sd3_prompt(
                self.text_encoder,
                self.text_encoder_2,
                self.text_encoder_3,
                self.tokenizer,
                self.tokenizer_2,
                self.tokenizer_3,
                negative_prompt or "",
                device,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                joint_attention_dim=joint_dim,
            )
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": negative,
            "negative_pooled_prompt_embeds": negative_pooled,
        }

    def _predict(self, image, colormap, control, mask, timestep, encoded, guidance_scale: float):
        do_cfg = encoded.get("negative_prompt_embeds") is not None and guidance_scale is not None and guidance_scale > 1
        if do_cfg:
            image = torch.cat([image, image], dim=0)
            colormap = torch.cat([colormap, colormap], dim=0)
            control = torch.cat([control, control], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            prompt_embeds = torch.cat([encoded["negative_prompt_embeds"], encoded["prompt_embeds"]], dim=0)
            pooled = torch.cat([encoded["negative_pooled_prompt_embeds"], encoded["pooled_prompt_embeds"]], dim=0)
            timestep = timestep.repeat(2)
            pred = forward_sd3_token_concat(
                self.transformer, image, colormap, control, mask, timestep, prompt_embeds, pooled
            )[0]
            uncond, cond = pred.chunk(2)
            return uncond + guidance_scale * (cond - uncond)
        return forward_sd3_token_concat(
            self.transformer,
            image,
            colormap,
            control,
            mask,
            timestep,
            encoded["prompt_embeds"],
            encoded["pooled_prompt_embeds"],
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
            guidance_scale = default_dit_guidance("sd3")
        if prompt is None:
            raise ValueError("Provide `prompt`.")
        prompt = prefix_task_prompt(prompt, task)
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1
        encoded = self.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
        )
        dtype = encoded["prompt_embeds"].dtype
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
        try:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas
            )
        except TypeError:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )
        latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor
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
                timestep = t.expand(image_latents.shape[0])
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
