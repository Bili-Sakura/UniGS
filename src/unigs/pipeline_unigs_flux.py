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

"""UniGS inference pipeline on a FLUX.1-Fill-dev DiT backbone.

Conditioning follows FLUX.1-Fill-dev: packed streams are concatenated on the
**channel** axis (last dim), not as extra sequence tokens::

    hidden = cat(img_64, cmap_64, control_64, mask_256, dim=-1)  # 448

The transformer jointly denoises packed image + colormap (``[B, S, 128]``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
except ImportError:
    from diffusers import DiffusionPipeline

try:
    from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
except ImportError:  # pragma: no cover - older Diffusers without FLUX
    FluxTransformer2DModel = None  # type: ignore[misc, assignment]
    FlowMatchEulerDiscreteScheduler = None  # type: ignore[misc, assignment]

from .backbones import FLUX_LATENT_CHANNELS
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .pipeline_unigs import (
    PipelineImageInput,
    UniGSPipelineOutput,
    _as_mask_tensor,
    _as_pil_rgb,
    retrieve_timesteps,
)
from .prompts import TASK_PROMPT_TEMPLATES, build_task_prompt
from .transformer import (
    adapt_unigs_transformer,
    calculate_shift,
    concat_fill_channels,
    decode_vae_latents,
    encode_vae_latents,
    pack_fill_mask,
    pack_latents,
    prepare_latent_image_ids,
    split_packed_pred,
    unpack_latents,
)


logger = logging.get_logger(__name__)


def encode_flux_prompt(
    text_encoder: CLIPTextModel,
    text_encoder_2: T5EncoderModel,
    tokenizer: CLIPTokenizer,
    tokenizer_2: T5TokenizerFast,
    prompt: Union[str, List[str]],
    device: torch.device,
    max_sequence_length: int = 512,
    num_images_per_prompt: int = 1,
):
    """CLIP pooled + T5 sequence embeddings, matching Diffusers Flux pipelines."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompts)

    clip_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    clip_out = text_encoder(clip_inputs.input_ids.to(device), output_hidden_states=False)
    pooled = clip_out.pooler_output.to(dtype=text_encoder.dtype, device=device)
    pooled = pooled.repeat(1, num_images_per_prompt)
    pooled = pooled.view(batch_size * num_images_per_prompt, -1)

    t5_inputs = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder_2(t5_inputs.input_ids.to(device), output_hidden_states=False)[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled, text_ids


class UniGSFluxPipeline(DiffusionPipeline):
    r"""
    UniGS on [`FluxTransformer2DModel`] (FLUX.1-Fill-dev).

    Packed image and colormap tokens are denoised jointly. The coarse mask and
    control latent are concatenated on the packed **channel** axis, matching
    `FluxFillPipeline` (plus a noisy colormap stream).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        if FluxTransformer2DModel is None:
            raise ImportError(
                "UniGSFluxPipeline requires Diffusers with FLUX support "
                "(`FluxTransformer2DModel`). Install `diffusers>=0.32.0`."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = getattr(self.vae.config, "latent_channels", None) or FLUX_LATENT_CHANNELS
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.latent_channels,
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            vae_latent_channels=self.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 77
        )
        self.default_sample_size = 64  # 64 * vae_scale_factor = 512, matching UniGS
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.palette = LocationAwarePalette()
        self.pdm = ProgressiveDichotomyModule()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a UniGS FLUX pipeline or adapt FLUX.1-Fill-dev (Diffusers contract)."""
        if FluxTransformer2DModel is None or FlowMatchEulerDiscreteScheduler is None:
            raise ImportError(
                "FLUX UniGS requires Diffusers with `FluxTransformer2DModel` "
                "and `FlowMatchEulerDiscreteScheduler` (diffusers>=0.32.0)."
            )
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(pipeline, "transformer", None) is not None:
            pipeline.transformer = adapt_unigs_transformer(pipeline.transformer, family="flux")
        return pipeline

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        device = device or self._execution_device
        if prompt_embeds is not None and pooled_prompt_embeds is not None:
            text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
            return prompt_embeds.to(device=device), pooled_prompt_embeds.to(device=device), text_ids
        return encode_flux_prompt(
            self.text_encoder,
            self.text_encoder_2,
            self.tokenizer,
            self.tokenizer_2,
            prompt,
            device,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )

    def _latent_hw(self, height: int, width: int) -> tuple[int, int]:
        latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
        return latent_h, latent_w

    def prepare_packed_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_h, latent_w = self._latent_hw(height, width)
        shape = (batch_size, self.latent_channels, latent_h, latent_w)
        if latents is None:
            image = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            colormap = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if latents.shape[1] == self.latent_channels * 2:
                image, colormap = latents.chunk(2, dim=1)
            else:
                image, colormap = latents, randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        image_tokens = pack_latents(image, batch_size, self.latent_channels, latent_h, latent_w)
        colormap_tokens = pack_latents(colormap, batch_size, self.latent_channels, latent_h, latent_w)
        return image_tokens, colormap_tokens

    def prepare_fill_condition(
        self,
        control_image: torch.Tensor,
        coarse_mask: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_h, latent_w = self._latent_hw(height, width)
        control_latents = encode_vae_latents(
            self.vae, control_image.to(device=device, dtype=dtype), generator=generator, sample_mode="argmax"
        )
        if control_latents.shape[0] < batch_size:
            control_latents = control_latents.repeat(batch_size // control_latents.shape[0], 1, 1, 1)
        control_tokens = pack_latents(
            control_latents, control_latents.shape[0], control_latents.shape[1], latent_h, latent_w
        )
        mask_tokens = pack_fill_mask(
            coarse_mask.to(device=device, dtype=dtype),
            height=latent_h,
            width=latent_w,
        )
        if mask_tokens.shape[0] < batch_size:
            mask_tokens = mask_tokens.repeat(batch_size // mask_tokens.shape[0], 1, 1)
        return control_tokens, mask_tokens

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
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 30.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        decode_masks: bool = True,
        pdm_delta: Optional[float] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Union[UniGSPipelineOutput, tuple]:
        if task not in TASK_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_PROMPT_TEMPLATES)}.")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor * 2} for FLUX packing, "
                f"got {height} and {width}."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide `prompt` or `prompt_embeds`.")
        if prompt is not None and isinstance(prompt, str) and not any(
            prompt.startswith(prefix) for prefix in ("inpainting:", "synthesis:", "referring:", "panoptic:")
        ):
            prompt = build_task_prompt(task, [prompt] if prompt else [])

        device = self._execution_device
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt if prompt is not None else "",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        dtype = prompt_embeds.dtype

        if image is not None:
            init_image = self.image_processor.preprocess(_as_pil_rgb(image), height=height, width=width).to(
                device=device, dtype=dtype
            )
        else:
            init_image = None
        if colormap is not None:
            colormap_tensor = self.image_processor.preprocess(_as_pil_rgb(colormap), height=height, width=width).to(
                device=device, dtype=dtype
            )
        else:
            colormap_tensor = None
        if mask_image is not None:
            coarse_mask = _as_mask_tensor(mask_image, height, width, device, dtype)
        else:
            coarse_mask = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)

        if task == "inpainting":
            if init_image is None:
                raise ValueError("`image` is required for the inpainting task.")
            control_image = init_image * (1.0 - coarse_mask)
        elif task == "synthesis":
            if colormap_tensor is None:
                raise ValueError("`colormap` is required for the synthesis task.")
            control_image = colormap_tensor
            coarse_mask = torch.ones_like(coarse_mask)
        elif task in {"referring", "entity"}:
            if init_image is None:
                raise ValueError(f"`image` is required for the {task} task.")
            control_image = init_image
            if task == "entity":
                coarse_mask = torch.ones_like(coarse_mask)
        else:
            raise ValueError(f"Unsupported task '{task}'.")

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist() if sigmas is None else sigmas
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

        image_tokens, colormap_tokens = self.prepare_packed_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )
        control_tokens, mask_tokens = self.prepare_fill_condition(
            control_image,
            coarse_mask,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
        )
        img_ids = prepare_latent_image_ids(latent_h, latent_w, device, dtype)
        target_latents = torch.cat([image_tokens, colormap_tokens], dim=-1)

        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.full((target_latents.shape[0],), guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                packed_image, packed_colormap = split_packed_pred(target_latents)
                hidden_states = concat_fill_channels(
                    packed_image, packed_colormap, control_tokens, mask_tokens
                )
                timestep = t.expand(target_latents.shape[0]).to(target_latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
                target_latents = self.scheduler.step(noise_pred, t, target_latents, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, target_latents)

        image_pred, colormap_pred = split_packed_pred(target_latents)
        image_latents = unpack_latents(image_pred, height, width, self.vae_scale_factor)
        colormap_latents = unpack_latents(colormap_pred, height, width, self.vae_scale_factor)

        if output_type == "latent":
            images, colormaps, entity_masks = image_latents, colormap_latents, None
        else:
            decoded_images = decode_vae_latents(self.vae, image_latents)
            decoded_colormaps = decode_vae_latents(self.vae, colormap_latents)
            images = self.image_processor.postprocess(decoded_images, output_type=output_type)
            colormaps = self.image_processor.postprocess(decoded_colormaps, output_type=output_type)
            entity_masks = None
            if decode_masks:
                pdm = self.pdm if pdm_delta is None else ProgressiveDichotomyModule(delta=pdm_delta)
                pdm.include_background = task == "entity"
                colormap_pils = colormaps if output_type == "pil" else self.image_processor.postprocess(
                    decoded_colormaps, output_type="pil"
                )
                entity_masks = [pdm.decode(cmap) for cmap in colormap_pils]

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images, colormaps, entity_masks)
        return UniGSPipelineOutput(images=images, colormaps=colormaps, masks=entity_masks)

    def inpaint(self, prompt, image, mask_image, **kwargs):
        return self(prompt=prompt, image=image, mask_image=mask_image, task="inpainting", **kwargs)

    def synthesize(self, prompt, colormap, **kwargs):
        return self(prompt=prompt, colormap=colormap, task="synthesis", **kwargs)

    def referring(self, prompt, image, mask_image=None, **kwargs):
        return self(prompt=prompt, image=image, mask_image=mask_image, task="referring", **kwargs)

    def segment(self, image, prompt: str = "panoptic: all entities.", **kwargs):
        return self(prompt=prompt, image=image, task="entity", **kwargs)
