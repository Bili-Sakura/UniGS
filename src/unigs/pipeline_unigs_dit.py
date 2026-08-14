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

"""UniGS inference pipeline for SD 3.5, Z-Image, and PixArt-α DiT backbones.

Conditioning follows FLUX.1-Fill-dev: the coarse mask and control latent are
concatenated on the **channel** axis of a 4D map. FLUX Fill stays on
[`UniGSFluxPipeline`] (packed last-dim concat).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer, CLIPTokenizer, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
except ImportError:
    from diffusers import DiffusionPipeline

from .backbones import (
    DIT_FAMILY_CHECKPOINTS,
    default_dit_guidance,
    default_dit_max_sequence_length,
    detect_dit_family_from_path,
    resolve_backbone,
    resolve_dit_family,
)
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .dit import (
    encode_pixart_prompt,
    encode_sd3_prompt,
    encode_zimage_prompt,
    forward_pixart_channel_concat,
    forward_sd3_channel_concat,
    forward_zimage_channel_concat,
    pixart_added_cond_kwargs,
    resize_mask_to_latents,
)
from .pipeline_unigs import (
    PipelineImageInput,
    UniGSPipelineOutput,
    _as_mask_tensor,
    _as_pil_rgb,
    retrieve_timesteps,
)
from .prompts import TASK_PROMPT_TEMPLATES, build_task_prompt
from .transformer import adapt_unigs_transformer, decode_vae_latents, encode_vae_latents, calculate_shift


logger = logging.get_logger(__name__)


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


def _import_zimage():
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    except ImportError as err:
        raise ImportError(
            "Z-Image UniGS requires a recent Diffusers with `ZImageTransformer2DModel` "
            "(install from source if your wheel is older than the Z-Image release)."
        ) from err
    return ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler


def _import_pixart():
    try:
        from diffusers import DPMSolverMultistepScheduler, PixArtTransformer2DModel
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as err:
        raise ImportError(
            "PixArt-α UniGS requires Diffusers `PixArtTransformer2DModel` and transformers T5."
        ) from err
    return PixArtTransformer2DModel, DPMSolverMultistepScheduler, T5EncoderModel, T5Tokenizer


def load_dit_transformer(family: str, pretrained_model_name_or_path: str, **load_kw):
    if family == "sd3":
        cls, _, _, _ = _import_sd3()
    elif family == "z_image":
        cls, _ = _import_zimage()
    elif family == "pixart":
        cls, _, _, _ = _import_pixart()
    else:
        raise ValueError(f"Unsupported DiT family '{family}'.")
    transformer = cls.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", **load_kw)
    return adapt_unigs_transformer(transformer, family=family)


class UniGSDiTPipeline(DiffusionPipeline):
    r"""
    UniGS on SD 3.5 Medium, Z-Image, or PixArt-α.

    Image and colormap latents are denoised jointly. The coarse mask and control
    latent are concatenated on the channel axis (Fill-style), not as extra
    sequence / omni tokens.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = [
        "tokenizer_2",
        "text_encoder_2",
        "tokenizer_3",
        "text_encoder_3",
        "safety_checker",
        "feature_extractor",
    ]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler,
        vae: AutoencoderKL,
        transformer,
        tokenizer,
        text_encoder,
        tokenizer_2=None,
        text_encoder_2=None,
        tokenizer_3=None,
        text_encoder_3=None,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
        unigs_dit_family: str = "sd3",
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
        family = unigs_dit_family or getattr(getattr(transformer, "config", None), "unigs_dit_family", None) or "sd3"
        self.register_to_config(unigs_dit_family=family, requires_safety_checker=requires_safety_checker)
        self.unigs_dit_family = family
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", None) or 16)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.latent_channels,
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
    def from_backbone(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        backbone: str = "sd3",
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        scheduler=None,
        **kwargs,
    ) -> "UniGSDiTPipeline":
        family = resolve_dit_family(backbone) or resolve_dit_family(pretrained_model_name_or_path) or "sd3"
        if family == "flux":
            from .pipeline_unigs_flux import UniGSFluxPipeline

            return UniGSFluxPipeline.from_fill(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                backbone=backbone,
                torch_dtype=torch_dtype,
                revision=revision,
                variant=variant,
                scheduler=scheduler,
                **kwargs,
            )
        pretrained_model_name_or_path = resolve_backbone(
            backbone=backbone,
            pretrained_model_name_or_path=pretrained_model_name_or_path or DIT_FAMILY_CHECKPOINTS[family],
        )
        load_kw = dict(revision=revision, variant=variant, torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", **load_kw)
        transformer = load_dit_transformer(family, pretrained_model_name_or_path, **load_kw)

        tokenizer_2 = text_encoder_2 = tokenizer_3 = text_encoder_3 = None
        if family == "sd3":
            SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, CLIPTextModelWithProjection, T5EncoderModel = _import_sd3()
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
            tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", revision=revision)
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
        elif family == "z_image":
            _, FlowMatchEulerDiscreteScheduler = _import_zimage()
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
            from transformers import AutoModel

            try:
                from transformers import Qwen2Model, Qwen2Tokenizer

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
        elif family == "pixart":
            _, DPMSolverMultistepScheduler, T5EncoderModel, T5Tokenizer = _import_pixart()
            try:
                tokenizer = T5Tokenizer.from_pretrained(
                    pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
                )
            except Exception:
                tokenizer = T5TokenizerFast.from_pretrained(
                    pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
                )
            text_encoder = T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", **load_kw
            )
            if scheduler is None:
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    pretrained_model_name_or_path, subfolder="scheduler"
                )
        else:
            raise ValueError(f"Unsupported DiT family '{family}'.")

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
            unigs_dit_family=family,
            **kwargs,
        )

    @property
    def family(self) -> str:
        return getattr(self, "unigs_dit_family", None) or self.config.get("unigs_dit_family", "sd3")

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
        family = self.family
        max_sequence_length = max_sequence_length or default_dit_max_sequence_length(family)
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(prompt, list) and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        if family == "sd3":
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
                joint_attention_dim=int(getattr(self.transformer.config, "joint_attention_dim", 4096)),
            )
            negative = None
            negative_pooled = None
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
                    joint_attention_dim=int(getattr(self.transformer.config, "joint_attention_dim", 4096)),
                )
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": negative,
                "negative_pooled_prompt_embeds": negative_pooled,
            }
        if family == "pixart":
            prompt_embeds, prompt_attention_mask = encode_pixart_prompt(
                self.text_encoder,
                self.tokenizer,
                prompt,
                device,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative = negative_mask = None
            if do_classifier_free_guidance:
                negative, negative_mask = encode_pixart_prompt(
                    self.text_encoder,
                    self.tokenizer,
                    negative_prompt or "",
                    device,
                    max_sequence_length=max_sequence_length,
                    num_images_per_prompt=num_images_per_prompt,
                )
            return {
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_embeds": negative,
                "negative_prompt_attention_mask": negative_mask,
            }
        prompt_embeds = encode_zimage_prompt(
            self.text_encoder,
            self.tokenizer,
            prompt,
            device,
            max_sequence_length=max_sequence_length,
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
        if self.family == "z_image":
            latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
            latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
            return latent_h, latent_w
        return int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor

    def _prepare_noisy_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator=None,
        latents: Optional[torch.Tensor] = None,
    ):
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
        return image, colormap

    def _prepare_context(
        self,
        control_image: torch.Tensor,
        coarse_mask: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator=None,
    ):
        latent_h, latent_w = self._latent_hw(height, width)
        control_latents = encode_vae_latents(
            self.vae, control_image.to(device=device, dtype=dtype), generator=generator, sample_mode="argmax"
        )
        if control_latents.shape[0] < batch_size:
            control_latents = control_latents.repeat(batch_size // control_latents.shape[0], 1, 1, 1)
        mask_latents = resize_mask_to_latents(
            coarse_mask.to(device=device, dtype=dtype),
            latent_height=latent_h,
            latent_width=latent_w,
        )
        if mask_latents.shape[0] < batch_size:
            mask_latents = mask_latents.repeat(batch_size // mask_latents.shape[0], 1, 1, 1)
        return control_latents.to(dtype=dtype), mask_latents.to(dtype=dtype)

    def _dit_forward(
        self,
        image,
        colormap,
        control,
        mask,
        timestep,
        encoded,
        guidance_scale: float,
        height: int,
        width: int,
    ):
        family = self.family
        has_uncond = encoded.get("negative_prompt_embeds") is not None
        do_cfg = has_uncond and (
            (family == "z_image" and guidance_scale is not None and guidance_scale > 0)
            or (family != "z_image" and guidance_scale is not None and guidance_scale > 1)
        )
        if family == "z_image":
            # Official Z-Image Turbo uses guidance 0; CFG>0 concatenates cond+uncond lists.
            if do_cfg:
                image_in = torch.cat([image, image], dim=0)
                colormap_in = torch.cat([colormap, colormap], dim=0)
                control_in = torch.cat([control, control], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
                caps = list(encoded["prompt_embeds"]) + list(encoded["negative_prompt_embeds"])
                timestep_in = timestep.repeat(2)
                pred = forward_zimage_channel_concat(
                    self.transformer, image_in, colormap_in, control_in, mask_in, timestep_in, caps, negate=True
                )[0]
                cond, uncond = pred.chunk(2)
                return uncond + guidance_scale * (cond - uncond)
            return forward_zimage_channel_concat(
                self.transformer, image, colormap, control, mask, timestep, encoded["prompt_embeds"], negate=True
            )[0]

        if family == "sd3":
            if do_cfg:
                image_in = torch.cat([image, image], dim=0)
                colormap_in = torch.cat([colormap, colormap], dim=0)
                control_in = torch.cat([control, control], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
                prompt_embeds = torch.cat(
                    [encoded["negative_prompt_embeds"], encoded["prompt_embeds"]], dim=0
                )
                pooled = torch.cat(
                    [encoded["negative_pooled_prompt_embeds"], encoded["pooled_prompt_embeds"]], dim=0
                )
                timestep_in = timestep.repeat(2)
                pred = forward_sd3_channel_concat(
                    self.transformer,
                    image_in,
                    colormap_in,
                    control_in,
                    mask_in,
                    timestep_in,
                    prompt_embeds,
                    pooled,
                )[0]
                uncond, cond = pred.chunk(2)
                return uncond + guidance_scale * (cond - uncond)
            return forward_sd3_channel_concat(
                self.transformer,
                image,
                colormap,
                control,
                mask,
                timestep,
                encoded["prompt_embeds"],
                encoded["pooled_prompt_embeds"],
            )[0]

        added = pixart_added_cond_kwargs(
            self.transformer,
            batch_size=image.shape[0],
            height=height,
            width=width,
            dtype=image.dtype,
            device=image.device,
            cfg_multiplier=2 if do_cfg else 1,
        )
        if do_cfg:
            image_in = torch.cat([image, image], dim=0)
            colormap_in = torch.cat([colormap, colormap], dim=0)
            control_in = torch.cat([control, control], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            prompt_embeds = torch.cat([encoded["negative_prompt_embeds"], encoded["prompt_embeds"]], dim=0)
            attn = torch.cat([encoded["negative_prompt_attention_mask"], encoded["prompt_attention_mask"]], dim=0)
            timestep_in = timestep.repeat(2)
            pred = forward_pixart_channel_concat(
                self.transformer,
                image_in,
                colormap_in,
                control_in,
                mask_in,
                timestep_in,
                prompt_embeds,
                encoder_attention_mask=attn,
                added_cond_kwargs=added,
            )[0]
            uncond, cond = pred.chunk(2)
            return uncond + guidance_scale * (cond - uncond)
        return forward_pixart_channel_concat(
            self.transformer,
            image,
            colormap,
            control,
            mask,
            timestep,
            encoded["prompt_embeds"],
            encoder_attention_mask=encoded.get("prompt_attention_mask"),
            added_cond_kwargs=added,
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
        if task not in TASK_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_PROMPT_TEMPLATES)}.")

        family = self.family
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if guidance_scale is None:
            guidance_scale = default_dit_guidance(family)

        if prompt is None:
            raise ValueError("Provide `prompt`.")
        if isinstance(prompt, str) and not any(
            prompt.startswith(prefix) for prefix in ("inpainting:", "synthesis:", "referring:", "panoptic:")
        ):
            prompt = build_task_prompt(task, [prompt] if prompt else [])

        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = (family == "z_image" and guidance_scale > 0) or (family != "z_image" and guidance_scale > 1)
        encoded = self.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
        )
        if family == "z_image":
            dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else next(self.transformer.parameters()).dtype
        else:
            dtype = encoded["prompt_embeds"].dtype

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

        if family in {"sd3", "z_image"} and sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()
        latent_h, latent_w = self._latent_hw(height, width)
        scheduler_kwargs = {}
        if family == "z_image":
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        try:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, **scheduler_kwargs
            )
        except TypeError:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )

        image_latents, colormap_latents = self._prepare_noisy_latents(
            batch_size * num_images_per_prompt, height, width, dtype, device, generator, latents
        )
        control_latents, mask_latents = self._prepare_context(
            control_image,
            coarse_mask,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if family == "z_image":
                    timestep = t.expand(image_latents.shape[0])
                    timestep = (1000 - timestep) / 1000
                elif family == "pixart":
                    latent_in_image = self.scheduler.scale_model_input(image_latents, t)
                    latent_in_cmap = self.scheduler.scale_model_input(colormap_latents, t)
                    current = t
                    if not torch.is_tensor(current):
                        current = torch.tensor([current], device=device)
                    elif current.ndim == 0:
                        current = current[None].to(device)
                    timestep = current.expand(image_latents.shape[0])
                    image_latents_in, colormap_latents_in = latent_in_image, latent_in_cmap
                else:
                    timestep = t.expand(image_latents.shape[0])
                    image_latents_in, colormap_latents_in = image_latents, colormap_latents

                if family != "pixart":
                    image_latents_in, colormap_latents_in = image_latents, colormap_latents

                noise_pred = self._dit_forward(
                    image_latents_in,
                    colormap_latents_in,
                    control_latents,
                    mask_latents,
                    timestep,
                    encoded,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                )
                target = torch.cat([image_latents, colormap_latents], dim=1)
                target = self.scheduler.step(noise_pred, t, target, return_dict=False)[0]
                image_latents, colormap_latents = target.chunk(2, dim=1)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, target)

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


def load_unigs_dit_pipeline(
    pretrained_model_name_or_path: Optional[str] = None,
    backbone: str = "sd3",
    **kwargs,
):
    family = (
        resolve_dit_family(backbone)
        or detect_dit_family_from_path(pretrained_model_name_or_path)
        or resolve_dit_family(pretrained_model_name_or_path)
    )
    if family == "flux":
        from .pipeline_unigs_flux import UniGSFluxPipeline

        return UniGSFluxPipeline.from_fill(
            pretrained_model_name_or_path=pretrained_model_name_or_path, backbone=backbone, **kwargs
        )
    return UniGSDiTPipeline.from_backbone(
        pretrained_model_name_or_path=pretrained_model_name_or_path, backbone=backbone, **kwargs
    )
