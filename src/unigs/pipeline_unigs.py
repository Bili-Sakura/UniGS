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

"""UniGS inference pipeline (community / research-example style).

The UNet denoises concatenated image+colormap latents conditioned on a coarse
mask, a control latent, and a task-prefixed CLIP prompt — the same protocol as
training (arxiv:2312.01985). Backbone is Stable Diffusion 1.5 or 2.1.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor

try:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
except ImportError:  # older / newer Diffusers layouts
    from diffusers import DiffusionPipeline

    try:
        from diffusers import StableDiffusionMixin
    except ImportError:

        class StableDiffusionMixin:  # type: ignore[no-redef]
            pass

try:
    from diffusers.schedulers import KarrasDiffusionSchedulers
except ImportError:
    KarrasDiffusionSchedulers = object

from .backbones import INPAINTING_BACKBONES, resolve_inpainting_checkpoint
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .prompts import TASK_PROMPT_TEMPLATES, build_task_prompt
from .unet import UNIGS_IN_CHANNELS, UNIGS_OUT_CHANNELS, adapt_unigs_unet


logger = logging.get_logger(__name__)

PipelineImageInput = Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]]


@dataclass
class UniGSPipelineOutput(BaseOutput):
    """Output of [`UniGSPipeline`]."""

    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    colormaps: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    masks: Optional[List[List[np.ndarray]]] = None


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of the provided encoder output.")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _as_pil_rgb(image: PipelineImageInput) -> PIL.Image.Image:
    if isinstance(image, list):
        image = image[0]
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        array = image.detach().cpu()
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        array = array.float().numpy()
        if array.max() <= 1.0:
            array = array * 255.0
        return PIL.Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(array).convert("RGB")


def _as_mask_tensor(
    mask: PipelineImageInput,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(mask):
        tensor = mask.float()
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None] if tensor.shape[0] != 1 else tensor.unsqueeze(0)
            if tensor.shape[1] != 1:
                tensor = tensor.mean(dim=1, keepdim=True)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")
        return (tensor > 0.5).to(device=device, dtype=dtype)

    if isinstance(mask, list):
        mask = mask[0]
    if isinstance(mask, PIL.Image.Image):
        mask = mask.convert("L").resize((width, height), resample=PIL.Image.NEAREST)
        array = np.asarray(mask).astype(np.float32) / 255.0
    else:
        array = np.asarray(mask).astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.max() > 1.5:
            array = array / 255.0
        array = np.array(PIL.Image.fromarray((array > 0.5).astype(np.uint8) * 255).resize((width, height), PIL.Image.NEAREST))
        array = array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None]
    return (tensor > 0.5).to(device=device, dtype=dtype)


class UniGSPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for unified image generation and entity-level segmentation (UniGS).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The UNet takes 13 input channels
    ``concat(z_t^image, z_t^colormap, coarse_mask, z_control)`` and predicts 8
    output channels ``concat(eps_image, eps_colormap)``.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder used to encode and decode images *and* colormaps.
        text_encoder ([`CLIPTextModel`]):
            Frozen CLIP text encoder from the inpainting checkpoint (SD 1.5 OpenAI CLIP
            or SD 2.1 OpenCLIP).
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer associated with `text_encoder`.
        unet ([`UNet2DConditionModel`]):
            Conditional UNet with `in_channels=13` and `out_channels=8`.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to denoise the encoded image / colormap latents.
        safety_checker:
            Optional NSFW checker. Disabled by default for this research pipeline.
        feature_extractor:
            Unused unless a safety checker is attached.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "coarse_mask", "control_latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if unet is not None:
            in_ch = getattr(unet.config, "in_channels", None)
            out_ch = getattr(unet.config, "out_channels", None)
            if in_ch != UNIGS_IN_CHANNELS or out_ch != UNIGS_OUT_CHANNELS:
                logger.warning(
                    "UniGS expects a UNet with in_channels=%s and out_channels=%s, got %s / %s. "
                    "Call `adapt_unigs_unet(unet)` (or `UniGSPipeline.from_inpainting`) before inference.",
                    UNIGS_IN_CHANNELS,
                    UNIGS_OUT_CHANNELS,
                    in_ch,
                    out_ch,
                )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.palette = LocationAwarePalette()
        self.pdm = ProgressiveDichotomyModule()

    @classmethod
    def from_inpainting(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        backbone: str = "sd15",
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
        variant: Optional[str] = None,
        scheduler=None,
        **kwargs,
    ) -> "UniGSPipeline":
        """Load an SD inpainting checkpoint and expand its UNet to UniGS channels.

        Args:
            pretrained_model_name_or_path:
                Hub id or local path to an SD *inpainting* pipeline. When omitted,
                `backbone` selects the default checkpoint (`sd15` or `sd21`).
            backbone:
                Shorthand for `stable-diffusion-v1-5/stable-diffusion-inpainting`
                (`sd15`) or `stabilityai/stable-diffusion-2-inpainting` (`sd21`).
        """
        pretrained_model_name_or_path = resolve_inpainting_checkpoint(
            backbone=backbone,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        unet = adapt_unigs_unet(unet)
        if scheduler is None:
            from diffusers import DDIMScheduler

            scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            **kwargs,
        )

    @classmethod
    def from_stable_diffusion(cls, pretrained_model_name_or_path: str, **kwargs) -> "UniGSPipeline":
        """Deprecated alias for :meth:`from_inpainting`. The path must be an SD inpainting checkpoint."""
        return cls.from_inpainting(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            if getattr(self.text_encoder.config, "use_attention_mask", False):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]
            else:
                outputs = self.text_encoder(
                    text_input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(outputs.hidden_states[-(clip_skip + 1)])

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        bs_embed, seq_len, extra = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, extra)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = list(negative_prompt)
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids.to(device)
            if getattr(self.text_encoder.config, "use_attention_mask", False):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(uncond_input_ids, attention_mask=attention_mask)[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, extra)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _encode_vae_image(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if image.shape[1] != 3:
            raise ValueError(f"Expected a 3-channel image tensor, got shape {tuple(image.shape)}")
        latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        return latents * self.vae.config.scaling_factor

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            UNIGS_OUT_CHANNELS,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents * self.scheduler.init_noise_sigma

    def prepare_condition_latents(
        self,
        control_image: torch.Tensor,
        coarse_mask: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        do_classifier_free_guidance: bool,
    ):
        control_latents = self._encode_vae_image(control_image.to(device=device, dtype=dtype), generator=generator)
        mask = torch.nn.functional.interpolate(
            coarse_mask.to(device=device, dtype=dtype),
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="nearest",
        )
        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size, 1, 1, 1)
        if control_latents.shape[0] < batch_size:
            control_latents = control_latents.repeat(batch_size, 1, 1, 1)
        if do_classifier_free_guidance:
            mask = torch.cat([mask] * 2)
            control_latents = torch.cat([control_latents] * 2)
        return mask, control_latents

    def check_inputs(self, prompt, height, width, callback_steps, prompt_embeds, negative_prompt, negative_prompt_embeds):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise TypeError(f"`prompt` must be str or list, got {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Provide either `negative_prompt` or `negative_prompt_embeds`, not both.")
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` has to be a positive integer, got {callback_steps}.")

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
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        decode_masks: bool = True,
        pdm_delta: Optional[float] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Union[UniGSPipelineOutput, tuple]:
        r"""
        Run the UniGS denoising loop.

        Args:
            prompt (`str` or `List[str]`):
                Task-prefixed text prompt, e.g. `"inpainting: generate dog."`. If you pass a
                bare noun phrase, it is wrapped with the template for `task`.
            image (`PIL.Image.Image`):
                Input RGB image. Required for `inpainting`, `referring`, and `entity`.
            mask_image (`PIL.Image.Image`):
                Coarse mask, white = region to fill. If omitted, the full image is used
                (synthesis / entity convention).
            colormap (`PIL.Image.Image`):
                Location-aware entity colormap. Required for `synthesis`; optional otherwise.
            task (`str`, defaults to `"inpainting"`):
                One of `"inpainting"`, `"synthesis"`, `"referring"`, `"entity"`.
            decode_masks (`bool`, defaults to `True`):
                If `True`, run the progressive dichotomy module on the decoded colormap.
        """
        if task not in TASK_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_PROMPT_TEMPLATES)}.")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is not None and isinstance(prompt, str) and not any(
            prompt.startswith(prefix) for prefix in ("inpainting:", "synthesis:", "referring:", "panoptic:")
        ):
            prompt = build_task_prompt(task, [prompt] if prompt else [])

        self.check_inputs(prompt, height, width, callback_steps, prompt_embeds, negative_prompt, negative_prompt_embeds)

        self._guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

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

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )
        mask, control_latents = self.prepare_condition_latents(
            control_image,
            coarse_mask,
            batch_size * num_images_per_prompt,
            height,
            width,
            dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, mask, control_latents], dim=1)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
                        std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                        noise_pred = noise_pred * (std_text / std_cfg) * guidance_rescale + noise_pred * (
                            1.0 - guidance_rescale
                        )

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image_latents, colormap_latents = latents.chunk(2, dim=1)
        if output_type == "latent":
            images = image_latents
            colormaps = colormap_latents
            entity_masks = None
        else:
            scaling = self.vae.config.scaling_factor
            decoded_images = self.vae.decode(image_latents / scaling, return_dict=False)[0]
            decoded_colormaps = self.vae.decode(colormap_latents / scaling, return_dict=False)[0]
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
