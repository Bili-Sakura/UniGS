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

"""Shared UniGS task / latent helpers used by the per-model pipelines."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from .colormap import ProgressiveDichotomyModule
from .dit import resize_mask_to_latents
from .pipeline_unigs import UniGSPipelineOutput
from .prompts import TASK_PROMPT_TEMPLATES, build_task_prompt
from .transformer import decode_vae_latents, encode_vae_latents


def prefix_task_prompt(prompt: Union[str, list], task: str):
    if prompt is not None and isinstance(prompt, str) and not any(
        prompt.startswith(prefix) for prefix in ("inpainting:", "synthesis:", "referring:", "panoptic:")
    ):
        return build_task_prompt(task, [prompt] if prompt else [])
    return prompt


def prepare_task_condition(task: str, init_image, colormap_tensor, coarse_mask):
    """Build the control image + coarse mask for one UniGS Table-2 task."""
    if task not in TASK_PROMPT_TEMPLATES:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_PROMPT_TEMPLATES)}.")
    if task == "inpainting":
        if init_image is None:
            raise ValueError("`image` is required for the inpainting task.")
        return init_image * (1.0 - coarse_mask), coarse_mask
    if task == "synthesis":
        if colormap_tensor is None:
            raise ValueError("`colormap` is required for the synthesis task.")
        return colormap_tensor, torch.ones_like(coarse_mask)
    if task in {"referring", "entity"}:
        if init_image is None:
            raise ValueError(f"`image` is required for the {task} task.")
        if task == "entity":
            coarse_mask = torch.ones_like(coarse_mask)
        return init_image, coarse_mask
    raise ValueError(f"Unsupported task '{task}'.")


def prepare_noisy_dual_latents(
    batch_size: int,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    dtype: torch.dtype,
    device: torch.device,
    generator=None,
    latents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (batch_size, latent_channels, latent_h, latent_w)
    if latents is None:
        image = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        colormap = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return image, colormap
    latents = latents.to(device=device, dtype=dtype)
    if latents.shape[1] == latent_channels * 2:
        return latents.chunk(2, dim=1)
    colormap = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents, colormap


def prepare_control_and_mask_latents(
    vae,
    control_image: torch.Tensor,
    coarse_mask: torch.Tensor,
    batch_size: int,
    latent_h: int,
    latent_w: int,
    latent_channels: int,
    dtype: torch.dtype,
    device: torch.device,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    control_latents = encode_vae_latents(
        vae, control_image.to(device=device, dtype=dtype), generator=generator, sample_mode="argmax"
    )
    if control_latents.shape[0] < batch_size:
        control_latents = control_latents.repeat(batch_size // control_latents.shape[0], 1, 1, 1)
    mask_latents = resize_mask_to_latents(
        coarse_mask.to(device=device, dtype=dtype),
        latent_height=latent_h,
        latent_width=latent_w,
        latent_channels=latent_channels,
    )
    if mask_latents.shape[0] < batch_size:
        mask_latents = mask_latents.repeat(batch_size // mask_latents.shape[0], 1, 1, 1)
    return control_latents.to(dtype=dtype), mask_latents.to(dtype=dtype)


def decode_dual_outputs(
    vae,
    image_processor,
    image_latents: torch.Tensor,
    colormap_latents: torch.Tensor,
    output_type: str,
    decode_masks: bool,
    pdm: ProgressiveDichotomyModule,
    task: str,
    pdm_delta: Optional[float],
    return_dict: bool,
):
    if output_type == "latent":
        images, colormaps, entity_masks = image_latents, colormap_latents, None
    else:
        decoded_images = decode_vae_latents(vae, image_latents)
        decoded_colormaps = decode_vae_latents(vae, colormap_latents)
        images = image_processor.postprocess(decoded_images, output_type=output_type)
        colormaps = image_processor.postprocess(decoded_colormaps, output_type=output_type)
        entity_masks = None
        if decode_masks:
            decoder = pdm if pdm_delta is None else ProgressiveDichotomyModule(delta=pdm_delta)
            decoder.include_background = task == "entity"
            colormap_pils = colormaps if output_type == "pil" else image_processor.postprocess(
                decoded_colormaps, output_type="pil"
            )
            entity_masks = [decoder.decode(cmap) for cmap in colormap_pils]
    if not return_dict:
        return (images, colormaps, entity_masks)
    return UniGSPipelineOutput(images=images, colormaps=colormaps, masks=entity_masks)
