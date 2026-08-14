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

"""Adapt DiT backbones to UniGS with context-token concatenation.

FLUX Fill concatenates the packed noisy latents, packed masked-image latents,
and packed mask **along the channel axis** (384-dim tokens). UniGS instead
follows FLUX.1 Kontext: every visual stream is a 64-dim packed token sequence,
and streams are concatenated along the **sequence** axis. Rotary ids use the
first coordinate to distinguish:

* ``0`` — noisy image (denoised)
* ``1`` — noisy colormap (denoised)
* ``2`` — control latent (context only)
* ``3`` — coarse mask (context only)

SD 3.5 and PixArt-α keep native ``in_channels`` and concat **after** patch
embed (plus a zero-init stream embedding). Z-Image uses native omni mode.
See ``dit.py``. Only the image + colormap prefix is the denoising target.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import (
    DIT_STREAM_COLORMAP,
    DIT_STREAM_CONTROL,
    DIT_STREAM_IMAGE,
    DIT_STREAM_MASK,
    FLUX_FILL_CHECKPOINT,
    FLUX_FILL_PACKED_IN_CHANNELS,
    FLUX_LATENT_CHANNELS,
    UNIGS_DIT_PACKED_IN_CHANNELS,
    UNIGS_DIT_PACKED_OUT_CHANNELS,
)


logger = logging.getLogger(__name__)


def pack_latents(
    latents: torch.Tensor,
    batch_size: Optional[int] = None,
    num_channels_latents: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """Pack ``[B, C, H, W]`` latents into 2×2 FLUX tokens ``[B, H/2 * W/2, C*4]``."""
    if latents.ndim != 4:
        raise ValueError(f"Expected 4D latents, got shape {tuple(latents.shape)}")
    batch_size = batch_size if batch_size is not None else latents.shape[0]
    num_channels_latents = num_channels_latents if num_channels_latents is not None else latents.shape[1]
    height = height if height is not None else latents.shape[2]
    width = width if width is not None else latents.shape[3]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Packed latent spatial size must be even, got {(height, width)}.")
    packed = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack FLUX tokens back to ``[B, C, H, W]`` latents.

    ``height`` / ``width`` are **pixel** sizes, matching Diffusers' Flux pipelines.
    """
    batch_size, _num_patches, channels = latents.shape
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    unpacked = latents.view(batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2)
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(batch_size, channels // 4, latent_height, latent_width)


def unpack_latents_from_packed_hw(
    latents: torch.Tensor,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Unpack using latent (not pixel) spatial size."""
    pixel_dummy_scale = 8
    return unpack_latents(
        latents,
        height=latent_height * pixel_dummy_scale,
        width=latent_width * pixel_dummy_scale,
        vae_scale_factor=pixel_dummy_scale,
    )


def prepare_latent_ids(
    packed_height: int,
    packed_width: int,
    device: torch.device,
    dtype: torch.dtype,
    stream_id: int = 0,
) -> torch.Tensor:
    """3D RoPE ids of shape ``[packed_height * packed_width, 3]`` = ``(t, h, w)``."""
    latent_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)
    latent_ids[..., 0] = float(stream_id)
    latent_ids[..., 1] = latent_ids[..., 1] + torch.arange(packed_height, device=device, dtype=dtype)[:, None]
    latent_ids[..., 2] = latent_ids[..., 2] + torch.arange(packed_width, device=device, dtype=dtype)[None, :]
    return latent_ids.reshape(packed_height * packed_width, 3)


def pack_mask_as_tokens(
    mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    latent_channels: int = FLUX_LATENT_CHANNELS,
) -> torch.Tensor:
    """Resize a coarse mask to the latent grid and pack it as 64-dim tokens.

    The mask is repeated across ``latent_channels`` so it shares the UniGS DiT
    ``x_embedder`` (64-in) with image / colormap / control tokens. Spatial RoPE
    matches the other streams; the stream id distinguishes it.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask.float(), size=(latent_height, latent_width), mode="nearest")
    mask = mask.expand(-1, latent_channels, -1, -1)
    return pack_latents(mask, mask.shape[0], latent_channels, latent_height, latent_width)


def concat_context_tokens(
    image_tokens: torch.Tensor,
    colormap_tokens: torch.Tensor,
    control_tokens: torch.Tensor,
    mask_tokens: torch.Tensor,
    packed_height: int,
    packed_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Sequence-concat UniGS DiT streams.

    Returns:
        hidden_states: ``[B, 4S, 64]``
        img_ids: ``[4S, 3]``
        target_seq_len: ``2S`` (image + colormap; condition tokens are dropped
            from the transformer output before the scheduler step)
    """
    for name, tokens in (
        ("image", image_tokens),
        ("colormap", colormap_tokens),
        ("control", control_tokens),
        ("mask", mask_tokens),
    ):
        if tokens.ndim != 3:
            raise ValueError(f"{name} tokens must be packed `[B, S, C]`, got {tuple(tokens.shape)}")

    hidden_states = torch.cat([image_tokens, colormap_tokens, control_tokens, mask_tokens], dim=1)
    device, dtype = image_tokens.device, image_tokens.dtype
    img_ids = torch.cat(
        [
            prepare_latent_ids(packed_height, packed_width, device, dtype, DIT_STREAM_IMAGE),
            prepare_latent_ids(packed_height, packed_width, device, dtype, DIT_STREAM_COLORMAP),
            prepare_latent_ids(packed_height, packed_width, device, dtype, DIT_STREAM_CONTROL),
            prepare_latent_ids(packed_height, packed_width, device, dtype, DIT_STREAM_MASK),
        ],
        dim=0,
    )
    target_seq_len = image_tokens.shape[1] + colormap_tokens.shape[1]
    return hidden_states, img_ids, target_seq_len


def split_target_tokens(
    model_pred: torch.Tensor,
    target_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop context-token predictions and split image / colormap packed outputs."""
    target = model_pred[:, :target_seq_len]
    seq = target_seq_len // 2
    return target[:, :seq], target[:, seq:]


def vae_shift(vae) -> float:
    return float(getattr(getattr(vae, "config", vae), "shift_factor", 0.0) or 0.0)


def encode_vae_latents(vae, image: torch.Tensor, generator=None, sample_mode: str = "sample") -> torch.Tensor:
    encoder_output = vae.encode(image)
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "argmax":
            latents = encoder_output.latent_dist.mode()
        else:
            latents = encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        latents = encoder_output.latents
    else:
        raise AttributeError("Could not access latents of the provided encoder output.")
    return (latents - vae_shift(vae)) * vae.config.scaling_factor


def decode_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    return vae.decode(latents / vae.config.scaling_factor + vae_shift(vae), return_dict=False)[0]


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def adapt_unigs_transformer(transformer, zero_init: bool = True, family: Optional[str] = None):
    """Adapt a DiT transformer to UniGS context-token concat.

    * **FLUX Fill** — shrink ``x_embedder`` 384→64 by copying packed noisy-latent
      columns. Streams share that embedder and are distinguished by sequence + RoPE.
    * **SD 3.5 / PixArt-α** — keep native ``in_channels``; register a zero-init
      ``unigs_stream_embed`` so stream 0 matches the pretrained image path.
    * **Z-Image** — keep native omni embedders; mark ``unigs_dit_family``.

    Already-adapted UniGS checkpoints are returned as-is.
    """
    from .dit import ensure_unigs_stream_embed, infer_transformer_family

    family = family or infer_transformer_family(transformer)
    if family in {"sd3", "pixart"}:
        ensure_unigs_stream_embed(transformer)
        if hasattr(transformer, "register_to_config"):
            transformer.register_to_config(unigs_dit_family=family)
        logger.info("Registered UniGS stream embeddings on %s DiT (native in_channels).", family)
        return transformer
    if family == "z_image":
        if hasattr(transformer, "register_to_config"):
            transformer.register_to_config(unigs_dit_family=family)
        logger.info("Z-Image DiT uses native omni context-token concat (no channel adapt).")
        return transformer

    old_in = int(transformer.config.in_channels)
    old_out = int(getattr(transformer.config, "out_channels", None) or old_in)

    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(unigs_dit_family="flux")

    if old_in == UNIGS_DIT_PACKED_IN_CHANNELS and old_out == UNIGS_DIT_PACKED_OUT_CHANNELS:
        logger.info("Transformer already has UniGS DiT packed channels (64 in / 64 out).")
        return transformer

    if old_in not in (FLUX_FILL_PACKED_IN_CHANNELS, UNIGS_DIT_PACKED_IN_CHANNELS):
        raise ValueError(
            f"Unsupported FluxTransformer `in_channels={old_in}`. Expected "
            f"{FLUX_FILL_PACKED_IN_CHANNELS} (FLUX.1-Fill-dev channel concat) or "
            f"{UNIGS_DIT_PACKED_IN_CHANNELS} (UniGS token concat). "
            f"Use `{FLUX_FILL_CHECKPOINT}`."
        )

    logger.info(
        "Adapting FLUX Fill transformer from %s-in/%s-out packed channels to UniGS "
        "DiT %s-in/%s-out (context-token concat, copy noisy-latent embedder=%s).",
        old_in,
        old_out,
        UNIGS_DIT_PACKED_IN_CHANNELS,
        UNIGS_DIT_PACKED_OUT_CHANNELS,
        zero_init,
    )

    if old_in != UNIGS_DIT_PACKED_IN_CHANNELS:
        old_embed = transformer.x_embedder
        new_embed = nn.Linear(
            UNIGS_DIT_PACKED_IN_CHANNELS,
            old_embed.out_features,
            bias=old_embed.bias is not None,
        )
        new_embed = new_embed.to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)
        with torch.no_grad():
            # Fill layout: [noisy(64), masked_image(64), mask(256)]. Keep noisy.
            new_embed.weight.copy_(old_embed.weight[:, :UNIGS_DIT_PACKED_IN_CHANNELS])
            if new_embed.bias is not None and old_embed.bias is not None:
                new_embed.bias.copy_(old_embed.bias)
            elif new_embed.bias is not None:
                new_embed.bias.zero_()
        transformer.x_embedder = new_embed
        transformer.register_to_config(in_channels=UNIGS_DIT_PACKED_IN_CHANNELS)

    if old_out != UNIGS_DIT_PACKED_OUT_CHANNELS:
        old_proj = transformer.proj_out
        patch_size = int(getattr(transformer.config, "patch_size", 1) or 1)
        new_out_features = patch_size * patch_size * UNIGS_DIT_PACKED_OUT_CHANNELS
        new_proj = nn.Linear(old_proj.in_features, new_out_features, bias=old_proj.bias is not None)
        new_proj = new_proj.to(device=old_proj.weight.device, dtype=old_proj.weight.dtype)
        with torch.no_grad():
            new_proj.weight.zero_()
            rows = min(old_proj.weight.shape[0], new_proj.weight.shape[0])
            new_proj.weight[:rows].copy_(old_proj.weight[:rows])
            if new_proj.bias is not None and old_proj.bias is not None:
                new_proj.bias.zero_()
                new_proj.bias[: min(old_proj.bias.shape[0], new_proj.bias.shape[0])].copy_(
                    old_proj.bias[: min(old_proj.bias.shape[0], new_proj.bias.shape[0])]
                )
        transformer.proj_out = new_proj
        transformer.register_to_config(out_channels=UNIGS_DIT_PACKED_OUT_CHANNELS)

    return transformer
