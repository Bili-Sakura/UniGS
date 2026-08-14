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

"""FLUX.1-Fill-dev transformer: pack/unpack, Fill-style mask, UniGS channel concat.

FLUX.1-Fill-dev concatenates **on the packed channel axis** (last dim)::

    hidden = cat(noisy_64, masked_image_64, mask_256, dim=-1)  # 384

UniGS follows the same style for all four streams::

    hidden = cat(img_64, cmap_64, control_64, mask_256, dim=-1)  # 448
    pred   = transformer(...)  # [B, S, 128] = packed image + packed colormap

A single RoPE ``img_ids`` grid is used (Fill's layout), not four stream ids.
Spatial DiTs (SD 3.5, PixArt, Z-Image) use the analog on 4D maps — see
:mod:`unigs.dit`.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import (
    FLUX_FILL_PACKED_IN_CHANNELS,
    FLUX_FILL_PACKED_LATENT,
    FLUX_FILL_PACKED_MASK,
    UNIGS_DIT_PACKED_IN_CHANNELS,
    UNIGS_DIT_PACKED_OUT_CHANNELS,
    family_from_pretrained,
    infer_transformer_family,
)


logger = logging.getLogger(__name__)

_FILL_MASK_FOLD = 8
_FILL_PACK_H = 2
_FILL_PACK_W = 2


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
    """3D RoPE ids of shape ``[packed_height * packed_width, 3]`` = ``(t, h, w)``.

    Fill-style UniGS uses a **single** stream (``stream_id=0``); extra ids were
    only needed for sequence-concat.
    """
    latent_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)
    latent_ids[..., 0] = float(stream_id)
    latent_ids[..., 1] = latent_ids[..., 1] + torch.arange(packed_height, device=device, dtype=dtype)[:, None]
    latent_ids[..., 2] = latent_ids[..., 2] + torch.arange(packed_width, device=device, dtype=dtype)[None, :]
    return latent_ids.reshape(packed_height * packed_width, 3)


def prepare_latent_image_ids(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """RoPE ids for one packed Fill stream from **latent** HxW."""
    return prepare_latent_ids(height // _FILL_PACK_H, width // _FILL_PACK_W, device, dtype, stream_id=0)


def fold_fill_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Fold a pixel-space mask into 64 channels at latent HxW (Fill's ``prepare_mask_latents``).

    ``mask`` is ``[B, 1, H_pix, W_pix]``. Fill interpolates to ``(H*8, W*8)``,
    then views 8×8 neighborhoods as extra channels::

        [B, 1, 8H, 8W] → [B, 64, H, W]
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    target_h = height * _FILL_MASK_FOLD
    target_w = width * _FILL_MASK_FOLD
    if mask.shape[-2:] != (target_h, target_w):
        mask = F.interpolate(mask.float(), size=(target_h, target_w), mode="nearest")
    batch = mask.shape[0]
    return (
        mask.view(batch, height, _FILL_MASK_FOLD, width, _FILL_MASK_FOLD)
        .permute(0, 2, 4, 1, 3)
        .reshape(batch, _FILL_MASK_FOLD * _FILL_MASK_FOLD, height, width)
    )


def pack_fill_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Pixel (or latent) mask → packed Fill mask tokens ``[B, S, 256]``."""
    return pack_latents(fold_fill_mask(mask, height, width))


def concat_fill_channels(
    packed_image: torch.Tensor,
    packed_colormap: torch.Tensor,
    packed_control: torch.Tensor,
    packed_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill-style last-dim concat: ``[B, S, 448] = 64+64+64+256``."""
    for name, tensor, channels in (
        ("image", packed_image, FLUX_FILL_PACKED_LATENT),
        ("colormap", packed_colormap, FLUX_FILL_PACKED_LATENT),
        ("control", packed_control, FLUX_FILL_PACKED_LATENT),
        ("mask", packed_mask, FLUX_FILL_PACKED_MASK),
    ):
        if tensor.ndim != 3 or tensor.shape[-1] != channels:
            raise ValueError(f"{name} packed channels must be {channels}, got {tuple(tensor.shape)}")
    return torch.cat([packed_image, packed_colormap, packed_control, packed_mask], dim=-1)


def split_packed_pred(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``[B, S, 128]`` transformer output into packed image + colormap (64 each)."""
    if packed.shape[-1] != UNIGS_DIT_PACKED_OUT_CHANNELS:
        raise ValueError(
            f"expected packed pred last dim {UNIGS_DIT_PACKED_OUT_CHANNELS}, got {packed.shape[-1]}"
        )
    return packed[..., :FLUX_FILL_PACKED_LATENT], packed[..., FLUX_FILL_PACKED_LATENT:]


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


def _copy_linear_out(src: nn.Linear, dst: nn.Linear) -> None:
    """Copy overlapping output rows; extra UniGS rows stay at their init (zeros)."""
    with torch.no_grad():
        dst.weight.zero_()
        n_out = min(src.weight.shape[0], dst.weight.shape[0])
        dst.weight[:n_out] = src.weight[:n_out]
        if src.bias is not None and dst.bias is not None:
            dst.bias.zero_()
            dst.bias[:n_out] = src.bias[:n_out]


def _copy_fill_x_embedder(src: nn.Linear, dst: nn.Linear) -> None:
    """Map Fill 384-in weights onto UniGS 448-in layout.

    Fill:  ``[noisy_64 | masked_image_64 | mask_256]``
    UniGS: ``[img_64 | cmap_64 | control_64 | mask_256]``

    Image and mask columns copy 1:1. Fill's masked-image columns become UniGS
    control. Colormap columns stay zero-init. A previous UniGS 64-in (token
    concat) checkpoint copies into the image slot only.
    """
    src_in = src.weight.shape[1]
    dst_in = dst.weight.shape[1]
    with torch.no_grad():
        dst.weight.zero_()
        if src_in == FLUX_FILL_PACKED_IN_CHANNELS and dst_in == UNIGS_DIT_PACKED_IN_CHANNELS:
            dst.weight[:, 0:64] = src.weight[:, 0:64]
            dst.weight[:, 128:192] = src.weight[:, 64:128]
            dst.weight[:, 192:448] = src.weight[:, 128:384]
        elif src_in == FLUX_FILL_PACKED_LATENT and dst_in == UNIGS_DIT_PACKED_IN_CHANNELS:
            dst.weight[:, :FLUX_FILL_PACKED_LATENT] = src.weight
        else:
            n_in = min(src_in, dst_in)
            dst.weight[:, :n_in] = src.weight[:, :n_in]
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


def _guess_flux_family(transformer) -> bool:
    in_ch = int(getattr(getattr(transformer, "config", None), "in_channels", 0) or 0)
    return hasattr(transformer, "x_embedder") and in_ch in {
        FLUX_FILL_PACKED_IN_CHANNELS,
        UNIGS_DIT_PACKED_IN_CHANNELS,
        FLUX_FILL_PACKED_LATENT,
    }


def adapt_unigs_transformer(transformer, zero_init: bool = True, family: Optional[str] = None, pretrained: Optional[str] = None):
    """Expand Fill (384→64) or a spatial DiT to UniGS channel-concat I/O.

    FLUX.1-Fill-dev: ``x_embedder`` 384→448, ``proj_out`` 64→128, with Fill
    column remapping. Spatial families are adapted in
    :func:`unigs.dit.adapt_spatial_transformer`.
    """
    del zero_init  # extra channels are always zero-init then overwritten where pretrained
    family = family or family_from_pretrained(pretrained, transformer) or infer_transformer_family(transformer)
    if family is None and _guess_flux_family(transformer):
        family = "flux"
    if family is None:
        raise TypeError(
            "Cannot infer DiT family for UniGS adapter; pass family='flux'|'sd3'|'z_image'|'pixart'."
        )

    if family != "flux":
        from .dit import adapt_spatial_transformer

        adapted = adapt_spatial_transformer(transformer, family)
        if hasattr(adapted, "register_to_config"):
            adapted.register_to_config(unigs_dit_family=family)
        elif hasattr(adapted, "config"):
            adapted.config.unigs_dit_family = family
        return adapted

    in_ch = int(getattr(transformer.config, "in_channels", 0))
    out_ch = int(getattr(transformer.config, "out_channels", 0))
    if in_ch == UNIGS_DIT_PACKED_IN_CHANNELS and out_ch == UNIGS_DIT_PACKED_OUT_CHANNELS:
        if hasattr(transformer, "register_to_config"):
            transformer.register_to_config(unigs_dit_family="flux")
        return transformer

    inner_dim = transformer.x_embedder.weight.shape[0]
    old_x = transformer.x_embedder
    transformer.x_embedder = nn.Linear(UNIGS_DIT_PACKED_IN_CHANNELS, inner_dim)
    nn.init.zeros_(transformer.x_embedder.weight)
    if transformer.x_embedder.bias is not None:
        nn.init.zeros_(transformer.x_embedder.bias)
    _copy_fill_x_embedder(old_x, transformer.x_embedder)

    old_out = transformer.proj_out
    transformer.proj_out = nn.Linear(inner_dim, UNIGS_DIT_PACKED_OUT_CHANNELS)
    nn.init.zeros_(transformer.proj_out.weight)
    if transformer.proj_out.bias is not None:
        nn.init.zeros_(transformer.proj_out.bias)
    _copy_linear_out(old_out, transformer.proj_out)

    transformer.config.in_channels = UNIGS_DIT_PACKED_IN_CHANNELS
    transformer.config.out_channels = UNIGS_DIT_PACKED_OUT_CHANNELS
    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(
            in_channels=UNIGS_DIT_PACKED_IN_CHANNELS,
            out_channels=UNIGS_DIT_PACKED_OUT_CHANNELS,
            unigs_dit_family="flux",
        )
    logger.info(
        "Adapted FLUX Fill transformer to UniGS channel concat (%s→%s in, %s→%s out).",
        in_ch,
        UNIGS_DIT_PACKED_IN_CHANNELS,
        out_ch,
        UNIGS_DIT_PACKED_OUT_CHANNELS,
    )
    return transformer


def maybe_adapt_unigs_transformer(transformer, family: Optional[str] = None, pretrained: Optional[str] = None):
    """Idempotent :func:`adapt_unigs_transformer`."""
    return adapt_unigs_transformer(transformer, family=family, pretrained=pretrained)
