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

"""Supported UniGS inpainting backbones: SD UNets and FLUX DiT."""

from __future__ import annotations

from typing import Optional


# UniGS is initialized from inpainting checkpoints, not base text-to-image.
UNET_BACKBONES = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "sd21": "stabilityai/stable-diffusion-2-inpainting",
}

# FLUX.1-Fill-dev is a DiT (MMDiT) inpainting backbone. UniGS does *not* keep
# Fill's channel-concat condition (`cat(latents, masked_image, mask)` along the
# packed channel axis). Conditioning is Kontext-style sequence / context-token
# concatenation instead; see `transformer.py`.
DIT_BACKBONES = {
    "flux": "black-forest-labs/FLUX.1-Fill-dev",
    "flux_fill": "black-forest-labs/FLUX.1-Fill-dev",
}

BACKBONES = {**UNET_BACKBONES, **DIT_BACKBONES}
INPAINTING_BACKBONES = BACKBONES  # backward-compatible alias

DEFAULT_BACKBONE = "sd15"
INPAINTING_UNET_IN_CHANNELS = 9

FLUX_FILL_CHECKPOINT = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_LATENT_CHANNELS = 16
FLUX_FILL_PACKED_IN_CHANNELS = 384  # packed noisy (64) + masked image (64) + mask (256)
UNIGS_DIT_PACKED_IN_CHANNELS = 64  # packed 16-channel latents only
UNIGS_DIT_PACKED_OUT_CHANNELS = 64

# RoPE stream ids for concatenated visual tokens (Kontext-style first axis).
DIT_STREAM_IMAGE = 0
DIT_STREAM_COLORMAP = 1
DIT_STREAM_CONTROL = 2
DIT_STREAM_MASK = 3


def is_dit_checkpoint(name_or_path: Optional[str]) -> bool:
    """Return True for FLUX Fill / UniGS-DiT Hub ids, shorthands, or local names."""
    if not name_or_path:
        return False
    if name_or_path in DIT_BACKBONES or name_or_path in DIT_BACKBONES.values():
        return True
    lowered = name_or_path.lower().replace("\\", "/")
    if lowered.rstrip("/").endswith("flux.1-fill-dev") or "flux.1-fill-dev" in lowered:
        return True
    if "flux1-fill" in lowered:
        return True
    return False


def is_unet_checkpoint(name_or_path: Optional[str]) -> bool:
    if not name_or_path:
        return False
    if name_or_path in UNET_BACKBONES or name_or_path in UNET_BACKBONES.values():
        return True
    return not is_dit_checkpoint(name_or_path)


def resolve_backbone(
    backbone: str = DEFAULT_BACKBONE,
    pretrained_model_name_or_path: Optional[str] = None,
) -> str:
    if pretrained_model_name_or_path is not None:
        return pretrained_model_name_or_path
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose one of {list(BACKBONES)} "
            "or pass an explicit `--pretrained_model_name_or_path`."
        )
    return BACKBONES[backbone]


def resolve_inpainting_checkpoint(
    backbone: str = DEFAULT_BACKBONE,
    pretrained_model_name_or_path: Optional[str] = None,
) -> str:
    """Backward-compatible alias for :func:`resolve_backbone`."""
    return resolve_backbone(
        backbone=backbone,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
