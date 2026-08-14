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

"""Supported UniGS backbones: SD inpainting UNets and DiT families.

UNet backbones must be *inpainting* checkpoints (9-in). DiT backbones may be
text-to-image or inpainting checkpoints; UniGS never uses channel-concat
conditioning on DiT. Conditioning is always context-token / sequence concat
(see ``transformer.py`` / ``dit.py``).
"""

from __future__ import annotations

import json
import os
from typing import Optional


# UniGS UNets are initialized from inpainting checkpoints, not base text-to-image.
UNET_BACKBONES = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "sd21": "stabilityai/stable-diffusion-2-inpainting",
}

# DiT families. `flux` is an inpainting Fill checkpoint (channel-concat is
# discarded). SD3.5 / Z-Image / PixArt-α are text-to-image DiTs adapted the
# same way: native in_channels, condition via extra visual token streams.
DIT_BACKBONES = {
    "flux": "black-forest-labs/FLUX.1-Fill-dev",
    "flux_fill": "black-forest-labs/FLUX.1-Fill-dev",
    "sd3": "stabilityai/stable-diffusion-3.5-medium",
    "sd3.5": "stabilityai/stable-diffusion-3.5-medium",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "sd3_medium": "stabilityai/stable-diffusion-3.5-medium",
    "z_image": "Tongyi-MAI/Z-Image-Turbo",
    "z-image": "Tongyi-MAI/Z-Image-Turbo",
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
    "pixart": "PixArt-alpha/PixArt-XL-2-512-MS",
    "pixart_alpha": "PixArt-alpha/PixArt-XL-2-512-MS",
    "pixart_512": "PixArt-alpha/PixArt-XL-2-512-MS",
    "pixart_1024": "PixArt-alpha/PixArt-XL-2-1024-MS",
}

# Canonical family name for each shorthand in DIT_BACKBONES.
DIT_BACKBONE_FAMILIES = {
    "flux": "flux",
    "flux_fill": "flux",
    "sd3": "sd3",
    "sd3.5": "sd3",
    "sd35": "sd3",
    "sd3_medium": "sd3",
    "z_image": "z_image",
    "z-image": "z_image",
    "zimage": "z_image",
    "pixart": "pixart",
    "pixart_alpha": "pixart",
    "pixart_512": "pixart",
    "pixart_1024": "pixart",
}

DIT_FAMILY_CHECKPOINTS = {
    "flux": "black-forest-labs/FLUX.1-Fill-dev",
    "sd3": "stabilityai/stable-diffusion-3.5-medium",
    "z_image": "Tongyi-MAI/Z-Image-Turbo",
    "pixart": "PixArt-alpha/PixArt-XL-2-512-MS",
}

DIT_FAMILY_DEFAULT_GUIDANCE = {
    "flux": 30.0,
    "sd3": 4.5,
    "z_image": 0.0,
    "pixart": 4.5,
}

DIT_FAMILY_MAX_SEQUENCE_LENGTH = {
    "flux": 512,
    "sd3": 256,
    "z_image": 512,
    "pixart": 120,
}

DIT_FLOW_FAMILIES = frozenset({"flux", "sd3", "z_image"})
DIT_EPSILON_FAMILIES = frozenset({"pixart"})

BACKBONES = {**UNET_BACKBONES, **DIT_BACKBONES}
INPAINTING_BACKBONES = BACKBONES  # backward-compatible alias

DEFAULT_BACKBONE = "sd15"
INPAINTING_UNET_IN_CHANNELS = 9

FLUX_FILL_CHECKPOINT = DIT_FAMILY_CHECKPOINTS["flux"]
SD3_CHECKPOINT = DIT_FAMILY_CHECKPOINTS["sd3"]
ZIMAGE_CHECKPOINT = DIT_FAMILY_CHECKPOINTS["z_image"]
PIXART_CHECKPOINT = DIT_FAMILY_CHECKPOINTS["pixart"]
PIXART_1024_CHECKPOINT = "PixArt-alpha/PixArt-XL-2-1024-MS"

FLUX_LATENT_CHANNELS = 16
FLUX_FILL_PACKED_IN_CHANNELS = 384  # packed noisy (64) + masked image (64) + mask (256)
UNIGS_DIT_PACKED_IN_CHANNELS = 64  # packed 16-channel latents only
UNIGS_DIT_PACKED_OUT_CHANNELS = 64

SD3_LATENT_CHANNELS = 16
ZIMAGE_LATENT_CHANNELS = 16
PIXART_LATENT_CHANNELS = 4

# RoPE / stream ids for concatenated visual tokens (Kontext-style first axis).
DIT_STREAM_IMAGE = 0
DIT_STREAM_COLORMAP = 1
DIT_STREAM_CONTROL = 2
DIT_STREAM_MASK = 3

_TRANSFORMER_CLASS_TO_FAMILY = {
    "FluxTransformer2DModel": "flux",
    "SD3Transformer2DModel": "sd3",
    "ZImageTransformer2DModel": "z_image",
    "PixArtTransformer2DModel": "pixart",
}


def _normalize_ckpt_id(name_or_path: str) -> str:
    return name_or_path.lower().replace("\\", "/").rstrip("/")


def resolve_dit_family(name_or_path: Optional[str]) -> Optional[str]:
    """Return canonical DiT family (`flux` / `sd3` / `z_image` / `pixart`) or None."""
    if not name_or_path:
        return None
    if name_or_path in DIT_BACKBONE_FAMILIES:
        return DIT_BACKBONE_FAMILIES[name_or_path]
    if name_or_path in DIT_FAMILY_CHECKPOINTS:
        return name_or_path
    for shorthand, checkpoint in DIT_BACKBONES.items():
        if name_or_path == checkpoint:
            return DIT_BACKBONE_FAMILIES[shorthand]

    lowered = _normalize_ckpt_id(name_or_path)
    basename = lowered.split("/")[-1]

    if basename.endswith("flux.1-fill-dev") or "flux.1-fill-dev" in lowered or "flux1-fill" in lowered:
        return "flux"
    if "z-image" in lowered or "z_image" in lowered or basename in {"zimage", "z-image-turbo"}:
        return "z_image"
    if "pixart" in lowered:
        return "pixart"
    if "stable-diffusion-3.5" in lowered or "stable-diffusion-3" in lowered:
        return "sd3"
    if basename in {"sd3.5-medium", "sd3-medium", "sd3.5", "sd35"}:
        return "sd3"
    return None


def detect_dit_family_from_path(path: Optional[str]) -> Optional[str]:
    """Resolve family from a Hub id, shorthand, or local UniGS / Diffusers dir."""
    family = resolve_dit_family(path)
    if family or not path:
        return family
    config_path = os.path.join(path, "transformer", "config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    family = config.get("unigs_dit_family")
    if family in DIT_FAMILY_CHECKPOINTS:
        return family
    return _TRANSFORMER_CLASS_TO_FAMILY.get(config.get("_class_name", ""))


def is_dit_checkpoint(name_or_path: Optional[str]) -> bool:
    """Return True for UniGS DiT Hub ids, shorthands, or local transformer dirs."""
    return detect_dit_family_from_path(name_or_path) is not None


def is_unet_checkpoint(name_or_path: Optional[str]) -> bool:
    if not name_or_path:
        return False
    if name_or_path in UNET_BACKBONES or name_or_path in UNET_BACKBONES.values():
        return True
    return not is_dit_checkpoint(name_or_path)


def dit_uses_flow_matching(family: Optional[str]) -> bool:
    return family in DIT_FLOW_FAMILIES


def default_dit_guidance(family: Optional[str]) -> float:
    return float(DIT_FAMILY_DEFAULT_GUIDANCE.get(family or "", 7.5))


def default_dit_max_sequence_length(family: Optional[str]) -> int:
    return int(DIT_FAMILY_MAX_SEQUENCE_LENGTH.get(family or "", 77))


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
