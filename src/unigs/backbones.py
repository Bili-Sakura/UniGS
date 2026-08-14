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

"""Supported Stable Diffusion *inpainting* backbones (paper default: SD 1.5 inpainting)."""

from __future__ import annotations

from typing import Optional

# UniGS is initialized from SD inpainting UNets (9 input channels), not base text-to-image.
INPAINTING_BACKBONES = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "sd21": "stabilityai/stable-diffusion-2-inpainting",
}

DEFAULT_BACKBONE = "sd15"
INPAINTING_UNET_IN_CHANNELS = 9


def resolve_inpainting_checkpoint(
    backbone: str = DEFAULT_BACKBONE,
    pretrained_model_name_or_path: Optional[str] = None,
) -> str:
    if pretrained_model_name_or_path is not None:
        return pretrained_model_name_or_path
    if backbone not in INPAINTING_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose one of {list(INPAINTING_BACKBONES)} "
            "or pass an explicit `--pretrained_model_name_or_path`."
        )
    return INPAINTING_BACKBONES[backbone]
