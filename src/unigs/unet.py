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

"""Adapt a Stable Diffusion 1.5 / 2.1 UNet to UniGS 13-in / 8-out channels.

UniGS concatenates, in this order (Eq. 8):

    z_t^i  (4)  noised image latents
    z_t^s  (4)  noised colormap latents
    m^c    (1)  coarse mask at latent resolution
    z^c    (4)  control latents (masked image, colormap, or full image)

and predicts 8 channels ``[image, colormap]``. Newly added weights are zeroed
as in InstructPix2Pix / the UniGS paper ("weight newly added channels as zero").
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from diffusers import UNet2DConditionModel


logger = logging.getLogger(__name__)

UNIGS_IN_CHANNELS = 13
UNIGS_OUT_CHANNELS = 8
LATENT_CHANNELS = 4


def _copy_conv_like(src: nn.Conv2d, dst: nn.Conv2d) -> None:
    dst.weight.zero_()
    if dst.bias is not None and src.bias is not None:
        dst.bias.zero_()
        n = min(src.bias.shape[0], dst.bias.shape[0])
        dst.bias[:n].copy_(src.bias[:n])


def adapt_unigs_unet(unet: UNet2DConditionModel, zero_init: bool = True) -> UNet2DConditionModel:
    """Expand ``conv_in`` / ``conv_out`` of an SD 1.5 or SD 2.1 UNet in-place.

    Supported source checkpoints:

    * text-to-image (4 in / 4 out) — extra in/out channels are zero-initialized
    * inpainting (9 in / 4 out) — noisy image, mask, and control channels are
      copied from the inpainting UNet; colormap channels are zero-initialized
    * already-adapted UniGS (13 in / 8 out) — returned unchanged
    """
    old_in = int(unet.config.in_channels)
    old_out = int(unet.config.out_channels)

    if old_in == UNIGS_IN_CHANNELS and old_out == UNIGS_OUT_CHANNELS:
        logger.info("UNet already has UniGS channels (13 in / 8 out).")
        return unet

    if old_in not in (4, 9, UNIGS_IN_CHANNELS):
        raise ValueError(
            f"Unsupported UNet `in_channels={old_in}`. Expected 4 (SD), 9 (SD inpainting), or 13 (UniGS)."
        )
    if old_out not in (4, UNIGS_OUT_CHANNELS):
        raise ValueError(
            f"Unsupported UNet `out_channels={old_out}`. Expected 4 (SD) or 8 (UniGS)."
        )

    logger.info(
        "Adapting UNet from %s-in/%s-out to UniGS %s-in/%s-out (zero-init extra channels=%s).",
        old_in,
        old_out,
        UNIGS_IN_CHANNELS,
        UNIGS_OUT_CHANNELS,
        zero_init,
    )

    if old_in != UNIGS_IN_CHANNELS:
        conv_in = unet.conv_in
        new_conv_in = nn.Conv2d(
            UNIGS_IN_CHANNELS,
            conv_in.out_channels,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
            bias=conv_in.bias is not None,
        )
        new_conv_in = new_conv_in.to(device=conv_in.weight.device, dtype=conv_in.weight.dtype)
        with torch.no_grad():
            _copy_conv_like(conv_in, new_conv_in)
            # [0:4] noisy image latents — always present.
            copy_in = min(old_in, LATENT_CHANNELS)
            new_conv_in.weight[:, :copy_in].copy_(conv_in.weight[:, :copy_in])
            if old_in == 9:
                # SD inpainting layout: [noisy(4), mask(1), masked_image(4)]
                # UniGS layout:         [noisy_img(4), noisy_cmap(4), mask(1), control(4)]
                new_conv_in.weight[:, 8:9].copy_(conv_in.weight[:, 4:5])
                new_conv_in.weight[:, 9:13].copy_(conv_in.weight[:, 5:9])
            elif not zero_init:
                nn.init.kaiming_normal_(new_conv_in.weight[:, copy_in:])
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=UNIGS_IN_CHANNELS)

    if old_out != UNIGS_OUT_CHANNELS:
        conv_out = unet.conv_out
        new_conv_out = nn.Conv2d(
            conv_out.in_channels,
            UNIGS_OUT_CHANNELS,
            kernel_size=conv_out.kernel_size,
            stride=conv_out.stride,
            padding=conv_out.padding,
            bias=conv_out.bias is not None,
        )
        new_conv_out = new_conv_out.to(device=conv_out.weight.device, dtype=conv_out.weight.dtype)
        with torch.no_grad():
            _copy_conv_like(conv_out, new_conv_out)
            new_conv_out.weight[:old_out].copy_(conv_out.weight[:old_out])
            if not zero_init:
                nn.init.kaiming_normal_(new_conv_out.weight[old_out:])
        unet.conv_out = new_conv_out
        unet.register_to_config(out_channels=UNIGS_OUT_CHANNELS)

    return unet
