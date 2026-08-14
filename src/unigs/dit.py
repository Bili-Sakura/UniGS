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

"""Spatial DiT UniGS adapters: Fill-style **channel** concat on 4D maps.

FLUX packs tokens then concatenates on the last dim (see :mod:`unigs.transformer`).
SD 3.5, PixArt-α, and Z-Image stay in ``[B, C, H, W]`` and concatenate on
``dim=1`` the same way Fill concatenates on packed channels::

    hidden = cat(z_image, z_colormap, z_control, mask_1ch, dim=1)  # 3C+1
    pred   = transformer(hidden, ...)  # native 4D forward, no token concat
    pred_image, pred_cmap = split along channels  (2 × native out)

The native transformer is used as-is after expanding PatchEmbed / ``x_embedder``
and the final projection. No extra stream embeddings, no sequence concat, no
omni nested lists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import (
    SPATIAL_MASK_CHANNELS,
    native_in_channels,
    native_out_channels,
    spatial_unigs_in_channels,
    spatial_unigs_out_channels,
)


PromptBatch = Union[str, List[str]]


def concat_spatial_unigs(
    latents: torch.Tensor,
    colormap_latents: torch.Tensor,
    control_latents: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``[B, 3C+1, H, W]`` channel concat (Fill-style, spatial analog)."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != latents.shape[-2:]:
        mask = F.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest").to(dtype=latents.dtype)
    if mask.shape[1] != SPATIAL_MASK_CHANNELS:
        mask = mask[:, :SPATIAL_MASK_CHANNELS]
    return torch.cat([latents, colormap_latents, control_latents, mask], dim=1)


def split_dual_output(pred: torch.Tensor, native_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``[B, 2*native_out, H, W]`` into image and colormap halves."""
    if pred.shape[1] != 2 * native_out:
        raise ValueError(f"expected {2 * native_out} output channels, got {pred.shape[1]}")
    return pred[:, :native_out], pred[:, native_out:]


def maybe_drop_learned_sigma(sample: torch.Tensor, latent_channels: int) -> torch.Tensor:
    """PixArt learned-sigma: keep the first ``latent_channels`` of an 8-ch half."""
    if sample.shape[1] == latent_channels:
        return sample
    if sample.shape[1] == 2 * latent_channels:
        return sample.chunk(2, dim=1)[0]
    return sample


def resize_mask_to_latents(
    mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    latent_channels: int = SPATIAL_MASK_CHANNELS,
) -> torch.Tensor:
    """Nearest-resize a coarse mask to the latent grid.

    UniGS spatial DiTs concat a **1-channel** mask (Fill-style). ``latent_channels``
    other than 1 repeats the mask (kept for tests / debugging).
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask.float(), size=(latent_height, latent_width), mode="nearest")
    if latent_channels <= 1:
        return mask[:, :1].to(dtype=mask.dtype)
    return mask.expand(-1, latent_channels, -1, -1).to(dtype=mask.dtype)


def _copy_conv2d_unigs_in(src: nn.Conv2d, dst: nn.Conv2d, native_c: int) -> None:
    """Copy image (+ control) PatchEmbed channels; colormap and mask stay zero.

    ``dst`` is ``3C+1`` in. Image occupies ``0:C``; control occupies ``2C:3C``
    (Fill's masked-image → UniGS control).
    """
    with torch.no_grad():
        dst.weight.zero_()
        src_in = src.weight.shape[1]
        dst_in = dst.weight.shape[1]
        if src_in == native_c and dst_in == spatial_unigs_in_channels(native_c):
            dst.weight[:, :native_c] = src.weight
            dst.weight[:, 2 * native_c : 3 * native_c] = src.weight
        else:
            n_in = min(src_in, dst_in)
            dst.weight[:, :n_in] = src.weight[:, :n_in]
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


def _copy_linear_out(src: nn.Linear, dst: nn.Linear) -> None:
    with torch.no_grad():
        dst.weight.zero_()
        n_out = min(src.weight.shape[0], dst.weight.shape[0])
        dst.weight[:n_out] = src.weight[:n_out]
        if src.bias is not None and dst.bias is not None:
            dst.bias.zero_()
            dst.bias[:n_out] = src.bias[:n_out]


def _copy_linear_in_spatial(src: nn.Linear, dst: nn.Linear, native_c: int) -> None:
    """Copy pretrained patch-linear columns for image (+ control) channel slots.

    PatchEmbed flattens each patch as ``(C, ph, pw)`` then Linear. UniGS input
    is ``(3C+1, ph, pw)``. Image occupies the first ``C`` of each spatial cell;
    control occupies the next ``C`` after colormap; mask is last and stays 0.
    """
    with torch.no_grad():
        dst.weight.zero_()
        patch_elems = src.weight.shape[1]
        if patch_elems % native_c != 0:
            n_in = min(src.weight.shape[1], dst.weight.shape[1])
            dst.weight[:, :n_in] = src.weight[:, :n_in]
        else:
            spatial = patch_elems // native_c
            unigs_c = spatial_unigs_in_channels(native_c)
            if dst.weight.shape[1] != unigs_c * spatial:
                n_in = min(src.weight.shape[1], dst.weight.shape[1])
                dst.weight[:, :n_in] = src.weight[:, :n_in]
            else:
                src_w = src.weight.view(src.weight.shape[0], native_c, spatial)
                dst_w = dst.weight.view(dst.weight.shape[0], unigs_c, spatial)
                dst_w[:, :native_c] = src_w
                dst_w[:, 2 * native_c : 3 * native_c] = src_w
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


def _expand_conv_patch_embed(proj: nn.Conv2d, new_in: int) -> nn.Conv2d:
    new = nn.Conv2d(
        new_in,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    nn.init.zeros_(new.weight)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    return new


def _expand_linear(layer: nn.Linear, new_in: Optional[int] = None, new_out: Optional[int] = None) -> nn.Linear:
    in_f = new_in if new_in is not None else layer.in_features
    out_f = new_out if new_out is not None else layer.out_features
    new = nn.Linear(in_f, out_f, bias=layer.bias is not None)
    nn.init.zeros_(new.weight)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    return new


def adapt_spatial_transformer(transformer: nn.Module, family: str) -> nn.Module:
    """Expand a spatial DiT's patch embed + output proj to UniGS ``3C+1`` / ``2*out``."""
    native_in = native_in_channels(family)
    native_out = native_out_channels(family)
    new_in = spatial_unigs_in_channels(native_in)
    new_out = spatial_unigs_out_channels(native_out)

    in_ch = int(getattr(transformer.config, "in_channels", native_in))
    out_ch = int(getattr(transformer.config, "out_channels", native_out))
    if in_ch == new_in and out_ch == new_out:
        return transformer

    if family in ("sd3", "pixart"):
        if not hasattr(transformer, "pos_embed") or not hasattr(transformer.pos_embed, "proj"):
            raise TypeError(f"{family} transformer has no pos_embed.proj PatchEmbed")
        old_proj = transformer.pos_embed.proj
        transformer.pos_embed.proj = _expand_conv_patch_embed(old_proj, new_in)
        _copy_conv2d_unigs_in(old_proj, transformer.pos_embed.proj, native_in)
        old_out = transformer.proj_out
        transformer.proj_out = _expand_linear(old_out, new_out=new_out)
        _copy_linear_out(old_out, transformer.proj_out)
    elif family == "z_image":
        old_x = transformer.all_x_embedder
        patch_vol = old_x.in_features // native_in
        transformer.all_x_embedder = _expand_linear(old_x, new_in=new_in * patch_vol)
        _copy_linear_in_spatial(old_x, transformer.all_x_embedder, native_in)
        old_final = transformer.all_final_layer.linear
        transformer.all_final_layer.linear = _expand_linear(old_final, new_out=new_out * patch_vol)
        _copy_linear_out(old_final, transformer.all_final_layer.linear)
        transformer.out_channels = new_out
        transformer.in_channels = new_in
    else:
        raise ValueError(f"no spatial adapter for family {family!r}")

    if hasattr(transformer, "config"):
        transformer.config.in_channels = new_in
        transformer.config.out_channels = new_out
        transformer.config.unigs_dit_family = family
        if hasattr(transformer, "register_to_config"):
            transformer.register_to_config(
                in_channels=new_in, out_channels=new_out, unigs_dit_family=family
            )
    return transformer


def _as_zimage_5d(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` → ``[B, C, 1, H, W]`` (Z-Image temporal axis)."""
    if latents.ndim == 4:
        return latents.unsqueeze(2)
    if latents.ndim == 5:
        return latents
    raise ValueError(f"Z-Image latents must be 4D or 5D, got {tuple(latents.shape)}")


def _stack_zimage_out(model_out, negate: bool) -> torch.Tensor:
    if torch.is_tensor(model_out):
        stacked = model_out.float()
    else:
        stacked = torch.stack([tensor.float() for tensor in model_out], dim=0)
    if stacked.ndim == 5:
        stacked = stacked.squeeze(2)
    if negate:
        stacked = -stacked
    return stacked


def forward_sd3_channel_concat(
    transformer,
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    return_dict: bool = False,
):
    """Native SD3 forward on Fill-style channel-concatenated latents."""
    hidden = concat_spatial_unigs(image, colormap, control, mask)
    sample = transformer(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]
    if return_dict:
        return {"sample": sample}
    return (sample,)


def forward_pixart_channel_concat(
    transformer,
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = False,
):
    """Native PixArt forward on Fill-style channel-concatenated latents."""
    hidden = concat_spatial_unigs(image, colormap, control, mask)
    sample = transformer(
        hidden_states=hidden,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        added_cond_kwargs=added_cond_kwargs,
        encoder_attention_mask=encoder_attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        attention_mask=attention_mask,
        return_dict=False,
    )[0]
    image_pred, colormap_pred = sample.chunk(2, dim=1)
    latent_channels = image.shape[1]
    image_pred = maybe_drop_learned_sigma(image_pred, latent_channels)
    colormap_pred = maybe_drop_learned_sigma(colormap_pred, latent_channels)
    sample = torch.cat([image_pred, colormap_pred], dim=1)
    if return_dict:
        return {"sample": sample}
    return (sample,)


def forward_zimage_channel_concat(
    transformer,
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: Sequence[torch.Tensor],
    negate: bool = True,
    return_dict: bool = False,
):
    """Native Z-Image forward (list of 5D maps, **not** omni nested lists)."""
    hidden = concat_spatial_unigs(image, colormap, control, mask)
    x = list(_as_zimage_5d(hidden).unbind(0))
    model_out = transformer(x, timestep, list(prompt_embeds), return_dict=False)[0]
    sample = _stack_zimage_out(model_out, negate=negate).to(dtype=image.dtype)
    if return_dict:
        return {"sample": sample}
    return (sample,)


# Backward-compatible names used by older call sites / docs.
forward_sd3_token_concat = forward_sd3_channel_concat
forward_pixart_token_concat = forward_pixart_channel_concat
forward_zimage_omni = forward_zimage_channel_concat


def encode_sd3_prompt(
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    prompt: PromptBatch,
    device: torch.device,
    max_sequence_length: int = 256,
    num_images_per_prompt: int = 1,
    joint_attention_dim: int = 4096,
):
    """CLIP-L + CLIP-G pooled/sequence + T5, matching Diffusers SD3 ``encode_prompt``."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompts)

    def _clip_encode(encoder, tok, texts):
        inputs = tok(
            texts,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        outputs = encoder(inputs.input_ids.to(device), output_hidden_states=True)
        pooled = outputs[0]
        hidden = outputs.hidden_states[-2]
        hidden = hidden.to(dtype=encoder.dtype, device=device)
        _, seq_len, _ = hidden.shape
        hidden = hidden.repeat(1, num_images_per_prompt, 1)
        hidden = hidden.view(batch_size * num_images_per_prompt, seq_len, -1)
        pooled = pooled.repeat(1, num_images_per_prompt)
        pooled = pooled.view(batch_size * num_images_per_prompt, -1)
        return hidden, pooled

    clip_hidden, clip_pooled = _clip_encode(text_encoder, tokenizer, prompts)
    clip_hidden_2, clip_pooled_2 = _clip_encode(text_encoder_2, tokenizer_2, prompts)
    clip_hidden = torch.cat([clip_hidden, clip_hidden_2], dim=-1)
    pooled = torch.cat([clip_pooled, clip_pooled_2], dim=-1)

    if text_encoder_3 is None or tokenizer_3 is None:
        t5_hidden = torch.zeros(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            joint_attention_dim,
            device=device,
            dtype=clip_hidden.dtype,
        )
    else:
        t5_inputs = tokenizer_3(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        t5_hidden = text_encoder_3(t5_inputs.input_ids.to(device))[0]
        t5_hidden = t5_hidden.to(dtype=text_encoder_3.dtype, device=device)
        _, seq_len, _ = t5_hidden.shape
        t5_hidden = t5_hidden.repeat(1, num_images_per_prompt, 1)
        t5_hidden = t5_hidden.view(batch_size * num_images_per_prompt, seq_len, -1)

    clip_hidden = F.pad(clip_hidden, (0, t5_hidden.shape[-1] - clip_hidden.shape[-1]))
    prompt_embeds = torch.cat([clip_hidden, t5_hidden], dim=-2)
    return prompt_embeds, pooled


def encode_pixart_prompt(
    text_encoder,
    tokenizer,
    prompt: PromptBatch,
    device: torch.device,
    max_sequence_length: int = 120,
    num_images_per_prompt: int = 1,
):
    """T5 sequence embeddings + attention mask for PixArt-α."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    batch_size = len(prompts)
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    attention_mask = inputs.attention_mask.to(device)
    hidden = text_encoder(inputs.input_ids.to(device), attention_mask=attention_mask)[0]
    hidden = hidden.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = hidden.shape
    hidden = hidden.repeat(1, num_images_per_prompt, 1)
    hidden = hidden.view(batch_size * num_images_per_prompt, seq_len, -1)
    attention_mask = attention_mask.repeat(1, num_images_per_prompt)
    attention_mask = attention_mask.view(batch_size * num_images_per_prompt, -1)
    return hidden, attention_mask


def encode_zimage_prompt(
    text_encoder,
    tokenizer,
    prompt: PromptBatch,
    device: torch.device,
    max_sequence_length: int = 512,
) -> List[torch.Tensor]:
    """Qwen chat-template embeddings; returns a variable-length list like ZImagePipeline."""
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    formatted = []
    for item in prompts:
        messages = [{"role": "user", "content": item}]
        try:
            formatted.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            )
        except TypeError:
            formatted.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
    inputs = tokenizer(
        formatted,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).bool()
    hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-2]
    return [hidden[i][attention_mask[i]] for i in range(hidden.shape[0])]


def pixart_added_cond_kwargs(
    transformer,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    cfg_multiplier: int = 1,
) -> Optional[Dict[str, torch.Tensor]]:
    if not getattr(transformer, "use_additional_conditions", False) and int(
        getattr(transformer.config, "sample_size", 0) or 0
    ) != 128:
        return {"resolution": None, "aspect_ratio": None}
    resolution = torch.tensor([height, width], device=device, dtype=dtype).repeat(batch_size * cfg_multiplier, 1)
    aspect_ratio = torch.tensor([float(height) / float(width)], device=device, dtype=dtype).repeat(
        batch_size * cfg_multiplier, 1
    )
    return {"resolution": resolution, "aspect_ratio": aspect_ratio}
