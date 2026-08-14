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

"""Family-specific DiT forwards for UniGS **context-token** concatenation.

FLUX.1-Fill-dev is the exception: it stays on Fill-style **channel** concat in
:mod:`unigs.transformer`. This module covers spatial DiTs:

* **SD 3.5** (``SD3Transformer2DModel``) — patch-embed each UniGS stream at its
  native HxW, add a zero-init stream embedding, concat on the sequence axis,
  unpatchify only the image + colormap prefix. Spatial stacking is impossible:
  ``pos_embed_max_size=96`` cannot hold 4×64.
* **PixArt-α** (``PixArtTransformer2DModel``) — same pos-embed-then-concat
  pattern. Output may include learned sigma (``out_channels=8``); only the
  first half is the epsilon / velocity used by the scheduler.
* **Z-Image** (``ZImageTransformer2DModel``) — native omni mode already accepts
  a nested list of images plus ``image_noise_mask``. Control + coarse mask are
  clean context streams; image and colormap are stacked on height as the single
  noisy target (omni unpatchify returns only the last stream).

None of these paths channel-concat the condition into ``in_channels``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import (
    DIT_STREAM_COLORMAP,
    DIT_STREAM_CONTROL,
    DIT_STREAM_IMAGE,
    DIT_STREAM_MASK,
)


logger = logging.getLogger(__name__)

PromptBatch = Union[str, List[str]]


def transformer_inner_dim(transformer) -> int:
    inner = getattr(transformer, "inner_dim", None)
    if inner is not None:
        return int(inner)
    config = transformer.config
    return int(config.num_attention_heads) * int(config.attention_head_dim)


def transformer_patch_size(transformer) -> int:
    return int(getattr(transformer.config, "patch_size", 2) or 2)


def transformer_out_channels(transformer) -> int:
    config = transformer.config
    out = getattr(config, "out_channels", None)
    if out is None:
        out = getattr(transformer, "out_channels", None)
    if out is None:
        out = getattr(config, "in_channels", 16)
    return int(out)


def infer_transformer_family(transformer) -> Optional[str]:
    family = getattr(getattr(transformer, "config", None), "unigs_dit_family", None)
    if family:
        return family
    name = transformer.__class__.__name__
    return {
        "FluxTransformer2DModel": "flux",
        "SD3Transformer2DModel": "sd3",
        "ZImageTransformer2DModel": "z_image",
        "PixArtTransformer2DModel": "pixart",
    }.get(name)


def ensure_unigs_stream_embed(transformer, num_streams: int = 4) -> nn.Parameter:
    """Zero-init per-stream bias so stream 0 matches the pretrained image path."""
    existing = getattr(transformer, "unigs_stream_embed", None)
    if isinstance(existing, nn.Parameter) and existing.shape[0] >= num_streams:
        return existing
    inner_dim = transformer_inner_dim(transformer)
    param = next(transformer.parameters())
    embed = nn.Parameter(torch.zeros(num_streams, 1, inner_dim, device=param.device, dtype=param.dtype))
    transformer.register_parameter("unigs_stream_embed", embed)
    return embed


def resize_mask_to_latents(
    mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    latent_channels: int,
) -> torch.Tensor:
    """Nearest-resize a coarse mask and repeat it across latent channels."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask.float(), size=(latent_height, latent_width), mode="nearest")
    return mask.expand(-1, latent_channels, -1, -1).to(dtype=mask.dtype)


def unpatchify_stream_tokens(
    hidden_states: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
    out_channels: int,
    num_streams: int = 2,
) -> List[torch.Tensor]:
    """Unpatchify the leading ``num_streams`` token groups back to ``[B, C, H, W]``.

    ``hidden_states`` is ``[B, N, patch_size**2 * out_channels]`` after ``proj_out``.
    Context tokens beyond the image + colormap prefix are dropped.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected packed tokens `[B, N, D]`, got {tuple(hidden_states.shape)}")
    patch_h = height // patch_size
    patch_w = width // patch_size
    seq = patch_h * patch_w
    target_len = num_streams * seq
    if hidden_states.shape[1] < target_len:
        raise ValueError(
            f"Token length {hidden_states.shape[1]} is shorter than {num_streams} streams of {seq} patches."
        )
    streams = []
    for index in range(num_streams):
        tokens = hidden_states[:, index * seq : (index + 1) * seq]
        tokens = tokens.reshape(
            hidden_states.shape[0], patch_h, patch_w, patch_size, patch_size, out_channels
        )
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        streams.append(tokens.reshape(hidden_states.shape[0], out_channels, height, width))
    return streams


def maybe_drop_learned_sigma(sample: torch.Tensor, latent_channels: int) -> torch.Tensor:
    if sample.shape[1] == latent_channels * 2:
        return sample.chunk(2, dim=1)[0]
    return sample


def _embed_spatial_stream(transformer, latents: torch.Tensor, stream_id: int) -> torch.Tensor:
    tokens = transformer.pos_embed(latents)
    stream_embed = getattr(transformer, "unigs_stream_embed", None)
    if stream_embed is not None:
        tokens = tokens + stream_embed[stream_id].to(dtype=tokens.dtype, device=tokens.device)
    return tokens


def _concat_unigs_streams(
    transformer,
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    hidden_states = torch.cat(
        [
            _embed_spatial_stream(transformer, image, DIT_STREAM_IMAGE),
            _embed_spatial_stream(transformer, colormap, DIT_STREAM_COLORMAP),
            _embed_spatial_stream(transformer, control, DIT_STREAM_CONTROL),
            _embed_spatial_stream(transformer, mask, DIT_STREAM_MASK),
        ],
        dim=1,
    )
    height, width = image.shape[-2], image.shape[-1]
    return hidden_states.contiguous(), height, width


def forward_sd3_token_concat(
    transformer,
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
):
    """SD3 MMDiT forward with four visual streams concatenated after patch embed."""
    hidden_states, height, width = _concat_unigs_streams(transformer, image, colormap, control, mask)
    temb = transformer.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        joint_attention_kwargs = dict(joint_attention_kwargs)
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states, ip_temb = transformer.image_proj(ip_adapter_image_embeds, timestep)
        joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and getattr(transformer, "gradient_checkpointing", False):
            ckpt = getattr(transformer, "_gradient_checkpointing_func", None)
            if ckpt is not None:
                encoder_hidden_states, hidden_states = ckpt(
                    block, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs
                )
            else:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    use_reentrant=False,
                )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    hidden_states = transformer.norm_out(hidden_states, temb)
    hidden_states = transformer.proj_out(hidden_states)
    image_pred, colormap_pred = unpatchify_stream_tokens(
        hidden_states,
        height=height,
        width=width,
        patch_size=transformer_patch_size(transformer),
        out_channels=transformer_out_channels(transformer),
        num_streams=2,
    )
    sample = torch.cat([image_pred, colormap_pred], dim=1)
    if return_dict:
        return {"sample": sample}
    return (sample,)


def forward_pixart_token_concat(
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
    """PixArt-α DiT forward with four visual streams concatenated after patch embed."""
    if getattr(transformer, "use_additional_conditions", False) and added_cond_kwargs is None:
        raise ValueError("`added_cond_kwargs` is required for PixArt checkpoints with sample_size=128.")

    hidden_states, height, width = _concat_unigs_streams(transformer, image, colormap, control, mask)
    batch_size = hidden_states.shape[0]
    dtype = hidden_states.dtype

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    timestep_emb, embedded_timestep = transformer.adaln_single(
        timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=dtype
    )
    if transformer.caption_projection is not None:
        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and getattr(transformer, "gradient_checkpointing", False):
            ckpt = getattr(transformer, "_gradient_checkpointing_func", None)
            if ckpt is not None:
                hidden_states = ckpt(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_emb,
                    cross_attention_kwargs,
                    None,
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_emb,
                    cross_attention_kwargs,
                    None,
                    use_reentrant=False,
                )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

    shift, scale = (transformer.scale_shift_table[None] + embedded_timestep[:, None].to(transformer.scale_shift_table.device)).chunk(
        2, dim=1
    )
    hidden_states = transformer.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
    hidden_states = transformer.proj_out(hidden_states)
    hidden_states = hidden_states.squeeze(1)

    image_pred, colormap_pred = unpatchify_stream_tokens(
        hidden_states,
        height=height,
        width=width,
        patch_size=transformer_patch_size(transformer),
        out_channels=transformer_out_channels(transformer),
        num_streams=2,
    )
    latent_channels = image.shape[1]
    image_pred = maybe_drop_learned_sigma(image_pred, latent_channels)
    colormap_pred = maybe_drop_learned_sigma(colormap_pred, latent_channels)
    sample = torch.cat([image_pred, colormap_pred], dim=1)
    if return_dict:
        return {"sample": sample}
    return (sample,)


def _as_zimage_5d(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` → ``[B, C, 1, H, W]`` (Z-Image temporal axis)."""
    if latents.ndim == 4:
        return latents.unsqueeze(2)
    if latents.ndim == 5:
        return latents
    raise ValueError(f"Z-Image latents must be 4D or 5D, got {tuple(latents.shape)}")


def prepare_zimage_omni_inputs(
    image: torch.Tensor,
    colormap: torch.Tensor,
    control: torch.Tensor,
    mask: torch.Tensor,
    prompt_embeds: Sequence[torch.Tensor],
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[int]], int]:
    """Pack UniGS streams into Z-Image omni nested lists.

    Omni ``unpatchify`` returns only the last image, so image + colormap are
    stacked on height as the noisy target. Control and mask stay clean context.
    """
    image_5d = _as_zimage_5d(image)
    colormap_5d = _as_zimage_5d(colormap)
    control_5d = _as_zimage_5d(control)
    mask_5d = _as_zimage_5d(mask)
    target = torch.cat([image_5d, colormap_5d], dim=-2)
    latent_height = image_5d.shape[-2]
    x: List[List[torch.Tensor]] = []
    cap: List[List[torch.Tensor]] = []
    noise_mask: List[List[int]] = []
    for index in range(image.shape[0]):
        caption = prompt_embeds[index]
        x.append([control_5d[index], mask_5d[index], target[index]])
        cap.append([caption, caption, caption])
        noise_mask.append([0, 0, 1])
    return x, cap, noise_mask, latent_height


def split_zimage_omni_output(
    model_out: Sequence[torch.Tensor],
    latent_height: int,
    negate: bool = True,
) -> torch.Tensor:
    """Stack omni outputs, drop the temporal axis, split image / colormap, optionally negate."""
    stacked = torch.stack([tensor.float() for tensor in model_out], dim=0)
    if stacked.ndim == 5:
        stacked = stacked.squeeze(2)
    if negate:
        stacked = -stacked
    image_pred, colormap_pred = stacked.split(latent_height, dim=-2)
    return torch.cat([image_pred, colormap_pred], dim=1)


def forward_zimage_omni(
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
    """Z-Image omni forward: control + mask context, stacked image/colormap target."""
    x, cap_feats, image_noise_mask, latent_height = prepare_zimage_omni_inputs(
        image, colormap, control, mask, prompt_embeds
    )
    model_out = transformer(
        x,
        timestep,
        cap_feats,
        return_dict=False,
        image_noise_mask=image_noise_mask,
    )[0]
    sample = split_zimage_omni_output(model_out, latent_height, negate=negate)
    sample = sample.to(dtype=image.dtype)
    if return_dict:
        return {"sample": sample}
    return (sample,)


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
