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

"""Fine-tune Stable Diffusion inpainting UNets or DiT backbones as UniGS.

UNet example:
    accelerate launch src/train_unigs.py \\
        --backbone=sd15 \\
        --coco_image_dir=/data/coco/train2017 \\
        --coco_annotation_file=/data/coco/annotations/instances_train2017.json \\
        --output_dir=unigs-sd15 \\
        --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=4 \\
        --learning_rate=5e-5 --max_train_steps=30000 --checkpointing_steps=5000 \\
        --mixed_precision=fp16 --task=joint

FLUX Fill DiT example (Fill-style channel concat; LoRA recommended):
    accelerate launch src/train_unigs.py \\
        --backbone=flux \\
        --coco_image_dir=/data/coco/train2017 \\
        --coco_annotation_file=/data/coco/annotations/instances_train2017.json \\
        --output_dir=unigs-flux-fill \\
        --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 \\
        --learning_rate=1e-4 --lora_rank=16 --max_train_steps=10000 \\
        --mixed_precision=bf16 --gradient_checkpointing --task=joint

SD 3.5 Medium / Z-Image / PixArt-α are text-to-image DiTs; UniGS conditions
those with context-token concat (not channel concat):

    accelerate launch src/train_unigs.py --backbone=sd3 ... --mixed_precision=bf16
    accelerate launch src/train_unigs.py --backbone=z_image ... --mixed_precision=bf16
    accelerate launch src/train_unigs.py --backbone=pixart ... --mixed_precision=fp16
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unigs.backbones import (
    INPAINTING_BACKBONES,
    INPAINTING_UNET_IN_CHANNELS,
    default_dit_max_sequence_length,
    dit_uses_flow_matching,
    is_dit_checkpoint,
    resolve_dit_family,
    resolve_inpainting_checkpoint,
)
from unigs.dataset import UniGSInstanceDataset, collate_fn, load_coco_records, records_from_hf_dataset
from unigs.pipeline_unigs import UniGSPipeline
from unigs.prompts import TASK_NAMES, TASK_PROMPT_TEMPLATES
from unigs.dit import (
    encode_pixart_prompt,
    encode_sd3_prompt,
    encode_zimage_prompt,
    forward_pixart_token_concat,
    forward_sd3_token_concat,
    forward_zimage_omni,
    pixart_added_cond_kwargs,
    resize_mask_to_latents,
)
from unigs.transformer import (
    adapt_unigs_transformer,
    concat_fill_channels,
    encode_vae_latents,
    pack_fill_mask,
    pack_latents,
    prepare_latent_image_ids,
    split_packed_pred,
    unpack_latents,
)
from unigs.unet import adapt_unigs_unet


if is_wandb_available():
    import wandb

try:
    check_min_version("0.27.0")
except Exception:
    pass

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniGS on an SD inpainting UNet or FLUX Fill DiT backbone.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="sd15",
        choices=list(INPAINTING_BACKBONES),
        help=(
            "Checkpoint shorthand when --pretrained_model_name_or_path is omitted "
            "(`sd15`, `sd21`, `flux`, `sd3`, `z_image`, `pixart`)."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Hub id or local path. UNet inpainting: "
            f"{INPAINTING_BACKBONES['sd15']} / {INPAINTING_BACKBONES['sd21']}. "
            "DiT: FLUX uses Fill-style channel concat; sd3 / z_image / pixart use "
            "context-token concat. "
            f"flux={INPAINTING_BACKBONES['flux']}, "
            f"sd3={INPAINTING_BACKBONES['sd3']}, "
            f"z_image={INPAINTING_BACKBONES['z_image']}, "
            f"pixart={INPAINTING_BACKBONES['pixart']}. "
            "Overrides --backbone when set."
        ),
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="joint",
        choices=list(TASK_NAMES) + ["joint"],
        help="Single UniGS task, or `joint` to mix the four Table-2 tasks.",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset id.")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--mask_column", type=str, default="masks")
    parser.add_argument("--label_column", type=str, default="labels")
    parser.add_argument("--objects_column", type=str, default="objects")
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default=None)
    parser.add_argument("--coco_annotation_file", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unigs-model",
        help="Where to save the UniGS pipeline (`save_pretrained`).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--max_entities", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=11)
    parser.add_argument("--arbitrary_mask_prob", type=float, default=0.5)
    parser.add_argument("--referring_neg_prob", type=float, default=0.2)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=48)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1)
    parser.add_argument("--loss_on_mask_only", action="store_true")
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_image", type=str, default=None)
    parser.add_argument("--validation_mask", type=str, default=None)
    parser.add_argument("--validation_colormap", type=str, default=None)
    parser.add_argument("--validation_task", type=str, default="inpainting", choices=list(TASK_NAMES))
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--tracker_project_name", type=str, default="unigs")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="T5 / Qwen token length for DiT backbones. Ignored for SD UNet backbones.",
    )
    parser.add_argument(
        "--train_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance embedding value while training distilled FLUX Fill. Ignored for other backbones.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Flow-matching timestep sampling / loss weighting (FLUX / SD3 / Z-Image).",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="If set, train LoRA on DiT attention projections (input adapters stay fully trainable).",
    )
    parser.add_argument("--lora_alpha", type=int, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and (args.coco_image_dir is None or args.coco_annotation_file is None):
        raise ValueError("Provide `--dataset_name` or both `--coco_image_dir` and `--coco_annotation_file`.")
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def _import_flux():
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from transformers import T5EncoderModel, T5TokenizerFast
    except ImportError as err:
        raise ImportError(
            "FLUX UniGS training requires diffusers>=0.32.0 with FluxTransformer2DModel "
            "and transformers T5EncoderModel / T5TokenizerFast."
        ) from err
    return FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, T5EncoderModel, T5TokenizerFast


def _apply_dit_lora(transformer, rank: int, alpha: Optional[int] = None):
    try:
        from peft import LoraConfig
    except ImportError as err:
        raise ImportError("Install peft to train DiT UniGS with LoRA: `pip install peft`.") from err
    transformer.requires_grad_(False)
    if getattr(transformer, "x_embedder", None) is not None:
        transformer.x_embedder.requires_grad_(True)
    if getattr(transformer, "unigs_stream_embed", None) is not None:
        transformer.unigs_stream_embed.requires_grad_(True)
    if getattr(transformer, "all_x_embedder", None) is not None:
        transformer.all_x_embedder.requires_grad_(True)
    pos_embed = getattr(transformer, "pos_embed", None)
    if pos_embed is not None and getattr(pos_embed, "proj", None) is not None:
        pos_embed.proj.requires_grad_(True)
    if getattr(transformer, "proj_out", None) is not None:
        transformer.proj_out.requires_grad_(True)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha or rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    return transformer


def _trainable_parameters(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return params


def _dit_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _sample_flow_timesteps(noise_scheduler, batch_size, device, args):
    try:
        from diffusers.training_utils import compute_density_for_timestep_sampling

        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
    except Exception:
        u = torch.rand(batch_size, device=device)
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    return noise_scheduler.timesteps[indices].to(device=device)


def _flow_loss_weighting(sigmas, args):
    try:
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        return compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    except Exception:
        return torch.ones_like(sigmas)


def _encode_flux_batch(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompts, device, max_sequence_length):
    from unigs.pipeline_unigs_flux import encode_flux_prompt

    return encode_flux_prompt(
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        prompts,
        device,
        max_sequence_length=max_sequence_length,
    )


def _unet_training_step(batch, args, vae, unet, text_encoder, tokenizer, noise_scheduler, weight_dtype):
    image_latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
    image_latents = image_latents * vae.config.scaling_factor
    colormap_latents = vae.encode(batch["colormap_values"].to(weight_dtype)).latent_dist.sample()
    colormap_latents = colormap_latents * vae.config.scaling_factor
    latents = torch.cat([image_latents, colormap_latents], dim=1)

    noise = torch.randn_like(latents)
    if args.noise_offset:
        noise = noise + args.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
    control_latents = vae.encode(batch["control_values"].to(weight_dtype)).latent_dist.mode()
    control_latents = control_latents * vae.config.scaling_factor
    coarse_mask = F.interpolate(
        batch["coarse_mask"].to(dtype=weight_dtype, device=latents.device),
        size=noisy_latents.shape[-2:],
        mode="nearest",
    )

    if args.conditioning_dropout_prob is not None and args.conditioning_dropout_prob > 0:
        random_p = torch.rand(bsz, device=latents.device)
        prompt_mask = (random_p < args.conditioning_dropout_prob).reshape(bsz, 1, 1)
        drop_ids = tokenizer(
            [""] * bsz,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(latents.device)
        null_conditioning = text_encoder(drop_ids)[0]
        encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

    model_input = torch.cat([noisy_latents, coarse_mask, control_latents], dim=1)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    model_pred = unet(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
    if args.loss_on_mask_only:
        loss_mask = coarse_mask.expand_as(model_pred)
        loss = F.mse_loss(model_pred.float() * loss_mask, target.float() * loss_mask, reduction="sum")
        return loss / torch.clamp(loss_mask.sum(), min=1.0)
    if args.snr_gamma is None:
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    snr = compute_snr(noise_scheduler, timesteps)
    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
    if noise_scheduler.config.prediction_type == "epsilon":
        mse_loss_weights = mse_loss_weights / snr
    elif noise_scheduler.config.prediction_type == "v_prediction":
        mse_loss_weights = mse_loss_weights / (snr + 1)
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    return loss.mean()


def _flux_training_step(
    batch,
    args,
    vae,
    transformer,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    weight_dtype,
    accelerator,
):
    prompts = list(batch["prompts"])
    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
    colormap_values = batch["colormap_values"].to(dtype=weight_dtype)
    control_values = batch["control_values"].to(dtype=weight_dtype)
    bsz = pixel_values.shape[0]
    device = accelerator.device

    with torch.no_grad():
        image_latents = encode_vae_latents(vae, pixel_values, sample_mode="sample")
        colormap_latents = encode_vae_latents(vae, colormap_values, sample_mode="sample")
        control_latents = encode_vae_latents(vae, control_values, sample_mode="argmax")
    image_latents = image_latents.to(dtype=weight_dtype)
    colormap_latents = colormap_latents.to(dtype=weight_dtype)
    control_latents = control_latents.to(dtype=weight_dtype)

    if args.conditioning_dropout_prob is not None and args.conditioning_dropout_prob > 0:
        drop = torch.rand(bsz, device=device) < args.conditioning_dropout_prob
        prompts = ["" if flag else prompt for flag, prompt in zip(drop.tolist(), prompts)]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = _encode_flux_batch(
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            prompts,
            device,
            args.max_sequence_length,
        )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
    text_ids = text_ids.to(dtype=weight_dtype)

    noise_image = torch.randn_like(image_latents)
    noise_colormap = torch.randn_like(colormap_latents)
    timesteps = _sample_flow_timesteps(noise_scheduler, bsz, image_latents.device, args)
    sigmas = _dit_sigmas(noise_scheduler, timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
    noisy_image = (1.0 - sigmas) * image_latents + sigmas * noise_image
    noisy_colormap = (1.0 - sigmas) * colormap_latents + sigmas * noise_colormap

    latent_h, latent_w = image_latents.shape[2], image_latents.shape[3]
    image_tokens = pack_latents(noisy_image)
    colormap_tokens = pack_latents(noisy_colormap)
    control_tokens = pack_latents(control_latents)
    mask_tokens = pack_fill_mask(
        batch["coarse_mask"].to(device=device, dtype=weight_dtype),
        height=latent_h,
        width=latent_w,
    )
    hidden_states = concat_fill_channels(image_tokens, colormap_tokens, control_tokens, mask_tokens)
    img_ids = prepare_latent_image_ids(latent_h, latent_w, image_tokens.device, image_tokens.dtype)

    if getattr(unwrap_model(accelerator, transformer).config, "guidance_embeds", False):
        guidance = torch.full((bsz,), args.train_guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    model_pred = transformer(
        hidden_states=hidden_states,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]
    pred_image_tokens, pred_colormap_tokens = split_packed_pred(model_pred)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    pred_image = unpack_latents(pred_image_tokens, args.resolution, args.resolution, vae_scale_factor)
    pred_colormap = unpack_latents(pred_colormap_tokens, args.resolution, args.resolution, vae_scale_factor)
    model_pred = torch.cat([pred_image, pred_colormap], dim=1)
    target = torch.cat([noise_image - image_latents, noise_colormap - colormap_latents], dim=1)

    weighting = _flow_loss_weighting(sigmas, args)
    if args.loss_on_mask_only:
        loss_mask = F.interpolate(
            batch["coarse_mask"].to(device=device, dtype=model_pred.dtype),
            size=model_pred.shape[-2:],
            mode="nearest",
        ).expand_as(model_pred)
        loss = F.mse_loss(model_pred.float() * loss_mask, target.float() * loss_mask, reduction="sum")
        return loss / torch.clamp(loss_mask.sum(), min=1.0)

    loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
    return loss.mean()


def _encode_unigs_visuals(vae, batch, weight_dtype, device):
    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
    colormap_values = batch["colormap_values"].to(dtype=weight_dtype)
    control_values = batch["control_values"].to(dtype=weight_dtype)
    with torch.no_grad():
        image_latents = encode_vae_latents(vae, pixel_values, sample_mode="sample")
        colormap_latents = encode_vae_latents(vae, colormap_values, sample_mode="sample")
        control_latents = encode_vae_latents(vae, control_values, sample_mode="argmax")
    return (
        image_latents.to(dtype=weight_dtype),
        colormap_latents.to(dtype=weight_dtype),
        control_latents.to(dtype=weight_dtype),
    )


def _drop_prompts(prompts, bsz, device, prob):
    if prob is None or prob <= 0:
        return prompts
    drop = torch.rand(bsz, device=device) < prob
    return ["" if flag else prompt for flag, prompt in zip(drop.tolist(), prompts)]


def _masked_or_weighted_loss(model_pred, target, batch, args, device, weighting=None):
    if args.loss_on_mask_only:
        loss_mask = F.interpolate(
            batch["coarse_mask"].to(device=device, dtype=model_pred.dtype),
            size=model_pred.shape[-2:],
            mode="nearest",
        ).expand_as(model_pred)
        loss = F.mse_loss(model_pred.float() * loss_mask, target.float() * loss_mask, reduction="sum")
        return loss / torch.clamp(loss_mask.sum(), min=1.0)
    if weighting is not None:
        loss = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        return loss.mean()
    if args.snr_gamma is None:
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def _sd3_training_step(
    batch,
    args,
    vae,
    transformer,
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    noise_scheduler,
    weight_dtype,
    accelerator,
):
    device = accelerator.device
    prompts = _drop_prompts(list(batch["prompts"]), batch["pixel_values"].shape[0], device, args.conditioning_dropout_prob)
    image_latents, colormap_latents, control_latents = _encode_unigs_visuals(vae, batch, weight_dtype, device)
    bsz = image_latents.shape[0]
    with torch.no_grad():
        prompt_embeds, pooled = encode_sd3_prompt(
            text_encoder,
            text_encoder_2,
            text_encoder_3,
            tokenizer,
            tokenizer_2,
            tokenizer_3,
            prompts,
            device,
            max_sequence_length=args.max_sequence_length,
            joint_attention_dim=int(getattr(unwrap_model(accelerator, transformer).config, "joint_attention_dim", 4096)),
        )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    pooled = pooled.to(dtype=weight_dtype)

    noise_image = torch.randn_like(image_latents)
    noise_colormap = torch.randn_like(colormap_latents)
    timesteps = _sample_flow_timesteps(noise_scheduler, bsz, image_latents.device, args)
    sigmas = _dit_sigmas(noise_scheduler, timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
    noisy_image = (1.0 - sigmas) * image_latents + sigmas * noise_image
    noisy_colormap = (1.0 - sigmas) * colormap_latents + sigmas * noise_colormap
    mask_latents = resize_mask_to_latents(
        batch["coarse_mask"].to(device=device, dtype=weight_dtype),
        latent_height=image_latents.shape[2],
        latent_width=image_latents.shape[3],
        latent_channels=image_latents.shape[1],
    )
    model_pred = forward_sd3_token_concat(
        transformer,
        noisy_image,
        noisy_colormap,
        control_latents,
        mask_latents,
        timesteps,
        prompt_embeds,
        pooled,
    )[0]
    target = torch.cat([noise_image - image_latents, noise_colormap - colormap_latents], dim=1)
    return _masked_or_weighted_loss(model_pred, target, batch, args, device, weighting=_flow_loss_weighting(sigmas, args))


def _zimage_training_step(
    batch,
    args,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    noise_scheduler,
    weight_dtype,
    accelerator,
):
    device = accelerator.device
    prompts = _drop_prompts(list(batch["prompts"]), batch["pixel_values"].shape[0], device, args.conditioning_dropout_prob)
    image_latents, colormap_latents, control_latents = _encode_unigs_visuals(vae, batch, weight_dtype, device)
    bsz = image_latents.shape[0]
    with torch.no_grad():
        prompt_embeds = encode_zimage_prompt(
            text_encoder, tokenizer, prompts, device, max_sequence_length=args.max_sequence_length
        )
    noise_image = torch.randn_like(image_latents)
    noise_colormap = torch.randn_like(colormap_latents)
    timesteps = _sample_flow_timesteps(noise_scheduler, bsz, image_latents.device, args)
    sigmas = _dit_sigmas(noise_scheduler, timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
    noisy_image = (1.0 - sigmas) * image_latents + sigmas * noise_image
    noisy_colormap = (1.0 - sigmas) * colormap_latents + sigmas * noise_colormap
    mask_latents = resize_mask_to_latents(
        batch["coarse_mask"].to(device=device, dtype=weight_dtype),
        latent_height=image_latents.shape[2],
        latent_width=image_latents.shape[3],
        latent_channels=image_latents.shape[1],
    )
    timestep = (1000 - timesteps) / 1000
    model_pred = forward_zimage_omni(
        transformer,
        noisy_image,
        noisy_colormap,
        control_latents,
        mask_latents,
        timestep,
        prompt_embeds,
        negate=True,
    )[0]
    target = torch.cat([noise_image - image_latents, noise_colormap - colormap_latents], dim=1)
    return _masked_or_weighted_loss(model_pred, target, batch, args, device, weighting=_flow_loss_weighting(sigmas, args))


def _pixart_training_step(
    batch,
    args,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    noise_scheduler,
    weight_dtype,
    accelerator,
):
    device = accelerator.device
    prompts = _drop_prompts(list(batch["prompts"]), batch["pixel_values"].shape[0], device, args.conditioning_dropout_prob)
    image_latents, colormap_latents, control_latents = _encode_unigs_visuals(vae, batch, weight_dtype, device)
    bsz = image_latents.shape[0]
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = encode_pixart_prompt(
            text_encoder, tokenizer, prompts, device, max_sequence_length=args.max_sequence_length
        )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    noise = torch.randn_like(torch.cat([image_latents, colormap_latents], dim=1))
    if args.noise_offset:
        noise = noise + args.noise_offset * torch.randn(
            (noise.shape[0], noise.shape[1], 1, 1), device=noise.device
        )
    latents = torch.cat([image_latents, colormap_latents], dim=1)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    noisy_image, noisy_colormap = noisy_latents.chunk(2, dim=1)
    mask_latents = resize_mask_to_latents(
        batch["coarse_mask"].to(device=device, dtype=weight_dtype),
        latent_height=image_latents.shape[2],
        latent_width=image_latents.shape[3],
        latent_channels=image_latents.shape[1],
    )
    added = pixart_added_cond_kwargs(
        unwrap_model(accelerator, transformer),
        batch_size=bsz,
        height=args.resolution,
        width=args.resolution,
        dtype=weight_dtype,
        device=device,
    )
    model_pred = forward_pixart_token_concat(
        transformer,
        noisy_image,
        noisy_colormap,
        control_latents,
        mask_latents,
        timesteps,
        prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        added_cond_kwargs=added,
    )[0]
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    if args.snr_gamma is not None:
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        if args.loss_on_mask_only:
            loss_mask = F.interpolate(
                batch["coarse_mask"].to(device=device, dtype=model_pred.dtype),
                size=model_pred.shape[-2:],
                mode="nearest",
            ).expand_as(model_pred)
            loss = F.mse_loss(model_pred.float() * loss_mask, target.float() * loss_mask, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            return loss.mean()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        return loss.mean()
    return _masked_or_weighted_loss(model_pred, target, batch, args, device)


def _dit_training_step(
    batch,
    args,
    vae,
    transformer,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    weight_dtype,
    accelerator,
    text_encoder_3=None,
    tokenizer_3=None,
):
    family = args.dit_family
    if family == "sd3":
        return _sd3_training_step(
            batch,
            args,
            vae,
            transformer,
            text_encoder,
            text_encoder_2,
            text_encoder_3,
            tokenizer,
            tokenizer_2,
            tokenizer_3,
            noise_scheduler,
            weight_dtype,
            accelerator,
        )
    if family == "z_image":
        return _zimage_training_step(
            batch, args, vae, transformer, text_encoder, tokenizer, noise_scheduler, weight_dtype, accelerator
        )
    if family == "pixart":
        return _pixart_training_step(
            batch, args, vae, transformer, text_encoder, tokenizer, noise_scheduler, weight_dtype, accelerator
        )
    return _flux_training_step(
        batch,
        args,
        vae,
        transformer,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        noise_scheduler,
        weight_dtype,
        accelerator,
    )


def _dit_transformer_class(family: str):
    if family == "sd3":
        from unigs.pipeline_unigs_sd3 import _import_sd3

        return _import_sd3()[0]
    if family == "z_image":
        from unigs.pipeline_unigs_zimage import _import_zimage

        return _import_zimage()[0]
    if family == "pixart":
        from unigs.pipeline_unigs_pixart import _import_pixart

        return _import_pixart()[0]
    return _import_flux()[0]


def _build_validation_pipeline(
    args,
    accelerator,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    unet,
    transformer,
    noise_scheduler,
    final: bool = False,
    text_encoder_3=None,
    tokenizer_3=None,
):
    if args.dit:
        family = args.dit_family
        if family == "flux":
            from unigs.pipeline_unigs_flux import UniGSFluxPipeline

            _, FlowMatchEulerDiscreteScheduler, _, _ = _import_flux()
            scheduler = (
                FlowMatchEulerDiscreteScheduler.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="scheduler"
                )
                if final
                else FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config)
            )
            return UniGSFluxPipeline(
                vae=unwrap_model(accelerator, vae),
                text_encoder=unwrap_model(accelerator, text_encoder),
                tokenizer=tokenizer,
                text_encoder_2=unwrap_model(accelerator, text_encoder_2),
                tokenizer_2=tokenizer_2,
                transformer=unwrap_model(accelerator, transformer),
                scheduler=scheduler,
            )
        if dit_uses_flow_matching(family):
            from diffusers import FlowMatchEulerDiscreteScheduler

            scheduler_cls = FlowMatchEulerDiscreteScheduler
        else:
            from diffusers import DPMSolverMultistepScheduler

            scheduler_cls = DPMSolverMultistepScheduler
        scheduler = (
            scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            if final
            else scheduler_cls.from_config(noise_scheduler.config)
        )
        vae_u = unwrap_model(accelerator, vae)
        transformer_u = unwrap_model(accelerator, transformer)
        text_u = unwrap_model(accelerator, text_encoder)
        if family == "sd3":
            from unigs.pipeline_unigs_sd3 import UniGSSD3Pipeline

            return UniGSSD3Pipeline(
                vae=vae_u,
                transformer=transformer_u,
                tokenizer=tokenizer,
                text_encoder=text_u,
                tokenizer_2=tokenizer_2,
                text_encoder_2=None if text_encoder_2 is None else unwrap_model(accelerator, text_encoder_2),
                tokenizer_3=tokenizer_3,
                text_encoder_3=None if text_encoder_3 is None else unwrap_model(accelerator, text_encoder_3),
                scheduler=scheduler,
            )
        if family == "z_image":
            from unigs.pipeline_unigs_zimage import UniGSZImagePipeline

            return UniGSZImagePipeline(
                vae=vae_u,
                transformer=transformer_u,
                tokenizer=tokenizer,
                text_encoder=text_u,
                scheduler=scheduler,
            )
        from unigs.pipeline_unigs_pixart import UniGSPixArtPipeline

        return UniGSPixArtPipeline(
            vae=vae_u,
            transformer=transformer_u,
            tokenizer=tokenizer,
            text_encoder=text_u,
            scheduler=scheduler,
        )
    scheduler = (
        DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        if final
        else DDIMScheduler.from_config(noise_scheduler.config)
    )
    return UniGSPipeline(
        vae=unwrap_model(accelerator, vae),
        text_encoder=unwrap_model(accelerator, text_encoder),
        tokenizer=tokenizer,
        unet=unwrap_model(accelerator, unet),
        scheduler=scheduler,
    )


def log_validation(pipeline, args, accelerator, weight_dtype, step):
    logger.info("Running validation...")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    image = Image.open(args.validation_image).convert("RGB") if args.validation_image else None
    mask = Image.open(args.validation_mask).convert("L") if args.validation_mask else None
    colormap = Image.open(args.validation_colormap).convert("RGB") if args.validation_colormap else None
    prompt = args.validation_prompt or TASK_PROMPT_TEMPLATES[args.validation_task].format(entities="object")

    autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(accelerator.device.type)
    images, colormaps = [], []
    with autocast_ctx:
        for _ in range(args.num_validation_images):
            output = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                colormap=colormap,
                task=args.validation_task,
                num_inference_steps=30,
                generator=generator,
                decode_masks=False,
            )
            images.extend(output.images)
            colormaps.extend(output.colormaps)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images + colormaps])
            tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [wandb.Image(img, caption=prompt) for img in images]
                    + [wandb.Image(img, caption=f"{prompt} [colormap]") for img in colormaps]
                }
            )
    return images, colormaps


def main():
    args = parse_args()
    args.pretrained_model_name_or_path = resolve_inpainting_checkpoint(
        backbone=args.backbone,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )
    args.dit_family = resolve_dit_family(args.backbone) or resolve_dit_family(args.pretrained_model_name_or_path)
    args.dit = args.dit_family is not None or is_dit_checkpoint(args.backbone) or is_dit_checkpoint(
        args.pretrained_model_name_or_path
    )
    if args.dit and args.dit_family is None:
        args.dit_family = "flux"
    if args.dit and args.max_sequence_length == 512:
        args.max_sequence_length = default_dit_max_sequence_length(args.dit_family)
    if args.dit and args.dit_family in {"flux", "z_image"} and args.resolution % 16 != 0:
        raise ValueError(
            f"{args.dit_family} packing requires --resolution divisible by 16 (VAE 8× and 2×2 pack), "
            f"got {args.resolution}."
        )
    if args.dit and args.resolution % 8 != 0:
        raise ValueError(f"DiT UniGS requires --resolution divisible by 8, got {args.resolution}.")
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both `--report_to=wandb` and `--hub_token`. Use `hf auth login` instead.")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
        else:
            repo_id = None
    else:
        repo_id = None

    tokenizer_3 = None
    text_encoder_3 = None
    if args.dit:
        from unigs.dit import load_dit_transformer

        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        if args.dit_family == "flux":
            FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, T5EncoderModel, T5TokenizerFast = _import_flux()
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler"
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
            )
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                revision=args.revision,
                variant=args.variant,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer", revision=args.non_ema_revision
            )
            transformer = adapt_unigs_transformer(transformer, family="flux")
        elif args.dit_family == "sd3":
            from transformers import CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast

            from unigs.pipeline_unigs_sd3 import _import_sd3

            _, FlowMatchEulerDiscreteScheduler, CLIPTextModelWithProjection, T5EncoderModel = _import_sd3()
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler"
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
            )
            try:
                tokenizer_3 = T5TokenizerFast.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision
                )
            except Exception:
                tokenizer_3 = None
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                revision=args.revision,
                variant=args.variant,
            )
            try:
                text_encoder_3 = T5EncoderModel.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder_3",
                    revision=args.revision,
                    variant=args.variant,
                )
            except Exception:
                text_encoder_3 = None
                tokenizer_3 = None
            transformer = load_dit_transformer(
                "sd3", args.pretrained_model_name_or_path, revision=args.non_ema_revision, variant=args.variant
            )
        elif args.dit_family == "z_image":
            from transformers import AutoModel, AutoTokenizer

            from unigs.pipeline_unigs_zimage import _import_zimage

            _, FlowMatchEulerDiscreteScheduler = _import_zimage()
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
            )
            tokenizer_2 = None
            try:
                from transformers import Qwen2Model

                text_encoder = Qwen2Model.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=args.revision,
                    variant=args.variant,
                )
            except Exception:
                text_encoder = AutoModel.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=args.revision,
                    variant=args.variant,
                )
            text_encoder_2 = None
            transformer = load_dit_transformer(
                "z_image", args.pretrained_model_name_or_path, revision=args.non_ema_revision, variant=args.variant
            )
        elif args.dit_family == "pixart":
            from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

            from unigs.pipeline_unigs_pixart import _import_pixart

            try:
                noise_scheduler = DDPMScheduler.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="scheduler"
                )
            except Exception:
                noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
            try:
                tokenizer = T5Tokenizer.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
                )
            except Exception:
                tokenizer = T5TokenizerFast.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
                )
            tokenizer_2 = None
            text_encoder = T5EncoderModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            )
            text_encoder_2 = None
            transformer = load_dit_transformer(
                "pixart", args.pretrained_model_name_or_path, revision=args.non_ema_revision, variant=args.variant
            )
        else:
            raise ValueError(f"Unsupported DiT family '{args.dit_family}'.")
        if args.lora_rank:
            transformer = _apply_dit_lora(transformer, args.lora_rank, args.lora_alpha)
        unet = None
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        if text_encoder_2 is not None:
            text_encoder_2.requires_grad_(False)
        if text_encoder_3 is not None:
            text_encoder_3.requires_grad_(False)
        transformer.train()
    else:
        tokenizer_2 = None
        text_encoder_2 = None
        transformer = None
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )
        if int(unet.config.in_channels) not in (INPAINTING_UNET_IN_CHANNELS, 13):
            raise ValueError(
                f"Expected an SD inpainting UNet ({INPAINTING_UNET_IN_CHANNELS} input channels), "
                f"got in_channels={unet.config.in_channels}. "
                f"Use --backbone sd15|sd21 or an inpainting Hub id such as {INPAINTING_BACKBONES['sd15']}."
            )
        unet = adapt_unigs_unet(unet)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()

    denoise_model = transformer if args.dit else unet

    if args.use_ema:
        if args.dit:
            raise ValueError("--use_ema is not supported for DiT backbones.")
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if args.dit:
            logger.warning("xformers memory-efficient attention is ignored for DiT backbones (uses SDPA).")
        elif is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it with `pip install xformers`.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        save_subfolder = "transformer" if args.dit else "unet"

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, save_subfolder))
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            if args.dit:
                loader = _dit_transformer_class(args.dit_family)
            else:
                loader = UNet2DConditionModel
            for _ in range(len(models)):
                model = models.pop()
                load_model = loader.from_pretrained(input_dir, subfolder=save_subfolder)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        denoise_model.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`.") from err
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        _trainable_parameters(denoise_model),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.coco_annotation_file is not None:
        logger.info("Loading COCO annotations from %s", args.coco_annotation_file)
        records = load_coco_records(args.coco_image_dir, args.coco_annotation_file)
    else:
        logger.info("Loading dataset %s from the Hub", args.dataset_name)
        hf_dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        split = hf_dataset["train"] if "train" in hf_dataset else hf_dataset[list(hf_dataset.keys())[0]]
        records = records_from_hf_dataset(
            split,
            image_column=args.image_column,
            mask_column=args.mask_column,
            label_column=args.label_column,
            objects_column=args.objects_column,
            caption_column=args.caption_column,
        )

    if args.max_train_samples is not None:
        records = records[: args.max_train_samples]
    if not records:
        raise ValueError("No training records were produced. Check the dataset columns / COCO paths.")

    train_dataset = UniGSInstanceDataset(
        records,
        tokenizer=None if args.dit else tokenizer,
        resolution=args.resolution,
        task=args.task,
        max_entities=args.max_entities,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        referring_neg_prob=args.referring_neg_prob,
        arbitrary_mask_prob=args.arbitrary_mask_prob,
        grid_size=args.grid_size,
        caption_column=args.caption_column,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    denoise_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        denoise_model, optimizer, train_dataloader, lr_scheduler
    )
    if args.dit:
        transformer = denoise_model
    else:
        unet = denoise_model
    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.dit:
        if text_encoder_2 is not None:
            text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_3 is not None:
            text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running UniGS training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Num epochs = %s", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %s", args.train_batch_size)
    logger.info("  Total train batch size = %s", total_batch_size)
    logger.info("  Gradient accumulation steps = %s", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %s", args.max_train_steps)
    logger.info("  Backbone = %s", args.pretrained_model_name_or_path)
    logger.info(
        "  Architecture = %s",
        (
            f"dit-{args.dit_family} (Fill-style channel concat)"
            if args.dit_family == "flux"
            else f"dit-{args.dit_family} (context-token concat)"
            if args.dit
            else "unet-channel-concat"
        ),
    )
    logger.info("  Task = %s", args.task)

    noise_scheduler_copy = copy.deepcopy(noise_scheduler) if args.dit else noise_scheduler

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            logger.info("Checkpoint '%s' does not exist. Starting a new run.", args.resume_from_checkpoint)
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        denoise_model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(denoise_model):
                if args.dit:
                    loss = _dit_training_step(
                        batch=batch,
                        args=args,
                        vae=vae,
                        transformer=transformer,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        noise_scheduler=noise_scheduler_copy,
                        weight_dtype=weight_dtype,
                        accelerator=accelerator,
                        text_encoder_3=text_encoder_3,
                        tokenizer_3=tokenizer_3,
                    )
                else:
                    loss = _unet_training_step(
                        batch=batch,
                        args=args,
                        vae=vae,
                        unet=unet,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                    )

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(denoise_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[0:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info("Saved state to %s", save_path)

                if (
                    args.validation_prompt is not None
                    and accelerator.is_main_process
                    and global_step % args.validation_steps == 0
                ):
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    pipeline = _build_validation_pipeline(
                        args=args,
                        accelerator=accelerator,
                        vae=vae,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        unet=unet,
                        transformer=transformer,
                        noise_scheduler=noise_scheduler,
                        text_encoder_3=text_encoder_3,
                        tokenizer_3=tokenizer_3,
                    )
                    log_validation(pipeline, args, accelerator, weight_dtype, global_step)
                    if args.use_ema:
                        ema_unet.restore(unet.parameters())
                    del pipeline
                    torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        denoise_model = unwrap_model(accelerator, denoise_model)
        if args.use_ema:
            ema_unet.copy_to(denoise_model.parameters())
        if args.dit and args.lora_rank:
            if hasattr(denoise_model, "fuse_lora"):
                denoise_model.fuse_lora(lora_scale=1.0)
            if hasattr(denoise_model, "unload_lora"):
                denoise_model.unload_lora()
        pipeline = _build_validation_pipeline(
            args=args,
            accelerator=accelerator,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=None if args.dit else denoise_model,
            transformer=denoise_model if args.dit else None,
            noise_scheduler=noise_scheduler,
            final=True,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of UniGS training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
