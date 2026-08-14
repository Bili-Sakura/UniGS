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

"""Fine-tune Stable Diffusion inpainting backbones as UniGS with 🤗 Accelerate.

Example:
    accelerate launch src/train_unigs.py \\
        --backbone=sd15 \\
        --coco_image_dir=/data/coco/train2017 \\
        --coco_annotation_file=/data/coco/annotations/instances_train2017.json \\
        --output_dir=unigs-sd15 \\
        --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=4 \\
        --learning_rate=5e-5 --max_train_steps=30000 --checkpointing_steps=5000 \\
        --mixed_precision=fp16 --task=joint
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

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

from unigs.backbones import INPAINTING_BACKBONES, INPAINTING_UNET_IN_CHANNELS, resolve_inpainting_checkpoint
from unigs.dataset import UniGSInstanceDataset, collate_fn, load_coco_records, records_from_hf_dataset
from unigs.pipeline_unigs import UniGSPipeline
from unigs.prompts import TASK_NAMES, TASK_PROMPT_TEMPLATES
from unigs.unet import adapt_unigs_unet


if is_wandb_available():
    import wandb

try:
    check_min_version("0.27.0")
except Exception:
    pass

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UniGS on an SD inpainting backbone.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="sd15",
        choices=list(INPAINTING_BACKBONES),
        help="Inpainting checkpoint shorthand when --pretrained_model_name_or_path is omitted.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Hub id or local path to an SD *inpainting* checkpoint "
            f"({INPAINTING_BACKBONES['sd15']} or {INPAINTING_BACKBONES['sd21']}). "
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

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it with `pip install xformers`.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
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
        unet.parameters(),
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
        tokenizer=tokenizer,
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

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

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
    logger.info("  Task = %s", args.task)

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
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
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
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
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
                    loss = loss / torch.clamp(loss_mask.sum(), min=1.0)
                elif args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
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
                    pipeline = UniGSPipeline(
                        vae=unwrap_model(accelerator, vae),
                        text_encoder=unwrap_model(accelerator, text_encoder),
                        tokenizer=tokenizer,
                        unet=unwrap_model(accelerator, unet),
                        scheduler=DDIMScheduler.from_config(noise_scheduler.config),
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
        unet = unwrap_model(accelerator, unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        pipeline = UniGSPipeline(
            vae=unwrap_model(accelerator, vae),
            text_encoder=unwrap_model(accelerator, text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
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
