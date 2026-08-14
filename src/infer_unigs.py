#!/usr/bin/env python
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

"""Run UniGS inference with a trained (or freshly adapted) pipeline.

Examples:

    python src/infer_unigs.py --task inpainting --prompt "dog" \\
        --image input.png --mask mask.png --output-dir out

    python src/infer_unigs.py --pipeline sd21 --task entity --image scene.png

    python src/infer_unigs.py --pipeline flux --task inpainting --prompt "dog" \\
        --image input.png --mask mask.png --output-dir out --dtype bf16

    python src/infer_unigs.py --pipeline sd3 --task inpainting --prompt "dog" \\
        --image input.png --mask mask.png --output-dir out --dtype bf16

    python src/infer_unigs.py --pipeline z_image --task entity --image scene.png --dtype bf16

    python src/infer_unigs.py --pipeline pixart --task synthesis --colormap layout.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unigs.backbones import (
    BACKBONES,
    default_dit_guidance,
    detect_dit_family_from_path,
    resolve_backbone,
    resolve_dit_family,
)
from unigs.colormap import ProgressiveDichotomyModule
from unigs.pipelines import load_unigs_pipeline
from unigs.prompts import TASK_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="UniGS inference.")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="sd15",
        choices=list(BACKBONES),
        help=(
            "Which UniGS class to `from_pretrained` when bootstrapping a Hub checkpoint: "
            "sd15/sd21 → UniGSPipeline, flux → UniGSFluxPipeline, sd3 → UniGSSD3Pipeline, "
            "z_image → UniGSZImagePipeline, pixart → UniGSPixArtPipeline. "
            "A trained UniGS save dir is selected from `model_index.json` `_class_name`."
        ),
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Hub id, local Diffusers dir, or trained UniGS `save_pretrained` directory.",
    )
    parser.add_argument("--task", type=str, default="inpainting", choices=list(TASK_NAMES))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--colormap", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="unigs-output")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG / DiT guidance. Defaults: 7.5 UNet, 30 FLUX, 4.5 SD3/PixArt, 0 Z-Image.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pdm-delta", type=float, default=10.0)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_backbone(
        backbone=args.pipeline,
        pretrained_model_name_or_path=args.pretrained_model,
    )
    family = resolve_dit_family(args.pipeline) or detect_dit_family_from_path(checkpoint)
    if args.guidance_scale is None:
        args.guidance_scale = default_dit_guidance(family) if family else 7.5

    pipeline = load_unigs_pipeline(checkpoint, family=family, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    pipeline.pdm = ProgressiveDichotomyModule(delta=args.pdm_delta, include_background=args.task == "entity")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    image = Image.open(args.image).convert("RGB") if args.image else None
    mask = Image.open(args.mask).convert("L") if args.mask else None
    colormap = Image.open(args.colormap).convert("RGB") if args.colormap else None
    prompt = args.prompt or {"entity": "panoptic: all entities."}.get(args.task, "object")

    output = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        colormap=colormap,
        task=args.task,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        decode_masks=True,
        pdm_delta=args.pdm_delta,
    )

    for i, (img, cmap) in enumerate(zip(output.images, output.colormaps)):
        img.save(os.path.join(args.output_dir, f"image_{i:02d}.png"))
        cmap.save(os.path.join(args.output_dir, f"colormap_{i:02d}.png"))
        if output.masks is not None:
            for j, entity in enumerate(output.masks[i]):
                Image.fromarray(entity * 255).save(os.path.join(args.output_dir, f"mask_{i:02d}_{j:02d}.png"))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
