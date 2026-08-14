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

    python src/infer_unigs.py --task entity --image scene.png --output-dir out

    python src/infer_unigs.py --pretrained-model path/to/trained-unigs \\
        --task synthesis --prompt "cat, sofa and lamp" --colormap layout.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unigs import UniGSPipeline, adapt_unigs_unet
from unigs.colormap import ProgressiveDichotomyModule
from unigs.prompts import TASK_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="UniGS inference.")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Trained UniGS dir, or an SD 1.5 / 2.1 checkpoint to adapt on the fly.",
    )
    parser.add_argument("--task", type=str, default="inpainting", choices=list(TASK_NAMES))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--colormap", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="unigs-output")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pdm-delta", type=float, default=10.0)
    parser.add_argument("--from-stable-diffusion", action="store_true", help="Expand SD conv_in/conv_out to 13/8.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.from_stable_diffusion:
        pipeline = UniGSPipeline.from_stable_diffusion(args.pretrained_model, torch_dtype=dtype)
    else:
        pipeline = UniGSPipeline.from_pretrained(args.pretrained_model, torch_dtype=dtype)
        if getattr(pipeline.unet.config, "in_channels", None) != 13:
            pipeline.unet = adapt_unigs_unet(pipeline.unet)
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
