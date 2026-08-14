# UniGS

Unofficial implementation of [UniGS: Unified Representation for Image Generation and Segmentation](https://arxiv.org/abs/2312.01985) on **Stable Diffusion inpainting** backbones (SD 1.5 / 2.1), written in native 🤗 Diffusers style so it can be dropped into `examples/research_projects/unigs` (training) and `examples/community` (inference) with almost no changes.

UniGS treats entity-level masks as an RGB **colormap** that lives in the same VAE latent space as images. A dual-output UNet denoises both jointly inside an inpainting protocol, which is enough to cover four tasks from one representation:

| Task | Coarse mask | Control latent | Prompt template |
| --- | --- | --- | --- |
| Multi-class multi-region inpainting | `Ω(M)` | VAE of the masked image | `inpainting: generate dog.` |
| Image synthesis | all ones | VAE of the colormap | `synthesis: generate dog, ground, and sky.` |
| Referring segmentation | `Ω(M)` | VAE of the full image | `referring: find dog.` |
| Entity segmentation | all ones | VAE of the full image | `panoptic: all entities.` |

The UNet starts from the standard SD **inpainting** `UNet2DConditionModel` (9 input channels) and is expanded to UniGS channels only:

* **13 in:** `concat(z_t^image, z_t^colormap, coarse_mask, z_control)` (Eq. 8)
* **8 out:** `concat(eps_image, eps_colormap)`

Newly added `conv_in` / `conv_out` weights are zero-initialized. Existing inpainting channels (noisy image, mask, control latent) are copied from the pretrained UNet; colormap channels are zero-initialized — matching the paper.

Base text-to-image checkpoints (`stable-diffusion-v1-5`, `stable-diffusion-2-1`, etc.) are **not** supported.

## Layout

This tree is intentionally close to a Diffusers research example:

```
src/
  train_unigs.py              # Accelerate trainer (like examples/instruct_pix2pix)
  infer_unigs.py              # CLI around the pipeline
  requirements.txt
  unigs/
    pipeline_unigs.py         # DiffusionPipeline (community-pipeline style)
    backbones.py              # sd15 / sd21 inpainting Hub ids
    unet.py                   # 9 → 13-in, 4 → 8-out adapter
    colormap.py               # location-aware palette Ψ + progressive dichotomy Φ
    coarse_mask.py            # Ω, Bezier / extended-bbox coarse masks
    prompts.py                # Table 2 templates
    dataset.py                # COCO / 🤗 Datasets → UniGS batches
```

## Installation

```bash
pip install -r src/requirements.txt
```

Supported inpainting backbones (use `--backbone` or pass the Hub id explicitly):

| Shorthand | Checkpoint |
| --- | --- |
| `sd15` (paper default) | `stable-diffusion-v1-5/stable-diffusion-inpainting` |
| `sd21` | `stabilityai/stable-diffusion-2-inpainting` |

## Training

The trainer freezes the VAE and CLIP text encoder and fine-tunes only the UNet, following the usual Diffusers Accelerate script.

COCO instances (paper setting: sample up to 4 entities, 512², 48 epochs):

```bash
accelerate launch src/train_unigs.py \
  --backbone=sd15 \
  --coco_image_dir=/data/coco/train2017 \
  --coco_annotation_file=/data/coco/annotations/instances_train2017.json \
  --output_dir=unigs-sd15 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --max_train_steps=30000 \
  --checkpointing_steps=5000 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --random_flip \
  --task=joint \
  --conditioning_dropout_prob=0.1
```

SD 2.1 inpainting:

```bash
accelerate launch src/train_unigs.py \
  --backbone=sd21 \
  --dataset_name=<your/dataset> \
  --image_column=image \
  --mask_column=masks \
  --label_column=labels \
  --task=referring \
  --output_dir=unigs-sd21-referring
```

`--task` can be `inpainting`, `synthesis`, `referring`, `entity`, or `joint` (sample ratios 0.3 / 0.3 / 0.2 / 0.2 from the supplementary). Referring training randomly replaces category names with negatives (`--referring_neg_prob`) so the text prompt has to match the coarse-mask region.

The run writes a full `UniGSPipeline` via `save_pretrained`, so the UNet config records `in_channels=13` and `out_channels=8`.

## Inference

Load a trained directory as a Diffusers pipeline:

```python
import torch
from PIL import Image
from unigs import UniGSPipeline

pipe = UniGSPipeline.from_pretrained("unigs-sd15", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = Image.open("scene.png").convert("RGB")
mask = Image.open("hole.png").convert("L")  # white = fill

out = pipe.inpaint("dog", image, mask, num_inference_steps=50)
out.images[0].save("inpainted.png")
out.colormaps[0].save("colormap.png")
# out.masks is a list of binary entity maps from the progressive dichotomy module
```

Bootstrap directly from an inpainting checkpoint (channels expanded; weights are pretrained-inpainting + zero-init colormap branches — fine-tune before serious use):

```python
from unigs import UniGSPipeline

pipe = UniGSPipeline.from_inpainting(backbone="sd15", torch_dtype=torch.float16)
# or: UniGSPipeline.from_inpainting("stabilityai/stable-diffusion-2-inpainting")
```

The other Table-2 tasks:

```python
out = pipe.synthesize("cat, sofa and lamp", colormap=layout)
out = pipe.referring("dog", image=image, mask_image=region)
out = pipe.segment(image)  # entity / panoptic
```

CLI:

```bash
python src/infer_unigs.py \
  --backbone sd15 \
  --task inpainting \
  --prompt dog \
  --image scene.png \
  --mask hole.png \
  --output-dir out
```

## Method notes

**Location-aware palette (Ψ).** Each RGB channel uses `{0, 64, 128, 192, 255}` (124 colors after dropping black). The image is tiled into an `11 × 11` grid; an entity inherits the color of the cell that contains its center of mass. Collision fallback walks to the next unused cell.

**Progressive dichotomy (Φ).** Depth-first 2-means on concatenated RGB + CIE Lab features, no assumed cluster count. A region stops splitting when the mean L2 distance to its centroid is below `δ` (default `10`). Entity segmentation keeps every cluster, including near-black regions.

**Coarse mask (Ω).** With probability `--arbitrary_mask_prob` a quadratic-Bezier blob is drawn around the entity bbox (Paint-by-Example / supplementary Alg. 1); otherwise an extended rectangle is used. Synthesis and entity segmentation pass an all-ones mask.

**VAE.** Colormaps are encoded and decoded with the same `AutoencoderKL` as RGB images. SD inpainting uses an 8× VAE downscale (`vae_scale_factor`).

## Mapping onto Diffusers

To upstream this as an official example:

1. Move `src/unigs/pipeline_unigs.py` → `examples/community/pipeline_unigs.py` (inline the small helpers, or keep the package).
2. Move `src/train_unigs.py` + `src/unigs/` → `examples/research_projects/unigs/`.
3. Load with `DiffusionPipeline.from_pretrained(..., custom_pipeline="pipeline_unigs")` once the community file is in tree.

No custom CUDA ops, no extra segmentation losses — training is standard latent-diffusion MSE on the 8-channel noise.
