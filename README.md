# UniGS

Unofficial implementation of [UniGS: Unified Representation for Image Generation and Segmentation](https://arxiv.org/abs/2312.01985) on **Stable Diffusion inpainting** UNets (SD 1.5 / 2.1) and several **DiT** backbones, written in native 🤗 Diffusers style so it can be dropped into `examples/research_projects/unigs` (training) and `examples/community` (inference) with almost no changes.

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

**FLUX.1-Fill-dev (DiT).** Fill's native condition is channel-concat of packed noisy latents, masked-image latents, and mask (`in_channels=384`). UniGS instead follows FLUX.1 Kontext: every stream is a 64-dim packed token sequence, concatenated on the **sequence** axis, with RoPE ids `(t, h, w)`:

* `t=0` noisy image (denoised)
* `t=1` noisy colormap (denoised)
* `t=2` control latent (context only)
* `t=3` coarse mask (context only)

`x_embedder` is shrunk 384→64 by copying Fill's packed noisy-latent columns. Text is CLIP pooled + T5; training is flow matching.

**Other DiT backbones (text-to-image is OK).** There is no inpainting SD 3.5 / Z-Image / PixArt-α checkpoint in the UniGS protocol. Those models keep **native** `in_channels` and still condition by extra visual **context tokens**, never by stacking mask+control on the channel axis:

| Family | Default Hub id | How condition is injected | Objective |
| --- | --- | --- | --- |
| SD 3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` | patch-embed each stream at native HxW, add a zero-init stream embedding, concat sequences, unpatchify image+colormap | flow matching (CLIP-L + CLIP-G + T5) |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | native omni nested list: clean control + mask, noisy image+colormap stacked on height (omni unpatchify returns only the last stream) | flow matching (Qwen; output negated like `ZImagePipeline`) |
| PixArt-α | `PixArt-alpha/PixArt-XL-2-512-MS` | same pos-embed-then-sequence-concat as SD3; 1024-MS needs `resolution` / `aspect_ratio` micro-conditions | epsilon diffusion (T5) |

SD3 cannot spatially stack four 64×64 streams: `pos_embed_max_size=96`. Sequence concat after `PatchEmbed` avoids that limit.

Base text-to-image **UNets** (`stable-diffusion-v1-5`, `stable-diffusion-2-1`, etc.) are **not** supported. DiT text-to-image checkpoints listed above are.

## Layout

This tree is intentionally close to a Diffusers research example:

```
src/
  train_unigs.py              # Accelerate trainer (like examples/instruct_pix2pix)
  infer_unigs.py              # CLI around the pipeline
  requirements.txt
  unigs/
    pipeline_unigs.py         # DiffusionPipeline (community-pipeline style)
    pipeline_unigs_flux.py    # FLUX Fill DiT pipeline (context-token concat)
    pipeline_unigs_dit.py     # SD 3.5 / Z-Image / PixArt-α DiT pipeline
    backbones.py              # sd15 / sd21 / flux / sd3 / z_image / pixart Hub ids
    unet.py                   # 9 → 13-in, 4 → 8-out adapter
    transformer.py            # Fill 384-in → UniGS 64-in token-concat adapter
    dit.py                    # SD3 / PixArt sequence concat + Z-Image omni
    colormap.py               # location-aware palette Ψ + progressive dichotomy Φ
    coarse_mask.py            # Ω, Bezier / extended-bbox coarse masks
    prompts.py                # Table 2 templates
    dataset.py                # COCO / 🤗 Datasets → UniGS batches
tests/
  test_dit_tokens.py          # pack / RoPE ids / Fill→64-in / SD3 stream embed / Z-Image omni
```

## Installation

```bash
pip install -r src/requirements.txt
```

Supported backbones (use `--backbone` or pass the Hub id explicitly):

| Shorthand | Checkpoint | Conditioning |
| --- | --- | --- |
| `sd15` (paper default) | `stable-diffusion-v1-5/stable-diffusion-inpainting` | channel concat (13-in UNet) |
| `sd21` | `stabilityai/stable-diffusion-2-inpainting` | channel concat (13-in UNet) |
| `flux` / `flux_fill` | `black-forest-labs/FLUX.1-Fill-dev` | context-token concat (64-in packed DiT) |
| `sd3` / `sd3.5` / `sd35` | `stabilityai/stable-diffusion-3.5-medium` | context-token concat (native 16-ch MMDiT) |
| `z_image` / `z-image` | `Tongyi-MAI/Z-Image-Turbo` | omni context-token concat (native 16-ch) |
| `pixart` / `pixart_alpha` | `PixArt-alpha/PixArt-XL-2-512-MS` | context-token concat (native 4-ch DiT) |
| `pixart_1024` | `PixArt-alpha/PixArt-XL-2-1024-MS` | same, with resolution micro-conditions |

FLUX.1-Fill-dev and SD 3.5 Medium are gated — accept the license on the Hub and `hf auth login` before training or bootstrapping. Z-Image needs a Diffusers build that includes `ZImageTransformer2DModel` (recent release or install from source).

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

FLUX.1-Fill-dev (context-token concat; LoRA is recommended because the transformer is 12B):

```bash
accelerate launch src/train_unigs.py \
  --backbone=flux \
  --coco_image_dir=/data/coco/train2017 \
  --coco_annotation_file=/data/coco/annotations/instances_train2017.json \
  --output_dir=unigs-flux-fill \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lora_rank=16 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --task=joint \
  --conditioning_dropout_prob=0.1
```

Resolution must be divisible by 16 (8× VAE and 2×2 packing). The saved pipeline is a `UniGSFluxPipeline` with `transformer.in_channels=64`.

SD 3.5 Medium (context-token concat after patch embed; T5-XXL is large — LoRA recommended):

```bash
accelerate launch src/train_unigs.py \
  --backbone=sd3 \
  --coco_image_dir=/data/coco/train2017 \
  --coco_annotation_file=/data/coco/annotations/instances_train2017.json \
  --output_dir=unigs-sd35-medium \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lora_rank=16 \
  --max_train_steps=10000 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --task=joint
```

Z-Image Turbo (omni context tokens; distilled, default guidance 0):

```bash
accelerate launch src/train_unigs.py \
  --backbone=z_image \
  --coco_image_dir=/data/coco/train2017 \
  --coco_annotation_file=/data/coco/annotations/instances_train2017.json \
  --output_dir=unigs-z-image \
  --resolution=512 \
  --train_batch_size=1 \
  --lora_rank=16 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --task=joint
```

PixArt-α 512 (epsilon diffusion, T5 only):

```bash
accelerate launch src/train_unigs.py \
  --backbone=pixart \
  --coco_image_dir=/data/coco/train2017 \
  --coco_annotation_file=/data/coco/annotations/instances_train2017.json \
  --output_dir=unigs-pixart \
  --resolution=512 \
  --train_batch_size=2 \
  --lora_rank=16 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --task=joint
```

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
# DiT: UniGSPipeline.from_inpainting(backbone="flux", torch_dtype=torch.bfloat16)
#   dispatches to UniGSFluxPipeline (context-token concat).
# Also: backbone="sd3" | "z_image" | "pixart" → UniGSDiTPipeline.
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

FLUX Fill (guidance default 30; bf16 recommended):

```bash
python src/infer_unigs.py \
  --backbone flux \
  --task inpainting \
  --prompt dog \
  --image scene.png \
  --mask hole.png \
  --dtype bf16 \
  --output-dir out
```

SD 3.5 Medium (guidance default 4.5):

```bash
python src/infer_unigs.py \
  --backbone sd3 \
  --task inpainting \
  --prompt dog \
  --image scene.png \
  --mask hole.png \
  --dtype bf16 \
  --output-dir out
```

Z-Image (guidance default 0) and PixArt-α (guidance default 4.5) use the same CLI with `--backbone z_image` or `--backbone pixart`.

## Method notes

**Location-aware palette (Ψ).** Each RGB channel uses `{0, 64, 128, 192, 255}` (124 colors after dropping black). The image is tiled into an `11 × 11` grid; an entity inherits the color of the cell that contains its center of mass. Collision fallback walks to the next unused cell.

**Progressive dichotomy (Φ).** Depth-first 2-means on concatenated RGB + CIE Lab features, no assumed cluster count. A region stops splitting when the mean L2 distance to its centroid is below `δ` (default `10`). Entity segmentation keeps every cluster, including near-black regions.

**Coarse mask (Ω).** With probability `--arbitrary_mask_prob` a quadratic-Bezier blob is drawn around the entity bbox (Paint-by-Example / supplementary Alg. 1); otherwise an extended rectangle is used. Synthesis and entity segmentation pass an all-ones mask.

**VAE.** Colormaps are encoded and decoded with the same `AutoencoderKL` as RGB images. SD inpainting and PixArt-α use an 8× VAE downscale (`vae_scale_factor`). FLUX Fill, SD 3.5, and Z-Image use a 16-channel VAE; FLUX/Z-Image also pack 2×2 tokens.

## Mapping onto Diffusers

To upstream this as an official example:

1. Move `src/unigs/pipeline_unigs.py` → `examples/community/pipeline_unigs.py` (inline the small helpers, or keep the package).
2. Move `src/train_unigs.py` + `src/unigs/` → `examples/research_projects/unigs/`.
3. Load with `DiffusionPipeline.from_pretrained(..., custom_pipeline="pipeline_unigs")` once the community file is in tree.

No custom CUDA ops, no extra segmentation losses — UNet training is standard latent-diffusion MSE on the 8-channel noise; FLUX / SD 3.5 / Z-Image training is flow-matching MSE on image + colormap latents; PixArt-α training is epsilon MSE on the same two streams.
