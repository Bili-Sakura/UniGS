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

"""Instance / panoptic dataset utilities for UniGS training."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF

from .coarse_mask import CoarseMaskGenerator
from .colormap import LocationAwarePalette, colormap_to_tensor
from .prompts import build_task_prompt, sample_negative_labels, sample_task


def _load_rgb(path_or_image) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    return Image.open(path_or_image).convert("RGB")


def polygons_to_mask(segmentation: Sequence, height: int, width: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    polygons = segmentation if segmentation and isinstance(segmentation[0], (list, tuple)) else [segmentation]
    for poly in polygons:
        if poly is None or len(poly) < 6:
            continue
        xy = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(xy, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def decode_coco_segmentation(segmentation, height: int, width: int) -> Optional[np.ndarray]:
    if segmentation is None:
        return None
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size", [height, width])
        if isinstance(counts, list):
            return _decode_uncompressed_rle(counts, size[0], size[1])
        try:
            from pycocotools import mask as mask_utils

            return mask_utils.decode(segmentation).astype(np.uint8)
        except Exception:
            return None
    if isinstance(segmentation, (list, tuple)):
        return polygons_to_mask(segmentation, height, width)
    return None


def _decode_uncompressed_rle(counts: Sequence[int], height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    value = 0
    for count in counts:
        end = min(pos + int(count), flat.size)
        if value:
            flat[pos:end] = 1
        pos = end
        value = 1 - value
        if pos >= flat.size:
            break
    return flat.reshape((width, height), order="F").T


def unique_id_masks(panoptic: np.ndarray, ignore_ids=(0,)) -> List[np.ndarray]:
    if panoptic.ndim == 3:
        packed = (
            panoptic[..., 0].astype(np.int32) * 256 * 256
            + panoptic[..., 1].astype(np.int32) * 256
            + panoptic[..., 2].astype(np.int32)
        )
    else:
        packed = panoptic.astype(np.int32)
    ignore = set(ignore_ids)
    masks = []
    for entity_id in np.unique(packed):
        if int(entity_id) in ignore:
            continue
        masks.append((packed == entity_id).astype(np.uint8))
    return masks


def _normalize_image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return tensor * 2.0 - 1.0


def paired_resize_crop_flip(
    image: Image.Image,
    colormap: Image.Image,
    coarse_mask: Image.Image,
    control: Image.Image,
    resolution: int,
    center_crop: bool,
    random_flip: bool,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    image = TF.resize(image, resolution, interpolation=TF.InterpolationMode.BILINEAR)
    control = TF.resize(control, resolution, interpolation=TF.InterpolationMode.BILINEAR)
    colormap = TF.resize(colormap, resolution, interpolation=TF.InterpolationMode.NEAREST)
    coarse_mask = TF.resize(coarse_mask, resolution, interpolation=TF.InterpolationMode.NEAREST)

    if center_crop:
        image = TF.center_crop(image, resolution)
        control = TF.center_crop(control, resolution)
        colormap = TF.center_crop(colormap, resolution)
        coarse_mask = TF.center_crop(coarse_mask, resolution)
    else:
        i, j, h, w = RandomCrop.get_params(image, (resolution, resolution))
        image = TF.crop(image, i, j, h, w)
        control = TF.crop(control, i, j, h, w)
        colormap = TF.crop(colormap, i, j, h, w)
        coarse_mask = TF.crop(coarse_mask, i, j, h, w)

    if random_flip and random.random() < 0.5:
        image = TF.hflip(image)
        control = TF.hflip(control)
        colormap = TF.hflip(colormap)
        coarse_mask = TF.hflip(coarse_mask)
    return image, colormap, coarse_mask, control


class UniGSInstanceDataset(Dataset):
    """Build UniGS training samples from COCO-style instance annotations or a Hub dataset.

    Each item contains the 4 tensors expected by the inpainting protocol:

    * ``pixel_values`` — RGB image in ``[-1, 1]``
    * ``colormap_values`` — location-aware entity colormap in ``[-1, 1]``
    * ``control_values`` — task-dependent control image in ``[-1, 1]``
    * ``coarse_mask`` — ``(1, H, W)`` mask, ``1`` = region to fill
    * ``prompt`` — task-prefixed text prompt (always)
    * ``input_ids`` — CLIP token ids when ``tokenizer`` is provided (UNet backbones)
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer=None,
        resolution: int = 512,
        task: str = "joint",
        max_entities: int = 4,
        center_crop: bool = False,
        random_flip: bool = True,
        referring_neg_prob: float = 0.2,
        arbitrary_mask_prob: float = 0.5,
        grid_size: int = 11,
        caption_column: Optional[str] = None,
    ):
        if not records:
            raise ValueError("UniGSInstanceDataset received an empty record list.")
        self.records = list(records)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.task = task
        self.max_entities = max_entities
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.referring_neg_prob = referring_neg_prob
        self.caption_column = caption_column
        self.palette = LocationAwarePalette(grid_size=grid_size)
        self.coarse_mask_generator = CoarseMaskGenerator(arbitrary_mask_prob=arbitrary_mask_prob)

    def _record_masks(self, record: Dict[str, Any], image: Image.Image) -> List[np.ndarray]:
        if "masks" in record and record["masks"] is not None:
            return [np.asarray(mask, dtype=np.uint8) for mask in record["masks"]]
        height = int(record.get("height") or image.height)
        width = int(record.get("width") or image.width)
        masks = []
        for segmentation in record.get("segmentations", []):
            mask = decode_coco_segmentation(segmentation, height, width)
            if mask is None:
                continue
            if mask.shape != (image.height, image.width):
                mask = np.array(Image.fromarray(mask * 255).resize(image.size, Image.NEAREST))
                mask = (mask > 0).astype(np.uint8)
            masks.append(mask)
        return masks

    def __len__(self) -> int:
        return len(self.records)

    def _sample_entities(self, masks: List[np.ndarray], labels: List[str], task: str):
        n = len(masks)
        if n == 0:
            return [], []
        if task == "entity" or n <= self.max_entities:
            return masks, labels
        indices = random.sample(range(n), k=self.max_entities)
        return [masks[i] for i in indices], [labels[i] for i in indices]

    def _control_and_mask(
        self,
        image: Image.Image,
        colormap: Image.Image,
        entity_masks: List[np.ndarray],
        task: str,
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        if task in {"synthesis", "entity"}:
            coarse = self.coarse_mask_generator.full(height, width)
        else:
            coarse = (
                self.coarse_mask_generator.from_masks(entity_masks)
                if entity_masks
                else self.coarse_mask_generator.full(height, width)
            )

        coarse_pil = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
        if task == "inpainting":
            black = Image.new("RGB", image.size, (0, 0, 0))
            control = Image.composite(black, image, coarse_pil)
            return control, coarse_pil
        if task == "synthesis":
            return colormap.copy(), coarse_pil
        # referring + entity: preserve the original image as control.
        return image.copy(), coarse_pil

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = _load_rgb(record["image"])
        masks = self._record_masks(record, image)
        labels: List[str] = [str(label) for label in record.get("labels", ["object"] * len(masks))]
        if len(labels) < len(masks):
            labels = labels + ["object"] * (len(masks) - len(labels))

        task = sample_task(self.task)
        entity_masks, entity_labels = self._sample_entities(masks, labels, task)
        if not entity_masks:
            dummy = np.zeros((image.height, image.width), dtype=np.uint8)
            dummy[image.height // 4 : 3 * image.height // 4, image.width // 4 : 3 * image.width // 4] = 1
            entity_masks, entity_labels = [dummy], ["object"]

        colormap = self.palette.encode_pil(entity_masks, size=image.size)
        control, coarse_mask = self._control_and_mask(image, colormap, entity_masks, task)

        image, colormap, coarse_mask, control = paired_resize_crop_flip(
            image,
            colormap,
            coarse_mask,
            control,
            resolution=self.resolution,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
        )

        prompt_labels = list(entity_labels)
        if task == "referring" and random.random() < self.referring_neg_prob:
            prompt_labels = sample_negative_labels(entity_labels)
        prompt = build_task_prompt(task, prompt_labels)
        if self.caption_column and record.get("caption") and task == "synthesis":
            prompt = f"synthesis: {record['caption']}"

        coarse_tensor = TF.to_tensor(coarse_mask)
        if coarse_tensor.max() > 1.0:
            coarse_tensor = coarse_tensor / 255.0
        coarse_tensor = (coarse_tensor > 0.5).float()

        # Rebuild inpainting control after geometric transforms so the hole matches the crop.
        pixel_values = _normalize_image_tensor(image)
        colormap_values = colormap_to_tensor(colormap)
        if task == "inpainting":
            control_values = pixel_values * (1.0 - coarse_tensor)
        elif task == "synthesis":
            control_values = colormap_values
        else:
            control_values = pixel_values

        sample: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "colormap_values": colormap_values,
            "control_values": control_values,
            "coarse_mask": coarse_tensor,
            "prompt": prompt,
            "task": task,
        }
        if self.tokenizer is not None:
            sample["input_ids"] = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        return sample


def collate_fn(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
        "colormap_values": torch.stack([ex["colormap_values"] for ex in examples]),
        "control_values": torch.stack([ex["control_values"] for ex in examples]),
        "coarse_mask": torch.stack([ex["coarse_mask"] for ex in examples]),
        "prompts": [ex["prompt"] for ex in examples],
    }
    if "input_ids" in examples[0]:
        batch["input_ids"] = torch.stack([ex["input_ids"] for ex in examples])
    return batch


def load_coco_records(image_dir: str, annotation_file: str, min_area: int = 32) -> List[Dict[str, Any]]:
    with open(annotation_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {int(cat["id"]): cat["name"] for cat in coco.get("categories", [])}
    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    records: List[Dict[str, Any]] = []
    for image_id, info in images.items():
        file_name = info["file_name"]
        path = file_name if os.path.isabs(file_name) else os.path.join(image_dir, file_name)
        if not os.path.exists(path):
            continue
        height, width = int(info.get("height", 0)), int(info.get("width", 0))
        segmentations, labels = [], []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("area", min_area) < min_area:
                continue
            if ann.get("segmentation") is None:
                continue
            segmentations.append(ann["segmentation"])
            labels.append(categories.get(int(ann.get("category_id", -1)), "object"))
        if not segmentations:
            continue
        records.append(
            {
                "image": path,
                "segmentations": segmentations,
                "labels": labels,
                "height": height,
                "width": width,
            }
        )
    return records


def records_from_hf_dataset(
    dataset,
    image_column: str = "image",
    mask_column: Optional[str] = "masks",
    label_column: Optional[str] = "labels",
    objects_column: Optional[str] = "objects",
    caption_column: Optional[str] = None,
    min_area: int = 32,
) -> List[Dict[str, Any]]:
    """Normalize a 🤗 Datasets object into UniGS records.

    Supported layouts:

    * ``image`` + ``masks`` (list of 2D arrays) + optional ``labels``
    * ``image`` + ``objects`` dict with ``mask`` / ``category`` / ``label`` fields
    * ``image`` + a panoptic map column (unique ids / colors)
    """
    records: List[Dict[str, Any]] = []
    for row in dataset:
        image = row[image_column]
        masks: List[np.ndarray] = []
        labels: List[str] = []

        if objects_column and objects_column in row and row[objects_column] is not None:
            objects = row[objects_column]
            obj_masks = objects.get("mask") or objects.get("masks") or objects.get("segmentation")
            obj_labels = objects.get("category") or objects.get("label") or objects.get("labels") or []
            if obj_masks is None and "bbox" in objects:
                obj_masks = []
            if obj_masks is not None:
                for i, mask in enumerate(obj_masks):
                    array = _coerce_mask(mask, image)
                    if array is None or int(array.sum()) < min_area:
                        continue
                    masks.append(array)
                    if i < len(obj_labels):
                        label = obj_labels[i]
                        labels.append(label if isinstance(label, str) else str(label))
                    else:
                        labels.append("object")
        elif mask_column and mask_column in row and row[mask_column] is not None:
            raw = row[mask_column]
            if isinstance(raw, Image.Image) or (hasattr(raw, "ndim") and getattr(raw, "ndim", 2) in (2, 3) and not isinstance(raw, list)):
                array = np.asarray(raw)
                masks = unique_id_masks(array)
                labels = ["object"] * len(masks)
            else:
                for i, mask in enumerate(raw):
                    array = _coerce_mask(mask, image)
                    if array is None or int(array.sum()) < min_area:
                        continue
                    masks.append(array)
                    if label_column and label_column in row and row[label_column] is not None and i < len(row[label_column]):
                        labels.append(str(row[label_column][i]))
                    else:
                        labels.append("object")

        if not masks:
            continue
        record = {"image": image, "masks": masks, "labels": labels}
        if caption_column and caption_column in row:
            record["caption"] = row[caption_column]
        records.append(record)
    return records


def _coerce_mask(mask, image: Image.Image) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] in (1, 3, 4) else array[0]
    if array.ndim != 2:
        return None
    if array.shape != (image.height, image.width) and isinstance(image, Image.Image):
        array = np.array(Image.fromarray((array > 0).astype(np.uint8) * 255).resize(image.size, Image.NEAREST))
    return (array > 0).astype(np.uint8)
