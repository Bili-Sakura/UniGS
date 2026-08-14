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

"""Coarse mask generator Ω used by the UniGS inpainting protocol (supplementary Alg. 1)."""

from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1


def mask_bbox(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _union_bbox(bboxes: Sequence[BBox]) -> Optional[BBox]:
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return x0, y0, x1, y1


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        float(np.clip(x0, 0, width)),
        float(np.clip(y0, 0, height)),
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
    )


def extend_bbox(bbox: BBox, width: int, height: int, scale: float = 1.3, jitter: float = 0.15) -> BBox:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    scale_x = scale * (1.0 + random.uniform(-jitter, jitter))
    scale_y = scale * (1.0 + random.uniform(-jitter, jitter))
    nw, nh = bw * scale_x, bh * scale_y
    extended = (cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5)
    return _clip_bbox(extended, width, height)


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, num_points: int = 18) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float64)[:, None]
    return (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t**2 * p2


def bezier_blob_mask(bbox: BBox, extended: BBox, width: int, height: int, point_jitter: int = 5) -> np.ndarray:
    """Irregular coarse mask obtained from four quadratic Bezier edges around a bbox."""
    x0, y0, x1, y1 = bbox
    ex0, ey0, ex1, ey1 = extended
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    top = _quadratic_bezier(np.array([x0, y0]), np.array([cx, ey0]), np.array([x1, y0]))
    right = _quadratic_bezier(np.array([x1, y0]), np.array([ex1, cy]), np.array([x1, y1]))
    down = _quadratic_bezier(np.array([x1, y1]), np.array([cx, ey1]), np.array([x0, y1]))
    left = _quadratic_bezier(np.array([x0, y1]), np.array([ex0, cy]), np.array([x0, y0]))

    points = np.concatenate([top, right, down, left], axis=0)
    if point_jitter > 0:
        noise = np.random.randint(-point_jitter, point_jitter + 1, size=points.shape)
        points = points + noise
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([tuple(p) for p in points.tolist()], fill=1)
    return np.asarray(image, dtype=np.uint8)


class CoarseMaskGenerator:
    """Sample a rectangular or free-form coarse mask covering one or more entities.

    ``1`` marks the region the UNet is asked to fill (image and/or colormap).
    """

    def __init__(
        self,
        arbitrary_mask_prob: float = 0.5,
        bbox_scale: float = 1.3,
        bbox_jitter: float = 0.15,
        point_jitter: int = 5,
    ):
        self.arbitrary_mask_prob = arbitrary_mask_prob
        self.bbox_scale = bbox_scale
        self.bbox_jitter = bbox_jitter
        self.point_jitter = point_jitter

    def from_masks(self, masks: Sequence[np.ndarray]) -> np.ndarray:
        if not masks:
            raise ValueError("`masks` must contain at least one entity mask.")
        height, width = masks[0].shape[-2:]
        bboxes = [bbox for bbox in (mask_bbox(mask) for mask in masks) if bbox is not None]
        if not bboxes:
            return np.zeros((height, width), dtype=np.uint8)
        bbox = _union_bbox(bboxes)
        return self.from_bbox(bbox, width=width, height=height)

    def from_bbox(self, bbox: BBox, width: int, height: int) -> np.ndarray:
        bbox = _clip_bbox(bbox, width, height)
        extended = extend_bbox(bbox, width, height, scale=self.bbox_scale, jitter=self.bbox_jitter)
        if random.random() < self.arbitrary_mask_prob:
            return bezier_blob_mask(bbox, extended, width, height, point_jitter=self.point_jitter)

        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = map(int, np.round(extended))
        x0, x1 = np.clip([x0, x1], 0, width)
        y0, y1 = np.clip([y0, y1], 0, height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
        return mask

    def full(self, height: int, width: int) -> np.ndarray:
        """All-ones coarse mask used by image synthesis and entity segmentation."""
        return np.ones((height, width), dtype=np.uint8)
