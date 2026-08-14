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

"""Location-aware colormap encoder and progressive dichotomy decoder (UniGS §3.2)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image


# Five quantization levels per RGB channel. Black is reserved for background,
# leaving 5**3 - 1 = 124 entity colors. Grid size must satisfy b**2 <= 124.
PALETTE_CHANNEL_VALUES: Tuple[int, ...] = (0, 64, 128, 192, 255)
BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)


def build_location_aware_colors() -> np.ndarray:
    colors = [
        (r, g, b)
        for r in PALETTE_CHANNEL_VALUES
        for g in PALETTE_CHANNEL_VALUES
        for b in PALETTE_CHANNEL_VALUES
        if (r, g, b) != BACKGROUND_COLOR
    ]
    return np.asarray(colors, dtype=np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in ``[0, 1]`` to CIE L*a*b* (D65). Used as PDM pixel features."""
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def _as_mask_list(masks: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return [masks.astype(np.uint8)]
        if masks.ndim == 3:
            return [masks[i].astype(np.uint8) for i in range(masks.shape[0])]
        raise ValueError(f"Expected masks with 2 or 3 dims, got shape {masks.shape}")
    return [np.asarray(mask).astype(np.uint8) for mask in masks]


def _centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


class LocationAwarePalette:
    """Encode entity masks into an RGB colormap (UniGS Ψ).

    The image is partitioned into a ``grid_size x grid_size`` lattice. Each cell
    owns a unique palette color, and an entity inherits the color of the cell
    that contains its center of mass. This keeps colors spatially consistent so
    the UNet's positional bias can recover them, unlike random color assignment.
    """

    def __init__(self, grid_size: int = 11, resolve_collisions: bool = True):
        colors = build_location_aware_colors()
        max_grid = int(np.floor(np.sqrt(len(colors))))
        if grid_size < 1 or grid_size > max_grid:
            raise ValueError(f"`grid_size` must be in [1, {max_grid}], got {grid_size}.")
        self.grid_size = grid_size
        self.colors = colors
        self.resolve_collisions = resolve_collisions

    def color_index_for_centroid(self, cy: float, cx: float, height: int, width: int) -> int:
        gy = min(int(cy / max(height, 1) * self.grid_size), self.grid_size - 1)
        gx = min(int(cx / max(width, 1) * self.grid_size), self.grid_size - 1)
        return gy * self.grid_size + gx

    def encode(
        self,
        masks: Union[np.ndarray, Sequence[np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Convert binary entity masks ``(n, h, w)`` to an RGB colormap ``(h, w, 3)``."""
        mask_list = _as_mask_list(masks)
        if not mask_list and (height is None or width is None):
            raise ValueError("Empty mask list requires explicit `height` and `width`.")
        if mask_list:
            height, width = mask_list[0].shape[-2:]
        colormap = np.zeros((height, width, 3), dtype=np.uint8)
        used: set = set()
        for mask in mask_list:
            center = _centroid(mask)
            if center is None:
                continue
            idx = self.color_index_for_centroid(center[0], center[1], height, width)
            if self.resolve_collisions and idx in used:
                for offset in range(1, len(self.colors)):
                    candidate = (idx + offset) % (self.grid_size * self.grid_size)
                    if candidate not in used:
                        idx = candidate
                        break
            used.add(idx)
            colormap[mask > 0] = self.colors[idx]
        return colormap

    def encode_pil(self, masks: Union[np.ndarray, Sequence[np.ndarray]], size=None) -> Image.Image:
        height, width = (None, None) if size is None else (size[1], size[0])
        colormap = self.encode(masks, height=height, width=width)
        return Image.fromarray(colormap, mode="RGB")


def _two_cluster_kmeans(
    features: np.ndarray,
    max_iter: int = 32,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary k-means (``BK`` in Eq. 6) over ``(n, d)`` pixel features."""
    n = features.shape[0]
    if n < 2:
        assignment = np.zeros(n, dtype=bool)
        return assignment, ~assignment

    # Initialize with the two most extreme points along the largest-variance axis.
    axis = int(np.argmax(features.var(axis=0)))
    order = np.argsort(features[:, axis])
    centers = np.stack([features[order[0]], features[order[-1]]], axis=0).astype(np.float64)

    assignment = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_assignment = distances[:, 1] < distances[:, 0]
        if new_assignment.all() or (~new_assignment).all():
            # Degenerate split: fall back to a median cut on the init axis.
            new_assignment = np.zeros(n, dtype=bool)
            new_assignment[order[n // 2 :]] = True
            assignment = new_assignment
            break
        if np.array_equal(assignment, new_assignment):
            break
        assignment = new_assignment
        new_centers = np.stack(
            [features[~assignment].mean(axis=0), features[assignment].mean(axis=0)],
            axis=0,
        )
        if np.linalg.norm(new_centers - centers) < eps:
            centers = new_centers
            break
        centers = new_centers
    return ~assignment, assignment


class ProgressiveDichotomyModule:
    """Decode a (possibly noisy) colormap into entity masks without knowing *n*.

    Depth-first binary clustering on concatenated RGB+Lab features. A cluster
    stops splitting once the mean L2 distance to its centroid is below ``delta``
    (Eq. 7). Default ``delta=10`` matches the UniGS ablation.
    """

    def __init__(
        self,
        delta: float = 10.0,
        min_area: int = 32,
        max_depth: int = 12,
        background_threshold: float = 8.0,
        include_background: bool = False,
    ):
        self.delta = delta
        self.min_area = min_area
        self.max_depth = max_depth
        self.background_threshold = background_threshold
        self.include_background = include_background

    def _features(self, colormap: np.ndarray) -> np.ndarray:
        rgb = colormap.astype(np.float32)
        lab = rgb_to_lab(np.clip(rgb / 255.0, 0.0, 1.0))
        return np.concatenate([rgb, lab], axis=-1)

    def decode(self, colormap: Union[np.ndarray, Image.Image, torch.Tensor]) -> List[np.ndarray]:
        if isinstance(colormap, Image.Image):
            colormap = np.asarray(colormap.convert("RGB"))
        elif torch.is_tensor(colormap):
            array = colormap.detach().cpu()
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = array.permute(1, 2, 0)
            colormap = (array.float().clamp(0, 1) * 255.0).numpy().astype(np.uint8)
        else:
            colormap = np.asarray(colormap)
            if colormap.dtype != np.uint8:
                if colormap.max() <= 1.0:
                    colormap = (np.clip(colormap, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    colormap = np.clip(colormap, 0, 255).astype(np.uint8)

        features = self._features(colormap)
        rgb_sum = colormap.astype(np.float32).sum(axis=-1)
        valid = np.ones(colormap.shape[:2], dtype=bool) if self.include_background else rgb_sum > self.background_threshold

        masks: List[np.ndarray] = []
        self._split(valid, features, masks, depth=0)
        return masks

    def _split(self, pixel_mask: np.ndarray, features: np.ndarray, out: List[np.ndarray], depth: int) -> None:
        ys, xs = np.nonzero(pixel_mask)
        if ys.size < self.min_area:
            return

        feats = features[ys, xs]
        centroid = feats.mean(axis=0)
        mean_l2 = float(((feats - centroid) ** 2).sum(axis=-1).mean())

        if mean_l2 < self.delta or depth >= self.max_depth:
            entity = np.zeros(pixel_mask.shape, dtype=np.uint8)
            entity[ys, xs] = 1
            out.append(entity)
            return

        left, right = _two_cluster_kmeans(feats)
        if left.sum() < self.min_area or right.sum() < self.min_area:
            entity = np.zeros(pixel_mask.shape, dtype=np.uint8)
            entity[ys, xs] = 1
            out.append(entity)
            return

        mask_left = np.zeros_like(pixel_mask, dtype=bool)
        mask_right = np.zeros_like(pixel_mask, dtype=bool)
        mask_left[ys[left], xs[left]] = True
        mask_right[ys[right], xs[right]] = True
        self._split(mask_left, features, out, depth + 1)
        self._split(mask_right, features, out, depth + 1)

    def decode_pil(self, colormap: Union[np.ndarray, Image.Image, torch.Tensor]) -> List[Image.Image]:
        return [Image.fromarray(mask * 255, mode="L") for mask in self.decode(colormap)]


def colormap_to_tensor(colormap: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """Convert an uint8 colormap to a ``[-1, 1]`` CHW tensor, matching VAE inputs."""
    if isinstance(colormap, Image.Image):
        colormap = np.asarray(colormap.convert("RGB"))
    array = torch.from_numpy(np.asarray(colormap)).float().permute(2, 0, 1) / 255.0
    return array * 2.0 - 1.0
