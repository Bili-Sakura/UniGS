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

"""Task prompt templates from UniGS Table 2."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple


TASK_NAMES = ("inpainting", "synthesis", "referring", "entity")

TASK_PROMPT_TEMPLATES = {
    "inpainting": "inpainting: generate {entities}.",
    "synthesis": "synthesis: generate {entities}.",
    "referring": "referring: find {entities}.",
    "entity": "panoptic: all entities.",
}

# Joint-training sample ratios from the supplementary material.
DEFAULT_TASK_SAMPLE_RATIOS: Dict[str, float] = {
    "inpainting": 0.3,
    "synthesis": 0.3,
    "referring": 0.2,
    "entity": 0.2,
}

# Fallback negative vocabulary for referring-segmentation training.
COCO_THING_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def format_entity_list(labels: Sequence[str]) -> str:
    unique = []
    seen = set()
    for label in labels:
        name = str(label).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)
    if not unique:
        return "object"
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def build_task_prompt(task: str, labels: Optional[Sequence[str]] = None) -> str:
    if task not in TASK_PROMPT_TEMPLATES:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_PROMPT_TEMPLATES)}.")
    template = TASK_PROMPT_TEMPLATES[task]
    if "{entities}" not in template:
        return template
    return template.format(entities=format_entity_list(labels or []))


def sample_negative_labels(
    positive_labels: Sequence[str],
    vocabulary: Sequence[str] = COCO_THING_CLASSES,
    k: Optional[int] = None,
) -> List[str]:
    """Replace referring categories with names that do not appear in the coarse mask."""
    positives = {str(label).strip().lower() for label in positive_labels}
    pool = [name for name in vocabulary if name.lower() not in positives]
    if not pool:
        pool = list(vocabulary)
    k = len(positive_labels) if k is None else k
    k = max(1, min(k, len(pool)))
    return random.sample(pool, k=k)


def sample_task(task: str, task_sample_ratios: Optional[Dict[str, float]] = None) -> str:
    if task != "joint":
        return task
    ratios = task_sample_ratios or DEFAULT_TASK_SAMPLE_RATIOS
    names = list(ratios.keys())
    weights = [float(ratios[name]) for name in names]
    return random.choices(names, weights=weights, k=1)[0]
