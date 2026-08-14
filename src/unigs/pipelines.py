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

"""Resolve the UniGS pipeline class, then load with `from_pretrained`."""

from __future__ import annotations

import json
import os
from typing import Optional


_CLASS_NAMES = {
    "UniGSPipeline": "sd15",
    "UniGSFluxPipeline": "flux",
    "UniGSSD3Pipeline": "sd3",
    "UniGSZImagePipeline": "z_image",
    "UniGSPixArtPipeline": "pixart",
}


def pipeline_class_for_family(family: Optional[str]):
    """Return the UniGS pipeline class for a canonical family name."""
    if family == "flux":
        from .pipeline_unigs_flux import UniGSFluxPipeline

        return UniGSFluxPipeline
    if family == "sd3":
        from .pipeline_unigs_sd3 import UniGSSD3Pipeline

        return UniGSSD3Pipeline
    if family == "z_image":
        from .pipeline_unigs_zimage import UniGSZImagePipeline

        return UniGSZImagePipeline
    if family == "pixart":
        from .pipeline_unigs_pixart import UniGSPixArtPipeline

        return UniGSPixArtPipeline
    from .pipeline_unigs import UniGSPipeline

    return UniGSPipeline


def _class_name_from_index(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    index_path = os.path.join(path, "model_index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle).get("_class_name")
    except (OSError, json.JSONDecodeError):
        return None


def pipeline_class_from_pretrained(pretrained_model_name_or_path: Optional[str], family: Optional[str] = None):
    """Pick the UniGS class from a save dir's `_class_name`, else from `family`."""
    class_name = _class_name_from_index(pretrained_model_name_or_path)
    if class_name in _CLASS_NAMES:
        return pipeline_class_for_family(_CLASS_NAMES[class_name])
    if family is None:
        from .backbones import detect_dit_family_from_path

        family = detect_dit_family_from_path(pretrained_model_name_or_path)
    return pipeline_class_for_family(family)


def load_unigs_pipeline(pretrained_model_name_or_path: str, family: Optional[str] = None, **kwargs):
    """`Cls.from_pretrained(...)` for the matching UniGS pipeline class."""
    cls = pipeline_class_from_pretrained(pretrained_model_name_or_path, family=family)
    return cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
