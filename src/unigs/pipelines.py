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

"""Per-model UniGS pipeline classes (no shared `backbone=` factory)."""

from __future__ import annotations

import json
import os
from typing import Optional


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


def load_base_pipeline(family: Optional[str], pretrained_model_name_or_path: Optional[str] = None, **kwargs):
    """Adapt a Hub checkpoint into the matching UniGS pipeline (no `backbone` arg)."""
    if family == "flux":
        from .pipeline_unigs_flux import UniGSFluxPipeline

        return UniGSFluxPipeline.from_fill(pretrained_model_name_or_path, **kwargs)
    if family == "sd3":
        from .pipeline_unigs_sd3 import UniGSSD3Pipeline

        return UniGSSD3Pipeline.from_sd3(pretrained_model_name_or_path, **kwargs)
    if family == "z_image":
        from .pipeline_unigs_zimage import UniGSZImagePipeline

        return UniGSZImagePipeline.from_zimage(pretrained_model_name_or_path, **kwargs)
    if family == "pixart":
        from .pipeline_unigs_pixart import UniGSPixArtPipeline

        return UniGSPixArtPipeline.from_pixart(pretrained_model_name_or_path, **kwargs)
    from .pipeline_unigs import UniGSPipeline

    return UniGSPipeline.from_inpainting(pretrained_model_name_or_path, **kwargs)


def load_saved_unigs_pipeline(path: str, **kwargs):
    """Load a `save_pretrained` UniGS dir by `model_index.json` `_class_name`."""
    index_path = os.path.join(path, "model_index.json")
    class_name = "UniGSPipeline"
    family = None
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        class_name = index.get("_class_name", class_name)
        family = index.get("unigs_dit_family")
    mapping = {
        "UniGSPipeline": "sd15",
        "UniGSFluxPipeline": "flux",
        "UniGSSD3Pipeline": "sd3",
        "UniGSZImagePipeline": "z_image",
        "UniGSPixArtPipeline": "pixart",
        "UniGSDiTPipeline": family,  # legacy combined DiT pipeline
    }
    family = mapping.get(class_name, family)
    if class_name == "UniGSDiTPipeline" and not family:
        from .backbones import detect_dit_family_from_path

        family = detect_dit_family_from_path(path)
    cls = pipeline_class_for_family(family)
    return cls.from_pretrained(path, **kwargs)
