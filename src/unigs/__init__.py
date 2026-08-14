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

"""
UniGS: Unified Representation for Image Generation and Segmentation.

This package follows the 🤗 Diffusers example / community-pipeline style so it can be
dropped into `examples/research_projects/unigs` (training) and
`examples/community/pipeline_unigs.py` (inference) with minimal changes.

Paper: https://arxiv.org/abs/2312.01985
"""

from .backbones import (
    BACKBONES,
    DEFAULT_BACKBONE,
    DIT_BACKBONES,
    FLUX_FILL_CHECKPOINT,
    INPAINTING_BACKBONES,
    INPAINTING_UNET_IN_CHANNELS,
    UNET_BACKBONES,
    is_dit_checkpoint,
    resolve_backbone,
    resolve_inpainting_checkpoint,
)
from .coarse_mask import CoarseMaskGenerator
from .colormap import LocationAwarePalette, ProgressiveDichotomyModule
from .dataset import UniGSInstanceDataset, collate_fn
from .pipeline_unigs import UniGSPipeline, UniGSPipelineOutput
from .prompts import TASK_PROMPT_TEMPLATES, build_task_prompt
from .transformer import adapt_unigs_transformer, concat_context_tokens, pack_latents
from .unet import UNIGS_IN_CHANNELS, UNIGS_OUT_CHANNELS, adapt_unigs_unet

try:
    from .pipeline_unigs_flux import UniGSFluxPipeline, encode_flux_prompt
except ImportError:  # Diffusers without FLUX
    UniGSFluxPipeline = None  # type: ignore[misc, assignment]
    encode_flux_prompt = None  # type: ignore[misc, assignment]


__all__ = [
    "BACKBONES",
    "CoarseMaskGenerator",
    "DEFAULT_BACKBONE",
    "DIT_BACKBONES",
    "FLUX_FILL_CHECKPOINT",
    "INPAINTING_BACKBONES",
    "INPAINTING_UNET_IN_CHANNELS",
    "LocationAwarePalette",
    "ProgressiveDichotomyModule",
    "UNET_BACKBONES",
    "UniGSFluxPipeline",
    "UniGSInstanceDataset",
    "UniGSPipeline",
    "UniGSPipelineOutput",
    "TASK_PROMPT_TEMPLATES",
    "UNIGS_IN_CHANNELS",
    "UNIGS_OUT_CHANNELS",
    "adapt_unigs_transformer",
    "adapt_unigs_unet",
    "build_task_prompt",
    "collate_fn",
    "concat_context_tokens",
    "encode_flux_prompt",
    "is_dit_checkpoint",
    "pack_latents",
    "resolve_backbone",
    "resolve_inpainting_checkpoint",
]
