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

"""Tests for FLUX DiT token packing and UniGS context-token concat."""

from __future__ import annotations

import os
import sys
import unittest

import torch
from torch import nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import leaf modules without executing `unigs/__init__.py` (Diffusers / PIL).
import types

_pkg = types.ModuleType("unigs")
_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "src", "unigs")]
sys.modules.setdefault("unigs", _pkg)

from unigs.backbones import (
    DIT_STREAM_COLORMAP,
    DIT_STREAM_CONTROL,
    DIT_STREAM_IMAGE,
    DIT_STREAM_MASK,
    FLUX_FILL_PACKED_IN_CHANNELS,
    UNIGS_DIT_PACKED_IN_CHANNELS,
    is_dit_checkpoint,
    resolve_backbone,
)
from unigs.transformer import (
    adapt_unigs_transformer,
    concat_context_tokens,
    pack_latents,
    pack_mask_as_tokens,
    split_target_tokens,
    unpack_latents,
)


class DummyConfig:
    def __init__(self, in_channels, out_channels, patch_size=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size


class DummyTransformer(nn.Module):
    def __init__(self, in_channels=FLUX_FILL_PACKED_IN_CHANNELS, out_channels=64, inner_dim=32):
        super().__init__()
        self.config = DummyConfig(in_channels, out_channels)
        self.x_embedder = nn.Linear(in_channels, inner_dim)
        with torch.no_grad():
            self.x_embedder.weight.zero_()
            self.x_embedder.weight[:, :UNIGS_DIT_PACKED_IN_CHANNELS] = 1.0
            if self.x_embedder.bias is not None:
                self.x_embedder.bias.fill_(0.5)
        self.proj_out = nn.Linear(inner_dim, out_channels)

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)


class BackboneTests(unittest.TestCase):
    def test_flux_shorthand_resolves_to_fill_dev(self):
        self.assertEqual(resolve_backbone("flux"), "black-forest-labs/FLUX.1-Fill-dev")
        self.assertEqual(resolve_backbone("flux_fill"), "black-forest-labs/FLUX.1-Fill-dev")

    def test_is_dit_checkpoint(self):
        self.assertTrue(is_dit_checkpoint("flux"))
        self.assertTrue(is_dit_checkpoint("black-forest-labs/FLUX.1-Fill-dev"))
        self.assertTrue(is_dit_checkpoint("/tmp/FLUX.1-Fill-dev"))
        self.assertFalse(is_dit_checkpoint("sd15"))
        self.assertFalse(is_dit_checkpoint("stable-diffusion-v1-5/stable-diffusion-inpainting"))


class PackingTests(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        latents = torch.arange(2 * 16 * 8 * 8, dtype=torch.float32).reshape(2, 16, 8, 8)
        packed = pack_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, 16, 64))
        restored = unpack_latents(packed, height=64, width=64, vae_scale_factor=8)
        self.assertTrue(torch.equal(restored, latents))

    def test_mask_tokens_match_packed_layout(self):
        mask = torch.ones(2, 1, 64, 64)
        tokens = pack_mask_as_tokens(mask, latent_height=8, latent_width=8, latent_channels=16)
        self.assertEqual(tuple(tokens.shape), (2, 16, 64))
        self.assertTrue(torch.allclose(tokens, torch.ones_like(tokens)))

    def test_context_token_concat_not_channel_concat(self):
        bsz, seq, dim = 2, 16, 64
        image = torch.zeros(bsz, seq, dim)
        colormap = torch.ones(bsz, seq, dim)
        control = torch.full((bsz, seq, dim), 2.0)
        mask = torch.full((bsz, seq, dim), 3.0)
        hidden, img_ids, target_seq = concat_context_tokens(image, colormap, control, mask, 4, 4)
        self.assertEqual(tuple(hidden.shape), (bsz, 4 * seq, dim))
        self.assertEqual(target_seq, 2 * seq)
        self.assertEqual(tuple(img_ids.shape), (4 * seq, 3))
        self.assertTrue(torch.equal(img_ids[:seq, 0], torch.full((seq,), float(DIT_STREAM_IMAGE))))
        self.assertTrue(torch.equal(img_ids[seq : 2 * seq, 0], torch.full((seq,), float(DIT_STREAM_COLORMAP))))
        self.assertTrue(torch.equal(img_ids[2 * seq : 3 * seq, 0], torch.full((seq,), float(DIT_STREAM_CONTROL))))
        self.assertTrue(torch.equal(img_ids[3 * seq :, 0], torch.full((seq,), float(DIT_STREAM_MASK))))
        pred = torch.randn(bsz, 4 * seq, dim)
        image_pred, cmap_pred = split_target_tokens(pred, target_seq)
        self.assertEqual(tuple(image_pred.shape), (bsz, seq, dim))
        self.assertTrue(torch.equal(image_pred, pred[:, :seq]))
        self.assertTrue(torch.equal(cmap_pred, pred[:, seq : 2 * seq]))


class AdapterTests(unittest.TestCase):
    def test_fill_embedder_keeps_noisy_latent_columns(self):
        transformer = DummyTransformer()
        old_weight = transformer.x_embedder.weight[:, :UNIGS_DIT_PACKED_IN_CHANNELS].clone()
        old_bias = transformer.x_embedder.bias.clone()
        adapted = adapt_unigs_transformer(transformer)
        self.assertEqual(adapted.config.in_channels, UNIGS_DIT_PACKED_IN_CHANNELS)
        self.assertEqual(adapted.x_embedder.in_features, UNIGS_DIT_PACKED_IN_CHANNELS)
        self.assertTrue(torch.equal(adapted.x_embedder.weight, old_weight))
        self.assertTrue(torch.equal(adapted.x_embedder.bias, old_bias))

    def test_already_adapted_is_noop(self):
        transformer = DummyTransformer(in_channels=UNIGS_DIT_PACKED_IN_CHANNELS, out_channels=64)
        embedder = transformer.x_embedder
        self.assertIs(adapt_unigs_transformer(transformer).x_embedder, embedder)


if __name__ == "__main__":
    unittest.main()
