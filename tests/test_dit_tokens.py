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

"""Tests for FLUX Fill channel concat and spatial DiT context-token concat."""

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
    FLUX_FILL_PACKED_LATENT,
    FLUX_FILL_PACKED_MASK,
    UNIGS_DIT_PACKED_IN_CHANNELS,
    UNIGS_DIT_PACKED_OUT_CHANNELS,
    default_dit_guidance,
    dit_uses_flow_matching,
    is_dit_checkpoint,
    resolve_backbone,
    resolve_dit_family,
)
from unigs.dit import (
    ensure_unigs_stream_embed,
    maybe_drop_learned_sigma,
    prepare_zimage_omni_inputs,
    resize_mask_to_latents,
    split_zimage_omni_output,
    unpatchify_stream_tokens,
)
from unigs.transformer import (
    adapt_unigs_transformer,
    concat_fill_channels,
    pack_fill_mask,
    pack_latents,
    prepare_latent_image_ids,
    split_packed_pred,
    unpack_latents,
)


class DummyConfig:
    def __init__(self, in_channels, out_channels, patch_size=1, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyTransformer(nn.Module):
    def __init__(self, in_channels=FLUX_FILL_PACKED_IN_CHANNELS, out_channels=64, inner_dim=32):
        super().__init__()
        self.config = DummyConfig(in_channels, out_channels)
        self.x_embedder = nn.Linear(in_channels, inner_dim)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        with torch.no_grad():
            self.x_embedder.weight.copy_(
                torch.arange(inner_dim * in_channels, dtype=torch.float32).reshape(inner_dim, in_channels) / 1000.0
            )
            if self.x_embedder.bias is not None:
                self.x_embedder.bias.fill_(0.5)
            self.proj_out.weight.copy_(
                torch.arange(out_channels * inner_dim, dtype=torch.float32).reshape(out_channels, inner_dim) / 100.0
            )
            if self.proj_out.bias is not None:
                self.proj_out.bias.fill_(0.25)

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
        self.assertTrue(is_dit_checkpoint("sd3"))
        self.assertTrue(is_dit_checkpoint("sd3.5"))
        self.assertTrue(is_dit_checkpoint("stabilityai/stable-diffusion-3.5-medium"))
        self.assertTrue(is_dit_checkpoint("z_image"))
        self.assertTrue(is_dit_checkpoint("Tongyi-MAI/Z-Image-Turbo"))
        self.assertTrue(is_dit_checkpoint("pixart"))
        self.assertTrue(is_dit_checkpoint("PixArt-alpha/PixArt-XL-2-512-MS"))
        self.assertTrue(is_dit_checkpoint("PixArt-alpha/PixArt-XL-2-1024-MS"))

    def test_resolve_dit_family_and_hub_ids(self):
        self.assertEqual(resolve_dit_family("sd3"), "sd3")
        self.assertEqual(resolve_backbone("sd3"), "stabilityai/stable-diffusion-3.5-medium")
        self.assertEqual(resolve_dit_family("z-image"), "z_image")
        self.assertEqual(resolve_backbone("z_image"), "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(resolve_dit_family("pixart_alpha"), "pixart")
        self.assertEqual(resolve_backbone("pixart"), "PixArt-alpha/PixArt-XL-2-512-MS")
        self.assertEqual(resolve_backbone("pixart_1024"), "PixArt-alpha/PixArt-XL-2-1024-MS")
        self.assertIsNone(resolve_dit_family("sd15"))
        self.assertTrue(dit_uses_flow_matching("sd3"))
        self.assertTrue(dit_uses_flow_matching("z_image"))
        self.assertFalse(dit_uses_flow_matching("pixart"))
        self.assertEqual(default_dit_guidance("sd3"), 4.5)
        self.assertEqual(default_dit_guidance("z_image"), 0.0)
        self.assertEqual(default_dit_guidance("pixart"), 4.5)
        self.assertEqual(default_dit_guidance("flux"), 30.0)


class FluxChannelConcatTests(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        latents = torch.arange(2 * 16 * 8 * 8, dtype=torch.float32).reshape(2, 16, 8, 8)
        packed = pack_latents(latents)
        self.assertEqual(tuple(packed.shape), (2, 16, 64))
        restored = unpack_latents(packed, height=64, width=64, vae_scale_factor=8)
        self.assertTrue(torch.equal(restored, latents))

    def test_fill_mask_packs_to_256_channels(self):
        mask = torch.ones(2, 1, 64, 64)
        tokens = pack_fill_mask(mask, height=8, width=8)
        self.assertEqual(tuple(tokens.shape), (2, 16, FLUX_FILL_PACKED_MASK))
        self.assertTrue(torch.allclose(tokens, torch.ones_like(tokens)))

    def test_channel_concat_not_sequence_concat(self):
        bsz, seq = 2, 16
        image = torch.zeros(bsz, seq, FLUX_FILL_PACKED_LATENT)
        colormap = torch.ones(bsz, seq, FLUX_FILL_PACKED_LATENT)
        control = torch.full((bsz, seq, FLUX_FILL_PACKED_LATENT), 2.0)
        mask = torch.full((bsz, seq, FLUX_FILL_PACKED_MASK), 3.0)
        hidden = concat_fill_channels(image, colormap, control, mask)
        self.assertEqual(tuple(hidden.shape), (bsz, seq, UNIGS_DIT_PACKED_IN_CHANNELS))
        self.assertTrue(torch.equal(hidden[..., :64], image))
        self.assertTrue(torch.equal(hidden[..., 64:128], colormap))
        self.assertTrue(torch.equal(hidden[..., 128:192], control))
        self.assertTrue(torch.equal(hidden[..., 192:], mask))
        pred = torch.randn(bsz, seq, UNIGS_DIT_PACKED_OUT_CHANNELS)
        image_pred, cmap_pred = split_packed_pred(pred)
        self.assertEqual(tuple(image_pred.shape), (bsz, seq, FLUX_FILL_PACKED_LATENT))
        self.assertTrue(torch.equal(image_pred, pred[..., :64]))
        self.assertTrue(torch.equal(cmap_pred, pred[..., 64:]))

    def test_fill_rope_ids_are_single_stream(self):
        ids = prepare_latent_image_ids(8, 8, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(ids.shape), (16, 3))
        self.assertTrue(torch.equal(ids[:, 0], torch.zeros(16)))

    def test_fill_embedder_remaps_masked_image_to_control(self):
        transformer = DummyTransformer()
        old_in = transformer.x_embedder.weight.clone()
        old_bias = transformer.x_embedder.bias.clone()
        old_out = transformer.proj_out.weight.clone()
        adapted = adapt_unigs_transformer(transformer)
        self.assertEqual(adapted.config.in_channels, UNIGS_DIT_PACKED_IN_CHANNELS)
        self.assertEqual(adapted.config.out_channels, UNIGS_DIT_PACKED_OUT_CHANNELS)
        self.assertEqual(adapted.x_embedder.in_features, UNIGS_DIT_PACKED_IN_CHANNELS)
        self.assertEqual(adapted.proj_out.out_features, UNIGS_DIT_PACKED_OUT_CHANNELS)
        self.assertTrue(torch.equal(adapted.x_embedder.weight[:, 0:64], old_in[:, 0:64]))
        self.assertTrue(torch.equal(adapted.x_embedder.weight[:, 64:128], torch.zeros_like(old_in[:, 0:64])))
        self.assertTrue(torch.equal(adapted.x_embedder.weight[:, 128:192], old_in[:, 64:128]))
        self.assertTrue(torch.equal(adapted.x_embedder.weight[:, 192:448], old_in[:, 128:384]))
        self.assertTrue(torch.equal(adapted.x_embedder.bias, old_bias))
        self.assertTrue(torch.equal(adapted.proj_out.weight[:64], old_out))
        self.assertTrue(torch.equal(adapted.proj_out.weight[64:], torch.zeros_like(old_out)))

    def test_already_adapted_is_noop(self):
        transformer = DummyTransformer(
            in_channels=UNIGS_DIT_PACKED_IN_CHANNELS, out_channels=UNIGS_DIT_PACKED_OUT_CHANNELS
        )
        embedder = transformer.x_embedder
        self.assertIs(adapt_unigs_transformer(transformer).x_embedder, embedder)


class SpatialTokenConcatTests(unittest.TestCase):
    def test_unpatchify_stream_prefix_drops_context(self):
        batch, patch, channels, height, width = 2, 2, 16, 8, 8
        seq = (height // patch) * (width // patch)
        hidden = torch.arange(batch * 4 * seq * patch * patch * channels, dtype=torch.float32).reshape(
            batch, 4 * seq, patch * patch * channels
        )
        image, colormap = unpatchify_stream_tokens(hidden, height, width, patch, channels, num_streams=2)
        self.assertEqual(tuple(image.shape), (batch, channels, height, width))
        self.assertEqual(tuple(colormap.shape), (batch, channels, height, width))
        self.assertFalse(torch.equal(image, colormap))

    def test_mask_resized_not_channel_concat(self):
        mask = torch.ones(2, 1, 64, 64)
        latents = resize_mask_to_latents(mask, 8, 8, 16)
        self.assertEqual(tuple(latents.shape), (2, 16, 8, 8))
        self.assertTrue(torch.allclose(latents, torch.ones_like(latents)))

    def test_learned_sigma_is_dropped(self):
        sample = torch.randn(2, 8, 8, 8)
        kept = maybe_drop_learned_sigma(sample, latent_channels=4)
        self.assertEqual(tuple(kept.shape), (2, 4, 8, 8))
        self.assertTrue(torch.equal(kept, sample[:, :4]))

    def test_sd3_stream_embed_is_zero_init(self):
        transformer = nn.Module()
        transformer.config = DummyConfig(
            in_channels=16, out_channels=16, patch_size=2, num_attention_heads=4, attention_head_dim=8
        )
        transformer.inner_dim = 32
        transformer.register_parameter("_dummy", nn.Parameter(torch.zeros(1)))

        def register_to_config(**kwargs):
            for key, value in kwargs.items():
                setattr(transformer.config, key, value)

        transformer.register_to_config = register_to_config
        adapted = adapt_unigs_transformer(transformer, family="sd3")
        self.assertEqual(adapted.config.unigs_dit_family, "sd3")
        self.assertEqual(adapted.config.in_channels, 16)
        self.assertEqual(tuple(adapted.unigs_stream_embed.shape), (4, 1, 32))
        self.assertTrue(torch.equal(adapted.unigs_stream_embed, torch.zeros_like(adapted.unigs_stream_embed)))
        ensure_unigs_stream_embed(adapted)
        self.assertIs(adapted.unigs_stream_embed, transformer.unigs_stream_embed)

    def test_zimage_omni_stacks_targets_and_keeps_context_clean(self):
        image = torch.zeros(2, 16, 8, 8)
        colormap = torch.ones(2, 16, 8, 8)
        control = torch.full((2, 16, 8, 8), 2.0)
        mask = torch.full((2, 16, 8, 8), 3.0)
        prompts = [torch.randn(5, 12), torch.randn(7, 12)]
        x, cap, noise_mask, latent_h = prepare_zimage_omni_inputs(image, colormap, control, mask, prompts)
        self.assertEqual(latent_h, 8)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 3)
        self.assertEqual(tuple(x[0][0].shape), (16, 1, 8, 8))
        self.assertEqual(tuple(x[0][2].shape), (16, 1, 16, 8))
        self.assertEqual(noise_mask[0], [0, 0, 1])
        self.assertEqual(len(cap[0]), 3)
        fake_out = [torch.cat([image[i], colormap[i]], dim=-2).unsqueeze(1) for i in range(2)]
        split = split_zimage_omni_output(fake_out, latent_h, negate=True)
        self.assertEqual(tuple(split.shape), (2, 32, 8, 8))
        restored_image, restored_cmap = split.chunk(2, dim=1)
        self.assertTrue(torch.equal(restored_image, -image))
        self.assertTrue(torch.equal(restored_cmap, -colormap))

    def test_stream_ids_are_distinct(self):
        self.assertEqual(
            (DIT_STREAM_IMAGE, DIT_STREAM_COLORMAP, DIT_STREAM_CONTROL, DIT_STREAM_MASK),
            (0, 1, 2, 3),
        )


if __name__ == "__main__":
    unittest.main()
