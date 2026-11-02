"""Exercise ICON prefix reuse with a real, small Qwen3-VL on CPU.

Run with the existing ICON environment, without installing pytest::

    .venv-icon/bin/python -m unittest discover -s tests \
        -p test_icon_context_kv_cache.py -v

The ordinary RLInf test environment has an older Transformers version; these
tests skip there because its Qwen3-VL implementation is incompatible.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/robometer_v2/robometer/models/context_kv_cache.py"

try:
    from transformers import Qwen3VLConfig, Qwen3VLModel
except ImportError:
    Qwen3VLConfig = Qwen3VLModel = None


def load_cache():
    spec = importlib.util.spec_from_file_location("icon_context_kv_cache_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FixedContextKVCache


def tiny_model():
    torch.manual_seed(712)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 1024,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=110,
        video_token_id=109,
        vision_start_token_id=111,
        vision_end_token_id=112,
        pad_token_id=0,
    )
    return Qwen3VLModel(config).eval()


def make_inputs(query_frames=2, prompts=((1, 5),), padding_side="left"):
    """Pack two context images plus query images in the real Qwen3 layout."""
    chunk = [111, 110, 110, 110, 110, 112, 113]
    sequences = [list(prompt) + chunk * 2 + [9] + chunk * query_frames + [2] for prompt in prompts]
    width = max(map(len, sequences))
    ids = torch.zeros(len(sequences), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        start = width - len(sequence) if padding_side == "left" else 0
        ids[row, start:start + len(sequence)] = torch.tensor(sequence)
        mask[row, start:start + len(sequence)] = 1
    generator = torch.Generator().manual_seed(815)
    image_count = len(sequences) * (2 + query_frames)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "pixel_values": torch.randn(image_count * 16, 24, generator=generator),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * image_count),
    }


class BenchmarkOutputConversionTests(unittest.TestCase):
    def test_success_sigmoid_uses_original_output_dtype_like_http(self):
        spec = importlib.util.spec_from_file_location(
            "icon_kv_benchmark_test", ROOT / "scripts/benchmark_icon_context_kv_cache.py"
        )
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        success = torch.tensor([[-0.5, 0.5]], dtype=torch.bfloat16)
        progress = torch.tensor([[[0.5, -0.5]]], dtype=torch.bfloat16)
        output = SimpleNamespace(progress_logits={"A": progress}, success_logits={"A": success})
        captured = benchmark.capture(output, lambda value: value.softmax(-1)[..., 1])
        self.assertEqual(captured["progress_logits/A"].dtype, torch.float32)
        self.assertTrue(torch.equal(captured["success_probs"], success.sigmoid().float()))
        self.assertFalse(torch.equal(captured["success_probs"], success.float().sigmoid()))


@unittest.skipIf(Qwen3VLModel is None, "Run in .venv-icon with Transformers 4.57.6")
class ContextKVCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cache_class = load_cache()

    def setUp(self):
        self.model = tiny_model()
        self.cache = self.cache_class(max_entries=8, max_bytes=64 * 1024**2)

    def baseline(self, inputs):
        with torch.no_grad():
            return self.model(**inputs, use_cache=False).last_hidden_state

    def cached(self, inputs, cache=None, **extra):
        with torch.no_grad():
            return (cache or self.cache).forward(
                self.model,
                context_frame_counts=torch.full((len(inputs["input_ids"]),), 2),
                context_progress_token_counts=torch.full((len(inputs["input_ids"]),), 2),
                **inputs,
                **extra,
            )

    def assert_active_equal(self, actual, expected, inputs):
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(
            actual[inputs["attention_mask"].bool()],
            expected[inputs["attention_mask"].bool()],
            rtol=2e-5,
            atol=2e-5,
        )

    def test_miss_and_repeated_hit_match_uncached_for_query_lengths(self):
        for length in (1, 2, 8):
            with self.subTest(query_frames=length):
                inputs = make_inputs(length)
                expected = self.baseline(inputs)
                self.assert_active_equal(self.cached(inputs), expected, inputs)
                self.assert_active_equal(self.cached(inputs), expected, inputs)
        info = self.cache.info()
        self.assertEqual(info["fallbacks"], 0)
        self.assertEqual(info["misses"], 1)
        self.assertEqual(info["hits"], 5)
        self.assertEqual(info["prefill_batches"], 1)
        self.assertEqual(info["entries"], 1)

    def test_changed_query_does_not_mutate_stored_prefix(self):
        inputs = make_inputs()
        expected = self.baseline(inputs)
        self.assert_active_equal(self.cached(inputs), expected, inputs)
        changed = copy.deepcopy(inputs)
        changed["pixel_values"][32:] += 0.37
        changed_expected = self.baseline(changed)
        self.assert_active_equal(self.cached(changed), changed_expected, changed)
        self.assert_active_equal(self.cached(inputs), expected, inputs)
        self.assertEqual(self.cache.info()["misses"], 1)
        self.assertEqual(self.cache.info()["hits"], 2)

    def test_instruction_and_context_image_changes_have_distinct_entries(self):
        inputs = make_inputs()
        self.cached(inputs)
        instruction = copy.deepcopy(inputs)
        instruction["input_ids"][0, 1] = 6
        self.assert_active_equal(self.cached(instruction), self.baseline(instruction), instruction)
        context = copy.deepcopy(inputs)
        context["pixel_values"][:32] += 0.5
        self.assert_active_equal(self.cached(context), self.baseline(context), context)
        self.assert_active_equal(self.cached(inputs), self.baseline(inputs), inputs)
        self.assertEqual(self.cache.info()["misses"], 3)
        self.assertEqual(self.cache.info()["entries"], 3)

    def test_context_image_geometry_change_has_distinct_prefix(self):
        inputs = make_inputs()
        self.cached(inputs)
        changed = copy.deepcopy(inputs)
        # Same number of pixels and visual tokens, different spatial positions.
        changed["image_grid_thw"][0] = torch.tensor([1, 2, 8])
        self.assert_active_equal(self.cached(changed), self.baseline(changed), changed)
        self.assertEqual(self.cache.info()["misses"], 2)
        self.assertEqual(self.cache.info()["entries"], 2)

    def test_left_and_right_padded_batches_match_active_states(self):
        for side in ("left", "right"):
            with self.subTest(padding=side):
                inputs = make_inputs(prompts=((1, 5), (1, 5, 6, 7, 8)), padding_side=side)
                expected = self.baseline(inputs)
                self.assert_active_equal(self.cached(inputs), expected, inputs)
                self.assert_active_equal(self.cached(inputs), expected, inputs)
        self.assertEqual(self.cache.info()["misses"], 2)
        self.assertEqual(self.cache.info()["hits"], 6)
        self.assertEqual(self.cache.info()["fallbacks"], 0)

    def test_batch_with_one_warm_and_one_missing_prefix(self):
        warm = make_inputs(prompts=((1, 5),))
        self.cached(warm)
        mixed = make_inputs(prompts=((1, 5), (1, 6, 7, 8)))
        expected = self.baseline(mixed)
        self.assert_active_equal(self.cached(mixed), expected, mixed)
        info = self.cache.info()
        self.assertEqual(info["hits"], 1)
        self.assertEqual(info["misses"], 2)
        self.assertEqual(info["prefill_batches"], 2)
        self.assertEqual(info["entries"], 2)
        self.assert_active_equal(self.cached(mixed), expected, mixed)
        self.assertEqual(self.cache.info()["hits"], 3)

    def test_identical_prefixes_in_one_batch_are_prefilled_only_once(self):
        inputs = make_inputs(prompts=((1, 5), (1, 5)))
        # Two query histories share identical context pixels and token prefixes.
        pixels_per_row = (2 + 2) * 16
        inputs["pixel_values"][pixels_per_row:pixels_per_row + 32] = inputs["pixel_values"][:32]
        expected = self.baseline(inputs)
        self.assert_active_equal(self.cached(inputs), expected, inputs)
        info = self.cache.info()
        self.assertEqual(info["misses"], 1)
        self.assertEqual(info["prefill_batches"], 1)
        self.assertEqual(info["entries"], 1)
        self.assert_active_equal(self.cached(inputs), expected, inputs)
        self.assertEqual(self.cache.info()["hits"], 2)

    def test_entry_limit_eviction_and_clear_preserve_results(self):
        cache = self.cache_class(max_entries=1, max_bytes=64 * 1024**2)
        first = make_inputs()
        second = make_inputs(prompts=((1, 6),))
        for inputs in (first, second, first):
            self.assert_active_equal(self.cached(inputs, cache=cache), self.baseline(inputs), inputs)
        self.assertEqual(cache.info()["entries"], 1)
        self.assertEqual(cache.info()["evictions"], 2)
        self.assertEqual(cache.info()["misses"], 3)
        cache.clear()
        self.assertEqual(cache.info()["entries"], 0)
        self.assertEqual(cache.info()["bytes"], 0)
        self.assert_active_equal(self.cached(first, cache=cache), self.baseline(first), first)

    def test_too_small_byte_budget_preserves_results(self):
        cache = self.cache_class(max_entries=8, max_bytes=1)
        inputs = make_inputs()
        self.assert_active_equal(self.cached(inputs, cache=cache), self.baseline(inputs), inputs)
        self.assert_active_equal(self.cached(inputs, cache=cache), self.baseline(inputs), inputs)
        self.assertEqual(cache.info()["entries"], 0)
        self.assertEqual(cache.info()["bytes"], 0)
        self.assertEqual(cache.info()["oversized_entries"], 2)

    def test_lru_hit_protects_recent_prefix_from_eviction(self):
        cache = self.cache_class(max_entries=2, max_bytes=64 * 1024**2)
        first = make_inputs(prompts=((1, 5),))
        second = make_inputs(prompts=((1, 6),))
        third = make_inputs(prompts=((1, 7),))
        for inputs in (first, second, first, third, first):
            self.assert_active_equal(self.cached(inputs, cache=cache), self.baseline(inputs), inputs)
        self.assertEqual(cache.info()["misses"], 3)
        self.assertEqual(cache.info()["hits"], 2)
        self.assertEqual(cache.info()["evictions"], 1)
        self.assert_active_equal(self.cached(second, cache=cache), self.baseline(second), second)
        self.assertEqual(cache.info()["misses"], 4)
        self.assertEqual(cache.info()["evictions"], 2)

    def test_training_mode_preserves_regular_forward(self):
        inputs = make_inputs()
        self.model.train()
        self.assert_active_equal(self.cached(inputs), self.baseline(inputs), inputs)
        self.assertEqual(self.cache.info()["fallback_reasons"]["training_or_gradients"], 1)

    def test_preference_request_preserves_regular_forward(self):
        inputs = make_inputs()
        self.assert_active_equal(self.cached(inputs, sample_type="preference"), self.baseline(inputs), inputs)
        self.assertEqual(self.cache.info()["fallback_reasons"]["sample_type"], 1)

    def test_weight_change_invalidates_prior_prefix(self):
        inputs = make_inputs()
        self.cached(inputs)
        with torch.no_grad():
            next(self.model.parameters()).add_(0.001)
        self.assert_active_equal(self.cached(inputs), self.baseline(inputs), inputs)
        self.assertEqual(self.cache.info()["invalidations"], 1)
        self.assertEqual(self.cache.info()["misses"], 2)

    def test_gradient_enabled_call_does_not_reuse_detached_prefix(self):
        inputs = make_inputs()
        self.cached(inputs)
        actual = self.cache.forward(
            self.model,
            context_frame_counts=torch.tensor([2]),
            context_progress_token_counts=torch.tensor([2]),
            **inputs,
        )
        self.assertTrue(actual.requires_grad)
        self.assertEqual(self.cache.info()["entries"], 0)
        self.assertEqual(self.cache.info()["fallback_reasons"]["training_or_gradients"], 1)
        self.assert_active_equal(actual, self.baseline(inputs), inputs)


if __name__ == "__main__":
    unittest.main()
