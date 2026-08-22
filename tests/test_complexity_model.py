"""Tests for the model/ pipeline: chunking, dedup, pooling, caching, leakage, heads.

These deliberately avoid loading a sentence-transformer. The encoder is a frozen
third-party artefact; what needs testing is the code around it — the plan that
decides what gets encoded, the reduction that turns chunk vectors into row
vectors, and the guards that keep the test split sealed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import SEGMENTS, PipelineConfig, accurate_config, fast_config  # noqa: E402
from model.data import Split, check_leakage  # noqa: E402
from model.embed import (  # noqa: E402
    _cache_path,
    build_chunk_plan,
    pack_chunks,
    pool_plan,
    segment_texts,
    subsample_chunks,
)
from model.evaluate import band_breakdown, confusion, regression_metrics  # noqa: E402
from model.features import Standardizer, band_of, static_feature_matrix  # noqa: E402
from model.heads import MeanBaseline, RidgeHead  # noqa: E402


def row(system="", user="", **features):
    defaults = {
        "request_id": features.pop("request_id", "chunk.jsonl:1"),
        "system_prompt": system,
        "user_prompt": user,
        "user_messages": [user] if user else [],
        "static_text_features": features or {},
    }
    return defaults


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_the_character_budget(self):
        text = "\n\n".join(f"block {i} " + "x" * 100 for i in range(20))
        chunks = pack_chunks(text, 300)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_oversized_paragraph_is_split_rather_than_dropped(self):
        chunks = pack_chunks("y" * 1000, 300)
        self.assertEqual(sum(len(c) for c in chunks), 1000)
        self.assertTrue(all(len(c) <= 300 for c in chunks))

    def test_a_shared_prefix_produces_shared_chunks(self):
        """Packing stays in phase while two prompts agree, so a common prefix
        deduplicates. This is where most of the real saving comes from: the Viktor
        system prompt is near-identical across requests."""
        shared = "\n\n".join(f"shared paragraph {i}" for i in range(6))
        first = pack_chunks(shared + "\n\nthen this request diverges", 120)
        second = pack_chunks(shared + "\n\nwhereas this one asks something else", 120)
        self.assertTrue(set(first) & set(second), "a shared prefix produced no shared chunk")

    def test_an_oversized_paragraph_splits_the_same_way_in_any_context(self):
        """Blocks longer than the budget are cut on their own boundaries, so an
        embedded log or code block deduplicates wherever it appears."""
        block = "L" * 500
        first = pack_chunks("short intro\n\n" + block, 120)
        second = pack_chunks("a rather longer introduction here\n\n" + block, 120)
        self.assertTrue(set(first) & set(second))

    def test_packing_does_not_align_after_a_differently_sized_prefix(self):
        """The limit of the approach, asserted so it is not mistaken for a guarantee:
        once two prompts differ in length before a shared run, greedy packing falls out
        of phase and the shared run is chunked differently. Content-defined boundaries
        were measured on the real export and recovered nothing (1.58x either way), so
        the simpler rule stands."""
        shared = "\n\n".join(f"shared paragraph {i}" for i in range(6))
        first = pack_chunks("tiny preamble\n\n" + shared, 120)
        second = pack_chunks("a considerably longer preamble than the other\n\n" + shared, 120)
        self.assertFalse(set(first) & set(second))

    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(pack_chunks("", 100), [])

    def test_subsample_keeps_first_and_last(self):
        chunks = [f"c{i}" for i in range(50)]
        kept = subsample_chunks(chunks, 6)
        self.assertLessEqual(len(kept), 6)
        self.assertEqual(kept[0], "c0")
        self.assertEqual(kept[-1], "c49")

    def test_zero_budget_means_uncapped_not_dropped(self):
        chunks = [f"c{i}" for i in range(10)]
        self.assertEqual(subsample_chunks(chunks, 0), chunks)

    def test_user_messages_are_never_embedded(self):
        """user_messages is exactly the join that produced user_prompt, so
        embedding it would double the cost for no new signal."""
        texts = segment_texts(row(system="sys", user="a\n\nb"))
        self.assertEqual(set(texts), set(SEGMENTS))
        self.assertNotIn("user_messages", texts)


class ChunkPlanTests(unittest.TestCase):
    def test_identical_chunks_are_encoded_once(self):
        rows = [row(system="shared system prompt", user=f"unique user {i}") for i in range(5)]
        plan = build_chunk_plan(rows, fast_config(max_chunk_chars=200))
        self.assertEqual(plan.total_chunks, 10)
        self.assertEqual(len(plan.unique_chunks), 6)  # 1 shared system + 5 distinct users
        self.assertGreater(plan.dedup_ratio, 1.0)

    def test_pooling_matches_a_naive_per_row_mean(self):
        rows = [
            row(system="a\n\nb\n\nc", user="d"),
            row(system="", user="a\n\nb"),
            row(system="z" * 900, user=""),
        ]
        config = fast_config(max_chunk_chars=10, system_max_chunks=8, user_max_chunks=8)
        plan = build_chunk_plan(rows, config)
        matrix = np.random.default_rng(0).normal(size=(len(plan.unique_chunks), 4)).astype(np.float32)
        pooled = pool_plan(plan, matrix)

        budgets = {"system": 8, "user": 8}
        expected = []
        for source in rows:
            texts = segment_texts(source)
            parts = []
            for segment in SEGMENTS:
                chunks = subsample_chunks(
                    pack_chunks(texts[segment], config.max_chunk_chars), budgets[segment]
                )
                if chunks:
                    vector = np.mean([matrix[plan.unique_chunks.index(c)] for c in chunks], axis=0)
                    vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
                else:
                    vector = np.zeros(4, dtype=np.float32)
                parts.append(vector)
            expected.append(np.concatenate(parts))
        np.testing.assert_allclose(pooled, np.array(expected, dtype=np.float32), atol=1e-6)

    def test_missing_segment_pools_to_zero(self):
        plan = build_chunk_plan([row(system="", user="hello")], fast_config())
        matrix = np.ones((len(plan.unique_chunks), 3), dtype=np.float32)
        pooled = pool_plan(plan, matrix)
        self.assertTrue(np.allclose(pooled[0, :3], 0.0))
        self.assertFalse(np.allclose(pooled[0, 3:], 0.0))

    def test_plan_is_deterministic(self):
        rows = [row(system=f"s{i}", user=f"u{i}") for i in range(8)]
        first = build_chunk_plan(rows, fast_config())
        second = build_chunk_plan(rows, fast_config())
        self.assertEqual(first.unique_chunks, second.unique_chunks)
        np.testing.assert_array_equal(first.flat_indices, second.flat_indices)


class CacheKeyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(system="system text here", user="user text here")]

    def _path(self, config):
        return _cache_path(config, build_chunk_plan(self.rows, config), "all")

    def test_same_settings_reuse_the_same_cache_entry(self):
        self.assertEqual(self._path(fast_config()), self._path(fast_config()))

    def test_changing_the_encoder_invalidates_the_cache(self):
        self.assertNotEqual(self._path(fast_config()), self._path(accurate_config()))

    def test_changing_chunking_invalidates_the_cache(self):
        self.assertNotEqual(self._path(fast_config()), self._path(fast_config(max_chunk_chars=480)))

    def test_changing_the_text_invalidates_the_cache(self):
        config = fast_config()
        original = self._path(config)
        self.rows = [row(system="a different system text", user="user text here")]
        self.assertNotEqual(original, self._path(config))


class LeakageTests(unittest.TestCase):
    def _split(self, name, ids, group, trajectory):
        return Split(
            name=name,
            request_ids=list(ids),
            inputs=[{"request_id": i} for i in ids],
            targets=[{"request_id": i} for i in ids],
            metadata=[{"request_id": i, "split_group_id": group, "trajectory_id": trajectory}
                      for i in ids],
        )

    def test_clean_splits_pass(self):
        splits = {
            "train": self._split("train", ["a"], "g1", "t1"),
            "validation": self._split("validation", ["b"], "g2", "t2"),
            "test": self._split("test", ["c"], "g3", "t3"),
        }
        result = check_leakage(splits, {"request_count": 3})
        self.assertEqual(result["requests"], 3)

    def test_a_prompt_group_spanning_two_splits_is_rejected(self):
        splits = {
            "train": self._split("train", ["a"], "shared", "t1"),
            "test": self._split("test", ["b"], "shared", "t2"),
        }
        with self.assertRaises(AssertionError):
            check_leakage(splits, {"request_count": 2})

    def test_a_duplicated_request_id_is_rejected(self):
        splits = {
            "train": self._split("train", ["a"], "g1", "t1"),
            "test": self._split("test", ["a"], "g2", "t2"),
        }
        with self.assertRaises(AssertionError):
            check_leakage(splits, {"request_count": 2})

    def test_a_miscounted_manifest_is_rejected(self):
        splits = {"train": self._split("train", ["a"], "g1", "t1")}
        with self.assertRaises(AssertionError):
            check_leakage(splits, {"request_count": 99})


class FeatureTests(unittest.TestCase):
    def test_static_features_are_log_compressed(self):
        matrix = static_feature_matrix([row(system_est_tokens=100000, user_est_tokens=0)])
        self.assertAlmostEqual(float(matrix[0][0]), float(np.log1p(100000)), places=3)
        self.assertEqual(float(matrix[0][1]), 0.0)

    def test_missing_features_default_to_zero(self):
        matrix = static_feature_matrix([row()])
        self.assertEqual(matrix.shape[1], 11)
        self.assertTrue(np.allclose(matrix, 0.0))

    def test_standardizer_fits_on_train_and_applies_unchanged(self):
        train = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
        scaler = Standardizer().fit(train)
        self.assertAlmostEqual(float(scaler.transform(train).mean()), 0.0, places=5)
        # Held-out rows are scaled with the *training* statistics, not their own.
        held_out = scaler.transform(np.array([[100.0]], dtype=np.float32))
        self.assertGreater(float(held_out[0][0]), 10.0)

    def test_band_rule_matches_the_dataset_thresholds(self):
        self.assertEqual(band_of(np.array([10.0]))[0], "low")
        self.assertEqual(band_of(np.array([35.0]))[0], "low")
        self.assertEqual(band_of(np.array([35.1]))[0], "medium")
        self.assertEqual(band_of(np.array([70.0]))[0], "medium")
        self.assertEqual(band_of(np.array([70.1]))[0], "high")


class HeadTests(unittest.TestCase):
    def _data(self, n=120, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 5)).astype(np.float32)
        y = (x @ np.array([3.0, -2.0, 1.0, 0.0, 0.5]))[:, None] + 50.0
        return x, np.hstack([y, y * 0.5, y * 0.2]).astype(np.float32)

    def test_ridge_recovers_a_linear_signal(self):
        x, y = self._data()
        head = RidgeHead((0.1, 1.0, 10.0)).fit(x[:80], y[:80], x[80:], y[80:])
        error = float(np.abs(head.predict(x[80:])[:, 0] - y[80:, 0]).mean())
        self.assertLess(error, 0.5)

    def test_ridge_selects_its_penalty_on_validation(self):
        x, y = self._data()
        head = RidgeHead((0.1, 1.0, 1000.0)).fit(x[:80], y[:80], x[80:], y[80:])
        self.assertIn(head.alpha, (0.1, 1.0, 1000.0))
        self.assertEqual(len(head.alpha_scores), 3)
        # the grid is scored, and the reported alpha is the argmin of it
        self.assertEqual(head.alpha, min(head.alpha_scores, key=lambda kv: kv[1])[0])

    def test_ridge_is_multi_output(self):
        x, y = self._data()
        head = RidgeHead((1.0,)).fit(x[:80], y[:80], x[80:], y[80:])
        self.assertEqual(head.predict(x[80:]).shape, (40, 3))

    def test_mean_baseline_predicts_the_training_mean(self):
        x, y = self._data()
        head = MeanBaseline().fit(x[:80], y[:80], x[80:], y[80:])
        np.testing.assert_allclose(head.predict(x[80:])[0], y[:80].mean(axis=0), rtol=1e-5)

    def test_unfitted_head_raises(self):
        with self.assertRaises(RuntimeError):
            RidgeHead().predict(np.zeros((1, 3), dtype=np.float32))


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction_scores_perfectly(self):
        y = np.array([10.0, 40.0, 80.0, 90.0])
        metrics = regression_metrics(y, y.copy())
        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["R2"], 1.0)
        self.assertEqual(metrics["band_accuracy"], 1.0)

    def test_constant_prediction_does_not_produce_nan(self):
        """The mean baseline has zero variance; correlation is defined as 0, not NaN."""
        y = np.array([10.0, 40.0, 80.0])
        metrics = regression_metrics(y, np.full(3, 40.0))
        self.assertEqual(metrics["pearson"], 0.0)
        self.assertEqual(metrics["spearman"], 0.0)
        self.assertFalse(np.isnan(metrics["R2"]))

    def test_spearman_is_rank_based_not_value_based(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = regression_metrics(y, np.array([10.0, 200.0, 3000.0, 40000.0]))
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertLess(metrics["pearson"], 1.0)

    def test_band_breakdown_covers_every_band(self):
        y = np.array([10.0, 50.0, 80.0])
        rows = band_breakdown(y, y.copy())
        self.assertEqual([r["band"] for r in rows], ["low", "medium", "high"])
        self.assertTrue(all(r["recall"] == 1.0 for r in rows))

    def test_confusion_rows_sum_to_the_band_counts(self):
        y = np.array([10.0, 50.0, 50.0, 80.0])
        table = confusion(y, np.array([10.0, 50.0, 80.0, 80.0]))
        self.assertEqual(sum(table["medium"].values()), 2)
        self.assertEqual(table["medium"]["high"], 1)


class ConfigTests(unittest.TestCase):
    def test_presets_differ_only_where_intended(self):
        self.assertEqual(fast_config().encoder, "static")
        self.assertEqual(accurate_config().encoder, "minilm")

    def test_with_returns_a_modified_copy(self):
        base = fast_config()
        modified = base.with_(head="ridge")
        self.assertEqual(base.head, "both")
        self.assertEqual(modified.head, "ridge")

    def test_unknown_encoder_is_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(encoder="nope").encoder_name

    def test_config_serialises_paths_as_strings(self):
        payload = json.dumps(fast_config().to_json())
        self.assertIn("complexity_dataset", payload)


if __name__ == "__main__":
    unittest.main()
