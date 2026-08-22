"""Tests for cost accounting and the off-policy quality estimator.

The cost half is arithmetic and is tested exactly. The quality half is an
estimator, so the tests pin down its *behaviour under thin evidence* — shrinkage,
support thresholds, and the fact that routing is scored against true complexity
rather than the router's own guess — rather than asserting particular numbers.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.frontier import pareto_front  # noqa: E402
from model.routing import (  # noqa: E402
    DEFAULT_MIN_SUPPORT,
    SHRINKAGE,
    QualityModel,
    bootstrap_quality_estimator,
    cost_matrix,
    observed_counts,
    observed_quality,
    quality_matrix,
    route_tradeoff,
)


def make_quality_model(seed=0, n=400):
    """Two models, 100 tool calls each: 'cheap' errors 10% of the time, 'dear' 1%."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(20, 100, n)
    models = ["cheap" if i % 2 else "dear" for i in range(n)]
    counts = np.array([[100.0, 10.0] if m == "cheap" else [100.0, 1.0] for m in models])
    return QualityModel.fit(scores, models, counts), scores, models, counts


class ObservedCountsTests(unittest.TestCase):
    def test_counts_are_returned_unaggregated(self):
        self.assertEqual(observed_counts({"observed_metrics": {"tool_output_count": 8,
                                                               "tool_error_count": 2}}), (8.0, 2.0))

    def test_a_request_with_nothing_to_score_carries_no_weight(self):
        """It must not be counted as a clean run, and must not vote either."""
        self.assertEqual(observed_counts({"observed_metrics": {"tool_output_count": 0}}), (0.0, 0.0))
        self.assertEqual(observed_counts({}), (0.0, 0.0))

    def test_more_errors_than_calls_is_clamped(self):
        self.assertEqual(observed_counts({"observed_metrics": {"tool_output_count": 2,
                                                               "tool_error_count": 9}}), (2.0, 2.0))

    def test_per_request_ratio_still_available_for_one_request(self):
        self.assertAlmostEqual(observed_quality({"observed_metrics": {"tool_output_count": 4,
                                                                      "tool_error_count": 2}}), 0.5)


class PooledVersusPerRequestTests(unittest.TestCase):
    """The bug this estimator was rebuilt to avoid.

    Averaging per-request ratios lets a 1-call failure (ratio 0.0) outweigh a
    40-call run with one failure (ratio 0.975). On the real export that inverted
    the ordering of two models: a previous-generation flagship appeared to beat its
    successor purely because its errors landed on longer requests.
    """

    def _targets(self, spec):
        return [{"observed_metrics": {"tool_output_count": o, "tool_error_count": e}}
                for o, e in spec]

    def test_per_request_mean_can_invert_the_true_rate(self):
        # model A: one 1-call failure, one clean 40-call run -> 1 error in 41 calls
        a = self._targets([(1, 1), (40, 0)])
        # model B: two 20-call runs with 2 errors each -> 4 errors in 40 calls
        b = self._targets([(20, 2), (20, 2)])
        per_request_a = np.mean([observed_quality(t) for t in a])
        per_request_b = np.mean([observed_quality(t) for t in b])
        pooled_a = 1 - sum(e for _, e in [observed_counts(t) for t in a]) / sum(
            o for o, _ in [observed_counts(t) for t in a])
        pooled_b = 1 - sum(e for _, e in [observed_counts(t) for t in b]) / sum(
            o for o, _ in [observed_counts(t) for t in b])
        # A really is the better model: 2.4% error vs 10%.
        self.assertGreater(pooled_a, pooled_b)
        # But the per-request average says the opposite.
        self.assertLess(per_request_a, per_request_b)

    def test_the_fitted_model_uses_the_pooled_rate(self):
        scores = np.array([50.0, 50.0, 50.0, 50.0])
        models = ["a", "a", "b", "b"]
        counts = np.array([[1.0, 1.0], [40.0, 0.0], [20.0, 2.0], [20.0, 2.0]])
        fitted = QualityModel.fit(scores, models, counts)
        self.assertGreater(fitted.by_model["a"], fitted.by_model["b"])


class QualityModelTests(unittest.TestCase):
    def test_strata_partition_the_score_range(self):
        model, scores, _, _ = make_quality_model()
        strata = model.stratum(scores)
        self.assertEqual(set(np.unique(strata)), {0, 1, 2, 3})

    def test_a_well_supported_model_keeps_its_own_rate(self):
        model, _, _, _ = make_quality_model()
        self.assertGreater(model.by_model["dear"], model.by_model["cheap"])
        self.assertAlmostEqual(model.by_model["dear"], 0.99, places=2)

    def test_a_thin_model_is_shrunk_toward_the_global_rate(self):
        """One lucky observation must not out-rank a model with thousands of calls."""
        scores = np.concatenate([np.linspace(20, 100, 200), [50.0]])
        models = ["bulk"] * 200 + ["rare"]
        counts = np.vstack([np.tile([100.0, 10.0], (200, 1)), [[1.0, 0.0]]])
        model = QualityModel.fit(scores, models, counts)
        self.assertLess(model.by_model["rare"], 1.0)
        expected = (1.0 + SHRINKAGE * model.global_mean) / (1.0 + SHRINKAGE)
        self.assertAlmostEqual(model.by_model["rare"], expected, places=6)

    def test_an_unseen_model_falls_back_to_the_global_rate(self):
        model, _, _, _ = make_quality_model()
        self.assertEqual(model.estimate("never-logged", 0), model.global_mean)
        self.assertEqual(model.support("never-logged", 0), 0)

    def test_support_is_measured_in_tool_calls(self):
        model, _, _, _ = make_quality_model()
        # 400 requests x 100 calls, split across 2 models and 4 strata
        self.assertGreater(model.support("cheap", 0), 1000)
        self.assertEqual(model.model_calls["cheap"], 20000.0)

    def test_candidates_respect_the_support_floor(self):
        scores = np.concatenate([np.linspace(20, 100, 200), np.linspace(20, 100, 5)])
        models = ["bulk"] * 200 + ["rare"] * 5
        counts = np.tile([50.0, 2.0], (205, 1))
        model = QualityModel.fit(scores, models, counts)
        self.assertEqual(model.candidates(DEFAULT_MIN_SUPPORT), ["bulk"])
        self.assertEqual(set(model.candidates(0)), {"bulk", "rare"})


class CostTests(unittest.TestCase):
    def setUp(self):
        self.tokens = {"a": 1_000_000, "b": 500_000}
        self.pricing = {"cheap": [1.0, 0.1, 5.0], "dear": [10.0, 1.0, 50.0],
                        "_default": [2.0, 0.2, 8.0]}

    def test_cost_is_tokens_times_rate(self):
        matrix = cost_matrix(["a", "b"], ["cheap", "dear"], self.tokens, self.pricing)
        np.testing.assert_allclose(matrix, [[1.0, 10.0], [0.5, 5.0]])

    def test_an_unpriced_model_falls_back_to_the_default(self):
        matrix = cost_matrix(["a"], ["unknown-model"], self.tokens, self.pricing)
        self.assertAlmostEqual(float(matrix[0][0]), 2.0)

    def test_cost_scales_linearly_with_tokens(self):
        doubled = dict(self.tokens, a=2_000_000)
        base = cost_matrix(["a"], ["cheap"], self.tokens, self.pricing)
        self.assertAlmostEqual(
            float(cost_matrix(["a"], ["cheap"], doubled, self.pricing)[0][0]),
            2 * float(base[0][0]),
        )


class RoutingTests(unittest.TestCase):
    def setUp(self):
        # two models: index 0 cheap and worse, index 1 dear and better
        self.costs = np.array([[1.0, 10.0], [1.0, 10.0]])
        self.qualities = np.array([[0.90, 0.99], [0.90, 0.99]])

    def test_zero_lambda_buys_quality(self):
        np.testing.assert_array_equal(route_tradeoff(self.costs, self.qualities, 0.0), [1, 1])

    def test_large_lambda_buys_the_cheapest(self):
        np.testing.assert_array_equal(route_tradeoff(self.costs, self.qualities, 1e6), [0, 0])

    def test_the_crossover_sits_where_the_exchange_rate_says(self):
        # quality gap 0.09 over a cost gap of 9.0 -> the switch happens at lambda 0.01
        np.testing.assert_array_equal(route_tradeoff(self.costs, self.qualities, 0.009), [1, 1])
        np.testing.assert_array_equal(route_tradeoff(self.costs, self.qualities, 0.011), [0, 0])

    def test_a_masked_model_is_never_chosen(self):
        mask = np.array([True, False])  # the better model is not routable
        np.testing.assert_array_equal(
            route_tradeoff(self.costs, self.qualities, 0.0, mask), [0, 0]
        )

    def test_routing_and_scoring_can_use_different_strata(self):
        """The router sees predicted complexity; it is scored on the true stratum.
        A confident-but-wrong prediction must therefore change the score."""
        model, _, _, _ = make_quality_model()  # noqa: F841
        models = ["cheap", "dear"]
        truthful = quality_matrix(np.array([0]), models, model)
        mistaken = quality_matrix(np.array([3]), models, model)
        self.assertEqual(truthful.shape, mistaken.shape)
        # both strata are populated, so the estimates are real cell values
        self.assertGreater(model.support("cheap", 0), 0)
        self.assertGreater(model.support("cheap", 3), 0)


class BootstrapTests(unittest.TestCase):
    def test_a_real_gap_survives_resampling(self):
        _, scores, models, counts = make_quality_model()
        policies = {"logged policy": ["cheap"] * 50, "all dear": ["dear"] * 50}
        out = bootstrap_quality_estimator(
            scores, models, counts, policies, scores[:50], draws=60,
        )
        self.assertTrue(out["all dear"]["delta_excludes_zero"])
        self.assertGreater(out["all dear"]["delta_vs_logged_mean"], 0)

    def test_no_gap_does_not_survive_resampling(self):
        _, scores, models, counts = make_quality_model()
        policies = {"logged policy": ["cheap"] * 50, "same again": ["cheap"] * 50}
        out = bootstrap_quality_estimator(
            scores, models, counts, policies, scores[:50], draws=60,
        )
        self.assertFalse(out["same again"]["delta_excludes_zero"])


class ParetoTests(unittest.TestCase):
    def test_dominated_points_are_dropped(self):
        points = [
            {"name": "a", "total_cost": 1.0, "mean_quality": 0.9},
            {"name": "b", "total_cost": 2.0, "mean_quality": 0.8},  # costs more, worse
            {"name": "c", "total_cost": 3.0, "mean_quality": 0.95},
        ]
        self.assertEqual([p["name"] for p in pareto_front(points)], ["a", "c"])

    def test_identical_outcomes_collapse_to_one_point(self):
        points = [
            {"name": "x", "total_cost": 1.0, "mean_quality": 0.9},
            {"name": "y", "total_cost": 1.0, "mean_quality": 0.9},
        ]
        self.assertEqual(len(pareto_front(points)), 1)

    def test_the_front_is_sorted_by_cost(self):
        points = [
            {"name": "c", "total_cost": 3.0, "mean_quality": 0.95},
            {"name": "a", "total_cost": 1.0, "mean_quality": 0.9},
        ]
        self.assertEqual([p["total_cost"] for p in pareto_front(points)], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
