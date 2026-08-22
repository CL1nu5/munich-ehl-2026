import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_catalog import (  # noqa: E402
    BENCHMARKS,
    _catalog_cards,
    dominance_pairs,
    ModelCard,
    Observation,
    build_catalog,
    fit_capability,
    naive_composite,
    quantile,
    recommended_tie_band,
    route_score,
    route_trajectory,
)

# Synthetic panel generated from a KNOWN additive truth, so the fit can be checked by
# recovery rather than by eyeballing. Coverage is deliberately ragged: `twin` has the
# same true ability as `broad` but is measured only on the easiest benchmark, which is
# exactly the shape that fools a coverage-weighted composite.
TRUE_ABILITY = {"broad": 1.5, "twin": 1.5, "hardspec": 1.6, "weak": 0.5}
TRUE_DIFFICULTY = {
    "swe_bench_verified": 1.0,     # everyone scores well -> "easy"
    "terminal_bench_2_1": 0.3,
    "agents_last_exam": -1.2,      # everyone scores poorly -> "hard"
}
ALL_THREE = list(TRUE_DIFFICULTY)


def synthetic(model_id, benchmarks, price):
    obs = [
        Observation(
            b,
            100.0 / (1.0 + math.exp(-(TRUE_ABILITY[model_id] + TRUE_DIFFICULTY[b]))),
            "synthetic",
        )
        for b in benchmarks
    ]
    return ModelCard(model_id, "f", "1", "t", price, price / 10, price * 5,
                     observations=obs)


BROAD = synthetic("broad", ALL_THREE, 5.0)
TWIN = synthetic("twin", ["swe_bench_verified"], 1.0)
HARDSPEC = synthetic("hardspec", ["terminal_bench_2_1", "agents_last_exam"], 8.0)
WEAK = synthetic("weak", ALL_THREE, 0.5)
PANEL = [BROAD, TWIN, HARDSPEC, WEAK]
SCORES = [float(x) for x in range(1, 101)]


class QuantileTests(unittest.TestCase):
    def test_interpolates_between_neighbours(self):
        self.assertAlmostEqual(quantile([0.0, 10.0], 0.5), 5.0)

    def test_clamps_out_of_range_probabilities(self):
        self.assertEqual(quantile([1.0, 2.0, 3.0], -1.0), 1.0)
        self.assertEqual(quantile([1.0, 2.0, 3.0], 2.0), 3.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            quantile([], 0.5)


class CapabilityFitTests(unittest.TestCase):
    def test_fit_recovers_the_true_ability_gaps(self):
        """Only ability DIFFERENCES are identifiable, so compare gaps, not levels."""
        fitted = fit_capability(PANEL)["models"]
        for a, b in [("hardspec", "weak"), ("broad", "weak"), ("hardspec", "broad")]:
            self.assertAlmostEqual(
                fitted[a]["ability"] - fitted[b]["ability"],
                TRUE_ABILITY[a] - TRUE_ABILITY[b],
                places=4, msg=f"ability gap {a}-{b} not recovered")

    def test_fit_recovers_the_true_benchmark_difficulty_gaps(self):
        difficulty = fit_capability(PANEL)["difficulty"]
        self.assertAlmostEqual(
            difficulty["swe_bench_verified"] - difficulty["agents_last_exam"],
            TRUE_DIFFICULTY["swe_bench_verified"] - TRUE_DIFFICULTY["agents_last_exam"],
            places=4)

    def test_additive_fit_corrects_the_naive_coverage_bias(self):
        """The whole reason the additive fit exists.

        `twin` has exactly `broad`'s ability but is measured only on the easy
        benchmark. The naive composite scores it higher for that reason alone; the
        additive fit knows the benchmark is easy for everyone and calls them equal.
        """
        naive = naive_composite(PANEL)
        self.assertGreater(naive["twin"], naive["broad"],
                           "fixture no longer exhibits the bias it exists to show")

        fitted = fit_capability(PANEL)["models"]
        self.assertAlmostEqual(fitted["twin"]["expected_score"],
                               fitted["broad"]["expected_score"], places=4)

    def test_saturated_benchmark_is_scored_as_the_easiest(self):
        difficulty = fit_capability(PANEL)["difficulty"]
        self.assertEqual(max(difficulty, key=difficulty.get), "swe_bench_verified")
        self.assertEqual(min(difficulty, key=difficulty.get), "agents_last_exam")

    def test_coverage_reflects_benchmark_weight_not_count(self):
        fitted = fit_capability(PANEL)["models"]
        expected = BENCHMARKS["swe_bench_verified"]["weight"] / sum(
            b["weight"] for b in BENCHMARKS.values())
        self.assertAlmostEqual(fitted["twin"]["coverage"], expected)
        self.assertEqual(fitted["twin"]["n_benchmarks"], 1)

    def test_model_without_evidence_is_imputed_below_its_donor(self):
        ghost = ModelCard("ghost", "f", "0", "t", 5.0, 0.5, 25.0,
                          imputed_from="broad", imputation_penalty=0.04)
        fitted = fit_capability(PANEL + [ghost])["models"]
        self.assertTrue(fitted["ghost"]["imputed"])
        self.assertEqual(fitted["ghost"]["coverage"], 0.0)
        self.assertLess(fitted["ghost"]["expected_score"],
                        fitted["broad"]["expected_score"])

    def test_fit_is_invariant_to_model_ordering(self):
        a = fit_capability(PANEL)["models"]
        b = fit_capability(list(reversed(PANEL)))["models"]
        for mid in a:
            self.assertAlmostEqual(a[mid]["expected_score"], b[mid]["expected_score"],
                                   places=6)


class DominanceTests(unittest.TestCase):
    """Direct head-to-head evidence must outrank indirect inference.

    Regression guard for a real bug: an earlier fit ranked gpt-5.6-terra above
    gpt-5.6-sol even though Sol beat Terra on all three benchmarks they share. Sol is
    unusually weak on SWE-bench Pro, a benchmark Terra was never measured on, and that
    leaked in transitively through models Terra never met.
    """

    def test_shipped_catalog_has_no_dominance_violations(self):
        cards = _catalog_cards()
        fitted = fit_capability(cards)["models"]
        offenders = [
            (winner, loser)
            for winner, loser, _ in dominance_pairs(cards)
            if fitted[winner]["ability"] < fitted[loser]["ability"] - 1e-12
        ]
        self.assertEqual(offenders, [], f"head-to-head winners ranked below losers: {offenders}")

    def test_sol_beats_terra_head_to_head_and_is_never_ranked_below_it(self):
        cards = {c.model_id: c for c in _catalog_cards()}
        sol, terra = cards["gpt-5.6-sol"], cards["gpt-5.6-terra"]
        shared = [n for n in BENCHMARKS
                  if sol.score(n) is not None and terra.score(n) is not None]
        self.assertGreaterEqual(len(shared), 3)
        for name in shared:
            self.assertGreater(sol.score(name), terra.score(name),
                               f"fixture changed: Sol no longer wins {name}")

        fitted = fit_capability(_catalog_cards())["models"]
        self.assertGreaterEqual(fitted["gpt-5.6-sol"]["ability"],
                                fitted["gpt-5.6-terra"]["ability"] - 1e-12)

    def test_repair_is_recorded_when_it_fires(self):
        fit = fit_capability(_catalog_cards())
        self.assertTrue(fit["dominance_repaired"],
                        "the shipped panel is expected to need one repair")
        pooled = {m for r in fit["dominance_repaired"] for m in r["pooled_members"]}
        self.assertIn("gpt-5.6-sol", pooled)
        self.assertIn("gpt-5.6-terra", pooled)

    def test_an_extra_weak_benchmark_cannot_sink_a_head_to_head_winner(self):
        """Synthetic version of the Sol/Terra bug, built from scratch."""
        winner = ModelCard("winner", "f", "1", "t", 5.0, 0.5, 25.0, observations=[
            Observation("terminal_bench_2_1", 88.0, "t"),
            Observation("agents_last_exam", 54.0, "t"),
            Observation("swe_bench_pro", 60.0, "t"),   # its one weak spot
        ])
        loser = ModelCard("loser", "f", "1", "t", 2.0, 0.2, 12.0, observations=[
            Observation("terminal_bench_2_1", 87.0, "t"),   # loses head-to-head
            Observation("agents_last_exam", 50.0, "t"),     # loses head-to-head
        ])                                                  # never measured on swe_bench_pro
        anchor_a = ModelCard("anchor_a", "f", "1", "t", 1.0, 0.1, 5.0, observations=[
            Observation("terminal_bench_2_1", 70.0, "t"),
            Observation("swe_bench_pro", 85.0, "t"),
        ])
        anchor_b = ModelCard("anchor_b", "f", "1", "t", 1.0, 0.1, 5.0, observations=[
            Observation("terminal_bench_2_1", 60.0, "t"),
            Observation("swe_bench_pro", 75.0, "t"),
        ])
        panel = [winner, loser, anchor_a, anchor_b]

        self.assertIn(("winner", "loser"),
                      [(w, l) for w, l, _ in dominance_pairs(panel)])
        fitted = fit_capability(panel)["models"]
        self.assertGreaterEqual(fitted["winner"]["ability"],
                                fitted["loser"]["ability"] - 1e-12)

    def test_min_shared_guards_against_single_benchmark_dominance(self):
        one = ModelCard("one", "f", "1", "t", 1.0, 0.1, 5.0,
                        observations=[Observation("terminal_bench_2_1", 90.0, "t")])
        two = ModelCard("two", "f", "1", "t", 1.0, 0.1, 5.0,
                        observations=[Observation("terminal_bench_2_1", 80.0, "t")])
        self.assertEqual(list(dominance_pairs([one, two], min_shared=2)), [])
        self.assertEqual([(w, l) for w, l, _ in dominance_pairs([one, two], min_shared=1)],
                         [("one", "two")])


class CutoffTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog(SCORES, cards=PANEL)

    def test_pre_margin_quantile_is_monotone_in_fitted_capability(self):
        """The cutoff itself is also pushed down by the evidence margin, so the
        monotonicity invariant lives on the pre-margin quantile."""
        ordered = sorted(self.catalog["models"],
                         key=lambda m: m["expected_score"], reverse=True)
        quantiles = [m["quantile_raw"] for m in ordered]
        self.assertEqual(quantiles, sorted(quantiles, reverse=True))

    def test_cutoffs_stay_inside_the_observed_score_range(self):
        for model in self.catalog["models"]:
            self.assertGreaterEqual(model["complexity_cutoff"], min(SCORES))
            self.assertLessEqual(model["complexity_cutoff"], max(SCORES))

    def test_risk_aversion_only_lowers_cutoffs(self):
        cautious = build_catalog(SCORES, risk_aversion=0.30,
                                 cards=PANEL)
        trusting = build_catalog(SCORES, risk_aversion=0.0,
                                 cards=PANEL)
        for lo, hi in zip(cautious["models"], trusting["models"]):
            self.assertEqual(lo["model_id"], hi["model_id"])
            self.assertLessEqual(lo["complexity_cutoff"], hi["complexity_cutoff"])

    def test_thin_evidence_earns_a_wider_safety_margin(self):
        by_id = {m["model_id"]: m for m in self.catalog["models"]}
        self.assertGreater(by_id["twin"]["safety_margin_quantile"],
                           by_id["broad"]["safety_margin_quantile"])

    def test_calibration_records_the_split_it_used(self):
        self.assertEqual(self.catalog["calibrated_on"]["split"], "train")
        self.assertEqual(self.catalog["calibrated_on"]["n_requests"], len(SCORES))

    def test_empty_training_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            build_catalog([], cards=PANEL)


class TieBandTests(unittest.TestCase):
    def test_band_is_positive_when_the_fit_has_residuals(self):
        self.assertGreater(recommended_tie_band(fit_capability(PANEL)),
                           0.0)

    def test_snapping_equalises_cutoffs_within_a_tier(self):
        snapped = build_catalog(SCORES, tie_band=100.0, snap_to_band=True,
                                cards=PANEL)
        tiers = {m["tier_group"] for m in snapped["models"]}
        self.assertEqual(len(tiers), 1, "a huge band should collapse every model")
        cutoffs = {m["complexity_cutoff"] for m in snapped["models"]}
        self.assertEqual(len(cutoffs), 1)

    def test_zero_band_separates_every_distinguishable_model(self):
        """A zero band still groups exact ties -- `broad` and `twin` are equal by
        construction -- but must not merge models with different scores."""
        split = build_catalog(SCORES, tie_band=0.0, cards=PANEL)
        groups = {}
        for m in split["models"]:
            groups.setdefault(m["tier_group"], set()).add(m["expected_score"])
        for scores in groups.values():
            self.assertEqual(len(scores), 1, "band 0 merged distinguishable models")
        self.assertEqual(len(groups), len({m["expected_score"]
                                           for m in split["models"]}))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_catalog(SCORES, cards=PANEL)
        self.by_id = {m["model_id"]: m for m in self.catalog["models"]}

    def test_easy_work_goes_to_the_weakest_sufficient_model(self):
        model, reason = route_score(min(SCORES), self.catalog)
        self.assertEqual(reason, "covered")
        weakest = min(self.catalog["models"], key=lambda m: m["expected_score"])
        self.assertEqual(model, weakest["model_id"])

    def test_selection_ignores_price(self):
        """Selection is capability-only, so making the strongest model also the
        cheapest must not attract easy work to it."""
        strong_and_cheap = ModelCard("strong_cheap", "f", "1", "t", 0.01, 0.001, 0.05,
                                     observations=[
            Observation("terminal_bench_2_1", 95.0, "t"),
            Observation("swe_bench_pro", 90.0, "t"),
        ])
        weak_and_dear = ModelCard("weak_dear", "f", "1", "t", 99.0, 9.9, 495.0,
                                  observations=[
            Observation("terminal_bench_2_1", 55.0, "t"),
            Observation("swe_bench_pro", 40.0, "t"),
        ])
        catalog = build_catalog(SCORES, cards=[strong_and_cheap, weak_and_dear])
        model, reason = route_score(min(SCORES), catalog)
        self.assertEqual(reason, "covered")
        self.assertEqual(model, "weak_dear",
                         "router chose the strongest model for the easiest request; "
                         "price appears to be leaking back into selection")

    def test_ranking_is_unchanged_by_price(self):
        """Rewriting every price must not move a single fitted score or cutoff."""
        import copy

        repriced = []
        for card in copy.deepcopy(PANEL):
            card.price_in, card.price_cached, card.price_out = 42.0, 4.2, 210.0
            repriced.append(card)
        rebuilt = build_catalog(SCORES, cards=repriced)

        baseline = {m["model_id"]: (m["expected_score"], m["complexity_cutoff"])
                    for m in self.catalog["models"]}
        for entry in rebuilt["models"]:
            self.assertEqual(baseline[entry["model_id"]],
                             (entry["expected_score"], entry["complexity_cutoff"]))
        self.assertEqual([m["model_id"] for m in rebuilt["models"]],
                         [m["model_id"] for m in self.catalog["models"]],
                         "catalog ordering changed when only prices changed")

    def test_work_above_every_cutoff_falls_back_to_the_frontier(self):
        model, reason = route_score(max(SCORES) + 10, self.catalog)
        self.assertEqual(reason, "fallback_frontier")
        self.assertEqual(model, self.catalog["frontier_model"])

    def test_a_model_is_never_chosen_above_its_own_cutoff(self):
        for score in SCORES:
            model, reason = route_score(score, self.catalog)
            if reason == "covered":
                self.assertLessEqual(score, self.by_id[model]["complexity_cutoff"])

    def test_restricting_candidates_is_respected(self):
        model, _ = route_score(min(SCORES), self.catalog, allowed=["broad"])
        self.assertEqual(model, "broad")

    def test_an_unevidenced_model_is_not_preferred_over_its_tier_peer(self):
        """Within a detectability band the ability ordering is noise, so the router must
        not hand work to a model with no public evidence just because its imputed score
        is nominally lower."""
        evidenced = ModelCard("evidenced", "f", "1", "t", 5.0, 0.5, 25.0, observations=[
            Observation("terminal_bench_2_1", 75.0, "t"),
            Observation("swe_bench_pro", 69.0, "t"),
        ])
        ghost = ModelCard("ghost", "f", "0", "t", 5.0, 0.5, 25.0,
                          imputed_from="evidenced", imputation_penalty=0.04)
        floor = ModelCard("floor", "f", "1", "t", 1.0, 0.1, 5.0, observations=[
            Observation("terminal_bench_2_1", 40.0, "t"),
            Observation("swe_bench_pro", 30.0, "t"),
        ])
        # Pin the band so the fixture does not depend on the residual-derived default.
        catalog = build_catalog(SCORES, cards=[evidenced, ghost, floor], tie_band=5.0)
        by_id = {m["model_id"]: m for m in catalog["models"]}
        self.assertEqual(by_id["ghost"]["tier_group"], by_id["evidenced"]["tier_group"],
                         "fixture expects the imputed model to share a tier")
        self.assertLess(by_id["ghost"]["expected_score"],
                        by_id["evidenced"]["expected_score"])

        chosen, _ = route_score(by_id["ghost"]["complexity_cutoff"], catalog)
        self.assertEqual(chosen, "evidenced",
                         "router preferred the model with zero public evidence")

    def test_selection_is_deterministic_across_tied_models(self):
        """`broad` and `twin` tie on ability by construction; the winner must not
        depend on dict or list ordering."""
        import copy

        first = route_score(min(SCORES), build_catalog(SCORES, cards=PANEL))
        second = route_score(min(SCORES),
                             build_catalog(SCORES, cards=list(reversed(copy.deepcopy(PANEL)))))
        self.assertEqual(first, second)

    def test_trajectory_is_routed_by_its_hardest_call(self):
        calls = [min(SCORES), min(SCORES), max(SCORES)]
        self.assertEqual(route_trajectory(calls, self.catalog),
                         route_score(max(SCORES), self.catalog))

    def test_empty_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            route_trajectory([], self.catalog)


if __name__ == "__main__":
    unittest.main()
