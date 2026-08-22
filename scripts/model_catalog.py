#!/usr/bin/env python3
"""Model catalog: public-benchmark evidence -> capability index -> quality-score cutoffs.

WHY THIS EXISTS
---------------
`build_complexity_dataset.py` produces a 0-100 weak-supervision complexity score per
request. A router needs the other half of the mapping: for each candidate model, the
complexity score above which we no longer trust it. This module derives that cutoff
from published benchmark results instead of hand-picked constants, so every threshold
is traceable to a citation.

THE ANONYMISATION CLAIM DOES NOT HOLD
-------------------------------------
AGENTS.md says the `model` ids are anonymised. They are not. Every id in the export is
a real, publicly documented model, and `scripts/pricing.json` reproduces the public
list price of each one to the cent:

    claude-opus-5    5.00 / 0.50 / 25.00   matches Anthropic's published Opus 5 price
    claude-fable-5  10.00 / 1.00 / 50.00   matches published Fable 5 price
    claude-sonnet-5  2.00 / 0.20 / 10.00   matches the Sonnet 5 introductory price
                                           ($2/$10, in effect through 2026-08-31)
    gpt-5.6-sol      5.00 / 0.50 / 30.00   matches OpenAI's Sol price
    gpt-5.6-terra    2.00 / 0.20 / 12.00   matches Terra (after the 2026-07-30 cut)
    gpt-5.6-luna     0.20 / 0.02 /  1.20   matches Luna (after the 2026-07-30 cut)

`sol`/`terra`/`luna` read like pseudonyms but are OpenAI's actual tier names for the
GPT-5.6 family (Sol flagship > Terra mid > Luna cheap). AGENTS.md says to trust a
posted price sheet over the briefing when the two conflict; the price sheet, the model
ids and the public record all agree, so public benchmark data applies directly.

Everything here is stdlib-only and offline: the benchmark numbers are a frozen,
hand-transcribed snapshot with a per-number source URL. Nothing is fetched at runtime.

WHAT IS ASSUMED, AND WHERE IT CAN FAIL, is documented in LIMITATIONS at the bottom of
this file and surfaced in the notebook. Read it before quoting any number from here.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

SNAPSHOT_DATE = "2026-08-22"

# --------------------------------------------------------------------------------------
# Benchmark panel
# --------------------------------------------------------------------------------------
# Weights encode how well each benchmark proxies THIS workload: long-horizon agentic
# tool use (function_call / apply_patch loops against a real toolset), not chat, not
# closed-book knowledge. A benchmark is in the panel only if it is (a) agentic and
# (b) reported for models in both families, so the composite cannot become a
# within-family scale that silently mis-ranks across families.

BENCHMARKS: dict[str, dict] = {
    "terminal_bench_2_1": {
        "weight": 0.40,
        "label": "Terminal-Bench 2.1",
        "rationale": (
            "Multi-step agentic work in a real terminal: the closest public proxy to a "
            "Viktor trajectory. Widest cross-family coverage in the panel."
        ),
    },
    "aa_coding_agent_index": {
        "weight": 0.15,
        "label": "AA Coding Agent Index",
        "rationale": (
            "Independent third-party composite over agentic coding tasks; spans both "
            "families, so it anchors the cross-family comparison."
        ),
    },
    "swe_bench_pro": {
        "weight": 0.22,
        "label": "SWE-bench Pro",
        "rationale": (
            "Harder, non-saturated variant of SWE-bench. Discriminates at the top of "
            "the range where SWE-bench Verified no longer does."
        ),
    },
    "swe_bench_verified": {
        "weight": 0.13,
        "label": "SWE-bench Verified",
        "rationale": (
            "Most widely reported coding benchmark, but saturating (88-97 across this "
            "catalog), so it compresses real capability gaps. Deliberately down-weighted."
        ),
    },
    "agents_last_exam": {
        "weight": 0.10,
        "label": "Agents' Last Exam",
        "rationale": "Long-horizon agentic reasoning; reported for both families.",
    },
}

# Benchmarks deliberately EXCLUDED from the composite, recorded so the choice is auditable.
EXCLUDED_BENCHMARKS = {
    "osworld": (
        "Reported as OSWorld-Verified for the Claude models and OSWorld 2.0 for GPT-5.6. "
        "The two variants are not comparable (Opus 4.8 scores 83.4 on Verified, Sol 62.6 "
        "on 2.0); including them would manufacture a cross-family gap that is a benchmark "
        "version difference, not a capability difference."
    ),
    "gpqa_diamond / aime": (
        "Closed-book reasoning, not agentic tool use, and not published for the GPT-5.6 "
        "tiers. Poor proxy for this workload."
    ),
    "humanitys_last_exam": (
        "Knowledge breadth rather than long-horizon execution; Claude-only coverage here."
    ),
}

SOURCES = {
    "vals_swebench": "https://www.vals.ai/benchmarks/swebench",
    "vellum_sonnet5": "https://www.vellum.ai/blog/claude-sonnet-5-benchmarks-explained",
    "vellum_gpt56": "https://www.vellum.ai/blog/gpt-5-6-sol-terra-luna-explained",
    "morph_claude": "https://www.morphllm.com/claude-benchmarks",
    "anthropic_opus5": "https://www.anthropic.com/news/claude-opus-5",
    "openai_gpt56": "https://openai.com/index/gpt-5-6/",
    "codingfleet_tb": "https://codingfleet.com/blog/terminal-bench-leaderboard-2026/",
}


@dataclass
class Observation:
    """One published benchmark number, with provenance and reported disagreement.

    `low`/`high` capture genuine cross-source or cross-harness disagreement. They are
    not error bars in a statistical sense -- they are the range the public record
    actually reports, which is the honest thing to propagate.
    """

    benchmark: str
    value: float
    source: str
    note: str = ""
    low: float | None = None
    high: float | None = None

    @property
    def disagreement(self) -> float:
        if self.low is None or self.high is None:
            return 0.0
        return self.high - self.low


@dataclass
class ModelCard:
    model_id: str
    family: str
    generation: str
    tier: str
    price_in: float          # USD / 1M uncached input tokens
    price_cached: float      # USD / 1M cached input tokens
    price_out: float         # USD / 1M output tokens
    observations: list[Observation] = field(default_factory=list)
    imputed_from: str | None = None
    imputation_penalty: float = 0.0
    notes: str = ""

    def score(self, benchmark: str) -> float | None:
        for obs in self.observations:
            if obs.benchmark == benchmark:
                return obs.value
        return None


# --------------------------------------------------------------------------------------
# The evidence, transcribed 2026-08-22. Every number carries its source.
# --------------------------------------------------------------------------------------

def _catalog_cards() -> list[ModelCard]:
    return [
        ModelCard(
            model_id="claude-fable-5", family="claude", generation="5", tier="frontier",
            price_in=10.0, price_cached=1.0, price_out=50.0,
            observations=[
                Observation("terminal_bench_2_1", 86.0, SOURCES["vellum_gpt56"],
                            "Sources disagree: 86.0 here, 88.0 vendor-reported, 80.5 on a "
                            "third harness. Midpoint of the reported range used.",
                            low=80.5, high=88.0),
                Observation("swe_bench_pro", 80.0, SOURCES["vellum_gpt56"],
                            "Highest SWE-bench Pro result in the catalog."),
                Observation("swe_bench_verified", 95.0, SOURCES["morph_claude"]),
                Observation("aa_coding_agent_index", 77.2, SOURCES["vellum_gpt56"]),
                Observation("agents_last_exam", 40.5, SOURCES["vellum_gpt56"],
                            "Well below the GPT-5.6 tiers on this one benchmark -- the "
                            "clearest disagreement in the panel."),
            ],
            notes="Most expensive model in the export and the top SWE-bench Pro scorer, "
                  "but not uniformly the strongest: it trails Sol/Terra/Luna on Agents' "
                  "Last Exam and trails Opus 5 on SWE-bench Verified.",
        ),
        ModelCard(
            model_id="claude-opus-5", family="claude", generation="5", tier="flagship",
            price_in=5.0, price_cached=0.5, price_out=25.0,
            observations=[
                Observation("terminal_bench_2_1", 84.6, SOURCES["codingfleet_tb"],
                            "Reported 84.6 with a refusal-fallback allowance, 81.3 if "
                            "those passes count as failures, 89.1 at max effort. The "
                            "widest single-model spread in the panel.",
                            low=81.3, high=89.1),
                Observation("swe_bench_verified", 97.0, SOURCES["vals_swebench"],
                            "Independent leaderboard, updated 2026-08-19; other sources "
                            "report 96.",
                            low=96.0, high=97.0),
            ],
            notes="Vendor positions it as a step change over Opus 4.8 at the same price. "
                  "No public SWE-bench Pro or AA-index number, so its evidence coverage "
                  "is thin and its cutoff carries a correspondingly larger safety margin.",
        ),
        ModelCard(
            model_id="claude-opus-4-8", family="claude", generation="4.8", tier="flagship",
            price_in=5.0, price_cached=0.5, price_out=25.0,
            observations=[
                Observation("terminal_bench_2_1", 74.6, SOURCES["vellum_sonnet5"]),
                Observation("swe_bench_pro", 69.2, SOURCES["vellum_sonnet5"]),
                Observation("swe_bench_verified", 88.6, SOURCES["vals_swebench"],
                            "Independent leaderboard, updated 2026-08-19."),
            ],
            notes="Previous-generation flagship at the same list price as Opus 5, which "
                  "makes it dominated on price-for-capability in this catalog.",
        ),
        ModelCard(
            model_id="claude-opus-4-6", family="claude", generation="4.6", tier="flagship",
            price_in=5.0, price_cached=0.5, price_out=25.0,
            observations=[],
            imputed_from="claude-opus-4-8", imputation_penalty=0.04,
            notes="NO public benchmark evidence collected for this id, and only 2 requests "
                  "in the export. Capability is imputed as one generation below Opus 4.8. "
                  "Evidence coverage is 0, so it receives the maximum safety margin.",
        ),
        ModelCard(
            model_id="claude-sonnet-5", family="claude", generation="5", tier="mid",
            price_in=2.0, price_cached=0.2, price_out=10.0,
            observations=[
                Observation("terminal_bench_2_1", 80.4, SOURCES["vellum_sonnet5"],
                            "Beats Opus 4.8 (74.6) despite costing 2.5x less."),
                Observation("swe_bench_pro", 63.2, SOURCES["vellum_sonnet5"]),
            ],
            notes="The interesting router target: mid-tier price, but out-scores the "
                  "previous-generation flagship on the agentic benchmark that best "
                  "matches this workload. Priced at the $2/$10 introductory rate, which "
                  "the price sheet uses and which expires 2026-08-31.",
        ),
        ModelCard(
            model_id="claude-sonnet-4-6", family="claude", generation="4.6", tier="mid",
            price_in=3.0, price_cached=0.3, price_out=15.0,
            observations=[
                Observation("terminal_bench_2_1", 67.0, SOURCES["vellum_sonnet5"]),
                Observation("swe_bench_pro", 58.1, SOURCES["vellum_sonnet5"]),
            ],
            notes="Weakest model in the catalog and more expensive than Sonnet 5: "
                  "strictly dominated. 1 request in the export.",
        ),
        ModelCard(
            model_id="gpt-5.6-sol", family="gpt", generation="5.6", tier="flagship",
            price_in=5.0, price_cached=0.5, price_out=30.0,
            observations=[
                Observation("terminal_bench_2_1", 88.8, SOURCES["vellum_gpt56"],
                            "Top Terminal-Bench score in the catalog."),
                Observation("swe_bench_pro", 64.6, SOURCES["vellum_gpt56"]),
                Observation("aa_coding_agent_index", 80.0, SOURCES["vellum_gpt56"]),
                Observation("agents_last_exam", 53.6, SOURCES["vellum_gpt56"]),
            ],
            notes="OpenAI does not publish SWE-bench Verified for the GPT-5.6 tiers, so "
                  "that benchmark is genuinely absent rather than imputed.",
        ),
        ModelCard(
            model_id="gpt-5.6-terra", family="gpt", generation="5.6", tier="mid",
            price_in=2.0, price_cached=0.2, price_out=12.0,
            observations=[
                Observation("terminal_bench_2_1", 87.4, SOURCES["vellum_gpt56"]),
                Observation("aa_coding_agent_index", 77.4, SOURCES["vellum_gpt56"]),
                Observation("agents_last_exam", 50.4, SOURCES["vellum_gpt56"]),
            ],
            notes="Within 1.4 points of Sol on Terminal-Bench at 40% of the input price. "
                  "The strongest cheap-substitution candidate in the catalog.",
        ),
        ModelCard(
            model_id="gpt-5.6-luna", family="gpt", generation="5.6", tier="cheap",
            price_in=0.2, price_cached=0.02, price_out=1.2,
            observations=[
                Observation("terminal_bench_2_1", 84.7, SOURCES["vellum_gpt56"]),
                Observation("aa_coding_agent_index", 74.6, SOURCES["vellum_gpt56"]),
                Observation("agents_last_exam", 50.3, SOURCES["vellum_gpt56"]),
            ],
            notes="25x cheaper than Sol on input yet within ~4 Terminal-Bench points. If "
                  "the benchmarks transfer to this workload at all, Luna is where the "
                  "savings are -- and it is also the claim most worth distrusting, since "
                  "3 of its 3 numbers come from a single source.",
        ),
    ]


# --------------------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------------------

def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an ascending sequence."""
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _logit(pct: float) -> float:
    p = min(max(pct / 100.0, 0.01), 0.99)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def naive_composite(cards: Sequence[ModelCard]) -> dict[str, float | None]:
    """Weighted mean of score_b(m) / max_m score_b(m) over the benchmarks m actually has.

    Kept only as a FOIL. It is biased by ragged coverage: a model measured on three
    benchmarks its family happens to be strong on outranks a model measured on five,
    because each model is averaged over a different panel. The notebook shows the two
    side by side -- this is the estimator the additive fit below replaces, not a
    second opinion to average with.
    """
    best: dict[str, float] = {}
    for name in BENCHMARKS:
        present = [s for s in (c.score(name) for c in cards) if s is not None]
        if present:
            best[name] = max(present)

    out: dict[str, float | None] = {}
    for card in cards:
        weighted = covered = 0.0
        for name, meta in BENCHMARKS.items():
            score = card.score(name)
            if score is None or not best.get(name):
                continue
            weighted += meta["weight"] * (score / best[name])
            covered += meta["weight"]
        out[card.model_id] = (weighted / covered) if covered else None
    return out


def dominance_pairs(cards: Sequence[ModelCard], min_shared: int = 2):
    """Yield (winner, loser, n_shared) where the winner beat the loser head-to-head.

    "Head-to-head" means: on every benchmark BOTH models were measured on, the winner
    scored at least as high, and strictly higher on at least one. Benchmarks only one of
    them has are ignored -- that is the whole point.

    `min_shared` guards against calling a single number a dominance relation.
    """
    by_id = {c.model_id: c for c in cards}
    for a in by_id.values():
        for b in by_id.values():
            if a.model_id == b.model_id:
                continue
            shared = [n for n in BENCHMARKS
                      if a.score(n) is not None and b.score(n) is not None]
            if len(shared) < min_shared:
                continue
            if (all(a.score(n) >= b.score(n) for n in shared)
                    and any(a.score(n) > b.score(n) for n in shared)):
                yield a.model_id, b.model_id, len(shared)


def fit_capability(cards: Sequence[ModelCard], *, max_iter: int = 20_000,
                   tol: float = 1e-13, min_shared: int = 2) -> dict:
    """Estimate a comparable capability per model from a ragged benchmark panel.

    Public benchmark coverage is ragged -- Opus 5 has no SWE-bench Pro result, the
    GPT-5.6 tiers have no SWE-bench Verified -- and the benchmarks also disagree
    systematically about which family is stronger. Both problems corrupt a naive
    composite, and they need separate fixes.

    Step 1, PAIRWISE COMPARISON ON SHARED SUPPORT. Every pair of models is compared only
    on the benchmarks BOTH were measured on, as a weighted mean logit difference. Those
    pairwise differences are then reconciled into one ability per model by weighted least
    squares over the comparison graph (solved by Gauss-Seidel; the graph is connected
    through Terminal-Bench). Because no comparison ever reads a benchmark only one side
    has, a model can no longer be rewarded or punished for its coverage alone. Scores are
    logit-transformed first so saturation is respected: the 2 points between 95 and 97 on
    SWE-bench Verified mean more than the 2 between 65 and 67.

    Step 2, DOMINANCE REPAIR. Least squares is still a global fit, so a model can be
    dragged below another through third parties. That is how an earlier version of this
    function ranked gpt-5.6-terra above gpt-5.6-sol even though Sol beat Terra on all
    three benchmarks they share -- Sol is unusually weak on SWE-bench Pro (-12 points
    against its own fit), and that leaked in transitively via models Terra never met.
    Direct evidence must outrank indirect inference, so abilities are projected onto the
    partial order of head-to-head dominance: any violating pair is pooled to their shared
    mean, repeated until no violation remains. Pooling equalises rather than reorders,
    which is the smallest correction that removes the false claim.

    Identifiability: only ability DIFFERENCES are determined. Benchmark difficulties are
    recovered afterwards as mean residuals and re-centred to a weighted mean of zero, so
    `expected_score` reads as "predicted score on a panel-average benchmark" and is
    comparable across models regardless of coverage.
    """
    observed = {
        name: {c.model_id: _logit(s)
               for c in cards for s in [c.score(name)] if s is not None}
        for name in BENCHMARKS
    }
    observed = {k: v for k, v in observed.items() if v}
    measured = [c for c in cards if any(c.model_id in v for v in observed.values())]
    if not measured:
        raise ValueError("no model has any benchmark observation")

    # -- Step 1a: pairwise weighted logit differences, on shared support only.
    neighbours: dict[str, list[tuple[str, float, float]]] = {c.model_id: [] for c in measured}
    for i, a in enumerate(measured):
        for b in measured[i + 1:]:
            shared = [n for n in observed
                      if a.model_id in observed[n] and b.model_id in observed[n]]
            if not shared:
                continue
            total = sum(BENCHMARKS[n]["weight"] for n in shared)
            diff = sum(BENCHMARKS[n]["weight"]
                       * (observed[n][a.model_id] - observed[n][b.model_id])
                       for n in shared) / total
            neighbours[a.model_id].append((b.model_id, total, diff))
            neighbours[b.model_id].append((a.model_id, total, -diff))

    # -- Step 1b: reconcile the pairwise differences into one ability per model.
    ability = {c.model_id: 0.0 for c in measured}
    for _ in range(max_iter):
        delta = 0.0
        for mid in ability:
            num = sum(w * (ability[other] + d) for other, w, d in neighbours[mid])
            den = sum(w for _, w, _ in neighbours[mid])
            if den:
                new = num / den
                delta = max(delta, abs(new - ability[mid]))
                ability[mid] = new
        centre = sum(ability.values()) / len(ability)
        for mid in ability:
            ability[mid] -= centre
        if delta < tol:
            break

    # -- Step 2: project onto the head-to-head dominance partial order.
    order = [(w, l) for w, l, _ in dominance_pairs(measured, min_shared=min_shared)]
    blocks = {c.model_id: {c.model_id} for c in measured}
    repaired = []
    for _ in range(len(measured) ** 2 + 1):
        breach = next(((w, l) for w, l in order
                       if ability[w] < ability[l] - 1e-12), None)
        if breach is None:
            break
        winner, loser = breach
        merged = blocks[winner] | blocks[loser]
        pooled = sum(ability[m] for m in merged) / len(merged)
        for mid in merged:
            ability[mid] = pooled
            blocks[mid] = merged
        repaired.append({"winner": winner, "loser": loser,
                         "pooled_members": sorted(merged)})

    # -- Benchmark difficulty, recovered as mean residual and re-centred.
    difficulty = {
        name: sum(row[mid] - ability[mid] for mid in row) / len(row)
        for name, row in observed.items()
    }
    wsum = sum(BENCHMARKS[n]["weight"] for n in observed)
    centre = sum(BENCHMARKS[n]["weight"] * difficulty[n] for n in observed) / wsum
    for name in difficulty:
        difficulty[name] -= centre
    for mid in ability:
        ability[mid] += centre

    total_weight = sum(b["weight"] for b in BENCHMARKS.values())
    out: dict[str, dict] = {}
    for card in cards:
        covered = sum(BENCHMARKS[n]["weight"] for n in observed
                      if card.model_id in observed[n])
        n_bench = sum(1 for n in observed if card.model_id in observed[n])
        resid = [observed[n][card.model_id] - ability[card.model_id] - difficulty[n]
                 for n in observed if card.model_id in observed[n]]
        out[card.model_id] = {
            "ability": ability.get(card.model_id),
            "expected_score": (None if card.model_id not in ability
                               else 100.0 * _sigmoid(ability[card.model_id])),
            "coverage": covered / total_weight if total_weight else 0.0,
            "n_benchmarks": n_bench,
            "residual_rms": (math.sqrt(sum(r * r for r in resid) / len(resid))
                             if resid else None),
        }

    for card in cards:
        if card.imputed_from and out[card.model_id]["ability"] is None:
            donor = out.get(card.imputed_from, {}).get("ability")
            if donor is not None:
                a = donor - card.imputation_penalty
                out[card.model_id].update(
                    ability=a, expected_score=100.0 * _sigmoid(a), imputed=True)

    return {"models": out, "difficulty": difficulty,
            "benchmarks_used": sorted(observed),
            "dominance_repaired": repaired}


def recommended_tie_band(fit: dict) -> float:
    """Smallest expected-score gap that is worth believing, in score points.

    Taken from the additive fit's own residual scale: the median per-model residual
    RMS, converted from logit units to score points at the catalog's median ability.
    Two models closer together than this are not separated by the evidence, and any
    routing decision that turns on their ordering is fitting noise.
    """
    resids = [m["residual_rms"] for m in fit["models"].values() if m.get("residual_rms")]
    abilities = [m["ability"] for m in fit["models"].values() if m.get("ability") is not None]
    if not resids or not abilities:
        return 0.0
    r, a = statistics.median(resids), statistics.median(abilities)
    return 100.0 * (_sigmoid(a + r) - _sigmoid(a - r)) / 2.0


def _assign_tiers(entries: list[dict], band: float) -> None:
    """Group models whose expected_score lies within `band` of the group leader."""
    tier, leader = 0, None
    for e in entries:
        exp = e["expected_score"]
        if exp is None:
            e["tier_group"] = None
            continue
        if leader is None or (leader - exp) > band:
            tier += 1
            leader = exp
        e["tier_group"] = tier
    for e in entries:
        peers = [o["model_id"] for o in entries
                 if o["tier_group"] == e["tier_group"] and o["model_id"] != e["model_id"]]
        e["indistinguishable_from"] = peers


def build_catalog(
    train_scores: Iterable[float],
    *,
    risk_aversion: float = 0.10,
    quantile_floor: float = 0.15,
    tie_band: float | None = None,
    snap_to_band: bool = False,
    cards: Sequence[ModelCard] | None = None,
) -> dict:
    """Derive a quality-score cutoff per model.

    The chain, one line each:

      1. ability        -- two-way additive fit over the benchmark panel (fit_capability)
      2. expected_score -- predicted score on a panel-average benchmark, 0..100
      3. spread         -- min-max rescale of expected_score ACROSS THIS CATALOG, 0..1
      4. raw quantile   -- spread mapped onto [quantile_floor, 1.0]
      5. adj. quantile  -- minus a safety margin that grows as evidence thins
      6. cutoff         -- that quantile of the OBSERVED train complexity distribution

    Step 6 is what keeps the numbers meaningful. A benchmark score says nothing about
    this dataset's 0-100 scale, so the fit is used only to ORDER and SPACE the models;
    the thresholds themselves are read off the empirical complexity distribution.

    `risk_aversion` is the knob that sweeps out a cost-quality frontier: 0.0 trusts the
    benchmarks fully, higher values demote every model toward covering only easy work.
    It is the maximum quantile any model can be demoted by.

    `train_scores` MUST be the training split only. Calibrating on validation or test
    would leak the evaluation set into the router -- the invariant the rest of this
    pipeline exists to protect.
    """
    scores = sorted(float(s) for s in train_scores)
    if not scores:
        raise ValueError("train_scores is empty; cannot calibrate cutoffs")
    cards = list(cards) if cards is not None else _catalog_cards()

    fit = fit_capability(cards)
    caps = fit["models"]
    naive = naive_composite(cards)

    expected = [c["expected_score"] for c in caps.values()
                if c["expected_score"] is not None]
    if not expected:
        raise ValueError("no model has a fitted capability; check the benchmark panel")
    lo_exp, hi_exp = min(expected), max(expected)
    span = hi_exp - lo_exp

    # Ties on fitted ability are real (see the dominance repair in fit_capability), so
    # break them deterministically: best-corroborated first, then cheapest.
    frontier_id = max(
        (c for c in cards if caps[c.model_id]["expected_score"] is not None),
        key=lambda c: (caps[c.model_id]["expected_score"],
                       caps[c.model_id]["coverage"], -c.price_in),
    ).model_id

    entries = []
    for card in cards:
        cap = caps[card.model_id]
        exp = cap["expected_score"]
        spread = 0.0 if span == 0 or exp is None else (exp - lo_exp) / span
        q_raw = quantile_floor + spread * (1.0 - quantile_floor)
        # Half the margin is a flat discount on benchmark->workload transfer; the other
        # half scales with how little evidence the model has.
        margin = risk_aversion * (0.5 + 0.5 * (1.0 - cap["coverage"]))
        q_adj = min(max(q_raw - margin, 0.0), 1.0)

        entries.append({
            "model_id": card.model_id,
            "family": card.family,
            "generation": card.generation,
            "tier": card.tier,
            "price_in_per_mtok": card.price_in,
            "price_cached_per_mtok": card.price_cached,
            "price_out_per_mtok": card.price_out,
            "ability_logit": None if cap["ability"] is None else round(cap["ability"], 4),
            "expected_score": None if exp is None else round(exp, 2),
            "naive_composite": (None if naive[card.model_id] is None
                                else round(naive[card.model_id], 4)),
            "evidence_coverage": round(cap["coverage"], 4),
            "n_benchmarks": cap["n_benchmarks"],
            "fit_residual_rms": (None if cap["residual_rms"] is None
                                 else round(cap["residual_rms"], 4)),
            "imputed": bool(cap.get("imputed")),
            "quantile_raw": round(q_raw, 4),
            "safety_margin_quantile": round(margin, 4),
            "quantile_used": round(q_adj, 4),
            "complexity_cutoff": round(quantile(scores, q_adj), 2),
            "is_frontier": card.model_id == frontier_id,
            "max_source_disagreement": round(
                max((o.disagreement for o in card.observations), default=0.0), 2),
            "notes": card.notes,
            "observations": [asdict(o) for o in card.observations],
        })

    entries.sort(key=lambda e: (-(e["expected_score"] or 0.0), e["price_in_per_mtok"]))

    band = recommended_tie_band(fit) if tie_band is None else tie_band
    _assign_tiers(entries, band)
    if snap_to_band:
        # Within a tier the ordering is noise, so every member inherits the tier's
        # highest cutoff and the router separates them on price alone.
        for tier in {e["tier_group"] for e in entries if e["tier_group"]}:
            members = [e for e in entries if e["tier_group"] == tier]
            top = max(m["complexity_cutoff"] for m in members)
            for m in members:
                m["complexity_cutoff"] = top

    return {
        "snapshot_date": SNAPSHOT_DATE,
        "risk_aversion": risk_aversion,
        "quantile_floor": quantile_floor,
        "calibrated_on": {
            "split": "train",
            "n_requests": len(scores),
            "score_min": round(scores[0], 2),
            "score_max": round(scores[-1], 2),
        },
        "frontier_model": frontier_id,
        "dominance_repaired": fit["dominance_repaired"],
        "tie_band": round(band, 3),
        "tie_band_snapped": snap_to_band,
        "benchmark_panel": dict(BENCHMARKS),
        "benchmark_difficulty_logit": {k: round(v, 4)
                                       for k, v in fit["difficulty"].items()},
        "excluded_benchmarks": EXCLUDED_BENCHMARKS,
        "models": entries,
        "limitations": LIMITATIONS,
    }


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------

def _price_key(entry: dict) -> tuple:
    # Rank by uncached input price: that is what cost_model.py actually bills, because
    # the export has no outputs and output cost is therefore excluded everywhere.
    return (entry["price_in_per_mtok"], entry["price_cached_per_mtok"],
            entry["price_out_per_mtok"], -(entry["expected_score"] or 0.0))


def route_score(complexity_score: float, catalog: dict,
                allowed: Sequence[str] | None = None) -> tuple[str, str]:
    """Cheapest catalogued model whose cutoff covers this complexity score.

    Returns (model_id, reason). When no model's cutoff reaches the score, the frontier
    model is returned with reason 'fallback_frontier' -- an explicit escape hatch rather
    than a fabricated cutoff of 100 on the top model.
    """
    models = catalog["models"]
    if allowed is not None:
        allow = set(allowed)
        models = [m for m in models if m["model_id"] in allow]
    if not models:
        raise ValueError("no candidate models available for routing")

    covering = [m for m in models if complexity_score <= m["complexity_cutoff"]]
    if covering:
        return min(covering, key=_price_key)["model_id"], "covered"
    frontier = max(models, key=lambda m: m["expected_score"] or 0.0)
    return frontier["model_id"], "fallback_frontier"


def route_trajectory(complexity_scores: Sequence[float], catalog: dict,
                     allowed: Sequence[str] | None = None) -> tuple[str, str]:
    """One model for a whole trajectory, chosen by its hardest call.

    The export's premise is one model per trajectory, and `cost_model.py` charges a
    cache reset on every mid-trajectory switch. Routing on the maximum keeps both
    properties: a single model, sized for the hardest thing it will be asked to do.
    """
    if not complexity_scores:
        raise ValueError("cannot route an empty trajectory")
    return route_score(max(complexity_scores), catalog, allowed=allowed)


LIMITATIONS = [
    "Benchmark scores are a frozen manual transcription (2026-08-22), not a live feed. "
    "Re-check them before reusing this catalog.",
    "Published benchmarks measure end-to-end task success on THEIR task distribution. "
    "Treating that as a ceiling on THIS dataset's complexity score assumes a model's "
    "failures are the hardest tasks. Models actually fail idiosyncratically, so a "
    "cutoff is a central tendency, not a guarantee.",
    "The complexity score is itself weak supervision, not measured quality. The cutoffs "
    "inherit every assumption baked into its weights.",
    "Harness and effort settings move scores more than model identity does at the top of "
    "the range: Opus 5's Terminal-Bench result spans 81.3-89.1 depending on refusal "
    "handling and effort. Any cutoff derived from a single number is over-precise.",
    "Coverage is ragged. Opus 5 has no public SWE-bench Pro or AA-index result, the "
    "GPT-5.6 tiers have no SWE-bench Verified, and claude-opus-4-6 has no evidence at "
    "all and is imputed from Opus 4.8. Thin evidence widens the safety margin but does "
    "not make the estimate sound.",
    "Several GPT-5.6 numbers trace to a single secondary source. Sources also disagree "
    "with each other (Fable 5 Terminal-Bench: 80.5 / 86.0 / 88.0), and secondary "
    "aggregators contradict each other on Sonnet 5's SWE-bench Verified badly enough "
    "that the benchmark was left out of its record rather than guessed.",
    "The catalog ranks by input price only, because the export has no outputs and the "
    "cost model excludes output tokens. A model with a high output price (Sol at $30/M) "
    "is therefore ranked more cheaply than a full accounting would rank it.",
    "claude-sonnet-5 is priced at its $2/$10 introductory rate, which expires "
    "2026-08-31. Every saving attributed to it shrinks when list price resumes.",
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", default="results/complexity_dataset/train_targets.jsonl",
                        help="train_targets.jsonl to calibrate cutoffs on (train split only)")
    parser.add_argument("--output", default="results/model_catalog.json")
    parser.add_argument("--risk-aversion", type=float, default=0.10)
    parser.add_argument("--quantile-floor", type=float, default=0.15)
    args = parser.parse_args()

    path = Path(args.targets)
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run scripts/build_complexity_dataset.py export/ first"
        )
    scores = [json.loads(line)["complexity_score"]
              for line in path.read_text().splitlines() if line.strip()]

    catalog = build_catalog(scores, risk_aversion=args.risk_aversion,
                            quantile_floor=args.quantile_floor)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2) + "\n")

    width = max(len(m["model_id"]) for m in catalog["models"])
    print(f"calibrated on {catalog['calibrated_on']['n_requests']} train requests "
          f"(risk_aversion={args.risk_aversion})")
    print(f"{'model':<{width}}  {'$in':>6}  {'exp':>6}  {'naive':>6}  {'evid':>5}  {'cutoff':>6}")
    for m in catalog["models"]:
        exp = "   n/a" if m["expected_score"] is None else f"{m['expected_score']:6.2f}"
        naive = "   n/a" if m["naive_composite"] is None else f"{m['naive_composite']:6.3f}"
        flag = " *" if m["is_frontier"] else ("  ~" if m["imputed"] else "")
        print(f"{m['model_id']:<{width}}  {m['price_in_per_mtok']:>6.2f}  {exp}  {naive}  "
              f"{m['evidence_coverage']:>5.2f}  {m['complexity_cutoff']:>6.2f}{flag}")
    print("exp = fitted expected score on a panel-average benchmark (used for cutoffs)")
    print("naive = coverage-biased composite, shown only for contrast")
    print("* frontier (routing fallback)   ~ imputed, no public evidence")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
