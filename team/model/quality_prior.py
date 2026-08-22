"""Compact quality prior for routing-time veto (fit on train labels)."""
from __future__ import annotations

from collections import defaultdict

OBSERVED_KEYS = [
    "execution_depth",
    "friction_recovery",
    "produced_work",
    "tool_breadth",
]


def logged_outcome_quality(target: dict) -> float:
    comps = target.get("observed_components") or {}
    if not comps:
        return 0.5
    return sum(float(comps.get(k, 0.0)) for k in OBSERVED_KEYS) / len(OBSERVED_KEYS)


class QualityPrior:
    """band×model cell means with model/global fallback."""

    def __init__(self, min_cell_count: int = 3):
        self.min_cell_count = min_cell_count
        self.cell: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.by_model: dict[str, float] = {}
        self.global_mean: float = 0.5

    @classmethod
    def from_targets(
        cls,
        targets: list[dict],
        models_by_id: dict[str, str],
        *,
        min_cell_count: int = 3,
    ) -> "QualityPrior":
        prior = cls(min_cell_count=min_cell_count)
        cells: dict[str, list[float]] = defaultdict(list)
        by_model: dict[str, list[float]] = defaultdict(list)
        all_q: list[float] = []

        for t in targets:
            rid = t["request_id"]
            model = models_by_id.get(rid)
            if not model:
                continue
            q = logged_outcome_quality(t)
            band = t["complexity_band"]
            cells[f"{band}|{model}"].append(q)
            by_model[model].append(q)
            all_q.append(q)

        prior.cell = {k: sum(v) / len(v) for k, v in cells.items()}
        prior.counts = {k: len(v) for k, v in cells.items()}
        prior.by_model = {m: sum(v) / len(v) for m, v in by_model.items()}
        prior.global_mean = sum(all_q) / len(all_q) if all_q else 0.5
        return prior

    def estimate(self, band: str, model: str) -> float:
        key = f"{band}|{model}"
        if self.counts.get(key, 0) >= self.min_cell_count and key in self.cell:
            return self.cell[key]
        if model in self.by_model:
            return self.by_model[model]
        return self.global_mean

    def to_dict(self) -> dict:
        return {
            "min_cell_count": self.min_cell_count,
            "cell": self.cell,
            "counts": self.counts,
            "by_model": self.by_model,
            "global_mean": self.global_mean,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QualityPrior":
        prior = cls(min_cell_count=d.get("min_cell_count", 3))
        prior.cell = d["cell"]
        prior.counts = d["counts"]
        prior.by_model = d["by_model"]
        prior.global_mean = d["global_mean"]
        return prior
