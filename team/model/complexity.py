"""Complexity score model: intrinsic + observed factors -> score + band."""
from __future__ import annotations

from .features import FEATURE_NAMES, vectorize
from .linear import RidgeRegressor

INTRINSIC_KEYS = [
    "reasoning_depth",
    "tool_action_complexity",
    "requirements_constraints",
    "domain_difficulty",
    "context_modality",
]

OBSERVED_KEYS = [
    "execution_depth",
    "friction_recovery",
    "produced_work",
    "tool_breadth",
]

OBSERVED_WEIGHT = 0.35
BAND_LOW = 40.0
BAND_HIGH = 70.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _scale100(v: float) -> float:
    return max(0.0, min(100.0, v))


class ComplexityModel:
    def __init__(self):
        self.intrinsic_reg = RidgeRegressor(alpha=2.0)
        self.observed_reg = RidgeRegressor(alpha=2.0)
        self.feature_means: list[float] | None = None
        self.feature_stds: list[float] | None = None

    def _normalize(self, x: list[list[float]], fit: bool = False) -> list[list[float]]:
        n_feat = len(x[0])
        if fit:
            self.feature_means = [sum(row[i] for row in x) / len(x) for i in range(n_feat)]
            self.feature_stds = []
            for i in range(n_feat):
                var = sum((row[i] - self.feature_means[i]) ** 2 for row in x) / max(1, len(x) - 1)
                self.feature_stds.append((var ** 0.5) or 1.0)
        assert self.feature_means and self.feature_stds
        out = []
        for row in x:
            out.append([(row[i] - self.feature_means[i]) / self.feature_stds[i] for i in range(n_feat)])
        return out

    def fit(self, features: list[dict], targets: list[dict]) -> "ComplexityModel":
        x_raw = [vectorize(f) for f in features]
        x = self._normalize(x_raw, fit=True)

        y_intrinsic = [[t["intrinsic_components"][k] * 100 for k in INTRINSIC_KEYS] for t in targets]
        y_observed = [[t["observed_difficulty"]] for t in targets]
        y_score = [[t["complexity_score"]] for t in targets]

        self.intrinsic_reg.fit(x, y_intrinsic)
        self.observed_reg.fit(x, y_observed)
        self.score_reg = RidgeRegressor(alpha=1.0).fit(x, y_score)
        return self

    def predict(self, features: dict) -> dict:
        x = self._normalize([vectorize(features)])[0]
        intrinsic_vec = self.intrinsic_reg.predict([x])[0]
        observed = self.observed_reg.predict([x])[0][0]
        score_direct = self.score_reg.predict([x])[0][0]

        intrinsic_components = {k: _clamp01(v / 100.0) for k, v in zip(INTRINSIC_KEYS, intrinsic_vec)}
        intrinsic_complexity = _scale100(sum(intrinsic_components.values()) / len(INTRINSIC_KEYS) * 100)

        observed_difficulty = _scale100(observed)
        score_blend = _scale100(
            (1 - OBSERVED_WEIGHT) * intrinsic_complexity + OBSERVED_WEIGHT * observed_difficulty
        )
        complexity_score = _scale100(0.5 * score_direct + 0.5 * score_blend)

        if complexity_score < BAND_LOW:
            band = "low"
        elif complexity_score < BAND_HIGH:
            band = "medium"
        else:
            band = "high"

        return {
            "complexity_score": round(complexity_score, 3),
            "complexity_band": band,
            "intrinsic_complexity": round(intrinsic_complexity, 3),
            "observed_difficulty": round(observed_difficulty, 3),
            "intrinsic_components": {k: round(v, 6) for k, v in intrinsic_components.items()},
            "observed_weight": OBSERVED_WEIGHT,
        }

    def to_dict(self) -> dict:
        return {
            "intrinsic_reg": self.intrinsic_reg.to_dict(),
            "observed_reg": self.observed_reg.to_dict(),
            "score_reg": self.score_reg.to_dict(),
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "observed_weight": OBSERVED_WEIGHT,
            "band_thresholds": [BAND_LOW, BAND_HIGH],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexityModel":
        m = cls()
        m.intrinsic_reg = RidgeRegressor.from_dict(d["intrinsic_reg"])
        m.observed_reg = RidgeRegressor.from_dict(d["observed_reg"])
        m.score_reg = RidgeRegressor.from_dict(d["score_reg"])
        m.feature_means = d["feature_means"]
        m.feature_stds = d["feature_stds"]
        return m
