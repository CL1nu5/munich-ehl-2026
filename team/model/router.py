"""Model router: complexity + static features -> model id."""
from __future__ import annotations

from .complexity import ComplexityModel
from .features import FEATURE_NAMES, vectorize
from .linear import SoftmaxRouter

# Route among common production models seen in export.
ROUTER_CLASSES = [
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]

BAND_TO_PREFERRED = {
    "low": "gpt-5.6-terra",
    "medium": "claude-sonnet-5",
    "high": "claude-opus-5",
}


class RouterModel:
    def __init__(self, complexity: ComplexityModel | None = None):
        self.complexity = complexity or ComplexityModel()
        self.classifier = SoftmaxRouter(classes=ROUTER_CLASSES)
        self.extra_means: list[float] | None = None
        self.extra_stds: list[float] | None = None

    def _augment(self, features: dict, complexity_pred: dict) -> list[float]:
        base = vectorize(features)
        extra = [
            complexity_pred["complexity_score"],
            complexity_pred["intrinsic_complexity"],
            complexity_pred["observed_difficulty"],
            {"low": 0.0, "medium": 1.0, "high": 2.0}[complexity_pred["complexity_band"]],
        ]
        return base + extra

    def _normalize_extra(self, rows: list[list[float]], fit: bool = False) -> list[list[float]]:
        n = len(rows[0])
        if fit:
            self.extra_means = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
            self.extra_stds = []
            for i in range(n):
                var = sum((r[i] - self.extra_means[i]) ** 2 for r in rows) / max(1, len(rows) - 1)
                self.extra_stds.append((var ** 0.5) or 1.0)
        assert self.extra_means and self.extra_stds
        out = []
        for row in rows:
            out.append([(row[i] - self.extra_means[i]) / self.extra_stds[i] for i in range(n)])
        return out

    def fit(self, features: list[dict], models: list[str]) -> "RouterModel":
        rows, labels = [], []
        for feat, model in zip(features, models):
            if model not in ROUTER_CLASSES:
                continue
            comp = self.complexity.predict(feat)
            rows.append(self._augment(feat, comp))
            labels.append(ROUTER_CLASSES.index(model))
        rows = self._normalize_extra(rows, fit=True)
        self.classifier.fit(rows, labels)
        return self

    def route(self, features: dict, logged_model: str | None = None) -> dict:
        comp = self.complexity.predict(features)
        row = self._normalize_extra([self._augment(features, comp)])[0]
        pred = self.classifier.predict([row])[0]
        proba = self.classifier.predict_proba([row])[0]
        rule = BAND_TO_PREFERRED.get(comp["complexity_band"], pred)

        # Blend: trust classifier, fall back to band rule when uncertain.
        max_p = max(proba)
        chosen = pred if max_p >= 0.35 else rule

        return {
            **comp,
            "routed_model": chosen,
            "classifier_model": pred,
            "rule_model": rule,
            "classifier_confidence": round(max_p, 4),
            "class_probs": {c: round(p, 4) for c, p in zip(ROUTER_CLASSES, proba)},
            "logged_model": logged_model,
        }

    def to_dict(self) -> dict:
        return {
            "complexity": self.complexity.to_dict(),
            "classifier": self.classifier.to_dict(),
            "extra_means": self.extra_means,
            "extra_stds": self.extra_stds,
            "router_classes": ROUTER_CLASSES,
            "band_to_preferred": BAND_TO_PREFERRED,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RouterModel":
        m = cls(complexity=ComplexityModel.from_dict(d["complexity"]))
        m.classifier = SoftmaxRouter.from_dict(d["classifier"])
        m.extra_means = d["extra_means"]
        m.extra_stds = d["extra_stds"]
        return m
