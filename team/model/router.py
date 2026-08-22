"""Model router: complexity + cost-downshift policy -> model id."""
from __future__ import annotations

from .complexity import ComplexityModel
from .features import vectorize
from .linear import SoftmaxRouter

ROUTER_CLASSES = [
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]

BAND_CHEAP = {
    "low": "gpt-5.6-terra",
    "medium": "claude-sonnet-5",
    "high": "claude-sonnet-5",
}

MODEL_COST_RANK = {
    "gpt-5.6-luna": 0,
    "gpt-5.6-terra": 1,
    "claude-sonnet-5": 2,
    "gpt-5.6-sol": 3,
    "claude-opus-5": 4,
    "claude-opus-4-8": 4,
    "claude-opus-4-6": 4,
    "claude-sonnet-4-6": 3,
    "claude-fable-5": 5,
}

DEFAULT_CONFIDENCE = 0.35


def model_tier(m: str) -> str:
    if m.startswith("claude-opus"):
        return "opus"
    if m.startswith("claude-sonnet"):
        return "sonnet"
    if m.startswith("claude-fable"):
        return "fable"
    return "gpt"


def cost_rank(m: str | None) -> int:
    if not m:
        return 99
    if m in MODEL_COST_RANK:
        return MODEL_COST_RANK[m]
    if m.startswith("gpt"):
        return 2
    if m.startswith("claude-sonnet"):
        return 2
    if m.startswith("claude-opus"):
        return 4
    if m.startswith("claude-fable"):
        return 5
    return 3


class RouterModel:
    def __init__(self, complexity: ComplexityModel | None = None, confidence_threshold: float = DEFAULT_CONFIDENCE):
        self.complexity = complexity or ComplexityModel()
        self.classifier = SoftmaxRouter(classes=ROUTER_CLASSES)
        self.extra_means: list[float] | None = None
        self.extra_stds: list[float] | None = None
        self.confidence_threshold = confidence_threshold

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
            comp = self.complexity.predict(feat, routing=True)
            rows.append(self._augment(feat, comp))
            labels.append(ROUTER_CLASSES.index(model))
        rows = self._normalize_extra(rows, fit=True)
        self.classifier.fit(rows, labels)
        return self

    def _band_target(self, band: str) -> str:
        return BAND_CHEAP.get(band, "claude-sonnet-5")

    def _pick_model(self, band: str, pred: str, max_p: float, logged_model: str | None) -> tuple[str, str]:
        logged = logged_model or pred
        band_target = self._band_target(band)
        if band == "low":
            return band_target, "downshift_low"
        if max_p >= self.confidence_threshold and cost_rank(pred) < cost_rank(logged):
            return pred, "classifier_downshift"
        if cost_rank(band_target) < cost_rank(logged):
            return band_target, f"downshift_{band}"
        return logged, "keep_logged"

    def route(self, features: dict, logged_model: str | None = None) -> dict:
        comp = self.complexity.predict(features, routing=True)
        row = self._normalize_extra([self._augment(features, comp)])[0]
        pred = self.classifier.predict([row])[0]
        proba = self.classifier.predict_proba([row])[0]
        max_p = max(proba)
        chosen, reason = self._pick_model(comp["complexity_band"], pred, max_p, logged_model)

        return {
            **comp,
            "routed_model": chosen,
            "classifier_model": pred,
            "rule_model": self._band_target(comp["complexity_band"]),
            "route_reason": reason,
            "classifier_confidence": round(max_p, 4),
            "confidence_threshold": self.confidence_threshold,
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
            "band_cheap": BAND_CHEAP,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RouterModel":
        m = cls(
            complexity=ComplexityModel.from_dict(d["complexity"]),
            confidence_threshold=d.get("confidence_threshold", DEFAULT_CONFIDENCE),
        )
        m.classifier = SoftmaxRouter.from_dict(d["classifier"])
        m.extra_means = d["extra_means"]
        m.extra_stds = d["extra_stds"]
        return m
