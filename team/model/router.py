"""Model router v3: complexity + guarded cost-downshift + quality veto."""
from __future__ import annotations

from .complexity import ComplexityModel
from .features import vectorize
from .linear import SoftmaxRouter
from .quality_prior import QualityPrior

ROUTER_CLASSES = [
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
]

# high: downshift opus/fable only (see _band_target)
BAND_CHEAP: dict[str, str | None] = {
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
DEFAULT_QUALITY_VETO_DELTA = 0.025


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
    def __init__(
        self,
        complexity: ComplexityModel | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        quality_veto_delta: float = DEFAULT_QUALITY_VETO_DELTA,
        quality_prior: QualityPrior | None = None,
    ):
        self.complexity = complexity or ComplexityModel()
        self.classifier = SoftmaxRouter(classes=ROUTER_CLASSES)
        self.extra_means: list[float] | None = None
        self.extra_stds: list[float] | None = None
        self.confidence_threshold = confidence_threshold
        self.quality_veto_delta = quality_veto_delta
        self.quality_prior = quality_prior

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

    def _band_target(self, band: str, logged: str) -> str | None:
        target = BAND_CHEAP.get(band)
        if band == "high" and model_tier(logged) not in ("opus", "fable"):
            return None
        return target

    def _allowed_downshift(self, candidate: str, logged: str) -> bool:
        if cost_rank(candidate) > cost_rank(logged):
            return False
        delta = cost_rank(logged) - cost_rank(candidate)
        # v2 failure mode: opus/fable -> gpt cross-family drops hurt quality
        if logged.startswith("claude-opus") and candidate.startswith("gpt"):
            return False
        if logged.startswith("claude-fable") and candidate.startswith("gpt"):
            return delta <= 1
        return delta <= 2

    def _quality_ok(self, band: str, candidate: str, logged: str) -> bool:
        if not self.quality_prior:
            return True
        est_c = self.quality_prior.estimate(band, candidate)
        est_l = self.quality_prior.estimate(band, logged)
        return est_c >= est_l - self.quality_veto_delta

    def _accept_candidate(
        self, candidate: str, logged: str, band: str, reason: str
    ) -> tuple[str, str] | None:
        if candidate == logged:
            return logged, "keep_logged"
        if not self._allowed_downshift(candidate, logged):
            return None
        if not self._quality_ok(band, candidate, logged):
            return None
        return candidate, reason

    def _pick_model(self, band: str, pred: str, max_p: float, logged_model: str | None) -> tuple[str, str]:
        logged = logged_model or pred

        if band == "low":
            band_target = self._band_target("low", logged)
            if band_target:
                accepted = self._accept_candidate(band_target, logged, band, "downshift_low")
                if accepted:
                    return accepted
            return logged, "keep_logged"

        if max_p >= self.confidence_threshold:
            accepted = self._accept_candidate(pred, logged, band, "classifier_downshift")
            if accepted:
                return accepted

        band_target = self._band_target(band, logged)
        if band_target:
            accepted = self._accept_candidate(band_target, logged, band, f"downshift_{band}")
            if accepted:
                return accepted

        return logged, "keep_logged"

    def route(self, features: dict, logged_model: str | None = None) -> dict:
        comp = self.complexity.predict(features, routing=True)
        row = self._normalize_extra([self._augment(features, comp)])[0]
        pred = self.classifier.predict([row])[0]
        proba = self.classifier.predict_proba([row])[0]
        max_p = max(proba)
        chosen, reason = self._pick_model(comp["complexity_band"], pred, max_p, logged_model)
        logged = logged_model or pred
        band_target = self._band_target(comp["complexity_band"], logged)

        return {
            **comp,
            "predicted_band": comp["complexity_band"],
            "predicted_score": comp["complexity_score"],
            "routed_model": chosen,
            "classifier_model": pred,
            "rule_model": band_target,
            "route_reason": reason,
            "classifier_confidence": round(max_p, 4),
            "confidence_threshold": self.confidence_threshold,
            "quality_veto_delta": self.quality_veto_delta,
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
            "quality_veto_delta": self.quality_veto_delta,
            "downshift_policy": "opus/fable->gpt blocked; max rank delta 2",
            "quality_prior": self.quality_prior.to_dict() if self.quality_prior else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RouterModel":
        prior = QualityPrior.from_dict(d["quality_prior"]) if d.get("quality_prior") else None
        m = cls(
            complexity=ComplexityModel.from_dict(d["complexity"]),
            confidence_threshold=d.get("confidence_threshold", DEFAULT_CONFIDENCE),
            quality_veto_delta=d.get("quality_veto_delta", DEFAULT_QUALITY_VETO_DELTA),
            quality_prior=prior,
        )
        m.classifier = SoftmaxRouter.from_dict(d["classifier"])
        m.extra_means = d["extra_means"]
        m.extra_stds = d["extra_stds"]
        return m
