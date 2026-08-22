"""Routing-time explanations: feature attribution + quality/cost context."""
from __future__ import annotations

from .features import FEATURE_NAMES, vectorize
from .router import RouterModel


def _ridge_contributions(
    weights: list[list[float]] | None,
    x_norm: list[float],
    feature_names: list[str],
    *,
    top_k: int = 5,
) -> list[dict]:
    """Per-feature contribution to a single Ridge output (linear attribution)."""
    if not weights or not x_norm:
        return []
    # weights: (n_feat+1) x 1 — use first output column
    contribs = []
    for i, name in enumerate(feature_names):
        w = weights[i][0] if i < len(weights) - 1 else 0.0
        contribs.append(
            {
                "feature": name,
                "value_norm": round(x_norm[i], 4),
                "weight": round(w, 6),
                "contribution": round(x_norm[i] * w, 4),
            }
        )
    contribs.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return contribs[:top_k]


def complexity_attribution(router: RouterModel, features: dict, *, top_k: int = 5) -> dict:
    """Top features pushing complexity score (score_reg linear model)."""
    cx = router.complexity
    raw = vectorize(features)
    x_norm = cx._normalize([raw])[0]
    score_weights = cx.score_reg.weights
    top = _ridge_contributions(score_weights, x_norm, FEATURE_NAMES, top_k=top_k)
    bias = score_weights[-1][0] if score_weights else 0.0
    linear_sum = sum(r["contribution"] for r in top) + bias
    return {
        "top_features": top,
        "linear_score_hint": round(linear_sum, 3),
        "band_thresholds": [cx.band_low, cx.band_high],
    }


def quality_prior_pair(router: RouterModel, band: str, logged: str | None, routed: str) -> dict:
    prior = router.quality_prior
    if not prior or not logged:
        return {}
    return {
        "prior_logged": round(prior.estimate(band, logged), 4),
        "prior_routed": round(prior.estimate(band, routed), 4),
        "prior_delta": round(prior.estimate(band, routed) - prior.estimate(band, logged), 4),
    }


def explain_decision(router: RouterModel, features: dict, logged_model: str | None = None) -> dict:
    """Full routing explanation for one request."""
    decision = router.route(features, logged_model=logged_model)
    band = decision["predicted_band"]
    logged = decision.get("logged_model") or decision["classifier_model"]
    routed = decision["routed_model"]
    qp = quality_prior_pair(router, band, logged, routed)
    return {
        **decision,
        **qp,
        "feature_attribution": complexity_attribution(router, features),
        "transition": f"{logged} -> {routed}",
        "route_changed": routed != logged,
    }
