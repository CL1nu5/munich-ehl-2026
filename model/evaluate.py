"""Metrics and plots for the complexity model.

The headline number is MAE on ``complexity_score``, but a router does not consume
a point estimate — it compares a request against a cutoff. Rank correlation and
band accuracy are therefore reported next to the regression error, because a
model can lose on MAE and still order requests correctly.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .features import band_of

BANDS = ("low", "medium", "high")


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, defined as 0 when either side is constant (the mean baseline)."""
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        # average ties so repeated scores do not get an arbitrary ordering
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]

    return _safe_pearson(rank(a), rank(b))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point-estimate quality plus the two ranking measures a router cares about."""
    residual = y_pred - y_true
    total = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "n": int(len(y_true)),
        "MAE": round(float(np.abs(residual).mean()), 3),
        "RMSE": round(float(np.sqrt((residual**2).mean())), 3),
        "R2": round(float(1 - (residual**2).sum() / total) if total > 0 else 0.0, 3),
        "pearson": round(_safe_pearson(y_true, y_pred), 3),
        "spearman": round(_spearman(y_true, y_pred), 3),
        "band_accuracy": round(float((band_of(y_true) == band_of(y_pred)).mean()), 3),
    }


def band_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Per-band error, to expose whether the model only works in the dense middle."""
    true_bands = band_of(y_true)
    rows = []
    for band in BANDS:
        mask = true_bands == band
        if not mask.any():
            rows.append({"band": band, "n": 0})
            continue
        rows.append({
            "band": band,
            "n": int(mask.sum()),
            "share": round(float(mask.mean()), 3),
            "MAE": round(float(np.abs(y_pred[mask] - y_true[mask]).mean()), 3),
            "mean_true": round(float(y_true[mask].mean()), 2),
            "mean_pred": round(float(y_pred[mask].mean()), 2),
            "recall": round(float((band_of(y_pred[mask]) == band).mean()), 3),
        })
    return rows


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, int]]:
    pairs = Counter(zip(band_of(y_true).tolist(), band_of(y_pred).tolist()))
    return {t: {p: pairs.get((t, p), 0) for p in BANDS} for t in BANDS}


def format_metrics_table(rows: list[dict], columns: tuple[str, ...] = ()) -> str:
    """Render a list of flat dicts as a fixed-width table."""
    if not rows:
        return "(no rows)"
    columns = columns or tuple(rows[0])
    widths = {
        c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = "  ".join(str(c).ljust(widths[c]) for c in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def plot_results(report: dict, y_true: np.ndarray, y_pred: np.ndarray, path=None):
    """Four-panel summary: fit, residuals, split comparison, and stage timings."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0][0]
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color="#3b6ea5", edgecolor="none")
    limits = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(limits, limits, color="0.4", linewidth=1, linestyle="--", label="perfect")
    for cut in (35, 70):
        ax.axvline(cut, color="0.75", linewidth=0.8)
        ax.axhline(cut, color="0.75", linewidth=0.8)
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("true complexity score"); ax.set_ylabel("predicted")
    metrics = report["test"]["metrics"]
    ax.set_title(f"Held-out test fit — MAE {metrics['MAE']}, "
                 f"$R^2$ {metrics['R2']}, band acc {metrics['band_accuracy']:.0%}")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0][1]
    residual = y_pred - y_true
    ax.scatter(y_true, residual, s=14, alpha=0.55, color="#a5533b", edgecolor="none")
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel("true complexity score"); ax.set_ylabel("predicted − true")
    ax.set_title("Residuals (positive = over-estimated difficulty)")

    ax = axes[1][0]
    comparison = report.get("comparison", [])
    if comparison:
        labels = [row["variant"] for row in comparison]
        values = [row["validation_MAE"] for row in comparison]
        colours = ["#3b6ea5" if row["variant"] == report["selected"] else "0.72"
                   for row in comparison]
        bars = ax.barh(range(len(labels)), values, color=colours)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("validation MAE (lower is better)")
        ax.set_title("Variants — selected in blue")
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va="center", fontsize=8)

    ax = axes[1][1]
    timings = report.get("timings", {})
    if timings:
        labels = list(timings); values = [timings[k] for k in labels]
        ax.bar(range(len(labels)), values, color="#5a8f5a")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("seconds")
        ax.set_title(f"Wall clock — {sum(values):.1f}s end to end")
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
