"""Off-policy quality estimation from validation labels + match table."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

def logged_outcome_quality(target: dict) -> float:
    """Recovered tool-call success rate for the logged-model execution.

    This is an outcome proxy, not end-to-end task success. The raw export omits
    the final call output, and tool failures may be environmental rather than
    attributable to the model.
    """
    metrics = target.get("observed_metrics") or {}
    outputs = float(metrics.get("tool_output_count", 0) or 0)
    errors = float(metrics.get("tool_error_count", 0) or 0)
    if outputs <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - errors / outputs))


def build_match_table(
    targets: list[dict],
    models_by_id: dict[str, str],
) -> dict:
    """Mean logged outcome quality by (complexity_band, model)."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    all_q: list[float] = []

    for t in targets:
        rid = t["request_id"]
        model = models_by_id.get(rid)
        if not model:
            continue
        q = logged_outcome_quality(t)
        band = t["complexity_band"]
        cells[(band, model)].append(q)
        by_model[model].append(q)
        all_q.append(q)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.5

    return {
        "cell": {f"{b}|{m}": mean(v) for (b, m), v in cells.items()},
        "by_model": {m: mean(v) for m, v in by_model.items()},
        "global": mean(all_q),
        "counts": {f"{b}|{m}": len(v) for (b, m), v in cells.items()},
    }


def estimate_routed_quality(
    routed_model: str,
    logged_model: str,
    band: str,
    logged_quality: float,
    table: dict,
    *,
    same_model_bonus: float = 0.0,
) -> tuple[float, str]:
    """Return (quality estimate, estimation method)."""
    if routed_model == logged_model:
        return logged_quality + same_model_bonus, "logged_observed"

    key = f"{band}|{routed_model}"
    cell = table["cell"].get(key)
    n = table["counts"].get(key, 0)
    if cell is not None and n >= 3:
        return cell, f"match_table_band_model(n={n})"

    model_mean = table["by_model"].get(routed_model)
    if model_mean is not None:
        return model_mean, "match_table_model"

    return table["global"], "global_mean"


def load_targets(path: Path) -> dict[str, dict]:
    return {json.loads(l)["request_id"]: json.loads(l) for l in open(path) if l.strip()}


def load_validation_targets(path: Path) -> dict[str, dict]:
    return load_targets(path)


def load_models_from_export(export_file: Path) -> dict[str, str]:
    out = {}
    with open(export_file) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            req = json.loads(line)
            out[f"{export_file.name}:{i}"] = req["model"]
    return out
