"""Off-policy quality estimation: match table + kNN fallback."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.5


def _normalize_rows(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not rows:
        return [], [], []
    n = len(rows[0])
    means = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
    stds = []
    for i in range(n):
        var = sum((r[i] - means[i]) ** 2 for r in rows) / max(1, len(rows) - 1)
        stds.append(math.sqrt(var) or 1.0)
    return [[(r[i] - means[i]) / stds[i] for i in range(n)] for r in rows], means, stds


def build_match_table(
    targets: list[dict],
    models_by_id: dict[str, str],
    lookup: dict[str, dict] | None = None,
) -> dict:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    all_q: list[float] = []
    knn_raw: list[list[float]] = []

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
        if lookup and rid in lookup:
            from model.features import vectorize

            knn_raw.append(vectorize(lookup[rid]))

    knn_vecs, knn_means, knn_stds = _normalize_rows(knn_raw)

    return {
        "cell": {f"{b}|{m}": _mean(v) for (b, m), v in cells.items()},
        "by_model": {m: _mean(v) for m, v in by_model.items()},
        "global": _mean(all_q),
        "counts": {f"{b}|{m}": len(v) for (b, m), v in cells.items()},
        "knn_vecs": knn_vecs,
        "knn_meta": [
            {
                "quality": logged_outcome_quality(t),
                "model": models_by_id.get(t["request_id"]),
                "band": t["complexity_band"],
            }
            for t in targets
            if t["request_id"] in (lookup or {})
        ],
        "knn_means": knn_means,
        "knn_stds": knn_stds,
    }


def _knn_quality(table: dict, features: dict, routed_model: str, k: int = 5) -> tuple[float | None, str]:
    if not table.get("knn_vecs") or not table.get("knn_means"):
        return None, ""
    from model.features import vectorize

    q = [(vectorize(features)[i] - table["knn_means"][i]) / table["knn_stds"][i] for i in range(len(table["knn_means"]))]
    pool = [
        (i, v)
        for i, v in enumerate(table["knn_vecs"])
        if table["knn_meta"][i]["model"] == routed_model
    ]
    if len(pool) < k:
        pool = list(enumerate(table["knn_vecs"]))
    if not pool:
        return None, ""

    scored = sorted(
        (math.sqrt(sum((a - b) ** 2 for a, b in zip(q, v))), table["knn_meta"][i]["quality"]) for i, v in pool
    )
    top = scored[:k]
    return _mean([q for _, q in top]), f"knn_k={len(top)}_model={routed_model}"


def estimate_routed_quality(
    routed_model: str,
    logged_model: str,
    band: str,
    logged_quality: float,
    table: dict,
    features: dict | None = None,
) -> tuple[float, str]:
    if routed_model == logged_model:
        return logged_quality, "logged_observed"

    if features is not None:
        knn_q, knn_method = _knn_quality(table, features, routed_model)
        if knn_q is not None and knn_method:
            return knn_q, knn_method

    key = f"{band}|{routed_model}"
    n = table["counts"].get(key, 0)
    cell = table["cell"].get(key)
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
            if line.strip():
                out[f"{export_file.name}:{i}"] = json.loads(line)["model"]
    return out
