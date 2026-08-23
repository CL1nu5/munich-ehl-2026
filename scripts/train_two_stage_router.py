#!/usr/bin/env python3
"""Train the two-stage complexity -> model-quality router.

Stage 1 learns to predict an execution-derived complexity score from only the
opening request. Stage 2 learns a shared quality response model from predicted
complexity, task features, and benchmark-informed model capability profiles.
At routing time the policy chooses the cheapest supported model whose predicted
quality is within epsilon of the best candidate.

All models are dependency-free ridge regressions and run offline. Evaluation is
performed on a deterministic model-stratified holdout with a doubly robust
off-policy estimator.

Usage: python scripts/train_two_stage_router.py
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import statistics
import re
from collections import Counter, defaultdict
from pathlib import Path

from cost_model import load_pricing, price_of
from build_feature_table import sanitize_routing_text
from load_trajectories import est_tokens, iter_requests


CALL_TYPES = {"function_call", "custom_tool_call"}
WORKLOAD_SIGNALS = {
    "continuation_tokens": 0.50,
    "tool_calls": 0.30,
    "distinct_tools": 0.20,
}
# Backwards-compatible name used in reports and by external notebooks.
COMPLEXITY_SIGNALS = WORKLOAD_SIGNALS
DEFAULT_TEXT_FEATURES = 96
LOG_FEATURES = {"x_total_context_tokens_est", "x_task_text_tokens_est"}
OPENING_FEATURES = (
    "x_total_context_tokens_est",
    "x_task_text_tokens_est",
    "x_input_items",
    "x_user_messages",
    "x_image_parts",
    "x_available_tools",
    "x_question_marks",
    "x_instruction_markers",
    "x_is_scheduled",
    "x_is_communication",
    "x_is_coding",
    "x_is_artifact",
    "x_is_research",
    "x_is_correction",
    "x_requires_multiple_steps",
    "x_requires_external_action",
    "x_high_risk_language",
    "x_available_communication_tools",
    "x_available_file_tools",
    "x_available_code_tools",
    "x_available_research_tools",
    "x_available_media_tools",
    "x_available_write_tools",
)
CATEGORIES = ("scheduled", "coding", "artifact", "research", "communication", "general")
RISKS = ("0", "1", "2", "3")
PROFILE_DIMENSIONS = (
    "tool_use", "coding", "reasoning", "instruction_following", "long_context", "multimodal"
)
QUALITY_TASK_FEATURES = (
    "x_is_scheduled", "x_is_communication", "x_is_coding", "x_is_artifact",
    "x_is_research", "x_requires_multiple_steps", "x_requires_external_action",
    "x_high_risk_language", "x_has_image", "x_available_tools",
)


def canonical_action(model):
    if "fable" in model:
        return "fable"
    if "sonnet" in model:
        return "sonnet"
    if "opus" in model:
        return "opus"
    if model.endswith("-sol"):
        return "sol"
    if model.endswith("-terra"):
        return "terra"
    if model.endswith("-luna"):
        return "luna"
    raise ValueError(f"unrecognized model id: {model}")


def raw_opening_value(row, feature):
    value = float(row.get(feature) or 0)
    return math.log1p(value) if feature in LOG_FEATURES else value


def request_signals(export_dir):
    signals = {}
    for chunk, line, request in iter_requests(export_dir):
        calls = [
            item for item in request.get("input", [])
            if isinstance(item, dict) and item.get("type") in CALL_TYPES
        ]
        signals[(chunk, line + 1)] = {
            "full_input_tokens": est_tokens(request.get("input", [])),
            "distinct_tools": len({item.get("name", "unknown") for item in calls}),
        }
    return signals


def attach_complexity_signals(rows, raw_signals):
    enriched = []
    for source in rows:
        row = dict(source)
        raw = raw_signals[(row["source_chunk"], int(row["source_line"]))]
        row["_continuation_tokens"] = max(
            0.0, raw["full_input_tokens"] - float(row["x_input_tokens_est"])
        )
        row["_tool_calls"] = float(row["y_observed_tool_calls"] or 0)
        row["_history_items"] = float(row["y_observed_history_items"] or 0)
        row["_distinct_tools"] = float(raw["distinct_tools"])
        row["_execution_friction"] = (
            float(row["y_tool_errors"] or 0)
            + float(row["y_unrecovered_tool_errors"] or 0)
            + float(row["y_user_correction"] or 0)
            + float(row["y_duplicate_successful_side_effects"] or 0)
        )
        enriched.append(row)
    return enriched


def signal_value(row, signal):
    return row[f"_{signal}"]


def fit_complexity_calibrator(rows):
    return {
        signal: sorted(signal_value(row, signal) for row in rows)
        for signal in WORKLOAD_SIGNALS
    }


def empirical_percentile(value, reference):
    return bisect.bisect_right(reference, value) / len(reference)


def observed_complexity(row, calibrator):
    return 100 * sum(
        weight * empirical_percentile(signal_value(row, signal), calibrator[signal])
        for signal, weight in WORKLOAD_SIGNALS.items()
    )


def observed_execution_risk(row):
    """Binary post-execution friction label, modeled separately from workload."""
    return float(signal_value(row, "execution_friction") > 0)


TEXT_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "not", "you", "your",
    "our", "are", "was", "were", "have", "has", "had", "into", "then", "than",
    "can", "will", "would", "should", "could", "but", "all", "any", "its", "it's",
    "system", "entity", "number", "date", "url", "path", "pii", "id",
}


def model_text(row):
    text = row.get("x_task_text_sanitized") or row.get("task_preview", "")
    return sanitize_routing_text(text)


def text_terms(row):
    words = [
        word for word in re.findall(r"[a-z][a-z']{2,}", model_text(row))
        if word not in TEXT_STOPWORDS
    ]
    return words + [f"{words[index]}__{words[index + 1]}" for index in range(len(words) - 1)]


def raw_text_vector(row, text_encoder):
    counts = Counter(text_terms(row))
    vector = [0.0] * len(text_encoder["vocabulary"])
    index = {term: position for position, term in enumerate(text_encoder["vocabulary"])}
    for term, count in counts.items():
        if term in index:
            position = index[term]
            vector[position] = (1.0 + math.log(count)) * text_encoder["idf"][position]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def fit_text_encoder(rows, max_features):
    documents = [Counter(text_terms(row)) for row in rows]
    document_frequency = Counter(term for document in documents for term in document)
    term_frequency = Counter()
    for document in documents:
        term_frequency.update(document)
    vocabulary = [
        term for term, _ in sorted(term_frequency.items(), key=lambda item: (-item[1], item[0]))
        if document_frequency[term] >= 3
    ][:max_features]
    encoder = {
        "vocabulary": vocabulary,
        "idf": [
            math.log((1 + len(rows)) / (1 + document_frequency[term])) + 1
            for term in vocabulary
        ],
    }
    vectors = [raw_text_vector(row, encoder) for row in rows]
    encoder["means"] = [
        statistics.fmean(vector[index] for vector in vectors)
        for index in range(len(vocabulary))
    ]
    encoder["scales"] = [
        statistics.pstdev(vector[index] for vector in vectors) or 1.0
        for index in range(len(vocabulary))
    ]
    return encoder


def fit_opening_encoder(rows, text_features=DEFAULT_TEXT_FEATURES):
    means, scales = {}, {}
    for feature in OPENING_FEATURES:
        values = [raw_opening_value(row, feature) for row in rows]
        means[feature] = statistics.fmean(values)
        scales[feature] = statistics.pstdev(values) or 1.0
    return {
        "features": list(OPENING_FEATURES),
        "means": means,
        "scales": scales,
        "categories": list(CATEGORIES),
        "risks": list(RISKS),
        "text": fit_text_encoder(rows, text_features),
    }


def encode_opening(row, encoder):
    vector = [1.0]
    vector.extend(
        (raw_opening_value(row, feature) - encoder["means"][feature])
        / encoder["scales"][feature]
        for feature in encoder["features"]
    )
    vector.extend(float(row["x_task_category"] == category) for category in encoder["categories"])
    vector.extend(float(row["x_risk_level"] == risk) for risk in encoder["risks"])
    text_encoder = encoder.get("text")
    if text_encoder:
        raw = raw_text_vector(row, text_encoder)
        vector.extend(
            (value - mean) / scale
            for value, mean, scale in zip(raw, text_encoder["means"], text_encoder["scales"])
        )
    return vector


def solve_linear_system(matrix, target):
    n = len(target)
    augmented = [list(matrix[i]) + [target[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[i][-1] for i in range(n)]


def fit_ridge(vectors, outcomes, penalty):
    p = len(vectors[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for vector, outcome in zip(vectors, outcomes):
        for i, left in enumerate(vector):
            xty[i] += left * outcome
            for j in range(i, p):
                xtx[i][j] += left * vector[j]
    for i in range(p):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
        if i:
            xtx[i][i] += penalty
    return solve_linear_system(xtx, xty)


def fit_multi_ridge(vectors, outcome_sets, penalty):
    """Fit several ridge heads over one encoder while factoring X'X only once."""
    p = len(vectors[0])
    targets = [[0.0] * p for _ in outcome_sets]
    xtx = [[0.0] * p for _ in range(p)]
    for row_index, vector in enumerate(vectors):
        for i, left in enumerate(vector):
            for target_index, outcomes in enumerate(outcome_sets):
                targets[target_index][i] += left * outcomes[row_index]
            for j in range(i, p):
                xtx[i][j] += left * vector[j]
    for i in range(p):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
        if i:
            xtx[i][i] += penalty

    augmented = [list(xtx[i]) + [target[i] for target in targets] for i in range(p)]
    for column in range(p):
        pivot = max(range(column, p), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular multi-output ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(p):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [
        [augmented[row][p + target_index] for row in range(p)]
        for target_index in range(len(outcome_sets))
    ]


def dot(coefficients, vector):
    return sum(coefficient * value for coefficient, value in zip(coefficients, vector))


def fit_complexity_model(rows, targets, risk_targets, penalty, text_features=DEFAULT_TEXT_FEATURES):
    encoder = fit_opening_encoder(rows, text_features)
    vectors = [encode_opening(row, encoder) for row in rows]
    coefficients, risk_coefficients = fit_multi_ridge(
        vectors, [targets, risk_targets], penalty
    )
    return {
        "encoder": encoder,
        "coefficients": coefficients,
        "risk_coefficients": risk_coefficients,
        "penalty": penalty,
    }


def fit_isotonic(predictions, outcomes):
    """Pool-adjacent-violators calibration represented as upper bounds + values."""
    grouped = []
    for prediction, outcome in sorted(zip(predictions, outcomes)):
        if grouped and prediction == grouped[-1]["upper"]:
            block = grouped[-1]
            block["sum"] += outcome
            block["weight"] += 1
        else:
            grouped.append({"lower": prediction, "upper": prediction, "sum": outcome, "weight": 1})
    index = 0
    while index < len(grouped) - 1:
        left = grouped[index]
        right = grouped[index + 1]
        if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
            index += 1
            continue
        grouped[index:index + 2] = [{
            "lower": left["lower"], "upper": right["upper"],
            "sum": left["sum"] + right["sum"],
            "weight": left["weight"] + right["weight"],
        }]
        index = max(0, index - 1)
    return {
        "upper_bounds": [block["upper"] for block in grouped],
        "values": [block["sum"] / block["weight"] for block in grouped],
    }


def calibrated_value(value, calibration):
    if not calibration:
        return value
    index = bisect.bisect_left(calibration["upper_bounds"], value)
    index = min(index, len(calibration["values"]) - 1)
    return calibration["values"][index]


def predict_complexity(row, model):
    raw = min(100.0, max(0.0, dot(model["coefficients"], encode_opening(row, model["encoder"]))))
    return min(100.0, max(0.0, calibrated_value(raw, model.get("workload_calibration"))))


def predict_execution_risk(row, model):
    raw = min(1.0, max(0.0, dot(model["risk_coefficients"], encode_opening(row, model["encoder"]))))
    return min(1.0, max(0.0, calibrated_value(raw, model.get("risk_calibration"))))


def template_key(row):
    """Normalized opening-task template used to keep recurring jobs in one split."""
    return model_text(row)[-500:]


def deterministic_folds(rows, folds):
    groups = defaultdict(list)
    for row in rows:
        groups[template_key(row)].append(row)
    sizes = [0] * folds
    group_fold = {}
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()),
    )
    for key, group_rows in ordered:
        fold = min(range(folds), key=lambda candidate: (sizes[candidate], candidate))
        group_fold[key] = fold
        sizes[fold] += len(group_rows)
    return {row["trajectory_id"]: group_fold[template_key(row)] for row in rows}


def out_of_fold_stage1(rows, targets, risk_targets, penalty, folds, text_features):
    assignments = deterministic_folds(rows, folds)
    predictions = {"workload": {}, "risk": {}}
    for fold in range(folds):
        fit_rows = [row for row in rows if assignments[row["trajectory_id"]] != fold]
        fit_targets = [
            targets[row["trajectory_id"]] for row in fit_rows
        ]
        fit_risks = [risk_targets[row["trajectory_id"]] for row in fit_rows]
        model = fit_complexity_model(fit_rows, fit_targets, fit_risks, penalty, text_features)
        for row in rows:
            if assignments[row["trajectory_id"]] == fold:
                predictions["workload"][row["trajectory_id"]] = predict_complexity(row, model)
                predictions["risk"][row["trajectory_id"]] = predict_execution_risk(row, model)
    return predictions


def template_grouped_three_way(rows, validation_fraction, test_fraction):
    groups = defaultdict(list)
    for row in rows:
        groups[template_key(row)].append(row)
    names = ("train", "validation", "test")
    fractions = (1.0 - validation_fraction - test_fraction, validation_fraction, test_fraction)
    targets = [len(rows) * fraction for fraction in fractions]
    partitions = [[] for _ in names]
    sizes = [0] * len(names)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()),
    )
    for _, group_rows in ordered:
        destination = min(
            range(len(names)),
            key=lambda index: (sizes[index] / max(targets[index], 1), index),
        )
        partitions[destination].extend(group_rows)
        sizes[destination] += len(group_rows)
    return tuple(partitions)


def stratified_three_way(rows, validation_fraction, test_fraction):
    """Compatibility alias: splits are now grouped by sanitized task template."""
    return template_grouped_three_way(rows, validation_fraction, test_fraction)


def quality_task_features(row, complexity, execution_risk):
    values = {
        "predicted_complexity": complexity / 100.0,
        "predicted_execution_risk": execution_risk,
    }
    for feature in QUALITY_TASK_FEATURES:
        values[feature] = float(row.get(feature) or 0)
    for category in CATEGORIES:
        values[f"category_{category}"] = float(row["x_task_category"] == category)
    for risk in RISKS:
        values[f"risk_{risk}"] = float(row["x_risk_level"] == risk)
    return values


def benchmark_fit(row, complexity, execution_risk, action, profiles):
    """Task-weighted public-benchmark prior, constrained to help rather than hurt."""
    profile = profiles[action]
    context_pressure = min(1.0, float(row["x_total_context_tokens_est"]) / 50_000)
    weights = {
        "tool_use": 1.0 + 2.0 * float(row["x_requires_external_action"]),
        "coding": 1.0 + 3.0 * float(row["x_is_coding"]),
        "reasoning": 1.0 + 2.0 * complexity / 100.0 + float(row["x_is_research"]),
        "instruction_following": (
            1.0 + 2.0 * float(row["x_requires_multiple_steps"])
            + float(row["x_high_risk_language"]) + execution_risk
        ),
        "long_context": 1.0 + 2.0 * context_pressure,
        "multimodal": 0.25 + 3.0 * float(row["x_has_image"]),
    }
    total = sum(weights.values())
    return sum(weights[name] * float(profile[name]) for name in PROFILE_DIMENSIONS) / total


def fit_dict_encoder(feature_dicts):
    names = sorted(feature_dicts[0])
    means, scales = {}, {}
    for name in names:
        values = [features[name] for features in feature_dicts]
        means[name] = statistics.fmean(values)
        scales[name] = statistics.pstdev(values) or 1.0
    return {"names": names, "means": means, "scales": scales}


def encode_dict(features, encoder):
    return [1.0] + [
        (features[name] - encoder["means"][name]) / encoder["scales"][name]
        for name in encoder["names"]
    ]


def fit_quality_model(rows, stage1_predictions, profiles, pricing, penalty):
    feature_dicts = [
        quality_task_features(
            row,
            stage1_predictions["workload"][row["trajectory_id"]],
            stage1_predictions["risk"][row["trajectory_id"]],
        )
        for row in rows
    ]
    capability = [
        benchmark_fit(
            row,
            stage1_predictions["workload"][row["trajectory_id"]],
            stage1_predictions["risk"][row["trajectory_id"]],
            canonical_action(row["logged_model"]), profiles,
        )
        for row in rows
    ]
    outcomes = [float(row["evaluation_outcome_score"]) for row in rows]
    assignments = deterministic_folds(rows, min(5, len(rows)))
    candidate_slopes = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
    losses = {}
    for slope in candidate_slopes:
        squared_errors = []
        for fold in sorted(set(assignments.values())):
            fit_indices = [
                index for index, row in enumerate(rows)
                if assignments[row["trajectory_id"]] != fold
            ]
            holdout_indices = [
                index for index, row in enumerate(rows)
                if assignments[row["trajectory_id"]] == fold
            ]
            encoder = fit_dict_encoder([feature_dicts[index] for index in fit_indices])
            coefficients = fit_ridge(
                [encode_dict(feature_dicts[index], encoder) for index in fit_indices],
                [
                    outcomes[index] - slope * (capability[index] - 0.80)
                    for index in fit_indices
                ],
                penalty,
            )
            for index in holdout_indices:
                prediction = dot(coefficients, encode_dict(feature_dicts[index], encoder))
                prediction += slope * (capability[index] - 0.80)
                prediction = min(1.0, max(0.0, prediction))
                squared_errors.append((prediction - outcomes[index]) ** 2)
        losses[slope] = statistics.fmean(squared_errors)
    # Capability is constrained to have a non-negative effect. Prefer the
    # smaller slope when cross-validation cannot distinguish two candidates.
    selected_slope = min(candidate_slopes, key=lambda slope: (losses[slope], slope))
    encoder = fit_dict_encoder(feature_dicts)
    coefficients = fit_ridge(
        [encode_dict(features, encoder) for features in feature_dicts],
        [
            outcome - selected_slope * (fit - 0.80)
            for outcome, fit in zip(outcomes, capability)
        ],
        penalty,
    )
    return {
        "encoder": encoder,
        "coefficients": coefficients,
        "penalty": penalty,
        "capability_slope": selected_slope,
        "capability_reference": 0.80,
        "capability_cv_mse": {str(key): value for key, value in losses.items()},
        "monotonic_capability": True,
        "price_excluded": True,
    }


def predict_quality(row, complexity, action, artifact, execution_risk=None):
    if execution_risk is None:
        execution_risk = predict_execution_risk(row, artifact["complexity_model"])
    features = quality_task_features(row, complexity, execution_risk)
    prediction = dot(
        artifact["quality_model"]["coefficients"],
        encode_dict(features, artifact["quality_model"]["encoder"]),
    )
    prediction += artifact["quality_model"]["capability_slope"] * (
        benchmark_fit(
            row, complexity, execution_risk, action, artifact["profiles"]
        ) - artifact["quality_model"]["capability_reference"]
    )
    return min(1.0, max(0.0, prediction))


def exact_key(row):
    return "|".join((row["x_task_category"], row["x_risk_level"], row["x_has_image"]))


def opening_distance(row, reference, encoder):
    squared = sum(
        (
            (raw_opening_value(row, feature) - reference["values"][feature])
            / encoder["scales"][feature]
        ) ** 2
        for feature in OPENING_FEATURES
    )
    return math.sqrt(squared / len(OPENING_FEATURES))


def fit_support(rows, opening_encoder, quantile):
    support = {}
    for action in sorted({canonical_action(row["logged_model"]) for row in rows}):
        action_rows = [row for row in rows if canonical_action(row["logged_model"]) == action]
        nearest = []
        for source in action_rows:
            candidates = [
                math.sqrt(sum(
                    ((raw_opening_value(source, feature) - raw_opening_value(target, feature))
                     / opening_encoder["scales"][feature]) ** 2
                    for feature in OPENING_FEATURES
                ) / len(OPENING_FEATURES))
                for target in action_rows
                if target is not source and exact_key(target) == exact_key(source)
            ]
            if candidates:
                nearest.append(min(candidates))
        support[action] = {
            "caliper": percentile(nearest, quantile) or 0.0,
            "reference_n": len(nearest),
            "rows": [
                {
                    "exact_key": exact_key(row),
                    "values": {feature: raw_opening_value(row, feature) for feature in OPENING_FEATURES},
                }
                for row in action_rows
            ],
        }
    return support


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def action_supported(row, action, artifact, min_stratum):
    support = artifact["support"].get(action)
    if not support:
        return False, None
    references = [ref for ref in support["rows"] if ref["exact_key"] == exact_key(row)]
    if len(references) < min_stratum:
        return False, None
    distance_value = min(
        opening_distance(row, reference, artifact["complexity_model"]["encoder"])
        for reference in references
    )
    return distance_value <= support["caliper"], distance_value


def estimated_opening_cost(row, action, artifact):
    tokens = float(row["x_total_context_tokens_est"])
    model = artifact["profiles"][action]["route_model"]
    return tokens * price_of(model, artifact["pricing"])[0] / 1_000_000


def choose_action(row, artifact, tolerance, min_stratum):
    complexity = predict_complexity(row, artifact["complexity_model"])
    execution_risk = predict_execution_risk(row, artifact["complexity_model"])
    candidates = [
        action for action, profile in artifact["profiles"].items()
        if profile.get("candidate")
    ]
    support = {
        action: action_supported(row, action, artifact, min_stratum)
        for action in candidates
    }
    available = [action for action in candidates if support[action][0]]
    if not available:
        fallback = artifact.get("fallback_action", "sonnet")
        quality = {
            fallback: predict_quality(
                row, complexity, fallback, artifact, execution_risk
            )
        }
        return fallback, complexity, quality, support, "fixed_fallback_no_support"
    quality = {
        action: predict_quality(row, complexity, action, artifact, execution_risk)
        for action in available
    }
    best = max(quality.values())
    acceptable = [action for action in available if quality[action] >= best - tolerance]
    chosen = min(
        acceptable,
        key=lambda action: (estimated_opening_cost(row, action, artifact), -quality[action]),
    )
    return chosen, complexity, quality, support, "learned_selection"


def fit_propensities(rows, actions):
    totals = Counter(exact_key(row) for row in rows)
    counts = Counter((exact_key(row), canonical_action(row["logged_model"])) for row in rows)
    return {
        key: {
            action: (counts[(key, action)] + 1) / (total + len(actions))
            for action in actions
        }
        for key, total in totals.items()
    }


def fit_artifact(
    rows, complexity_targets, risk_targets, stage1_predictions,
    profiles_config, pricing, args,
):
    profiles = profiles_config["profiles"]
    complexity_model = fit_complexity_model(
        rows, [complexity_targets[row["trajectory_id"]] for row in rows],
        [risk_targets[row["trajectory_id"]] for row in rows],
        args.complexity_penalty, args.text_features,
    )
    complexity_model["workload_calibration"] = fit_isotonic(
        [stage1_predictions["workload"][row["trajectory_id"]] for row in rows],
        [complexity_targets[row["trajectory_id"]] for row in rows],
    )
    complexity_model["risk_calibration"] = fit_isotonic(
        [stage1_predictions["risk"][row["trajectory_id"]] for row in rows],
        [risk_targets[row["trajectory_id"]] for row in rows],
    )
    calibrated_stage1 = {
        "workload": {
            row["trajectory_id"]: calibrated_value(
                stage1_predictions["workload"][row["trajectory_id"]],
                complexity_model["workload_calibration"],
            )
            for row in rows
        },
        "risk": {
            row["trajectory_id"]: calibrated_value(
                stage1_predictions["risk"][row["trajectory_id"]],
                complexity_model["risk_calibration"],
            )
            for row in rows
        },
    }
    quality_model = fit_quality_model(
        rows, calibrated_stage1, profiles, pricing, args.quality_penalty
    )
    actions = sorted(profiles)
    artifact = {
        "version": 5,
        "policy": "constrained_workload_risk_benchmark_router",
        "quality_tolerance": None,
        "fallback_action": args.fallback_action,
        "minimum_exact_stratum": args.min_stratum,
        "complexity_signals": WORKLOAD_SIGNALS,
        "execution_risk_signal": "any tool error, user correction, or duplicate successful side effect",
        "complexity_model": complexity_model,
        "quality_model": quality_model,
        "profiles": profiles,
        "profile_metadata": {
            "description": profiles_config["description"],
            "dimensions": profiles_config["dimensions"],
            "sources": profiles_config["sources"],
        },
        "pricing": pricing,
        "support": fit_support(rows, complexity_model["encoder"], args.caliper_quantile),
        "behavior_propensity": fit_propensities(rows, actions),
        "actions": actions,
    }
    return artifact


def propensity(row, action, artifact, floor):
    probabilities = artifact["behavior_propensity"].get(exact_key(row), {})
    return max(floor, probabilities.get(action, 1 / len(artifact["actions"])))


def stabilized_dr_value(details):
    direct = statistics.fmean(row["direct_policy_quality"] for row in details)
    matched = [row for row in details if row["matched_observed_action"]]
    weight_sum = sum(row["inverse_propensity"] for row in matched)
    correction = (
        sum(row["inverse_propensity"] * row["outcome_residual"] for row in matched)
        / weight_sum
        if weight_sum else 0.0
    )
    raw = direct + correction
    return min(1.0, max(0.0, raw)), raw


def effective_sample_size(details):
    weights = [
        row["inverse_propensity"]
        for row in details if row["matched_observed_action"]
    ]
    return (sum(weights) ** 2 / sum(weight * weight for weight in weights)) if weights else 0.0


def bootstrap_stabilized_delta(details, iterations, seed):
    rng = random.Random(seed)
    n = len(details)
    deltas = []
    for _ in range(iterations):
        sample = [details[rng.randrange(n)] for _ in range(n)]
        value, _ = stabilized_dr_value(sample)
        observed = statistics.fmean(row["observed_quality"] for row in sample)
        deltas.append(value - observed)
    return percentile(deltas, 0.025), percentile(deltas, 0.975)


def correlation(left, right):
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def binary_auc(outcomes, predictions):
    """Tie-aware ROC AUC; returns 0.5 when only one class is present."""
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    if not positives or not negatives:
        return 0.5
    ordered = sorted(zip(predictions, outcomes))
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(outcome for _, outcome in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap_ci(values, iterations, seed):
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def evaluate(holdout, artifact, complexity_targets, args, tolerance):
    details = []
    for row in holdout:
        chosen, predicted_complexity, quality, support, reason = choose_action(
            row, artifact, tolerance, args.min_stratum
        )
        incumbent = canonical_action(row["logged_model"])
        observed = float(row["evaluation_outcome_score"])
        predicted_risk = predict_execution_risk(row, artifact["complexity_model"])
        direct_quality = predict_quality(
            row, predicted_complexity, chosen, artifact, predicted_risk
        )
        matched = chosen == incumbent
        inverse_propensity = (
            1.0 / propensity(row, incumbent, artifact, args.propensity_floor)
            if matched else 0.0
        )
        observed_model_quality = predict_quality(
            row, predicted_complexity, incumbent, artifact, predicted_risk
        )
        details.append({
            "trajectory_id": row["trajectory_id"],
            "logged_model": row["logged_model"],
            "incumbent_action": incumbent,
            "chosen_action": chosen,
            "chosen_model": artifact["profiles"][chosen]["route_model"],
            "changed": int(chosen != incumbent),
            "decision_reason": reason,
            "observed_complexity": round(complexity_targets[row["trajectory_id"]], 6),
            "predicted_complexity": round(predicted_complexity, 6),
            "observed_execution_risk": int(observed_execution_risk(row)),
            "predicted_execution_risk": round(predicted_risk, 6),
            "predicted_quality": round(direct_quality, 6),
            "observed_quality": observed,
            "direct_policy_quality": direct_quality,
            "matched_observed_action": int(matched),
            "inverse_propensity": inverse_propensity,
            "outcome_residual": observed - observed_model_quality,
            "incumbent_opening_cost_usd_est": estimated_opening_cost(row, incumbent, artifact),
            "policy_opening_cost_usd_est": estimated_opening_cost(row, chosen, artifact),
            "supported_actions": "|".join(action for action, result in support.items() if result[0]),
        })
    observed_complexity_values = [row["observed_complexity"] for row in details]
    predicted_complexity_values = [row["predicted_complexity"] for row in details]
    errors = [predicted - observed for predicted, observed in zip(predicted_complexity_values, observed_complexity_values)]
    risk_outcomes = [row["observed_execution_risk"] for row in details]
    risk_predictions = [row["predicted_execution_risk"] for row in details]
    policy_quality, policy_quality_unbounded = stabilized_dr_value(details)
    observed_quality = statistics.fmean(row["observed_quality"] for row in details)
    quality_delta = policy_quality - observed_quality
    ci_low, ci_high = bootstrap_stabilized_delta(details, args.bootstrap, args.seed)
    incumbent_cost = sum(row["incumbent_opening_cost_usd_est"] for row in details)
    policy_cost = sum(row["policy_opening_cost_usd_est"] for row in details)
    route_counts = Counter(row["chosen_action"] for row in details)
    summary = {
        "quality_tolerance": tolerance,
        "holdout_n": len(details),
        "complexity_mae": statistics.fmean(abs(error) for error in errors),
        "complexity_rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "complexity_correlation": correlation(observed_complexity_values, predicted_complexity_values),
        "execution_risk_prevalence": statistics.fmean(risk_outcomes),
        "execution_risk_brier": statistics.fmean(
            (prediction - outcome) ** 2
            for prediction, outcome in zip(risk_predictions, risk_outcomes)
        ),
        "execution_risk_auc": binary_auc(risk_outcomes, risk_predictions),
        "changed_routes": sum(row["changed"] for row in details),
        "route_counts": dict(route_counts),
        "observed_quality": observed_quality,
        "direct_policy_quality": statistics.fmean(
            row["direct_policy_quality"] for row in details
        ),
        "dr_policy_quality": policy_quality,
        "dr_policy_quality_unbounded": policy_quality_unbounded,
        "dr_quality_delta": quality_delta,
        "dr_quality_delta_ci_low": ci_low,
        "dr_quality_delta_ci_high": ci_high,
        "matched_actions": sum(row["matched_observed_action"] for row in details),
        "overlap_effective_sample_size": effective_sample_size(details),
        "incumbent_opening_cost_usd_est": incumbent_cost,
        "policy_opening_cost_usd_est": policy_cost,
        "cost_savings_pct_est": (incumbent_cost - policy_cost) / incumbent_cost,
    }
    return summary, details


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(summary):
    result = summary["test_result"]
    return f"""# Two-stage complexity and benchmark-informed router

## Architecture

1. An opening-context model predicts a 0–100 workload target from later token growth, tool calls, and distinct tools. Sanitized TF-IDF features capture what the task asks for, without identifiers.
2. A separate opening-context head estimates execution-friction risk; errors and corrections are not treated as task workload.
3. A constrained quality model learns task difficulty and adds a non-negative, task-weighted benchmark-capability effect. Price is excluded from quality prediction.
4. The policy chooses the cheapest supported model within **{summary['selected_quality_tolerance']:.3f}** of the best predicted quality.

The validation selector requires non-negative estimated savings, stabilized quality loss no worse than **{summary['quality_loss_budget']:.3f}**, direct-model quality loss no worse than **{summary['direct_quality_loss_budget']:.3f}**, and overlap effective sample size of at least **{summary['minimum_overlap_ess_fraction']:.0%}** of the validation set.

## Template-held-out final test

- Template-grouped train / validation / final test: **{summary['initial_train_n']} / {summary['validation_n']} / {summary['test_n']}** usable trajectories. Recurring sanitized task templates never cross partitions.
- Workload model: MAE **{result['complexity_mae']:.2f}** points, RMSE **{result['complexity_rmse']:.2f}**, correlation **{result['complexity_correlation']:.3f}**.
- Execution-risk model: Brier score **{result['execution_risk_brier']:.3f}**, ROC AUC **{result['execution_risk_auc']:.3f}** at prevalence **{result['execution_risk_prevalence']:.1%}**.
- Routes changed: **{result['changed_routes']}/{result['holdout_n']}**; route counts: **{result['route_counts']}**.
- Estimated opening-input cost: **${result['incumbent_opening_cost_usd_est']:.4f} → ${result['policy_opening_cost_usd_est']:.4f}**, saving **{result['cost_savings_pct_est']:.1%}**.
- Direct-model quality: **{result['direct_policy_quality']:.3f}**. Stabilized doubly robust quality: **{result['observed_quality']:.3f} → {result['dr_policy_quality']:.3f}**; delta **{result['dr_quality_delta']:+.3f}**, bootstrap 95% CI **[{result['dr_quality_delta_ci_low']:+.3f}, {result['dr_quality_delta_ci_high']:+.3f}]**. Policy/behavior matches: **{result['matched_actions']}**; overlap ESS: **{result['overlap_effective_sample_size']:.1f}**.

## Benchmark and pricing assumptions

The challenge model ids are anonymized. Public provider pages, BFCL, and SWE-bench inform normalized ordinal analogue priors and an assumed pricing scenario; they do not establish the real model mapping. Cross-family capability spacing is varied in sensitivity analysis, and organizer-supplied prices should replace the defaults if available.

## Limitations

- The export contains estimated rather than measured token usage.
- Observed execution workload is affected by both intrinsic task difficulty and the model that happened to run.
- Only one factual model outcome is observed per task; the quality estimate remains off-policy and depends on overlap and confounding assumptions.
- The final response is missing, so most quality labels are telemetry proxies rather than semantic gold labels.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="results/calibrated_evaluation_table.csv")
    parser.add_argument("--export", default="export")
    parser.add_argument("--profiles", default="models/benchmark_priors.json")
    parser.add_argument("--artifact", default="models/two_stage_router.json")
    parser.add_argument("--details", default="results/two_stage_test.csv")
    parser.add_argument("--complexity-output", default="results/complexity_scores.csv")
    parser.add_argument("--summary-json", default="results/two_stage_summary.json")
    parser.add_argument("--summary-md", default="results/two_stage_summary.md")
    parser.add_argument("--frontier", default="results/two_stage_validation_frontier.csv")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--complexity-penalty", type=float, default=150.0)
    parser.add_argument("--text-features", type=int, default=DEFAULT_TEXT_FEATURES)
    parser.add_argument("--quality-penalty", type=float, default=15.0)
    parser.add_argument("--quality-tolerances", default="0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05")
    parser.add_argument("--quality-loss-budget", type=float, default=0.02)
    parser.add_argument("--direct-quality-loss-budget", type=float, default=0.02)
    parser.add_argument("--minimum-overlap-ess-fraction", type=float, default=0.20)
    parser.add_argument("--caliper-quantile", type=float, default=0.95)
    parser.add_argument("--min-stratum", type=int, default=3)
    parser.add_argument("--fallback-action", default="sonnet")
    parser.add_argument("--propensity-floor", type=float, default=0.03)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    with Path(args.evaluation).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    usable = [row for row in rows if row["evaluation_outcome_usable"] == "1"]
    enriched = attach_complexity_signals(usable, request_signals(args.export))
    train, validation, test = stratified_three_way(
        enriched, args.validation_fraction, args.test_fraction
    )
    profiles_config = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    pricing = load_pricing()

    train_calibrator = fit_complexity_calibrator(train)
    train_targets = {
        row["trajectory_id"]: observed_complexity(row, train_calibrator)
        for row in train + validation
    }
    train_risk_targets = {
        row["trajectory_id"]: observed_execution_risk(row)
        for row in train + validation
    }
    train_oof_stage1 = out_of_fold_stage1(
        train, train_targets, train_risk_targets,
        args.complexity_penalty, args.oof_folds, args.text_features,
    )
    validation_artifact = fit_artifact(
        train, train_targets, train_risk_targets, train_oof_stage1,
        profiles_config, pricing, args,
    )
    tolerances = sorted({float(value) for value in args.quality_tolerances.split(",")})
    validation_frontier = []
    for tolerance in tolerances:
        result, _ = evaluate(
            validation, validation_artifact, train_targets, args, tolerance
        )
        validation_frontier.append(result)
    feasible = [
        result for result in validation_frontier
        if result["dr_quality_delta"] >= -args.quality_loss_budget
        and result["direct_policy_quality"] >= (
            result["observed_quality"] - args.direct_quality_loss_budget
        )
        and result["overlap_effective_sample_size"] >= (
            result["holdout_n"] * args.minimum_overlap_ess_fraction
        )
        and result["cost_savings_pct_est"] >= 0
    ]
    if feasible:
        selected = max(feasible, key=lambda result: result["cost_savings_pct_est"])
    else:
        selected = max(
            validation_frontier,
            key=lambda result: (result["dr_quality_delta"], result["cost_savings_pct_est"]),
        )
    selected_tolerance = selected["quality_tolerance"]

    development = train + validation
    development_calibrator = fit_complexity_calibrator(development)
    development_targets = {
        row["trajectory_id"]: observed_complexity(row, development_calibrator)
        for row in development + test
    }
    development_risk_targets = {
        row["trajectory_id"]: observed_execution_risk(row)
        for row in development + test
    }
    development_oof_stage1 = out_of_fold_stage1(
        development, development_targets, development_risk_targets,
        args.complexity_penalty, args.oof_folds, args.text_features,
    )
    test_artifact = fit_artifact(
        development, development_targets, development_risk_targets, development_oof_stage1,
        profiles_config, pricing, args,
    )
    test_artifact["quality_tolerance"] = selected_tolerance
    test_summary, details = evaluate(
        test, test_artifact, development_targets, args, selected_tolerance
    )

    # Refit both stages on all usable rows after freezing the holdout result.
    final_calibrator = fit_complexity_calibrator(enriched)
    final_targets = {
        row["trajectory_id"]: observed_complexity(row, final_calibrator)
        for row in enriched
    }
    final_risk_targets = {
        row["trajectory_id"]: observed_execution_risk(row)
        for row in enriched
    }
    final_oof_stage1 = out_of_fold_stage1(
        enriched, final_targets, final_risk_targets,
        args.complexity_penalty, args.oof_folds, args.text_features,
    )
    final_artifact = fit_artifact(
        enriched, final_targets, final_risk_targets, final_oof_stage1,
        profiles_config, pricing, args,
    )
    final_artifact["complexity_calibrator"] = final_calibrator
    final_artifact["quality_tolerance"] = selected_tolerance
    final_artifact["training_rows"] = len(enriched)
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(final_artifact, indent=2) + "\n", encoding="utf-8")

    complexity_rows = [
        {
            "trajectory_id": row["trajectory_id"],
            "logged_model": row["logged_model"],
            "task_category": row["x_task_category"],
            "risk_level": row["x_risk_level"],
            "opening_context_tokens_est": row["x_total_context_tokens_est"],
            "observed_complexity": round(final_targets[row["trajectory_id"]], 6),
            "predicted_complexity": round(predict_complexity(row, final_artifact["complexity_model"]), 6),
            "observed_execution_risk": int(final_risk_targets[row["trajectory_id"]]),
            "predicted_execution_risk": round(
                predict_execution_risk(row, final_artifact["complexity_model"]), 6
            ),
        }
        for row in enriched
    ]

    summary = {
        "architecture": "opening workload + execution-risk models -> shared benchmark-informed quality regression -> cost-constrained selection",
        "initial_train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "initial_train_actions": dict(Counter(canonical_action(row["logged_model"]) for row in train)),
        "validation_actions": dict(Counter(canonical_action(row["logged_model"]) for row in validation)),
        "test_actions": dict(Counter(canonical_action(row["logged_model"]) for row in test)),
        "complexity_signal_weights": WORKLOAD_SIGNALS,
        "text_features": args.text_features,
        "split_strategy": "sanitized task-template grouped",
        "quality_loss_budget": args.quality_loss_budget,
        "direct_quality_loss_budget": args.direct_quality_loss_budget,
        "minimum_overlap_ess_fraction": args.minimum_overlap_ess_fraction,
        "selected_quality_tolerance": selected_tolerance,
        "validation_frontier": validation_frontier,
        "benchmark_mapping_confidence": {
            action: profile["mapping_confidence"]
            for action, profile in profiles_config["profiles"].items()
        },
        "test_result": test_summary,
        "artifact": args.artifact,
    }
    write_csv(args.details, details)
    write_csv(args.complexity_output, complexity_rows)
    write_csv(args.frontier, validation_frontier)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path(args.summary_md).write_text(markdown_summary(summary), encoding="utf-8")

    print(
        f"train={len(train)} validation={len(validation)} test={len(test)} "
        f"actions={summary['initial_train_actions']}"
    )
    print(
        f"selected tolerance={selected_tolerance:.3f} on validation: "
        f"saving={selected['cost_savings_pct_est']:.1%} "
        f"quality_delta={selected['dr_quality_delta']:+.4f}"
    )
    print(
        f"workload: MAE={test_summary['complexity_mae']:.2f} "
        f"RMSE={test_summary['complexity_rmse']:.2f} "
        f"r={test_summary['complexity_correlation']:.3f}"
    )
    print(
        f"execution risk: Brier={test_summary['execution_risk_brier']:.3f} "
        f"AUC={test_summary['execution_risk_auc']:.3f}"
    )
    print(
        f"policy: changed={test_summary['changed_routes']}/{len(test)} "
        f"saving={test_summary['cost_savings_pct_est']:.1%} "
        f"quality_delta={test_summary['dr_quality_delta']:+.4f} "
        f"CI=[{test_summary['dr_quality_delta_ci_low']:+.4f}, "
        f"{test_summary['dr_quality_delta_ci_high']:+.4f}]"
    )
    print(f"wrote {args.artifact}, {args.details}, {args.summary_md}")


if __name__ == "__main__":
    main()
