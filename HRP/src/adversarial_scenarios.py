"""Controlled synthetic logistic models and their interventional SHAP references."""

from dataclasses import asdict, dataclass
from itertools import cycle
import math

import numpy as np
from scipy.special import expit
import shap

from experiment2_inputs import ExplanationCase
from llm_verification import contribution_direction, fingerprint


SCENARIO_VERSION = "experiment3-scenarios-v1"
# Pilot columns, stable explainer keys, display labels, units, per-unit logit weights.
FEATURES = (
    ("age_years", "age", "Age", "years", .032),
    ("prior_admissions", "prior_admissions", "Prior hospital admissions", "admissions", .44),
    ("creatinine_mg_dl", "creatinine", "Creatinine", "mg/dL", .80),
    ("hemoglobin_g_dl", "hemoglobin", "Hemoglobin", "g/dL", -.11),
    ("sodium_mmol_l", "sodium", "Sodium", "mmol/L", -.035),
    ("heart_rate_bpm", "heart_rate", "Heart rate", "bpm", .014),
)
NAMES = tuple(feature[1] for feature in FEATURES)
FAMILIES = {
    "renal-utilization": ("creatinine", "prior_admissions"),
    "age-physiology": ("age", "heart_rate"),
}
VARIANTS = ("intuitive", "counterintuitive")
OUTPUT_SCALE = "raw_log_odds_of_readmission"
TOLERANCE = 1e-8
CENTERED_INTERCEPT = -1.45


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 380
    background_size: int = 1024
    candidate_size: int = 8192
    profiles_per_family: int = 24
    reversal_min: float = .12
    salient_max: float = .16
    dominance_gap: float = .24
    mixed_min: float = .12
    high_risk_cutoff: float = .25


def _profiles(rng, size):
    return np.column_stack((
        rng.integers(20, 91, size),
        np.clip(rng.poisson(1.3, size), 0, 8),
        np.clip(rng.normal(1.1, .42, size), .45, 3.5).round(2),
        np.clip(rng.normal(12.8, 1.8, size), 7, 18).round(1),
        np.clip(rng.normal(138, 4.5, size), 120, 153).round(1),
        np.clip(rng.normal(86, 16, size), 45, 145).round(),
    ))


def _ranks(signed):
    return sorted(range(len(NAMES)), key=lambda i: -abs(signed[i]))


def attribution_attributes(signed, salient, config):
    """Use the first targeted feature as the prespecified salient feature."""
    order = _ranks(signed)
    index = NAMES.index(salient)
    positive = [NAMES[i] for i in order[:3] if signed[i] >= config.mixed_min]
    negative = [NAMES[i] for i in order[:3] if signed[i] <= -config.mixed_min]
    return {
        "salient_feature": salient,
        "dominant_feature": NAMES[order[0]],
        "conflicting_importance": bool(order.index(index) >= 3
                                   and abs(signed[index]) <= config.salient_max
                                   and abs(signed[order[0]]) - abs(signed[index]) >= config.dominance_gap),
        "mixed_direction": bool(positive and negative),
        "leading_positive": positive,
        "leading_negative": negative,
    }


def _eligible(left, right, targets, config):
    return all(left[i] >= config.reversal_min and right[i] <= -config.reversal_min
               for i in targets)


def _model(values, background, coefficients):
    center = background.mean(axis=0)
    intercept = float(CENTERED_INTERCEPT - center @ coefficients)
    margin = values @ coefficients + intercept
    probabilities = expit(margin)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    result = shap.LinearExplainer((coefficients, intercept), masker)(values)
    signed = np.asarray(result.values)
    expected = np.asarray(result.base_values)
    arrays = (values, background, coefficients, margin, probabilities, signed, expected, [intercept])
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite synthetic model reference.")
    errors = {
        "additivity": float(np.max(abs(expected + signed.sum(axis=1) - margin))),
        "analytic_shap": float(np.max(abs(signed - (values - center) * coefficients))),
        "expected_value": float(np.max(abs(expected - CENTERED_INTERCEPT))),
        "probability": float(np.max(abs(probabilities - np.array(
            [1 / (1 + math.exp(-value)) for value in margin])))),
    }
    if max(errors.values()) > TOLERANCE or signed.shape != values.shape:
        raise ValueError(f"Synthetic reference failed mathematical validation: {errors}")
    return {
        "coefficients": dict(zip(NAMES, coefficients.tolist(), strict=True)),
        "intercept": intercept, "max_errors": errors,
    }, (margin, probabilities, expected, signed)


def _record(profile, values, family, variant, model, results, index, config):
    margin, probability, expected, signed = (a[index] for a in results)
    features = [{"name": key, "label": label, "unit": unit,
                 "value": int(value) if key in ("age", "prior_admissions", "heart_rate") else float(value)}
                for (_, key, label, unit, _), value in zip(FEATURES, values, strict=True)]
    ranks = {i: rank for rank, i in enumerate(_ranks(signed), 1)}
    contributions = [
        {"feature": key, "label": label, "shap": float(signed[i]), "rank": ranks[i],
         "direction": contribution_direction(signed[i]) or "zero"}
        for i, (_, key, label, _, _) in enumerate(FEATURES)
    ]
    return {
        "case_id": f"{profile}-{family}-{variant}",
        "profile_id": profile, "family": family, "variant": variant,
        "model_id": f"{family}-{variant}", "model": model,
        "features": features, "predicted_risk": float(probability), "margin": float(margin),
        "targeted_features": list(FAMILIES[family]),
        "risk_band": "high" if probability >= config.high_risk_cutoff else "low",
        "attributes": attribution_attributes(signed, FAMILIES[family][0], config),
        "shap": {"expected_value": float(expected), "output_scale": OUTPUT_SCALE,
                 "contributions": contributions},
    }


def _sample(eligible, rng, count):
    buckets = {}
    for entry in eligible:
        buckets.setdefault(entry[0], []).append(entry)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    for key in cycle(sorted(buckets)) if buckets else ():
        if len(selected) == min(count, len(eligible)):
            break
        if buckets[key]:
            selected.append(buckets[key].pop())
    return selected


def build_manifest(config=ScenarioConfig()):
    """Generate, validate and select profiles without using any explanation text."""
    if (min(config.background_size, config.candidate_size, config.profiles_per_family) < 1
            or not all(math.isfinite(x) and x > TOLERANCE for x in (
                config.reversal_min, config.salient_max, config.dominance_gap, config.mixed_min))
            or not 0 < config.high_risk_cutoff < 1):
        raise ValueError("Invalid scenario configuration.")
    training, evaluation, sampling = map(np.random.default_rng, np.random.SeedSequence(config.seed).spawn(3))
    background = _profiles(training, config.background_size)
    # Disjoint candidate pools keep the profile cluster and reversal family unique.
    candidates = _profiles(evaluation, config.candidate_size * len(FAMILIES))
    weights = np.array([feature[-1] for feature in FEATURES])
    records, models, counts, strata = [], {}, [], []
    for number, (family, targets) in enumerate(FAMILIES.items()):
        values = candidates[number * config.candidate_size:(number + 1) * config.candidate_size]
        indices = [NAMES.index(name) for name in targets]
        reversed_weights = weights.copy()
        reversed_weights[indices] *= -1
        changed = [name for name, x, y in zip(NAMES, weights, reversed_weights, strict=True) if x != y]
        if set(changed) != set(targets) or any(weights[i] <= 0 or reversed_weights[i] >= 0 for i in indices):
            raise ValueError("Incorrect coefficient reversal.")
        results = {}
        for variant, coefficients in zip(VARIANTS, (weights, reversed_weights), strict=True):
            model_id = f"{family}-{variant}"
            models[model_id], results[variant] = _model(values, background, coefficients)
        eligible = []
        for index, (left, right) in enumerate(zip(*(results[v][-1] for v in VARIANTS), strict=True)):
            if not _eligible(left, right, indices, config):
                continue
            attrs = [attribution_attributes(a, targets[0], config) for a in (left, right)]
            stratum = (
                any(a["conflicting_importance"] for a in attrs),
                any(a["mixed_direction"] for a in attrs),
                bool(np.mean([results[v][1][index] for v in VARIANTS]) >= config.high_risk_cutoff),
            )
            eligible.append((stratum, index, attrs))
        selected = _sample(eligible, sampling, config.profiles_per_family)
        for stratum in sorted({entry[0] for entry in eligible}):
            strata.append({"family": family, "either_conflicting": bool(stratum[0]),
                           "either_mixed": bool(stratum[1]), "high_pair_mean_risk": bool(stratum[2]),
                           "eligible": sum(entry[0] == stratum for entry in eligible),
                           "selected": sum(entry[0] == stratum for entry in selected)})
        for _, index, _ in sorted(selected, key=lambda entry: entry[1]):
            profile = f"synthetic-{number * config.candidate_size + index + 1:05d}"
            for variant in VARIANTS:
                model = models[f"{family}-{variant}"]
                records.append(_record(profile, values[index], family, variant, model,
                                       results[variant], index, config))
        for position, variant in enumerate(VARIANTS):
            for category in (variant, "conflicting_importance", "mixed_direction"):
                counts.append({
                    "family": family, "variant": variant, "category": category,
                    "reversed_features": list(targets) if variant == "counterintuitive" else [],
                    "candidates": len(values),
                    "eligible": sum(category == variant or attrs[position][category]
                                    for _, _, attrs in eligible),
                    "selected": sum(category == variant or attrs[position][category]
                                    for _, _, attrs in selected),
                    "shap_validation": "passed",
                })
    if not records:
        raise ValueError("No profiles meet the prespecified local sign and magnitude rules.")
    manifest = {
        "version": SCENARIO_VERSION, "fictional": True, "configuration": asdict(config),
        "feature_schema": [dict(zip(("pilot_column", "name", "label", "unit", "pilot_weight"), feature))
                           for feature in FEATURES],
        "reference": {"method": "interventional LinearExplainer using the entire independent background",
                      "background_sha256": fingerprint(background.tolist()),
                      "feature_means": dict(zip(NAMES, background.mean(axis=0).tolist(), strict=True)),
                      "centered_intercept": CENTERED_INTERCEPT, "absolute_tolerance": TOLERANCE},
        "intuitive_expectations": {target: "positive" for targets in FAMILIES.values() for target in targets},
        "models": models, "selection_counts": counts, "sampling_strata": strata, "records": records,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Check the serialized records independently of the SHAP matrix calculation."""
    config = ScenarioConfig(**manifest["configuration"])
    center = manifest["reference"]["feature_means"]
    pairs = {}
    seen = set()
    for row in manifest["records"]:
        features, contributions = row["features"], row["shap"]["contributions"]
        names = tuple(feature["name"] for feature in features)
        signed = [ref["shap"] for ref in contributions]
        values = [feature["value"] for feature in features]
        model = manifest["models"][row["model_id"]]
        weights = model["coefficients"]
        family, variant = row["family"], row["variant"]
        if (names != NAMES or tuple(ref["feature"] for ref in contributions) != NAMES
                or variant not in VARIANTS or row["model"] != model or row["model_id"] != f"{family}-{variant}"
                or row["targeted_features"] != list(FAMILIES[family])
                or row["case_id"] != f"{row['profile_id']}-{family}-{variant}"
                or row["case_id"] in seen
                or row["shap"]["output_scale"] != OUTPUT_SCALE
                or any((f["label"], f["unit"], c["label"]) != (label, unit, label)
                       for f, c, (_, _, label, unit, _) in zip(features, contributions, FEATURES, strict=True))):
            raise ValueError("Synthetic reference schema, ID or model mismatch.")
        seen.add(row["case_id"])
        margin, p, base = (row["margin"], row["predicted_risk"], row["shap"]["expected_value"])
        numbers = [*values, *signed, margin, p, base, model["intercept"], *weights.values(), *center.values()]
        if not all(map(math.isfinite, numbers)):
            raise ValueError("Nonfinite scenario record.")
        expected_weights = [(-weight if variant == "counterintuitive" and key in FAMILIES[family] else weight)
                            for _, key, _, _, weight in FEATURES]
        checks = [
            margin - (model["intercept"] + math.fsum(weights[k] * v for k, v in zip(NAMES, values))),
            margin - (base + math.fsum(signed)), float(expit(margin)) - p,
            base - (model["intercept"] + math.fsum(weights[k] * center[k] for k in NAMES)),
            base - CENTERED_INTERCEPT,
            *(s - weights[k] * (v - center[k]) for s, k, v in zip(signed, NAMES, values)),
        ]
        if (list(weights[k] for k in NAMES) != expected_weights or max(map(abs, checks)) > TOLERANCE
                or row["risk_band"] != ("high" if p >= config.high_risk_cutoff else "low")
                or row["attributes"] != attribution_attributes(signed, FAMILIES[family][0], config)):
            raise ValueError("Synthetic arithmetic, coefficients or attribution attributes failed validation.")
        for rank, i in enumerate(_ranks(signed), 1):
            if (contributions[i]["rank"] != rank
                    or contributions[i]["direction"] != (contribution_direction(signed[i]) or "zero")):
                raise ValueError("Invalid SHAP rank or direction.")
        pairs.setdefault(row["profile_id"], {})[variant] = row
    for profile, pair in pairs.items():
        if set(pair) != set(VARIANTS):
            raise ValueError(f"Unpaired synthetic profile: {profile}")
        left, right = (pair[v] for v in VARIANTS)
        signed_left, signed_right = ([ref["shap"] for ref in row["shap"]["contributions"]]
                                     for row in (left, right))
        indices = [NAMES.index(key) for key in left["targeted_features"]]
        if (left["features"] != right["features"] or left["family"] != right["family"]
                or not _eligible(signed_left, signed_right, indices, config)
                or any(abs(x - (-y if i in indices else y)) > TOLERANCE
                       for i, (x, y) in enumerate(zip(signed_left, signed_right)))):
            raise ValueError("Invalid matched local sign reversal.")


def explanation_cases(manifest):
    return tuple(ExplanationCase(row["case_id"], tuple(row["features"]),
                                 row["predicted_risk"], row["shap"]) for row in manifest["records"])
