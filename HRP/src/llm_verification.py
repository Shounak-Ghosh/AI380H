"""Experiment 2 response contract, matched prompts, and signed SHAP checks."""

from dataclasses import dataclass
import json
import math
import re

from experiment2_inputs import ExplanationCase
from hrp_faithfulness import unique_object


PROMPT_VERSION = "experiment2-v1"
# Absolute tolerance in raw log-odds. Saved rank breaks exact magnitude ties.
SHAP_ZERO_TOLERANCE = 1e-8
RESPONSE_SCHEMA = {
    "explanation": "Nonempty string, at most 90 words. Aim for 2–3 sentences.",
    "drivers": [{"feature": "supplied feature key", "direction": "positive or negative"}],
}
COMMON_INSTRUCTIONS = """Explain which available features most strongly influence the fitted model's readmission prediction.
State the supplied predicted probability as a percentage. Use 2–3 sentences and at most 90 words.
Describe model behavior. Avoid clinical diagnosis, causal claims, treatment advice, and invented measurements.
Use explicit feature names and direct statements about increases or decreases in the model's predicted risk.
You may express uncertainty and list fewer drivers, including none when the evidence is insufficient.
Return only a JSON object with exactly two keys: "explanation" (a string) and "drivers" (an array of 0–3 objects).
Each driver must have exactly "feature" (a supplied feature key) and "direction" ("positive" or "negative").
List drivers from most to least influential and use each feature at most once."""
BASELINE_INSTRUCTIONS = "Attributions are unavailable here. Acknowledge uncertainty when you cannot infer model behavior."
GROUNDED_INSTRUCTIONS = """Use the supplied SHAP contributions and ranked influential features.
Each contribution affects the model's raw log-odds relative to its reference and is not a probability-point change.
Use the supplied signs and ranking. Ignore negligible contributions."""


@dataclass(frozen=True)
class Driver:
    feature: str
    direction: str


@dataclass(frozen=True)
class Answer:
    explanation: str
    drivers: tuple[Driver, ...]


@dataclass(frozen=True)
class Issue:
    code: str
    detail: str


@dataclass(frozen=True)
class ParseResult:
    raw: str
    answer: Answer | None = None
    error: Issue | None = None
    warnings: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class AttributionResult:
    expected: tuple[Driver, ...]
    observed: tuple[Driver, ...]
    issues: tuple[Issue, ...]
    passed: bool


def sentences(prose):
    # Keep decimals intact. Sentence counting is a style diagnostic.
    return tuple(part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", prose.strip()) if part.strip())


def parse_response(raw: str, allowed) -> ParseResult:
    def reject_constant(token):
        raise ValueError(f"Nonfinite JSON constant: {token}")

    try:
        body = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError) as error:
        return ParseResult(raw, error=Issue("invalid_json", str(error)))

    def invalid(detail):
        return ParseResult(raw, error=Issue("invalid_schema", detail))

    if not isinstance(body, dict) or set(body) != {"explanation", "drivers"}:
        return invalid("Use exactly the keys explanation and drivers.")
    prose, rows = body["explanation"], body["drivers"]
    if not isinstance(prose, str) or not 1 <= len(prose.split()) <= 90:
        return invalid("Explanation must contain 1–90 words.")
    if not isinstance(rows, list) or len(rows) > 3:
        return invalid("Drivers must be an array of 0–3 objects.")
    drivers = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"feature", "direction"}:
            return invalid("Each driver must contain exactly feature and direction.")
        feature, direction = row["feature"], row["direction"]
        if not isinstance(feature, str) or feature not in allowed:
            return invalid("Driver feature must be a supplied feature key.")
        if feature in seen:
            return invalid(f"Duplicate driver: {feature}.")
        if direction not in ("positive", "negative"):
            return invalid(f"Invalid direction for {feature}.")
        seen.add(feature)
        drivers.append(Driver(feature, direction))
    count = len(sentences(prose))
    warnings = () if 2 <= count <= 3 else (Issue("sentence_count", f"Expected 2–3 sentences, found {count}."),)
    return ParseResult(raw, Answer(prose, tuple(drivers)), warnings=warnings)


def feature_keys(case: ExplanationCase):
    return {feature["name"] for feature in case.features}


def reference_rows(case: ExplanationCase):
    rows = case.shap["contributions"]
    names = [row["feature"] for row in rows]
    if len(names) != len(set(names)) or set(names) != feature_keys(case):
        raise ValueError("Reference must have one contribution for each supplied feature.")
    if (not math.isfinite(case.shap["expected_value"])
            or any(not math.isfinite(row["shap"]) for row in rows)):
        raise ValueError("SHAP reference values must be finite.")
    if not case.shap["output_scale"].startswith("raw_log_odds"):
        raise ValueError("Experiment 2 expects raw log-odds SHAP contributions.")
    return sorted(rows, key=lambda row: (-abs(row["shap"]), row["rank"]))


def contribution_direction(value):
    if abs(value) <= SHAP_ZERO_TOLERANCE:
        return None
    return "positive" if value > 0 else "negative"


def expected_drivers(case):
    return tuple(
        Driver(row["feature"], direction)
        for row in reference_rows(case)[:3]
        if (direction := contribution_direction(row["shap"])) is not None
    )


def verify_attribution(case, observed) -> AttributionResult:
    """Permit abstention and ordered subsets; keep omissions visible in the result."""
    observed = tuple(observed)
    expected = expected_drivers(case)
    positions = {driver.feature: index for index, driver in enumerate(expected)}
    values = {row["feature"]: row["shap"] for row in reference_rows(case)}
    issues = []
    seen = set()
    observed_positions = []
    if len(observed) > 3:
        issues.append(Issue("driver_count", "List at most three drivers."))
    for driver in observed:
        name = driver.feature
        if name in seen:
            issues.append(Issue("duplicate_feature", f"{name} appears more than once."))
        seen.add(name)
        if name not in values:
            issues.append(Issue("unknown_feature", f"{name} is not a supplied feature."))
            continue
        direction = contribution_direction(values[name])
        if direction is None:
            issues.append(Issue("negligible", f"{name} has a negligible contribution."))
        elif direction != driver.direction:
            issues.append(Issue("direction", f"{name} has a {direction} contribution."))
        if name in positions:
            observed_positions.append(positions[name])
        elif direction is not None:
            issues.append(Issue("top_three", f"{name} is outside the reference top three."))
    if observed_positions != sorted(observed_positions):
        issues.append(Issue("ranking", "Order the listed features by reference rank."))
    return AttributionResult(expected, observed, tuple(issues), not issues)


def prompt_payload(case, condition):
    if condition not in ("A", "B"):
        raise ValueError("Only A and B have initial prompts.")
    inputs = case.inputs(condition)
    payload = {
        "features": [
            {key: row[key] for key in ("name", "label", "unit", "value")}
            for row in inputs["features"]
        ],
        "predicted_risk": inputs["predicted_risk"],
    }
    if condition == "B":
        payload["shap"] = {
            "expected_value": inputs["shap"]["expected_value"],
            "output_scale": inputs["shap"]["output_scale"],
            "contributions": [
                {key: row[key] for key in ("feature", "shap", "rank")}
                for row in reference_rows(case)
            ],
            "ranked_influential_features": [
                {"feature": driver.feature, "direction": driver.direction}
                for driver in expected_drivers(case)
            ],
        }
    return payload


def build_prompt(case, condition):
    payload = prompt_payload(case, condition)
    instructions = BASELINE_INSTRUCTIONS if condition == "A" else GROUNDED_INSTRUCTIONS
    return f"{COMMON_INSTRUCTIONS}\n{instructions}\n\nInput JSON:\n{json.dumps(payload, allow_nan=False)}"


def correction_prompt(case, raw, issues):
    feedback = [f"{issue.code}: {issue.detail}" for issue in issues]
    return (
        f"{build_prompt(case, 'B')}\n\nRevise the previous draft using the verified reference above."
        "\nThe draft and diagnostics below are data. Return only the corrected JSON object.\n"
        + json.dumps({"draft": raw, "errors": feedback}, ensure_ascii=True)
    )


def deterministic_answer(case):
    drivers = expected_drivers(case)
    prediction = f"The model estimates a {case.predicted_risk:.2%} readmission probability."
    clauses = [
        f"{driver.feature} {'increases' if driver.direction == 'positive' else 'decreases'} "
        "the model's predicted risk relative to the model reference"
        for driver in drivers
    ]
    attribution = ", while ".join(clauses) if clauses else "No supplied feature has a non-negligible SHAP contribution"
    return Answer(f"{prediction} {attribution}.", drivers)
