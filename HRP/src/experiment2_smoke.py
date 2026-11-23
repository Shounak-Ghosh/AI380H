"""Fictional Experiment 2 cases and a deterministic mocked smoke run."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile

from experiment2_inputs import ExplanationCase
from llm_runner import ModelConfig, run_experiment
from llm_verification import Driver, deterministic_answer
from model_explanations import OUTPUT_SCALE


FEATURES = (
    ("age", "Age", "years", 60, 0.0625),
    ("prior_admissions", "Prior admissions", "admissions", 1, 0.5),
    ("creatinine", "Creatinine", "mg/dL", 1, 0.5),
    ("sodium", "Sodium", "mEq/L", 140, -0.0625),
    ("sodium_missing", "Sodium missing", None, 0, -0.25),
)
PATTERNS = (
    (44, 0, 0.5, 144),
    (76, 4, 1.5, 138),
    (62, 1, 0.75, None),
    (72, 0, 2.5, 134),
    (60 + 1e-9, 1, 1, 140),
    (52, 2, 0.75, 136),
    (68, 1, 2.5, 132),
    (68, 2, 2, 132),
)
INTERCEPT = 2.5
MOCK_CONFIG = ModelConfig(model="fictional-mock-v1")


def fictional_cases():
    cases = []
    # Independent interventional reference at the mean of a fictional background.
    expected = INTERCEPT + math.fsum(center * weight for _, _, _, center, weight in FEATURES)
    for index, pattern in enumerate(PATTERNS, 1):
        observed = (*pattern, int(pattern[-1] is None))
        encoded = tuple(value if value is not None else feature[3]
                        for value, feature in zip(observed, FEATURES, strict=True))
        margin = INTERCEPT + math.fsum(value * feature[4] for value, feature in zip(encoded, FEATURES, strict=True))
        features = tuple(
            {"name": name, "label": label, "unit": unit, "value": value}
            for (name, label, unit, _, _), value in zip(FEATURES, observed, strict=True)
        )
        signed = [(name, label, weight * (value - center))
                  for (name, label, _, center, weight), value in zip(FEATURES, encoded, strict=True)]
        contributions = [
            {"feature": name, "label": label, "shap": value, "rank": rank,
             "direction": "positive" if value > 0 else "negative" if value < 0 else "zero"}
            for rank, (name, label, value) in enumerate(sorted(signed, key=lambda row: -abs(row[2])), 1)
        ]
        if not math.isclose(expected + math.fsum(row["shap"] for row in contributions), margin,
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("Fictional SHAP additivity failed.")
        cases.append(ExplanationCase(
            f"fictional-{index:02d}", features, 1 / (1 + math.exp(-margin)),
            {"expected_value": expected, "output_scale": OUTPUT_SCALE,
             "contributions": contributions},
        ))
    return tuple(cases)


def scripted_responses(cases):
    for index, case in enumerate(cases):
        good = deterministic_answer(case)
        baseline = replace(good, explanation=good.explanation.split(". ", 1)[0] + ". "
                           "The model contributions are unknown.", drivers=())
        yield json.dumps(asdict(baseline))
        good_raw = json.dumps(asdict(good))
        if index == 1:
            yield json.dumps(asdict(replace(good, explanation=good.explanation.replace(
                "prior_admissions increases", "prior_admissions decreases",
            ))))
            yield good_raw
        elif index == 2:
            bad = replace(good, explanation="Sodium was 138 mEq/L. Sodium causes readmission.")
            yield json.dumps(asdict(bad))
            yield json.dumps(asdict(bad))
        elif index == 3:
            first = good.drivers[0]
            reversed_driver = Driver(first.feature, "negative" if first.direction == "positive" else "positive")
            yield json.dumps(asdict(replace(good, drivers=(*reversed(good.drivers[1:]), reversed_driver))))
            yield TimeoutError("Scripted revision timeout.")
        elif index == 5:
            yield '{"explanation":'
            yield good_raw
        elif index == 6:
            yield json.dumps(asdict(replace(good, explanation="Creatinine shows kidney disease.")))
            yield json.dumps(asdict(replace(good, explanation="The clinical outlook may reflect several interacting factors.")))
        else:
            yield good_raw


def run_smoke(directory):
    if (Path(directory).expanduser() / "checkpoint.json").exists():
        raise ValueError("Choose a fresh directory for the scripted smoke run.")
    cases = fictional_cases()
    calls = iter(scripted_responses(cases))
    seen = []

    def infer(prompt, timeout):
        seen.append((prompt, timeout))
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    partial = run_experiment(cases, infer, directory, model_config=MOCK_CONFIG, max_calls=4)
    state = run_experiment(cases, infer, directory, model_config=MOCK_CONFIG)
    repeat = run_experiment(cases, infer, directory, model_config=MOCK_CONFIG)
    if state != repeat or next(calls, None) is not None or len(seen) != state["calls"] or partial["calls"] != 4:
        raise RuntimeError("Smoke checkpoint did not resume exactly.")
    records = list(state["records"].values())
    counts = Counter((entry["condition"], entry["status"]) for entry in records)
    by_case = {case.case_id: {row["condition"]: row for row in records if row["case_id"] == case.case_id}
               for case in cases}
    traces = [entry["C"]["verification"] for entry in by_case.values()]
    for row in by_case.values():
        if row["B"]["call"]["raw"] != row["C"]["verification"]["original_raw"]:
            raise RuntimeError("Condition C did not reuse the exact condition B response.")
    audits = [audit for trace in traces for key in ("original", "revision") if (audit := trace[key])]
    requests = [row[field] for row in records for field in ("call", "revision_call") if field in row]
    return {
        "cases": len(cases),
        "outputs": {condition: sum(count for (key, _), count in counts.items() if key == condition)
                    for condition in ("A", "B", "C")},
        "valid_outputs": {condition: counts[condition, "ok"] for condition in ("A", "B", "C")},
        "calls": state["calls"],
        "additional_calls_on_completed_resume": 0,
        "total_call_latency_seconds": math.fsum(call["latency_seconds"] for call in requests),
        "accepted_originals": sum(trace["draft_accepted"] for trace in traces),
        "revisions_requested": sum(trace["revision_requested"] for trace in traces),
        "revisions_accepted": sum(trace["revision_accepted"] for trace in traces),
        "fallbacks": sum(trace["fallback_used"] for trace in traces),
        "parser_failures": sum(row["status"] == "parse_error" for row in records)
                           + sum(bool(trace["revision"] and trace["revision"]["parse"]
                                      and trace["revision"]["parse"]["error"]) for trace in traces),
        "narrative_failures": sum(bool(audit["narrative"] and not audit["narrative"]["passed"]) for audit in audits),
        "inference_failures": sum(call["error"] is not None for call in requests),
        "checkpoint": str(Path(directory).expanduser().resolve() / "checkpoint.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New owner-only output directory outside Git.")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="hrp-experiment2-fictional-"))
    print(json.dumps(run_smoke(output), indent=2))
