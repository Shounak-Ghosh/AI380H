"""Compare Experiment 2 explanations against a common saved SHAP reference."""

from collections import Counter, defaultdict
from statistics import mean

from llm_verification import (
    SHAP_ZERO_TOLERANCE, VERIFIER_VERSION, Answer, Driver, case_fingerprints, sentences, verify_attribution,
)
from narrative_verifier import missing_direction_claims, verify_narrative


SCORER_VERSION = "experiment2-scorer-v1"
QUALITY_METRICS = (
    "top_three_overlap", "signed_top_three_agreement", "rank_position_agreement",
    "exact_top_three_ordering", "missing_influential_drivers", "incorrect_feature_directions",
    "structured_correct", "structured_complete", "explanation_correct", "explanation_complete",
    "missing_prose_drivers", "narrative_contradictions", "unsupported_model_claims",
    "narrative_claims_checked", "narrative_claims_unparsed",
)
UNSUPPORTED = {"unsupported_feature", "causal", "clinical", "scale", "negligible", "false_abstention"}
CONDITIONS = ("A", "B", "C")


def score_answer(case, answer: Answer):
    """Score feature ranks separately from direction, and count prose findings by clause."""
    attribution = verify_attribution(case, answer.drivers)
    expected = attribution.expected
    expected_names = tuple(driver.feature for driver in expected)
    observed_names = tuple(driver.feature for driver in answer.drivers)
    narrative = verify_narrative(case, answer)
    missing_prose = len(missing_direction_claims(expected, narrative))

    def count_clauses(statuses, codes=None):
        return len({claim.clause_index for claim in narrative.claims
                    if claim.status in statuses and (codes is None or claim.code in codes)})

    return {
        "top_three_overlap": len(set(expected_names) & set(observed_names)),
        "signed_top_three_agreement": len(set(expected) & set(answer.drivers)),
        "rank_position_agreement": sum(x == y for x, y in zip(expected_names, observed_names)),
        "exact_top_three_ordering": expected_names == observed_names,
        "missing_influential_drivers": len(attribution.missing),
        "incorrect_feature_directions": sum(issue.code == "direction" for issue in attribution.issues),
        "structured_correct": attribution.passed,
        "structured_complete": attribution.complete,
        "explanation_correct": attribution.passed and narrative.passed,
        "explanation_complete": attribution.complete and missing_prose == 0,
        "missing_prose_drivers": missing_prose,
        "narrative_contradictions": count_clauses({"contradicted"}),
        "unsupported_model_claims": count_clauses({"contradicted"}, UNSUPPORTED),
        "narrative_claims_checked": count_clauses({"supported", "contradicted"}),
        "narrative_claims_unparsed": count_clauses({"ambiguous"}),
        "non_attribution_clauses": count_clauses({"non_attribution"}),
        "word_count": len(answer.explanation.split()),
        "sentence_count": len(sentences(answer.explanation)),
    }


def _score_record(case, record):
    condition = record["condition"]
    verification = record.get("verification")
    call = record.get("revision_call" if condition == "C" else "call")
    parsed = verification["revision"]["parse"] if verification and verification["revision"] else record.get("parse")
    source = verification["source"] if verification else (
        "independent_generation" if condition in ("A", "B") else None
    )
    body = (verification["answer"] if verification else parsed["answer"]) if record["status"] == "ok" else None
    answer = Answer(body["explanation"], tuple(Driver(**driver) for driver in body["drivers"])) if body else None
    metrics = score_answer(case, answer) if answer else None
    return {
        "case_id": case.case_id, "condition": condition, "status": record["status"], "source": source,
        "quality": metrics,
        "parse_failure": bool(parsed and parsed["error"]),
        "inference_failure": call["error"]["code"] if call and call["error"] else None,
        "calls": int(call is not None), "latency_seconds": call["latency_seconds"] if call else 0.0,
        "verification": {
            "original_status": verification["original"]["status"],
            "revision_status": verification["revision"]["status"] if verification["revision"] else None,
            "final_status": verification["status"],
            **{key: verification[key] for key in (
                "draft_accepted", "revision_requested", "revision_accepted", "revision_rejected", "fallback_used",
            )},
        } if verification else None,
    }


def _aggregate(rows):
    evaluated = [row["quality"] for row in rows if row["quality"] is not None]
    interventions = [row["verification"] for row in rows if row["verification"] is not None]
    return {
        "records": len(rows), "evaluated": len(evaluated), "statuses": dict(Counter(row["status"] for row in rows)),
        "parse_failures": sum(row["parse_failure"] for row in rows),
        "inference_failures": dict(Counter(row["inference_failure"] for row in rows if row["inference_failure"])),
        "calls": sum(row["calls"] for row in rows),
        "latency_seconds": sum(row["latency_seconds"] for row in rows),
        "quality_means": {name: mean(score[name] for score in evaluated) for name in QUALITY_METRICS}
                         if evaluated else {},
        "original_verification_statuses": dict(Counter(row["original_status"] for row in interventions)),
        "revision_statuses": dict(Counter(row["revision_status"] for row in interventions if row["revision_status"])),
        "intervention_counts": {
            key: sum(row[key] for row in interventions)
            for key in ("draft_accepted", "revision_requested", "revision_accepted", "revision_rejected", "fallback_used")
        },
    }


def _paired_metrics(pairs):
    return {
        "pairs": len(pairs),
        "metrics": {
            name: {
                "left_mean": mean(left["quality"][name] for left, _ in pairs),
                "right_mean": mean(right["quality"][name] for _, right in pairs),
                "mean_change": mean(right["quality"][name] - left["quality"][name] for left, right in pairs),
            }
            for name in QUALITY_METRICS
        } if pairs else {},
    }


def _paired(per_case, left, right):
    eligible = []
    strata = defaultdict(list)
    for rows in per_case.values():
        a, b = rows[left], rows[right]
        if a["quality"] is not None and b["quality"] is not None:
            eligible.append((a, b))
            strata[f"{a['source']} → {b['source']}"].append((a, b))
    return {
        **_paired_metrics(eligible),
        "excluded_pairs": len(per_case) - len(eligible),
        "status_pairs": dict(Counter(f"{rows[left]['status']} → {rows[right]['status']}"
                                     for rows in per_case.values())),
        "by_source": {source: _paired_metrics(pairs) for source, pairs in strata.items()},
    }


def score_experiment(cases, state):
    """Score paired condition records. Missing outputs stay missing in quality averages."""
    cases = tuple(cases)
    manifest = state["manifest"]
    if (manifest.get("cases") != case_fingerprints(cases) or manifest.get("scorer_version") != SCORER_VERSION
            or manifest.get("verifier_version") != VERIFIER_VERSION
            or manifest.get("shap_zero_tolerance") != SHAP_ZERO_TOLERANCE):
        raise ValueError("Scoring requires the checkpoint's cases and compatible verifier and scorer versions.")
    records = state["records"].values()
    indexed = {(row["case_id"], row["condition"]): row for row in records}
    expected = {(case.case_id, condition) for case in cases for condition in CONDITIONS}
    if indexed.keys() != expected or len(indexed) != len(records):
        raise ValueError("Checkpoint must contain exactly one record per case and condition.")
    per_case = {}
    for case in cases:
        grounded = indexed[case.case_id, "B"]
        verified = indexed[case.case_id, "C"].get("verification")
        if verified and verified["original_raw"] != grounded["call"]["raw"]:
            raise ValueError("Condition C must retain the exact original B draft.")
        per_case[case.case_id] = {
            condition: _score_record(case, indexed[case.case_id, condition]) for condition in CONDITIONS
        }
    scores = [row for rows in per_case.values() for row in rows.values()]
    operational = _aggregate(scores)
    if operational["calls"] != state["calls"]:
        raise ValueError("Checkpoint call count does not match recorded requests.")
    sources = defaultdict(list)
    for row in scores:
        if row["source"]:
            sources[row["condition"], row["source"]].append(row)
    return {
        "manifest": manifest, "per_case": scores, "operations": {
            key: operational[key] for key in ("records", "statuses", "parse_failures", "inference_failures",
                                              "calls", "latency_seconds", "intervention_counts")
        },
        "by_condition": {condition: _aggregate([row for row in scores if row["condition"] == condition])
                         for condition in CONDITIONS},
        "by_condition_and_source": {
            condition: {source: _aggregate(rows) for (key, source), rows in sources.items() if key == condition}
            for condition in CONDITIONS
        },
        "paired": {f"{left}_to_{right}": _paired(per_case, left, right)
                   for left, right in (("A", "B"), ("B", "C"), ("A", "C"))},
    }
