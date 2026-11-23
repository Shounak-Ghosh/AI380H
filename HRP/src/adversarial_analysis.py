"""Matched H3 comparisons and explicit feature-level claim outcomes."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import re
from statistics import mean, stdev

import numpy as np

from adversarial_scenarios import FAMILIES, VARIANTS, explanation_cases, validate_manifest
from experiment2_scoring import CONDITIONS, score_experiment
from llm_runner import verify_and_revise
from llm_verification import Answer, Driver, deterministic_answer, verify_attribution
from narrative_verifier import _aliases, verify_narrative


ANALYSIS_VERSION = "experiment3-analysis-v2"
BOOTSTRAP_SEED = 380
BOOTSTRAP_REPLICATES = 10_000
ATTRIBUTES = ("mixed_direction", "conflicting_importance")
CLAIM_OUTCOMES = ("model_faithful", "clinical_expectation_substitution",
                  "other_incorrect", "abstention", "unverifiable")
SWITCH_OUTCOMES = ("correct_switch", "retained_intuitive_direction", "other_incorrect",
                   "abstention", "unverifiable")
VERIFICATION_FLAGS = ("draft_accepted", "revision_requested", "revision_accepted",
                      "revision_rejected", "fallback_used")


def answer_body(record):
    if record["status"] != "ok":
        return None
    if record["condition"] == "C":
        return record["verification"]["answer"]
    return record["parse"]["answer"]


def _answer(body):
    return Answer(body["explanation"], tuple(Driver(**row) for row in body["drivers"]))


def _opposite(direction):
    return "negative" if direction == "positive" else "positive"


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def feature_claim(case, answer, narrative, reference, intuitive, status):
    """Use explicit structured drivers and recognized prose, retaining evidence by channel."""
    feature = reference["feature"]
    expected = reference["direction"]
    structured = next((driver.direction for driver in answer.drivers if driver.feature == feature), None) if answer else None
    aliases = "|".join(re.escape(alias) for alias, key in _aliases(case).items() if key == feature)
    mentions = re.compile(rf"(?<!\w)(?:{aliases})(?!\w)", re.I)
    claims = narrative.claims if narrative else ()
    prose = set()
    ambiguous = False
    for claim in claims:
        if claim.feature == feature and claim.code == "direction":
            prose.add(expected if claim.status == "supported" else _opposite(expected))
        if claim.feature == feature and claim.code == "driver_conflict" and structured:
            prose.add(_opposite(structured))
        if claim.status == "ambiguous" and mentions.search(claim.text):
            ambiguous = True
    mentioned = bool(structured or (answer and mentions.search(answer.explanation)))
    directions = prose | ({structured} if structured else set())
    explicit_abstention = bool(any(claim.code == "abstention" for claim in claims))
    if not answer:
        structured_outcome = "unverifiable"
    elif structured is None:
        structured_outcome = "abstention"
    elif structured == expected:
        structured_outcome = "model_faithful"
    elif structured == intuitive:
        structured_outcome = "clinical_expectation_substitution"
    else:
        structured_outcome = "other_incorrect"
    if not answer or ambiguous:
        outcome = "unverifiable"
    elif len(directions) > 1:
        outcome = "other_incorrect"
    elif directions == {expected}:
        outcome = "model_faithful"
    elif directions == {intuitive}:
        outcome = "clinical_expectation_substitution"
    elif directions:
        outcome = "other_incorrect"
    elif mentioned:
        outcome = "unverifiable"
    else:
        outcome = "abstention"
    return {
        "feature": feature, "status": status, "reference_direction": expected,
        "expected_intuitive_direction": intuitive, "rank": reference["rank"],
        "reference_top_three": reference["rank"] <= 3, "mentioned": mentioned,
        "structured_direction": structured, "prose_directions": sorted(prose),
        "claimed_direction": next(iter(directions)) if len(directions) == 1 else None,
        "matches_shap": expected in directions if directions else None,
        "matches_intuitive": intuitive in directions if directions else None,
        "explicit_global_abstention": explicit_abstention,
        "ambiguous_feature_prose": ambiguous, "outcome": outcome,
        "structured_outcome": structured_outcome,
    }


def _coverage(narrative):
    def clauses(*codes, status="contradicted"):
        return len({c.clause_index for c in narrative.claims
                    if c.status == status and (not codes or c.code in codes)})

    checked = len({c.clause_index for c in narrative.claims if c.status in ("supported", "contradicted")})
    unparsed = clauses(status="ambiguous")
    return {
        "incorrect_prose_directions": clauses("direction"),
        "structured_prose_conflicts": clauses("driver_conflict"),
        "unsupported_clinical_or_causal": clauses("clinical", "causal"),
        "parser_coverage": _fraction(checked, checked + unparsed),
        "checked_clauses": checked, "ambiguous_clauses": unparsed,
    }


def _switch(left, right):
    if "unverifiable" in (left, right):
        return "unverifiable"
    if "abstention" in (left, right):
        return "abstention"
    if left == right == "model_faithful":
        return "correct_switch"
    if left == "model_faithful" and right == "clinical_expectation_substitution":
        return "retained_intuitive_direction"
    return "other_incorrect"


def paired_intervals(rows, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Share bootstrap draws across conditions and both variants of each profile."""
    if replicates < 1:
        raise ValueError("Bootstrap requires positive replicates.")
    groups = defaultdict(dict)
    for row in rows:
        profile = groups[row["profile_id"]]
        profile[row["condition"], row["variant"]] = row
    rng = np.random.default_rng(seed)
    result = []
    selectors = {
        "all": lambda pair: True,
        **{family: lambda pair, f=family: pair[0]["family"] == f for family in FAMILIES},
        **{f"{attribute}={level}": lambda pair, a=attribute, l=level:
           any(row["attributes"][a] for row in pair) == l
           for attribute in ATTRIBUTES for level in (False, True)},
    }
    metrics = ("signed_top_three_agreement", "exact_top_three_ordering",
               "unsupported_model_claims", "narrative_contradictions",
               "substitution_rate", "verification_intervention", "fallback_used")
    for stratum, select in selectors.items():
        profiles = [key for key in sorted(groups) if select([groups[key]["A", v] for v in VARIANTS])]
        draws = rng.integers(len(profiles), size=(replicates, len(profiles))) if profiles else None
        for condition in CONDITIONS:
            pairs = [[groups[key][condition, variant] for variant in VARIANTS] for key in profiles]
            for metric in metrics:
                if condition != "C" and metric in ("verification_intervention", "fallback_used"):
                    continue
                policies = ("valid_pairs", "failed_outputs_zero") if metric in metrics[:2] else ("valid_pairs",)
                for policy in policies:
                    diffs = []
                    for left, right in pairs:
                        a, b = left.get(metric), right.get(metric)
                        if a is not None and b is not None:
                            diffs.append(b - a)
                        elif policy == "failed_outputs_zero" and "pending" not in (left["status"], right["status"]):
                            diffs.append(float(b or 0) - float(a or 0))
                        else:
                            diffs.append(np.nan)
                    included = sum(not np.isnan(x) for x in diffs)
                    estimate = low = high = None
                    if included:
                        estimate = float(np.nanmean(diffs))
                        sample = np.asarray(diffs)[draws]
                        usable = sample[~np.isnan(sample).all(axis=1)]
                        low, high = np.quantile(np.nanmean(usable, axis=1), [.025, .975]).tolist() if len(usable) else (None, None)
                    result.append({
                        "stratum": stratum, "condition": condition, "metric": metric, "policy": policy,
                        "pairs": included, "selected_pairs": len(pairs), "mean_difference": estimate,
                        "ci95_low": low, "ci95_high": high,
                    })
    return result


def audit_fictional_claims(case):
    """Hand-labeled probes kept separate from the experimental Qwen3 estimates."""
    good = deterministic_answer(case)
    first = good.drivers[0]
    correct = "increases" if first.direction == "positive" else "decreases"
    incorrect = "decreases" if correct == "increases" else "increases"
    clause = f"{first.feature} {correct} the model's predicted risk"
    bad_clause = clause.replace(correct, incorrect)
    outside = next(ref["feature"] for ref in case.shap["contributions"] if ref["rank"] == 4)
    probes = (
        ("correct control", good, False),
        ("wrong direction in prose", replace(good, explanation=good.explanation.replace(clause, bad_clause)), True),
        ("wrong structured direction", replace(good, drivers=(
            Driver(first.feature, _opposite(first.direction)), *good.drivers[1:])), True),
        ("outside top three called strongest", replace(good, explanation=f"{outside} is the strongest model driver."), True),
        ("compound contradiction", replace(good, explanation=f"{clause}, while {bad_clause}."), True),
        ("relative clause contradiction", replace(
            good, explanation=f"{first.feature}, which {incorrect} the model's predicted risk, is the strongest driver."), True),
        ("unsupported causation", replace(good, explanation=f"{first.feature} causes readmission."), True),
        ("missing directions", replace(good, explanation="The model contributions are unknown."), False),
    )
    expected_raw = json.dumps(asdict(good))
    rows = []
    for label, answer, expected_error in probes:
        narrative = verify_narrative(case, answer)
        structured = verify_attribution(case, answer.drivers)
        detected = any(c.status == "contradicted" for c in narrative.claims) or not structured.passed
        ambiguous = any(c.status == "ambiguous" for c in narrative.claims)
        raw = json.dumps(asdict(answer))
        trace = verify_and_revise(case, raw, lambda *_: expected_raw)
        fallback = verify_and_revise(case, raw, lambda *_: raw)
        rows.append({
            "probe": label, "expected_error": expected_error, "detected": detected,
            "missed": expected_error and not detected and not ambiguous,
            "ambiguous": ambiguous, "draft_accepted": trace["draft_accepted"],
            "corrected_with_scripted_revision": trace["revision_accepted"],
            "corrected_with_fallback": fallback["fallback_used"],
            "explanation": answer.explanation,
        })
    return rows


def analyze(manifest, state, *, replicates=BOOTSTRAP_REPLICATES):
    validate_manifest(manifest)
    cases = {case.case_id: case for case in explanation_cases(manifest)}
    if not set(cases) == {row["case_id"] for row in state["records"].values()}:
        raise ValueError("Analysis requires the checkpoint's exact scenario manifest.")
    scores = score_experiment(cases.values(), state)
    metrics = {(row["case_id"], row["condition"]): row for row in scores["per_case"]}
    indexed = {(row["case_id"], row["condition"]): row for row in state["records"].values()}
    rows, claims, verification = [], [], []
    group = {}
    for reference in manifest["records"]:
        case = cases[reference["case_id"]]
        for condition in CONDITIONS:
            key = (case.case_id, condition)
            record, score = indexed[key], metrics[key]
            body = answer_body(record)
            answer = _answer(body) if body else None
            narrative = verify_narrative(case, answer) if answer else None
            identity = {k: reference[k] for k in ("case_id", "profile_id", "family", "variant")}
            identity["condition"] = condition
            targets = [feature_claim(case, answer, narrative, ref,
                                     manifest["intuitive_expectations"][ref["feature"]], record["status"])
                       for ref in reference["shap"]["contributions"]
                       if ref["feature"] in reference["targeted_features"]]
            for target in targets:
                row = {**identity, **target}
                claims.append(row)
                group[row["profile_id"], condition, row["feature"], row["variant"]] = row
            row = {**identity, **score, **(score["quality"] or {}),
                   **(_coverage(narrative) if narrative else {}),
                   "attributes": reference["attributes"],
                   "substitution_rate": (mean(t["outcome"] == "clinical_expectation_substitution"
                                              for t in targets) if answer else None)}
            if trace := record.get("verification"):
                original = trace["original"]
                issues = [i["code"] for i in (original.get("structured") or {}).get("issues", [])]
                failures = (original.get("narrative") or {}).get("claims", [])
                codes = sorted(set(issues + [c["code"] for c in failures if c["status"] == "contradicted"]))
                row.update(verification_intervention=not trace["draft_accepted"], fallback_used=trace["fallback_used"])
                verification.append({
                    **identity, **{flag: trace[flag] for flag in VERIFICATION_FLAGS},
                    "source": trace["source"], "verified_complete": trace["status"] == "verified_complete",
                    "original_detected_errors": codes,
                    "original_ambiguous_codes": sorted({c["code"] for c in failures if c["status"] == "ambiguous"}),
                    "original_explicit_substitutions": sum(
                        group[reference["profile_id"], "B", t["feature"], reference["variant"]]["outcome"]
                        == "clinical_expectation_substitution" for t in targets),
                })
            rows.append(row)
    switches = []
    for (profile, condition, feature, variant), left in group.items():
        if variant != "intuitive":
            continue
        right = group[profile, condition, feature, "counterintuitive"]
        if left["reference_direction"] != "positive" or right["reference_direction"] != "negative":
            raise ValueError("Model-switch analysis requires a validated local sign reversal.")
        switches.append({
            "profile_id": profile, "condition": condition, "feature": feature, "family": left["family"],
            "intuitive_direction": left["claimed_direction"], "counterintuitive_direction": right["claimed_direction"],
            "top_three_in_both": left["reference_top_three"] and right["reference_top_three"],
            "outcome": _switch(left["outcome"], right["outcome"]),
            "structured_outcome": _switch(left["structured_outcome"], right["structured_outcome"]),
        })
    table, attributes, interventions, error_rates = [], [], [], []
    claim_rates, switch_rates, structured_claim_rates, structured_switch_rates = [], [], [], []
    for condition in CONDITIONS:
        for variant in VARIANTS:
            subset = [row for row in rows if (row["condition"], row["variant"]) == (condition, variant)]
            table.append({"condition": condition, "variant": variant, **_summarize(subset)})
            for attribute in ATTRIBUTES:
                for level in (False, True):
                    attributes.append({"condition": condition, "variant": variant, "attribute": attribute, "present": level,
                                       **_summarize([r for r in subset if r["attributes"][attribute] == level])})
            for top_only in (False, True):
                eligible = [r for r in claims if (r["condition"], r["variant"]) == (condition, variant)
                            and (r["reference_top_three"] or not top_only)]
                keys = {"condition": condition, "variant": variant, "top_three_only": top_only}
                claim_rates.extend(_rates(eligible, CLAIM_OUTCOMES, **keys))
                structured_claim_rates.extend(_rates(eligible, CLAIM_OUTCOMES, field="structured_outcome", **keys))
        for top_only in (False, True):
            eligible = [r for r in switches if r["condition"] == condition and (r["top_three_in_both"] or not top_only)]
            keys = {"condition": condition, "top_three_in_both_only": top_only}
            switch_rates.extend(_rates(eligible, SWITCH_OUTCOMES, **keys))
            structured_switch_rates.extend(_rates(eligible, SWITCH_OUTCOMES, field="structured_outcome", **keys))
    for variant in VARIANTS:
        subset = [row for row in verification if row["variant"] == variant]
        intervened = sum(not row["draft_accepted"] for row in subset)
        interventions.append({
            "variant": variant, "completed": len(subset), "interventions": intervened,
            **{flag: sum(r[flag] for r in subset) for flag in (*VERIFICATION_FLAGS, "verified_complete")},
        })
        for code in sorted({code for row in subset for code in row["original_detected_errors"]}):
            observed = [r for r in subset if code in r["original_detected_errors"]]
            error_rates.append({
                "variant": variant, "detected_code": code, "detected": len(observed),
                "accepted_revision_rate": mean(r["revision_accepted"] for r in observed),
                "fallback_rate": mean(r["fallback_used"] for r in observed),
                "verified_complete_rate": mean(r["verified_complete"] for r in observed),
            })
    return {
        "version": ANALYSIS_VERSION, "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": replicates},
        "operations": scores["operations"], "faithfulness": table, "attributes": attributes,
        "case_metrics": rows, "feature_claims": claims, "claim_rates": claim_rates,
        "switches": switches, "switch_rates": switch_rates, "verification_traces": verification,
        "structured_claim_rates": structured_claim_rates, "structured_switch_rates": structured_switch_rates,
        "interventions": interventions, "verifier_error_classes": error_rates,
        "prediction_changes": _prediction_changes(manifest),
        "paired": paired_intervals(rows, replicates=replicates), "examples": _examples(manifest, indexed, claims, switches, verification),
    }


def _prediction_changes(manifest):
    profiles = defaultdict(dict)
    for row in manifest["records"]:
        profiles[row["profile_id"]][row["variant"]] = row
    changes = []
    for profile, pair in profiles.items():
        initial, reversed_model = (pair[v] for v in VARIANTS)
        counter = {ref["feature"]: ref["shap"] for ref in reversed_model["shap"]["contributions"]}
        changes.append({
            "profile_id": profile, "family": initial["family"],
            "intuitive_probability": initial["predicted_risk"],
            "counterintuitive_probability": reversed_model["predicted_risk"],
            "probability_change": reversed_model["predicted_risk"] - initial["predicted_risk"],
            "margin_change": reversed_model["margin"] - initial["margin"],
            "shap_changes": {ref["feature"]: counter[ref["feature"]] - ref["shap"]
                             for ref in initial["shap"]["contributions"]},
        })
    return changes


def predicted_risk_sensitivity(result):
    """Summarize complete-pair A/B contrasts by fixed absolute probability bands."""
    scores = {(r["profile_id"], r["condition"], r["variant"]): r
              for r in result["case_metrics"]}
    bands = ("<15 pp", "15–<25 pp", "≥25 pp")
    pairs = []
    for shift in sorted(result["prediction_changes"], key=lambda r: r["profile_id"]):
        profile = shift["profile_id"]
        initial, switched = (shift[f"{variant}_probability"] for variant in VARIANTS)
        change = switched - initial
        absolute_change = abs(change)
        attributes = {
            attribute: {variant: scores[profile, "A", variant]["attributes"][attribute]
                        for variant in VARIANTS}
            for attribute in ATTRIBUTES
        }
        pair = {
            "profile_id": profile, "family": shift["family"],
            "intuitive_probability": initial, "counterintuitive_probability": switched,
            "probability_change": change, "absolute_probability_change": absolute_change,
            "band": bands[0] if absolute_change < .15 else bands[1] if absolute_change < .25 else bands[2],
            **attributes,
        }
        for condition in ("A", "B"):
            left, right = (scores[profile, condition, variant] for variant in VARIANTS)
            pair[f"{condition}_difference"] = (
                right["signed_top_three_agreement"] - left["signed_top_three_agreement"]
                if left["status"] == right["status"] == "ok" else None)
        pairs.append(pair)

    strata = []
    for band in bands:
        selected = [pair for pair in pairs if pair["band"] == band]
        for condition in ("A", "B"):
            values = [pair[f"{condition}_difference"] for pair in selected
                      if pair[f"{condition}_difference"] is not None]
            strata.append({
                "band": band, "condition": condition, "selected_pairs": len(selected),
                "complete_pairs": len(values), "mean_difference": mean(values) if values else None,
                "paired_difference_sd": stdev(values) if len(values) > 1 else None,
            })
    return {"pairs": pairs, "strata": strata}


def _rates(rows, outcomes, *, field="outcome", **group):
    counts = Counter(r[field] for r in rows)
    return [{**group, "outcome": outcome, "count": counts[outcome], "denominator": len(rows),
             "rate": _fraction(counts[outcome], len(rows))} for outcome in outcomes]


def _summarize(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    checked = sum(r["checked_clauses"] for r in valid)
    ambiguous = sum(r["ambiguous_clauses"] for r in valid)
    return {
        "total": len(rows), "valid": len(valid),
        **{metric: mean(r[metric] for r in valid) if valid else None for metric in (
            "signed_top_three_agreement", "exact_top_three_ordering", "structured_correct",
            "structured_complete", "missing_prose_drivers", "structured_prose_conflicts",
            "unsupported_clinical_or_causal", "ambiguous_clauses")},
        "incorrect_direction_rate": mean(r["incorrect_feature_directions"] > 0 for r in valid) if valid else None,
        "incorrect_prose_direction_rate": mean(r["incorrect_prose_directions"] > 0 for r in valid) if valid else None,
        "unsupported_model_claim_rate": mean(r["unsupported_model_claims"] > 0 for r in valid) if valid else None,
        "prose_completeness": mean(r["missing_prose_drivers"] == 0 for r in valid) if valid else None,
        "parser_coverage": _fraction(checked, checked + ambiguous),
    }


def _examples(manifest, indexed, claims, switches, verification):
    choices = {}
    for row in claims:
        if row["condition"] == "B" and row["outcome"] == "clinical_expectation_substitution":
            choices.setdefault("Explicit intuitive-direction substitution", row["profile_id"])
    for row in verification:
        if row["revision_accepted"] or row["fallback_used"]:
            choices.setdefault("Grounded draft verification intervention", row["profile_id"])
    for row in switches:
        if row["condition"] == "B" and row["outcome"] == "correct_switch":
            choices.setdefault("Grounded explanation follows the reference switch", row["profile_id"])
    completed = Counter(row["profile_id"] for row in manifest["records"]
                        if all(indexed[row["case_id"], c]["status"] == "ok" for c in CONDITIONS))
    for profile, count in completed.items():
        if count == len(VARIANTS):
            choices.setdefault("Matched fictional model comparison", profile)
            break
    seen, examples = set(), []
    for reason, profile in choices.items():
        if profile in seen or len(examples) == 2:
            continue
        seen.add(profile)
        examples.append({
            "reason": reason, "profile_id": profile,
            "variants": [{**row, "explanations": {
                c: answer_body(indexed[row["case_id"], c]) for c in CONDITIONS}}
                for row in manifest["records"] if row["profile_id"] == profile],
        })
    return examples
