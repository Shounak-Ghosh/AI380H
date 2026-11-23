"""Frozen verifier execution and independent Experiment 4 analysis."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from llm_runner import _audit
from llm_verification import Answer, VERIFIER_VERSION
from narrative_verifier import verify_narrative
from verifier_challenge_set import (
    ERROR, FIXTURE, ROOT, blinded_worksheet, checksum, explanation_case, load_challenge,
    validate_challenge, write_json,
)

PINNED_VERSION = "experiment2-verifier-v3"
PINNED_FILES = {
    "llm_verification.py": "8587981795811045a74ae26f43d3563111cf52860734d6c66a6b3070b2d4318a",
    "narrative_verifier.py": "9fc388cf26fd1fcd0d13c11d9ac4877e2674d11e3b76d28f26bcca0889c49f78",
    "llm_runner.py": "28a32ed3c020046be2835e3a2476cf79402db3db6196f1d0dbb2d887bca3baa8",
    "hrp_faithfulness.py": "3a91cc7691bf10f3c2c72b861a7c0eadb944dec563cec5b6a4339454193496ba",
    "experiment2_inputs.py": "dc47bf586c4440f8a2ccfcddc54402c74c8a6b2ec64c5eded5976b32560f6a80",
}
ANALYSIS_VERSION = "experiment4-analysis-v1"
BOOTSTRAP_SEED = 3804
REPLICATES = 10_000
ACCEPTED, EXPLICIT, AMBIGUOUS, INCOMPLETE, SCHEMA, OTHER = (
    "accepted_as_faithful", "explicit_error_detected", "ambiguous_or_unparsed",
    "incomplete_but_otherwise_correct", "schema_invalid", "other_verification_failure",
)
CODES = {
    "direction": {"direction"}, "rank": {"ranking"}, "importance": {"negligible", "top_three", "ranking"},
    "probability": {"prediction"}, "feature": {"unsupported_feature"}, "measurement": {"observation"},
    "missingness": {"observation", "missing_observation"}, "scale": {"scale"}, "causal": {"causal"},
    "diagnosis": {"clinical"}, "conflict": {"driver_conflict"},
}


def frozen_verifier():
    actual = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in PINNED_FILES}
    if VERIFIER_VERSION != PINNED_VERSION or actual != PINNED_FILES:
        raise ValueError("The frozen Experiment 2/3 verifier version or source hashes differ.")
    return dict(version=VERIFIER_VERSION, files=actual, code_sha256=checksum(actual), commit="a51a7b1")


def verify_input(case, raw):
    """Run the same parser and release checks as the normal revision workflow."""
    trace, issues = _audit(case, {"raw": raw, "error": None})
    narrative = trace["narrative"]
    claims = narrative["claims"] if narrative else ()
    structured = trace["structured"]
    explicit = bool((structured and structured["issues"]) or
                    any(c["status"] == "contradicted" for c in claims))
    ambiguous = any(c["status"] == "ambiguous" for c in claims)
    accepted = not issues
    if trace["status"] == "parse_error":
        outcome = SCHEMA
    elif accepted:
        outcome = ACCEPTED
    elif explicit:
        outcome = EXPLICIT
    elif ambiguous:
        outcome = AMBIGUOUS
    elif trace["status"] == "correct_incomplete":
        outcome = INCOMPLETE
    else:
        outcome = OTHER
    # Removing the submitted driver list isolates the narrative layer from cross-layer agreement.
    standalone = (asdict(verify_narrative(case, Answer(trace["parse"]["answer"]["explanation"], ())))
                  if narrative else None)
    return dict(trace=trace, issues=[asdict(issue) for issue in issues], outcome=outcome,
                explicit_error=explicit, ambiguous=ambiguous, release_accepted=accepted,
                narrative_standalone=standalone)


def evaluate(bundle):
    frozen_verifier()
    validate_challenge(bundle)
    cases = {key: explanation_case(ref) for key, ref in bundle["references"].items()}
    return [dict(challenge_id=item["challenge_id"], **verify_input(cases[item["case_id"]], item["raw"]))
            for item in bundle["inputs"]]


def join_results(bundle, predictions):
    labels = {g["challenge_id"]: g for g in bundle["labels"]}
    if len(predictions) != len(labels) or {p["challenge_id"] for p in predictions} != set(labels):
        raise ValueError("Predictions do not cover the frozen challenge.")
    return [{**labels[p["challenge_id"]], **p} for p in predictions]


def bootstrap(records, statistic, *, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Resample whole profiles. Records hold profile ID and additive counts."""
    totals = defaultdict(list)
    for cluster, values in records:
        totals[cluster].append(values)
    if not totals:
        return dict(numerator=0, denominator=0, estimate=None, ci95_low=None, ci95_high=None, clusters=0)
    data = np.array([np.sum(totals[c], axis=0) for c in sorted(totals)], dtype=float)
    numerator, denominator = statistic(data.sum(axis=0))
    limits = (None, None)
    if len(data) >= 2 and denominator and replicates > 0:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(len(data), [1 / len(data)] * len(data), size=replicates)
        numerators, denominators = statistic((weights @ data).T)
        draws = numerators[denominators > 0] / denominators[denominators > 0]
        if len(draws):
            limits = tuple(float(x) for x in np.quantile(draws, (.025, .975)))
    return dict(numerator=int(numerator), denominator=int(denominator),
                estimate=float(numerator / denominator) if denominator else None,
                ci95_low=limits[0], ci95_high=limits[1], clusters=len(data))


def rate(rows, event, replicates=REPLICATES):
    return bootstrap([(r["base_reference_id"], (int(event(r)), 1)) for r in rows],
                     lambda v: (v[0], v[1]), replicates=replicates)


def _metric_functions():
    return {
        "precision": lambda tp, fp, tn, fn: (tp, tp + fp),
        "recall": lambda tp, fp, tn, fn: (tp, tp + fn),
        "specificity": lambda tp, fp, tn, fn: (tn, tn + fp),
        "false_positive_rate": lambda tp, fp, tn, fn: (fp, tn + fp),
        "false_negative_rate": lambda tp, fp, tn, fn: (fn, tp + fn),
        "f1": lambda tp, fp, tn, fn: (2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": lambda tp, fp, tn, fn: (tp * (tn + fp) + tn * (tp + fn),
                                                     2 * (tp + fn) * (tn + fp)),
    }


def binary_metrics(rows, gold, predict, *, replicates=REPLICATES):
    counts = []
    for row in rows:
        actual, predicted = bool(gold(row)), bool(predict(row))
        index = {(True, True): 0, (False, True): 1, (False, False): 2, (True, False): 3}[actual, predicted]
        counts.append((row["base_reference_id"], tuple(int(i == index) for i in range(4))))
    totals = np.sum([v for _, v in counts], axis=0) if counts else [0] * 4
    return {"n": len(rows), "confusion": dict(zip(("TP", "FP", "TN", "FN"), map(int, totals), strict=True)),
            "metrics": {key: bootstrap(counts, lambda values: fn(*values), replicates=replicates)
                        for key, fn in _metric_functions().items()}}


def primary_cohort(rows, include_boundary=False):
    return [r for r in rows if (include_boundary or r["group"] in "AB")
            and r["schema_validity"] == "valid" and r["faithfulness"] in ("faithful", "unfaithful")]


def _policy_results(rows, replicates):
    gold = lambda r: r["faithfulness"] == "unfaithful"
    return {key: binary_metrics(rows, gold, predicate, replicates=replicates)
            for key, predicate in {
                "explicit": lambda r: r["explicit_error"],
                "release": lambda r: not r["release_accepted"],
            }.items()}


def _paraphrase_pairs(faithful, replicates):
    by_id = {r["challenge_id"]: r for r in faithful}
    pairs = defaultdict(list)
    for row in faithful:
        control = by_id[row["direct_control_id"]]
        if row["challenge_id"] == control["challenge_id"]:
            continue
        pair = dict(control_id=control["challenge_id"], paraphrase_id=row["challenge_id"],
                    style=row["style"], base_reference_id=row["base_reference_id"],
                    direct_accepted=control["release_accepted"], paraphrase_accepted=row["release_accepted"])
        pairs[row["style"]].append(pair)
    return [
        dict(style=style, pairs=matched, discordant=sum(p["direct_accepted"] != p["paraphrase_accepted"]
                                                       for p in matched),
             **bootstrap([(p["base_reference_id"], (int(p["direct_accepted"]) - int(p["paraphrase_accepted"]), 1))
                          for p in matched], lambda v: (v[0], v[1]), replicates=replicates))
        for style, matched in sorted(pairs.items())
    ]


def _category_outcome(row, category, layer=None):
    code_key = next(key for key, name in ERROR.items() if name == category)
    codes = CODES[code_key]
    structured = row["trace"]["structured"]
    narrative = row["trace"]["narrative"]
    gold_claims = [c for c in row["claims"] if category in c["errors"] and c["status"] == "unfaithful"]
    for c in gold_claims:
        if layer is not None and c["layer"] != layer:
            continue
        checks = (structured or {}).get("issues", ()) if c["layer"] == "structured" and code_key != "conflict" else [
            n for n in (narrative or {}).get("claims", ()) if n["status"] == "contradicted"
            and (not c["feature"] or not n["feature"] or n["feature"] == c["feature"])
        ]
        if any(check["code"] in codes for check in checks):
            return "explicitly_detected"
    if row["release_accepted"]:
        return "incorrectly_accepted"
    return "ambiguous_rejection" if row["ambiguous"] else "missed_other_rejection"


def _error_matrix(unfaithful, replicates):
    categories = sorted({e for r in unfaithful for e in r["errors"] if e in {ERROR[k] for k in CODES}})
    cohorts = {category: [r for r in unfaithful if category in r["errors"]] for category in categories}
    for recipe in ("compound_direction", "relative_direction", "passive_direction", "both_layers"):
        cohorts[recipe] = [r for r in unfaithful if r["mutation"]["recipe"] == recipe]
    cohorts["both_layers_prose_direction"] = cohorts["both_layers"]
    result = []
    for category, cohort in cohorts.items():
        criterion = category if category in categories else ERROR["direction"]
        layer = "prose" if category == "both_layers_prose_direction" else None
        outcomes = {r["challenge_id"]: _category_outcome(r, criterion, layer) for r in cohort}
        result.append(dict(category=category, n=len(cohort), **{
            event: rate(cohort, lambda r: outcomes[r["challenge_id"]] == event, replicates)
            for event in ("explicitly_detected", "ambiguous_rejection", "incorrectly_accepted", "missed_other_rejection")
        }))
    return result


def _layers(rows, replicates):
    registry = {}
    checks = {
        "structured": (
            lambda r: any(c["layer"] == "structured" and c["status"] == "unfaithful" for c in r["claims"]),
            lambda r: bool(r["trace"]["structured"]["issues"]),
        ),
        "narrative": (
            lambda r: any(c["layer"] == "prose" and c["status"] == "unfaithful" for c in r["claims"]),
            lambda r: any(c["status"] == "contradicted" for c in r["narrative_standalone"]["claims"]),
        ),
    }
    for key, (gold, predicted) in checks.items():
        registry[key] = binary_metrics(rows, gold, predicted, replicates=replicates)
    membership = lambda r: ERROR["importance"] in {
        e for c in r["claims"] if c["layer"] == "structured" for e in c["errors"]}
    missing = lambda r: any(c["layer"] == "structured" and c["status"] == "omitted" for c in r["claims"])
    registry["structured_checks"] = {
        "membership_accuracy": rate(rows, lambda r: membership(r) ==
                                    any(i["code"] == "top_three" for i in r["trace"]["structured"]["issues"]), replicates),
        "completeness_accuracy": rate(rows, lambda r: missing(r) == (not r["trace"]["structured"]["complete"]), replicates),
    }
    for kind, code in (("direction", "direction"), ("rank", "ranking"), ("membership", "top_three")):
        subset = [r for r in rows if any(c["layer"] == "structured" and c["status"] == "unfaithful"
                                         and ERROR["importance" if kind == "membership" else kind] in c["errors"]
                                         for c in r["claims"])]
        registry["structured_checks"][f"{kind}_error_recall"] = rate(
            subset, lambda r: any(i["code"] == code for i in r["trace"]["structured"]["issues"]), replicates)
    clause_counts = []
    for row in rows:
        clauses = defaultdict(set)
        for c in row["narrative_standalone"]["claims"]:
            clauses[c["clause_index"]].add(c["status"])
        checked = sum(bool(states & {"supported", "contradicted"}) for states in clauses.values())
        unknown = sum("ambiguous" in states for states in clauses.values())
        clause_counts.append((row["base_reference_id"], (checked, checked + unknown)))
    registry["narrative_checks"] = {
        "parser_coverage": bootstrap(clause_counts, lambda v: (v[0], v[1]), replicates=replicates),
        "ambiguity_per_response": rate(rows, lambda r: any(c["status"] == "ambiguous"
                                       for c in r["narrative_standalone"]["claims"]), replicates),
    }
    for kind in ("causal", "diagnosis"):
        subset = [r for r in rows if ERROR[kind] in r["errors"]]
        registry["narrative_checks"][f"{kind}_recall"] = rate(subset, lambda r: any(
            c["status"] == "contradicted" and c["code"] in CODES[kind]
            for c in r["narrative_standalone"]["claims"]), replicates)
    registry["layer_controls"] = dict(Counter(
        f"structured_{'wrong' if checks['structured'][0](r) else 'correct'}__"
        f"prose_{'wrong' if checks['narrative'][0](r) else 'correct'}" for r in rows))
    return registry


def analyze(bundle, predictions, *, replicates=REPLICATES):
    rows = join_results(bundle, predictions)
    primary = primary_cohort(rows)
    faithful = [r for r in primary if r["group"] == "A"]
    unfaithful = [r for r in primary if r["group"] == "B"]
    strata = {"all": faithful, **{style: [r for r in faithful if r["style"] == style]
                                 for style in sorted({r["style"] for r in faithful})}}
    for category in sorted({r["linguistic_construction"] for r in faithful}):
        strata["category_" + category] = [r for r in faithful if r["linguistic_construction"] == category]
    rejection = [dict(style=key, n=len(stratum), **{
        name: rate(stratum, predicate, replicates)
        for name, predicate in {
            "accepted": lambda r: r["release_accepted"],
            "false_rejection": lambda r: not r["release_accepted"],
            "ambiguous_rejection": lambda r: r["outcome"] == AMBIGUOUS,
            "missing_parsed_directions": lambda r: bool(r["trace"].get("missing_prose_drivers")),
            "incorrect_explicit_classification": lambda r: r["explicit_error"],
        }.items()}) for key, stratum in strata.items()]
    return {
        "bootstrap": dict(seed=BOOTSTRAP_SEED, replicates=replicates, unit="paired_profile"),
        "primary": _policy_results(primary, replicates),
        "all_eligible_sensitivity": _policy_results(primary_cohort(rows, True), replicates),
        "false_rejection": rejection, "paraphrases": _paraphrase_pairs(faithful, replicates),
        "error_matrix": _error_matrix(unfaithful, replicates), "layers": _layers(primary, replicates),
        "boundary": [dict(faithfulness=key[0], completeness=key[1], schema=key[2], outcome=key[3], n=n)
                     for key, n in sorted(Counter((r["faithfulness"], r["completeness"], r["schema_validity"],
                                                  r["outcome"]) for r in rows if r["group"] == "C").items())],
        "outcomes": {group: dict(Counter(r["outcome"] for r in rows if r["group"] == group))
                     for group in "ABC"},
        "multi_error_cases": [r["challenge_id"] for r in unfaithful if len(
            [c for c in r["claims"] if c["status"] == "unfaithful"]) > 1],
        "examples": [dict(challenge_id=r["challenge_id"], gold=r["faithfulness"],
                          explanation=r["original_explanation"], outcome=r["outcome"],
                          issues=sorted({i["code"] for i in r["issues"]}))
                     for pool, condition in ((faithful, lambda r: not r["release_accepted"]),
                                             (unfaithful, lambda r: not r["explicit_error"]))
                     for r in [next((r for r in pool if condition(r)), None)] if r],
    }


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def save_results(directory, bundle, predictions, analysis, corrections, plan):
    from verifier_evaluation_figures import save_figures

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    worksheet, crosswalk = blinded_worksheet(bundle)
    artifacts = {"predictions.json": predictions, "analysis.json": analysis, "scripted_corrections.json": corrections,
                 "live_selection.json": plan, "blinded_worksheet.json": worksheet, "blinded_crosswalk.json": crosswalk}
    for name, data in artifacts.items():
        write_json(directory / name, data)
    figures = save_figures(directory, analysis, corrections)
    sources = ("verifier_challenge_set", "verifier_evaluation", "verifier_evaluation_figures", "verifier_correction")
    manifest = dict(
        version=ANALYSIS_VERSION, frozen_verifier=frozen_verifier(), challenge_sha256=checksum(bundle),
        dataset_version=bundle["version"], seed=bundle["seed"], bootstrap=analysis["bootstrap"],
        rubric_sha256=_sha(ROOT / "rubric.md"),
        code_sha256={key: _sha(Path(__file__).with_name(key + ".py")) for key in sources},
        software={"python": platform.python_version(), **{
            p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "requests", "pandas")}},
        files={name: _sha(directory / name) for name in [*artifacts, *figures]},
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def review(directory=FIXTURE / "results", bundle=None):
    directory = Path(directory)
    bundle = load_challenge() if bundle is None else bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["version"] != ANALYSIS_VERSION or manifest["frozen_verifier"] != frozen_verifier()
            or manifest["challenge_sha256"] != checksum(bundle) or manifest["rubric_sha256"] != _sha(ROOT / "rubric.md")):
        raise ValueError("Saved results do not match the frozen design or verifier.")
    for name, expected in manifest["files"].items():
        if _sha(directory / name) != expected:
            raise ValueError(f"Unvalidated saved artifact: {name}")
    return {key: json.loads((directory / filename).read_text()) for key, filename in (
        ("analysis", "analysis.json"), ("predictions", "predictions.json"),
        ("corrections", "scripted_corrections.json"), ("live_plan", "live_selection.json"))} | {"manifest": manifest}


def notebook_run(mode, *, directory=FIXTURE / "results", save=False, live_directory=None, max_calls=30):
    from verifier_correction import live_revision, scripted_corrections, select_live

    bundle = load_challenge()
    result = dict(bundle=bundle, validation=validate_challenge(bundle), frozen=frozen_verifier())
    if mode == "setup":
        return result
    if mode == "fictional":
        return dict(result, corrections=scripted_corrections(bundle))
    if mode == "review":
        return dict(result, **review(directory, bundle))
    if mode == "live_revision":
        if live_directory is None:
            raise ValueError("Set an owner-only Experiment 4 live directory outside Git.")
        plan = review(directory, bundle)["live_plan"]
        return dict(result, live=live_revision(bundle, plan, live_directory, max_calls=max_calls))
    if mode != "evaluate":
        raise ValueError("Choose setup, review, evaluate, fictional, or live_revision.")
    predictions = evaluate(bundle)
    analysis = analyze(bundle, predictions)
    corrections = scripted_corrections(bundle)
    plan = select_live(bundle, predictions)
    result.update(predictions=predictions, analysis=analysis, corrections=corrections, live_plan=plan)
    if save:
        result["manifest"] = save_results(directory, bundle, predictions, analysis, corrections, plan)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "review", "evaluate", "fictional", "live_revision"))
    parser.add_argument("--directory", type=Path, default=FIXTURE / "results")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--live-directory", type=Path)
    parser.add_argument("--max-calls", type=int, default=30)
    args = parser.parse_args()
    result = notebook_run(args.mode, directory=args.directory, save=args.save,
                          live_directory=args.live_directory, max_calls=args.max_calls)
    print(json.dumps({"validation": result["validation"],
                      **({"live": result["live"]["summary"]} if "live" in result else {}),
                      **({"outcomes": result["analysis"]["outcomes"]} if "analysis" in result else {})}, indent=2))
