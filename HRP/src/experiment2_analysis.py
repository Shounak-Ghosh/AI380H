"""Read-only review and derived statistics for the Experiment 2 checkpoint."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from statistics import mean, median
from tempfile import TemporaryDirectory

import numpy as np

from experiment2_scoring import CONDITIONS, UNSUPPORTED, score_experiment
from experiment2_smoke import run_smoke
from llm_runner import CHECKPOINT_VERSION, Ollama, run_experiment
from llm_verification import Answer, Driver, PROMPT_VERSION
from narrative_verifier import verify_narrative


ANALYSIS_VERSION = "experiment2-analysis-v1"
BOOTSTRAP_SEED = 380
BOOTSTRAP_REPLICATES = 10_000
REVIEW_SEED = 383
PRIMARY_METRICS = ("top_three_overlap", "signed_top_three_agreement",
                   "rank_position_agreement", "exact_top_three_ordering")
STRUCTURED_METRICS = (*PRIMARY_METRICS, "missing_influential_drivers",
                      "incorrect_feature_directions", "structured_correct", "structured_complete")
REASONS = {
    "Missing explicit directional prose": {"missing_prose_drivers"},
    "Unparsed or ambiguous wording": {"unparsed", "ambiguous_subject"},
    "Incorrect SHAP direction": {"direction"},
    "Unsupported clinical interpretation": {"clinical", "causal"},
    "Numerical scale error": {"scale"},
    "Ranking error": {"ranking"},
}
INTERVENTIONS = ("draft_accepted", "revision_requested", "revision_accepted",
                 "revision_rejected", "fallback_used")


def _outside_git(path):
    if any(os.path.lexists(directory / ".git") for directory in (path, *path.parents)):
        raise ValueError("Keep Experiment 2 checkpoints and analysis outside Git.")


def read_checkpoint(path):
    """Read the original bytes once and retain their SHA-256 for derived files."""
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Use the actual checkpoint file.")
    path = path.resolve(strict=True)
    _outside_git(path.parent)
    for candidate, mode, predicate in ((path.parent, 0o700, stat.S_ISDIR),
                                       (path, 0o600, stat.S_ISREG)):
        info = candidate.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode or not predicate(info.st_mode):
            raise ValueError("Use an owner-only checkpoint (0700 directory, 0600 file).")
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _validate_run(state, provenance, cases, *, completed):
    manifest = state["manifest"]
    if (manifest["checkpoint_version"] != CHECKPOINT_VERSION
            or manifest["prompt_version"] != PROMPT_VERSION or manifest["provenance"] != provenance):
        raise ValueError("The saved run must match the expected format, prompt and pinned provenance.")
    configuration = manifest["configuration"]
    if (configuration["inference"] != "ollama" or configuration["model"]["model"] != "qwen3:8b"
            or not configuration["model"]["digest"]
            or configuration["generation"] != {"format": "json", "stream": False, "think": False}):
        raise ValueError("The checkpoint requires the recorded local Qwen3 generation configuration.")
    score = score_experiment(cases, state)
    if completed and (state["pause_reason"] is not None or score["operations"]["statuses"].get("pending", 0)):
        raise ValueError("Review requires a completed run. Resume it explicitly with real mode.")
    for row in state["records"].values():
        if row["condition"] != "C" or row["status"] != "ok":
            continue
        trace = row["verification"]
        outcomes = {
            "grounded_draft": trace["draft_accepted"],
            "model_revision": trace["revision_accepted"],
            "deterministic_fallback": trace["fallback_used"],
        }
        if (trace["status"] != "verified_complete" or sum(outcomes.values()) != 1
                or not outcomes.get(trace["source"])
                or trace["revision_requested"] != ("revision_call" in row)
                or trace["revision_call"] != row.get("revision_call")):
            raise ValueError("Incomplete verification trace in the saved run.")
    return score


def notebook_run(mode, *, checkpoint, run_directory=None, runtime_seconds=14_400):
    """Review is read-only. Only an explicit, unfinished real run can discover Ollama."""
    if mode == "fictional":
        with TemporaryDirectory(prefix="hrp-experiment2-fictional-") as directory:
            report = run_smoke(directory)
        report.pop("checkpoint")
        return {"fictional": report}
    if mode not in ("review", "real"):
        raise ValueError("Choose review, fictional, or real mode.")

    checkpoint = Path(checkpoint).expanduser()
    target = checkpoint
    if mode == "real":
        if run_directory is None or not Path(run_directory).expanduser().is_absolute():
            raise ValueError("Set a dedicated absolute private real-run directory.")
        directory = Path(run_directory).expanduser().resolve()
        _outside_git(directory)
        # A new run uses a dedicated leaf, never the original checkpoint's directory.
        if directory == checkpoint.resolve().parent or directory.name == "MIMIC-IV":
            raise ValueError("Choose a dedicated run directory separate from the original benchmark.")
        target = directory / "checkpoint.json"
        if directory.exists() and not target.exists() and any(directory.iterdir()):
            raise ValueError("Choose an empty directory for a new run.")
    saved = read_checkpoint(target) if mode == "review" or target.exists() else None

    from experiment2_inputs import load_explanation_cases

    provenance, cases = load_explanation_cases()
    if len(cases) != 72:
        raise ValueError("The real benchmark requires the pinned 72 cases.")
    if saved is not None:
        state, digest = saved
        scoring = _validate_run(state, provenance, cases, completed=mode == "review")
        if mode == "review" or not scoring["operations"]["statuses"].get("pending", 0):
            return {"state": state, "checkpoint_sha256": digest, "cases": cases, "score": scoring}

    client = Ollama.discover()
    if not client.config.digest:
        raise ValueError("Real inference requires an installed model digest.")
    run_experiment(cases, client, target.parent, provenance=provenance,
                   max_calls=3 * len(cases), runtime_seconds=runtime_seconds)
    state, digest = read_checkpoint(target)
    return {"state": state, "checkpoint_sha256": digest, "cases": cases,
            "score": _validate_run(state, provenance, cases, completed=False)}


def paired_intervals(scores, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Percentile bootstrap of right-minus-left differences, sampling whole cases."""
    if replicates < 1:
        raise ValueError("Use at least one bootstrap replicate.")
    cases = {}
    for row in scores:
        cases.setdefault(row["case_id"], {})[row["condition"]] = row["quality"]
    rng = np.random.default_rng(seed)
    results = []
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        aligned = [(row[left], row[right]) for row in cases.values()]
        complete = [(a, b) for a, b in aligned if a is not None and b is not None]
        for analysis, pairs in (("complete_case", complete), ("failure_inclusive", aligned)):
            n = len(pairs)
            draws = rng.integers(0, n, size=(replicates, n)) if n else None
            for metric in PRIMARY_METRICS:
                values = np.array([float((b or {}).get(metric, 0)) - float((a or {}).get(metric, 0))
                                   for a, b in pairs])
                interval = np.quantile(values[draws].mean(axis=1), (0.025, 0.975)).tolist() if n else [None, None]
                results.append({
                    "comparison": f"{left}→{right}", "analysis": analysis, "metric": metric,
                    "pairs": n, "observed_valid_pairs": len(complete), "excluded_pairs": len(aligned) - n,
                    "mean_difference": float(values.mean()) if n else None,
                    "ci95_low": interval[0], "ci95_high": interval[1],
                })
    return results


def _reason_codes(audit):
    if not audit.get("narrative"):
        return {audit["status"]}
    codes = {claim["code"] for claim in audit["narrative"]["claims"]
             if claim["status"] in ("ambiguous", "contradicted")}
    codes.update(issue["code"] for issue in audit["structured"]["issues"])
    if audit["missing_prose_drivers"]:
        codes.add("missing_prose_drivers")
    if audit["structured"]["missing"]:
        codes.add("missing_drivers")
    return codes


def rejection_rows(traces):
    """Count response-level categories. A response can trigger several categories."""
    stages = {
        "original B": [_reason_codes(trace["original"]) for trace in traces],
        "rejected revision": [_reason_codes(trace["revision"]) for trace in traces if trace["revision_rejected"]],
    }
    known = set().union(*REASONS.values())
    return [
        {"stage": stage, "reason": reason, "count": count, "denominator": len(codes),
         "percent": 100 * count / len(codes) if codes else None}
        for stage, codes in stages.items()
        for reason, count in [
            *((label, sum(bool(value & keys) for value in codes)) for label, keys in REASONS.items()),
            ("Other", sum(bool(value - known) for value in codes)),
        ]
    ]


def inference_costs(records):
    """Count each call once overall. B plus C describes full C delivery."""
    calls = {condition: [] for condition in CONDITIONS}
    for row in records:
        request = row.get("revision_call" if row["condition"] == "C" else "call")
        if request is not None:
            calls[row["condition"]].append(request["latency_seconds"])
    series = {**calls, "C delivery (B + C)": calls["B"] + calls["C"],
              "Total unique": [seconds for group in calls.values() for seconds in group]}
    return [{"stage": stage, "calls": len(values), "seconds": sum(values),
             "mean_seconds": mean(values) if values else None,
             "median_seconds": median(values) if values else None}
            for stage, values in series.items()]


def _narrative_counts(case, body):
    answer = Answer(body["explanation"], tuple(Driver(**driver) for driver in body["drivers"]))
    findings = verify_narrative(case, answer).claims
    categories = {
        "verified_directional_clauses": lambda c: c.code == "direction" and c.status == "supported",
        "incorrect_directional_clauses": lambda c: c.code == "direction" and c.status == "contradicted",
        "unsupported_claim_clauses": lambda c: c.code in UNSUPPORTED and c.status == "contradicted",
        "contradicted_clauses": lambda c: c.status == "contradicted",
        "checked_clauses": lambda c: c.status in ("supported", "contradicted"),
        "unparsed_clauses": lambda c: c.status == "ambiguous",
        "descriptive_clauses": lambda c: c.status == "non_attribution",
    }
    return {name: len({c.clause_index for c in findings if matches(c)}) for name, matches in categories.items()}


def select_review_examples(state, seed=REVIEW_SEED):
    """Select one private trace per criterion with fixed RNG and stable checkpoint order."""
    candidates = {}
    for row in state["records"].values():
        v = row.get("verification")
        if not v:
            continue
        audit = v["original"]
        structured, narrative = audit["structured"], audit["narrative"]
        predicates = {
            "correct structured drivers": bool(structured and structured["passed"]),
            "incorrect structured drivers": bool(structured and not structured["passed"]),
            "narrative contradiction": bool(narrative and any(c["status"] == "contradicted" for c in narrative["claims"])),
            "ambiguous prose": bool(narrative and any(c["status"] == "ambiguous" for c in narrative["claims"])),
            "accepted revision": v["revision_accepted"], "deterministic fallback": v["fallback_used"],
        }
        for label, selected in predicates.items():
            if selected:
                candidates.setdefault(label, []).append(row["case_id"])
    rng = np.random.default_rng(seed)
    return {label: sorted(ids)[int(rng.integers(len(ids)))] for label, ids in sorted(candidates.items())}


def analyze(run):
    """Derive aggregate tables and private scoring rows without modifying the run."""
    score, state = run["score"], run["state"]
    case_map = {case.case_id: case for case in run["cases"]}
    records = list(state["records"].values())
    traces = [row["verification"] for row in records if row.get("verification")]
    narrative = {condition: Counter() for condition in CONDITIONS}
    for row in records:
        if row["status"] != "ok":
            continue
        condition = row["condition"]
        body = row["verification"]["answer"] if condition == "C" else row["parse"]["answer"]
        narrative[condition].update(_narrative_counts(case_map[row["case_id"]], body))
    for row in score["per_case"]:
        if row["quality"] is not None:
            counts = narrative[row["condition"]]
            counts["missing_top_three_directions"] += row["quality"]["missing_prose_drivers"]
            counts["complete_prose_outputs"] += row["quality"]["missing_prose_drivers"] == 0

    structured, prose = [], []
    for condition in CONDITIONS:
        group = score["by_condition"][condition]
        valid, total = group["evaluated"], group["records"]
        means = group["quality_means"]
        structured.append({"condition": condition, "valid": valid, "total": total,
                           **{metric: means.get(metric) for metric in STRUCTURED_METRICS}})
        counts = narrative[condition]
        covered = counts["checked_clauses"] + counts["unparsed_clauses"]
        prose.append({"condition": condition, "valid": valid, **counts,
                      "parser_coverage": counts["checked_clauses"] / covered if covered else None,
                      "prose_completeness": counts["complete_prose_outputs"] / valid if valid else None})

    audits = [trace["original"] for trace in traces]
    grounded_audit = {
        "rejected": sum(not trace["draft_accepted"] for trace in traces),
        "structured_correct": sum(bool(a["structured"] and a["structured"]["passed"]) for a in audits),
        "structured_correct_and_ambiguous": sum(bool(a["structured"] and a["structured"]["passed"]
                                                     and _reason_codes(a) & REASONS["Unparsed or ambiguous wording"])
                                                 for a in audits),
        "responses_with_contradictions": sum(bool(a["narrative"] and any(c["status"] == "contradicted"
                                                                        for c in a["narrative"]["claims"]))
                                            for a in audits),
    }
    return {
        "operations": score["operations"], "structured": structured, "narrative": prose,
        "rejections": rejection_rows(traces), "grounded_audit": grounded_audit,
        "interventions": [{"outcome": key, "count": sum(trace[key] for trace in traces)}
                          for key in INTERVENTIONS] +
                         [{"outcome": "verified_complete", "count": sum(v["status"] == "verified_complete" for v in traces)}],
        "costs": inference_costs(records), "paired": paired_intervals(score["per_case"]),
        "by_condition_and_source": score["by_condition_and_source"],
        "private_scores": score["per_case"], "review_selection": select_review_examples(state),
    }


def save_analysis(directory, result, checkpoint_sha256, run_manifest, figures, *, manual_observations=None):
    """Write a new owner-only derived bundle. The original checkpoint is never opened for writing."""
    directory = Path(directory).expanduser().resolve()
    _outside_git(directory)
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.mkdir(mode=0o700)
    manifest = {
        "checkpoint_sha256": checkpoint_sha256, "analysis_version": ANALYSIS_VERSION,
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_version": run_manifest["prompt_version"], "verifier_version": run_manifest["verifier_version"],
        "scorer_version": run_manifest["scorer_version"], "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "manual_review": {"seed": REVIEW_SEED, "selection": "One case per criterion; overlap allowed",
                          "observations_included": manual_observations is not None},
    }
    bundles = {
        "experiment2_analysis_manifest.json": manifest,
        "experiment2_summary.json": {key: result[key] for key in
                                      ("operations", "grounded_audit", "narrative", "costs", "by_condition_and_source")},
        "experiment2_case_scores.private.json": result["private_scores"],
        "experiment2_manual_review.private.json": {
            "selection": result["review_selection"], "observations": manual_observations,
        },
    }
    for name, content in bundles.items():
        with (directory / name).open("x") as file:
            json.dump(content, file, indent=2, allow_nan=False)
    for name, key in (("condition_metrics", "structured"), ("paired_comparisons", "paired"),
                      ("interventions", "interventions"), ("verifier_diagnostics", "rejections"),
                      ("narrative_metrics", "narrative"), ("inference_costs", "costs")):
        rows = result[key]
        with (directory / f"experiment2_{name}.csv").open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    for name, figure in figures.items():
        figure.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
        figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    for file in directory.iterdir():
        file.chmod(0o600)
    return directory
