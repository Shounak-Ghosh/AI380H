"""Read-only validation and aggregate reporting for the five frozen HRP studies."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path.home() / "Library/Application Support/AI380H/MIMIC-IV"
VERSION = "results-synthesis-v1"
MODES = ("review", "validate", "export", "fictional")
HEX = re.compile(r"[0-9a-f]{64}")
ARMS = ("A", "B", "C")
SIGNED = "signed_top_three_agreement"
FAITH = "mathematical_faithfulness"
READ = "readability_1_to_5"
CLEAR = "clarity_1_to_5"
COMPARISONS = ("B − A", "C − B", "C − A")
UNITS = {
    1: ("MIMIC-IV v3.1", "admission (patient-clustered test uncertainty)", "prediction model and SHAP reference"),
    2: ("MIMIC-IV v3.1 risk-stratified sample", "matched admission", "baseline, grounded draft and verified delivery"),
    3: ("Fictional attribution reversals", "matched fictional profile", "intuitive and counterintuitive model variants"),
    4: ("Fictional verifier challenge", "challenge response (profile-clustered)", "detection, release and correction"),
    5: ("MIMIC-IV v3.1 selected triplets", "matched admission", "single-reviewer faithfulness and text quality"),
}
E2_TABLES = {
    "condition_metrics": "structured", "paired_comparisons": "paired",
    "interventions": "interventions", "narrative_metrics": "narrative",
    "inference_costs": "costs", "verifier_diagnostics": "rejections",
}
PENDING = "Intra-reviewer consistency analysis pending."


class ValidationError(ValueError):
    """An unavailable, incompatible or inconsistent source."""


@dataclass(frozen=True)
class Sources:
    cohort_root: Path = field(default_factory=lambda: Path(os.environ.get("MIMIC_OUTPUT_DIR", DEFAULT_ROOT)))
    model_root: Path | None = None
    experiment2_analysis: Path | None = None
    experiment3_run: Path | None = None
    human_directory: Path = DEFAULT_ROOT / "experiment5-private-v1"
    human_version: str = "results-v1"

    @property
    def models(self):
        return Path(self.model_root or os.environ.get(
            "HRP_MODEL_OUTPUT_DIR", self.cohort_root / "model-runs")).expanduser()


@dataclass
class Study:
    data: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    images: dict = field(default_factory=dict, repr=False)
    fictional: bool = False

    @property
    def validated(self):
        return not self.fictional and set(self.data) == set(UNITS) and all(
            row["status"] == "validated" for row in self.provenance
        )


def require(condition, detail):
    if not condition:
        raise ValidationError(detail)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def indexed_files(directory, files, required=()):
    require(isinstance(files, dict) and files and set(required) <= files.keys(),
            "The checksum index is missing required entries.")
    for name, expected in files.items():
        require(Path(name).name == name and name not in {".", ".."}
                and isinstance(expected, str) and HEX.fullmatch(expected),
                "The checksum index contains an invalid artifact name or digest.")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Missing indexed artifact {name}.")
        require(digest(path) == expected, f"Checksum mismatch for {name}.")


def source(number, version, analysis, observations, records, artifacts, **metadata):
    dataset, unit, design = UNITS[number]
    return dict(experiment=number, version=version, analysis=analysis, observations=observations,
                records=records, artifacts=artifacts, dataset=dataset, unit=unit, design=design,
                status="validated", **metadata)


def _model(sources):
    from experiment2_inputs import MODEL_RUN_ID, load_explanation_cases
    from modeling import load_cohort, read_json

    path = sources.models / MODEL_RUN_ID
    provenance, cases = load_explanation_cases(sources.models)
    manifest = read_json(path / "manifest.json")
    cohort = sources.cohort_root.expanduser() / provenance["cohort_id"]
    _, partitions, cohort_info = load_cohort(cohort)
    require(cohort_info == manifest["source"], "The cohort or partition counts disagree with the model bundle.")
    metrics = ("roc_auc", "average_precision", "prevalence", "brier", "mean_probability",
               "threshold", "precision", "recall", "f1")
    performance = []
    for split in ("validation", "test"):
        for family in ("logistic", "xgboost"):
            report = manifest[split][family]
            values = (report["raw"] if report["probability_source"] == "raw" else report["sigmoid_oof"]
                      ) if split == "validation" else report
            performance.append(dict(
                split=split, model=family, n=len(partitions[split]),
                probability_source=report["probability_source"], threshold=report["threshold"],
                **{key: values.get(key) for key in metrics if key != "threshold"},
                **{f"{key}_ci95": report.get("intervals", {}).get("percentile_95", {}).get(key)
                   for key in ("roc_auc", "average_precision")},
            ))
    with (path / "shap_importance.csv").open(newline="") as stream:
        importance = [{"feature": row["feature"], "mean_abs_shap": float(row["mean_abs_shap"])}
                      for row in csv.DictReader(stream)]
    artifacts = {"model bundle/manifest.json": provenance["manifest_sha256"],
                 "cohort/readiness_summary.json": digest(cohort / "private_audit/readiness_summary.json"),
                 **{f"cohort/{name}": checksum for name, checksum in provenance["source_sha256"].items()},
                 **{f"model bundle/{name}": manifest["files_sha256"][name] for name in (
                     "shap_importance.csv", "shap_reference.json", "shap_diagnostic_summary.json",
                     "case_contributions.parquet", "explanation_cases.parquet", "precision_recall.png",
                     "shap_importance.png")}}
    return (dict(performance=performance, partitions=manifest["source"]["partitions"],
                 selected=manifest["primary"], importance=importance[:10],
                 shap_diagnostics=read_json(path / "shap_diagnostic_summary.json"),
                 xgboost_calibration={key: manifest["validation"]["xgboost"][key] for key in (
                     "raw", "sigmoid_oof", "sigmoid_slope", "sigmoid_intercept")}),
            source(1, "model-bundle-v1", "frozen model evaluation v1", cohort_info["cohort"]["admissions"],
                   None, artifacts, cohort=provenance["cohort_id"], model=MODEL_RUN_ID,
                   model_versions={key: manifest["software"]["packages"][key]
                                   for key in ("xgboost", "shap")}),
            {"01_test_precision_recall": path / "precision_recall.png",
             "02_global_shap": path / "shap_importance.png"}, (provenance, cases))


def _experiment2(sources, reference):
    from experiment2_analysis import (ANALYSIS_VERSION, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
                                      ORIGINAL_CHECKPOINT_SHA256, _validate_run, analyze, read_checkpoint)

    path = sources.cohort_root.expanduser()
    directory = Path(sources.experiment2_analysis or
                     path / "llm-runs/experiment2-qwen3-primary/analysis-v1").expanduser()
    state, checkpoint = read_checkpoint(path / "checkpoint.json")
    require(checkpoint == ORIGINAL_CHECKPOINT_SHA256, "Experiment 2 checkpoint checksum differs from the pinned run.")
    provenance, cases = reference
    score = _validate_run(state, provenance, cases, completed=True)
    require(len(cases) == 72 and score["operations"]["records"] == 216,
            "Experiment 2 requires 72 complete A/B/C case selections.")
    run = state["manifest"]
    metadata, manifest_hash = read_checkpoint(directory / "experiment2_analysis_manifest.json")
    require(metadata["checkpoint_sha256"] == checkpoint
            and metadata["analysis_version"] in {"experiment2-analysis-v1", ANALYSIS_VERSION}
            and metadata["bootstrap_seed"] == BOOTSTRAP_SEED
            and metadata["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
            and all(metadata[key] == run[key] for key in ("prompt_version", "verifier_version", "scorer_version")),
            "Experiment 2 saved analysis has incompatible checkpoint, analysis or scoring provenance.")
    # The v1 export has no file index. Reconcile against the unchanged analysis
    # definitions in v2, then report the saved aggregate values.
    verified = analyze(dict(state=state, cases=cases, checkpoint_sha256=checkpoint, score=score))
    summary_path = directory / "experiment2_summary.json"
    summary, summary_hash = read_checkpoint(summary_path)
    summary_fields = ("operations", "grounded_audit", "narrative", "costs", "by_condition_and_source")
    require(summary == {key: verified[key] for key in summary_fields},
            "Experiment 2 saved summary differs from read-only checkpoint analysis.")
    data = {key: summary[key] for key in summary_fields[:-1]}
    artifacts = {"checkpoint.json": checkpoint, "experiment2_analysis_manifest.json": manifest_hash,
                 summary_path.name: summary_hash}
    for stem, key in E2_TABLES.items():
        table = directory / f"experiment2_{stem}.csv"
        with table.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in verified[key]]
        require(rows == expected, f"Saved Experiment 2 table {stem} disagrees with frozen scoring.")
        data[key] = [{name: (row[name] == "True" if isinstance(value, bool) else type(value)(row[name]))
                             if value is not None else None
                      for name, value in original.items()}
                     for row, original in zip(rows, verified[key], strict=True)]
    return data, source(2, f"checkpoint-format-{run['checkpoint_version']}", metadata["analysis_version"],
                        len(cases), score["operations"]["records"],
                        artifacts | {f"experiment2_{stem}.csv": digest(directory / f"experiment2_{stem}.csv")
                                     for stem in E2_TABLES},
                        cohort=provenance["cohort_id"], model=provenance["model_id"],
                        qwen=run["configuration"]["model"]["model"], qwen_digest=run["configuration"]["model"]["digest"],
                        prompt=run["prompt_version"], verifier=run["verifier_version"], scorer=run["scorer_version"],
                        replay_analysis=ANALYSIS_VERSION, bootstrap_seed=BOOTSTRAP_SEED,
                        bootstrap_replicates=BOOTSTRAP_REPLICATES), {}


def _experiment3(sources):
    from adversarial_analysis import ANALYSIS_VERSION
    from adversarial_runner import RUNNER_VERSION, _saved_run
    from adversarial_scenarios import validate_manifest
    from experiment2_analysis import read_checkpoint
    from human_evaluation_sampling import SCENARIO_SHA, SYNTHETIC_SHA
    from llm_runner import ModelConfig
    from llm_verification import fingerprint

    directory = Path(sources.experiment3_run or sources.cohort_root.expanduser() /
                     "llm-runs/experiment3-adversarial/qwen3-v1").expanduser()
    index, index_hash = read_checkpoint(directory / "run_manifest.json")
    require((index["experiment"], index["mode"], index["complete"], index["runner_version"],
             index["analysis_version"], index["checkpoint_sha256"], index["scenario_sha256"])
            == (3, "real", True, RUNNER_VERSION, ANALYSIS_VERSION, SYNTHETIC_SHA, SCENARIO_SHA),
            "Experiment 3 index has incompatible frozen provenance.")
    indexed_files(directory, index["files"], ("checkpoint.json", "scenario_manifest.json", "analysis.json",
                                               "01_h3_interaction.png"))
    manifest, _ = read_checkpoint(directory / "scenario_manifest.json")
    require(fingerprint(manifest) == SCENARIO_SHA, "Experiment 3 scenario fingerprint mismatch.")
    validate_manifest(manifest)
    state, checkpoint, config = _saved_run(directory, manifest, "real", ModelConfig(),
                                           request_timeout=None, completed=True)
    result, result_hash = read_checkpoint(directory / "analysis.json")
    require(checkpoint == index["files"]["checkpoint.json"] == SYNTHETIC_SHA
            and result_hash == index["files"]["analysis.json"] and result["version"] == ANALYSIS_VERSION,
            "Experiment 3 analysis does not match the validated checkpoint or analysis version.")
    n = len(manifest["records"]) // 2
    require(n == 48 and len(state["records"]) == result["operations"]["records"] == n * 6
            and len(result["faithfulness"]) == 6 and all(row["total"] == n for row in result["faithfulness"])
            and all(row["pairs"] <= row["selected_pairs"] <= n for row in result["paired"]),
            "Experiment 3 analysis does not preserve 48 pairs and six condition/variant groups.")
    # Explicit selection of aggregate fields excludes example profiles and response traces.
    data = {key: result[key] for key in ("bootstrap", "operations", "faithfulness", "paired",
                                        "structured_claim_rates", "claim_rates", "interventions")}
    return data, source(3, manifest["version"], ANALYSIS_VERSION, n, n * 6,
                        {"run_manifest.json": index_hash, **index["files"]},
                        qwen=config.model, qwen_digest=config.digest,
                        verifier=state["manifest"]["verifier_version"], scorer=state["manifest"]["scorer_version"],
                        runner=RUNNER_VERSION, scenario_fingerprint=SCENARIO_SHA), {
                            "04_adversarial_agreement": directory / "01_h3_interaction.png"}


def validate_live_summary(summary, manifest):
    """Check the public supplement against its frozen benchmark and internal totals."""
    from verifier_live_annotation import COUNTS, VERSION as LIVE_VERSION

    study, provenance, overall = (summary[key] for key in ("study", "provenance", "overall"))
    verifier = manifest["frozen_verifier"]
    expected = dict(challenge_sha256=manifest["challenge_sha256"],
                    live_selection_sha256=manifest["files"]["live_selection.json"],
                    frozen_verifier_version=verifier["version"], frozen_verifier_code_sha256=verifier["code_sha256"],
                    analysis_version=LIVE_VERSION)
    require(summary["version"] == LIVE_VERSION and all(provenance[k] == v for k, v in expected.items())
            and all(HEX.fullmatch(provenance[key]) for key in (
                "live_checkpoint_sha256", "annotation_sha256", "crosswalk_sha256")),
            "Live summary provenance does not match the frozen benchmark and selection.")
    n = sum(COUNTS.values())
    strata = summary["by_stratum"]
    require(study["fictional_cases_only"] is True and study["review_blinded_to_verifier_decision"] is True
            and study["reviewers"] == 1 and study["selected_drafts"] == n
            and len(strata) == len(COUNTS) and {row["stratum"] for row in strata} == COUNTS.keys(),
            "Live summary study or stratum schema is inconsistent.")
    counts = ("selected_drafts", "completed_revisions", "valid", "faithful", "unfaithful", "indeterminate",
              "complete", "faithful_complete_valid", "accepted", "rejected", "fallback")
    for row in [overall, *strata]:
        size = n if row is overall else COUNTS[row["stratum"]]
        require(all(type(row[k]) is int and 0 <= row[k] <= size for k in counts)
                and row["selected_drafts"] == row["completed_revisions"] == size
                and row["faithful"] + row["unfaithful"] + row["indeterminate"] == size
                and row["accepted"] + row["rejected"] == size and row["fallback"] == row["rejected"]
                and row["faithful_complete_valid"] <= min(row[k] for k in ("faithful", "complete", "valid")),
                "Live summary counts or delivery arithmetic are inconsistent.")
    require(all(sum(row[key] for row in strata) == overall[key] for key in counts)
            and overall["reserved_requests"] == n, "Live summary stratum totals disagree.")
    matrix = summary["confusion_matrix"]
    accepted, rejected = matrix["accepted"], matrix["rejected"]
    exclusions = summary["binary_exclusions"]
    classifications = summary["classification_counts"]
    require(set(matrix) == {"accepted", "rejected"}
            and set(exclusions) == {"invalid_response", "indeterminate"}
            and all(type(v) is int and 0 <= v <= n for v in exclusions.values())
            and sum(exclusions.values()) + sum(accepted.values()) + sum(rejected.values()) == n
            and exclusions["invalid_response"] == n - overall["valid"]
            and exclusions["indeterminate"] <= overall["indeterminate"]
            and sum(accepted.values()) <= overall["accepted"] and sum(rejected.values()) <= overall["rejected"]
            and set(accepted) == set(rejected) == {"faithful", "unfaithful"}
            and all(type(v) is int and 0 <= v <= n for v in (*accepted.values(), *rejected.values()))
            and all(accepted[k] + rejected[k] <= overall[k] for k in accepted)
            and classifications == dict(correct_acceptance=accepted["faithful"], false_acceptance=accepted["unfaithful"],
                                        correct_rejection=rejected["unfaithful"], false_rejection=rejected["faithful"])
            and summary["final_delivery"] == dict(model_revision=overall["accepted"],
                                                  deterministic_fallback=overall["fallback"]),
            "Live summary confusion matrix or delivery counts disagree.")
    rates = summary["rates"]
    require(rates["false_rejection_faithful_complete_valid"]["denominator"] == overall["faithful_complete_valid"]
            and rates["false_rejection_faithful_complete_valid"]["numerator"] <= rejected["faithful"]
            and rates["false_acceptance_unfaithful_valid"]["numerator"] == accepted["unfaithful"]
            and rates["false_acceptance_unfaithful_valid"]["denominator"] ==
            accepted["unfaithful"] + rejected["unfaithful"]
            and rates["fallback"]["numerator"] == overall["fallback"] and rates["fallback"]["denominator"] == n,
            "Live summary rate denominators disagree.")
    for rate in [*rates.values(), *(v for row in strata for v in row.values() if isinstance(v, dict))]:
        a, b = rate["numerator"], rate["denominator"]
        require(type(a) is int and type(b) is int and 0 <= a <= b <= n
                and rate["rate"] == (a / b if b else None), "Live summary contains an invalid rate.")
    return summary


def _experiment4():
    from verifier_evaluation import review
    from verifier_challenge_set import FIXTURE, load_challenge
    from verifier_live_annotation import load_public_summary

    bundle = load_challenge()
    saved = review(bundle=bundle)
    manifest, analysis = saved["manifest"], saved["analysis"]
    directory = FIXTURE / "results"
    live_path = directory / "live_annotation_summary.json"
    summary = validate_live_summary(load_public_summary(live_path), manifest)
    n = len(bundle["labels"])
    require(n == 180 and len(saved["predictions"]) == n, "Experiment 4 requires 180 frozen challenge responses.")
    for policy in analysis["primary"].values():
        require(policy["n"] == sum(policy["confusion"].values()) == 120,
                "Experiment 4 primary confusion matrix has the wrong denominator.")
        for metric in policy["metrics"].values():
            require(isclose(metric["estimate"], metric["numerator"] / metric["denominator"]),
                    "Experiment 4 metric disagrees with its numerator or denominator.")
    data = {key: analysis[key] for key in ("bootstrap", "primary", "all_eligible_sensitivity", "false_rejection",
                                          "paraphrases", "error_matrix", "layers", "boundary", "outcomes",
                                          "multi_error_cases")}
    data["paraphrases"] = [{key: value for key, value in row.items() if key != "pairs"}
                           for row in analysis["paraphrases"]]
    data["composition"] = dict(sorted(Counter(row["group"] for row in bundle["labels"]).items()))
    data["scripted_corrections"] = saved["corrections"]["summary"]
    data["live"] = {key: summary[key] for key in ("version", "overall", "rates", "classification_counts",
                                                 "rejection_reasons", "final_delivery", "binary_exclusions")}
    data["live"]["reviewers"] = summary["study"]["reviewers"]
    artifacts = {f"experiment4/v1/{name}": digest(FIXTURE / name)
                 for name in ("challenge.json", "gold.json", "gold_validation.json")}
    artifacts.update({f"experiment4/v1/results/{name}": digest(directory / name)
                      for name in ("manifest.json", "analysis.json", "predictions.json",
                                   "scripted_corrections.json", "live_selection.json", live_path.name,
                                   "01_explicit_confusion.png", "02_release_confusion.png")})
    return data, source(4, manifest["dataset_version"], manifest["version"], n, n, artifacts,
                        binary_responses=120, live_drafts=summary["study"]["selected_drafts"],
                        verifier=manifest["frozen_verifier"]["version"], live_analysis=summary["version"],
                        live_validation="Public schema, arithmetic and benchmark provenance"), {
                            "05a_explicit_detection": directory / "01_explicit_confusion.png"}


def _human_snapshot(directory, version):
    from experiment2_analysis import read_checkpoint

    require(re.fullmatch(r"results-v[1-9][0-9]*", version), "Use a versioned Experiment 5 results-vN snapshot.")
    output = directory / version
    index, index_hash = read_checkpoint(output / "index.json")
    indexed_files(output, index, ("aggregate.json", "01_faithfulness.png", "02_text_ratings.png",
                                  "04_matched_changes.png"))
    result, checksum = read_checkpoint(output / "aggregate.json")
    require(checksum == index["aggregate.json"], "Experiment 5 aggregate changed during validation.")
    return result, index, index_hash


def validate_human_summary(data, n):
    require(data["study"] == "primary" and data["clusters"] == n and data["explanations"] == n * 3
            and set(data["by_condition"]) == set(ARMS), "Experiment 5 aggregate does not match selected triplets.")
    for arm, row in data["by_condition"].items():
        faithful = row["faithfulness"]["faithful"]
        determinate = faithful + row["faithfulness"]["unfaithful"]
        require(set(row["faithfulness"]) == {"faithful", "unfaithful", "indeterminate", "invalid"}
                and set(row["completeness"]) == {"complete", "partial", "insufficient", "not_applicable"}
                and all(type(count) is int and 0 <= count <= n
                        for count in (*row["faithfulness"].values(), *row["completeness"].values()))
                and row["n"] == sum(row["faithfulness"].values()) == sum(row["completeness"].values()) == n
                and row["determinate_n"] == determinate
                and row["determinate_faithful_rate"] == (faithful / determinate if determinate else None)
                and row["all_selected_faithful_rate"] == faithful / n,
                "Experiment 5 faithfulness or completeness arithmetic is inconsistent.")
        for field in (CLEAR, READ):
            grade = row[field]
            require(set(grade["distribution"]) == set("12345")
                    and sum(grade["distribution"].values()) == grade["n"] <= n,
                    "Experiment 5 ordinal score counts are inconsistent.")
        verifier = data["human_vs_frozen_verifier"][arm]
        for field, counts in verifier["discordance"].items():
            require(sum(verifier[field].values()) == n and
                    0 <= counts["human_faithful_automated_flagged"] +
                    counts["human_unfaithful_automated_clear"] <= counts["comparable_n"] <= n,
                    "Experiment 5 automated comparison denominator is inconsistent.")
    require(set(data["paired"]) == set(COMPARISONS), "Experiment 5 is missing a paired contrast.")
    for metric in (FAITH, "completeness", CLEAR, READ):
        for contrast in COMPARISONS:
            row = data["paired"][contrast][metric]
            eligible = row["eligible_n"]
            bounds = row["mean_change_ci95"]
            require(row["matched_n"] == sum(row["transitions"].values()) == n
                    and 0 <= eligible <= n
                    and sum(row[key] for key in ("improved", "worsened", "unchanged")) == eligible
                    and ((row["mean_change"] is None and bounds is None and not eligible)
                         or (eligible and isinstance(bounds, list) and len(bounds) == 2
                             and bounds[0] <= row["mean_change"] <= bounds[1])),
                    "Experiment 5 matched counts or interval are inconsistent.")
    require(sum(row["n"] for row in data["condition_c_by_final_source"].values()) == n,
            "Experiment 5 delivery source counts do not sum to C.")


def _experiment5(sources):
    from experiment2_analysis import ORIGINAL_CHECKPOINT_SHA256
    from experiment2_inputs import COHORT_RUN_ID, MANIFEST_SHA256, MODEL_RUN_ID
    from human_evaluation_analysis import VERSION as HUMAN_VERSION
    from human_evaluation_rubric import repeat_lock
    from human_evaluation_sampling import read_packet
    from human_explanation_evaluation import _annotation_state
    from llm_verification import VERIFIER_VERSION, fingerprint

    directory = Path(sources.human_directory).expanduser()
    manifest, packet = read_packet(directory)
    primary, repeats, hashes, locked = _annotation_state(directory, manifest, packet)
    require(locked and (manifest["study"], manifest["reviewers"], manifest["primary_selected"], len(primary))
            == ("primary", 1, 72, 72) and manifest["selected_by_condition"] == dict.fromkeys(ARMS, 24)
            and manifest["repeat_selected"] == manifest["repeat_target"] == 12,
            "Experiment 5 requires the completed, locked single-reviewer primary pass and 12 selected repeats.")
    result, index, index_hash = _human_snapshot(directory, sources.human_version)
    provenance = result["provenance"]
    require(result["analysis_version"] == provenance["analysis_version"] == HUMAN_VERSION
            and provenance["study_manifest_sha256"] == fingerprint(manifest)
            and provenance["source_sha256"] == manifest["source_sha256"] ==
            {"checkpoint": ORIGINAL_CHECKPOINT_SHA256, "model_manifest": MANIFEST_SHA256}
            and provenance["annotation_sha256"]["primary"] == hashes["primary"]
            and provenance["crosswalk_sha256"] == manifest["files"]["crosswalk.json"]
            and all(provenance[key] == manifest[key] for key in ("rubric_sha256", "protocol_sha256")),
            "Experiment 5 source, annotation, protocol or analysis provenance mismatch.")
    validate_human_summary(result, 24)
    frozen_repeats = result["repeat_consistency"]
    require(frozen_repeats["selected"] == frozen_repeats["target"] == 12
            and 0 <= frozen_repeats["completed"] <= len(repeats) <= 12
            and frozen_repeats["missing_from_selected"] == 12 - frozen_repeats["completed"]
            and bool(provenance["annotation_sha256"]["repeat"]) == bool(frozen_repeats["completed"])
            and (frozen_repeats["completed"] < len(repeats) or
                 provenance["annotation_sha256"]["repeat"] == hashes["repeat"]),
            "Experiment 5 snapshot repeat counts are inconsistent with current annotations.")
    if sources.human_version != "results-v1":
        original, _, _ = _human_snapshot(directory, "results-v1")
        require({k: v for k, v in original.items() if k not in {"repeat_consistency", "provenance"}} ==
                {k: v for k, v in result.items() if k not in {"repeat_consistency", "provenance"}}
                and original["provenance"]["annotation_sha256"]["primary"] == hashes["primary"],
                "New human snapshot changes the frozen primary results.")
    completed = (len(repeats) == 12 and provenance["annotation_sha256"]["repeat"] == hashes["repeat"]
                 and frozen_repeats["completed"] == 12 and repeat_lock(directory, manifest, hashes["repeat"], 12))
    repeat_status = {"selected": 12, "recorded": len(repeats), "in_snapshot": frozen_repeats["completed"],
                     "status": "Repeat consistency available." if completed else PENDING}
    data = {key: result[key] for key in ("explanations", "clusters", "by_condition", "paired",
                                         "ratings_by_faithfulness", "human_vs_frozen_verifier",
                                         "condition_c_by_final_source", "bootstrap_seed", "bootstrap_replicates")}
    data.update(reviewers=manifest["reviewers"], primary_locked=True, repeat_status=repeat_status,
                repeat_consistency=frozen_repeats if completed else None)
    output = directory / sources.human_version
    return data, source(5, sources.human_version, HUMAN_VERSION, result["clusters"], result["explanations"],
                        {"results/index.json": index_hash, "results/aggregate.json": index["aggregate.json"]},
                        cohort=COHORT_RUN_ID, model=MODEL_RUN_ID, verifier=VERIFIER_VERSION,
                        reviewers=manifest["reviewers"], protocol=manifest["protocol_version"],
                        rubric=manifest["rubric_version"], primary_lock="validated",
                        annotation_snapshot_sha256=hashes["primary"],
                        input_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA256), {
                            "06_local_human_faithfulness": output / "01_faithfulness.png",
                            "07_local_human_text_ratings": output / "02_text_ratings.png",
                            "08_local_human_matched_changes": output / "04_matched_changes.png"}


def load_sources(sources=None):
    """Validate each source separately and return only aggregate data and logical provenance."""
    sources = sources or Sources()
    study = Study()
    reference = None
    for number, load in enumerate((
        lambda: _model(sources), lambda: _experiment2(sources, reference),
        lambda: _experiment3(sources), _experiment4, lambda: _experiment5(sources),
    ), 1):
        try:
            require(number != 2 or reference is not None, "Experiment 1 pinned model/cohort validation is required.")
            data, provenance, images, *extra = load()
            images = {name: path.read_bytes() for name, path in images.items()}
            if extra:
                reference = extra[0]
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, AssertionError) as error:
            # Display only controlled diagnostics. Exceptions from private readers may include paths or record data.
            if isinstance(error, ValidationError):
                detail = str(error)
            elif isinstance(error, FileNotFoundError):
                detail = f"Missing required artifact {Path(error.filename).name}." if error.filename else "Missing required artifact."
            else:
                detail = f"Frozen source validator failed ({type(error).__name__}). Check source paths, schema and provenance."
            dataset, unit, design = UNITS[number]
            study.provenance.append(dict(experiment=number, dataset=dataset, unit=unit, design=design,
                                         status="unavailable" if isinstance(error, FileNotFoundError) else "invalid",
                                         diagnostic=detail))
        else:
            study.data[number] = data
            study.provenance.append(provenance)
            study.images.update(images)
    return study


def fictional_study():
    from human_explanation_evaluation import fictional_results

    result, _ = fictional_results()
    validate_human_summary(result, result["clusters"])
    return Study(data={5: result}, fictional=True)


def paired_grounding(data, comparison):
    return next(row for row in data["paired"] if row["comparison"] == comparison
                and row["analysis"] == "complete_case" and row["metric"] == SIGNED)


def paired_reversal(data, arm):
    return next(row for row in data["paired"] if row["stratum"] == "all" and row["condition"] == arm
                and row["policy"] == "valid_pairs" and row["metric"] == SIGNED)


def paired_result(row, decimals=3):
    return f'{row["mean_difference"]:+.{decimals}f} ({row["pairs"]} pairs)'


def interval(low, high):
    return f"[{low:+.3f}, {high:+.3f}]" if low is not None and high is not None else "Unavailable"


def direction_status(row, direction=1):
    lo, hi = row["mean_change_ci95"] or (None, None)
    if lo is None:
        return "Inconclusive"
    if (direction == 1 and lo > 0) or (direction == -1 and hi < 0):
        return "Supported"
    if (direction == 1 and hi < 0) or (direction == -1 and lo > 0):
        return "Unsupported"
    return "Inconclusive"


def hypothesis_rows(study, *, public=False):
    """Map each original hypothesis to its designated metric and matched direction."""
    rows = []
    evidence = {row["experiment"]: row.get("analysis") for row in study.provenance}

    def add(exp, label, hypothesis, metric, contrast, result, uncertainty, status, interpretation, limitation):
        rows.append(dict(experiment=exp, hypothesis=label, research_question=hypothesis, metric=metric,
                         comparison=contrast, result=result, ci95=uncertainty, assessment=status,
                         interpretation=interpretation, limitation=limitation, evidence=evidence.get(exp)))

    if 1 in study.data:
        fit = next(row for row in study.data[1]["performance"] if row["split"] == "test" and row["model"] == "xgboost")
        add(1, "Model reference", "What prediction and attribution reference supports the study?",
            "Test average precision", "Held-out XGBoost vs test prevalence",
            f'{fit["average_precision"]:.3f} vs {fit["prevalence"]:.3f} (n={fit["n"]:,})',
            interval(*fit["average_precision_ci95"]), "Descriptive", "XGBoost was selected on validation.",
            "Prediction discrimination measures a different outcome from explanation faithfulness.")

    if 2 in study.data:
        data = study.data[2]
        for label, contrast, question, interpretation, limitation in (
            ("H1", "A→B", "SHAP grounding improves model-feature and direction agreement.",
             "Grounded drafts better match the fitted model's signed drivers.", "Free-form prose coverage was low."),
            ("H2", "B→C", "Verification, correction and fallback improve final agreement and completeness.",
             "The final system improved signed agreement and produced complete verified outputs.",
             "Model revision and deterministic fallback have different delivery outcomes."),
        ):
            row = paired_grounding(data, contrast)
            n = row["pairs"]
            lo, hi = row["ci95_low"], row["ci95_high"]
            status = direction_status({"mean_change_ci95": [lo, hi]})
            result = paired_result(row)
            if label == "H2":
                complete = next(r["count"] for r in data["interventions"] if r["outcome"] == "verified_complete")
                result += f"; {complete}/{n} verified complete"
                if complete < n and status == "Supported":
                    status = "Partially supported"
            add(2, label, question, "Mean signed top-three matches (0–3)", contrast, result,
                interval(lo, hi), status, interpretation, limitation)

    if 3 in study.data:
        reversal = [paired_reversal(study.data[3], arm) for arm in ARMS]
        negative = sum(direction_status({"mean_change_ci95": [row["ci95_low"], row["ci95_high"]]}, -1)
                       == "Supported" for row in reversal)
        add(3, "H3", "Faithfulness may decline when attributions conflict with clinical expectations.",
            "Mean signed top-three matches (0–3)", "Counterintuitive − intuitive, by condition",
            "; ".join(f"{row['condition']} {paired_result(row)}" for row in reversal),
            "; ".join(f"{r['condition']} {interval(r['ci95_low'], r['ci95_high'])}" for r in reversal),
            "Supported" if negative == 3 else "Partially supported" if negative else "Inconclusive",
            "A and B showed losses in this benchmark. C retained agreement under fallback-heavy delivery.",
            "The synthetic model switch can also change probability.")

    if 4 in study.data:
        data = study.data[4]
        structured, narrative = (data["layers"][key]["metrics"]["recall"] for key in ("structured", "narrative"))
        explicit, release = (data["primary"][key]["metrics"]["false_positive_rate"] for key in ("explicit", "release"))

        def rate_text(row):
            return f"{row['numerator']}/{row['denominator']}"

        add(4, "H4", "Structured errors are detected more reliably and unfamiliar faithful prose is withheld.",
            "Layer recall and false-positive rate", "Structured vs narrative; explicit vs release",
            f"Recall {rate_text(structured)} vs {rate_text(narrative)}; false positives "
            f"{rate_text(explicit)} vs {rate_text(release)}",
            "; ".join(f"{label} {interval(row['ci95_low'], row['ci95_high'])}" for label, row in (
                ("Structured", structured), ("Narrative", narrative), ("Explicit", explicit), ("Release", release))),
            "Supported" if structured["estimate"] > narrative["estimate"] and release["estimate"] >
            explicit["estimate"] else "Inconclusive", "Parser and release behavior diverged on the authored challenge.",
            "Layer-specific errors have different denominators. No formal between-layer interval was estimated.")

    if 5 in study.data and not study.fictional:
        data = study.data[5]
        for label, comparison, field, direction, question, interpretation in (
            ("H5a", "B − A", FAITH, 1, "B has higher human-assessed faithfulness than A.",
             "The reviewer more often judged grounded drafts faithful."),
            ("H5b", "C − B", READ, -1, "Fallback-heavy verified delivery may have lower readability.",
             "Readability declined in C. Delivery source and repetitive wording were not randomized."),
        ):
            row = data["paired"][comparison][field]
            lo, hi = row["mean_change_ci95"] or (None, None)
            status = direction_status(row, direction)
            if label == "H5b" and status == "Supported":
                status = "Partially supported"
            add(5, label, question, "Determinate faithfulness difference" if field == FAITH else "Readability points (1–5)",
                comparison, interpretation if public else f"{row['mean_change']:+.3f} ({row['eligible_n']}/{row['matched_n']} pairs)",
                "Detailed intervals in local review" if public else interval(lo, hi), status, interpretation,
                "Single reviewer and a selected sample of complete triplets.")
        shift = data["paired"]["C − A"]
        faith, readability = shift[FAITH], shift[READ]
        add(5, "H5c", "Faithfulness and readability may move separately.", "Faithfulness and readability",
            "C − A, endpoint-specific matched denominators",
            "The endpoints moved in opposite directions." if public else
            f"Faithfulness {faith['mean_change']:+.3f} (n={faith['eligible_n']}); "
            f"readability {readability['mean_change']:+.3f} (n={readability['eligible_n']})",
            "Detailed intervals in local review" if public else
            f"Faithfulness {interval(*faith['mean_change_ci95'])}; readability {interval(*readability['mean_change_ci95'])}",
            "Supported" if direction_status(faith) == direction_status(readability, -1) == "Supported" else "Inconclusive",
            "The paired condition-level movements were distinct.", "This comparison does not estimate individual-level dependence.")
        add(5, "H5d", "Optional synthetic ratings may find plausible but counterintuitive-model-inconsistent prose.",
            "Human faithfulness by variant", "Optional secondary study", "Outside the primary snapshot", "Unavailable",
            "Inconclusive", "The primary human study does not address this hypothesis.",
            "The synthetic automated benchmark measures a different endpoint.")
    return rows


def human_condition_rows(data):
    return [dict(condition=arm, selected=r["n"], **r["faithfulness"], determinate=r["determinate_n"],
                 determinate_rate=r["determinate_faithful_rate"], all_selected_rate=r["all_selected_faithful_rate"],
                 determinate_ci95=r["determinate_faithful_ci95"], **r["completeness"],
                 clarity_mean=r[CLEAR]["mean"], clarity_median=r[CLEAR]["median"],
                 readability_mean=r[READ]["mean"], readability_median=r[READ]["median"])
            for arm, r in data["by_condition"].items()]


def human_comparison_rows(data):
    return [dict(comparison=comparison, metric=metric, **data["paired"][comparison][metric])
            for comparison in COMPARISONS for metric in (FAITH, "completeness", CLEAR, READ)]


def human_verifier_rows(data):
    rows = []
    for arm, group in data["human_vs_frozen_verifier"].items():
        for check, counts in group["discordance"].items():
            left = counts["human_faithful_automated_flagged"]
            right = counts["human_unfaithful_automated_clear"]
            comparable = counts["comparable_n"]
            rows.append(dict(condition=arm, check=check, concordant=comparable - left - right,
                             **counts, outside_binary=data["by_condition"][arm]["n"] - comparable,
                             labels=group[check]))
    return rows


def verifier_rows(data):
    return [dict(cohort=cohort, decision=decision, n=row["n"], **row["confusion"])
            for cohort in ("primary", "all_eligible_sensitivity") for decision, row in data[cohort].items()]


def verifier_metrics(data):
    return [dict(cohort=cohort, decision=decision, metric=metric, **values)
            for cohort in ("primary", "all_eligible_sensitivity") for decision, row in data[cohort].items()
            for metric, values in row["metrics"].items()]


def coarsened_human_rows(data):
    """Whole-condition summaries only. Coarsen before making any shareable figure."""
    return [
        dict(condition=arm, selected=row["n"],
             faithful_share_band=rate_band(row["all_selected_faithful_rate"]),
             clarity_rounded_half_point=round(row[CLEAR]["mean"] * 2) / 2,
             readability_rounded_half_point=round(row[READ]["mean"] * 2) / 2)
        for arm, row in data["by_condition"].items()
    ]


def rate_band(rate):
    lower = min(4, int(rate * 5)) * 20
    return f"{lower}–{lower + 20}%"


def report_tables(study):
    """Publication projection. Keep small human outcome cells and linked contrasts local."""
    e1, e2, e3, e4, e5 = (study.data[i] for i in UNITS)
    return {
        "01_design": [{k: r.get(k) for k in ("experiment", "dataset", "design", "unit", "observations", "records")}
                      for r in study.provenance],
        "02_model_performance": e1["performance"],
        "03_explanation_comparison": e2["structured"],
        "03b_grounding_paired": e2["paired"],
        "04_adversarial_comparison": e3["faithfulness"],
        "04b_adversarial_paired": [r for r in e3["paired"] if r["stratum"] == "all"],
        "05_verifier_confusion": verifier_rows(e4),
        "05b_verifier_metrics": verifier_metrics(e4),
        "05c_live_revision": [dict(outcome=key, numerator=e4["live"]["overall"][key],
                                   denominator=e4["live"]["overall"]["selected_drafts"])
                              for key in ("faithful", "accepted", "fallback", "unfaithful")] +
                             [dict(outcome="faithful_complete_valid_withheld", **e4["live"]["rates"][
                                 "false_rejection_faithful_complete_valid"])],
        "06_human_ratings": coarsened_human_rows(e5),
        "07_hypothesis_results": hypothesis_rows(study, public=True),
    }


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_export(directory):
    files = json.loads((directory / "output_index.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()} - {"output_index.json"}
    require(set(files) == actual and {"master_results.json", "hypothesis_summary.json", "source_manifest.json"} <= actual,
            "Synthesis output index is incomplete.")
    require(all(digest(directory / name) == checksum for name, checksum in files.items()),
            "Synthesis output checksum mismatch.")


def export_study(study, root=ROOT / "results_synthesis", version="v1"):
    from results_synthesis_figures import public_figures

    require(study.validated, "Export requires validation of all five frozen sources.")
    require(re.fullmatch(r"v[1-9][0-9]*", version), "Use a synthesis version of the form vN.")
    root = Path(root)
    target = root / version
    root.mkdir(parents=True, exist_ok=True)
    # Reserve the destination atomically. A concurrent exporter cannot overwrite it.
    target.mkdir()
    try:
        tables = report_tables(study)
        with TemporaryDirectory(prefix=".synthesis-", dir=root) as temporary:
            stage = Path(temporary)
            (stage / "tables").mkdir()
            (stage / "figures").mkdir()
            for name, rows in tables.items():
                with (stage / "tables" / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                    writer.writeheader()
                    writer.writerows(rows)
            _write_json(stage / "master_results.json", dict(version=version, analysis=VERSION, tables=tables))
            _write_json(stage / "hypothesis_summary.json", tables["07_hypothesis_results"])
            _write_json(stage / "source_manifest.json", dict(version=version, analysis=VERSION, sources=study.provenance))
            captions = public_figures(study, tables, stage / "figures")
            (stage / "README.md").write_text(
                "# Frozen HRP results synthesis\n\n"
                "Notebook 07 integrates five previous experiments. This is an aggregate report bundle. "
                "Exact human outcome counts, paired intervals, risk and delivery subgroups, and human–verifier "
                "cross-tabulations are available in authorized local review. Public human faithfulness is binned "
                "in 20 percentage-point bands (lower bound included, upper excluded except 100%). "
                "Text ratings are rounded to the nearest half-point. Tables retain separate study denominators.\n\n"
                "## Figure captions\n\n" + "\n\n".join(f"**{name}** — {caption}" for name, caption in captions.items()) +
                "\n", encoding="utf-8")
            _write_json(stage / "output_index.json", {
                str(path.relative_to(stage)): digest(path) for path in sorted(stage.rglob("*")) if path.is_file()})
            validate_export(stage)
            # The target reservation is empty and owned by this call.
            stage.rename(target)
    except BaseException:
        target.rmdir()
        raise
    return target


def run(mode="review", *, sources=None, output_root=None, version="v1"):
    require(mode in MODES, "Choose review, validate, export or fictional.")
    if mode == "fictional":
        return fictional_study()
    study = load_sources(sources)
    if mode == "export":
        failures = [f'Experiment {r["experiment"]}: {r["diagnostic"]}'
                    for r in study.provenance if r["status"] != "validated"]
        require(not failures, "Export validation failed. " + " ".join(failures))
        export_study(study, output_root or ROOT / "results_synthesis", version)
    if mode == "validate":
        require(re.fullmatch(r"v[1-9][0-9]*", version), "Use a synthesis version of the form vN.")
        existing = (Path(output_root) if output_root else ROOT / "results_synthesis") / version
        if existing.exists():
            validate_export(existing)
    return study


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES, nargs="?", default="review")
    parser.add_argument("--human-directory", type=Path, default=Sources().human_directory)
    parser.add_argument("--human-version", default="results-v1")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        study = run(args.mode, sources=Sources(human_directory=args.human_directory, human_version=args.human_version),
                    output_root=args.output_root, version=args.version)
    except (ValidationError, FileExistsError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps(study.provenance if not study.fictional else {"status": "Fictional software fixture"},
                     ensure_ascii=False, indent=2))
    if not study.validated and not study.fictional:
        sys.exit(1)


if __name__ == "__main__":
    main()
