"""Train or regenerate private Experiment 1 artifacts without querying BigQuery."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from model_explanations import Attribution, generate_cases
from modeling import (
    DEFAULT_COHORT_ROOT, KEYS, Settings, audit_features, calibrate, choose_cohort, choose_primary,
    encoding_map, evaluate, feature_metadata, fit_candidates, load_cohort, model_input,
    private_directory, raw_margin, read_json, sha256,
)
from model_plots import evaluation_plots, global_plots, local_plots

FAMILIES = ("logistic", "xgboost")
FORMAT_VERSION = 1
CODE_FILES = ("modeling.py", "model_explanations.py", "model_plots.py", "run_models.py")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fingerprint():
    source = Path(__file__).resolve().parent
    return {
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap",
            "pyarrow", "matplotlib", "joblib", "threadpoolctl",
        )},
        "code_sha256": {name: sha256(source / name) for name in CODE_FILES},
        "uv_lock_sha256": sha256(source.parent / "uv.lock"),
    }


@contextmanager
def staging_run(root):
    requested = Path(root).expanduser()
    if not requested.exists():
        private_directory(requested.parent)
        requested.mkdir(mode=0o700, exist_ok=True)
    root = private_directory(requested)
    run_id = f"models-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix=".model-stage-", dir=root) as stage:
        work = Path(stage) / run_id
        work.mkdir(mode=0o700)
        yield work
        work.rename(root / run_id)


def finalize(work, manifest):
    files = {str(path.relative_to(work)): sha256(path) for path in sorted(work.rglob("*"))
             if path.is_file() and path.name != "manifest.json"}
    write_json(work / "manifest.json", dict(manifest, run_id=work.name, files_sha256=files))


def read_manifest(run):
    return read_json(private_directory(run) / "manifest.json")


def verify_run(run):
    run = private_directory(run)
    manifest = read_manifest(run)
    if manifest["format_version"] != FORMAT_VERSION or manifest["run_id"] != run.name:
        raise ValueError("Unknown model bundle format or incorrect run ID")
    actual = {str(path.relative_to(run)): sha256(path) for path in sorted(run.rglob("*"))
              if path.is_file() and path.name != "manifest.json"}
    if not actual or actual != manifest["files_sha256"]:
        raise ValueError("Model bundle file checksums disagree with the manifest")
    return manifest


def load_fitted(run):
    """Deserialize only bundles written in a trusted owner-only local directory."""
    manifest = verify_run(run)
    run = private_directory(run)
    return manifest, joblib.load(run / "fitted.joblib")


def latest_model_run(root):
    root = Path(root).expanduser()
    return max((path for path in root.glob("models-*") if (path / "manifest.json").is_file()),
               default=None)


class Experiment:
    def __init__(self, cohort_run, settings):
        self.settings = settings
        self.schema, self.partitions, self.source = load_cohort(cohort_run)
        self.train, self.validation, self.test = (self.partitions[n] for n in
                                                  ("train", "validation", "test"))
        self.feature_audits = {"train": audit_features(self.train, self.schema)}
        self.models, self.candidates = {}, {}

    def fit(self, family):
        self.models[family], self.candidates[family] = fit_candidates(
            family, self.train, self.validation, self.schema, self.settings,
        )
        return pd.DataFrame(self.candidates[family]).drop(columns="config")

    def validation_decisions(self):
        label = self.schema["label"]
        x = model_input(self.validation, self.schema)
        y = self.validation[label]
        self.validation_predictions = self.validation[KEYS + [label]].copy()
        self.reports, self.calibrators = {}, {}
        for family in FAMILIES:
            margins = raw_margin(self.models[family], x)
            calibrator, report, raw, oof = calibrate(y, margins, self.validation.subject_id, self.settings)
            self.calibrators[family], self.reports[family] = calibrator, report
            self.validation_predictions[f"{family}_margin"] = margins
            self.validation_predictions[f"{family}_raw"] = raw
            self.validation_predictions[f"{family}_sigmoid_oof"] = oof
        self.primary = choose_primary(self.reports, self.settings)
        return self.primary

    def evaluate_test(self):
        # Inspect test and validation features after freezing every model decision.
        for name in ("validation", "test"):
            self.feature_audits[name] = audit_features(self.partitions[name], self.schema)
        self.test_metrics, self.test_scores = {}, {}
        for family in FAMILIES:
            metrics, probability = evaluate(self.models[family], self.calibrators[family],
                                            self.reports[family], self.test, self.schema, self.settings)
            winner = next(row for row in self.candidates[family] if row["selected"])
            self.test_metrics[family] = dict(metrics, training_seconds=winner["training_seconds"])
            self.test_scores[family] = probability
        return comparison(self.test_metrics)

    def prepare_shap(self):
        family = self.primary["model"]
        self.attribution = Attribution(family, self.models[family], self.train, self.schema, self.settings)

    def explain_global(self):
        self.global_frame = self.test.sample(n=min(self.settings.global_n, len(self.test)),
                                             random_state=self.settings.seed + 1)
        self.global_values, self.global_diagnostics = self.attribution.explain(self.global_frame)
        return diagnostic_summary(self.global_diagnostics)

    def select_cases(self, model_id):
        family = self.primary["model"]
        self.cases, self.case_values, self.case_summary = generate_cases(
            self.test, self.test_scores[family], self.attribution, self.calibrators[family],
            self.reports[family]["probability_source"], model_id, self.settings,
        )
        return self.case_summary

    def save(self, root):
        with staging_run(root) as work:
            self.cases["model_id"] = work.name
            metadata = feature_metadata(self.schema)
            pd.concat([table.assign(partition=name) for name, table in self.feature_audits.items()],
                      ignore_index=True).to_csv(work / "feature_audit.csv", index=False)
            write_json(work / "feature_schema.json", self.schema)
            write_json(work / "feature_metadata.json", metadata)
            write_json(work / "encoding_map.json", {
                key: encoding_map(model.named_steps["preprocess"]) for key, model in self.models.items()
            })
            joblib.dump(dict(models=self.models, calibrators=self.calibrators), work / "fitted.joblib")
            self.validation_predictions.to_parquet(work / "validation_predictions.parquet", index=False)
            test_predictions = self.test[KEYS + [self.schema["label"]]].assign(**self.test_scores)
            test_predictions.to_parquet(work / "test_predictions.parquet", index=False)
            comparison(self.test_metrics).to_csv(work / "test_comparison.csv")
            evaluation_plots(work, test_predictions[self.schema["label"]], self.test_scores,
                             self.reports, self.test_metrics)
            manifest = dict(format_version=FORMAT_VERSION, model_id=work.name, source=self.source,
                            settings=self.settings.manifest(), software=fingerprint(),
                            candidates=self.candidates, validation=self.reports,
                            primary=self.primary, test=self.test_metrics)
            save_global(work, metadata, self.attribution, self.global_frame,
                        self.global_values, self.global_diagnostics)
            save_cases(work, metadata, self.cases, self.case_values, self.case_summary)
            finalize(work, manifest)
        return private_directory(root) / work.name


def comparison(metrics):
    excluded = {"intervals", "calibration_curve", "confusion_matrix"}
    rows = {}
    for family, result in metrics.items():
        row = {key: value for key, value in result.items() if key not in excluded}
        for metric, (lower, upper) in result["intervals"]["percentile_95"].items():
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = lower, upper
        rows[family] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")


def diagnostic_summary(diagnostics):
    return dict(rows=len(diagnostics), max_abs_error=float(diagnostics.abs_error.max()),
                mean_expected_value=float(diagnostics.expected_value.mean()))


def save_global(work, metadata, attribution, frame, values, diagnostics):
    frame[KEYS].join(values).to_parquet(work / "global_shap.parquet", index=False)
    frame[KEYS].join(diagnostics).to_parquet(work / "global_shap_diagnostics.parquet", index=False)
    importance = global_plots(work, values, metadata, attribution.settings.seed)
    importance.to_csv(work / "shap_importance.csv", index_label="feature")
    write_json(work / "shap_reference.json", attribution.metadata)
    write_json(work / "shap_diagnostic_summary.json", diagnostic_summary(diagnostics))


def save_cases(work, metadata, cases, values, summary):
    cases.to_parquet(work / "explanation_cases.parquet", index=False)
    values.to_parquet(work / "case_contributions.parquet", index=False)
    write_json(work / "case_summary.json", summary)
    local_plots(work, cases, values, metadata)


def regenerate(saved, cohort_run, root, *, recompute_shap, regenerate_cases):
    if not (recompute_shap or regenerate_cases):
        raise ValueError("Select a regeneration action")
    saved = private_directory(saved)
    manifest, fitted = load_fitted(saved)
    settings = Settings(**manifest["settings"])
    experiment = Experiment(cohort_run, settings)
    if experiment.source != manifest["source"]:
        raise ValueError("Selected cohort does not match the saved model source")
    software = fingerprint()
    for key in ("python", "packages", "code_sha256", "uv_lock_sha256"):
        if software[key] != manifest["software"][key]:
            raise ValueError("Use the original model software and source versions for regeneration")
    experiment.primary = manifest["primary"]
    experiment.reports = manifest["validation"]
    experiment.models, experiment.calibrators = fitted["models"], fitted["calibrators"]
    experiment.prepare_shap()
    reference = read_json(saved / "shap_reference.json")
    if experiment.attribution.metadata != reference:
        raise ValueError("SHAP reference differs from the saved model run")
    family = experiment.primary["model"]
    metadata = feature_metadata(experiment.schema)
    with staging_run(root) as work:
        shutil.copytree(saved, work, dirs_exist_ok=True)
        work.chmod(0o700)
        if recompute_shap:
            experiment.explain_global()
            save_global(work, metadata, experiment.attribution, experiment.global_frame,
                        experiment.global_values, experiment.global_diagnostics)
        if regenerate_cases:
            predictions = pd.read_parquet(saved / "test_predictions.parquet")
            predicted = experiment.test[KEYS].merge(predictions[KEYS + [family]], on=KEYS,
                                                   how="left", validate="1:1")
            experiment.test_scores = {family: predicted[family].to_numpy()}
            experiment.select_cases(manifest["model_id"])
            save_cases(work, metadata, experiment.cases, experiment.case_values,
                       experiment.case_summary)
        manifest.update(parent_run_id=manifest["run_id"], regeneration_software=software)
        finalize(work, manifest)
    return private_directory(root) / work.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "review", "regenerate"), default="review", nargs="?")
    parser.add_argument("--cohort-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--recompute-shap", action="store_true")
    parser.add_argument("--regenerate-cases", action="store_true")
    args = parser.parse_args()
    cohort_root = Path(os.environ.get("MIMIC_OUTPUT_DIR", DEFAULT_COHORT_ROOT)).expanduser()
    output_root = args.output_root or Path(os.environ.get("HRP_MODEL_OUTPUT_DIR", cohort_root / "model-runs"))
    saved = args.model_run or latest_model_run(output_root)
    if args.mode == "review":
        if saved is None:
            parser.error("No saved model run. Pass --model-run or explicitly select train.")
        run = saved
    else:
        cohort_run = choose_cohort(cohort_root, args.cohort_run or os.environ.get("HRP_COHORT_RUN"))
        if args.mode == "train":
            experiment = Experiment(cohort_run, Settings())
            print("Validated cohort:", experiment.source["id"], flush=True)
            for family in FAMILIES:
                print(f"Training {family} candidates", flush=True)
                print(experiment.fit(family).to_string(index=False), flush=True)
            experiment.validation_decisions()
            print("Primary model fixed:", experiment.primary["model"], flush=True)
            experiment.evaluate_test()
            print("Computing held-out SHAP and selecting local cases", flush=True)
            experiment.prepare_shap()
            experiment.explain_global()
            experiment.select_cases(model_id=None)
            run = experiment.save(output_root)
        else:
            if saved is None:
                parser.error("Regeneration requires a saved model run.")
            run = regenerate(saved, cohort_run, output_root, recompute_shap=args.recompute_shap,
                             regenerate_cases=args.regenerate_cases)
    manifest = verify_run(run)
    print(f"Private model bundle: {run}")
    print(json.dumps(manifest["primary"], indent=2))
    print(comparison(manifest["test"]).to_string())
    print(json.dumps(read_json(Path(run) / "shap_diagnostic_summary.json"), indent=2))
    print(json.dumps(read_json(Path(run) / "case_summary.json")["selected"]))


if __name__ == "__main__":
    main()
