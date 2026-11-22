"""Read-only input contract for the pinned Experiment 2 explanation cases."""

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from modeling import DEFAULT_COHORT_ROOT, read_json, sha256
from run_models import verify_run

MODEL_RUN_ID = "models-20261122T230904Z-511d7f06"
COHORT_RUN_ID = "run-20261122T212339Z-c71630d2"
MANIFEST_SHA256 = "072840d4f1b0ea5a3d3b26311de8a9d275d92751d1655201ce36c4fe46ab0756"


@dataclass(frozen=True)
class ExplanationCase:
    case_id: str
    features: tuple[dict, ...]
    predicted_risk: float
    shap: dict

    def inputs(self, condition):
        """Return only the fields supplied to the LLM in condition A, B or C."""
        if condition not in {"A", "B", "C"}:
            raise ValueError("Choose condition A, B or C")
        payload = {
            "features": [dict(feature) for feature in self.features],
            "predicted_risk": self.predicted_risk,
        }
        if condition in {"B", "C"}:
            payload["shap"] = {
                "expected_value": self.shap["expected_value"],
                "output_scale": self.shap["output_scale"],
                "contributions": [dict(row) for row in self.shap["contributions"]],
            }
        return payload


def load_explanation_cases(model_root=None):
    """Return bundle provenance and the 72 cases in saved order."""
    cohort_root = Path(os.environ.get("MIMIC_OUTPUT_DIR", DEFAULT_COHORT_ROOT))
    root = Path(model_root or os.environ.get("HRP_MODEL_OUTPUT_DIR", cohort_root / "model-runs"))
    run = root.expanduser() / MODEL_RUN_ID
    manifest = verify_run(run)
    if (sha256(run / "manifest.json") != MANIFEST_SHA256
            or manifest["model_id"] != MODEL_RUN_ID
            or manifest["source"]["id"] != COHORT_RUN_ID):
        raise ValueError("Experiment 2 requires the pinned Experiment 1 bundle and cohort")

    feature_metadata = read_json(run / "feature_metadata.json")
    reference = read_json(run / "shap_reference.json")
    primary = manifest["primary"]["model"]
    validation = manifest["validation"][primary]
    provenance = {
        "model_id": manifest["model_id"],
        "model_family": primary,
        "cohort_id": manifest["source"]["id"],
        "source_sha256": manifest["source"]["sha256"],
        "manifest_sha256": MANIFEST_SHA256,
        "software": manifest["software"],
        "probability_source": validation["probability_source"],
        "calibration": {
            "applied": validation["probability_source"] == "sigmoid",
            "candidate_sigmoid_slope": validation["sigmoid_slope"],
            "candidate_sigmoid_intercept": validation["sigmoid_intercept"],
        },
        "shap_reference": {
            key: value for key, value in reference.items() if key != "background_admission_ids"
        },
    }

    rows = pd.read_parquet(run / "explanation_cases.parquet")
    contributions = pd.read_parquet(run / "case_contributions.parquet")
    by_admission = {key: group for key, group in contributions.groupby("hadm_id", sort=False)}
    cases = tuple(
        _prepare_case(i, row, by_admission[row["hadm_id"]], feature_metadata)
        for i, row in enumerate(rows.to_dict("records"), 1)
    )
    return provenance, cases


def _prepare_case(number, row, contributions, metadata):
    features = tuple(
        {"name": name, "label": feature["label"], "unit": feature["unit"],
         "value": None if pd.isna(row[name]) else row[name]}
        for name, feature in metadata.items()
    )
    ranked = contributions.sort_values("rank")[["feature", "shap", "direction", "rank"]]
    entries = tuple(
        {**entry, "label": metadata[entry["feature"]]["label"]}
        for entry in ranked.to_dict("records")
    )
    return ExplanationCase(
        case_id=f"case-{number:03d}",
        features=features,
        predicted_risk=float(row["predicted_risk"]),
        shap={"expected_value": float(row["expected_value"]),
              "output_scale": row["shap_output_scale"], "contributions": entries},
    )
