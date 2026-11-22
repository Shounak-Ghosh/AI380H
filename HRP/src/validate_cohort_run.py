"""Validate a saved private MIMIC-IV cohort and write an aggregate readiness summary."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

from cohort import FLOW_REASONS, LABEL, audit_cohort, partition, validate_cohort
from features import LAB_ITEMS, LAB_TIMES, LABS, feature_schema, verified_lab_mapping
from report import build_report, lab_range_audit
from run_cohort import DATASET, DEFAULT_DIRECTORY, DEFAULT_PROJECT, SQL_DIR


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_saved_run(run):
    """Audit the data contract and saved reports without querying BigQuery."""
    cohort = pd.read_parquet(run / "cohort.parquet")
    splits = pd.read_parquet(run / "splits.parquet")
    schema = json.loads((run / "feature_schema.json").read_text())
    require(schema == feature_schema(), "Saved predictor schema differs from code")
    validate_cohort(cohort)
    require(cohort.columns.is_unique and cohort.subject_id.notna().all(),
            "Invalid cohort columns or missing subject ID")
    allowed_predictors = [feature["name"] for feature in schema["features"]]
    require(len(allowed_predictors) == len(set(allowed_predictors)), "Duplicate predictor names")
    cohort[allowed_predictors]  # Verify the downstream allowlist is loadable.

    expected_splits = partition(cohort)
    pd.testing.assert_frame_equal(expected_splits, splits.sort_values("hadm_id", ignore_index=True))
    joined = cohort.merge(splits, on=["subject_id", "hadm_id"], validate="1:1")
    require(len(joined) == len(cohort) == len(splits), "Incomplete split assignment")
    require(joined.groupby("subject_id").partition.nunique().eq(1).all(), "Patient split leakage")
    require(joined.groupby("partition")[LABEL].nunique().eq(2).all(),
            "A partition has missing outcome classes")

    observed = cohort[LABS].notna().sum(axis=1)
    require(observed.eq(cohort.selected_labs.fillna(0)).all(), "Selected lab count mismatch")
    measured = observed.gt(0)
    for column in LAB_TIMES:
        clock = cohort[column]
        require(clock.notna().eq(measured).all(), f"Missing timing audit: {column}")
        require(clock[measured].between(cohort.admittime[measured], cohort.prediction_time[measured])
                .all(), f"Out-of-window lab timing: {column}")
    require((cohort.first_lab_charttime.le(cohort.last_lab_charttime)
             & cohort.last_lab_charttime.le(cohort.last_lab_storetime))[measured].all(),
            "Lab chart/store chronology mismatch")
    for name, _, _, _, _, allow_zero in LAB_ITEMS:
        values = cohort[name].dropna()
        require((np.isfinite(values) & (values.ge(0) if allow_zero else values.gt(0))).all(),
                f"Invalid lab values for {name}")
        require(cohort[f"{name}_missing"].eq(cohort[name].isna()).all(),
                f"Lab indicator mismatch for {name}")

    audit = run / "private_audit"
    exact_flow = pd.read_csv(audit / "exact_flow.csv")
    require(exact_flow.disposition.tolist() == list(FLOW_REASONS)
            and exact_flow.admissions.ge(0).all()
            and int(exact_flow.iloc[-1].admissions) == len(cohort), "Invalid eligibility flow")
    source_schema = pd.read_csv(run / "aggregates/source_schema.csv").set_index("table")
    require(exact_flow.admissions.sum() == source_schema.loc["admissions", "source_rows"],
            "Eligibility flow does not reconcile to source records")
    mapping = verified_lab_mapping(pd.read_csv(audit / "lab_dictionary_unit_profile.csv"))
    ranges = lab_range_audit(cohort)
    pd.testing.assert_frame_equal(ranges, pd.read_csv(audit / "lab_ranges.csv"), check_dtype=False)
    for name in ("admission_categories", "source_nulls", "query_costs"):
        require(not pd.read_csv(audit / f"{name}.csv").empty, f"Empty audit file: {name}")

    report = build_report(cohort, exact_flow, splits, mapping)
    for name, table in report.items():
        actual = pd.read_csv(run / "aggregates" / f"{name}.csv")
        pd.testing.assert_frame_equal(
            table.reset_index(drop=True), actual, check_dtype=False, check_categorical=False,
            obj=f"aggregate {name}",
        )
    for name in ("cohort_flow", "label_distribution", "lab_missingness"):
        with Image.open(run / "aggregates" / f"{name}.png") as plot:
            plot.verify()

    checksums = {}
    for name in ("cohort.parquet", "splits.parquet", "feature_schema.json"):
        with (run / name).open("rb") as handle:
            checksums[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return cohort, exact_flow, {
        "cohort": report["cohort_summary"].iloc[0].to_dict(),
        "exact_flow": exact_flow.to_dict("records"),
        "labs": mapping.merge(report["lab_missingness"], on="feature").to_dict("records"),
        "lab_completeness": report["lab_completeness"].to_dict("records"),
        "lab_ranges": ranges.to_dict("records"),
        "partitions": report["partitions"].to_dict("records"),
        "sha256": checksums,
        "passed": ["counts and eligibility", "binary outcome timing", "predictor allowlist",
                   "prior-use consistency", "seeded patient partitions", "lab timing bounds",
                   "dictionary and units", "lab indicators and range audit", "saved aggregates"],
        "limitations": [
            "Raw lab event IDs and individual lab clocks are not stored. Timing extrema and the "
            "selected count are verified from Parquet. Event identity and ranking rely on the "
            "two-key SQL join and offline fixtures.",
            "Administrative fields lack timestamped snapshots. In-system absence of a return "
            "does not establish complete 30-day follow-up. Date shifts preserve within-patient "
            "intervals and prevent a common cutoff on shifted calendar dates.",
            "Range flags identify values for review. No chart adjudication was performed.",
        ],
    }


def check_source(cohort, exact_flow, project):
    """Recompute primary and sensitivity results using only the source timelines."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    location = client.get_dataset(DATASET, timeout=30).location
    cap = 100_000_000

    def fetch(name):
        sql = (SQL_DIR / f"{name}.sql").read_text()
        estimate = client.query(sql, location=location, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False,
        )).total_bytes_processed
        require(estimate <= cap, f"{name} dry run exceeds 100 MB")
        job = client.query(sql, location=location, job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=cap,
        ))
        data = job.result().to_dataframe(create_bqstorage_client=False)
        return data, {"query": f"{name}.sql", "estimated_bytes": estimate,
                      "billed_bytes": job.total_bytes_billed, "cache_hit": job.cache_hit}

    with ThreadPoolExecutor(max_workers=2) as executor:
        (admissions, aq), (patients, pq) = executor.map(fetch, ("cohort", "patients"))
    index, flow, sensitivity = audit_cohort(admissions, patients)
    pd.testing.assert_frame_equal(index, cohort[index.columns])
    pd.testing.assert_frame_equal(flow, exact_flow, check_dtype=False)
    additional = int(sensitivity.later_qualifying.sum())
    positives = int(cohort[LABEL].sum())
    return {
        "raw_source_admissions": len(admissions),
        "primary_labels_dates_and_features_exact_match": True,
        "primary_eligibility_flow_exact_match": True,
        "queries": [aq, pq],
        "later_qualifying_by_primary_exclusion": sensitivity.to_dict("records"),
        "primary": {"admissions": len(cohort), "positives": positives,
                    "prevalence_pct": 100 * positives / len(cohort)},
        "broader_endpoint": {"admissions": len(cohort) + additional,
                             "positives": positives + additional,
                             "prevalence_pct": 100 * (positives + additional) / (len(cohort) + additional)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Defaults to the latest run in MIMIC_OUTPUT_DIR")
    parser.add_argument("--check-source", action="store_true",
                        help="Query admission and patient timelines with a 100 MB per-query cap")
    args = parser.parse_args()
    root = Path(os.environ.get("MIMIC_OUTPUT_DIR", DEFAULT_DIRECTORY)).expanduser()
    run = (args.run_dir or max((path for path in root.glob("run-*") if path.is_dir()),
                              default=root)).expanduser().resolve()
    require(not run.is_relative_to(Path(__file__).resolve().parents[2])
            and not run.stat().st_mode & 0o077, "Expected an owner-only run outside Git")
    cohort, exact_flow, summary = validate_saved_run(run)
    previous_audit = run / "private_audit/endpoint_sensitivity.json"
    if args.check_source:
        summary["endpoint_sensitivity"] = check_source(
            cohort, exact_flow, os.environ.get("MIMIC_BQ_PROJECT", DEFAULT_PROJECT)
        )
    elif previous_audit.exists():
        previous = json.loads(previous_audit.read_text())
        require(previous["primary"]["admissions"] == len(cohort)
                and previous["primary"]["positives"] == int(cohort[LABEL].sum())
                and previous["raw_source_admissions"] == int(exact_flow.admissions.sum()),
                "Saved source audit counts disagree with the cohort")
        summary["endpoint_sensitivity"] = previous
        summary["source_audit_reused"] = True
    target = run / "private_audit/readiness_summary.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(handle.name).replace(target)
    print(f"Private readiness summary: {target}")


if __name__ == "__main__":
    main()
