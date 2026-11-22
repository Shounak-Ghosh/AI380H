"""Run or dry-run the MIMIC-IV cohort workflow with local ADC."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from uuid import uuid4

from google.cloud import bigquery
import pandas as pd

from cohort import build_cohort, partition
from features import LAB_ITEMS, assemble, feature_schema, verified_lab_mapping
from report import build_report, lab_range_audit, make_plots

DATASET = "physionet-data.mimiciv_3_1_hosp"
DEFAULT_PROJECT = "mimic-iii-iv-508014"
DEFAULT_DIRECTORY = Path.home() / "Library/Application Support/AI380H/MIMIC-IV"
DEFAULT_CAP = 10 * 1024**3
SQL_DIR = Path(__file__).resolve().parent / "sql"
REQUIRED_TYPES = {
    "patients": {"subject_id": "INTEGER", "gender": "STRING", "anchor_age": "INTEGER",
                 "anchor_year": "INTEGER", "dod": "DATE"},
    "admissions": {"subject_id": "INTEGER", "hadm_id": "INTEGER", "admittime": "DATETIME",
                   "dischtime": "DATETIME", "deathtime": "DATETIME", "admission_type": "STRING",
                   "admission_location": "STRING", "discharge_location": "STRING",
                   "insurance": "STRING", "hospital_expire_flag": "INTEGER"},
    "d_labitems": {"itemid": "INTEGER", "label": "STRING", "fluid": "STRING", "category": "STRING"},
    "labevents": {"labevent_id": "INTEGER", "subject_id": "INTEGER", "hadm_id": "INTEGER",
                  "itemid": "INTEGER", "charttime": "DATETIME", "storetime": "DATETIME",
                  "valuenum": "FLOAT", "valueuom": "STRING"},
}


def inspect_schema(client):
    dataset = client.get_dataset(DATASET, timeout=30)
    rows = []
    for table, required in REQUIRED_TYPES.items():
        metadata = client.get_table(f"{DATASET}.{table}", timeout=30)
        actual = {field.name: field.field_type for field in metadata.schema}
        for name, expected in required.items():
            if actual.get(name) != expected:
                raise ValueError(f"{table}.{name}: expected {expected}, got {actual.get(name)}")
        rows.append({"table": table, "source_rows": metadata.num_rows,
                     "used_fields": ", ".join(f"{name} {kind}" for name, kind in required.items())})
    return dataset.location, pd.DataFrame(rows)


def lab_params():
    items = [
        bigquery.StructQueryParameter(
            None, bigquery.ScalarQueryParameter("feature", "STRING", feature),
            bigquery.ScalarQueryParameter("itemid", "INT64", itemid),
            bigquery.ScalarQueryParameter("unit", "STRING", unit),
            bigquery.ScalarQueryParameter("allow_zero", "BOOL", allow_zero),
        ) for feature, itemid, _, _, unit, allow_zero in LAB_ITEMS
    ]
    return [bigquery.ArrayQueryParameter("lab_items", "STRUCT", items)]


def inspect_queries(client, location):
    plans = {}
    for name in ("patients", "cohort", "lab_itemids", "first_24h_labs"):
        sql = (SQL_DIR / f"{name}.sql").read_text()
        parameters = lab_params() if name == "first_24h_labs" else []
        dry = client.query(sql, location=location, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False, query_parameters=parameters,
        ))
        plans[name] = (sql, parameters, dry.total_bytes_processed)
    return plans


def run_workflow(client, location, plans, byte_cap=DEFAULT_CAP):
    oversized = {name: size for name, (_, _, size) in plans.items() if size > byte_cap}
    if oversized:
        raise RuntimeError(f"Dry run exceeds per-query cap of {byte_cap:,} bytes: {oversized}")
    frames, costs = {}, []
    for name in ("lab_itemids", "patients", "cohort", "first_24h_labs"):
        sql, params, estimated = plans[name]
        print(f"Running {name} (dry run {estimated:,} bytes)", flush=True)
        job = client.query(sql, location=location, job_config=bigquery.QueryJobConfig(
            query_parameters=params, maximum_bytes_billed=byte_cap,
        ))
        frames[name] = job.result().to_dataframe(create_bqstorage_client=False)
        costs.append({"query": name, "dry_run_bytes": estimated, "billed_bytes": job.total_bytes_billed,
                      "cache_hit": job.cache_hit})
        if name == "lab_itemids":
            mapping = verified_lab_mapping(frames[name])
    admissions, patients = frames["cohort"], frames["patients"]
    index, flow = build_cohort(admissions, patients)
    repeated, repeated_flow = build_cohort(admissions.iloc[::-1], patients.iloc[::-1])
    pd.testing.assert_frame_equal(index, repeated)
    pd.testing.assert_frame_equal(flow, repeated_flow)
    if index.empty or flow.admissions.sum() != len(admissions):
        raise ValueError("Empty cohort or source rows missing from eligibility flow")
    data = assemble(index, frames["first_24h_labs"])
    pd.testing.assert_frame_equal(data, assemble(repeated, frames["first_24h_labs"].iloc[::-1]))
    splits = partition(data)
    pd.testing.assert_frame_equal(splits, partition(data.iloc[::-1]))
    report = build_report(data, flow, splits, mapping)
    audit = {
        "lab_ranges": lab_range_audit(data),
        "lab_dictionary_unit_profile": frames["lab_itemids"],
        "admission_categories": admissions.groupby(
            ["admission_type", "admission_location", "discharge_location"], dropna=False
        ).size().reset_index(name="admissions"),
        "source_nulls": pd.concat({"admissions": admissions.isna().sum(),
                                   "patients": patients.isna().sum()}).rename(
            "missing"
        ).rename_axis(["source", "field"]).reset_index(),
        "query_costs": pd.DataFrame(costs),
    }
    return data, splits, flow, report, audit


def save_run(directory, schema, data, splits, flow, report, audit):
    directory = directory.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if directory.is_relative_to(repo):
        raise ValueError("Choose a private local output directory outside the repository")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_mode & 0o077:
        raise PermissionError(f"Set owner-only permissions on the output directory: chmod 700 {directory}")
    name = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix=".cohort-", dir=directory) as staging:
        work = Path(staging) / name
        work.mkdir(mode=0o700)
        data.to_parquet(work / "cohort.parquet", index=False)
        splits.to_parquet(work / "splits.parquet", index=False)
        (work / "feature_schema.json").write_text(json.dumps(feature_schema(), indent=2) + "\n")
        aggregate = work / "aggregates"
        aggregate.mkdir()
        for label, frame in {"source_schema": schema, **report}.items():
            frame.to_csv(aggregate / f"{label}.csv", index=False)
        private_audit = work / "private_audit"
        private_audit.mkdir()
        for label, frame in {"exact_flow": flow, **audit}.items():
            frame.to_csv(private_audit / f"{label}.csv", index=False)
        for label, fig in make_plots(data, flow, report).items():
            fig.savefig(aggregate / f"{label}.png", dpi=200)
        return work.rename(directory / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("MIMIC_BQ_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("MIMIC_OUTPUT_DIR", DEFAULT_DIRECTORY)
    ))
    parser.add_argument("--max-bytes-billed", type=int, default=DEFAULT_CAP)
    parser.add_argument("--run", action="store_true", help="Execute queries and write local artifacts")
    args = parser.parse_args()
    client = bigquery.Client(project=args.project)
    location, schema = inspect_schema(client)
    print(schema[["table", "source_rows"]].to_string(index=False), flush=True)
    plans = inspect_queries(client, location)
    for name, (_, _, size) in plans.items():
        print(f"{name}: {size:,} dry-run bytes", flush=True)
    if args.run:
        data, splits, flow, report, audit = run_workflow(
            client, location, plans, byte_cap=args.max_bytes_billed
        )
        output = save_run(args.output_dir, schema, data, splits, flow, report, audit)
        print(report["cohort_summary"].to_string(index=False))
        print(report["lab_missingness"].to_string(index=False))
        print(report["partitions"].to_string(index=False))
        print(f"Local run: {output}", flush=True)


if __name__ == "__main__":
    main()
