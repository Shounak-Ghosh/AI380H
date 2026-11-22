# HRP local Qwen3 + SHAP pilot (fictional data only)

## Prerequisites
- M4 Mac Mini (16 GB) with Ollama and `qwen3:8b` (`ol list` or `ollama list`).
- VS Code with the **Python** and **Jupyter** extensions installed.
- `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)).

## Set up with uv in VS Code
1. Open this project folder using **File → Open Folder...**.
2. Open **Terminal → New Terminal** in VS Code and run:
   ```bash
   uv --version
   uv python install 3.12
   uv sync
   ```
   `uv sync` creates the project's `.venv` automatically and installs dependencies from `pyproject.toml`. It also writes `uv.lock` if one does not exist. You do **not** need `python -m venv`, `pip install`, or to activate the environment.
3. Start the **Ollama app**, then check:
   ```bash
   ol list                  # or: ollama list
   curl http://127.0.0.1:11434/api/tags
   ```
   If the app is not running, start it. Alternatively `ollama serve` runs the local server (do not start it a second time if already running).
4. Open `hrp_qwen3_shap_pilot.ipynb`. Click **Select Kernel** → **Python Environments** → select `./.venv/bin/python` (Python 3.12). The notebook kernel uses dependencies from uv's environment.
5. Click **Run All** to build the fictional cohort and SHAP references. Enable `RUN_SINGLE_CASE` for the original walkthrough or set `BENCHMARK_MODE` to `smoke` (12 cases) or `full` (72 cases) for local inference. The first Qwen3 inference can be slower while the model loads.
6. To check GPU usage after invoking the model: `ol ps`.

## Useful uv commands
```bash
uv sync                         # install/synchronize project dependencies
uv add package-name             # add a dependency and update pyproject.toml/uv.lock
uv remove package-name          # remove a dependency
uv run python -V                # run Python in project environment
uv run python -m pip --version  # not necessary; pip may not be installed
```

## Contents
- Artificial patient generation, logistic regression, genuine `shap.LinearExplainer` explanations and additivity checks.
- Baseline, SHAP-grounded, and SHAP-verified Qwen3 responses through local Ollama, with at most one revision per case.
- Balanced risk and SHAP sign strata across three fictional models, including two with reversed fitted coefficients.
- Ranked agreement, separate prose checks, paired summaries, and plots.
- No MIMIC data or cloud LLM APIs used.

Run the offline parser, scoring, and revision tests with `uv run python -m unittest discover -s tests -p test_faithfulness.py`.

Section 9 audits 12 patient triplets and rescores the recovered full run from `hrp_benchmark_responses.json`. Set the inference modes to off before using Run All for offline review. The audit includes the full explanation text, manual labels, parser coverage, and paired revision costs.

## Troubleshooting
- **Kernel missing:** install VS Code Python/Jupyter extensions, rerun `uv sync`, and choose the interpreter at `.venv/bin/python`.
- **Connection refused:** launch Ollama; `curl http://127.0.0.1:11434/api/tags` should respond.
- **Model not found:** `ol pull qwen3:8b` (or `ollama pull qwen3:8b`).
- **Slow inference:** close memory-intensive apps; the notebook disables thinking. You can try a smaller Ollama model if installed, after changing `MODEL_NAME`.
- **Dependency problems:** `uv sync --reinstall` reinstalls the locked dependencies.

## Privacy
This pilot uses independently generated fictional patients with artificial coefficients. For later MIMIC work, keep data and derived artifacts within an authorized environment and follow the applicable PhysioNet DUA.

# MIMIC-IV readmission cohort

## Setup and access

From `HRP`, use the existing Python 3.12 environment with `uv sync --locked`.
In VS Code, open `src/01_mimic_cohort.ipynb` and select the project’s
`.venv/bin/python` interpreter. The cohort uses BigQuery, pandas, NumPy,
PyArrow and Matplotlib from the locked environment.

Use a credentialed PhysioNet account with the MIMIC-IV data use agreement and
BigQuery dataset access. The ADC principal needs permission to create BigQuery
jobs on the billing project (for example, BigQuery Job User), a billing-enabled
project with the BigQuery API active, and dataset/table read access from
PhysioNet. Use the existing local ADC. If setting it up on another machine,
run `gcloud auth application-default login` using the approved account.
If Google requires a quota project, run
`gcloud auth application-default set-quota-project PROJECT_ID` with
`serviceusage.services.use` on that project.

The default billing project is `mimic-iii-iv-508014`. Override it with
`MIMIC_BQ_PROJECT`, the CLI `--project` flag, or the notebook configuration.
The input dataset is `physionet-data.mimiciv_3_1_hosp` in its reported BigQuery
location. [PhysioNet access and release notes](https://physionet.org/content/mimiciv/3.1/)

## Run or inspect

The notebook defaults to schema inspection and dry runs. Set `RUN_COHORT = True`
and run all cells to generate a new cohort. Set `EXISTING_RUN` to a saved run
path to inspect aggregate results without calling BigQuery. The CLI uses
the same code and SQL.

```bash
uv run python src/run_cohort.py
uv run python src/run_cohort.py --run
uv run python -m unittest discover -s tests
```

The four dry runs estimate about 15.28 GB combined at the observed source
sizes. Execution enforces a 10 GiB maximum billed per query, including the
unit audit and first-day lab extraction. To set a different budget, pass
`--max-bytes-billed` or edit the notebook setting. The query cost audit
records estimates, billed bytes and cache hits. Tests use fictional fixtures
and require no Google credentials. The existing Ollama connection smoke
script is available directly as `uv run python tests/test_local_model_connection.py`.

## Cohort contract

- **Index eligibility.** An adult at admission, using
  `anchor_age + year(admittime) - anchor_year`, with a stay of at least
  24 hours, known survival and no discharge to an acute hospital or psychiatric
  facility. The long-term acute care discharge category remains eligible.
  Keep repeat eligible stays. Reject missing IDs, all copies of
  duplicate admission IDs, missing/negative intervals, and remaining encounters
  for identifiable patients whose source timeline is compromised. Exclude both
  sides of overlaps and ties at admission start. Treat encounters with the same
  discharge and next admission instant as possible continuations.
- **Outcome.** Label the first subsequent admission as a qualifying observed
  event when it starts strictly after discharge and at or before 30 days,
  is `DIRECT EMER.`, `EW EMER.`, or `URGENT`, and has a known non-transfer
  source. Same-day returns with a positive gap qualify. Incoming hospital,
  internal psychiatric, and ambulatory-surgery transfers, overlapping returns,
  elective, same-day surgical and ambulatory observation encounters, and other
  ambiguous first returns in the window are separate exclusions. Skilled nursing
  facility referrals can qualify when the recorded admission type is acute.
  A known death by the end of day 30 censors follow-up unless the qualifying
  return occurs before or on the recorded death date. A later emergency after a
  competing first encounter is outside this endpoint.
- **History and baseline features.** Count valid observed encounters started
  before the index and discharged no later than index admission. Compute elapsed
  days from the most recent such discharge, missing for zero prior encounters.
  Use age, recorded gender, admission type/location and insurance as baseline
  features. Discharge information, date of death, outcome, identifiers, timestamps
  and validation columns are outside the predictor allowlist.
- **Labs.** Hematology WBC 51301, hemoglobin 51222, platelets 51265, and
  chemistry sodium 50983, potassium 50971, creatinine 50912, BUN 51006,
  glucose 50931 and bicarbonate 50882. The v3.1 item dictionary lists Blood
  for all nine. These are CBC and routine blood chemistry definitions. In the
  unit audit, the uncommon duplicate chemistry codes for creatinine, glucose,
  potassium, sodium and BUN had no linked admission IDs. Whole-blood blood
  gas or alternative methods are excluded. Official release notes document
  that v3.1 corrected unintended item-ID changes in v3.0 to match v2.2.
  This pipeline verifies the current dictionary and observed canonical units
  for each run. [MIMIC-IV v3.1 release notes](https://physionet.org/content/mimiciv/3.1/#release-notes)
- **Lab timing.** Require matching patient and hospital admission IDs,
  admission ≤ chart time ≤ store time ≤ landmark. Use store time as a proxy
  for result availability and select the earliest eligible chart time.
  Resolve ties by store time, event ID and numeric value. Use verified units
  without conversion. Exclude nonfinite, nonnumeric, negative and out-of-unit
  values, and zeros outside cell counts. Preserve absent results and per-lab
  missingness indicators. The first live run showed an upper tail in several
  analytes, including bicarbonate and glucose. The private range audit records
  central percentiles, extrema and review-threshold counts. These records have
  no independent chart adjudication, so this cohort retains finite values in
  the verified unit without upper clipping. [MIMIC-IV laboratory timestamp documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)

Neither absence of a recorded return nor absence of a recorded death proves
complete 30-day observation. MIMIC-IV captures one system and has a finite
accrual window. Patient-level date shifting prevents applying a single maximum
shifted calendar timestamp across patients. Observation exclusions and the
24-hour stay requirement condition the population, limiting generalizability.
The date-based age proxy and mapped older anchor ages are approximate. The
source does not provide historical snapshots of administrative fields or
definitive bedside visibility times for lab results.

## Revalidate a saved run

Run `uv run python src/validate_cohort_run.py` to verify the latest private
run without querying BigQuery. Use `--run-dir` for an explicit run. The command
checks the predictor contract, saved timing bounds, missingness indicators,
seeded splits and descriptive outputs, then writes
`private_audit/readiness_summary.json`.

The optional `--check-source` flag re-reads only the admission and patient
timelines, with a 100 MB billing cap per query. It verifies the primary
cohort row for row and reports a sensitivity analysis. That analysis adds
positive cases with a later qualifying non-overlapping return by day 30,
on or before recorded death, after a primary exclusion for an immediate,
overlapping, transfer, planned or ambiguous first return. Indices with
unresolved follow-up stay excluded. The sensitivity analysis is descriptive
and leaves the primary cohort, labels and partitions intact.

## Local artifacts and Experiment 1

On macOS the default output directory is
`~/Library/Application Support/AI380H/MIMIC-IV`. Each run gets an owner-only
subdirectory. For another local volume, set `MIMIC_OUTPUT_DIR` or
`--output-dir` to an owner-only directory outside Git and cloud sync.
The restricted run holds `cohort.parquet`, `splits.parquet`,
`feature_schema.json`, `private_audit/` and `aggregates/`. The notebook never
displays patient rows. Clear outputs before committing and keep patient-level
extracts, model artifacts and audit files local. Aggregate tables mask small
positive cells below 11 and apply complementary suppression to count tables.
Review aggregates, plots and combinations of tables under the applicable
PhysioNet DUA before sharing.

In `02_readmission_models.ipynb`, point to the local run directory.

```python
import json
from pathlib import Path
import pandas as pd

run = Path("/absolute/path/to/run-directory")
schema = json.loads((run / "feature_schema.json").read_text())
cohort = pd.read_parquet(run / "cohort.parquet")
splits = pd.read_parquet(run / "splits.parquet")
dataset = cohort.merge(splits, on=["subject_id", "hadm_id"], validate="1:1")
predictors = [feature["name"] for feature in schema["features"]]
train = dataset.loc[dataset["partition"].eq("train")]
X_train = train[predictors]
y_train = train[schema["label"]]
```

Splits use a seeded permutation of sorted unique patient IDs with 70/15/15
patient targets and seed 380. All encounters for a patient share a partition.
Prevalence is reported after splitting. Fit imputation, encoding, scaling or
feature selection using training data only.
