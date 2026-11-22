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

Splits use a seeded permutation of sorted unique patient IDs with 70/15/15
patient targets and seed 380. All encounters for a patient share a partition.

# Experiment 1 modeling and SHAP

From `HRP`, run `uv python install 3.12` and `uv sync --locked --python 3.12`.
On macOS, install the OpenMP runtime with `brew install libomp` for the
XGBoost CPU wheel. Open `src/02_readmission_models.ipynb` in VS Code with
the HRP `.venv/bin/python` kernel. Modeling runs locally from saved Parquet
files and uses neither a BigQuery connection nor Ollama.

The notebook defaults to `MODE = "review"`. Set `MODEL_RUN` to an explicit
private bundle or review the latest saved bundle under `OUTPUT_ROOT`.
Set `MODE = "train"` to run Experiment 1. Set `COHORT_RUN` (or the
`HRP_COHORT_RUN` environment variable) to a validated run path. With no
explicit path, the notebook selects the latest `run-*` with a readiness
manifest under `MIMIC_OUTPUT_DIR` and verifies its input checksums, outcome
counts and original patient partitions. `MODE = "regenerate"` uses the
saved fitted models with `RECOMPUTE_SHAP` and/or `REGENERATE_CASES` and
writes a new bundle with the same fitted model ID. Use the original
modeling code, locked software versions and cohort for regeneration.

The command-line entry point uses the same implementation.

```bash
uv run --locked python src/run_models.py review
uv run --locked python src/run_models.py train
uv run --locked python src/run_models.py regenerate --recompute-shap --regenerate-cases
uv run --locked python -m unittest discover -s tests
```

All three model commands support `--output-root`. Use `--cohort-run` to
pin a validated cohort and `--model-run` to pin a saved bundle for review
or regeneration. In an already synchronized environment,
`uv run --offline --no-sync` uses the installed packages offline.

## Fitting and evaluation

The saved cohort schema defines the predictor list and outcome. Both
pipelines fit median imputation and one-hot encoding on training data.
Logistic Regression uses L2 regularization and scales numeric predictors.
XGBoost uses CPU histogram trees and unscaled, median-filled predictors.
Both preserve the existing missingness indicators and use uniform class
weights. Select the best of two candidates in each family by validation
average precision, then Brier score and candidate order.

The validation analysis compares raw probabilities with out-of-fold
sigmoid forecasts in three patient-disjoint folds. Use sigmoid
calibration if it improves Brier score by at least 0.0001 and the fold
and final slopes are positive. Select the F1-maximizing validation
threshold, preferring the higher threshold in a tie. Favor XGBoost when
its validation AP gain is at least 0.005, it wins in a majority of
validation folds and its selected Brier score is within 0.001 of
Logistic Regression or better. Select the model and thresholds before
scoring the held-out test set. Report ROC-AUC, AP with test prevalence
as a reference, threshold metrics and calibration, with 95% percentile
intervals for ROC-AUC and AP from 300 patient-cluster bootstraps.
[scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

SHAP explains the raw model margin in log-odds. It uses a seeded sample
of 64 training admissions as the reference and up to 1,000 test
admissions for global summaries. Signed contributions for all one-hot
columns of a categorical feature are summed, with separate contributions
for explicit lab missingness indicators. The expected value plus those
grouped contributions must reconstruct the model margin within an
absolute tolerance of 0.0001. A sigmoid-adjusted risk, when selected,
is a separate output from the additive explanation.
[SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)

## Private output and Experiment 2 handoff

Bundles are written under `~/Library/Application Support/AI380H/MIMIC-IV/model-runs`
by default. Override with `HRP_MODEL_OUTPUT_DIR` or `OUTPUT_ROOT` to use
another owner-only directory outside Git and cloud sync. Each bundle
contains the two fitted pipelines, any selected sigmoid calibrators,
the source cohort and artifact checksums, software versions, validation
predictions, test results, encoded feature maps, evaluation plots, SHAP
reference and diagnostics, and the local case tables. Keep the bundle in
the authorized local environment for Experiment 2.

Experiment 2 can load the model metadata and private cases without
retraining. From the HRP Python environment, with `src` on `sys.path`:

```python
from pathlib import Path
import pandas as pd
from run_models import verify_run

run = Path("/absolute/path/to/private/models-run")
manifest = verify_run(run)
cases = pd.read_parquet(run / "explanation_cases.parquet")
contributions = pd.read_parquet(run / "case_contributions.parquet")
case_features = contributions.merge(cases, on="hadm_id", validate="m:1")
```

`explanation_cases.parquet` contains 24 admissions in each of the three
test-risk tertiles, with 72 distinct patients. It records the final
predicted risk, raw probability, original feature values, baseline,
raw margin, reconstruction, and model ID. `case_contributions.parquet`
contains one row per admission and original feature with its signed
value, direction and rank. Use `feature_metadata.json` for display
labels and units. `shap_reference.json` records the reference definition
and `case_summary.json` records sampling rules, outcomes and limitations.
Use `load_fitted` only with a trusted local bundle if inference is needed.
