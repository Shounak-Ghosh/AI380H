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

### Pinned inputs for Experiment 2

Use model run `models-20261122T230904Z-511d7f06` and cohort run
`run-20261122T212339Z-c71630d2`. The loader looks for that exact model run
under `HRP_MODEL_OUTPUT_DIR` (or `MIMIC_OUTPUT_DIR/model-runs`) and verifies
its full manifest digest and artifact checksums. An explicit `model_root`
can point to another copy of the same bundle. The loader does not select
the newest run. It reads the validated cases and references without loading
fitted models, refreshing labels, or regenerating explanations.

| Pinned file | SHA-256 |
|---|---|
| Model `manifest.json` | `072840d4f1b0ea5a3d3b26311de8a9d275d92751d1655201ce36c4fe46ab0756` |
| Source `cohort.parquet` | `c622ff49a264ecfc027d014e7dc23f5cea91cf1b518fe8c8fa4ead1fbcc188da` |
| Source `feature_schema.json` | `769177bf3ee85cf22b2e95418e24a8cb80c8d0b1771d8af401738e430edc587f` |

The run used Python 3.12.11, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1
and pandas 2.3.3. The manifest also records the other package versions,
the split-file checksum, the lockfile checksum and the modeling code hashes.

From `03_llm_experiments.ipynb`, with `HRP/src` on `sys.path`:

```python
from experiment2_inputs import load_explanation_cases

provenance, cases = load_explanation_cases()
case = cases[0]
baseline = case.inputs("A")
grounded = case.inputs("B")
to_verify = case.inputs("C")
```

Use `case.case_id` for pairing runs locally. The loader uses
`feature_metadata.json` for readable feature names and units,
`explanation_cases.parquet` for the original values and final predicted
risk, `case_contributions.parquet` for the signed ranks, and
`shap_reference.json` for the reference metadata. The inputs omit observed
outcomes and MIMIC identifiers. A missing original value is represented
as `null`, with its separate model indicator retained where applicable.
Keep the case payloads in the authorized local environment.

The saved XGBoost run uses the raw classifier probability. Its logistic
link maps the raw log-odds margin to the displayed risk. A candidate
sigmoid was evaluated and left inactive. If reading the source table
directly, `predicted_risk` is the final probability, `raw_probability` is
the classifier probability and `raw_margin` is the pre-link log-odds.
SHAP decomposes the raw margin as expected value plus the sum of grouped
contributions. SHAP magnitudes are in log-odds. They cannot be read as
percentage-point changes in risk. A positive contribution raises the raw
margin relative to the training reference and a negative contribution
lowers it. A clinically plausible explanation can still disagree with
the fitted model. Attribution describes model behavior and establishes no
causal clinical effect.

| Condition | Experiment 2 input and release rule |
|---|---|
| A — Baseline | Supply only the observed feature values and final predicted risk. |
| B — SHAP-grounded | Supply the same values and risk plus the expected value, output scale, and contributions ordered by absolute magnitude. Return the initial explanation. |
| C — SHAP-verified | Supply the same input as B. Check the draft's driver claims against the saved original-feature keys, ranks and signs. Reject unsupported, reversed or causal claims. Allow one revision with the specific discrepancies, validate the revision, and use a deterministic explanation from the saved attributions if it still fails. Record the validation and fallback outcome separately from the explanation. |

Use the same response shape for all three conditions. Ask for an
`explanation` paragraph of two or three sentences, at most 90 words, and a
`drivers` list of up to three ordered records with `feature` (the supplied
feature key) and `direction` (`positive` or `negative`). The paragraph
includes the estimated 30-day risk, describes the listed drivers as model
associations and avoids treatment advice. For C, use the structured list
for exact attribution checks and review the paragraph for unsupported
wording. Passing the structured check alone cannot establish faithfulness
of every sentence. Apply the same length and style to the verified fallback.

### Reproducibility and review

Reuse the saved 72 cases and contribution rows. The original sampler ranks
the held-out test set by final risk, breaks probability ties by admission ID,
and divides the ranking into tertiles. It samples 144 candidates per tertile
with seed 382 and selects 24 distinct patients in each using seed 383.
Selection rotates across lab availability, contribution direction patterns
and leading features. It sets no quotas for those attributes. The reference
is a seed-380 sample of 64 training admissions and the SHAP reconstruction
tolerance is 0.0001 log-odds. To reproduce the selection for an audit,
use `generate_cases()` with the pinned fitted model, source cohort, settings
and saved test probabilities. The saved case order and references are the
inputs for all downstream comparisons.

The cases are a risk-stratified evaluation set from one hospital system.
Case frequencies do not estimate cohort prevalence or model performance
in a new population. Keep the model bundle and case-level artifacts local.
The modeling notebook retains its evaluation outputs, including three
individual-admission SHAP charts with feature labels and signed
contributions. Review those charts in the authorized course context before
any public redistribution. [MIMIC-IV data use agreement](https://physionet.org/content/mimiciv/view-dua/3.1/)

## Experiment 2 — LLM Explanation Faithfulness

[03_llm_experiments.ipynb](src/03_llm_experiments.ipynb) compares baseline
Qwen3 explanations (A), SHAP-grounded drafts (B), and the verified
system output (C) on the pinned 72-case XGBoost sample described above.
C checks the exact B draft, allows one correction, and uses a
deterministic explanation if the correction fails.

### Review and analysis

The notebook defaults to `MODE = "review"` and makes no Ollama requests.
It reads the completed private `MIMIC-IV/checkpoint.json` and verifies
the pinned provenance, case fingerprints, versions and paired records.
Set `REVIEW_CHECKPOINT` to review another completed owner-only checkpoint.
The historical checkpoint stays at its original location.

The results cover structured agreement, narrative-verifier coverage,
interventions, and inference latency. Paired differences include fixed-seed
95% bootstrap intervals and a sensitivity analysis that scores invalid
outputs as failures on bounded attribution metrics. The original run used
216 requests and produced 214 valid final outputs. Eight Qwen3 revisions
passed. Sixty-four C outputs used deterministic fallback.

Leave `ANALYSIS_EXPORT_DIRECTORY` unset for read-only review. Set it to a
new private directory outside Git to export tables, aggregate figures,
the source-checkpoint checksum and versioned metadata. Case-level scores
and manual-review selection remain in the owner-only analysis bundle.
Generated prose and patient-level inputs stay inside the authorized
local environment.

### Fictional validation and real execution

Use `MODE = "fictional"` or the smoke command to validate the workflow with
scripted responses and fabricated cases. The offline suite needs neither
Ollama nor MIMIC-IV data.

```bash
uv run --locked python -m unittest discover -s tests
uv run --locked python src/experiment2_smoke.py
```

Use `MODE = "real"` only to start or resume a benchmark. Set
`REAL_RUN_DIRECTORY` to a dedicated absolute directory outside Git, such
as `~/Library/Application Support/AI380H/MIMIC-IV/llm-runs/experiment2-qwen3-replication`
after expanding `~`. The notebook requires a local model digest and uses
the pinned inputs. The same run directory resumes a pause atomically.
A completed checkpoint is reviewed without new inference. Real execution
does not target the original checkpoint's directory.

### Interpretation limits

A and B permit abstention. C requires the full set of non-negligible
top-three drivers and explicit supported directions in prose. Ambiguous or
unparsed prose triggers intervention and lowers parser coverage. It is
unverified rather than automatically false. Report accepted revisions
separately from deterministic fallbacks. The 72-case sample is deliberately
risk-stratified within one hospital system. Attribution describes model
behavior relative to the reference and supports no causal or clinical
deployment conclusions.

## Experiment 3 — Adversarial Faithfulness Challenge

[04_adversarial_faithfulness.ipynb](src/04_adversarial_faithfulness.ipynb)
tests whether Qwen3 tracks the current model when attributions conflict
with prespecified intuitive expectations. Experiment 3 uses fictional
profiles with age (years), prior admissions (count), creatinine (mg/dL),
hemoglobin (g/dL), sodium (mmol/L), and heart rate (bpm). The synthetic
logistic models adapt the pilot's per-unit weights. One variant flips
creatinine (+0.80 to −0.80) and admissions (+0.44 to −0.44). Another flips
age (+0.032 to −0.032) and heart rate (+0.014 to −0.014). The other slopes
within each pair are shared. A common expected margin of −1.45 determines
each model's raw intercept from the same background means.

Seed 380 generates disjoint candidate pools and a shared 1,024-profile
background. Interventional LinearExplainer references are tested against
the linear formula, probability reconstruction and margin additivity
at absolute tolerance 1e-8. Both targeted features must change from at
least +0.12 to at most −0.12 raw log-odds. Matching keeps feature values
fixed and supplies the model-specific probabilities. The probability can
therefore indirectly cue a model change even in A.

Within each reversal family, sampling cycles through eligible strata
defined by pair-mean risk (cutoff 25%), mixed direction in either
variant, and conflicting importance in either variant. Mixed direction
requires at least one contribution of +0.12 or more and one of −0.12 or
less among the top three. Conflicting importance requires creatinine or
age, depending on the family, outside the top three with magnitude at
most 0.16 and an absolute gap of at least 0.24 from the leading feature.
The manifest reports eligibility, selection, model-specific attributions
and the overlapping challenge attributes.

A receives the feature values, labels, units and probability. B also
receives SHAP. C checks the exact B draft using the Experiment 2 verifier.
C can request one Qwen3 revision and use the deterministic fallback.
Separate results distinguish Qwen3's accepted revisions from final
system delivery. The analysis reports signed top-three agreement, rank,
structured and narrative errors, explicit intuitive-direction
substitution, and within-profile direction switches. The paired
counterintuitive-minus-intuitive contrasts use a 10,000-replicate
profile-cluster bootstrap (seed 380). A sensitivity analysis counts
terminal failures as zero on bounded agreement metrics. Omitted
features, explicit abstentions, and unverifiable claims are reported.
The narrative verifier's grammar can miss claims or leave them
ambiguous. An independent fictional probe set illustrates those limits.

### Reference-run findings

The matched sample contains 48 synthetic profiles, 96 model–profile cases,
and 288 A/B/C records. Maximum saved SHAP additivity error was
3.55e-15 log-odds. Negative paired differences mean lower signed
top-three agreement under counterintuitive model behavior.

| Condition | Intuitive mean | Counterintuitive mean | Complete pairs | Paired difference (95% CI) |
|---|---:|---:|---:|---:|
| A · Baseline | 1.659 | 0.652 | 44 | −1.000 [−1.227, −0.762] |
| B · SHAP-grounded | 2.958 | 2.729 | 48 | −0.229 [−0.479, −0.021] |
| C · Verified system | 3.000 | 3.000 | 48 | 0.000 [0.000, 0.000] |

The reference run supports H3 in A and shows a smaller adverse effect
in B. The failure-inclusive A estimate was −0.896. B and C did not
change. C required intervention on 48 of 48 cases in each variant.
Qwen3 supplied seven accepted revisions (four intuitive, three
counterintuitive). The other 89 outputs used deterministic fallback.
All 96 C outputs were verified complete.

Structured clinical-expectation substitution appeared in 71/96
counterintuitive target opportunities in A, 10/96 in B, and 0/96 in C.
These counts describe explicit JSON directions. The combined
structured-and-prose rubric found 11/96, 0/96, and 0/96, respectively,
and marked 57/96 grounded target opportunities unverifiable. An
independent reading of the saved matched example identified misleading
prose alongside correct JSON directions.

The supplemental risk-band analysis uses saved pairs only. The
baseline difference was more adverse in bands with larger absolute
probability changes. Differences in B remained negative across bands
without increasing monotonically. The switch holds features fixed
and changes both attributions and predicted risk. The results cannot
isolate a causal effect of the probability cue or of the sign reversals
alone.

### Offline review and provenance

The notebook defaults to auto mode. It reviews a completed run offline
or displays setup without requesting inference. Explicit fictional mode
exercises scripted A/B/C responses. Explicit real mode starts or resumes
local Qwen3 8B on the full synthetic sample. The runner validates the
model digest, input fingerprints, shared implementations, and scenario
manifest checksum. Set an optional cumulative call cap with
`--max-calls` and a per-invocation limit with `--runtime-seconds`.
A recorded failed request is a terminal observation. An interruption
during an uncommitted Ollama response may require repeating that call.

```bash
uv run --locked python -m unittest discover -s tests
uv run --locked python src/adversarial_runner.py review
```

Review leaves the saved files untouched by default. The reference
checkpoint SHA-256 is
`52ecf7c17a9cfd9108c264af7e8f0886a1754be23edfb261597deb1cf6052a1b`.
Its canonical scenario fingerprint is
`68387030b1bd39b0bc243a81704624a0bc9c1f19bb2bac8edce09420693a7478`.
The reference uses checkpoint format 2, runner `experiment3-runner-v1`,
scenarios `experiment3-scenarios-v1`, prompt `experiment2-v3`, verifier
`experiment2-verifier-v3`, scorer `experiment2-scorer-v1`, and saved
analysis `experiment3-analysis-v2`. The notebook computes the new
risk-band supplement in memory.

Private run files are owner-only, outside Git, and default to
`~/Library/Application Support/AI380H/MIMIC-IV/llm-runs/experiment3-adversarial/`
in the respective `qwen3-v1` and `fictional-v1` directories. Set
`--directory` to use a different dedicated private directory. The
bundle includes a scenario manifest, resumable checkpoint, case and
aggregate analyses, five figures and a checksummed run index.
Review exports require an explicit `--save` or notebook
`SAVE_BUNDLE = True`.

The optional aggregate-only screen reads the pinned XGBoost sample
from existing private SHAP files. Its flags are exploratory and remain
outside the matched synthetic H3 analysis. The synthetic coefficients
encode experimental assumptions and support no clinical or causal
conclusions. Neither pipeline requires changes to Experiment 2.

Experiment 4 should use a separately labeled verifier challenge set
to study false rejections of faithful prose, missed contradictions,
correct structured-error rejections, and ambiguity. Evaluate model
revision separately from fallback, and measure readability and
informativeness under constrained explanation formats.

## Experiment 4 — Independent Verifier Effectiveness Evaluation

[05_verifier_evaluation.ipynb](src/05_verifier_evaluation.ipynb) tests H4
on a fixed fictional benchmark. The hypothesis is that the deterministic
verifier catches structured attribution errors more reliably than errors
phrased in unfamiliar prose and will withhold some mathematically faithful
explanations. The [rubric](experiment4/rubric.md) defines mathematical
faithfulness, completeness and schema validity separately. Labels come from
authored claims, controlled mutations and validated synthetic SHAP
references. They are assigned before running the verifier.

The design has 60 faithful complete responses, 60 authored mathematical
corruptions and 60 boundary cases. The primary binary cohort includes the
first two groups. The boundary analysis and an all-eligible sensitivity
analysis cover faithful incompleteness, indeterminate prose, schema errors
and additional errors. A positive is gold-unfaithful. Explicit detection
requires a contradicted or unsupported claim. The release policy counts
all withhold decisions, including ambiguity and incompleteness. Report
both confusion matrices, false positive and false negative rates, missed
error categories and matched false rejection. Confidence intervals
resample whole fictional profiles.

### Offline results and reproduction

The notebook defaults to review and loads the
[checksummed results](experiment4/v1/results/manifest.json) without Ollama.
Evaluate replays the pinned source hashes and fixed challenge. Setup
validates the inputs and labels. Fictional mode runs the scripted correction
check. The results directory contains gold-separated case traces, analyses,
a blinded review worksheet with a separate crosswalk, and seven figures
in PNG and PDF.

```bash
uv run --locked python src/verifier_challenge_set.py
uv run --locked python src/verifier_evaluation.py review
uv run --locked python src/verifier_evaluation.py evaluate --save
uv run --locked python src/verifier_evaluation.py fictional
uv run --locked python -m unittest discover -s tests
```

In the reference run, explicit checking found 52 of 60 erroneous responses
and flagged zero of 60 faithful responses. The gate withheld all 60 errors
and 36 of 60 faithful responses. Eight mathematical errors were withheld
as ambiguous. The standalone narrative verifier explicitly detected 36 of
48 narrative-error cases, and the structured layer detected all 16
structured-error cases. Seven of eleven scripted cases used deterministic
fallback. Final delivery was faithful in all eleven and complete in ten.
The accepted incomplete script omitted the predicted probability.

### Optional local correction and annotation

Live mode is opt-in. Use an already-running local Ollama server with the
recorded Qwen3 8B digest and a dedicated owner-only directory outside Git.
The frozen plan selects ten IDs each from rejected mathematical errors,
faithful prose and faithful incomplete responses. Settings are temperature
0, seed 42, the existing correction prompt and at most one generation
request per example. The checkpoint records requests before generation and
keeps returned traces private. Its cumulative cap supports pause and resume.
The completed live supplement used 30 selected fictional drafts, ten each
with mathematical errors, faithful prose withheld by the gate, and faithful
but incomplete responses. One human reviewer evaluated revisions against
the frozen rubric and synthetic SHAP references while blinded to verifier
decisions, as reported in the study procedure. The reviewer judged 28 of 30
revisions faithful, complete and valid. The gate accepted 16 of those
revisions and withheld 12 (42.9%). It withheld both unfaithful revisions,
each with incorrect driver order. One reviewer, two unfaithful revisions,
and a selected fictional sample limit generalization. Blinding reduces
one potential source of bias, and inter-reviewer agreement was not measured.

The notebook loads the [published live aggregate](experiment4/v1/results/live_annotation_summary.json)
without accessing private files. To verify the aggregate against the saved
sources, set the three paths to your existing private files and run:

```bash
uv run --locked python src/verifier_live_annotation.py validate \
  --checkpoint "$LIVE_CHECKPOINT" \
  --annotations "$LIVE_ANNOTATIONS" \
  --crosswalk "$LIVE_CROSSWALK"
```

Validation is read-only and reports unavailable inputs separately. Use the
explicit `export` action with the same paths to regenerate the aggregate
after reviewing discrepancies. Both actions use saved records and make no
model requests. Keep the checkpoint, annotation workbook and crosswalk
outside Git.

Optional saved-prose audit settings in the notebook create owner-only
worksheets from the pinned Experiment 2 or 3 runs. The benchmark requires no
MIMIC-IV data, and the saved-prose audit exports no patient-level text to
Git. The earlier verifier, checkpoints and scores are preserved. Experiment
5 can use the blinded worksheet to measure reviewer agreement and assess
faithfulness, clarity and readability. The separate saved-prose analysis remains pending.
