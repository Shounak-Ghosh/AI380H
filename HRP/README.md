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
