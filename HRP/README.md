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
5. Click **Run All**, or run cells individually. The first Qwen3 inference can be slower while the model loads.
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
- Baseline Qwen3 prompt vs SHAP-grounded Qwen3 prompt, using **only** the localhost Ollama API.
- Feature/direction agreement, plus manual review guidance.
- No MIMIC data or cloud LLM APIs used.

## Troubleshooting
- **Kernel missing:** install VS Code Python/Jupyter extensions, rerun `uv sync`, and choose the interpreter at `.venv/bin/python`.
- **Connection refused:** launch Ollama; `curl http://127.0.0.1:11434/api/tags` should respond.
- **Model not found:** `ol pull qwen3:8b` (or `ollama pull qwen3:8b`).
- **Slow inference:** close memory-intensive apps; the notebook disables thinking. You can try a smaller Ollama model if installed, after changing `MODEL_NAME`.
- **Dependency problems:** `uv sync --reinstall` reinstalls the locked dependencies.

## Privacy
This pilot uses independently generated fictional patients with artificial coefficients. For later MIMIC work, keep data and derived artifacts within an authorized environment and follow the applicable PhysioNet DUA.
