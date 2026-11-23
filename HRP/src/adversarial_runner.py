"""Experiment 3 setup, scripted validation, explicit local inference and offline review."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

from adversarial_analysis import ANALYSIS_VERSION, analyze, audit_fictional_claims
from adversarial_scenarios import FAMILIES, build_manifest, explanation_cases, validate_manifest
from experiment2_analysis import read_checkpoint
from experiment2_scoring import SCORER_VERSION
from llm_runner import CHECKPOINT_VERSION, ModelConfig, Ollama, _locked_directory, _save, run_experiment
from llm_verification import (
    PROMPT_VERSION, VERIFIER_VERSION, build_prompt, case_fingerprints, deterministic_answer, fingerprint,
)


RUNNER_VERSION = "experiment3-runner-v1"
DEFAULT_ROOT = Path.home() / "Library/Application Support/AI380H/MIMIC-IV/llm-runs/experiment3-adversarial"
FICTIONAL_CONFIG = ModelConfig(model="experiment3-scripted-v1")
VERSIONS = ("llm_runner.py", "llm_verification.py", "narrative_verifier.py", "experiment2_scoring.py")
DIGEST = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def _provenance(manifest, mode):
    return {
        "experiment": 3, "runner_version": RUNNER_VERSION, "mode": mode,
        "scenario_manifest_sha256": fingerprint(manifest),
        "scenario_version": manifest["version"],
        "shared_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in VERSIONS},
    }


def _fictional_manifest(manifest):
    chosen = {}
    for row in manifest["records"]:
        ranks = {ref["feature"]: ref["rank"] for ref in row["shap"]["contributions"]}
        if row["variant"] == "intuitive" and any(ranks[k] <= 3 for k in row["targeted_features"]):
            chosen.setdefault(row["family"], row["profile_id"])
    if chosen.keys() != FAMILIES.keys():
        raise ValueError("The scripted exercise requires an eligible pair per family.")
    records = [row for row in manifest["records"] if row["profile_id"] in chosen.values()]
    selection_counts = []
    for count in manifest["selection_counts"]:
        category = count["category"]
        selected = sum(row["family"] == count["family"] and row["variant"] == count["variant"]
                       and (category == row["variant"] or row["attributes"][category]) for row in records)
        selection_counts.append({**count, "selected": selected})
    return {**manifest, "selection_counts": selection_counts, "records": records,
            "sampling_strata": [], "scripted_subset_of": fingerprint(manifest)}


def scripted_inference(manifest):
    """Map prompts to canned responses so the exercise also works after resume."""
    responses, revisions = {}, {}
    for row, case in zip(manifest["records"], explanation_cases(manifest), strict=True):
        good = deterministic_answer(case)
        baseline = replace(good, explanation=good.explanation.split(". ", 1)[0] +
                           ". The model contributions are unknown.", drivers=())
        raw = json.dumps(asdict(good))
        draft = raw
        revision = raw
        if row["variant"] == "counterintuitive":
            target = next(driver.feature for driver in good.drivers if driver.feature in row["targeted_features"])
            wrong = replace(good, explanation=good.explanation.replace(f"{target} decreases", f"{target} increases"),
                            drivers=tuple(replace(driver, direction="positive") if driver.feature == target else driver
                                          for driver in good.drivers))
            draft = json.dumps(asdict(wrong))
            if row["family"] == "age-physiology":
                revision = draft
        responses[build_prompt(case, "A")] = json.dumps(asdict(baseline))
        responses[build_prompt(case, "B")] = draft
        revisions[build_prompt(case, "B")] = revision

    def infer(prompt, timeout):
        if prompt in responses:
            return responses[prompt]
        return revisions[prompt.split("\n\nRevise the previous draft", 1)[0]]

    return infer


def _run_directory(path):
    path = Path(path).expanduser().resolve()
    if path.name == "MIMIC-IV" or not path.name:
        raise ValueError("Use a dedicated Experiment 3 run directory.")
    return path


def _check_directory(directory, provenance, *, create=False, manifest=None):
    marker = directory / "scenario_manifest.json"
    if directory.exists() and not marker.exists() and any(
            item.name != ".run.lock" for item in directory.iterdir()):
        raise ValueError("Choose a dedicated Experiment 3 directory with no existing run artifacts.")
    with _locked_directory(directory) as output:
        marker = output / "scenario_manifest.json"
        entries = set(p.name for p in output.iterdir()) - {".run.lock"}
        if marker.exists():
            if marker.is_symlink() or fingerprint(json.loads(marker.read_text())) != provenance["scenario_manifest_sha256"]:
                raise ValueError("Incompatible Experiment 3 scenario manifest.")
        elif entries:
            raise ValueError("Choose a dedicated Experiment 3 directory with no existing run artifacts.")
        elif create:
            _save(marker, manifest)


def _saved_run(directory, manifest, mode, config, *, request_timeout, completed):
    state, checksum = read_checkpoint(directory / "checkpoint.json")
    saved = state["manifest"]
    configuration = saved["configuration"]
    recorded = ModelConfig(**configuration["model"])
    if mode == "real":
        if (not DIGEST.fullmatch(recorded.digest or "")
                or recorded != replace(config, digest=recorded.digest)
                or (config.digest is not None and recorded.digest != config.digest)):
            raise ValueError("Incompatible or unpinned saved Qwen3 model configuration.")
    elif recorded != FICTIONAL_CONFIG:
        raise ValueError("Incompatible scripted configuration.")
    if (saved["provenance"] != _provenance(manifest, mode)
            or saved["checkpoint_version"] != CHECKPOINT_VERSION or saved["prompt_version"] != PROMPT_VERSION
            or saved["verifier_version"] != VERIFIER_VERSION or saved["scorer_version"] != SCORER_VERSION
            or saved["cases"] != case_fingerprints(explanation_cases(manifest))
            or configuration["inference"] != ("ollama" if mode == "real" else "custom")
            or configuration["generation"] != {"format": "json", "stream": False, "think": False}
            or (request_timeout is not None and configuration["request_timeout"] != request_timeout)):
        raise ValueError("Incompatible Experiment 3 checkpoint, inputs or versions.")
    if completed and (state["pause_reason"] is not None or any(
            row["status"] == "pending" for row in state["records"].values())):
        raise ValueError("Review requires a completed run. Resume it in the original execution mode.")
    for row in state["records"].values():
        if row["condition"] != "C" or row["status"] != "ok":
            continue
        trace = row["verification"]
        accepted = {
            "grounded_draft": trace["draft_accepted"],
            "model_revision": trace["revision_accepted"],
            "deterministic_fallback": trace["fallback_used"],
        }
        if (trace["status"] != "verified_complete" or sum(accepted.values()) != 1
                or not accepted.get(trace["source"])
                or trace["revision_requested"] != ("revision_call" in row)
                or trace["revision_call"] != row.get("revision_call")):
            raise ValueError("Inconsistent Condition C verification trace.")
    return state, checksum, recorded


def save_results(directory, manifest, result, checkpoint_sha256, mode):
    """Save a checksum-indexed set of private, derived artifacts."""
    from adversarial_figures import build_figures
    import matplotlib.pyplot as plt

    # Keep the checkpoint and scenario files in place and commit the bundle index last.
    with _locked_directory(directory) as output:
        if hashlib.sha256((output / "checkpoint.json").read_bytes()).hexdigest() != checkpoint_sha256:
            raise ValueError("Checkpoint changed while computing the analysis. Repeat review.")
        _save(output / "analysis.json", result)
        artifacts = ["scenario_manifest.json", "checkpoint.json", "analysis.json"]
        for name, figure in build_figures(result).items():
            target = output / f"{name}.png"
            fd, temp = tempfile.mkstemp(dir=output, suffix=".png")
            os.close(fd)
            try:
                figure.savefig(temp, dpi=180)
                os.replace(temp, target)
            finally:
                Path(temp).unlink(missing_ok=True)
                plt.close(figure)
            artifacts.append(target.name)
        _save(output / "run_manifest.json", {
            "experiment": 3, "mode": mode, "complete": "pending" not in result["operations"]["statuses"],
            "runner_version": RUNNER_VERSION, "analysis_version": ANALYSIS_VERSION,
            "checkpoint_sha256": checkpoint_sha256, "scenario_sha256": fingerprint(manifest),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "shap", "scipy")},
            "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifacts},
        })


def notebook_run(mode="auto", *, directory=None, config=ModelConfig(),
                 max_calls=None, runtime_seconds=14_400, request_timeout=90, save=False, replicates=10_000):
    """Default to offline setup/review. Real inference requires an explicit mode."""
    if mode not in ("auto", "setup", "review", "fictional", "real"):
        raise ValueError("Choose auto, setup, review, fictional, or real.")
    directory = _run_directory(directory or DEFAULT_ROOT / ("fictional-v1" if mode == "fictional" else "qwen3-v1"))
    checkpoint = directory / "checkpoint.json"
    if mode == "auto":
        mode = "setup"
        if checkpoint.exists():
            state, _ = read_checkpoint(checkpoint)
            if state["pause_reason"] is None and all(r["status"] != "pending" for r in state["records"].values()):
                mode = "review"
    if mode == "review":
        state, _ = read_checkpoint(checkpoint)
        execution_mode = state["manifest"]["provenance"].get("mode")
        if execution_mode not in ("real", "fictional"):
            raise ValueError("This checkpoint is not an Experiment 3 run.")
        manifest = json.loads((directory / "scenario_manifest.json").read_text())
        validate_manifest(manifest)
        state, checksum, _ = _saved_run(directory, manifest, execution_mode, config,
                                        request_timeout=None, completed=True)
    else:
        manifest = build_manifest()
        if mode == "setup":
            if save:
                _check_directory(directory, _provenance(manifest, "real"), create=True, manifest=manifest)
            return {"mode": mode, "manifest": manifest, "result": None, "directory": directory}
        execution_mode = mode
        if mode == "fictional":
            manifest = _fictional_manifest(manifest)
        cases = explanation_cases(manifest)
        provenance = _provenance(manifest, mode)
        _check_directory(directory, provenance, create=True, manifest=manifest)
        recorded = config
        if checkpoint.exists():
            old, _, recorded = _saved_run(directory, manifest, mode, config,
                                           request_timeout=request_timeout, completed=False)
        else:
            old = None
        if old is None or any(r["status"] == "pending" for r in old["records"].values()):
            if mode == "real":
                if config.model != ModelConfig().model:
                    raise ValueError("Experiment 3 real mode requires qwen3:8b.")
                infer = Ollama.discover(recorded)
                if not DIGEST.fullmatch(infer.config.digest or ""):
                    raise ValueError("The installed local model must report a valid digest.")
                recorded = infer.config
            else:
                infer, recorded = scripted_inference(manifest), FICTIONAL_CONFIG
            run_experiment(cases, infer, directory, model_config=recorded, provenance=provenance,
                           max_calls=max_calls if max_calls is not None else 3 * len(cases),
                           runtime_seconds=runtime_seconds, request_timeout=request_timeout)
        state, checksum, _ = _saved_run(directory, manifest, mode, config,
                                        request_timeout=request_timeout, completed=False)
    result = analyze(manifest, state, replicates=replicates)
    result["fictional_verifier_probes"] = audit_fictional_claims(explanation_cases(manifest)[0])
    if save:
        save_results(directory, manifest, result, checksum, execution_mode)
    return {"mode": mode, "execution_mode": execution_mode, "manifest": manifest, "state": state,
            "result": result, "directory": directory, "checkpoint_sha256": checksum}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "review", "fictional", "real"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--runtime-seconds", type=float, default=14_400)
    parser.add_argument("--digest", help="Optional expected Qwen3 digest.")
    parser.add_argument("--save", action="store_true", help="Export derived files when reviewing a run.")
    args = parser.parse_args()
    run = notebook_run(args.mode, directory=args.directory, max_calls=args.max_calls,
                       runtime_seconds=args.runtime_seconds, config=ModelConfig(digest=args.digest),
                       save=args.mode != "review" or args.save)
    print(json.dumps({"directory": str(run["directory"]), "model_profiles": len(run["manifest"]["records"]),
                      "operations": run["result"]["operations"] if run["result"] else None,
                      "pause_reason": run.get("state", {}).get("pause_reason"),
                      "checkpoint_sha256": run.get("checkpoint_sha256")}, indent=2))
