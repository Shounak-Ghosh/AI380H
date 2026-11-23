"""Local Experiment 2 execution with an atomic, resumable request checkpoint."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Protocol
from urllib.parse import urlsplit

import requests

from hrp_faithfulness import local_request
from llm_verification import (
    PROMPT_VERSION, Issue, build_prompt, correction_prompt, deterministic_answer, feature_keys,
    parse_response, prompt_payload, verify_attribution,
)
from narrative_verifier import verify_narrative


CHECKPOINT_VERSION = 1
CONDITIONS = ("A", "B", "C")


@dataclass(frozen=True)
class ModelConfig:
    model: str = "qwen3:8b"
    endpoint: str = "http://127.0.0.1:11434"
    digest: str | None = None
    temperature: float = 0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 512


class Inference(Protocol):
    def __call__(self, prompt: str, timeout: float) -> str: ...


class InferenceError(Exception):
    def __init__(self, code, description, raw=None):
        super().__init__(description)
        self.code = code
        self.raw = raw


class Ollama:
    """The pilot transport disables redirects and proxy/environment inheritance."""

    def __init__(self, config=ModelConfig()):
        parsed = urlsplit(config.endpoint)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Ollama requires http://127.0.0.1 with an explicit port.")
        self.config = config

    def _request(self, method, path, timeout, **kwargs):
        try:
            return local_request(method, self.config.endpoint.rstrip("/") + path, timeout, **kwargs)
        except requests.Timeout as error:
            raise InferenceError("timeout", str(error)) from error
        except requests.ConnectionError as error:
            raise InferenceError("connection_failure", str(error)) from error
        except (requests.RequestException, ValueError, RecursionError) as error:
            raise InferenceError("invalid_ollama_response", str(error)) from error

    @classmethod
    def discover(cls, config=ModelConfig(), timeout=5):
        """Read the installed digest, including checking a caller-supplied pin."""
        client = cls(config)
        models = client._request("GET", "/api/tags", timeout).get("models")
        if not isinstance(models, list):
            raise InferenceError("invalid_ollama_response", "Ollama tags lacks a model list.")
        entry = next((row for row in models if isinstance(row, dict)
                      and config.model in (row.get("name"), row.get("model"))), None)
        if entry is None:
            raise InferenceError("invalid_ollama_response", f"Local model {config.model} is unavailable.")
        digest = entry.get("digest") or None
        if digest is not None and not isinstance(digest, str):
            raise InferenceError("invalid_ollama_response", "Ollama returned an invalid model digest.")
        if config.digest is not None and digest != config.digest:
            raise InferenceError("invalid_ollama_response", "Installed model digest differs from the pin.")
        return cls(replace(config, digest=digest))

    def __call__(self, prompt, timeout):
        config = self.config
        body = self._request("POST", "/api/generate", timeout, json={
            "model": config.model, "prompt": prompt, "format": "json",
            "stream": False, "think": False,
            "options": {
                "temperature": config.temperature, "seed": config.seed,
                "num_ctx": config.num_ctx, "num_predict": config.num_predict,
            },
        })
        raw = body.get("response")
        raw = raw if isinstance(raw, str) else None
        if body.get("done_reason") == "length" or body.get("done") is False:
            raise InferenceError("truncated_generation", "Ollama generation did not complete.", raw)
        if (body.get("done") is not True or raw is None
                or body.get("model", config.model) != config.model):
            raise InferenceError("invalid_ollama_response", "Invalid generate response envelope.", raw)
        return raw


def _call(infer, prompt, timeout):
    started = time.monotonic()
    raw, issue = None, None
    try:
        response = infer(prompt, timeout)
        if not isinstance(response, str):
            raise InferenceError("invalid_ollama_response", "Inference must return response text.")
        raw = response
    except InferenceError as error:
        raw, issue = error.raw, Issue(error.code, str(error))
    except (TimeoutError, requests.Timeout) as error:
        issue = Issue("timeout", str(error) or "Inference timed out.")
    except (ConnectionError, requests.ConnectionError) as error:
        issue = Issue("connection_failure", str(error))
    return {
        "raw": raw, "error": asdict(issue) if issue else None,
        "latency_seconds": time.monotonic() - started, "timeout_seconds": timeout,
        "prompt_sha256": _fingerprint(prompt),
    }


def _audit(case, call):
    if call["error"]:
        return {"parse": None, "structured": None, "narrative": None}, (Issue(**call["error"]),)
    parsed = parse_response(call["raw"], feature_keys(case))
    trace = {"parse": asdict(parsed), "structured": None, "narrative": None}
    if parsed.error:
        return trace, (parsed.error,)
    structured = verify_attribution(case, parsed.answer.drivers)
    narrative = verify_narrative(case, parsed.answer)
    trace.update(structured=asdict(structured), narrative=asdict(narrative))
    issues = structured.issues + tuple(
        Issue(claim.code, f"{claim.detail} Text: {claim.text}")
        for claim in narrative.claims if claim.status in ("contradicted", "ambiguous")
    )
    return trace, issues


def verify_and_revise(case, grounded_draft, infer, *, timeout=90):
    """Accept B, request one revision, or render an independently checked fallback."""
    return _verify_and_revise(
        case, {"raw": grounded_draft, "error": None},
        lambda prompt: _call(infer, prompt, timeout),
    )


def _verify_and_revise(case, draft_call, revise):
    original, issues = _audit(case, draft_call)
    result = {
        "original_raw": draft_call["raw"], "original": original, "revision_call": None, "revision": None,
        "draft_accepted": not issues, "revision_requested": False, "revision_accepted": False,
        "revision_rejected": False, "fallback_used": False, "fallback_reason": None,
    }
    if not issues:
        return dict(result, source="grounded_draft", answer=original["parse"]["answer"])
    if not draft_call["error"]:
        call = revise(correction_prompt(case, draft_call["raw"], issues))
        revision, issues = _audit(case, call)
        result.update(revision_call=call, revision=revision, revision_requested=True,
                      revision_accepted=not issues, revision_rejected=bool(issues))
        if not issues:
            return dict(result, source="model_revision", answer=revision["parse"]["answer"])
    answer = deterministic_answer(case)
    structured = verify_attribution(case, answer.drivers)
    narrative = verify_narrative(case, answer)
    if not structured.passed or not narrative.passed:
        raise RuntimeError("Deterministic fallback failed its verifier.")
    return dict(result, source="deterministic_fallback", answer=asdict(answer),
                fallback_used=True, fallback_reason=[asdict(issue) for issue in issues],
                fallback_validation={"structured": asdict(structured), "narrative": asdict(narrative)})


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_key(case_id, condition, prompt_version, model_configuration):
    return _fingerprint([case_id, condition, prompt_version, model_configuration])


@contextmanager
def _locked_directory(directory):
    directory = Path(directory).expanduser().resolve()
    if any(os.path.lexists(parent / ".git") for parent in (directory, *directory.parents)):
        raise ValueError("Store Experiment 2 checkpoints outside Git.")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Checkpoint directory must be owned by the current user with mode 0700.")
    lock = os.open(directory / ".run.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield directory
    finally:
        os.close(lock)


def _save(path, state):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class _BudgetReached(Exception):
    pass


def run_experiment(cases, infer: Inference, directory, *, model_config=None,
                   prompt_version=PROMPT_VERSION, request_timeout=90,
                   max_calls=30, runtime_seconds=900):
    """Resume one checkpoint. The call cap is cumulative and the wall budget per invocation.

    Failed attempts are recorded as terminal observations. A budget pause keeps
    pending work resumable. The endpoint cannot deduplicate a request interrupted
    before its response is committed to disk.
    """
    if (not isinstance(max_calls, int) or max_calls < 0
            or any(not math.isfinite(value) or value <= 0 for value in (request_timeout, runtime_seconds))):
        raise ValueError("Use a nonnegative integer call cap and positive finite time budgets.")
    started = time.monotonic()
    cases = tuple(cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    adapter_config = infer.config if isinstance(infer, Ollama) else None
    if adapter_config is not None and model_config is not None and adapter_config != model_config:
        raise ValueError("Runner configuration must match the Ollama adapter.")
    configuration = {
        "inference": "ollama" if adapter_config else "custom",
        "model": asdict(adapter_config or model_config or ModelConfig()),
        "request_timeout": request_timeout,
        "generation": {"format": "json", "stream": False, "think": False},
    }
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "prompt_version": prompt_version, "configuration": configuration,
        "cases": [{"id": case.case_id, "input_sha256": _fingerprint(prompt_payload(case, "B"))}
                  for case in cases],
    }
    records = {
        checkpoint_key(case.case_id, condition, prompt_version, configuration): {
            "case_id": case.case_id, "condition": condition, "status": "pending",
        }
        for case in cases for condition in CONDITIONS
    }
    with _locked_directory(directory) as output:
        checkpoint = output / "checkpoint.json"
        if os.path.lexists(checkpoint):
            info = checkpoint.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("Checkpoint must be an owner-only regular file.")
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state["manifest"] != manifest or state["records"].keys() != records.keys():
                raise ValueError("Incompatible checkpoint input, prompt, or model configuration.")
        else:
            state = {"manifest": manifest, "records": records, "calls": 0, "pause_reason": None}
            _save(checkpoint, state)

        def checkpointed_call(record, field, prompt):
            if field in record:
                return record[field]
            remaining = runtime_seconds - (time.monotonic() - started)
            if state["calls"] >= max_calls:
                raise _BudgetReached("call_limit")
            if remaining <= 0:
                raise _BudgetReached("runtime_limit")
            record[field] = _call(infer, prompt, min(request_timeout, remaining))
            state["calls"] += 1
            _save(checkpoint, state)
            return record[field]

        try:
            state["pause_reason"] = None
            for case in cases:
                entries = [
                    state["records"][checkpoint_key(case.case_id, condition, prompt_version, configuration)]
                    for condition in CONDITIONS
                ]
                for entry in entries:
                    if entry["status"] != "pending":
                        continue
                    condition = entry["condition"]
                    if condition == "C":
                        entry["verification"] = _verify_and_revise(
                            case, entries[1]["call"],
                            lambda prompt: checkpointed_call(entry, "revision_call", prompt),
                        )
                        entry["status"] = "ok"
                    else:
                        call = checkpointed_call(entry, "call", build_prompt(case, condition))
                        if call["error"]:
                            entry["status"] = "inference_error"
                        else:
                            parsed = parse_response(call["raw"], feature_keys(case))
                            entry.update(parse=asdict(parsed), status="parse_error" if parsed.error else "ok")
                    _save(checkpoint, state)
        except _BudgetReached as error:
            state["pause_reason"] = str(error)
        _save(checkpoint, state)
        # Use the same JSON-native representation for fresh and reloaded runs.
        return json.loads(checkpoint.read_text(encoding="utf-8"))
