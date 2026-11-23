"""Scripted corrections, private live revision checkpoints, and audit worksheets."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import median
import time

from llm_runner import ModelConfig, Ollama, _call, _locked_directory, _save, _verify_and_revise, verify_and_revise
from llm_verification import PROMPT_VERSION, fingerprint
from verifier_challenge_set import SEED, base, canonical, checksum, explanation_case, flip, response, sign, values

LIVE_VERSION = "experiment4-live-revision-v1"
DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
HISTORICAL_ROOT = Path.home() / "Library/Application Support/AI380H/MIMIC-IV"


def _fallback_gold(reference, body):
    """Compare fallback delivery with independently derived numeric propositions."""
    signed = values(reference)
    top = tuple(signed)[:3]
    desired = [dict(feature=key, direction=sign(signed[key])) for key in top]
    quoted = body["explanation"]
    match = re.fullmatch(r"The model estimates a ([0-9.]+)% readmission probability\. (.*)\.", quoted)
    if not match:
        return False
    directions = re.findall(r"(\w+) (increases|decreases) the model's predicted risk relative to the model reference",
                            match[2])
    reconstructed = ", while ".join(f"{key} {verb} the model's predicted risk relative to the model reference"
                                     for key, verb in directions)
    return (abs(float(match[1]) / 100 - reference["case"]["predicted_risk"]) <= .00005
            and list(body["drivers"]) == desired and reconstructed == match[2]
            and directions == [(d["feature"], "increases" if d["direction"] == "positive" else "decreases")
                               for d in desired])


def scripted_corrections(bundle):
    refs = list(bundle["references"].values())
    specs = (
        ("original_accepted", "good", None, None, None),
        ("fully_corrected", "bad", "good", "faithful", True),
        ("partially_corrected", "both", "bad", "unfaithful", True),
        ("unchanged", "bad", "bad", "unfaithful", True),
        ("new_error", "bad", "wrong_risk", "unfaithful", True),
        ("invalid_json", "bad", "invalid", "indeterminate", False),
        ("timeout", "bad", "timeout", None, None),
        ("faithful_ambiguous_revision", "bad", "relative", "faithful", True),
        ("correct_drivers_wrong_prose", "bad", "wrong_prose", "unfaithful", True),
        ("faithful_false_rejection_repaired", "relative", "good", "faithful", True),
        ("accepted_omits_probability", "bad", "no_probability", "faithful", False),
    )
    rows = []
    for reference, (name, draft, revision, revision_gold, revision_complete) in zip(refs[:len(specs)], specs, strict=True):
        case = explanation_case(reference)
        first = tuple(values(reference))[0]
        good, drivers, _ = base(reference)
        wrong_drivers = [dict(row) for row in drivers]
        wrong_drivers[0]["direction"] = flip(wrong_drivers[0]["direction"])
        wrong_prose = base(reference, prose_flip=first)[0]
        responses = {
            "good": response(good, drivers),
            "bad": response(good, wrong_drivers),
            "both": response(base(reference, wrong_risk=True)[0], wrong_drivers),
            "wrong_risk": response(base(reference, wrong_risk=True)[0], drivers),
            "wrong_prose": response(wrong_prose, drivers),
            "relative": response(base(reference, "relative")[0], drivers),
            "invalid": "{",
            "no_probability": response(good.split(". ", 1)[1], drivers),
        }
        calls = []

        def infer(prompt, timeout):
            calls.append(checksum(prompt))
            if revision == "timeout":
                raise TimeoutError("Scripted correction timeout.")
            return responses[revision]

        trace = verify_and_revise(case, responses[draft], infer)
        original_gold = "faithful" if draft in ("good", "relative") else "unfaithful"
        if trace["fallback_used"]:
            final_gold = final_complete = _fallback_gold(reference, trace["answer"])
            if not final_gold:
                raise ValueError("Independent fallback arithmetic or text check failed.")
        elif trace["draft_accepted"]:
            final_gold, final_complete = original_gold == "faithful", True
        else:
            final_gold, final_complete = revision_gold == "faithful", revision_complete
        rows.append(dict(
            scenario=name, case_id=case.case_id, original_gold=original_gold,
            original_correctly_rejected=not trace["draft_accepted"] and original_gold == "unfaithful",
            original_falsely_rejected=not trace["draft_accepted"] and original_gold == "faithful",
            revision_gold=revision_gold, revision_complete=revision_complete,
            requests=len(calls), new_error_introduced=name == "new_error",
            final_mathematically_faithful=final_gold, final_complete=final_complete, trace=trace,
        ))
    return {"rows": rows, "summary": correction_summary(rows)}


def correction_summary(rows):
    totals = Counter()
    seconds = []
    for row in rows:
        trace = row["trace"]
        original = row["original_gold"]
        gold = row.get("revision_gold")
        complete = row.get("revision_complete")
        totals["requests"] += row["requests"]
        if trace["revision_call"]:
            seconds.append(trace["revision_call"]["latency_seconds"])
        totals["valid_revisions"] += bool(trace["revision"] and trace["revision"]["parse"]
                                          and trace["revision"]["parse"]["answer"])
        totals["original_accepted"] += trace["draft_accepted"]
        totals["original_correctly_rejected"] += row["original_correctly_rejected"]
        totals["original_falsely_rejected"] += row["original_falsely_rejected"]
        totals["accepted_revisions"] += trace["revision_accepted"]
        totals["failed_revisions"] += trace["revision_rejected"]
        totals["fallback"] += trace["fallback_used"]
        totals["pending_gold"] += gold == "pending_review"
        totals["genuinely_corrected"] += original == "unfaithful" and gold == "faithful"
        totals["accepted_genuine_corrections"] += (original == "unfaithful" and gold == "faithful"
                                                   and trace["revision_accepted"])
        totals["full_genuine_corrections"] += (original == "unfaithful" and gold == "faithful"
                                               and complete is True and trace["revision_accepted"])
        totals["false_rejection_repairs"] += row["original_falsely_rejected"] and gold == "faithful" and trace["revision_accepted"]
        totals["accepted_unfaithful"] += gold == "unfaithful" and trace["revision_accepted"]
        totals["rejected_faithful_revision"] += gold == "faithful" and trace["revision_rejected"]
        totals["new_errors"] += row.get("new_error_introduced") is True
        totals["final_faithful"] += row["final_mathematically_faithful"] is True
        totals["final_complete"] += row["final_complete"] is True
    rates = {}
    if not totals["pending_gold"]:
        for name, num, den in (
            ("model_revision_success", "full_genuine_corrections", "original_correctly_rejected"),
            ("false_rejection_repair", "false_rejection_repairs", "original_falsely_rejected"),
            ("final_system_correctness", "final_faithful", None),
            ("final_system_completeness", "final_complete", None),
        ):
            n = totals[den] if den else len(rows)
            rates[name] = dict(numerator=totals[num], denominator=n, estimate=totals[num] / n if n else None)
    rates["fallback_frequency"] = dict(numerator=totals["fallback"], denominator=len(rows),
                                       estimate=totals["fallback"] / len(rows) if rows else None)
    return dict(attempted=len(rows), **totals, rates=rates, latency_seconds=sum(seconds),
                median_latency_seconds=median(seconds) if seconds else None)


def select_live(bundle, predictions):
    gates = {p["challenge_id"]: p["release_accepted"] for p in predictions}
    rules = {
        "math_error": lambda g: g["group"] == "B" and g["faithfulness"] == "unfaithful",
        "faithful_prose": lambda g: g["group"] == "A" and g["faithfulness"] == "faithful",
        "incomplete": lambda g: g["group"] == "C" and g["faithfulness"] == "faithful"
        and g["schema_validity"] == "valid" and g["completeness"] == "incomplete",
    }
    plan = []
    for bucket, qualifies in rules.items():
        ids = [g["challenge_id"] for g in bundle["labels"] if not gates[g["challenge_id"]] and qualifies(g)]
        if len(ids) < 10:
            raise ValueError(f"Insufficient eligible cases for the fixed live stratum: {bucket}.")
        plan.extend(dict(challenge_id=uid, stratum=bucket) for uid in
                    sorted(ids, key=lambda uid: checksum([SEED, uid]))[:10])
    # Interleave strata so a reduced call budget samples all three.
    return [plan[i + shift] for i in range(10) for shift in (0, 10, 20)]


def _read_owner_file(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("Expected an owner-only regular file with mode 0600.")
    return json.loads(path.read_text())


def live_revision(bundle, plan, directory, *, max_calls=30, runtime_seconds=3600, timeout=90, adapter=None):
    """Checkpoint one local revision per preregistered ID. Never replay an interrupted request."""
    from verifier_evaluation import frozen_verifier

    if not 0 <= max_calls <= len(plan) or not math.isfinite(runtime_seconds) or runtime_seconds < 0 or timeout <= 0:
        raise ValueError("Invalid live execution bounds.")
    start = time.monotonic()
    config = ModelConfig(digest=DIGEST)
    inputs = {r["challenge_id"]: r for r in bundle["inputs"]}
    labels = {r["challenge_id"]: r for r in bundle["labels"]}
    header = dict(version=LIVE_VERSION, challenge_sha256=checksum(bundle), verifier=frozen_verifier(),
                  prompt=PROMPT_VERSION, plan=plan, model=asdict(config), timeout=timeout)
    with _locked_directory(directory) as output:
        path = output / (LIVE_VERSION + ".json")
        if os.path.lexists(path):
            state = _read_owner_file(path)
            if state["manifest"] != header:
                raise ValueError("Live checkpoint design, model, or plan differs.")
        else:
            if set(p.name for p in output.iterdir()) - {".run.lock"}:
                raise ValueError("Choose a dedicated Experiment 4 live directory.")
            state = dict(manifest=header, requests=0, rows={})
            _save(path, state)
        for item in plan:
            uid = item["challenge_id"]
            entry = state["rows"].get(uid)
            if entry is not None and "result" in entry:
                continue
            remaining = runtime_seconds - (time.monotonic() - start)
            if entry is None and (state["requests"] >= max_calls or remaining <= 0):
                break
            if entry is None and adapter is None:
                adapter = Ollama.discover(config)
            case_input, label = inputs[uid], labels[uid]
            reference = bundle["references"][case_input["case_id"]]
            case = explanation_case(reference)

            def revise(prompt):
                nonlocal entry
                if entry is not None:
                    if "call" not in entry:
                        entry["call"] = dict(raw=None, error=dict(code="interrupted_request",
                                                                 detail="Request state unknown after interruption."),
                                             latency_seconds=0., timeout_seconds=timeout, prompt_sha256=fingerprint(prompt))
                        _save(path, state)
                    return entry["call"]
                entry = state["rows"][uid] = {}
                state["requests"] += 1
                _save(path, state)
                entry["call"] = _call(adapter, prompt, min(timeout, remaining))
                _save(path, state)
                return entry["call"]

            trace = _verify_and_revise(case, dict(raw=case_input["raw"], error=None), revise)
            if not trace["revision_requested"]:
                raise ValueError("A selected revision case unexpectedly passed the original gate.")
            delivered = _fallback_gold(reference, trace["answer"]) if trace["fallback_used"] else None
            entry["result"] = dict(
                **item, original_gold=label["faithfulness"], requests=1, trace=trace,
                original_correctly_rejected=label["faithfulness"] == "unfaithful",
                original_falsely_rejected=label["faithfulness"] == "faithful",
                revision_gold="pending_review" if trace["revision_call"]["raw"] is not None else None,
                final_mathematically_faithful=delivered, final_complete=delivered,
            )
            entry["result"] = json.loads(canonical(entry["result"]))
            _save(path, state)
        rows = [entry["result"] for entry in state["rows"].values() if "result" in entry]
        summary = correction_summary(rows)
        summary["reserved_requests"] = state["requests"]
        return dict(rows=rows, summary=summary, checkpoint=str(path))


def saved_prose_worksheet(directory, experiment, *, checkpoint=None, model_root=None, per_stratum=3):
    """Read pinned checkpoints and save blinded raw text with model references privately."""
    from experiment2_analysis import ORIGINAL_CHECKPOINT_SHA256, read_checkpoint

    if experiment == 2:
        from experiment2_inputs import load_explanation_cases
        path = Path(checkpoint or HISTORICAL_ROOT / "checkpoint.json")
        state, digest = read_checkpoint(path)
        if digest != ORIGINAL_CHECKPOINT_SHA256:
            raise ValueError("Expected the pinned Experiment 2 source checkpoint.")
        _, cases = load_explanation_cases(model_root)
    elif experiment == 3:
        from adversarial_scenarios import explanation_cases, validate_manifest
        path = Path(checkpoint or HISTORICAL_ROOT / "llm-runs/experiment3-adversarial/qwen3-v1/checkpoint.json")
        state, digest = read_checkpoint(path)
        if digest != "52ecf7c17a9cfd9108c264af7e8f0886a1754be23edfb261597deb1cf6052a1b":
            raise ValueError("Expected the pinned Experiment 3 source checkpoint.")
        scenarios = _read_owner_file(path.with_name("scenario_manifest.json"))
        validate_manifest(scenarios)
        if fingerprint(scenarios) != state["manifest"]["provenance"]["scenario_manifest_sha256"]:
            raise ValueError("Scenario reference mismatch.")
        cases = explanation_cases(scenarios)
    else:
        raise ValueError("Choose Experiment 2 or 3.")
    from llm_verification import case_fingerprints
    if case_fingerprints(cases) != state["manifest"]["cases"]:
        raise ValueError("Saved-prose reference fingerprints differ from the checkpoint.")
    references = {case.case_id: case for case in cases}
    strata = {}
    for record in state["records"].values():
        trace = record.get("verification")
        if not trace:
            continue
        case_id = record["case_id"]
        audit = trace["original"]
        structured = audit.get("structured")
        prose = (audit.get("parse") or {}).get("answer")
        text = prose["explanation"] if prose else ""
        claims = (audit.get("narrative") or {}).get("claims", ())
        probes = {
            "correct_structured_rejected": (structured and structured["passed"] and not trace["draft_accepted"], "draft"),
            "correct_structured_ambiguous": (structured and structured["passed"] and
                                             any(c["status"] == "ambiguous" for c in claims), "draft"),
            "wrong_structured": (structured and not structured["passed"], "draft"),
            "explicit_contradiction": (any(c["status"] == "contradicted" for c in claims), "draft"),
            "compound": (bool(re.search(r"\b(?:and|while)\b", text, re.I)), "draft"),
            "relative": (bool(re.search(r"\b(?:which|whose|that)\b", text, re.I)), "draft"),
            "accepted_revision": (trace["revision_accepted"], "revision"),
            "fallback": (trace["fallback_used"], "fallback"),
        }
        for stratum, (eligible, stage) in probes.items():
            if eligible:
                raw = (trace["original_raw"] if stage == "draft" else trace["revision_call"]["raw"]
                       if stage == "revision" else canonical_body(trace["answer"]))
                strata.setdefault(stratum, {})[(case_id, stage)] = raw
    selected = {}
    for stratum, candidates in strata.items():
        for key in sorted(candidates, key=lambda key: checksum([SEED, stratum, key]))[:per_stratum]:
            selected.setdefault(key, dict(raw=candidates[key], selection=[]))["selection"].append(stratum)
    worksheet, crosswalk = [], {}
    for i, key in enumerate(sorted(selected, key=lambda key: checksum([SEED, "blind", key])), 1):
        rid = f"e{experiment}-review-{i:03}"
        worksheet.append(dict(review_id=rid, raw=selected[key]["raw"], reference=references[key[0]].inputs("B"),
                              reviewer_1=None, reviewer_2=None, adjudication=None))
        crosswalk[rid] = dict(case_id=key[0], stage=key[1], strata=selected[key]["selection"])
    with _locked_directory(directory) as output:
        for suffix, data in (("worksheet", worksheet), ("crosswalk", crosswalk)):
            target = output / f"experiment{experiment}-{suffix}.json"
            if os.path.lexists(target):
                raise FileExistsError(target)
            _save(target, dict(checkpoint_sha256=digest, data=data))
    return dict(rows=len(worksheet), source_sha256=digest)


def canonical_body(body):
    return response(body["explanation"], body["drivers"])
