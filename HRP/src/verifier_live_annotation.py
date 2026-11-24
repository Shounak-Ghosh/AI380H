"""Reproduce the Experiment 4 live supplement from saved records without inference."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import posixpath
import re
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from llm_runner import ModelConfig
from llm_verification import PROMPT_VERSION
from verifier_challenge_set import ERROR, FIXTURE, checksum, require, sign
from verifier_correction import DIGEST, LIVE_VERSION
from verifier_evaluation import frozen_verifier

VERSION = "experiment4-live-annotation-v1"
SUMMARY = FIXTURE / "results" / "live_annotation_summary.json"
STRATA = {"math_error": "Mathematical errors", "faithful_prose": "Faithful prose withheld",
          "incomplete": "Faithful but incomplete"}
COUNTS = dict.fromkeys(STRATA, 10)
LABELS = {
    "Mathematical faithfulness": {"faithful", "unfaithful", "indeterminate"},
    "Completeness": {"complete", "incomplete", "not applicable"},
    "Response validity": {"valid", "invalid", "no response"},
    "Unsupported claims": {"yes", "no", "indeterminate"},
}
ANNOTATION_FIELDS = (
    "Review ID", "Original Qwen3 revision (verbatim)", "Reference probability", "Expected top three (ranked)",
    *LABELS, "Error categories", "Reviewer notes", "Reviewer identity",
)
REFERENCE_FIELDS = (
    "Review ID", "SHAP rank", "Feature key", "Observed value", "Measurement status", "Observed units",
    "Signed SHAP (raw log-odds)", "Direction", "Expected driver", "Baseline (raw log-odds)",
)
NON_MATH_ERRORS = {ERROR[key] for key in ("omission", "ambiguous", "malformed")}
UNSUPPORTED_ERRORS = {ERROR[key] for key in ("feature", "causal", "diagnosis")}
HASH = re.compile(r"[0-9a-f]{64}")
XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _sha(data):
    return sha256(data).hexdigest()


def _unique_object(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON key in source artifact.")
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _index(rows, key, description):
    result = {}
    for row in rows:
        value = row[key]
        require(isinstance(value, str) and value.strip() and value not in result,
                f"Missing or duplicate {description}.")
        result[value] = row
    return result


def load_frozen_context(directory=FIXTURE):
    """Read and hash the saved benchmark and selection, without rebuilding cases."""
    directory = Path(directory)
    members = {name: (directory / name).read_bytes()
               for name in ("challenge.json", "gold.json", "gold_validation.json")}
    bundle = dict(_json(members["challenge.json"]), labels=_json(members["gold.json"]))
    manifest = _json((directory / "results/manifest.json").read_bytes())
    plan_bytes = (directory / "results/live_selection.json").read_bytes()
    verifier = frozen_verifier()
    require(_sha(plan_bytes) == manifest["files"]["live_selection.json"], "Frozen selection checksum mismatch.")
    require(checksum(bundle) == _json(members["gold_validation.json"])["challenge_sha256"]
            == manifest["challenge_sha256"] and verifier == manifest["frozen_verifier"],
            "Frozen challenge or verifier provenance mismatch.")
    require(_sha((directory.parent / "rubric.md").read_bytes()) == manifest["rubric_sha256"],
            "Frozen rubric checksum mismatch.")
    plan = _json(plan_bytes)
    _index(plan, "challenge_id", "challenge ID in selection plan")
    require(Counter(row["stratum"] for row in plan) == COUNTS, "Incorrect frozen stratum totals.")
    return dict(bundle=bundle, challenge_sha256=manifest["challenge_sha256"], verifier=verifier, plan=plan,
                selection_sha256=_sha(plan_bytes), files={name: _sha(data) for name, data in members.items()})


def load_live_checkpoint(path, context):
    data = Path(path).read_bytes()
    state = _json(data)
    manifest = state["manifest"]
    plan = _index(manifest["plan"], "challenge_id", "challenge ID in checkpoint plan")
    require(manifest["version"] == LIVE_VERSION and manifest["prompt"] == PROMPT_VERSION
            and manifest["model"] == asdict(ModelConfig(digest=DIGEST))
            and manifest["challenge_sha256"] == context["challenge_sha256"]
            and manifest["verifier"] == context["verifier"] and manifest["plan"] == context["plan"],
            "Checkpoint metadata does not match the frozen study.")
    timeout = manifest["timeout"]
    require(type(timeout) in (float, int) and math.isfinite(timeout) and timeout > 0,
            "Invalid checkpoint request timeout.")
    rows = state["rows"]
    require(state["requests"] == len(rows) == len(plan) == sum(COUNTS.values()) and set(rows) == set(plan),
            "Checkpoint does not cover the 30 reserved requests and selected drafts.")
    incoming = _index(context["bundle"]["inputs"], "challenge_id", "frozen challenge ID")
    for uid, row in rows.items():
        call, result = row["call"], row["result"]
        raw = call["raw"]
        require(isinstance(raw, str) and raw.strip() and call["error"] is None,
                f"Missing or interrupted completed revision for {uid}.")
        require(type(call["timeout_seconds"]) in (float, int) and 0 < call["timeout_seconds"] <= timeout
                and type(call["latency_seconds"]) in (float, int) and math.isfinite(call["latency_seconds"])
                and call["latency_seconds"] >= 0 and HASH.fullmatch(call["prompt_sha256"]),
                f"Invalid request metadata for {uid}.")
        trace = result["trace"]
        revised = trace["revision"]
        accepted = trace["revision_accepted"]
        require(result["challenge_id"] == uid and result["stratum"] == plan[uid]["stratum"]
                and result["requests"] == 1 and trace["revision_requested"] is True
                and trace["draft_accepted"] is False and trace["original_raw"] == incoming[uid]["raw"]
                and trace["revision_call"] == call and revised["parse"]["raw"] == raw,
                f"Checkpoint revision linkage mismatch for {uid}.")
        require(type(accepted) is bool and accepted == (revised["status"] == "verified_complete")
                and trace["revision_rejected"] is (not accepted) and trace["fallback_used"] is (not accepted)
                and bool(trace["fallback_reason"]) is (not accepted)
                and trace["source"] == ("model_revision" if accepted else "deterministic_fallback")
                and trace["status"] == "verified_complete",
                f"Inconsistent saved verifier outcome for {uid}.")
    return state, _sha(data)


def load_crosswalk(path, checkpoint, checkpoint_sha, context):
    data = Path(path).read_bytes()
    body = _json(data)
    require(body["format"] == "experiment4-live-review-crosswalk-v1",
            "Unrecognized review crosswalk.")
    require(body["checkpoint_sha256"] == checkpoint_sha
            and body["challenge_sha256"] == context["challenge_sha256"]
            and body["frozen_verifier"] == context["verifier"]
            and body["frozen_input_sha256"] == context["files"], "Crosswalk source checksum mismatch.")
    cases = _index(body["cases"], "review_id", "review ID in crosswalk")
    by_case = _index(body["cases"], "challenge_id", "challenge ID in crosswalk")
    require(set(by_case) == set(checkpoint["rows"]), "Crosswalk challenge IDs differ from completed requests.")
    bundle = context["bundle"]
    incoming = _index(bundle["inputs"], "challenge_id", "frozen challenge ID")
    gold = _index(bundle["labels"], "challenge_id", "frozen label ID")
    for uid, row in by_case.items():
        entry = checkpoint["rows"][uid]
        ref_id = incoming[uid]["case_id"]
        label = gold[uid]
        reference = bundle["references"][ref_id]
        stratum = entry["result"]["stratum"]
        expected_stratum = {"A": "faithful_prose", "B": "math_error", "C": "incomplete"}[label["group"]]
        require(row["raw_sha256"] == _sha(entry["call"]["raw"].encode())
                and row["fictional_case_id"] == ref_id and row["base_reference_id"] == reference["base_reference_id"]
                and row["stratum"] == stratum == expected_stratum
                and row["original_faithfulness"] == label["faithfulness"] == entry["result"]["original_gold"]
                and row["original_completeness"] == label["completeness"]
                and row["original_schema_validity"] == label["schema_validity"],
                f"Crosswalk does not match the frozen input, label or revision for {uid}.")
    return cases, _sha(data)


def _xlsx_tables(data):
    """Read the two visible review tables from the saved Office workbook."""
    def strings(element):
        return "".join(node.text or "" for node in element.iter(XML + "t"))

    def table(archive, target, shared, fields):
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.iter(XML + "row"):
            cells = {}
            for cell in row.findall(XML + "c"):
                key = re.sub(r"[0-9]+$", "", cell.attrib["r"])
                require(key not in cells and cell.find(XML + "f") is None, "Duplicate cell or formula in review data.")
                kind, element = cell.get("t"), cell.find(XML + "v")
                value = None if element is None else element.text
                if kind == "inlineStr":
                    value = strings(cell)
                elif value is not None and kind == "s":
                    value = shared[int(value)]
                elif value is not None and kind in (None, "n"):
                    value = float(value)
                elif value is not None:
                    require(kind == "str", "Unsupported cell type in review data.")
                cells[key] = value
            if any(value not in (None, "") for value in cells.values()):
                require(all(len(key) == 1 and "A" <= key <= chr(64 + len(fields)) or value in (None, "")
                            for key, value in cells.items()), "Unexpected populated column in review worksheet.")
                rows.append([cells.get(chr(65 + index)) for index in range(len(fields))])
        require(rows and len(set(rows[0])) == len(fields) and set(rows[0]) == set(fields),
                "Unexpected review workbook column headers.")
        return [dict(zip(rows[0], row, strict=True)) for row in rows[1:]]

    with ZipFile(BytesIO(data)) as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels if r.get("Type", "").endswith("/worksheet")}
        sheets = {sheet.attrib["name"]: targets[sheet.attrib[REL + "id"]]
                  for sheet in book.iter(XML + "sheet")}
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [strings(cell) for cell in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        return tuple(table(archive, posixpath.normpath(posixpath.join("xl", sheets[name])).lstrip("/"),
                           shared, fields) for name, fields in (("Annotations", ANNOTATION_FIELDS),
                                                               ("SHAP reference", REFERENCE_FIELDS)))


def _validate_reference(rid, row, annotated, case):
    def equal(actual, expected):
        if type(expected) in (int, float):
            return type(actual) in (int, float) and math.isclose(actual, expected, abs_tol=1e-15, rel_tol=1e-14)
        return (actual or None) == expected

    signed = sorted(case["shap"]["contributions"], key=lambda feature: feature["rank"])
    top = "\n".join(f'{f["rank"]}. {f["feature"]} · {sign(f["shap"])}'
                    for f in signed[:3] if sign(f["shap"]) != "negligible")
    require(row["Expected top three (ranked)"] == top
            and equal(row["Reference probability"], case["predicted_risk"]),
            f"Worksheet probability or top-three reference mismatch for {rid}.")
    features = {feature["name"]: feature for feature in case["features"]}
    require(set(annotated) == set(features), f"Missing or extra SHAP reference features for {rid}.")
    for f in signed:
        key, rank, shap = f["feature"], f["rank"], f["shap"]
        value, units = features[key]["value"], features[key]["unit"]
        direction = sign(shap)
        expected = [rid, rank, key, value, "Missing" if value is None else "Observed", units, shap, direction,
                    "Yes" if rank <= 3 and direction != "negligible" else "No", case["shap"]["expected_value"]]
        require(all(equal(annotated[key][field], match) for field, match in zip(REFERENCE_FIELDS, expected, strict=True)),
                f"Worksheet signed SHAP or measurement reference mismatch for {rid}.")


def load_reviewer_annotations(path, crosswalk, checkpoint, context):
    data = Path(path).read_bytes()
    annotations, references = _xlsx_tables(data)
    indexed = _index(annotations, "Review ID", "review ID in annotations")
    by_reference = {}
    for row in references:
        rid, feature = row["Review ID"], row["Feature key"]
        feature_rows = by_reference.setdefault(rid, {})
        require(feature not in feature_rows, f"Duplicate SHAP reference row for {rid}.")
        feature_rows[feature] = row
    require(set(indexed) == set(by_reference) == set(crosswalk),
            "Missing or unmatched reviewer annotation or SHAP reference.")
    records = []
    reviewers = set()
    for rid, row in indexed.items():
        link = crosswalk[rid]
        raw = row["Original Qwen3 revision (verbatim)"]
        require(isinstance(raw, str) and _sha(raw.encode()) == link["raw_sha256"],
                f"Revision text mismatch for {rid}.")
        _validate_reference(rid, row, by_reference[rid],
                            context["bundle"]["references"][link["fictional_case_id"]]["case"])
        labels = {name: str(row[name] or "").strip().casefold() for name in LABELS}
        for name, allowed in LABELS.items():
            require(labels[name] in allowed, f"Missing or invalid {name} label for {rid}.")
        faithfulness, completeness, validity, unsupported = (labels[name] for name in LABELS)
        codes = [code.strip() for code in str(row["Error categories"] or "").casefold().split(",")]
        errors = set() if codes == ["none"] else set(codes)
        reviewer = row["Reviewer identity"]
        require(isinstance(reviewer, str) and reviewer.strip(), f"Missing reviewer identity for {rid}.")
        require(errors <= set(ERROR.values()) and (not errors or len(errors) == len(codes))
                and (not errors or str(row["Reviewer notes"] or "").strip()), f"Invalid error annotation for {rid}.")
        require(validity != "no response" and (completeness == "not applicable") == (validity != "valid")
                and (ERROR["malformed"] in errors) == (validity == "invalid")
                and (completeness != "incomplete" or ERROR["omission"] in errors)
                and (ERROR["omission"] not in errors or completeness != "complete")
                and (faithfulness == "unfaithful") == bool(errors - NON_MATH_ERRORS)
                and (faithfulness != "indeterminate" or ERROR["ambiguous"] in errors)
                and (unsupported != "yes" or faithfulness == "unfaithful")
                and (not errors & UNSUPPORTED_ERRORS or unsupported == "yes"),
                f"Gold labels and rubric error categories disagree for {rid}.")
        reviewers.add(reviewer.strip())
        trace = checkpoint["rows"][link["challenge_id"]]["result"]["trace"]
        revised = trace["revision"]
        claims = (revised["narrative"] or {}).get("claims", ())
        issues = sorted({issue["code"] for issue in trace["fallback_reason"] or ()})
        records.append(dict(stratum=link["stratum"], faithfulness=faithfulness, completeness=completeness,
                            validity=validity, errors=sorted(errors), accepted=trace["revision_accepted"],
                            fallback=trace["fallback_used"], issues=issues,
                            ambiguous=any(c["status"] == "ambiguous" for c in claims),
                            explicit=bool((revised["structured"] or {}).get("issues")) or
                            any(c["status"] == "contradicted" for c in claims)))
    require(len(reviewers) == 1, "Incorrect reviewer count for this single-reviewer study.")
    return records, len(reviewers), _sha(data)


def _rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def _is_ready(row):
    return row["faithfulness"] == "faithful" and row["validity"] == "valid" and row["completeness"] == "complete"


def _counts(records):
    return dict(selected_drafts=len(records), completed_revisions=len(records),
                valid=sum(r["validity"] == "valid" for r in records),
                faithful=sum(r["faithfulness"] == "faithful" for r in records),
                unfaithful=sum(r["faithfulness"] == "unfaithful" for r in records),
                indeterminate=sum(r["faithfulness"] == "indeterminate" for r in records),
                complete=sum(r["completeness"] == "complete" for r in records),
                faithful_complete_valid=sum(_is_ready(r) for r in records),
                accepted=sum(r["accepted"] for r in records),
                rejected=sum(not r["accepted"] for r in records),
                fallback=sum(r["fallback"] for r in records))


def aggregate(records, reviewers, provenance):
    """Publish only counts, source digests and fixed study metadata."""
    overall = _counts(records)
    overall["reserved_requests"] = provenance["reserved_requests"]
    binary = [r for r in records if r["validity"] == "valid" and r["faithfulness"] != "indeterminate"]
    matrix = {decision: {gold: sum(r["accepted"] is accepted and r["faithfulness"] == gold for r in binary)
                         for gold in ("faithful", "unfaithful")}
              for decision, accepted in (("accepted", True), ("rejected", False))}
    accepted, rejected = matrix["accepted"], matrix["rejected"]
    exclusions = Counter("invalid_response" if r["validity"] != "valid" else "indeterminate"
                         for r in records if r["validity"] != "valid" or r["faithfulness"] == "indeterminate")
    faithful_rejected = [r for r in records if r["faithfulness"] == "faithful" and not r["accepted"]]
    signatures = Counter(tuple(r["issues"]) for r in faithful_rejected)
    issue_counts = Counter(code for r in faithful_rejected for code in r["issues"])
    explained = {"ambiguous_subject", "unparsed", "missing_prose_drivers"}
    by_stratum = []
    for key, label in STRATA.items():
        rows = [r for r in records if r["stratum"] == key]
        counts = _counts(rows)
        entry = dict(stratum=key, label=label, **counts)
        ready_and_accepted = sum(_is_ready(r) and r["accepted"] for r in rows)
        if key == "math_error":
            entry["mathematical_correction"] = _rate(counts["faithful"], len(rows))
            entry["accepted_full_correction"] = _rate(ready_and_accepted, len(rows))
        elif key == "faithful_prose":
            entry["accepted_faithful_rewrite"] = _rate(ready_and_accepted, len(rows))
        else:
            entry["completeness_repair"] = _rate(counts["complete"], len(rows))
            entry["accepted_faithful_completion"] = _rate(ready_and_accepted, len(rows))
        by_stratum.append(entry)
    # Whitelist the provenance fields so private paths and join keys cannot leave this layer.
    published_provenance = {key: provenance[key] for key in (
        "challenge_sha256", "live_checkpoint_sha256", "annotation_sha256", "crosswalk_sha256",
        "live_selection_sha256", "frozen_verifier_version", "frozen_verifier_code_sha256",
    )}
    return dict(
        version=VERSION,
        study=dict(selected_drafts=len(records), reviewers=reviewers, fictional_cases_only=True,
                   review_blinded_to_verifier_decision=True,
                   blinding_evidence="Reported review procedure. Source files do not independently establish blinding."),
        provenance=dict(**published_provenance, analysis_version=VERSION),
        overall=overall, by_stratum=by_stratum, confusion_matrix=matrix,
        binary_exclusions=dict(invalid_response=exclusions["invalid_response"],
                               indeterminate=exclusions["indeterminate"]),
        classification_counts=dict(correct_acceptance=accepted["faithful"], false_acceptance=accepted["unfaithful"],
                                   correct_rejection=rejected["unfaithful"], false_rejection=rejected["faithful"]),
        rates=dict(
            false_rejection_faithful_complete_valid=_rate(
                sum(_is_ready(r) and not r["accepted"] for r in records), overall["faithful_complete_valid"]),
            false_acceptance_unfaithful_valid=_rate(accepted["unfaithful"],
                                                   accepted["unfaithful"] + rejected["unfaithful"]),
            fallback=_rate(overall["fallback"], len(records)),
        ),
        rejection_reasons=dict(
            faithful_rejected=len(faithful_rejected),
            ambiguous_or_unparsed=sum(r["ambiguous"] or "unparsed" in r["issues"] for r in faithful_rejected),
            missing_parsed_prose_directions=issue_counts["missing_prose_drivers"],
            explicit_contradiction=sum(r["explicit"] for r in faithful_rejected),
            other_issue_codes={code: count for code, count in sorted(issue_counts.items()) if code not in explained},
            issue_code_counts=dict(sorted(issue_counts.items())),
            combinations=[dict(issue_codes=list(signature), count=count) for signature, count in sorted(signatures.items())],
        ),
        unfaithful_revision_error_counts=dict(sorted(Counter(
            code for r in records if r["faithfulness"] == "unfaithful" for code in r["errors"]).items())),
        final_delivery=dict(model_revision=overall["accepted"], deterministic_fallback=overall["fallback"]),
        limitations=[
            "One human reviewer. Inter-reviewer agreement was not estimated.",
            "The 30 fictional drafts were selected from previously rejected examples.",
            "Few unfaithful revisions limit what the false-acceptance result can establish.",
            "Fictional SHAP references and selected drafts limit generalization to real hospital data and natural LLM prose.",
            "Reported blinding reduces one source of bias, and human judgments can still contain errors.",
        ],
    )


def reproduce_live_annotation_summary(*, checkpoint, annotations, crosswalk, fixture=FIXTURE):
    context = load_frozen_context(fixture)
    state, checkpoint_hash = load_live_checkpoint(checkpoint, context)
    mapping, crosswalk_hash = load_crosswalk(crosswalk, state, checkpoint_hash, context)
    records, reviewers, annotation_hash = load_reviewer_annotations(annotations, mapping, state, context)
    return aggregate(records, reviewers, dict(
        reserved_requests=state["requests"], challenge_sha256=context["challenge_sha256"],
        live_checkpoint_sha256=checkpoint_hash, annotation_sha256=annotation_hash,
        crosswalk_sha256=crosswalk_hash, live_selection_sha256=context["selection_sha256"],
        frozen_verifier_version=context["verifier"]["version"],
        frozen_verifier_code_sha256=context["verifier"]["code_sha256"],
    ))


def load_public_summary(path=SUMMARY):
    """Load the published summary for display. This does not validate its private provenance."""
    summary = _json(Path(path).read_bytes())
    require(summary["version"] == VERSION, "Unrecognized live annotation summary.")
    return summary


def validate_live_annotation_summary(*, checkpoint, annotations, crosswalk, summary=SUMMARY, fixture=FIXTURE):
    paths = dict(checkpoint=checkpoint, annotations=annotations, crosswalk=crosswalk)
    missing = [name for name, path in paths.items() if path is None or not Path(path).is_file()]
    if missing:
        return dict(status="private_source_unavailable", missing=missing)
    try:
        expected = reproduce_live_annotation_summary(**paths, fixture=fixture)
        published = load_public_summary(summary)
        mismatches = sorted(k for k in set(expected) | set(published)
                            if k not in expected or k not in published or checksum(expected[k]) != checksum(published[k]))
        if mismatches:
            return dict(status="mismatch", fields=mismatches)
        return dict(status="validated", selected_drafts=expected["study"]["selected_drafts"],
                    reviewers=expected["study"]["reviewers"])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OSError, BadZipFile, ET.ParseError) as error:
        return dict(status="invalid", detail=str(error))


def export_live_annotation_summary(*, checkpoint, annotations, crosswalk, summary=SUMMARY, fixture=FIXTURE):
    """Explicitly publish aggregate data. Keep verification and private inputs read-only."""
    source = dict(checkpoint=checkpoint, annotations=annotations, crosswalk=crosswalk)
    path = Path(summary)
    require(all(path.resolve() != Path(p).resolve() and
                (not path.exists() or not path.samefile(p)) for p in source.values()),
            "Aggregate output must be separate from its private sources.")
    if path.exists():
        load_public_summary(path)
    aggregate_summary = reproduce_live_annotation_summary(**source, fixture=fixture)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(aggregate_summary, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.close()
            temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return aggregate_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="validate", choices=("validate", "export"))
    for flag in ("checkpoint", "annotations", "crosswalk"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    sources = {key: value for key, value in vars(args).items() if key != "action"}
    if args.action == "validate":
        result = validate_live_annotation_summary(**sources)
    else:
        if any(sources[key] is None for key in ("checkpoint", "annotations", "crosswalk")):
            parser.error("Export requires all three private sources.")
        try:
            data = export_live_annotation_summary(**sources)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, OSError, BadZipFile, ET.ParseError) as error:
            parser.error(str(error))
        result = dict(status="exported", selected_drafts=data["study"]["selected_drafts"])
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("validated", "exported", "private_source_unavailable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
