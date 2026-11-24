import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verifier_live_annotation as live


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def put_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_xlsx(path, annotations, references):
    """Small inline-string workbook testing the same two tables as a saved review."""
    rel_type = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    document = Element(live.XML + "workbook")
    sheets = SubElement(document, live.XML + "sheets")
    rels = Element(rel_type + "Relationships")
    with ZipFile(path, "w") as wb:
        for i, (name, headings, rows) in enumerate((
            ("Annotations", live.ANNOTATION_FIELDS, annotations),
            ("SHAP reference", live.REFERENCE_FIELDS, references),
        ), 1):
            target = f"worksheets/sheet{i}.xml"
            SubElement(sheets, live.XML + "sheet", {"name": name, live.REL + "id": f"rId{i}"})
            SubElement(rels, rel_type + "Relationship", Id=f"rId{i}", Target=target,
                       Type=live.REL[1:-1] + "/worksheet")
            sheet = Element(live.XML + "worksheet")
            data = SubElement(sheet, live.XML + "sheetData")
            for j, values in enumerate([headings, *rows], 1):
                row = SubElement(data, live.XML + "row", r=str(j))
                for k, value in enumerate(values):
                    if value is None:
                        continue
                    cell = SubElement(row, live.XML + "c", r=f"{chr(65+k)}{j}")
                    if isinstance(value, (int, float)):
                        SubElement(cell, live.XML + "v").text = str(value)
                    else:
                        cell.set("t", "inlineStr")
                        string = SubElement(cell, live.XML + "is")
                        SubElement(string, live.XML + "t").text = value
            wb.writestr("xl/" + target, tostring(sheet))
        wb.writestr("xl/workbook.xml", tostring(document))
        wb.writestr("xl/_rels/workbook.xml.rels", tostring(rels))


class LiveAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self._tmp_cleanup)
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fixture = root / "fixture"
        results = self.fixture / "results"
        results.mkdir(parents=True)
        self.sources = {key: root / filename for key, filename in
                        (("checkpoint", "source.json"), ("crosswalk", "mapping.json"), ("annotations", "review.xlsx"))}
        self.summary = root / "public.json"
        self.verifier = live.frozen_verifier()
        self.plan, incoming, gold, self.annotations, self.references, self.mapping = [], [], [], [], [], []
        state_rows = {}
        features = [{"name": name, "value": value, "unit": "units"} for name, value in
                    (("f1", 5), ("f2", 6), ("f3", None))]
        contributions = [{"feature": f["name"], "shap": value, "rank": i} for i, (f, value) in
                         enumerate(zip(features, (.7, -.4, .2), strict=True), 1)]
        top = "1. f1 · positive\n2. f2 · negative\n3. f3 · positive"
        case = dict(features=features, predicted_risk=.32,
                    shap=dict(contributions=contributions, expected_value=-1.2))
        for i in range(30):
            j = i // 10
            group, stratum = ("B", "A", "C")[j], tuple(live.STRATA)[j]
            rid, uid = f"private-id-{i:02}", f"challenge-{i:02}"
            raw, original = f"PRIVATE_REVISION_{i}", f"Original {i}"
            self.plan.append(dict(challenge_id=uid, stratum=stratum))
            incoming.append(dict(challenge_id=uid, case_id="ref", raw=original))
            faithfulness = "unfaithful" if j == 0 else "faithful"
            completeness = "incomplete" if j == 2 else "complete"
            gold.append(dict(challenge_id=uid, group=group, faithfulness=faithfulness,
                             completeness=completeness, schema_validity="valid"))
            accepted, wrong = i % 10 < 5, i in (0, 15, 25)
            call = dict(raw=raw, error=None, latency_seconds=1.0, timeout_seconds=90, prompt_sha256="1" * 64)
            audit = dict(parse=dict(raw=raw), status="verified_complete" if accepted else "verification_failed",
                         narrative=dict(claims=[] if accepted else [dict(status="ambiguous")]), structured=dict(issues=[]))
            trace = dict(revision_requested=True, draft_accepted=False, revision_call=call, original_raw=original,
                         revision=audit, revision_accepted=accepted, revision_rejected=not accepted,
                         fallback_used=not accepted, source="model_revision" if accepted else "deterministic_fallback",
                         fallback_reason=None if accepted else [dict(code="unparsed"), dict(code="missing_prose_drivers")],
                         status="verified_complete")
            state_rows[uid] = dict(call=call, result=dict(challenge_id=uid, stratum=stratum,
                                                         requests=1, trace=trace, original_gold=faithfulness))
            self.mapping.append(dict(review_id=rid, challenge_id=uid, stratum=stratum, fictional_case_id="ref",
                                     base_reference_id="base", original_faithfulness=faithfulness,
                                     original_completeness=completeness, original_schema_validity="valid",
                                     raw_sha256=sha256(raw.encode()).hexdigest()))
            self.annotations.append([rid, raw, .32, top, "Unfaithful" if wrong else "Faithful", "Complete", "Valid",
                                     "No", "incorrect_attribution_ranking" if wrong else "None",
                                     "PRIVATE_REVIEW_NOTE" if wrong else None, "PRIVATE_REVIEWER"])
            for f, signed in zip(features, contributions, strict=True):
                self.references.append([rid, signed["rank"], f["name"], f["value"],
                                        "Observed" if f["value"] is not None else "Missing", f["unit"],
                                        signed["shap"], live.sign(signed["shap"]), "Yes", -1.2])
        bundle = dict(inputs=incoming, references={"ref": dict(case=case, base_reference_id="base")})
        challenge_hash = live.checksum(dict(bundle, labels=gold))
        put_json(self.fixture / "challenge.json", bundle)
        put_json(self.fixture / "gold.json", gold)
        put_json(self.fixture / "gold_validation.json", dict(challenge_sha256=challenge_hash))
        rubric = root / "rubric.md"
        rubric.write_text("Synthetic rubric")
        plan_file = results / "live_selection.json"
        put_json(plan_file, self.plan)
        put_json(results / "manifest.json", dict(frozen_verifier=self.verifier, challenge_sha256=challenge_hash,
                                                  rubric_sha256=digest(rubric),
                                                  files={"live_selection.json": digest(plan_file)}))
        self.state = dict(requests=30, rows=state_rows, manifest=dict(
            version=live.LIVE_VERSION, verifier=self.verifier, prompt=live.PROMPT_VERSION, timeout=90,
            challenge_sha256=challenge_hash, model=live.asdict(live.ModelConfig(digest=live.DIGEST)), plan=self.plan))
        self.crosswalk = dict(format="experiment4-live-review-crosswalk-v1", challenge_sha256=challenge_hash,
                              frozen_verifier=self.verifier, checkpoint="/PRIVATE_DATA/checkpoint.json",
                              frozen_input_sha256={name: digest(self.fixture / name) for name in
                                                  ("challenge.json", "gold.json", "gold_validation.json")},
                              cases=self.mapping)
        self.write_sources()

    def _tmp_cleanup(self):
        self.tmp.cleanup()

    def write_sources(self):
        put_json(self.sources["checkpoint"], self.state)
        self.crosswalk["checkpoint_sha256"] = digest(self.sources["checkpoint"])
        put_json(self.sources["crosswalk"], self.crosswalk)
        write_xlsx(self.sources["annotations"], self.annotations, self.references)

    def export(self):
        return live.export_live_annotation_summary(**self.sources, summary=self.summary, fixture=self.fixture)

    def validate(self):
        return live.validate_live_annotation_summary(**self.sources, summary=self.summary, fixture=self.fixture)

    def invalid_sources(self, detail):
        self.write_sources()
        report = self.validate()
        self.assertEqual(report["status"], "invalid", report)
        self.assertIn(detail, report["detail"])

    def test_valid_30_record_study_and_false_rejection(self):
        summary = self.export()
        self.assertEqual(self.validate(), dict(status="validated", selected_drafts=30, reviewers=1))
        self.assertEqual(summary["confusion_matrix"], {
            "accepted": {"faithful": 14, "unfaithful": 1}, "rejected": {"faithful": 13, "unfaithful": 2}})
        self.assertEqual(summary["rates"]["false_rejection_faithful_complete_valid"],
                         dict(numerator=13, denominator=27, rate=13/27))
        self.assertEqual(summary["rates"]["false_acceptance_unfaithful_valid"]["denominator"], 3)
        self.assertEqual(summary["rejection_reasons"]["missing_parsed_prose_directions"], 13)
        self.assertEqual([s["selected_drafts"] for s in summary["by_stratum"]], [10] * 3)

    def test_duplicate_challenge_id(self):
        self.state["manifest"]["plan"] = [self.plan[0], *self.plan[:-1]]
        self.invalid_sources("duplicate challenge ID")

    def test_missing_reviewer_annotation(self):
        self.annotations.pop()
        self.invalid_sources("Missing or unmatched reviewer annotation")

    def test_duplicate_review_id(self):
        self.annotations[1][0] = self.annotations[0][0]
        self.invalid_sources("duplicate review ID")

    def test_incorrect_reviewer_count(self):
        self.annotations[1][-1] = "OTHER_REVIEWER"
        self.invalid_sources("Incorrect reviewer count")

    def test_incorrect_crosswalk(self):
        self.mapping[0]["fictional_case_id"] = "wrong-ref"
        self.invalid_sources("Crosswalk does not match")

    def test_invalid_gold_label(self):
        self.annotations[0][4] = "Unknown"
        self.invalid_sources("invalid Mathematical faithfulness")

    def test_inconsistent_error_annotation(self):
        self.annotations[0][4] = "Faithful"
        self.invalid_sources("Gold labels and rubric")

    def test_missing_revision(self):
        self.state["rows"]["challenge-00"]["call"]["raw"] = None
        self.invalid_sources("Missing or interrupted completed revision")

    def test_missing_or_altered_checkpoint_metadata(self):
        model = self.state["manifest"].pop("model")
        self.invalid_sources("model")
        self.state["manifest"]["model"] = dict(model, temperature=0.6)
        self.invalid_sources("Checkpoint metadata")

    def test_wrong_published_values_and_denominators(self):
        good = self.export()
        for section, edit in (
            ("overall", lambda s: s.update(accepted=20)),
            ("rates", lambda s: s["false_rejection_faithful_complete_valid"].update(denominator=30)),
            ("rates", lambda s: s["false_rejection_faithful_complete_valid"].update(rate=.5)),
            ("by_stratum", lambda s: s[0].update(selected_drafts=12)),
        ):
            with self.subTest(section=section):
                changed = copy.deepcopy(good)
                edit(changed[section])
                put_json(self.summary, changed)
                self.assertEqual(self.validate(), dict(status="mismatch", fields=[section]))

    def test_saved_verifier_outcome_mismatch(self):
        self.state["rows"]["challenge-00"]["result"]["trace"]["revision_accepted"] = False
        self.invalid_sources("Inconsistent saved verifier outcome")

    def test_altered_annotations_change_checksum(self):
        self.export()
        self.annotations[0][-2] = "Corrected private notes"
        self.write_sources()
        self.assertEqual(self.validate(), dict(status="mismatch", fields=["provenance"]))

    def test_checkpoint_checksum_mismatch(self):
        self.export()
        self.sources["checkpoint"].write_bytes(self.sources["checkpoint"].read_bytes() + b" ")
        self.assertIn("Crosswalk source checksum mismatch", self.validate()["detail"])

    def test_altered_reference_fails(self):
        self.references[0][6] = -.8
        self.invalid_sources("Worksheet signed SHAP")

    def test_altered_frozen_manifest_fails(self):
        path = self.fixture / "gold_validation.json"
        put_json(path, dict(challenge_sha256="2" * 64))
        self.assertIn("Frozen challenge", self.validate()["detail"])

    def test_missing_private_source(self):
        self.sources["annotations"].unlink()
        self.assertEqual(self.validate(), dict(status="private_source_unavailable", missing=["annotations"]))

    def test_deterministic_read_only_no_inference_and_no_private_text(self):
        with patch("llm_runner.Ollama.discover", side_effect=AssertionError("unexpected local inference")), patch(
                "requests.sessions.Session.request", side_effect=AssertionError("unexpected request")):
            self.export()
            checksum = digest(self.summary)
            self.export()
            self.assertEqual(digest(self.summary), checksum)
            paths = [self.summary, *self.sources.values()]
            before = [(digest(path), path.stat().st_mtime_ns) for path in paths]
            self.assertEqual(self.validate()["status"], "validated")
            self.assertEqual(before, [(digest(path), path.stat().st_mtime_ns) for path in paths])
        public = self.summary.read_text()
        for marker in ("PRIVATE_", "private-id-", "challenge-00", "checkpoint.json", "ref\""):
            self.assertNotIn(marker, public)
