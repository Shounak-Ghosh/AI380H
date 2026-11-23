"""Fictional references and authored labels. No verifier imports."""

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from experiment2_inputs import ExplanationCase

ROOT = Path(__file__).resolve().parents[1] / "experiment4"
FIXTURE = ROOT / "v1"
VERSION = "experiment4-challenge-v1"
SEED = 404
FEATURES = (
    ("age", "Age", "years", 60),
    ("prior_admissions", "Prior hospital admissions", "admissions", 1),
    ("creatinine", "Creatinine", "mg/dL", 1),
    ("hemoglobin", "Hemoglobin", "g/dL", 13),
    ("sodium", "Sodium", "mmol/L", 138),
    ("heart_rate", "Heart rate", "bpm", 80),
)
ALIASES = {"age": "patient age", "prior_admissions": "previous admissions",
           "creatinine": "serum creatinine"}
STYLES = ("direct", "alias", "compound", "relative", "shared", "passive", "qualified", "pronoun", "reference")
LANGUAGE = dict(zip(STYLES, ("direct", "direct", "coordinated_or_compound", "relative_clause",
                            "shared_predicate", "passive", "hedged_or_qualified",
                            "implicit_reference_or_pronoun", "other"), strict=True))
MUTATIONS = (
    ("json_direction", "direct"), ("json_rank", "direct"), ("json_membership", "direct"),
    ("compound_direction", "compound"), ("relative_direction", "relative"),
    ("passive_direction", "passive"), ("probability", "direct"), ("rank", "direct"),
    ("importance", "direct"), ("scale", "direct"), ("causal", "direct"), ("feature", "direct"),
    ("measurement", "direct"), ("diagnosis", "direct"), ("both_layers", "relative"),
)
BOUNDARIES = ("short_list", "missing_prose", "abstain", "ambiguous_magnitude", "unsupported",
              "ambiguous_feature", "invalid_json", "unknown_key", "duplicate_driver", "wrong_structure",
              "invalid_direction", "overlong", "missingness", "probability_only", "ordered_subset")
ERROR = {
    "direction": "incorrect_feature_direction", "rank": "incorrect_attribution_ranking",
    "importance": "incorrect_feature_importance", "probability": "incorrect_probability",
    "feature": "unsupported_model_feature", "measurement": "invented_measurement",
    "missingness": "incorrect_missing_value", "scale": "incorrect_shap_scale",
    "causal": "unsupported_causal_claim", "diagnosis": "unsupported_clinical_diagnosis",
    "conflict": "structured_prose_contradiction", "omission": "missing_required_attribution",
    "ambiguous": "ambiguous_or_indeterminate", "malformed": "malformed_response",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def checksum(value):
    return sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def sign(value):
    return "negligible" if abs(value) <= 1e-8 else "positive" if value > 0 else "negative"


def references():
    result = {}
    for profile in range(6):
        observed = (64 + 2 * profile, 2 + profile % 3, 1.5 + profile / 8,
                    12.5, None if profile % 2 == 0 else 136, 88 + profile)
        names = ("age", "prior_admissions", "creatinine")
        top = names[profile % 3:] + names[:profile % 3]
        largest = .75 + profile / 16
        reference_shap = dict(zip(top, (largest, largest if profile in (1, 3) else .5,
                                        -.25 - profile / 32), strict=True))
        reference_shap.update(hemoglobin=.125, sodium=.0625 if observed[4] is not None else 0,
                              heart_rate=2 ** -30)
        features = [dict(name=key, label=label, unit=unit, value=value)
                    for (key, label, unit, _), value in zip(FEATURES, observed, strict=True)]
        means = {key: mean for key, _, _, mean in FEATURES}
        encoded = {r["name"]: means[r["name"]] if r["value"] is None else r["value"] for r in features}
        for variant, multiplier in enumerate((1, -1)):
            ref_id = f"exp4-reference-{profile * 2 + variant + 1:02}"
            signed = {key: value * multiplier for key, value in reference_shap.items()}
            weights = {k: signed[k] / (x - means[k]) if x != means[k] else 0 for k, x in encoded.items()}
            expected = -2 + profile / 8 + variant / 32
            intercept = expected - math.fsum(weights[k] * means[k] for k in means)
            raw = intercept + math.fsum(weights[k] * x for k, x in encoded.items())
            order = sorted(signed, key=lambda k: -abs(signed[k]))
            rows = [dict(feature=k, label=next(r["label"] for r in features if r["name"] == k),
                         shap=signed[k], rank=i, direction=sign(signed[k])) for i, k in enumerate(order, 1)]
            result[ref_id] = {
                "base_reference_id": f"exp4-profile-{profile + 1:02}", "variant": variant,
                "case": dict(case_id=ref_id, features=features, predicted_risk=1 / (1 + math.exp(-raw)),
                             shap=dict(expected_value=expected, output_scale="raw_log_odds_of_readmission",
                                       contributions=rows)),
                "model": dict(means=means, weights=weights, intercept=intercept),
            }
    return result


def explanation_case(reference):
    case = reference["case"]
    return ExplanationCase(case["case_id"], tuple(case["features"]), case["predicted_risk"], case["shap"])


def values(reference):
    return {row["feature"]: row["shap"] for row in reference["case"]["shap"]["contributions"]}


def claim(text, kind, stated, expected, feature=None, *, layer="prose", status="faithful", errors=()):
    return dict(text=text, kind=kind, feature=feature, claimed=stated, reference=expected,
                layer=layer, status=status, errors=[ERROR[e] for e in errors])


def flip(direction):
    return {"positive": "negative", "negative": "positive"}[direction]


def direction_lines(style, top, signed):
    verb = {k: ("increases" if signed[k] == "positive" else "decreases") for k in top}
    risk = "the model's predicted risk"
    direct = {k: f"{k} {verb[k]} {risk}" for k in top}
    if style == "compound":
        return [(", and ".join(direct.values()), top)]
    if style == "shared":
        groups = [tuple(k for k in top if signed[k] == direction) for direction in ("positive", "negative")]
        return [(" and ".join(keys) + f" {verb[keys[0]][:-1] if len(keys) > 1 else verb[keys[0]]} {risk}",
                 keys) for keys in groups if keys]
    if style == "pronoun":
        a, b, c = top
        return [(f"{direct[a]}, while {b} {verb[b]} it and {c} {verb[c]} that same estimate", top)]
    render = {
        "direct": lambda k: direct[k],
        "alias": lambda k: f"{ALIASES[k]} {verb[k]} {risk}",
        "relative": lambda k: f"{k}, which {verb[k]} {risk}, is among the three leading drivers",
        "passive": lambda k: f"The predicted risk is {'raised' if signed[k] == 'positive' else 'lowered'} by {k}",
        "qualified": lambda k: f"For this model and reference, {direct[k]}",
        "reference": lambda k: f"Relative to the reference, {k} shifts modeled log-odds "
                                f"{'upward' if signed[k] == 'positive' else 'downward'}",
    }
    return [(render[style](k), (k,)) for k in top]


def base(reference, style="direct", *, prose_flip=None, wrong_risk=False, omit=()):
    signed = values(reference)
    top = tuple(signed)[:3]
    expected = {k: sign(signed[k]) for k in top}
    stated = dict(expected)
    if prose_flip:
        stated[prose_flip] = flip(stated[prose_flip])
    probability = reference["case"]["predicted_risk"]
    shown = f"{probability + .073 if wrong_risk else probability:.2%}"
    heading = f"The model estimates a {shown} readmission probability"
    claims = [claim(heading, "probability", float(shown.rstrip("%")) / 100, probability,
                    status="unfaithful" if wrong_risk else "faithful", errors=("probability",) if wrong_risk else ())]
    lines = direction_lines(style, tuple(k for k in top if k not in omit), stated)
    for text, keys in lines:
        for key in keys:
            wrong = key == prose_flip
            claims.append(claim(text, "direction", stated[key], expected[key], key,
                                status="unfaithful" if wrong else "faithful",
                                errors=("direction", "conflict") if wrong else ()))
            if style == "relative":
                claims.append(claim(text, "membership", True, True, key))
    sep = ", while " if style in ("direct", "alias", "shared", "pronoun", "compound") else "; "
    prose = heading + "." + (" " + sep.join(text for text, _ in lines) + "." if lines else "")
    return prose, [dict(feature=k, direction=expected[k]) for k in top], claims


def response(prose, drivers):
    return canonical(dict(explanation=prose, drivers=drivers))


def append(prose, claims, text, kind, stated, expected, key=None, *, status="unfaithful", errors=()):
    claims.append(claim(text, kind, stated, expected, key, status=status, errors=errors))
    return prose + " " + text + "."


def corrupt(reference, recipe, style):
    signed = values(reference)
    first, _, third = tuple(signed)[:3]
    prose_flip = first if recipe.endswith("_direction") and recipe != "json_direction" or recipe == "both_layers" else None
    prose, drivers, claims = base(reference, style, prose_flip=prose_flip, wrong_risk=recipe == "probability")
    observed = {r["name"]: r["value"] for r in reference["case"]["features"]}
    additions = {
        "rank": (f"{third} is the strongest model contributor", 1, 3, third),
        "importance": ("Heart rate is the strongest model contributor", "strongest", "negligible", "heart_rate"),
        "scale": (f"The SHAP contribution for {first} is {signed[first]:+.4f} percentage points",
                  "probability_points", "raw_log_odds", first),
        "causal": ("Prior hospital admissions cause readmission", "causal", "not_established", "prior_admissions"),
        "feature": ("Insurance status increases the model's predicted risk", "model_feature", "absent", "insurance_status"),
        "measurement": ("Creatinine was 3.25 mg/dL", 3.25, observed["creatinine"], "creatinine"),
        "diagnosis": ("The creatinine contribution establishes kidney disease", "kidney_disease",
                      "not_established", "creatinine"),
    }
    if recipe in ("json_direction", "both_layers"):
        row = drivers[int(recipe == "both_layers")]
        row["direction"] = flip(row["direction"])
    elif recipe == "json_rank":
        drivers[0], drivers[2] = drivers[2], drivers[0]
    elif recipe == "json_membership":
        drivers[2] = dict(feature="hemoglobin", direction=sign(signed["hemoglobin"]))
    elif recipe in additions:
        text, stated, expected, key = additions[recipe]
        prose = append(prose, claims, text, recipe, stated, expected, key, errors=(recipe,))
    return prose, drivers, claims


def boundary(reference, recipe):
    top = tuple(values(reference))[:3]
    omit = {"missing_prose": top[-1:], "abstain": top, "probability_only": top}
    prose, drivers, claims = base(reference, omit=omit.get(recipe, ()))
    valid, raw = True, None
    if recipe == "short_list":
        drivers = drivers[:1]
    elif recipe == "ordered_subset":
        drivers = drivers[::2]
    elif recipe in ("abstain", "probability_only"):
        drivers = []
        if recipe == "abstain":
            prose = append(prose, claims, "I decline to identify the model's drivers", "abstention",
                           None, None, status="faithful")
    elif recipe.startswith("ambiguous"):
        text = "The age contribution was considerable" if recipe.endswith("magnitude") else "That one outweighed the rest"
        prose = append(prose, claims, text, "ambiguity", None, None,
                       status="indeterminate", errors=("ambiguous",))
    elif recipe == "unsupported":
        prose = append(prose, claims, "The model uses neighborhood deprivation", "feature",
                       "model_feature", "absent", "neighborhood_deprivation", errors=("feature",))
    elif recipe == "missingness":
        missing = next(r["value"] for r in reference["case"]["features"] if r["name"] == "sodium") is None
        prose = append(prose, claims, "Sodium was measured" if missing else "Sodium was missing",
                       "missingness", not missing, missing, "sodium", errors=("missingness",))
    elif recipe != "missing_prose":
        valid = False
        if recipe == "invalid_json":
            raw = response(prose, drivers)[:-1]
        elif recipe == "unknown_key":
            drivers[0]["feature"] = "age_legacy"
        elif recipe == "duplicate_driver":
            drivers[1] = dict(drivers[0])
        elif recipe == "wrong_structure":
            raw = canonical(dict(text=prose, drivers=drivers))
        elif recipe == "invalid_direction":
            drivers[0]["direction"] = "up"
        elif recipe == "overlong":
            prose += " Model summary." * 45
    return prose, drivers, claims, valid, raw


def structured_claims(drivers, signed):
    top = tuple(signed)[:3]
    order = [row["feature"] for row in drivers if row["feature"] in top]
    expected = sorted(order, key=top.index)
    if len(order) > 1:
        wrong = order != expected
        yield claim(canonical(drivers), "rank", order, expected, layer="structured",
                    status="unfaithful" if wrong else "faithful", errors=("rank",) if wrong else ())
    for row in drivers:
        key, stated = row["feature"], row["direction"]
        actual = sign(signed[key])
        wrong = stated != actual
        yield claim(canonical(row), "direction", stated, actual, key, layer="structured",
                    status="unfaithful" if wrong else "faithful", errors=("direction", "conflict") if wrong else ())
        yield claim(canonical(row), "importance", True, key in top, key, layer="structured",
                    status="faithful" if key in top else "unfaithful", errors=() if key in top else ("importance",))


def record(reference, group, slot, recipe, style):
    case_id = reference["case"]["case_id"]
    uid = f"{case_id.replace('reference', group)}-{slot + 1:02}"
    parent, parent_drivers, _ = base(reference, style)
    valid, override = True, None
    if group == "A":
        prose, drivers, claims = base(reference, style)
    elif group == "B":
        prose, drivers, claims = corrupt(reference, recipe, style)
    else:
        prose, drivers, claims, valid, override = boundary(reference, recipe)
    signed = values(reference)
    top = set(tuple(signed)[:3])
    missing = False
    if valid:
        claims.extend(structured_claims(drivers, signed))
        for layer in ("structured", "prose"):
            seen = {c["feature"] for c in claims if c["layer"] == layer and c["kind"] == "direction"}
            for key in sorted(top - seen):
                missing = True
                claims.append(claim(None, "omission", None, sign(signed[key]), key, layer=layer,
                                    status="omitted", errors=("omission",)))
    else:
        claims.append(claim(None, "schema", "invalid", "valid", layer="response",
                            status="invalid", errors=("malformed",)))
    statuses = {c["status"] for c in claims}
    faithfulness = ("unfaithful" if "unfaithful" in statuses else
                    "indeterminate" if "indeterminate" in statuses else "faithful")
    errors = sorted({e for c in claims for e in c["errors"]})
    rationale = {"faithful": "Assessable claims agree with the reference.",
                 "unfaithful": "At least one claim disagrees with the reference or exceeds its scope.",
                 "indeterminate": "An unresolved claim has more than one material reading."}[faithfulness]
    changed = [part for part, condition in (("prose", prose != parent), ("drivers", drivers != parent_drivers),
                                            ("encoding", override is not None)) if condition]
    if group == "A" and style != "direct":
        changed = ["prose"]
        parent, parent_drivers, _ = base(reference)
    language = {"ambiguous_feature": "implicit_reference_or_pronoun",
                "ambiguous_magnitude": "hedged_or_qualified",
                "probability_only": "other", "abstain": "other"}.get(recipe, LANGUAGE[style])
    return (
        dict(challenge_id=uid, case_id=case_id, raw=override if override is not None else response(prose, drivers)),
        dict(challenge_id=uid, case_id=case_id, base_reference_id=reference["base_reference_id"],
             group=group, direct_control_id=f"{case_id.replace('reference', 'A')}-01",
             original_explanation=prose, structured_drivers=drivers,
             expected_signs={k: sign(v) for k, v in signed.items()}, expected_order=list(signed),
             faithfulness=faithfulness, completeness=("not_applicable" if not valid else
                                                       "incomplete" if missing else "complete"),
             schema_validity="valid" if valid else "invalid", style=style, linguistic_construction=language,
             errors=errors, claims=claims, judgment=rationale + (" " + ", ".join(errors) + "." if errors else ""),
             mutation=dict(recipe=recipe, changed_parts=changed, parent_sha256=checksum(response(parent, parent_drivers))),
             dataset_version=VERSION, seed=SEED, provenance="Authored fictional template and controlled transformation."),
    )


def build_challenge():
    refs, inputs, labels = references(), [], []
    for i, reference in enumerate(refs.values()):
        choices = {
            "A": [("paraphrase", "direct")] + [
                ("paraphrase", STYLES[1 + (SEED + i * 4 + j) % 8]) for j in range(4)],
            "B": [MUTATIONS[(SEED + i * 5 + j) % 15] for j in range(5)],
            "C": [(BOUNDARIES[(SEED + i * 5 + j) % 15], "direct") for j in range(5)],
        }
        for group, recipes in choices.items():
            for slot, (recipe, style) in enumerate(recipes):
                data, gold = record(reference, group, slot, recipe, style)
                inputs.append(data)
                labels.append(gold)
    return dict(version=VERSION, seed=SEED, references=refs, inputs=inputs, labels=labels)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_challenge(bundle):
    refs, inputs, labels = (bundle[k] for k in ("references", "inputs", "labels"))
    clusters = Counter(r["base_reference_id"] for r in refs.values())
    require(len(clusters) == 6 and set(clusters.values()) == {2}, "Expected six paired profiles.")
    residuals = dict(additivity=0., logit=0., feature_formula=0.)
    for key, ref in refs.items():
        case, model = ref["case"], ref["model"]
        rows = case["shap"]["contributions"]
        signed = values(ref)
        require(key == case["case_id"] and set(signed) == {r["name"] for r in case["features"]}
                == {feature[0] for feature in FEATURES} and len(rows) == len(FEATURES), "Feature mismatch.")
        require([r["rank"] for r in rows] == list(range(1, 7))
                and sorted(map(abs, signed.values()), reverse=True) == list(map(abs, signed.values())),
                "Invalid reference order.")
        means, weights = model["means"], model["weights"]
        encoded = {r["name"]: means[r["name"]] if r["value"] is None else r["value"] for r in case["features"]}
        margin = model["intercept"] + math.fsum(weights[k] * encoded[k] for k in signed)
        expected = model["intercept"] + math.fsum(weights[k] * means[k] for k in signed)
        p = case["predicted_risk"]
        residuals["additivity"] = max(residuals["additivity"], abs(margin - expected - math.fsum(signed.values())),
                                     abs(expected - case["shap"]["expected_value"]))
        residuals["logit"] = max(residuals["logit"], abs(math.log(p / (1 - p)) - margin))
        residuals["feature_formula"] = max(residuals["feature_formula"],
                                          *(abs(v - weights[k] * (encoded[k] - means[k])) for k, v in signed.items()))
    require(max(residuals.values()) < 1e-12, "Synthetic logit or SHAP inconsistency.")
    ids = [i["challenge_id"] for i in inputs]
    require(len(ids) == len(set(ids)) == len({i["raw"] for i in inputs}) == 180
            and ids == [g["challenge_id"] for g in labels], "Non-unique or out-of-order records.")
    groups = Counter(g["group"] for g in labels)
    require(groups == {"A": 60, "B": 60, "C": 60} and
            Counter((g["case_id"], g["group"]) for g in labels) == {(k, g): 5 for k in refs for g in "ABC"},
            "Unbalanced allocation.")
    by_id = {g["challenge_id"]: g for g in labels}
    require(len({g["original_explanation"] for g in labels if g["group"] == "A"}) == 60,
            "Duplicate faithful prose.")
    for item, g in zip(inputs, labels, strict=True):
        require(set(item) == {"challenge_id", "case_id", "raw"}, "Label leakage.")
        ref = refs[item["case_id"]]
        signed = values(ref)
        require(g["case_id"] == item["case_id"] and g["base_reference_id"] == ref["base_reference_id"]
                and g["expected_signs"] == {k: sign(v) for k, v in signed.items()}
                and g["expected_order"] == list(signed), "Incorrect gold reference link.")
        require(g["claims"] and g["provenance"] and g["judgment"]
                and g["errors"] == sorted({e for c in g["claims"] for e in c["errors"]}), "Incomplete gold.")
        for c in g["claims"]:
            if c["layer"] == "prose" and c["text"]:
                require(c["text"] in g["original_explanation"], "Invalid claim text.")
            if c["kind"] == "direction":
                require(c["reference"] == sign(signed[c["feature"]])
                        and (c["claimed"] == c["reference"]) == (c["status"] == "faithful"), "Invalid gold direction.")
        if g["group"] == "A":
            direct = by_id[g["direct_control_id"]]
            directions = lambda row: {(c["feature"], c["claimed"]) for c in row["claims"] if c["kind"] == "direction"}
            require(g["faithfulness"] == "faithful" and g["completeness"] == "complete"
                    and directions(g) == directions(direct)
                    and g["structured_drivers"] == direct["structured_drivers"], "Unmatched paraphrase.")
        if g["group"] == "B":
            recipe = g["mutation"]["recipe"]
            changed = ["drivers"] if recipe.startswith("json") else (
                ["prose", "drivers"] if recipe == "both_layers" else ["prose"])
            require(g["faithfulness"] == "unfaithful" and g["schema_validity"] == "valid"
                    and g["mutation"]["changed_parts"] == changed
                    and sum(c["status"] == "unfaithful" for c in g["claims"]) == (2 if recipe == "both_layers" else 1),
                    "Mutation exceeded its intended scope.")
    require(checksum(bundle) == checksum(build_challenge()), "Deterministic design replay differs.")
    return dict(examples=len(inputs), references=len(refs), clusters=len(clusters), groups=dict(groups),
                gold_counts=dict(Counter(g["faithfulness"] for g in labels)),
                max_residuals=residuals, challenge_sha256=checksum(bundle))


def freeze(directory=FIXTURE):
    directory = Path(directory)
    bundle = build_challenge()
    validation = validate_challenge(bundle)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("gold.json", bundle["labels"]), ("gold_validation.json", validation),
                       ("challenge.json", {k: v for k, v in bundle.items() if k != "labels"})):
        path = directory / name
        if path.exists():
            require(checksum(json.loads(path.read_text())) == checksum(data), f"Frozen artifact differs: {name}")
        else:
            write_json(path, data)
    return validation


def load_challenge(directory=FIXTURE):
    directory = Path(directory)
    bundle = json.loads((directory / "challenge.json").read_text())
    bundle["labels"] = json.loads((directory / "gold.json").read_text())
    require(validate_challenge(bundle) == json.loads((directory / "gold_validation.json").read_text()),
            "Frozen challenge checksum mismatch.")
    return bundle


def blinded_worksheet(bundle, per_group=10):
    ids = []
    for group in "ABC":
        members = [g["challenge_id"] for g in bundle["labels"] if g["group"] == group]
        ids.extend(sorted(members, key=lambda uid: checksum([SEED, "sample", uid]))[:per_group])
    inputs = {i["challenge_id"]: i for i in bundle["inputs"]}
    worksheet, crosswalk = [], {}
    for i, uid in enumerate(sorted(ids, key=lambda uid: checksum([SEED, "blind", uid])), 1):
        rid = f"review-{i:02}"
        source = inputs[uid]
        worksheet.append(dict(review_id=rid, raw=source["raw"],
                              reference=explanation_case(bundle["references"][source["case_id"]]).inputs("B"),
                              reviewer_1=None, reviewer_2=None, adjudication=None))
        crosswalk[rid] = uid
    return worksheet, crosswalk


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
