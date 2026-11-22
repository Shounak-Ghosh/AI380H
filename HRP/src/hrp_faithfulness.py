import json
import re
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
import shap
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


CONDITIONS = ("Baseline", "SHAP-grounded", "SHAP-verified")
METRICS = (
    "top3_overlap", "rank_positions", "exact_top3_order", "signed_top3_matches",
    "prose_missing", "prose_unsupported", "prose_wrong_direction",
)
ABSTENTION = "The model contributions are unknown."
REVERSED = {
    "Original": {},
    "Reversed utilization and renal": {"prior_admissions": 0.44, "creatinine_mg_dl": 0.80},
    "Reversed age and pulse": {"age_years": 0.032, "heart_rate_bpm": 0.014},
}
SENTENCE = re.compile(
    r"`?(?P<feature>[a-z][a-z0-9_]+)`? (?P<direction>increases|decreases) "
    r"(?:the )?(?:model's )?predicted risk(?: relative to the (?:model )?baseline)?",
    re.IGNORECASE,
)


def build_cohort(raw_test, train_z, test_z, raw_train, original_model, synthetic_logit, seed):
    rng = np.random.default_rng(seed)
    partitions = np.array_split(rng.permutation(len(test_z)), len(REVERSED))
    background = train_z.sample(n=min(100, len(train_z)), random_state=seed)
    cases = []
    for (scenario, reversed_weights), positions in zip(REVERSED.items(), partitions, strict=True):
        model = original_model
        if reversed_weights:
            logits = synthetic_logit.loc[raw_train.index].copy()
            for feature, weight in reversed_weights.items():
                logits -= 2 * weight * raw_train[feature]
            logits += synthetic_logit.loc[raw_train.index].mean() - logits.mean()
            labels = rng.binomial(1, expit(logits))
            model = LogisticRegression(max_iter=2000, random_state=seed).fit(train_z, labels)
            for feature in reversed_weights:
                feature_index = train_z.columns.get_loc(feature)
                if not model.coef_[0, feature_index] < 0 < original_model.coef_[0, feature_index]:
                    raise ValueError(f"{scenario}: fitted sign did not reverse for {feature}")

        subset = test_z.iloc[positions]
        probabilities = model.predict_proba(subset)[:, 1]
        effects = shap.LinearExplainer(model, background)(subset)
        np.testing.assert_allclose(
            effects.base_values + effects.values.sum(axis=1), model.decision_function(subset), atol=1e-5
        )
        low, high = np.quantile(probabilities, [0.25, 0.75])
        order = np.argsort(-np.abs(effects.values), axis=1, kind="stable")
        top_values = np.take_along_axis(effects.values, order[:, :3], axis=1)
        for risk, risk_mask in (("Low", probabilities <= low), ("High", probabilities >= high)):
            candidates = {
                "Positive": np.flatnonzero(risk_mask & (top_values > 1e-9).any(axis=1)),
                "Negative": np.flatnonzero(risk_mask & (top_values < -1e-9).any(axis=1)),
            }
            used = set()
            for sign in sorted(candidates, key=lambda label: len(candidates[label])):
                eligible = [position for position in candidates[sign] if position not in used]
                if len(eligible) < 6:
                    raise ValueError(f"{scenario}, {risk}, {sign}: only {len(eligible)} eligible cases")
                for position in rng.choice(eligible, size=6, replace=False):
                    used.add(position)
                    patient_id = subset.index[position]
                    signed = {
                        subset.columns[index]: float(effects.values[position, index])
                        for index in order[position]
                    }
                    cases.append({
                        "patient_id": patient_id, "scenario": scenario, "risk": risk, "sign": sign,
                        "prediction": probabilities[position], "values": raw_test.loc[patient_id].to_dict(),
                        "shap": signed, "reversed_features": tuple(reversed_weights),
                    })
    rng.shuffle(cases)
    return cases


def local_request(method, url, timeout, **kwargs):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.port is None or parsed.query or parsed.fragment):
        raise ValueError("Use a numeric loopback Ollama URL with an explicit port.")
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, url, timeout=timeout, allow_redirects=False, **kwargs)
        if response.status_code != 200:
            raise requests.RequestException(f"Local Ollama returned HTTP {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Expected an Ollama JSON object.")
        return body


def generate(prompt, url, model, timeout=90):
    body = local_request("POST", f"{url}/api/generate", timeout, json={
        "model": model, "prompt": prompt, "stream": False, "think": False, "format": "json",
        "keep_alive": "5m",
        "options": {"temperature": 0, "seed": 42, "num_predict": 384, "num_ctx": 4096},
    })
    if body.get("done") is not True or body.get("done_reason") == "length":
        raise ValueError("Ollama returned an incomplete generation.")
    if not isinstance(body.get("response"), str):
        raise ValueError("Ollama response text is missing.")
    return body["response"]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_answer(raw, allowed):
    if not isinstance(raw, str) or len(raw) > 6000:
        raise ValueError("Invalid response size or type.")
    clean = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.I | re.S)
    try:
        answer = json.loads(fenced[1] if fenced else clean, object_pairs_hook=unique_object)
    except RecursionError as error:
        raise ValueError("Response JSON is nested too deeply.") from error
    if not isinstance(answer, dict) or set(answer) != {"explanation", "drivers"}:
        raise ValueError("Expected exactly explanation and drivers.")
    prose, drivers = answer["explanation"], answer["drivers"]
    if not isinstance(prose, str) or not prose.strip() or len(prose) > 1200:
        raise ValueError("Expected a nonempty explanation of at most 1200 characters.")
    if not isinstance(drivers, list) or len(drivers) > 3:
        raise ValueError("Expected a list of up to three drivers.")
    seen = set()
    for driver in drivers:
        if not isinstance(driver, dict) or set(driver) != {"feature", "direction"}:
            raise ValueError("Invalid driver object.")
        feature, direction = driver["feature"], driver["direction"]
        if not isinstance(feature, str) or feature not in allowed or feature in seen:
            raise ValueError("Unknown or duplicate driver.")
        if direction not in ("increases", "decreases"):
            raise ValueError("Invalid driver direction.")
        seen.add(feature)
    return answer


def score_answer(answer, signed):
    directions = {
        feature: "increases" if value > 0 else "decreases"
        for feature, value in signed.items() if abs(value) > 1e-9
    }
    top = sorted(directions, key=lambda feature: -abs(signed[feature]))[:3]
    drivers = answer["drivers"]
    ranked = [driver["feature"] for driver in drivers]
    driver_errors = [
        f"{driver['feature']} has the wrong direction or no nonzero contribution"
        for driver in drivers if directions.get(driver["feature"]) != driver["direction"]
    ]
    prose_features, prose_correct = set(), set()
    unsupported, wrong = [], []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer["explanation"].strip()):
        sentence = sentence.strip().removesuffix(".")
        if not sentence or sentence + "." == ABSTENTION:
            continue
        claim = SENTENCE.fullmatch(sentence)
        if not claim or claim["feature"].lower() not in directions:
            unsupported.append(sentence)
            continue
        feature = claim["feature"].lower()
        prose_features.add(feature)
        if directions[feature] == claim["direction"].lower():
            prose_correct.add(feature)
        else:
            wrong.append(sentence)
    missing = [feature for feature in top if feature not in prose_features]
    return {
        "top3_overlap": len(set(top) & set(ranked)),
        "rank_positions": sum(expected == actual for expected, actual in zip(top, ranked)),
        "exact_top3_order": ranked == top,
        "signed_top3_matches": sum(
            directions[driver["feature"]] == driver["direction"]
            for driver in drivers if driver["feature"] in top
        ),
        "prose_missing": len(missing),
        "prose_unsupported": len(unsupported),
        "prose_wrong_direction": len(wrong),
        "prose_correct_features": prose_correct,
        "feedback": (
            ([] if ranked == top else ["Order drivers by the top three absolute SHAP values."])
            + driver_errors
            + [f"Add the missing prose driver {feature}." for feature in missing]
            + [f"Remove or rewrite unsupported wording: {sentence!r}" for sentence in unsupported]
            + [f"Correct the direction in: {sentence!r}" for sentence in wrong]
        ),
    }


def make_prompt(case, supplied_shap):
    feature_values = json.dumps(case["values"])
    prompt = (
        "Explain a fictional model's predicted risk relative to its baseline. "
        "Use no medical inference, causality, or treatment advice. "
        "Return one JSON object with explanation (plain English) and drivers (array). "
        "Use one sentence per driver in explanation: '<feature> increases predicted risk.' "
        "or '<feature> decreases predicted risk.' Use exact feature names. "
        "If contributions are unknown you may abstain with an empty drivers array "
        f"and explanation '{ABSTENTION}' "
        "List up to three drivers in descending order of absolute effect. "
        "Each driver must have feature and direction (increases or decreases).\n"
        f"Fictional inputs: {feature_values}\n"
        f"Fitted model probability: {case['prediction']:.6f}\n"
    )
    if supplied_shap:
        prompt += "Signed SHAP values in log-odds, in input order:\n" + "\n".join(
            f"{feature}: {case['shap'][feature]:+.9f}" for feature in case["values"]
        )
    return prompt


def write_verified_explanation(case, proposed_drivers, infer=None, timeout=90):
    signed = case["shap"]
    ordered = sorted(signed, key=lambda feature: -abs(signed[feature]))
    drivers = [
        {"feature": feature, "direction": "increases" if signed[feature] > 0 else "decreases"}
        for feature in ordered if abs(signed[feature]) > 1e-9
    ][:3]
    template = " ".join(
        f"{driver['feature']} {driver['direction']} predicted risk."
        for driver in drivers
    ) or ABSTENTION
    result = {
        "drivers": drivers, "draft_corrected": proposed_drivers != drivers,
        "explanation": template, "writer_status": "template", "writer_seconds": 0.0,
    }
    if infer is None or not drivers:
        return result
    prompt = (
        "Write a fictional model explanation from the verified drivers below, in their given rank order. "
        "Use exactly one sentence per driver. Use its exact feature name and direction, then "
        "'predicted risk' or 'predicted risk relative to the model baseline'. "
        "Add no other claims. Return only a JSON object with an explanation string.\n"
        f"Verified drivers in rank order: {json.dumps(drivers)}"
    )
    start = time.monotonic()
    try:
        raw = infer(prompt, timeout)
        result["writer_raw"] = raw
        if not isinstance(raw, str) or len(raw) > 1200:
            raise ValueError("Unexpected writer response type or size.")
        answer = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(answer, dict) or set(answer) != {"explanation"} or not isinstance(answer["explanation"], str):
            raise ValueError("Expected a single explanation string.")
        prose = answer["explanation"].strip()
        sentences = re.split(r"(?<=\.)\s+", prose)
        claims = [SENTENCE.fullmatch(sentence.removesuffix(".")) for sentence in sentences]
        observed = [
            {"feature": claim["feature"].lower(), "direction": claim["direction"].lower()}
            for claim in claims if claim
        ]
        if len(claims) != len(drivers) or observed != drivers:
            raise ValueError("Writer prose did not match the verified claims and order.")
        result.update(explanation=prose, writer_status="accepted")
    except (requests.RequestException, ValueError, RecursionError) as error:
        result.update(writer_status="fallback", writer_error=str(error))
    result["writer_seconds"] = time.monotonic() - start
    return result


def run_benchmark(cases, infer, call_limit, wall_seconds, request_seconds=90):
    deadline = time.monotonic() + wall_seconds
    records = []
    calls = 0

    def attempt(prompt, case):
        nonlocal calls
        remaining = deadline - time.monotonic()
        if calls >= call_limit or remaining < 1:
            return {"status": "budget_skipped"}
        calls += 1
        start = time.monotonic()
        try:
            raw = infer(prompt, min(request_seconds, remaining))
        except (requests.RequestException, ValueError) as error:
            return {"status": "request_error", "error": str(error), "seconds": time.monotonic() - start}
        result = {"raw": raw, "seconds": time.monotonic() - start}
        try:
            answer = parse_answer(raw, case["values"])
        except ValueError as error:
            return {**result, "status": "parse_error", "error": str(error)}
        scored = score_answer(answer, case["shap"])
        target = set(list(case["shap"])[:3]) & set(case["reversed_features"])
        return {
            **result, **scored, **answer, "status": "ok",
            "reversed_top3_count": len(target),
            "reversed_prose_correct": len(target & scored["prose_correct_features"]),
        }

    consecutive_failed_cases = 0
    for case in cases:
        meta = {key: case[key] for key in ("patient_id", "scenario", "risk", "sign")}
        grounded_prompt = make_prompt(case, supplied_shap=True)
        baseline = attempt(make_prompt(case, supplied_shap=False), case)
        grounded = attempt(grounded_prompt, case)
        verification = "accepted"
        revision = {}
        verified = grounded
        if grounded["status"] in ("ok", "parse_error") and (
            grounded["status"] == "parse_error" or grounded["feedback"]
        ):
            feedback = grounded.get("feedback", [grounded.get("error")])
            revision = attempt(
                grounded_prompt + "\nRevise this draft using the checks. Return replacement JSON.\n"
                + json.dumps({"draft": grounded["raw"], "checks": feedback}),
                case,
            )
            if revision["status"] == "ok":
                verified = revision
                verification = "revised_with_flags" if revision["feedback"] else "revised_clean"
            else:
                verification = f"revision_{revision['status']}_retained_draft"
        elif grounded["status"] != "ok":
            verification = "no_draft"
        for condition, result in zip(CONDITIONS, (baseline, grounded, verified), strict=True):
            record = {**meta, **result, "condition": condition}
            if condition == CONDITIONS[2]:
                record.update(
                    verification=verification, revision_raw=revision.get("raw"),
                    revision_error=revision.get("error"),
                    seconds=grounded.get("seconds", 0) + revision.get("seconds", 0),
                )
            records.append(record)
        if baseline["status"] == grounded["status"] == "request_error":
            consecutive_failed_cases += 1
        else:
            consecutive_failed_cases = 0
        if consecutive_failed_cases >= 2:
            call_limit = calls
    return pd.DataFrame(records), calls
