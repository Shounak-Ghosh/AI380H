import re

import pandas as pd

from hrp_faithfulness import ABSTENTION, CONDITIONS, SENTENCE, score_answer


ALIASES = {
    "age_years": r"(?:patient's )?age",
    "prior_admissions": r"prior (?:hospital )?admissions",
    "creatinine_mg_dl": r"creatinine(?: levels?)?",
    "hemoglobin_g_dl": r"hemoglobin(?: levels?)?",
    "sodium_mmol_l": r"sodium(?: levels?)?",
    "heart_rate_bpm": r"heart rate",
}
FEATURE = re.compile(
    r"\b(?:" + "|".join(f"(?P<{name}>{name}|{alias})" for name, alias in ALIASES.items()) + r")\b",
    re.I,
)
FEATURE_LIST = r"@(?:\s*(?:,\s*(?:and\s+)?|\band\s+)@)*"
DIRECTION = re.compile(
    rf"(?P<features>{FEATURE_LIST})(?:,\s*which)?\s+(?P<direction>increas|decreas)(?:es?|ing)\s+"
    r"(?:(?:the |model's |predicted |model )*(?:risk|likelihood)|it)\b"
)
RANK = re.compile(
    rf"(?P<leading>{FEATURE_LIST})\s+(?:having|being|has|have)\s+the strongest\b"
    rf"|most significant contributors being\s+(?:the\s+)?(?P<trailing>{FEATURE_LIST})(?=[.,;!?]|$)"
)


def prose_claims(text):
    mentions, directions, rankings = set(), set(), set()
    legacy_mentions = set()
    legacy_unscored = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        sentence = sentence.strip().removesuffix(".")
        if not sentence or sentence + "." == ABSTENTION:
            continue
        legacy = SENTENCE.fullmatch(sentence)
        if legacy:
            legacy_mentions.add(legacy["feature"].lower())
        else:
            legacy_unscored += 1
        features = [match.lastgroup for match in FEATURE.finditer(sentence)]
        mentions.update(features)
        masked = FEATURE.sub("@", sentence.lower().replace("`", ""))
        offsets = [match.start() for match in re.finditer("@", masked)]
        names = dict(zip(offsets, features, strict=True))
        for match in DIRECTION.finditer(masked):
            direction = match["direction"] + "es"
            directions.update(
                (names[offset], direction)
                for offset in offsets if match.start("features") <= offset < match.end("features")
            )
        for match in RANK.finditer(masked):
            group = "leading" if match["leading"] else "trailing"
            rankings.add(tuple(
                names[offset] for offset in offsets if match.start(group) <= offset < match.end(group)
            ))
    return mentions, directions, rankings, legacy_mentions, legacy_unscored


def rescore_prose(responses, cohort):
    by_id = {case["patient_id"]: case for case in cohort}
    if responses.duplicated(["patient_id", "condition"]).any():
        raise ValueError("Each patient and condition must have a single response.")
    results = []
    for response in responses.loc[responses.status == "ok"].to_dict("records"):
        case = by_id[response["patient_id"]]
        text = response["explanation"]
        mentions, directions, rankings, legacy, legacy_unscored = prose_claims(text)
        signed = case["shap"]
        top = tuple(sorted(signed, key=lambda feature: -abs(signed[feature]))[:3])
        expected = {
            feature: "increases" if value > 0 else "decreases"
            for feature, value in signed.items() if abs(value) > 1e-9
        }
        correct = {(feature, sign) for feature, sign in directions if expected.get(feature) == sign}
        wrong = directions - correct
        bad_ranks = {rank for rank in rankings if set(rank) != set(top[:len(rank)])}
        original = score_answer(response, signed)
        results.append({
            **response, **{key: case[key] for key in ("scenario", "risk", "sign")},
            "top": top, "reference": expected, "reversed": case["reversed_features"],
            "mentions": mentions, "directions": directions, "rankings": rankings,
            "legacy_mentions": legacy & set(expected), "legacy_unscored_sentences": legacy_unscored,
            "missing": len(set(top) - mentions), "wrong_direction": len(wrong),
            "unsupported_rank": len(bad_ranks), "direction_claims_checked": len(directions),
            "rank_claims_checked": len(rankings),
            "correct_top3_directions": len({feature for feature, _ in correct} & set(top)),
            "reversed_top3": len(set(top) & set(case["reversed_features"])),
            "reversed_correct": len({feature for feature, _ in correct} & set(top) & set(case["reversed_features"])),
            "legacy_missing": original["prose_missing"],
            "signed_top3_matches": original["signed_top3_matches"],
            "rank_positions": original["rank_positions"],
        })
    return pd.DataFrame(results)


def calibrate(scored, labels):
    audit = scored.merge(labels, on=["patient_id", "condition"], validate="one_to_one")
    if len(audit) != len(labels):
        raise ValueError("The human labels do not match the recovered responses.")
    for column in ("mentions", "directions", "rankings"):
        audit[f"exact_{column}"] = [
            observed == expected
            for observed, expected in zip(audit[column], audit[f"human_{column}"], strict=True)
        ]
    audit["legacy_mentions_exact"] = [
        observed == expected for observed, expected in zip(audit.legacy_mentions, audit.human_mentions, strict=True)
    ]
    audit["human_missing"] = [
        len(set(top) - mentions) for top, mentions in zip(audit.top, audit.human_mentions, strict=True)
    ]
    audit["human_wrong_direction"] = [
        sum(reference.get(feature) != sign for feature, sign in directions)
        for reference, directions in zip(audit.reference, audit.human_directions, strict=True)
    ]
    audit["human_unsupported_rank"] = [
        sum(set(rank) != set(top[:len(rank)]) for rank in rankings)
        for top, rankings in zip(audit.top, audit.human_rankings, strict=True)
    ]
    return audit


def compare_revisions(scored):
    measures = {
        "signed_top3_matches": 1, "rank_positions": 1, "missing": -1,
        "wrong_direction": -1, "unsupported_rank": -1, "correct_top3_directions": 1,
    }
    paired = scored.loc[scored.condition.isin(CONDITIONS[1:])]
    wide = paired.pivot(index="patient_id", columns="condition", values=[*measures, "seconds"])
    missing_conditions = set(CONDITIONS[1:]) - set(wide.columns.get_level_values("condition"))
    if missing_conditions:
        return pd.DataFrame(), pd.Series(dtype=float)
    delta = (
        wide.xs(CONDITIONS[2], axis=1, level="condition")
        - wide.xs(CONDITIONS[1], axis=1, level="condition")
    ).dropna(subset=list(measures))
    rows = []
    for measure, direction in measures.items():
        gain = delta[measure] * direction
        rows.append({
            "metric": measure, "helped": (gain > 0).sum(), "hurt": (gain < 0).sum(),
            "unchanged": (gain == 0).sum(), "mean_change": delta[measure].mean(),
        })
    return pd.DataFrame(rows).set_index("metric"), delta.seconds
