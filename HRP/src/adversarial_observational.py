"""Optional aggregate-only screen of the pinned saved XGBoost attribution sample."""

from collections import Counter

from experiment2_inputs import load_explanation_cases
from llm_verification import reference_rows


# Operational screen cutoffs, chosen before reading the saved cases.
# These are study assumptions for flagging examples, not clinical cutoffs.
SCREEN = {"age": (70, "positive"), "prior_admissions": (2, "positive"),
          "creatinine": (1.5, "positive")}
MIN_MAGNITUDE = .12
SALIENT_MAX = .16
DOMINANCE_GAP = .24


def screen_saved_cases(model_root=None):
    """Read the existing 72-case bundle and release only aggregate counts."""
    provenance, cases = load_explanation_cases(model_root)
    if provenance["model_family"] != "xgboost":
        raise ValueError("The observational screen requires the pinned XGBoost model.")
    counts = Counter(dict.fromkeys(("any_flag", "unexpected_direction", "small_salient_contribution",
                                    "mixed_leading_contributions", "utilization_or_missingness_over_lab"), 0))
    exposed, unexpected = Counter(dict.fromkeys(SCREEN, 0)), Counter(dict.fromkeys(SCREEN, 0))
    for case in cases:
        values = {row["name"]: row["value"] for row in case.features}
        ranked = reference_rows(case)
        refs = {ref["feature"]: ref for ref in ranked}
        dominant = ranked[0]
        flags = set()
        if any(r["shap"] >= MIN_MAGNITUDE for r in ranked[:3]) and any(
                r["shap"] <= -MIN_MAGNITUDE for r in ranked[:3]):
            flags.add("mixed_leading_contributions")
        for feature, (cutoff, sign) in SCREEN.items():
            if values.get(feature) is None or values[feature] < cutoff:
                continue
            exposed[feature] += 1
            ref = refs[feature]
            if abs(ref["shap"]) >= MIN_MAGNITUDE and ref["direction"] != sign:
                unexpected[feature] += 1
                flags.add("unexpected_direction")
            if ref["rank"] > 3 and abs(ref["shap"]) <= SALIENT_MAX and (
                    abs(dominant["shap"]) - abs(ref["shap"]) >= DOMINANCE_GAP):
                flags.add("small_salient_contribution")
                name = dominant["feature"]
                if feature == "creatinine" and (name == "prior_admissions" or name.endswith("_missing")):
                    flags.add("utilization_or_missingness_over_lab")
        counts.update(flags)
        counts["any_flag"] += bool(flags)
    return {"model_id": provenance["model_id"], "screened": len(cases), "flags": dict(counts),
            "above_screen_cutoff": dict(exposed), "unexpected_direction_by_feature": dict(unexpected)}
