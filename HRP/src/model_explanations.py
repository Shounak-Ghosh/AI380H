"""Private SHAP validation and held-out sampling for Experiment 1."""

import numpy as np
import pandas as pd
from scipy.special import expit

from modeling import KEYS, calibrated_probability, encoding_map, model_input, predictor_names, raw_margin

STRATA = ("low", "medium", "high")
CASES_PER_STRATUM = 24
OUTPUT_SCALE = "raw_log_odds_of_observed_30_day_readmission"


class Attribution:
    def __init__(self, family, model, train, schema, settings):
        import shap

        self.model = model
        self.schema = schema
        self.settings = settings
        self.preprocessor = model.named_steps["preprocess"]
        self.mapping = encoding_map(self.preprocessor)
        self.names = predictor_names(schema)
        self.groups = np.asarray([self.names.index(record["feature"]) for record in self.mapping])
        background = train.sample(n=min(settings.background_n, len(train)),
                                  random_state=settings.seed).sort_index()
        data = self.preprocessor.transform(model_input(background, schema))
        masker = shap.maskers.Independent(data, max_samples=len(background))
        classifier = model.named_steps["classifier"]
        if family == "xgboost":
            self.explainer = shap.TreeExplainer(classifier, data=masker,
                                                feature_perturbation="interventional",
                                                model_output="raw")
        else:
            self.explainer = shap.LinearExplainer(classifier, masker)
        self.metadata = dict(explainer=type(self.explainer).__name__, background_partition="train",
                             background_n=len(background), seed=settings.seed,
                             atol=settings.shap_atol, rtol=0, output_scale=OUTPUT_SCALE,
                             grouping="Sum all one-hot contributions for each original predictor; "
                                      "retain explicit lab missingness indicators as predictors.",
                             background_admission_ids=background.hadm_id.tolist())

    def explain(self, frame):
        features = model_input(frame, self.schema)
        x = self.preprocessor.transform(features)
        if self.metadata["explainer"] == "TreeExplainer":
            # Check the actual model margin below with an explicit absolute tolerance.
            explanation = self.explainer(x, check_additivity=False)
        else:
            explanation = self.explainer(x)
        values = np.asarray(explanation.values)
        if values.shape != (len(frame), len(self.mapping)):
            raise ValueError("Unexpected SHAP dimensions or output ordering")
        baseline = np.broadcast_to(np.asarray(explanation.base_values).reshape(-1), (len(frame),))
        grouped = np.column_stack([values[:, self.groups == i].sum(axis=1)
                                   for i in range(len(self.names))])
        margin = raw_margin(self.model, features)
        reconstructed = baseline + grouped.sum(axis=1)
        if not np.allclose(values.sum(axis=1), grouped.sum(axis=1), atol=1e-10, rtol=0):
            raise ValueError("Grouping changed the sum of SHAP contributions")
        if not np.allclose(margin, reconstructed, atol=self.settings.shap_atol, rtol=0):
            raise ValueError(f"SHAP additivity failed on raw margins; max absolute error "
                             f"{np.max(np.abs(margin - reconstructed)):.6g}")
        raw_probability = self.model.predict_proba(features)[:, 1]
        if not np.allclose(raw_probability, expit(margin), atol=1e-6, rtol=0):
            raise ValueError("Raw classifier probability disagrees with the logistic link")
        return pd.DataFrame(grouped, columns=self.names, index=frame.index), pd.DataFrame(
            dict(expected_value=baseline, sum_shap=grouped.sum(axis=1),
                 reconstructed_margin=reconstructed, raw_margin=margin,
                 raw_probability=raw_probability, abs_error=np.abs(reconstructed - margin)),
            index=frame.index,
        )


def risk_strata(test, probabilities):
    """Use empirical rank tertiles; break exact probability ties by admission ID."""
    if len(probabilities) != len(test) or not np.isfinite(probabilities).all():
        raise ValueError("Expected one finite predicted risk per test admission")
    scores = test[KEYS].assign(predicted_risk=probabilities)
    ordered = scores.sort_values(["predicted_risk", "hadm_id"])
    for name, positions in zip(STRATA, np.array_split(np.arange(len(ordered)), 3), strict=True):
        ordered.loc[ordered.index[positions], "risk_stratum"] = name
    return ordered


def pick_cases(pool, diversity, seed):
    """Choose distinct patients, rotating across observed lab and SHAP subgroups."""
    ordered = pool.join(diversity).sample(frac=1, random_state=seed)
    # Process the most constrained tertiary group first. Grouping by buckets
    # spreads selections over observed patterns without imposing hard quotas.
    counts = ordered.groupby("risk_stratum").subject_id.nunique()
    used, selected = set(), []
    for stratum in sorted(STRATA, key=lambda name: (counts.get(name, 0), STRATA.index(name))):
        candidates = ordered.loc[ordered.risk_stratum.eq(stratum)]
        buckets = [iter(zip(bucket.index, bucket.subject_id))
                   for _, bucket in candidates.groupby(list(diversity.columns), sort=False)]
        added = []
        while buckets and len(added) < CASES_PER_STRATUM:
            remaining = []
            for bucket in buckets:
                for index, subject in bucket:
                    if subject not in used:
                        used.add(subject)
                        added.append(index)
                        remaining.append(bucket)
                        break
                if len(added) == CASES_PER_STRATUM:
                    break
            buckets = remaining
        if len(added) < CASES_PER_STRATUM:
            raise ValueError(f"Insufficient distinct {stratum}-risk patients in the fixed candidate "
                             f"pool ({len(added)} eligible selections). Increase the pool explicitly.")
        selected.extend(added)
    return ordered.loc[selected].sort_values(["predicted_risk", "hadm_id"])


def generate_cases(test, probabilities, attribution, calibrator, probability_source, model_id, settings):
    scores = risk_strata(test, probabilities)
    candidates = scores.groupby("risk_stratum", sort=True, group_keys=False).apply(
        lambda group: group.sample(n=min(settings.case_pool_per_stratum, len(group)),
                                   random_state=settings.seed + 2),
        include_groups=False,
    ).index
    pool = scores.loc[candidates]
    values, diagnostics = attribution.explain(test.loc[pool.index])
    schema = attribution.schema
    indicators = [f["name"] for f in schema["features"] if f["kind"] == "indicator"]
    missing = test.loc[pool.index, indicators].sum(axis=1)
    availability = np.select([missing.eq(0), missing.eq(len(indicators))], ["all", "none"], "some")
    positive, negative = values.gt(1e-10).any(axis=1), values.lt(-1e-10).any(axis=1)
    sign = np.select([positive & negative, positive, negative],
                     ["mixed", "positive_only", "negative_only"], "zero")
    strongest = values.abs().idxmax(axis=1)
    diversity = pd.DataFrame(dict(lab_availability=availability, contribution_pattern=sign,
                                  leading_feature=strongest), index=pool.index)
    picked = pick_cases(pool, diversity, settings.seed + 3)
    frame = test.loc[picked.index]
    local_values = values.loc[picked.index]
    local_diagnostics = diagnostics.loc[picked.index]
    final_p = calibrated_probability(local_diagnostics.raw_margin, calibrator)
    if not np.allclose(final_p, picked.predicted_risk, atol=1e-6, rtol=0):
        raise ValueError("Case probabilities disagree with the saved test predictions")
    cases = picked.join(frame[[schema["label"], *attribution.names]]).join(local_diagnostics)
    cases["probability_source"] = probability_source
    cases["shap_output_scale"] = OUTPUT_SCALE
    cases["model_id"] = model_id
    signed = local_values.to_numpy()
    order = np.argsort(-np.abs(signed), axis=1, kind="stable")
    direction = np.select([signed > 0, signed < 0], ["positive", "negative"], "zero")
    # Long form makes the values, direction and ranking unambiguous in Parquet.
    records = pd.DataFrame({
        "hadm_id": np.repeat(cases.hadm_id.to_numpy(), len(attribution.names)),
        "feature": np.asarray(attribution.names)[order].ravel(),
        "shap": np.take_along_axis(signed, order, axis=1).ravel(),
        "direction": np.take_along_axis(direction, order, axis=1).ravel(),
        "rank": np.tile(np.arange(1, len(attribution.names) + 1), len(cases)),
    })
    summary = dict(
        selected=picked.groupby("risk_stratum").size().to_dict(),
        unique_patients=int(picked.subject_id.nunique()),
        pool_size=pool.groupby("risk_stratum").size().to_dict(),
        lab_availability=pd.crosstab(picked.risk_stratum, picked.lab_availability).to_dict(),
        contribution_pattern=pd.crosstab(picked.risk_stratum, picked.contribution_pattern).to_dict(),
        leading_feature_counts=picked.leading_feature.value_counts().to_dict(),
        risk_bounds=scores.groupby("risk_stratum").predicted_risk.agg(["min", "max"]).to_dict(),
        max_abs_shap_error=float(local_diagnostics.abs_error.max()),
        pool_max_abs_shap_error=float(diagnostics.abs_error.max()),
        limitations=[
            "Availability, contribution direction, and leading feature guide diversity without quotas. "
            "Observed counts reflect the finite candidate pools.",
            "Clinical obviousness has no prespecified rating; leading-feature diversity is measurable.",
            "Risk groups are ranked tertiles; exact probability ties can cross a boundary.",
        ],
    )
    return cases, records, summary
