"""Offline tests using entirely fictional admission records."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import feature_schema
from model_explanations import Attribution, generate_cases, risk_strata
from modeling import (
    COHORT_FILES, PARTITIONS, Settings, calibrate, choose_cohort, cohort_counts, encoding_map,
    fit_candidates, join_partitions, load_cohort, model_input, predictor_names, probability_metrics,
    raw_margin, select_threshold, sha256,
)
from run_models import Experiment, load_fitted, regenerate, verify_run


def fictional_data():
    schema = feature_schema()
    n, rng = 750, np.random.default_rng(330)
    partition = np.repeat(PARTITIONS, 250)
    age = rng.integers(18, 88, n)
    frame = pd.DataFrame(dict(subject_id=np.arange(n), hadm_id=np.arange(n) + 10_000, age=age))
    frame.loc[1, "subject_id"] = 0
    for feature in schema["features"]:
        name, kind = feature["name"], feature["kind"]
        if kind == "categorical":
            frame[name] = rng.choice(["A_B", "C", None], n)
        elif kind == "indicator":
            frame[name] = frame[name.removesuffix("_missing")].isna().astype("int8")
        elif name != "age":
            frame[name] = rng.choice([1.0, 2.0, 3.0, np.nan], n)
    frame[schema["label"]] = rng.binomial(1, expit((age - 50) / 12))
    frame["return_admittime"] = pd.Timestamp("2150-01-01")
    frame["discharge_location"] = frame[schema["label"]].astype(str)
    return schema, frame, frame[["subject_id", "hadm_id"]].assign(partition=partition)


def save_fixture(root, schema, frame, splits):
    directory = root / "run-0001"
    directory.mkdir(mode=0o700)
    (directory / "private_audit").mkdir(mode=0o700)
    frame.to_parquet(directory / "cohort.parquet")
    splits.to_parquet(directory / "splits.parquet")
    (directory / "feature_schema.json").write_text(json.dumps(schema))
    joined = join_partitions(frame, splits, schema)
    readiness = {
        "passed": ["fictional test contract"],
        "cohort": cohort_counts(frame, schema["label"]),
        "partitions": [dict(partition=p, **cohort_counts(joined.loc[joined.partition.eq(p)],
                                                       schema["label"])) for p in PARTITIONS],
        "sha256": {name: sha256(directory / name) for name in COHORT_FILES},
    }
    (directory / "private_audit/readiness_summary.json").write_text(json.dumps(readiness))
    return directory


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema, cls.frame, cls.splits = fictional_data()
        cls.label = cls.schema["label"]
        cls.joined = join_partitions(cls.frame, cls.splits, cls.schema)
        cls.train, cls.validation, cls.test = (cls.joined.loc[cls.joined.partition.eq(p)]
                                              for p in PARTITIONS)
        cls.settings = replace(Settings(), lr_cs=(1.0,), xgb_candidates=((2, 12, 0.2),),
                               threads=1, bootstrap_reps=8, background_n=16, global_n=30)
        cls.models = {family: fit_candidates(family, cls.train, cls.validation,
                                             cls.schema, cls.settings)[0]
                      for family in ("logistic", "xgboost")}

    def test_coverage_patient_disjoint_and_validated_selector(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = save_fixture(root, self.schema, self.frame, self.splits)
            (root / "run-0002").mkdir()
            self.assertEqual(choose_cohort(root), run.resolve())
            _, partitions, source = load_cohort(run)
            self.assertEqual(source["cohort"]["admissions"], 750)
            self.assertFalse(set(partitions["train"].subject_id) & set(partitions["test"].subject_id))
            with (run / "feature_schema.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "checksums"):
                load_cohort(run)
        with self.assertRaisesRegex(ValueError, "coverage"):
            join_partitions(self.frame, self.splits.iloc[1:], self.schema)
        broken = self.splits.copy()
        broken.loc[1, "partition"] = "test"
        with self.assertRaisesRegex(ValueError, "Patient split leakage"):
            join_partitions(self.frame, broken, self.schema)
        with self.assertRaisesRegex(ValueError, "unique"):
            join_partitions(pd.concat([self.frame, self.frame.iloc[:1]]), self.splits, self.schema)
        broken = self.frame.copy()
        broken.loc[0, self.label] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            join_partitions(broken, self.splits, self.schema)

    def test_training_only_imputation_encoding_and_leakage(self):
        x = model_input(self.train, self.schema)
        self.assertEqual(x.columns.tolist(), predictor_names(self.schema))
        for column in [self.label, "subject_id", "hadm_id", "partition", "discharge_location",
                       "return_admittime"]:
            self.assertNotIn(column, x)
        tampered = deepcopy(self.schema)
        tampered["features"].append({"name": "discharge_location", "kind": "categorical"})
        with self.assertRaisesRegex(ValueError, "approved cohort contract"):
            predictor_names(tampered)
        preprocessor = self.models["logistic"].named_steps["preprocess"]
        pipeline = preprocessor.named_transformers_["numeric"]
        numeric = preprocessor.transformers_[0][2]
        np.testing.assert_allclose(pipeline.named_steps["impute"].statistics_,
                                   x[numeric].median())
        probe = model_input(self.validation.iloc[:2], self.schema)
        probe["gender"] = ["unseen-category", np.nan]
        transformed = preprocessor.transform(probe)
        mapping = encoding_map(preprocessor)
        self.assertEqual([m["transformed"] for m in mapping], preprocessor.get_feature_names_out().tolist())
        gender = [i for i, record in enumerate(mapping) if record["feature"] == "gender"]
        np.testing.assert_array_equal(transformed[0, gender], 0)
        self.assertEqual(transformed[1, gender].sum(), 1)
        self.assertIn("A_B", [mapping[i]["category"] for i in gender])

    def test_models_are_deterministic(self):
        x = model_input(self.test, self.schema)
        for family, model in self.models.items():
            repeated, candidates = fit_candidates(family, self.train, self.validation,
                                                  self.schema, self.settings)
            self.assertTrue(candidates[0]["converged"])
            np.testing.assert_array_equal(repeated.predict_proba(x), model.predict_proba(x))

    def test_metric_and_threshold_conventions(self):
        y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.35, 0.2])
        metrics = probability_metrics(y, p)
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], (1 + 2 / 3) / 2)
        threshold = select_threshold(y, p)
        self.assertEqual(threshold, 0.2)
        self.assertEqual(f1_score(y, p >= threshold), max(f1_score(y, p >= t) for t in p))
        self.assertEqual(select_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6]), 0.9)

    def test_grouped_sigmoid_decision(self):
        rng = np.random.default_rng(2)
        latent = rng.normal(size=2000)
        y = rng.binomial(1, expit(latent - 1))
        groups = np.repeat(np.arange(1000), 2)
        calibrator, report, raw, oof = calibrate(y, 3 * latent + 2, groups, self.settings)
        self.assertIsNotNone(calibrator)
        self.assertEqual(report["probability_source"], "sigmoid")
        self.assertLess(report["sigmoid_oof"]["brier"], report["raw"]["brier"])
        self.assertEqual(report["threshold"], select_threshold(y, oof))
        self.assertEqual(len(report["folds"]), 3)
        unadjusted = replace(self.settings, min_brier_gain=1.0)
        calibrator, report, raw, _ = calibrate(y, latent - 1, groups, unadjusted)
        self.assertIsNone(calibrator)
        self.assertEqual(report["threshold"], select_threshold(y, raw))

    def test_grouped_shap_and_case_sample(self):
        for family, model in self.models.items():
            with self.subTest(family=family):
                attribution = Attribution(family, model, self.train, self.schema, self.settings)
                values, diagnostics = attribution.explain(self.test.iloc[:20])
                self.assertEqual(values.columns.tolist(), predictor_names(self.schema))
                np.testing.assert_allclose(values.sum(axis=1), diagnostics.sum_shap)
                self.assertLessEqual(diagnostics.abs_error.max(), self.settings.shap_atol)
                p = expit(raw_margin(model, model_input(self.test, self.schema)))
                cases, contributions, summary = generate_cases(
                    self.test, p, attribution, None, "raw", "fixture", self.settings,
                )
                repeated, repeated_values, _ = generate_cases(
                    self.test, p, attribution, None, "raw", "fixture", self.settings,
                )
                pd.testing.assert_frame_equal(cases, repeated)
                pd.testing.assert_frame_equal(contributions, repeated_values)
                self.assertEqual(summary["selected"], {"low": 24, "medium": 24, "high": 24})
                self.assertEqual(cases.subject_id.nunique(), 72)
                expected = risk_strata(self.test, p)
                self.assertTrue(cases.risk_stratum.eq(expected.loc[cases.index].risk_stratum).all())
                self.assertEqual(len(contributions), len(cases) * len(values.columns))
                self.assertEqual(contributions.groupby("hadm_id")["rank"].min().tolist(), [1] * 72)

    def test_bundle_round_trip_and_regeneration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = save_fixture(root, self.schema, self.frame, self.splits)
            experiment = Experiment(cohort, self.settings)
            for family in self.models:
                experiment.fit(family)
            experiment.validation_decisions()
            experiment.evaluate_test()
            experiment.prepare_shap()
            experiment.explain_global()
            experiment.select_cases(None)
            run = experiment.save(root / "model-runs")
            manifest, restored = load_fitted(run)
            self.assertEqual(manifest["source"]["id"], cohort.name)
            for family in self.models:
                x = model_input(self.test, self.schema)
                np.testing.assert_array_equal(experiment.models[family].predict_proba(x),
                                              restored["models"][family].predict_proba(x))
            regenerated = regenerate(run, cohort, root / "model-runs", recompute_shap=True,
                                     regenerate_cases=True)
            new = verify_run(regenerated)
            self.assertEqual(new["model_id"], run.name)
            self.assertEqual(new["parent_run_id"], run.name)
            for name in ("explanation_cases.parquet", "case_contributions.parquet", "global_shap.parquet"):
                pd.testing.assert_frame_equal(pd.read_parquet(run / name),
                                              pd.read_parquet(regenerated / name))
            (regenerated / "fitted.joblib").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksums"):
                load_fitted(regenerated)


if __name__ == "__main__":
    unittest.main()
