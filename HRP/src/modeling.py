"""Offline model fitting and evaluation for validated HRP cohort runs."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from features import LAB_ITEMS, feature_schema

DEFAULT_COHORT_ROOT = Path.home() / "Library/Application Support/AI380H/MIMIC-IV"
KEYS = ["subject_id", "hadm_id"]
PARTITIONS = ("train", "validation", "test")
COHORT_FILES = ("cohort.parquet", "splits.parquet", "feature_schema.json")
MISSING_CATEGORY = "__HRP_MISSING__"


@dataclass(frozen=True)
class Settings:
    seed: int = 380
    threads: int = 4
    lr_cs: tuple = (0.1, 1.0)
    # (depth, trees, learning rate)
    xgb_candidates: tuple = ((3, 240, 0.06), (5, 320, 0.05))
    calibration_folds: int = 3
    min_brier_gain: float = 0.0001
    min_primary_ap_gain: float = 0.005
    max_primary_brier_penalty: float = 0.001
    bootstrap_reps: int = 300
    background_n: int = 64
    global_n: int = 1000
    case_pool_per_stratum: int = 144
    shap_atol: float = 0.0001

    def manifest(self):
        return asdict(self)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def private_directory(path):
    path = Path(path).expanduser().resolve(strict=True)
    if (not path.is_dir() or path.stat().st_mode & 0o077
            or any((parent / ".git").exists() for parent in (path, *path.parents))):
        raise ValueError("Expected an owner-only directory outside Git")
    return path


def choose_cohort(root=DEFAULT_COHORT_ROOT, explicit=None):
    """Select the newest run with a readiness manifest; fail on a corrupt selection."""
    if explicit is not None:
        return private_directory(explicit)
    root = Path(root).expanduser()
    candidates = [
        path for path in root.glob("run-*")
        if (path / "private_audit/readiness_summary.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No validated cohort runs in {root}")
    return private_directory(max(candidates))


def predictor_names(schema):
    # The saved schema drives column order and types. The cohort contract
    # independently rejects a modified schema that adds a leakage column.
    if schema != feature_schema():
        raise ValueError("Saved predictor schema differs from the approved cohort contract")
    return [feature["name"] for feature in schema["features"]]


def feature_metadata(schema):
    predictor_names(schema)
    labels = {
        "age": ("Age", "years"),
        "gender": ("Gender", None),
        "admission_type": ("Admission type", None),
        "admission_location": ("Admission location", None),
        "insurance": ("Insurance", None),
        "prior_admissions": ("Prior admissions", "admissions"),
        "days_since_previous_discharge": ("Days since previous discharge", "days"),
    }
    for name, _, label, _, unit, _ in LAB_ITEMS:
        labels[name] = (label, unit)
        labels[f"{name}_missing"] = (f"{label} missing", None)
    return {
        f["name"]: {**f, "label": labels[f["name"]][0],
                    "unit": f.get("unit", labels[f["name"]][1])}
        for f in schema["features"]
    }


def join_partitions(cohort, splits, schema):
    names = predictor_names(schema)
    label = schema["label"]
    for frame in (cohort, splits):
        if frame[KEYS].isna().any().any() or not frame.hadm_id.is_unique:
            raise ValueError("Admission IDs must be unique and both identifiers present")
    joined = cohort[[*KEYS, label, *names]].merge(
        splits[[*KEYS, "partition"]], on=KEYS, how="outer", validate="1:1", indicator=True,
    )
    if not joined.pop("_merge").eq("both").all():
        raise ValueError("Cohort and splits lack one-to-one admission coverage")
    if not joined[label].isin([0, 1]).all():
        raise ValueError("Expected a valid binary outcome for every admission")
    if set(joined.partition) != set(PARTITIONS):
        raise ValueError("Expected exactly train, validation and test partitions")
    if not joined.groupby("subject_id").partition.nunique().eq(1).all():
        raise ValueError("Patient split leakage")
    if not joined.groupby("partition")[label].nunique().eq(2).all():
        raise ValueError("Every partition must contain both classes")
    return joined.sort_values("hadm_id", ignore_index=True)


def cohort_counts(data, label):
    return dict(patients=int(data.subject_id.nunique()), admissions=len(data),
                readmissions=int(data[label].sum()), readmission_pct=100 * data[label].mean())


def load_cohort(run):
    run = private_directory(run)
    readiness = read_json(run / "private_audit/readiness_summary.json")
    checksums = {name: sha256(run / name) for name in COHORT_FILES}
    if not readiness["passed"] or checksums != readiness["sha256"]:
        raise ValueError("Cohort checksums disagree with the readiness manifest")
    schema = read_json(run / "feature_schema.json")
    names = predictor_names(schema)
    label = schema["label"]
    cohort = pd.read_parquet(run / "cohort.parquet", columns=[*KEYS, label, *names])
    splits = pd.read_parquet(run / "splits.parquet")
    data = join_partitions(cohort, splits, schema)
    counts = cohort_counts(data, label)
    partitions = {name: data.loc[data.partition.eq(name)].copy() for name in PARTITIONS}
    summary = [dict(partition=name, **cohort_counts(frame, label))
               for name, frame in partitions.items()]
    pd.testing.assert_frame_equal(pd.DataFrame([counts]),
                                   pd.DataFrame([readiness["cohort"]]),
                                   check_dtype=False, check_like=True)
    pd.testing.assert_frame_equal(pd.DataFrame(summary),
                                   pd.DataFrame(readiness["partitions"]),
                                   check_dtype=False, check_like=True)
    return schema, partitions, {"id": run.name, "sha256": checksums, "cohort": counts,
                                "partitions": summary}


def audit_features(frame, schema):
    rows = []
    for name, metadata in feature_metadata(schema).items():
        values = frame[name]
        observed = values.dropna()
        row = dict(feature=name, dtype=str(values.dtype), missing=int(values.isna().sum()),
                   missing_pct=100 * values.isna().mean())
        if metadata["kind"] == "categorical":
            if (not observed.map(lambda value: isinstance(value, str)).all()
                    or observed.eq(MISSING_CATEGORY).any()):
                raise ValueError(f"Invalid categorical values: {name}")
            row["categories"] = observed.nunique()
        else:
            if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(observed).all():
                raise ValueError(f"Expected finite numeric values or missing: {name}")
            if metadata["kind"] == "indicator" and not values.isin([0, 1]).all():
                raise ValueError(f"Expected a complete binary missingness indicator: {name}")
            row.update(min=observed.min(), max=observed.max())
        rows.append(row)
    return pd.DataFrame(rows)


def model_input(frame, schema):
    features = frame[predictor_names(schema)].copy()
    for feature in schema["features"]:
        name = feature["name"]
        if feature["kind"] == "categorical":
            # Normalize pandas' nullable string representations to sklearn's NaN.
            features[name] = features[name].astype(object).where(features[name].notna(), np.nan)
    return features


def preprocessing(schema, scale):
    predictor_names(schema)
    columns = {kind: [f["name"] for f in schema["features"] if f["kind"] == kind]
               for kind in ("numeric", "indicator", "categorical")}
    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), columns["numeric"]),
        ("indicator", "passthrough", columns["indicator"]),
        ("categorical", categorical, columns["categorical"]),
    ])


def encoding_map(preprocessor):
    """Map columns by transformer position and encoder categories, without parsing names."""
    records = []
    names = preprocessor.get_feature_names_out()
    for block, transformer, originals in preprocessor.transformers_:
        if block == "remainder":
            continue
        if block == "categorical":
            pairs = [(name, str(category)) for name, categories in
                     zip(originals, transformer.named_steps["onehot"].categories_, strict=True)
                     for category in categories]
        else:
            pairs = [(name, None) for name in originals]
        for transformed, (original, category) in zip(
            names[preprocessor.output_indices_[block]], pairs, strict=True,
        ):
            records.append(dict(transformed=transformed, feature=original, category=category))
    if [record["transformed"] for record in records] != names.tolist():
        raise ValueError("Encoding map order differs from the fitted transformer")
    return records


def probability_metrics(y, probabilities):
    return dict(roc_auc=roc_auc_score(y, probabilities),
                average_precision=average_precision_score(y, probabilities),
                brier=brier_score_loss(y, probabilities),
                mean_probability=float(np.mean(probabilities)),
                prevalence=float(np.mean(y)))


def fit_candidates(family, train, validation, schema, settings):
    x_train, x_val = (model_input(frame, schema) for frame in (train, validation))
    y_train, y_val = (frame[schema["label"]] for frame in (train, validation))
    if family == "logistic":
        configs = [{"C": c, "solver": "lbfgs", "max_iter": 1000,
                    "random_state": settings.seed} for c in settings.lr_cs]
        estimator = LogisticRegression  # lbfgs uses L2 regularization by default
    elif family == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier
        configs = [
            dict(max_depth=depth, n_estimators=trees, learning_rate=rate, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=2, tree_method="hist",
                 device="cpu", n_jobs=settings.threads, random_state=settings.seed,
                 objective="binary:logistic", eval_metric="logloss")
            for depth, trees, rate in settings.xgb_candidates
        ]
    else:
        raise ValueError(f"Unknown model family: {family}")
    rows, best, best_key = [], None, None
    for position, config in enumerate(configs):
        model = Pipeline([("preprocess", preprocessing(schema, scale=family == "logistic")),
                          ("classifier", estimator(**config))])
        with threadpool_limits(limits=settings.threads), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            started = perf_counter()
            model.fit(x_train, y_train)
            elapsed = perf_counter() - started
            probabilities = model.predict_proba(x_val)[:, 1]
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        for item in caught:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
        row = dict(candidate=position, config=config, training_seconds=elapsed,
                   converged=converged, **probability_metrics(y_val, probabilities))
        if family == "logistic":
            row["iterations"] = int(model.named_steps["classifier"].n_iter_[0])
        rows.append(row)
        key = (row["average_precision"], -row["brier"], -position)
        if converged and (best_key is None or key > best_key):
            best, best_key = model, key
    if best is None:
        raise RuntimeError(f"No converged candidate for {family}")
    selected = -best_key[2]
    for row in rows:
        row["selected"] = row["candidate"] == selected
    return best, rows


def raw_margin(model, features):
    classifier = model.named_steps["classifier"]
    transformed = model.named_steps["preprocess"].transform(features)
    if isinstance(classifier, LogisticRegression):
        return classifier.decision_function(transformed)
    return classifier.predict(transformed, output_margin=True)


def select_threshold(y, probabilities):
    """Maximize validation F1. Among ties, prefer the higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total,
                   out=np.zeros_like(total), where=total > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def calibration_bins(y, p):
    observed, forecast = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return dict(mean_predicted=forecast.tolist(), observed_fraction=observed.tolist())


def calibrated_probability(margin, calibrator=None):
    if calibrator is None:
        return expit(margin)
    return calibrator.predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]


def calibrate(y, margins, groups, settings):
    """Assess Platt calibration with patient-disjoint out-of-fold validation predictions."""
    y, margins, groups = map(np.asarray, (y, margins, groups))
    x = margins.reshape(-1, 1)
    raw = calibrated_probability(margins)
    oof = np.empty(len(y))
    fold_results = []
    folds = GroupKFold(n_splits=settings.calibration_folds, shuffle=True,
                       random_state=settings.seed)
    for fit, assess in folds.split(x, y, groups):
        if np.unique(y[fit]).size != 2 or np.unique(y[assess]).size != 2:
            raise ValueError("Calibration folds must contain both outcome classes")
        sigmoid = LogisticRegression(C=1000, max_iter=1000).fit(x[fit], y[fit])
        oof[assess] = calibrated_probability(margins[assess], sigmoid)
        fold_results.append(dict(
            raw_ap=average_precision_score(y[assess], raw[assess]),
            sigmoid_brier=brier_score_loss(y[assess], oof[assess]),
            raw_brier=brier_score_loss(y[assess], raw[assess]),
            sigmoid_slope=float(sigmoid.coef_[0, 0]),
        ))
    raw_metrics, oof_metrics = probability_metrics(y, raw), probability_metrics(y, oof)
    final = LogisticRegression(C=1000, max_iter=1000).fit(x, y)
    use_sigmoid = (
        raw_metrics["brier"] - oof_metrics["brier"] >= settings.min_brier_gain
        and final.coef_[0, 0] > 0 and all(row["sigmoid_slope"] > 0 for row in fold_results)
    )
    chosen = oof if use_sigmoid else raw
    report = dict(probability_source="sigmoid" if use_sigmoid else "raw",
                  raw=raw_metrics, sigmoid_oof=oof_metrics, folds=fold_results,
                  raw_curve=calibration_bins(y, raw), sigmoid_oof_curve=calibration_bins(y, oof),
                  sigmoid_slope=float(final.coef_[0, 0]), sigmoid_intercept=float(final.intercept_[0]),
                  threshold=select_threshold(y, chosen))
    return final if use_sigmoid else None, report, raw, oof


def choose_primary(reports, settings):
    logistic, xgb = (reports[name] for name in ("logistic", "xgboost"))
    chosen = lambda report: report["sigmoid_oof"] if report["probability_source"] == "sigmoid" else report["raw"]
    ap_gain = xgb["raw"]["average_precision"] - logistic["raw"]["average_precision"]
    brier_penalty = chosen(xgb)["brier"] - chosen(logistic)["brier"]
    wins = sum(a["raw_ap"] > b["raw_ap"]
               for a, b in zip(xgb["folds"], logistic["folds"], strict=True))
    tree_wins = (ap_gain >= settings.min_primary_ap_gain
                 and brier_penalty <= settings.max_primary_brier_penalty
                 and wins > len(xgb["folds"]) / 2)
    criteria = dict(xgb_validation_ap_gain=ap_gain, xgb_selected_brier_penalty=brier_penalty,
                    xgb_validation_fold_ap_wins=wins, validation_folds=len(xgb["folds"]))
    rationale = (
        "XGBoost met the validation AP, probability quality, and fold consistency criteria. "
        "CPU histogram training and TreeExplainer support the explanation workflow."
        if tree_wins else
        "Logistic Regression leads or XGBoost missed a prespecified validation criterion. "
        "The linear model offers a simpler interpretation and efficient LinearExplainer support."
    )
    return dict(model="xgboost" if tree_wins else "logistic", rationale=rationale, **criteria)


def clustered_intervals(y, p, patients, reps, seed):
    """Percentile intervals from resampled patients, weighted by cluster multiplicity."""
    y, p = np.asarray(y), np.asarray(p)
    patients, patient_index = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(reps):
        drawn = rng.integers(len(patients), size=len(patients))
        weights = np.bincount(drawn, minlength=len(patients))[patient_index]
        present = weights > 0
        if np.unique(y[present]).size == 2:
            scores.append((roc_auc_score(y, p, sample_weight=weights),
                           average_precision_score(y, p, sample_weight=weights)))
    bounds = {}
    if scores:
        bounds = dict(zip(("roc_auc", "average_precision"),
                          np.quantile(scores, [0.025, 0.975], axis=0).T.tolist(), strict=True))
    return dict(requested=reps, valid=len(scores), percentile_95=bounds)


def evaluate(model, calibrator, report, test, schema, settings):
    x = model_input(test, schema)
    y = test[schema["label"]].to_numpy()
    with threadpool_limits(limits=settings.threads):
        started = perf_counter()
        margins = raw_margin(model, x)
        p = calibrated_probability(margins, calibrator)
        elapsed = perf_counter() - started
    labels = p >= report["threshold"]
    metrics = dict(**probability_metrics(y, p), threshold=report["threshold"],
                   probability_source=report["probability_source"], prediction_seconds=elapsed,
                   precision=precision_score(y, labels, zero_division=0),
                   recall=recall_score(y, labels, zero_division=0),
                   f1=f1_score(y, labels, zero_division=0),
                   confusion_matrix=confusion_matrix(y, labels, labels=[0, 1]).tolist(),
                   calibration_curve=calibration_bins(y, p),
                   intervals=clustered_intervals(y, p, test.subject_id, settings.bootstrap_reps,
                                                 settings.seed))
    return metrics, p
