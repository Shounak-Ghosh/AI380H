"""Laboratory feature contract and one-to-one assembly."""

import numpy as np
import pandas as pd

from cohort import BASE_FEATURES, LABEL, SPLIT_SEED, validate_cohort

# Feature, v3.1 itemid, dictionary label, category, source unit, allow zero.
# All selected rows have fluid Blood. Chemistry uses the standard blood
# chemistry analyte, with serum/plasma convention inferred from the method.
LAB_ITEMS = (
    ("wbc", 51301, "White Blood Cells", "Hematology", "K/uL", True),
    ("hemoglobin", 51222, "Hemoglobin", "Hematology", "g/dL", False),
    ("platelets", 51265, "Platelet Count", "Hematology", "K/uL", True),
    ("sodium", 50983, "Sodium", "Chemistry", "mEq/L", False),
    ("potassium", 50971, "Potassium", "Chemistry", "mEq/L", False),
    ("creatinine", 50912, "Creatinine", "Chemistry", "mg/dL", False),
    ("bun", 51006, "Urea Nitrogen", "Chemistry", "mg/dL", False),
    ("glucose", 50931, "Glucose", "Chemistry", "mg/dL", False),
    ("bicarbonate", 50882, "Bicarbonate", "Chemistry", "mEq/L", False),
)
LABS = [row[0] for row in LAB_ITEMS]
LAB_TIMES = ["first_lab_charttime", "last_lab_charttime", "last_lab_storetime"]
CATEGORICAL = ["gender", "admission_type", "admission_location", "insurance"]


def verified_lab_mapping(profile):
    selected = []
    for feature, itemid, label, category, unit, _ in LAB_ITEMS:
        item = profile.loc[profile.itemid.eq(itemid)]
        dictionary = item[["label", "fluid", "category"]].drop_duplicates()
        canonical = item.loc[item.valueuom.str.strip().str.lower().eq(unit.lower())]
        if (list(dictionary.itertuples(index=False, name=None)) != [(label, "Blood", category)]
                or canonical.numeric_measurements.sum() == 0):
            raise ValueError(f"Unexpected v3.1 dictionary definition or unit for item {itemid}")
        selected.append({
            "feature": feature, "itemid": itemid, "label": label, "fluid": "Blood",
            "category": category, "unit": unit,
            "numeric_in_unit": int(canonical.numeric_measurements.sum()),
        })
    return pd.DataFrame(selected)


def assemble(index, labs):
    data = index.merge(labs, on="hadm_id", how="left", validate="1:1")
    observed = data[LABS].notna()
    count = observed.sum(axis=1)
    if not count.eq(data.selected_labs.fillna(0)).all():
        raise ValueError("Selected lab counts disagree with wide columns")
    for name in LAB_TIMES:
        data[name] = pd.to_datetime(data[name])
        time = data[name]
        if not (time.notna().eq(count.gt(0)).all()
                and time.dropna().between(data.admittime[time.notna()],
                                          data.prediction_time[time.notna()]).all()):
            raise ValueError(f"Lab timing validation failed: {name}")
    measured = count.gt(0)
    if not (data.first_lab_charttime.le(data.last_lab_charttime)
            & data.last_lab_charttime.le(data.last_lab_storetime))[measured].all():
        raise ValueError("Invalid order of lab charting and result availability")
    for feature, _, _, _, _, allow_zero in LAB_ITEMS:
        values = data[feature].dropna().astype(float)
        if not (np.isfinite(values) & (values.ge(0) if allow_zero else values.gt(0))).all():
            raise ValueError(f"Invalid numeric lab values: {feature}")
        data[feature] = data[feature].astype("float64")
        data[f"{feature}_missing"] = (~observed[feature]).astype("int8")
    validate_cohort(data)
    return data.sort_values("hadm_id", ignore_index=True)


def feature_schema():
    sources = {
        "age": "patients.anchor_age, patients.anchor_year, admissions.admittime",
        "gender": "patients.gender",
        "admission_type": "admissions.admission_type",
        "admission_location": "admissions.admission_location",
        "insurance": "admissions.insurance",
        "prior_admissions": "earlier admission starts with discharge at or before index admission",
        "days_since_previous_discharge": "index start minus most recent completed discharge",
    }
    features = [
        {"name": name, "kind": "categorical" if name in CATEGORICAL else "numeric",
         "source": sources[name]} for name in BASE_FEATURES
    ]
    for name, itemid, _, _, unit, _ in LAB_ITEMS:
        features.extend([
            {"name": name, "kind": "numeric", "source": "labevents.valuenum",
             "itemid": itemid, "unit": unit},
            {"name": f"{name}_missing", "kind": "indicator", "source": f"{name} is missing"},
        ])
    return {"dataset": "physionet-data.mimiciv_3_1_hosp", "label": LABEL,
            "split_seed": SPLIT_SEED, "patient_group": "subject_id", "join_key": "hadm_id",
            "features": features}
