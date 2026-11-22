"""Admission eligibility, observed readmission labels, and patient partitions."""

import numpy as np
import pandas as pd

LABEL = "readmitted_30d_unplanned"
HORIZON = pd.Timedelta("30d")
LANDMARK = pd.Timedelta("1d")
SPLIT_SEED = 380
UNPLANNED = {"DIRECT EMER.", "EW EMER.", "URGENT"}
PLANNED = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION", "AMBULATORY OBSERVATION"}
TRANSFER_IN = {"TRANSFER FROM HOSPITAL", "INTERNAL TRANSFER TO OR FROM PSYCH",
               "AMBULATORY SURGERY TRANSFER"}
TRANSFER_OUT = {"ACUTE HOSPITAL", "PSYCH FACILITY"}
BASE_FEATURES = ["age", "gender", "admission_type", "admission_location", "insurance",
                 "prior_admissions", "days_since_previous_discharge"]
FLOW_REASONS = (
    "missing_identifier", "duplicate_admission_id", "invalid_admission_interval",
    "incomplete_patient_timeline", "overlapping_index_admission", "missing_age",
    "under_18", "stay_under_24h", "in_hospital_death_or_unknown",
    "discharge_to_acute_or_psych", "same_instant_return", "overlapping_return",
    "competing_transfer", "competing_planned", "ambiguous_return",
    "death_before_30d", "eligible",
)


def admission_age(anchor_age, anchor_year, admittime):
    return anchor_age + admittime.dt.year - anchor_year


def build_cohort(admissions, patients):
    """Return eligible index admissions and mutually exclusive row-level flow counts."""
    a = admissions.copy().reset_index(drop=True)
    p = patients.copy()
    for name in ("admittime", "dischtime", "deathtime"):
        a[name] = pd.to_datetime(a[name])
    p["dod"] = pd.to_datetime(p["dod"])
    if p.subject_id.isna().any() or p.subject_id.duplicated().any():
        raise ValueError("Patient identifiers must be nonnull and unique")

    reason = pd.Series(pd.NA, index=a.index, dtype="string")

    def exclude(mask, code):
        reason.loc[reason.isna() & mask.fillna(False)] = code

    exclude(a.subject_id.isna() | a.hadm_id.isna(), "missing_identifier")
    exclude(a.hadm_id.duplicated(keep=False), "duplicate_admission_id")
    exclude(a.admittime.isna() | a.dischtime.isna() | a.dischtime.lt(a.admittime),
            "invalid_admission_interval")
    incomplete_subjects = a.loc[reason.notna(), "subject_id"].dropna()

    timeline = a.loc[reason.isna()].sort_values(["subject_id", "admittime", "hadm_id"])
    timeline = timeline.join(p.set_index("subject_id"), on="subject_id", validate="m:1")
    groups = timeline.groupby("subject_id", sort=False)
    latest_end = groups.dischtime.cummax()
    previous_max = latest_end.groupby(timeline.subject_id, sort=False).shift()
    next_start = groups.admittime.shift(-1)
    overlap = (previous_max.gt(timeline.admittime) | next_start.lt(timeline.dischtime)
               | timeline.duplicated(["subject_id", "admittime"], keep=False))
    next_overlap = overlap.groupby(timeline.subject_id, sort=False).shift(-1, fill_value=False)
    next_type = groups.admission_type.shift(-1)
    next_location = groups.admission_location.shift(-1)
    gap = next_start - timeline.dischtime
    in_window = gap.gt(pd.Timedelta(0)) & gap.le(HORIZON)
    timeline["age"] = admission_age(timeline.anchor_age, timeline.anchor_year, timeline.admittime)
    timeline["prediction_time"] = timeline.admittime + LANDMARK

    def exclude_index(mask, code):
        exclude(mask.reindex(a.index, fill_value=False), code)

    exclude_index(timeline.subject_id.isin(incomplete_subjects), "incomplete_patient_timeline")
    exclude_index(overlap, "overlapping_index_admission")
    exclude_index(timeline.age.isna(), "missing_age")
    exclude_index(timeline.age.lt(18), "under_18")
    exclude_index(timeline.dischtime.lt(timeline.prediction_time), "stay_under_24h")
    exclude_index(~timeline.hospital_expire_flag.eq(0).fillna(False) | timeline.deathtime.notna(),
                  "in_hospital_death_or_unknown")
    exclude_index(timeline.discharge_location.isin(TRANSFER_OUT), "discharge_to_acute_or_psych")
    exclude_index(gap.eq(pd.Timedelta(0)), "same_instant_return")
    exclude_index(in_window & next_overlap, "overlapping_return")
    exclude_index(in_window & next_location.isin(TRANSFER_IN), "competing_transfer")
    exclude_index(in_window & next_type.isin(PLANNED), "competing_planned")
    exclude_index(in_window & (~next_type.isin(UNPLANNED) | next_location.isna()
                              | next_location.eq("INFORMATION NOT AVAILABLE")),
                  "ambiguous_return")
    death_in_window = timeline.dod.le((timeline.dischtime + HORIZON).dt.normalize())
    return_before_or_on_death = next_start.dt.normalize().le(timeline.dod)
    exclude_index(death_in_window & ~(in_window & return_before_or_on_death), "death_before_30d")

    eligible = timeline.loc[reason.reindex(timeline.index).isna()].copy()
    eligible[LABEL] = in_window.loc[eligible.index].astype("int8")
    eligible["return_admittime"] = next_start.loc[eligible.index].where(eligible[LABEL].eq(1))
    eligible["eligible"] = True
    eligible = add_history(eligible, timeline)
    flow = (reason.fillna("eligible").value_counts().reindex(FLOW_REASONS, fill_value=0)
            .rename_axis("disposition").reset_index(name="admissions"))
    columns = ["subject_id", "hadm_id", "admittime", "dischtime", "prediction_time",
               "hospital_expire_flag", "return_admittime", "eligible", *BASE_FEATURES, LABEL]
    return eligible[columns].sort_values("hadm_id").reset_index(drop=True), flow


def add_history(index, timeline):
    """Count earlier-started valid encounters discharged by the index admission."""
    history = timeline[["subject_id", "admittime", "dischtime", "hadm_id"]].sort_values(
        ["dischtime", "admittime", "hadm_id"]
    )
    history["prior_admissions"] = history.groupby("subject_id", sort=False).cumcount() + 1
    history = history[["subject_id", "dischtime", "prior_admissions"]].rename(
        columns={"dischtime": "previous_discharge"}
    )
    # An eligible index has positive duration and no overlapping or same-start records.
    history_at_index = pd.merge_asof(
        index[["subject_id", "hadm_id", "admittime"]].sort_values("admittime"),
        history, left_on="admittime", right_on="previous_discharge",
        by="subject_id", direction="backward",
    ).set_index("hadm_id")
    out = index.join(history_at_index[["previous_discharge", "prior_admissions"]],
                     on="hadm_id", validate="1:1")
    out["prior_admissions"] = out.prior_admissions.fillna(0).astype("int64")
    out["days_since_previous_discharge"] = (
        (out.admittime - out.previous_discharge) / pd.Timedelta("1d")
    )
    return out


def validate_cohort(cohort):
    gap = cohort.return_admittime - cohort.dischtime
    checks = {
        "unique, nonnull admission IDs": cohort.hadm_id.is_unique and cohort.hadm_id.notna().all(),
        "eligible adults": cohort.eligible.all() and cohort.age.ge(18).all(),
        "24-hour landmark and stay": (cohort.prediction_time - cohort.admittime).eq(LANDMARK).all()
        and cohort.dischtime.ge(cohort.prediction_time).all(),
        "survived hospitalization": cohort.hospital_expire_flag.eq(0).all(),
        "binary labels and positive timing": cohort[LABEL].isin([0, 1]).all()
        and (gap.gt(pd.Timedelta(0)) & gap.le(HORIZON)).eq(cohort[LABEL].eq(1)).all()
        and gap.notna().eq(cohort[LABEL].eq(1)).all(),
        "prior utilization": cohort.prior_admissions.ge(0).all()
        and cohort.days_since_previous_discharge.isna().eq(cohort.prior_admissions.eq(0)).all()
        and cohort.days_since_previous_discharge.dropna().ge(0).all(),
    }
    for name, passed in checks.items():
        if not passed:
            raise ValueError(f"Cohort validation failed: {name}")


def partition(cohort, seed=SPLIT_SEED):
    """Shuffle sorted unique patients, using rounded cumulative 70/85% cutoffs."""
    subjects = np.sort(cohort.subject_id.unique())
    shuffled = np.random.default_rng(seed).permutation(subjects)
    train_end, validation_end = round(.70 * len(subjects)), round(.85 * len(subjects))
    labels = np.repeat(["train", "validation", "test"],
                       [train_end, validation_end - train_end, len(subjects) - validation_end])
    by_patient = pd.DataFrame({"subject_id": shuffled, "partition": labels})
    assignments = cohort[["subject_id", "hadm_id"]].merge(
        by_patient, on="subject_id", how="left", validate="m:1"
    ).sort_values("hadm_id", ignore_index=True)
    if assignments.partition.isna().any() or not assignments.hadm_id.is_unique:
        raise ValueError("Invalid split coverage")
    return assignments
