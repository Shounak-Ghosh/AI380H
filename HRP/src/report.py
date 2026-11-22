"""Local descriptive evidence and plots for the cohort notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cohort import FLOW_REASONS, LABEL
from features import LABS

MIN_CELL = 11
QUALITY_END = FLOW_REASONS.index("overlapping_index_admission") + 1
CLINICAL_END = FLOW_REASONS.index("discharge_to_acute_or_psych") + 1


def suppress_counts(counts):
    """Mask small positive cells and a second cell when totals could expose one."""
    result = counts.astype("Int64").copy()
    small = result.between(1, MIN_CELL - 1)
    if small.sum() == 1:
        complement = result[~small & result.gt(0)]
        if not complement.empty:
            small.loc[complement.idxmin()] = True
    return result.mask(small)


def build_report(data, flow, splits, mapping):
    report = {"lab_mapping": mapping.copy()}
    flow = flow.copy()
    # The flow diagram presents stage totals. Suppress within each stage so
    # subtracting the other detailed counts cannot reconstruct a rare cell.
    flow["admissions"] = pd.concat([
        suppress_counts(flow.admissions.iloc[:QUALITY_END]),
        suppress_counts(flow.admissions.iloc[QUALITY_END:CLINICAL_END]),
        suppress_counts(flow.admissions.iloc[CLINICAL_END:-1]),
        flow.admissions.iloc[-1:].astype("Int64"),
    ])
    report["cohort_flow"] = flow
    labels = data[LABEL].value_counts().reindex([0, 1], fill_value=0)
    safe_counts = suppress_counts(labels)
    report["cohort_summary"] = pd.DataFrame([{
        "patients": data.subject_id.nunique(), "admissions": len(data),
        "readmissions": safe_counts[1],
        "readmission_pct": 100 * safe_counts[1] / len(data) if len(data) else np.nan,
    }])
    report["labels"] = safe_counts.rename_axis(LABEL).reset_index(name="admissions")
    for field in ("admission_type", "prior_admissions"):
        values = data[field].fillna("(missing)") if field == "admission_type" else pd.cut(
            data[field], [-1, 0, 1, 2, 4, 9, float("inf")], labels=["0", "1", "2", "3–4", "5–9", "10+"]
        )
        counts = values.value_counts()
        report[field] = suppress_counts(counts).rename_axis(field).reset_index(name="admissions")
    report["age_summary"] = data.age.describe().drop(["min", "max"]).to_frame("age").T
    availability = []
    lab_summary = []
    for lab in LABS:
        missing = data[lab].isna().sum()
        counts = suppress_counts(pd.Series({"missing": missing, "available": len(data) - missing}))
        availability.append({"feature": lab, **counts.to_dict(),
                             "missing_pct": 100 * counts["missing"] / len(data) if len(data) else np.nan})
        group_counts = suppress_counts(data.groupby(LABEL)[lab].count().reindex([0, 1], fill_value=0))
        for outcome, count in group_counts.items():
            stat = {"feature": lab, LABEL: outcome, "n": count}
            if pd.notna(count) and count >= MIN_CELL:
                values = data.loc[data[LABEL].eq(outcome), lab]
                stat.update(mean=values.mean(), sd=values.std(), p25=values.quantile(.25),
                            median=values.median(), p75=values.quantile(.75))
            lab_summary.append(stat)
    report["lab_missingness"] = pd.DataFrame(availability)
    lab_count = data[LABS].notna().sum(axis=1)
    completeness = pd.Series({"none": lab_count.eq(0).sum(),
                              "some": lab_count.between(1, len(LABS) - 1).sum(),
                              "all": lab_count.eq(len(LABS)).sum()})
    report["lab_completeness"] = suppress_counts(completeness).rename_axis(
        "labs_available"
    ).reset_index(name="admissions")
    report["labs_by_outcome"] = pd.DataFrame(lab_summary)
    labeled = splits.merge(data[["hadm_id", LABEL]], on="hadm_id", validate="1:1")
    split_rows = []
    for name, group in labeled.groupby("partition", sort=False):
        counts = suppress_counts(group[LABEL].value_counts().reindex([0, 1], fill_value=0))
        split_rows.append({"partition": name, "patients": group.subject_id.nunique(),
                           "admissions": len(group), "readmissions": counts[1],
                           "readmission_pct": 100 * counts[1] / len(group)})
    report["partitions"] = pd.DataFrame(split_rows).set_index("partition").reindex(
        ["train", "validation", "test"]
    ).reset_index()
    return report


def lab_range_audit(data):
    """Private distribution checks. High-value review thresholds never clip source data."""
    thresholds = [100, 30, 1500, 180, 10, 25, 200, 1000, 60]
    rows = []
    for lab, high in zip(LABS, thresholds, strict=True):
        values = data[lab].dropna()
        rows.append({"feature": lab, "min": values.min(), "p01": values.quantile(.01),
                     "p99": values.quantile(.99), "max": values.max(),
                     "review_above": high, "n_above": int(values.gt(high).sum())})
    return pd.DataFrame(rows)


def make_plots(data, flow, report):
    figures = {}
    stages = flow.set_index("disposition").admissions
    included = len(data)
    first = int(stages.sum())
    quality = first - int(stages.loc[:"overlapping_index_admission"].sum())
    clinical = quality - int(stages.loc["missing_age":"discharge_to_acute_or_psych"].sum())
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.set_axis_off()
    labels = [f"Source admission records • {first:,}",
              f"Valid non-overlapping index candidates • {quality:,}",
              f"Adult stays ≥24 h, survived, no outgoing acute transfer • {clinical:,}",
              f"Eligible after outcome assessment • {included:,}"]
    for idx, (y, label) in enumerate(zip([.88, .63, .38, .13], labels, strict=True)):
        ax.text(.5, y, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=.55", "facecolor": "#e8f2f6", "edgecolor": "#30687d"})
        if idx:
            ax.annotate("", xy=(.5, y + .055), xytext=(.5, y + .195), xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": "#30687d"})
    figures["cohort_flow"] = fig
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    label_counts = report["labels"].admissions.to_numpy(dtype=float, na_value=np.nan)
    ax.bar(["No observed qualifying return", "Observed unplanned return"], label_counts,
           color=["#447d96", "#cf6e53"])
    for i, value in enumerate(label_counts):
        if np.isnan(value):
            ax.text(i, 0, "Suppressed", ha="center", va="bottom")
    ax.set_ylabel("Admissions")
    ax.set_title("30-day in-system readmission")
    figures["label_distribution"] = fig
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    missing = report["lab_missingness"]
    rates = missing.missing_pct.to_numpy(dtype=float, na_value=np.nan)
    ax.barh(missing.feature, rates, color="#447d96")
    for i, value in enumerate(rates):
        if np.isnan(value):
            ax.text(1, i, "Suppressed", va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Admissions missing (%)", title="First-24-hour lab missingness")
    figures["lab_missingness"] = fig
    return figures
