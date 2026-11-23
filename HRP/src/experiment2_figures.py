"""Aggregate report figures derived from the Experiment 2 analysis tables."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


COLORS = ("#56728a", "#157d7c", "#834f88")
CONDITIONS = ("A · Baseline", "B · SHAP-grounded", "C · SHAP-verified")


def _axes(title, ylabel, *, size=(8, 4.2)):
    figure, axis = plt.subplots(figsize=size, layout="constrained")
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.22)
    return figure, axis


def _condition_bars(rows, metric, title, ylabel, maximum):
    figure, axis = _axes(title, ylabel)
    values = [row[metric] for row in rows]
    bars = axis.bar(CONDITIONS, values, color=COLORS, width=.58)
    axis.set_ylim(0, maximum * 1.16)
    if maximum == 1:
        axis.set_yticks(np.linspace(0, 1, 5), labels=[f"{i}%" for i in range(0, 101, 25)])
    else:
        axis.set_yticks(range(maximum + 1))
    axis.bar_label(bars, labels=[f"{i:.1%}" if maximum == 1 else f"{i:.2f}" for i in values], padding=4)
    axis.set_xlabel("Valid outputs · " + " / ".join(f"{row['condition']} {row['valid']}/{row['total']}" for row in rows))
    return figure


def build_figures(result):
    structured, prose = result["structured"], result["narrative"]
    figures = {
        "experiment2_signed_agreement": _condition_bars(
            structured, "signed_top_three_agreement", "Signed top-three agreement",
            "Mean matching feature–direction pairs (max. 3)", 3),
        "experiment2_exact_ranking": _condition_bars(
            structured, "exact_top_three_ordering", "Exact top-three feature order",
            "Share of valid outputs", 1),
    }

    figure, axis = _axes("Structured correctness and prose coverage", "Share", size=(9, 4.8))
    series = (
        ("Structured correct", [r["structured_correct"] for r in structured], "#56728a"),
        ("Structured complete", [r["structured_complete"] for r in structured], "#157d7c"),
        ("All top directions in prose", [r["prose_completeness"] for r in prose], "#834f88"),
        ("Parser coverage", [r["parser_coverage"] for r in prose], "#b66c37"),
    )
    for index, (label, values, color) in enumerate(series):
        axis.bar(np.arange(3) + (index - 1.5) * .19, values, width=.18, label=label, color=color)
    axis.set_xticks(range(3), CONDITIONS)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylim(0, 1.08)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, frameon=False)
    figures["experiment2_structure_and_prose"] = figure

    counts = {row["outcome"]: row["count"] for row in result["interventions"]}
    figure, axis = _axes("Condition C output sources", "Case count")
    bars = axis.bar(["Accepted B draft", "Accepted Qwen3 revision", "Deterministic fallback"],
                    [counts[key] for key in ("draft_accepted", "revision_accepted", "fallback_used")], color=COLORS, width=.58)
    axis.bar_label(bars, padding=4)
    axis.set_ylim(0, max(1, counts["verified_complete"]) * 1.07)
    figures["experiment2_intervention_sources"] = figure

    reasons = [row for row in result["rejections"] if row["stage"] == "original B"]
    figure, axis = _axes("Why the verifier rejected the grounded drafts", "", size=(9, 4.6))
    bars = axis.barh([row["reason"] for row in reasons], [row["count"] for row in reasons], color="#157d7c")
    axis.invert_yaxis()
    axis.set_xlim(0, max(row["denominator"] for row in reasons) * 1.09)
    axis.bar_label(bars, padding=4)
    axis.set_xlabel("Grounded responses · categories can overlap")
    figures["experiment2_grounded_rejections"] = figure

    costs = result["costs"][:3]
    figure, axis = _axes("Recorded request time", "Sum of request latency (minutes)")
    values = [row["seconds"] / 60 for row in costs]
    bars = axis.bar(["A generation", "B generation", "C revision only"], values, color=COLORS, width=.58)
    axis.bar_label(bars, labels=[f"{v:.1f} min\n{row['calls']} calls" for v, row in zip(values, costs)], padding=4)
    axis.set_ylim(0, max(values) * 1.25)
    axis.set_xlabel("Full C delivery includes B generation and C revision")
    figures["experiment2_request_cost"] = figure

    primary = [row for row in result["paired"]
               if row["metric"] == "signed_top_three_agreement" and row["analysis"] == "complete_case"]
    figure, axis = _axes("Paired improvement in signed agreement", "", size=(8, 3.4))
    for index, (row, color) in enumerate(zip(primary, COLORS)):
        axis.hlines(index, row["ci95_low"], row["ci95_high"], color=color, linewidth=2)
        axis.scatter(row["mean_difference"], index, color=color, s=46, zorder=3)
    axis.set_yticks(range(len(primary)), [f"{r['comparison']} · n={r['pairs']}" for r in primary])
    axis.invert_yaxis()
    axis.axvline(0, color="#646464", linewidth=1)
    axis.set_xlabel("Mean paired difference and 95% bootstrap interval")
    axis.xaxis.set_major_locator(MaxNLocator(6))
    figures["experiment2_paired_improvement"] = figure
    return figures
