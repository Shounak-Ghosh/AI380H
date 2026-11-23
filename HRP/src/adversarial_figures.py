"""Aggregate report figures for the synthetic H3 benchmark."""

import numpy as np
from matplotlib.ticker import PercentFormatter

from adversarial_analysis import ATTRIBUTES, CLAIM_OUTCOMES
from experiment2_figures import _axes


VARIANT_LABELS = ("Intuitive", "Counterintuitive")
COLORS = ("#247b83", "#a55327", "#476399", "#9c9a8d", "#694875")
CLAIM_LABELS = ("Model-faithful", "Intuitive substitution", "Other incorrect",
                "Abstention or omission", "Unverifiable")


def _stacked(title, ylabel, ticks, series):
    figure, axis = _axes(title, ylabel, size=(9, 4.8))
    bottom = np.zeros(len(ticks))
    for index, (name, values) in enumerate(series):
        axis.bar(ticks, values, bottom=bottom, width=.68, label=name, color=COLORS[index])
        bottom += values
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylim(0, 1.05)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    return figure


def build_figures(result):
    figure, axis = _axes("H3 · Signed top-three agreement", "Mean matches (out of 3)")
    for condition, color in zip("ABC", COLORS):
        rows = [r for r in result["faithfulness"] if r["condition"] == condition]
        axis.plot(VARIANT_LABELS, [r["signed_top_three_agreement"] for r in rows],
                  marker="o", color=color, label=f"Condition {condition}")
    axis.set_ylim(0, 3.12)
    axis.legend(frameon=False)
    figures = {"01_h3_interaction": figure}

    figure, axis = _axes("Correct direction switches among drivers ranked in both models",
                        "Share of eligible feature–profile pairs", size=(9, 4.8))
    for offset, column, label, color in (
        (-.18, "structured_switch_rates", "Structured drivers", COLORS[0]),
        (.18, "switch_rates", "Structured drivers and prose", COLORS[1]),
    ):
        rows = [r for r in result[column] if r["top_three_in_both_only"] and r["outcome"] == "correct_switch"]
        bars = axis.bar(np.arange(3) + offset, [r["rate"] or 0 for r in rows],
                        width=.34, color=color, label=label)
        axis.bar_label(bars, labels=[f"{r['count']}/{r['denominator']}" for r in rows], padding=3)
    axis.set_xticks(range(3), [f"Condition {c}" for c in "ABC"])
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylim(0, 1.13)
    axis.legend(loc="upper left", frameon=False)
    figures["02_model_switch"] = figure

    rows = [r for r in result["claim_rates"] if not r["top_three_only"]]
    figures["03_clinical_substitution"] = _stacked(
        "Targeted feature outcomes in structured drivers and prose", "Share of targeted feature claims",
        [f"{c}\n{v}" for c in "ABC" for v in VARIANT_LABELS],
        [(label, [r["rate"] or 0 for r in rows if r["outcome"] == outcome])
         for label, outcome in zip(CLAIM_LABELS, CLAIM_OUTCOMES, strict=True)])

    rows = result["interventions"]
    figures["04_verification_outcomes"] = _stacked(
        "Condition C delivery sources", "Share of completed C outputs",
        [f"{label} · n={row['completed']}" for label, row in zip(VARIANT_LABELS, rows, strict=True)],
        [(label, [row[key] / row["completed"] if row["completed"] else 0 for row in rows])
         for label, key in (("Accepted original", "draft_accepted"), ("Accepted Qwen3 revision", "revision_accepted"),
                            ("Deterministic fallback", "fallback_used"))])

    figure, axis = _axes("Agreement by additional attribution properties", "Mean matches (out of 3)", size=(10, 4.8))
    rows = result["attributes"]
    groups = [(variant, attribute, condition) for variant in ("intuitive", "counterintuitive")
              for attribute in ATTRIBUTES for condition in "ABC"]
    x = np.arange(len(groups))
    for offset, present, label, color in ((-.18, False, "Attribute absent", COLORS[0]),
                                           (.18, True, "Attribute present", COLORS[1])):
        values = [next(r["signed_top_three_agreement"] for r in rows
                       if (r["variant"], r["attribute"], r["condition"], r["present"])
                       == (*group, present)) for group in groups]
        axis.bar(x + offset, [value if value is not None else np.nan for value in values],
                 width=.36, label=label, color=color)
    axis.set_xticks(x, [condition for _, _, condition in groups])
    for index, label in enumerate(("Intuitive\nMixed directions", "Intuitive\nConflicting importance",
                                   "Counterintuitive\nMixed directions", "Counterintuitive\nConflicting importance")):
        axis.text(index * 3 + 1, -.13, label, transform=axis.get_xaxis_transform(), ha="center", va="top")
    axis.set_ylim(0, 3.2)
    axis.legend(frameon=False, loc="upper center", ncol=2)
    figures["05_attribution_properties"] = figure
    return figures
