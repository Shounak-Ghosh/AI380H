"""Publication-ready static summaries of the fixed Experiment 4 challenge."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

BLUE = "#195577"
ORANGE = "#ad5822"
GREEN = "#2b786b"
GRAY = "#757b86"


def _confusion(result, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.1), layout="constrained")
    c = result["confusion"]
    matrix = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", size=19,
                color="white" if value > matrix.max() / 2 else "#12283a")
    ax.set(xticks=[0, 1], xticklabels=["No error", "Error"], yticks=[0, 1],
           yticklabels=["Faithful", "Unfaithful"], xlabel="Policy prediction", ylabel="Independent gold",
           title=f"{title}\n{result['n']} valid, determinate responses")
    return fig


def _intervals(ax, data, names, color, *, xlabel="Share of responses"):
    for i, (stat, name) in enumerate(zip(data, names, strict=True)):
        x = stat["estimate"]
        if x is None:
            continue
        low, high = stat["ci95_low"], stat["ci95_high"]
        if low is not None:
            ax.plot([low, high], [i, i], color=color, lw=2)
        ax.scatter([x], [i], color=color, zorder=3)
        ax.text(1.03, i, f'{stat["numerator"]}/{stat["denominator"]}', va="center", size=9)
    ax.set(yticks=list(range(len(names))), yticklabels=names, xlim=(-.03, 1.19), xlabel=xlabel)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _errors(analysis):
    rows = analysis["error_matrix"]
    fig, ax = plt.subplots(figsize=(8.8, 6.8), layout="constrained")
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, label, color in (
        ("explicitly_detected", "Explicitly detected", BLUE),
        ("ambiguous_rejection", "Ambiguous rejection", ORANGE),
        ("missed_other_rejection", "Other rejection", GRAY),
        ("incorrectly_accepted", "Incorrectly accepted", GREEN),
    ):
        counts = np.array([r[key]["numerator"] for r in rows])
        ax.barh(y, counts, left=left, label=label, color=color)
        left += counts
    ax.set(yticks=y, yticklabels=[r["category"].replace("_", " ") + f'  (n={r["n"]})' for r in rows],
           xlabel="Responses within category", title="Which mathematical errors did the verifier identify?")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncols=2, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def _false_rejection(analysis):
    rows = [r for r in analysis["false_rejection"] if r["style"] != "all" and not r["style"].startswith("category_")]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    names = {"reference": "Reference-relative", "pronoun": "Pronominal", "alias": "Direct with aliases"}
    _intervals(ax, [r["false_rejection"] for r in rows],
               [names.get(r["style"], r["style"].title()) for r in rows], ORANGE)
    ax.set_title("False rejection of faithful, complete prose")
    ax.set_xlabel("False rejection rate with 95% profile bootstrap intervals")
    return fig


def _layers(analysis):
    fig, ax = plt.subplots(figsize=(7.4, 3.9), layout="constrained")
    data, names = [], []
    for layer in ("structured", "narrative"):
        for metric in ("recall", "specificity"):
            data.append(analysis["layers"][layer]["metrics"][metric])
            names.append(f"{layer.title()} {metric}")
    _intervals(ax, data, names, BLUE, xlabel="Rate with 95% profile bootstrap intervals")
    ax.set_title("Explicit error detection by verification layer")
    return fig


def _corrections(corrections):
    s = corrections["summary"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    labels = ["Original accepted", "Full genuine revision", "Faithful prose repaired",
              "Accepted revision incomplete", "Failed revision", "Fallback delivered"]
    values = [s["original_accepted"], s["full_genuine_corrections"], s["false_rejection_repairs"],
              sum(r["trace"]["revision_accepted"] and r["revision_complete"] is False for r in corrections["rows"]),
              s["failed_revisions"], s["fallback"]]
    ax.barh(labels, values, color=[BLUE, GREEN, GREEN, ORANGE, GRAY, BLUE])
    for i, value in enumerate(values):
        ax.text(value + .12, i, str(value), va="center")
    ax.set(xlim=(0, max(values) + 1), xlabel="Scripted scenarios",
           title=f"Correction and delivery across {s['attempted']} scenarios")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def _policies(analysis):
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    data, names = [], []
    for name, key in (("Explicit detection", "explicit"), ("Release rejection", "release")):
        for metric in ("precision", "recall"):
            names.append(f"{name} · {metric}")
            data.append(analysis["primary"][key]["metrics"][metric])
    _intervals(ax, data, names, BLUE, xlabel="Rate with 95% profile bootstrap intervals")
    ax.set_title("Precision and recall for two fixed policies")
    return fig


def build_figures(analysis, corrections):
    return {
        "01_explicit_confusion": _confusion(analysis["primary"]["explicit"], "Explicit error detection"),
        "02_release_confusion": _confusion(analysis["primary"]["release"], "Conservative release gate"),
        "03_error_types": _errors(analysis), "04_false_rejection": _false_rejection(analysis),
        "05_verification_layers": _layers(analysis), "06_corrections": _corrections(corrections),
        "07_policy_precision_recall": _policies(analysis),
    }


def save_figures(directory, analysis, corrections):
    names = []
    for stem, figure in build_figures(analysis, corrections).items():
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            figure.savefig(Path(directory) / name, dpi=170, metadata=metadata)
            names.append(name)
        plt.close(figure)
    return names
