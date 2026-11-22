"""Write local Experiment 1 model comparisons and attribution figures."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

MODEL_LABELS = {"logistic": "Logistic Regression", "xgboost": "XGBoost"}


def save(fig, target):
    try:
        fig.savefig(target, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def evaluation_plots(directory, y, scores, reports, metrics):
    for title, filename, xlabel, ylabel, reference in (
        ("Test ROC", "roc", "False positive rate", "True positive rate", ([0, 1], [0, 1])),
        ("Test precision–recall", "precision_recall", "Recall", "Precision",
         ([0, 1], [y.mean(), y.mean()])),
        ("Test calibration", "calibration", "Mean predicted probability",
         "Observed event fraction", ([0, 1], [0, 1])),
        ("Validation calibration", "validation_calibration", "Mean predicted probability",
         "Observed event fraction", ([0, 1], [0, 1])),
    ):
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.plot(*reference, "--", color="0.55", label="Prevalence" if filename == "precision_recall"
                else "Reference")
        for family, probability in scores.items():
            label = MODEL_LABELS[family]
            if filename == "roc":
                x, values, _ = roc_curve(y, probability)
                ax.plot(x, values, label=f"{label} ({metrics[family]['roc_auc']:.3f})")
            elif filename == "precision_recall":
                values, x, _ = precision_recall_curve(y, probability)
                ax.step(x, values, where="post",
                        label=f"{label} ({metrics[family]['average_precision']:.3f})")
            elif filename == "calibration":
                curve = metrics[family]["calibration_curve"]
                ax.plot(curve["mean_predicted"], curve["observed_fraction"], "o-", label=label)
            else:
                raw = reports[family]["raw_curve"]
                sigmoid = reports[family]["sigmoid_oof_curve"]
                line, = ax.plot(raw["mean_predicted"], raw["observed_fraction"], "o-", label=f"{label} raw")
                ax.plot(sigmoid["mean_predicted"], sigmoid["observed_fraction"], "s:",
                        color=line.get_color(), label=f"{label} sigmoid (OOF)")
        if filename.endswith("calibration"):
            largest = max(np.max(line.get_xydata()) for line in ax.lines[1:])
            ax.set(xlim=(0, min(1, largest * 1.12)), ylim=(0, min(1, largest * 1.12)))
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
        save(fig, directory / f"{filename}.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for (family, result), ax in zip(metrics.items(), axes, strict=True):
        matrix = np.asarray(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, f"{count:,}", ha="center", va="center",
                    color="white" if count > matrix.max() / 2 else "black")
        ax.set(title=f"{MODEL_LABELS[family]} · {result['threshold']:.3f}", xlabel="Predicted",
               ylabel="Observed", xticks=[0, 1], yticks=[0, 1])
    save(fig, directory / "confusion_matrices.png")


def global_plots(directory, contributions, metadata, seed):
    importance = contributions.abs().mean().sort_values()
    labels = [metadata[name]["label"] for name in importance.index]
    fig, ax = plt.subplots(figsize=(7, 8), layout="constrained")
    ax.barh(labels, importance, color="#22677b")
    ax.set(xlabel="Mean absolute SHAP (raw log-odds)", title="Held-out feature importance")
    save(fig, directory / "shap_importance.png")

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    for row, name in enumerate(importance.index):
        ax.scatter(contributions[name], row + rng.uniform(-0.23, 0.23, len(contributions)),
                   s=6, alpha=0.18, color="#22677b", edgecolors="none")
    ax.axvline(0, color="0.5", linewidth=1)
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels,
           xlabel="Signed SHAP contribution (raw log-odds)", title="Held-out SHAP distribution")
    save(fig, directory / "shap_summary.png")
    return importance.sort_values(ascending=False).rename("mean_abs_shap")


def local_plots(directory, cases, contributions, metadata):
    for stratum, group in cases.groupby("risk_stratum"):
        middle = group.sort_values("predicted_risk").iloc[len(group) // 2]
        features = contributions.loc[contributions.hadm_id.eq(middle.hadm_id)].head(10).iloc[::-1]
        labels = [metadata[name]["label"] for name in features.feature]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.barh(labels, features.shap, color=np.where(features.shap > 0, "#ae5139", "#22677b"))
        ax.axvline(0, color="0.5", linewidth=1)
        ax.set(title=f"Representative {stratum}-risk admission",
               xlabel="Contribution to raw log-odds relative to training baseline")
        save(fig, directory / f"local_{stratum}.png")
