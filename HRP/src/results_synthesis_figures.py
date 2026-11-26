"""Curated figures for Notebook 07 and its aggregate report bundle."""

from io import BytesIO

from results_synthesis import ARMS, VERSION, coarsened_human_rows


def pipeline_figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.2, 2.15), layout="constrained")
    ax.set(xlim=(-.52, 4.52), ylim=(0, 1))
    ax.set_title("Study design across five experiments", loc="left", fontweight="bold")
    ax.axis("off")
    steps = ("XGBoost + SHAP\nModel reference", "A / B / C\nMatched admissions",
             "Attribution reversals\nFictional profiles", "Verifier challenge\nFictional responses",
             "Human evaluation\nMatched admissions")
    labels = [ax.text(index, .54, label, ha="center", va="center", fontsize=10.5,
                      bbox=dict(boxstyle="round,pad=.55", edgecolor="#456a8a", facecolor="#eef4f8"))
              for index, label in enumerate(steps)]
    for index, (left, right) in enumerate(zip(labels, labels[1:])):
        ax.annotate("", xy=(index + 1, .54), xytext=(index, .54),
                    arrowprops={"arrowstyle": "->", "color": "#456a8a", "lw": 1.2,
                                "patchA": left.get_bbox_patch(), "patchB": right.get_bbox_patch()})
    ax.text(2.5, .11, "Human ratings use saved A/B/C deliveries from Experiment 2.",
            ha="center", fontsize=9, color="#33465b")
    return fig


def grounding_figure(data):
    import matplotlib.pyplot as plt
    from experiment2_figures import build_figures

    figures = build_figures(data)
    figure = figures.pop("experiment2_signed_agreement")
    for unused in figures.values():
        plt.close(unused)
    return figure


def release_figure(data):
    from verifier_evaluation_figures import _confusion

    fig = _confusion(data["primary"]["release"], "Conservative release gate")
    fig.axes[0].set(xticklabels=["Released", "Withheld"], xlabel="Release decision")
    return fig


def human_figure(rows, *, fictional=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), width_ratios=(1.1, 1), layout="constrained")
    fig.suptitle("Human faithfulness and text quality by condition" if not fictional else
                 "Scripted ratings · fictional software fixture", fontweight="bold")
    colors = ("#456a8a", "#168578", "#a75039")
    faith, text = axes
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        low, high = (int(value) for value in row["faithful_share_band"].rstrip("%").split("–"))
        faith.vlines(index, low, high, color=color, lw=11, alpha=.8)
        faith.text(index, low - 4, f"{low}–{high}%", ha="center", va="top", fontsize=9)
    faith.set(xlim=(-.5, 2.5), ylim=(0, 106), ylabel="Faithful among selected (%)",
             title="Share in 20-point bands")
    for offset, field, label, color in ((-.17, "clarity_rounded_half_point", "Clarity", "#456a8a"),
                                        (.17, "readability_rounded_half_point", "Readability", "#a75039")):
        text.scatter([i + offset for i in range(3)], [row[field] for row in rows],
                     marker="o" if offset < 0 else "D", color=color, label=label, zorder=3)
    text.set(xlim=(-.5, 2.5), ylim=(.85, 5.23), yticks=range(1, 6), title="Means rounded to half-points",
             ylabel="Rating (1–5)")
    text.legend(loc="lower right", frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_xticks(range(3), [f"{arm}\nn={row['selected']}" for arm, row in zip(ARMS, rows, strict=True)])
        ax.set_xlabel("Condition")
    return fig


def figure_bytes(figure):
    import matplotlib.pyplot as plt

    try:
        stream = BytesIO()
        figure.savefig(stream, format="png", dpi=190, metadata={"Software": VERSION})
        return stream.getvalue()
    finally:
        plt.close(figure)


def captions(study):
    version = {row["experiment"]: row.get("analysis") for row in study.provenance}
    c = {
        "00_pipeline": f"{VERSION}. The study reuses frozen reference artifacts and deliveries across experiments.",
    }
    if 1 in study.data:
        d = study.data[1]
        test = next(row for row in d["performance"] if row["split"] == "test" and row["model"] == "xgboost")
        c["01_test_precision_recall"] = (
            f"Experiment 1 · {version[1]}. Held-out test set, n={test['n']:,} admissions. "
            "The horizontal line marks prevalence. Patient-cluster intervals for average precision are in the performance table.")
        c["02_global_shap"] = (
            f"Experiment 1 · {version[1]}. Mean absolute grouped SHAP contribution in raw log-odds across "
            f"{d['shap_diagnostics']['rows']:,} test admissions. Descriptive feature importance.")
    if 2 in study.data:
        sizes = ", ".join(f"{r['condition']} {r['valid']}/{r['total']}" for r in study.data[2]["structured"])
        c["03_grounding_agreement"] = (
            f"Experiment 2 · {version[2]}. Descriptive means among valid explanations ({sizes}). "
            "See the matched comparison table for 95% bootstrap intervals.")
    if 3 in study.data:
        sizes = ", ".join(f"{r['condition']} {r['variant']} {r['valid']}/{r['total']}"
                          for r in study.data[3]["faithfulness"])
        c["04_adversarial_agreement"] = (
            f"Experiment 3 · {version[3]}. Valid/selected outputs by arm and variant ({sizes}). "
            "The paired contrast table gives whole-profile 95% intervals. C includes deterministic fallbacks.")
    if 4 in study.data:
        c["05a_explicit_detection"] = (
            f"Experiment 4 · {version[4]}. Primary comparison of 120 fictional responses. "
            "A positive is a mathematical error. Only explicitly detected errors count as flags.")
        c["05b_release_rejection"] = (
            f"Experiment 4 · {version[4]}. The same 120 fictional responses. Withholding includes ambiguity "
            "and incompleteness. The metrics table reports profile-cluster intervals for both policies.")
    if 5 in study.data:
        d = study.data[5]
        c["06_human_overview"] = (
            f"Experiment 5 · {version[5]}. One reviewer, {d['clusters']} selected admissions. "
            "Faithfulness uses all selected ratings. Vertical segments are disclosure bands, and text means "
            "are rounded to half-points. The bands and points are descriptive.")
        c["06_local_human_faithfulness"] = (
            f"Experiment 5 · {version[5]}. One reviewer, {d['clusters']} selected admissions. "
            "Points use determinate judgments and lines are admission-bootstrap 95% intervals.")
        c["07_local_human_text_ratings"] = (
            f"Experiment 5 · {version[5]}. Ordinal rating counts, {d['clusters']} selected per condition. "
            "Descriptive distributions. See the paired table for contrasts and intervals.")
        c["08_local_human_matched_changes"] = (
            f"Experiment 5 · {version[5]}. Within-admission mean changes for {d['clusters']} matched admissions. "
            "Lines are admission-bootstrap 95% intervals. Eligible counts differ by outcome and appear in the paired table.")
    return c


def public_figures(study, tables, directory):
    metadata = captions(study)
    output = {
        "00_pipeline": figure_bytes(pipeline_figure()),
        "01_test_precision_recall": study.images["01_test_precision_recall"],
        "02_global_shap": study.images["02_global_shap"],
        "03_grounding_agreement": figure_bytes(grounding_figure(study.data[2])),
        "04_adversarial_agreement": study.images["04_adversarial_agreement"],
        "05a_explicit_detection": study.images["05a_explicit_detection"],
        "05b_release_rejection": figure_bytes(release_figure(study.data[4])),
        "06_human_overview": figure_bytes(human_figure(tables["06_human_ratings"])),
    }
    for name, image in output.items():
        (directory / f"{name}.png").write_bytes(image)
    return {name: metadata[name] for name in output}


def fictional_figure(study):
    return human_figure(coarsened_human_rows(study.data[5]), fictional=True)
