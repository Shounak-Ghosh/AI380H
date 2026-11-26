# Frozen HRP results synthesis

Notebook 07 integrates five previous experiments. This is an aggregate report bundle. Exact human outcome counts, paired intervals, risk and delivery subgroups, and human–verifier cross-tabulations are available in authorized local review. Public human faithfulness is binned in 20 percentage-point bands (lower bound included, upper excluded except 100%). Text ratings are rounded to the nearest half-point. Tables retain separate study denominators.

## Figure captions

**00_pipeline** — results-synthesis-v1. The study reuses frozen reference artifacts and deliveries across experiments.

**01_test_precision_recall** — Experiment 1 · frozen model evaluation v1. Held-out test set, n=53,379 admissions. The horizontal line marks prevalence. Patient-cluster intervals for average precision are in the performance table.

**02_global_shap** — Experiment 1 · frozen model evaluation v1. Mean absolute grouped SHAP contribution in raw log-odds across 1,000 test admissions. Descriptive feature importance.

**03_grounding_agreement** — Experiment 2 · experiment2-analysis-v1. Descriptive means among valid explanations (A 70/72, B 72/72, C 72/72). See the matched comparison table for 95% bootstrap intervals.

**04_adversarial_agreement** — Experiment 3 · experiment3-analysis-v2. Valid/selected outputs by arm and variant (A intuitive 44/48, A counterintuitive 46/48, B intuitive 48/48, B counterintuitive 48/48, C intuitive 48/48, C counterintuitive 48/48). The paired contrast table gives whole-profile 95% intervals. C includes deterministic fallbacks.

**05a_explicit_detection** — Experiment 4 · experiment4-analysis-v1. Primary comparison of 120 fictional responses. A positive is a mathematical error. Only explicitly detected errors count as flags.

**05b_release_rejection** — Experiment 4 · experiment4-analysis-v1. The same 120 fictional responses. Withholding includes ambiguity and incompleteness. The metrics table reports profile-cluster intervals for both policies.

**06_human_overview** — Experiment 5 · experiment5-analysis-v1. One reviewer, 24 selected admissions. Faithfulness uses all selected ratings. Vertical segments are disclosure bands, and text means are rounded to half-points. The bands and points are descriptive.
