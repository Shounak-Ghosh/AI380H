# Experiment 5 reviewer guide

Use [the rubric](rubric.md), [protocol](protocol.md) and [fictional calibration set](calibration.md) before opening the primary review. Work in a local terminal from `HRP`.

```bash
uv run --locked python src/human_explanation_evaluation.py calibrate
```

The practice session uses the live rating prompts and shows rubric feedback after the six examples. Practice answers are temporary.

```bash
uv run --locked python src/human_explanation_evaluation.py setup --write-worksheets
uv run --locked python src/human_explanation_evaluation.py annotate --pass primary
```

The default private study directory is `~/Library/Application Support/AI380H/MIMIC-IV/experiment5-private-v1`. Use `--directory` consistently if you need a different owner-only directory outside Git. The annotation command saves each completed row. Press Ctrl+C to pause and run the same command to resume. The primary pass locks at completion. After a break of preferably three or more days, open a fresh terminal for the second pass.

```bash
uv run --locked python src/human_explanation_evaluation.py annotate --pass repeat
```

Use `review` to inspect progress without writing. Use `analyze` to validate source, selection and annotation checksums and calculate aggregates after the primary pass locks. `--export results-v1` saves a versioned aggregate JSON and report figures in the private study directory. If the annotation snapshot changes, pick a new export version. Keep `private-example-ids.json` and every other case-level file private. Use `fictional` to exercise both studies with scripted software fixtures. For optional Experiment 3 setup and analysis, add `--study synthetic`. Its worksheet is separate and has no repeat pass.

Experiment completion requires 72 valid primary annotations, all 12 repeat ratings or transparent reporting of missing repeats, verified source and annotation provenance, matched contrasts, intra-reviewer repeat consistency, reproducible aggregates and figures, and a passing local test suite. The study directory stores the investigator-only crosswalk separately from the reviewer worksheets and references.
