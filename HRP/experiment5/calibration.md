# Fictional reviewer calibration

The six fictional examples let you practice with the same prompts used in the primary pass. From the `HRP` directory, run:

```bash
uv run --locked python src/human_explanation_evaluation.py calibrate
```

Score clarity and readability before the reference appears. Then rate the explanation against the shared fictional probability and signed SHAP contributions. After you finish the set, the terminal displays rubric notes about faithfulness and completeness. Use the scale anchors for your clarity and readability judgments.

Practice answers are temporary. Press Ctrl+C to exit and rerun the command to start over. The [practice source](calibration.json) includes the explanations, shared reference and feedback.
