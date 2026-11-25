# Experiment 5 human rating rubric

Version `experiment5-rubric-v1`

Read only the explanation prose for clarity and readability. Assign both ratings before revealing the model reference and structured drivers. Use the reference for the remaining judgments. Judge model statements against the saved prediction and signed SHAP contributions. A positive SHAP contribution increases the model output relative to its reference on the stated output scale. A negative contribution decreases it. SHAP values express model attribution and do not establish clinical mechanisms or causation. The predicted readmission probability can be calibrated separately from the SHAP output scale. Feature rank follows absolute SHAP magnitude.

## Mathematical faithfulness

| Label | Rule |
|---|---|
| `faithful` | All explicit model claims agree with the saved probability, directions, ranks and other supported model information. An explanation can be faithful and incomplete. |
| `unfaithful` | At least one material explicit claim contradicts the reference or goes beyond what it supports. Cite the claim and the relevant reference in the notes. An explicit contradiction takes precedence over ambiguity. |
| `indeterminate` | The meaning of at least one material claim is too ambiguous to judge, with no explicit contradiction established. Note the ambiguity. |
| `invalid` | No meaningful evaluation is possible. Explain why. Text scores may be blank if the prose is unusable. |

Check both prose and structured drivers when supplied. Allow ordinary, correctly labeled rounding of the probability. Claims about population effects, diagnoses, treatment, prognosis or mechanism need support beyond this local attribution. Appropriate abstention about unidentified drivers can be mathematically faithful.

## Completeness

| Label | Rule |
|---|---|
| `complete` | Gives the correct predicted probability, relevant top-three drivers with correct directions and sufficient context to explain the model behavior. |
| `partial` | Gives some correct probability or driver information, with a material omission, incorrect direction, wrong probability or insufficient context for full coverage. |
| `insufficient` | Gives too little correct information to explain the prediction, even if the prose appears plausible. |
| `not_applicable` | Use for invalid explanations. |

## Clarity and readability

| Score | Clarity for a reasonably educated non-specialist | Naturalness and readability |
|---|---|---|
| 1 | Very difficult to understand | Highly awkward or fragmented |
| 2 | Mostly unclear | Repetitive or difficult to read |
| 3 | Understandable with effort | Acceptable with noticeably mechanical phrasing |
| 4 | Clear with minor issues | Mostly natural and fluent |
| 5 | Exceptionally clear | Natural, coherent and concise |

Allow appropriately explained technical terms. Score what the prose communicates, even when its claims later prove inaccurate. Use `unusable` at a text prompt if reading the prose is impossible.

## Uncertainty

Use `appropriate` for claims bounded by the model evidence, `overconfident` for overstated certainty or mechanism, `excessively_vague` for hedging that obscures meaningful supported information, or `not_applicable` for invalid text or no applicable uncertainty judgment.

## Unsupported claims and evidence notes

Select all applicable codes from `incorrect_direction`, `incorrect_rank`, `incorrect_probability`, `wrong_shap_scale`, `invented_feature`, `unsupported_clinical_inference`, `unsupported_causal_claim`, `prose_json_contradiction` and `other`. Briefly cite the wording and supporting reference for every `unfaithful` judgment or selected code. Keep notes inside the private study directory.
