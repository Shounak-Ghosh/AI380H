# Experiment 5 review protocol

Version `experiment5-protocol-v1`

One reviewer with prior project and model-explanation familiarity evaluates the same 72 saved explanation records in a condition-blinded, single-reviewer evaluation. The primary sample contains 24 MIMIC-IV admissions, eight each from the frozen low-, medium- and high-predicted-risk strata. Each case contributes baseline A, grounded draft B and final verified delivery C. B and C may contain identical wording. Their records represent different delivery conditions for the same case.

## Selection and preparation

The frozen Experiment 2 checkpoint and its model reference are eligible sources. A case qualifies when all three saved deliveries have usable, nonempty parsed explanation prose. Missing, failed, malformed and truncated outputs are counted in an exclusion ledger before sampling. Select a fixed-seed sample of up to eight complete triplets per risk group. If a group falls short, keep its available complete triplets, report the shortfall and retain the 72-record completion target. Selection never uses human judgments. Optional Experiment 3 review uses up to four complete fictional profiles per reversal family, with both variants and all three conditions per profile. Each variant retains its own prediction and signed SHAP reference. Its records and results stay in a separate study directory.

Before formal review, use six to eight separate fictional calibration examples covering faithful, unfaithful, incomplete and ambiguous explanations. Practice the two-stage rating procedure and record any rubric interpretation decisions before beginning the primary pass. Calibration ratings stay outside the reported evaluation. Use the frozen rubric version throughout the study.

## Primary rating pass

Work locally in a private terminal using the reviewer command in [the Experiment 5 guide](README.md). The worksheet and its separate reference file expose only blind IDs, explanation and model reference fields. The terminal first displays the prose and asks for clarity and readability. It then displays the calibrated probability, feature values and units, signed SHAP contributions, model driver ranks and explanation drivers. Rate faithfulness, completeness and uncertainty, then select any unsupported-claim codes and enter brief evidence notes. The terminal saves each completed row and resumes with the next pending row. Take breaks as needed. Avoid review of the private investigator crosswalk or the automated findings during annotation.

## Repeat rating pass

After the primary pass locks, preferably wait at least three days. Use the repeat reviewer command. The repeat pass contains 12 primary explanations, four per condition from a separate fixed-seed draw, with new blind IDs and an independently randomized order. The terminal presents each item afresh and never displays the primary answers. Repeat ratings are saved separately and are used solely to measure intra-reviewer repeat consistency. The report includes missing repeats and their denominator. Exact agreement includes all matched labels. Quadratic-weighted Cohen’s kappa applies only to ordered endpoints, with at least eight eligible repeat pairs and neither pass having a category at or above 85%. `not_applicable` completeness is excluded from weighted kappa.

## Privacy and analysis

Keep review data in a dedicated owner-only directory outside Git. It contains source-derived prose, feature values, annotations, repeat linkage, condition codes and delivery information. Reviewers should use the terminal interface and rubric. Keep calibration separate and use local access controls. After annotation, validate the source, packet and annotation hashes before joining conditions. Aggregate results are exportable by explicit request. Review the disclosure policy before sharing aggregates, especially small strata. Do not include raw MIMIC text, patient data, case identifiers or reviewer notes in public files or notebook output.

Analyze within matched admissions. Paired contrast labels show right minus left. Faithfulness compares determinate judgments and reports all selected denominators as a sensitivity measure. Bootstrap percentages and paired contrasts by whole admission, or by whole fictional profile for optional synthetic results. The frozen verifier comparison preserves separate structured correctness, narrative flag and release status indicators. Narrative `no_flag` indicates that the scorer recorded neither a contradiction nor an ambiguous claim. It can coexist with omitted attributions. Release status applies to the C workflow. Illustrative real examples remain private.

## Limitations

A single reviewer with prior familiarity with the project and model explanation framework provides the annotations. No inter-reviewer reliability estimate is available. Hiding condition labels, automated judgments and C delivery source reduces direct exposure to study metadata, while expectations and recognizably deterministic fallback wording may still influence ratings. Repeat ratings measure within-reviewer stability. They cannot establish correctness or external validity. The findings are exploratory and do not establish clinical usability or definitive clinical ground truth. Comparisons by C delivery source describe groups selected by the verifier, with potential confounding by original explanation difficulty.
