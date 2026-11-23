# Experiment 4 gold rubric and design

Judge claims from the supplied fitted-model reference before exposing verifier
outputs. A response is **faithful** when its explicit assessable claims agree,
**unfaithful** when any claim is contradicted or asserts knowledge the reference
cannot support, and **indeterminate** when unresolved meaning could change the
judgment and no claim is known false. Abstention can be faithful. A malformed
response can contain faithful prose. Schema validity refers to the two-key
JSON contract, known unique feature names, up to three drivers, binary
directions, and an explanation of 1–90 words. Sentence count is advisory.
Plain text has no response-schema judgment.

Completeness requires the predicted probability, every non-negligible top-three
driver in the list, and each top-three direction in prose. Presence and
correctness are judged separately. Structured rank correctness uses the
reference ordering, including its explicit tie-break. Prose may truthfully
describe tied contributions as jointly strongest. Compare SHAP magnitudes for
rank and treat absolute values at most 1e-8 as negligible. Direction describes
a local contribution in raw log-odds under an increasing logistic link.
Compare displayed probabilities at their stated precision. Raw SHAP units are
not probability points. Imputation does not establish that a lab was measured.
Feature attribution cannot establish a diagnosis or clinical causality.

Record each claim's text, layer, feature, stated quantity, reference quantity,
status, and applicable errors. Categories cover direction, importance, rank,
probability, unsupported feature use, measurement, missingness, SHAP scale,
causality, diagnosis, JSON/prose conflict, omission, ambiguity, and malformed
responses. Overlapping error categories count once per case per category.

Twelve linear logistic references form six paired profiles with fixed patient
features and reversed coefficients. Each reference contributes five faithful
paraphrases, five controlled corruptions, and five boundary cases. The direct
control is paired with all four alternative wordings. Recipes and wordings
are assigned by a fixed rotation. The planned primary comparison uses valid,
determinate A and B responses with mathematical errors as positives. Group C
is a separate boundary cohort. A sensitivity analysis includes all valid,
determinate examples. Bootstrap samples draw entire profiles, retaining both
coefficient directions and matched paraphrases. These are authored gold
judgments. Independent human review has not occurred.

For optional live correction, rank candidate IDs by a fixed seeded hash and
take ten rejected examples from each of groups B errors, A faithful prose, and
C faithful valid incomplete responses. One correction request per ID, local
Qwen3 8B only, temperature zero, seed 42, and the Experiment 2/3 model digest.
A returned revision remains pending independent annotation.
