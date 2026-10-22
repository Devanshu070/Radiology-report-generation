# Report evaluation

The repository's shared evaluator is `evaluation/res.py`. It provides a local, RES-style text comparison for comparing a generated report with a reference report while giving higher heuristic weight to findings changed from the supplied template and to clinically meaningful tokens.

This implementation is an approximation. It is not an official scorer, a clinical validation method, or evidence that generated reports are safe for clinical use. Lower scores indicate closer textual agreement under this implementation.

## Interface

```python
from evaluation.res import res_proxy

result = res_proxy(
    template_text=template_report,
    reference_text=reference_report,
    prediction_text=generated_report,
)
```

The returned mapping contains:

- `res_proxy`: combined score from 0 to 1.
- `findings_proxy`: weighted findings score from 0 to 1.
- `impression_proxy`: impression score from 0 to 1.
- `parse_error`: empty for a parsed prediction, or a reason when the prediction is malformed.

Zero means a token-level match after the evaluator's normalization. A malformed prediction receives the maximum score of 1.0. A malformed template or reference raises `ValueError`, because those inputs define the comparison.

## Scoring formula

The combined score is a weighted blend of findings and impression disagreement:

```text
RES_proxy = 0.65 × Findings_proxy + 0.35 × Impression_proxy
```

For findings, let each `i` represent a labeled reference field or the unlabelled findings text, when present. Let `dᵢ` be the normalized weighted word-edit distance between that reference text and the corresponding prediction text. For labeled fields, assign `aᵢ = 3` if the reference content differs from the template and `aᵢ = 1` otherwise. Unlabelled reference findings use `aᵢ = 1`. Let `U` be the count of predicted fields absent from the reference, plus one if the prediction contains unlabelled findings when the reference does not. Then:

```text
Findings_proxy = min(1, (Σᵢ aᵢ × dᵢ + U) / max(Σᵢ aᵢ, 1))
Impression_proxy = d(reference_impression, predicted_impression)
```

Each `d(reference, prediction)` is computed using a weighted word-edit distance. If `R` and `P` are the normalized token sequences, token weights are `w(t)`, and `D(R, P)` is the minimum weighted insertion/deletion/substitution cost, then:

```text
d(R, P) = D(R, P) / max(Σᵣ∈R w(r), Σₚ∈P w(p))
```

An exact normalized token match scores 0; the distance is capped at 1. Insertions and deletions cost the weight of the inserted or deleted token. Substitutions cost 0 for equal tokens and otherwise the larger of the two token weights. Negation, laterality, severity, acuity, numbers, and selected units receive higher heuristic weights than common function words. Numbered list prefixes and common spelled-out metric-unit aliases are normalized before comparison.

At a high level, evaluation parses exactly one `FINDINGS` section followed by one `IMPRESSION` section, compares structured fields, weights changed reference fields more heavily than unchanged template fields, penalizes unexpected predicted content, then combines the findings and impression scores. A malformed prediction receives the maximum score of 1; malformed template or reference input raises `ValueError`.

The token list, field grammar, parent-heading inference, unit aliases, and weighting are explicit heuristics. They do not capture all clinical meaning, and token-level agreement cannot determine whether a report is medically correct.

## Usage in the project

Both report-generation approaches should call this module for local evaluation rather than maintaining separate scorer copies. Keep evaluation examples synthetic. Report aggregate scores alongside failure counts and clinician review; do not present proxy scores alone as evidence of clinical performance.
