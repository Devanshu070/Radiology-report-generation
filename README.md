# Clinical Radiology Report Completion

Radiologists often dictate findings under time pressure while working from a structured, modality-specific examination template. For X-Ray, MRI, CT, and Ultrasound reports, the task is to turn transcribed dictation into a complete report: place observations in the appropriate fields, replace only normal statements the dictation contradicts, preserve other applicable normal text, and retain clinical details such as negation, laterality, severity, and measurements.

## Synthetic example

The following example is fictional and illustrates the intended transformation. It is not a real patient report or clinical guidance.

### 1. Input examination template

```text
FINDINGS:
LUNGS: Lungs are clear without focal air-space opacity.
PLEURA: No pleural effusion or pneumothorax.
CARDIOMEDIASTINAL SILHOUETTE: Within normal limits.
OSSEOUS STRUCTURES: No acute osseous abnormality.

IMPRESSION:
No acute cardiopulmonary abnormality.
```

### 2. Physician dictation

```text
There is a small right pleural effusion with adjacent right basilar linear
atelectatic opacity. No pneumothorax. The cardiomediastinal silhouette is
within normal limits. No acute osseous abnormality.
```

### 3. Intended completed report

```text
FINDINGS:
LUNGS: Right basilar linear atelectatic opacity.
PLEURA: Small right pleural effusion. No pneumothorax.
CARDIOMEDIASTINAL SILHOUETTE: Within normal limits.
OSSEOUS STRUCTURES: No acute osseous abnormality.

IMPRESSION:
1. Small right pleural effusion with adjacent right basilar linear atelectatic opacity.
2. No pneumothorax.
```

The example shows the core challenge: dictated right-sided findings replace conflicting normal template statements, while unchanged cardiomediastinal and osseous findings remain. Laterality and negation are retained, and the impression summarizes the current case.
