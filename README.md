# Clinical Radiology Report Completion

Radiologists dictate observations under time pressure while using structured templates for X-Ray, MRI, CT, and Ultrasound exams. The challenge is to place dictated findings in the right fields, replace only contradicted normal statements, preserve the rest of the template, and retain details such as negation, laterality, and measurements.

## Example

Fictional example; not a real patient report or clinical guidance.

**Template**

```text
FINDINGS:
LUNGS: Clear.
PLEURA: No effusion or pneumothorax.
HEART: Normal size.

IMPRESSION:
No acute cardiopulmonary abnormality.
```

**Dictation**

```text
Small right pleural effusion with right basilar atelectatic opacity. No pneumothorax.
```

**Completed report**

```text
FINDINGS:
LUNGS: Right basilar atelectatic opacity.
PLEURA: Small right pleural effusion. No pneumothorax.
HEART: Normal size.

IMPRESSION:
1. Small right pleural effusion with right basilar atelectatic opacity.
2. No pneumothorax.
```

## Project guides

- [In-context report completion](docs/in-context-rag.md) retrieves relevant training examples and uses them to guide report generation.
- [Biomedical retrieval and QLoRA](docs/hybrid-retrieval-qlora.md) combines biomedical text retrieval with fine-tuning of a Qwen model.
- [Evaluation and RES-style proxy](docs/evaluation.md) describes the local metric, its formula, and its limitations.

**Reported result:** the second approach achieved an RES score of **0.21**. The linked evaluation guide describes the repository's local proxy; compare scores only when evaluation setups match.

## Data and setup

Clinical CSVs are excluded from the repository. For local workflows, place authorized data in `data/private/` or set `CLINICAL_DATA_ROOT`. Public tests use synthetic fixtures and do not need private data.

Install base dependencies with `python -m pip install -r requirements.txt`. QLoRA training dependencies are optional and listed in `requirements-finetune.txt`.

Generated reports require qualified human review. The RES-style proxy is a development tool, not clinical validation or evidence of safety.
