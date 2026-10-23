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

## Evaluation

The project includes a shared, local RES-style report comparison in [`evaluation/res.py`](evaluation/res.py). It compares a completed report with a reference while accounting for which findings changed from the input template. Lower scores indicate closer text agreement under this implementation.

This repository implements an explicitly defined **RES-style proxy**, not the official RES metric. Its combined score is:

```text
RES_proxy = 0.65 × Findings_proxy + 0.35 × Impression_proxy
```

The findings score uses weighted normalized word-edit distances, assigning greater weight to fields changed from the template and penalizing unexpected fields. Negation, laterality, severity, acuity, and measurements receive higher heuristic token weights. See [`docs/evaluation.md`](docs/evaluation.md) for the full formula, scoring details, and limitations.

This proxy supports development and regression testing; it is not clinical validation or evidence of clinical safety. Public tests use synthetic examples and do not require private clinical datasets.

## In-Context RAG workflow

The `in_context_rag/` package contains an end-to-end local workflow: it selects up to five demonstrations from the training partition, constructs a structured Groq request, validates the response, assembles the completed report, calculates the local RES proxy, and can create an escaped HTML review page.

Install the base dependencies and configure local data and provider credentials:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY or one or more numbered GROQ_API_KEY_N values in .env.
# CLINICAL_DATA_ROOT is optional; it defaults to data/private/.
```

The application loads the ignored root `.env` file without overriding values
already exported in the shell. It supports a single `GROQ_API_KEY` and the
numbered `GROQ_API_KEY_1` through `GROQ_API_KEY_10` slots; configured keys are
rotated between requests. The current pipeline uses Groq only; legacy
OpenRouter/Ollama settings from earlier experiments are not consumed. Keep
real credentials in `.env` and never commit them.

Prepare a deterministic split, then run the held-out test workflow and create a review page:

```bash
python -m in_context_rag.cli prepare
python -m in_context_rag.cli run --split test --top-k 5
python -m in_context_rag.cli review
```

The stages can also be run separately with `retrieve`, `generate`, and `evaluate`. Generated reports, embedding/response caches, split manifests, and review pages are written under ignored `artifacts/`. Before sending any clinical text to a hosted model, confirm that the data is appropriately de-identified and that the intended use is authorized. Model output requires qualified human review and is not a clinical decision.
