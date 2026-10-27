# In-Context Report Completion

This guide covers the local workflow implemented in `in_context_rag/`. It owns its own data loading, template parsing, validation, retrieval, generation, and report review. Its preprocessing is independent of the QLoRA workflow.

## Data and preprocessing

By default, the loader reads `data/private/train.csv` from the repository root. Set `CLINICAL_DATA_ROOT` to use a different local directory. Training rows require these columns:

`case_id`, `modality`, `body_part`, `study_description`, `patient_age_band`, `patient_sex`, `template_content`, `dictation`, and `report`.

Inference rows may omit `report` when loaded with `require_report=False`. Extra CSV columns are ignored. The repository does not include the clinical CSV files.

The split manifest deterministically assigns case IDs to train, validation, and test partitions using a seed. Only labeled training cases can be selected as few-shot examples; validation and test IDs are excluded to prevent leakage. The manifest and supplied records must agree exactly.

Template preprocessing extracts exactly one `FINDINGS` and one `IMPRESSION` section into prompt variables while preserving their text. Pydantic schemas validate loaded cases, prompt cases, few-shot pairs, and generated sections.

## Installation and configuration

From the repository root:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` or numbered `GROQ_API_KEY_1` through `GROQ_API_KEY_10` in `.env`. The project loads the root `.env` without overriding shell variables. Keep real credentials private. The default model is `openai/gpt-oss-120b`; generation parameters can be set through the documented environment variables in `.env.example`.

Hosted generation sends the current dictation, template text, and selected demonstrations to Groq. Use only data that is appropriately de-identified and authorized for that processing.

## Run the workflow

Prepare deterministic partitions and retrieve examples for held-out cases:

```bash
python -m in_context_rag.cli prepare
python -m in_context_rag.cli retrieve --split test --top-k 5
```

Generate reports, evaluate against references, and create a local HTML review page:

```bash
python -m in_context_rag.cli generate --split test --top-k 5
python -m in_context_rag.cli evaluate --split test
python -m in_context_rag.cli review
```

The combined `run` command executes the held-out test workflow:

```bash
python -m in_context_rag.cli run --split test --top-k 5
```

Use `python -m in_context_rag.cli --help` and command-specific `--help` for available options. Split manifests, retrieval and response caches, generated reports, evaluation outputs, and HTML review pages are written beneath ignored `artifacts/`.

## Engineering boundaries

- Demonstrations are selected from the training partition only, with a maximum of five examples per request.
- The model returns findings and impression bodies; local assembly verifies findings field labels and their order against the template.
- Provider requests use bounded retries, and structured responses are validated before use.
- Evaluation uses the shared local proxy in [`../evaluation/res.py`](../evaluation/res.py); see [the evaluation guide](evaluation.md) for its formula and limitations.
- Model output requires qualified human review and is not a clinical decision.
