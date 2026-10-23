# In-Context RAG

This package owns its clinical-data loading, template parsing, validation,
retrieval, generation, evaluation, and review logic. It does not share
preprocessing with the QLoRA approach.

## Local input data

By default, the loader reads `data/private/train.csv` from the repository root.
Set `CLINICAL_DATA_ROOT` to use a different local data directory. The expected
training columns are `case_id`, `modality`, `body_part`, `study_description`,
`patient_age_band`, `patient_sex`, `template_content`, `dictation`, and
`report`. Inference rows may omit `report` when loaded with
`require_report=False`. Extra CSV columns are ignored; source rows and reports
are not included in this repository.

```python
from in_context_rag.preprocess import (
    create_split_manifest,
    load_cases,
    to_prompt_case,
    training_pool,
)

cases = load_cases()  # CLINICAL_DATA_ROOT/train.csv, or data/private/train.csv
splits = create_split_manifest(cases, seed=42)
demonstrations = training_pool(cases, splits)
prompt_case = to_prompt_case(cases[0])
```

The split manifest assigns each case deterministically to train, validation,
or test based on a seeded hash of its case ID, independent of CSV row order.
Only labeled training cases are returned by `training_pool`; validation and
test IDs are excluded so they cannot become few-shot demonstrations. The
manifest and exact input records must match when constructing that pool.

`to_prompt_case` separates the template's single `FINDINGS` section and single
`IMPRESSION` section into prompt variables while preserving their text. The
`in_context_rag.cli` module exposes `prepare`, `retrieve`, `generate`,
`evaluate`, `review`, and `run` commands. See the root README for the complete
CLI sequence. Generation sends dictation and selected demonstrations to the
configured Groq API, so only use appropriately de-identified data with
authorization for the intended processing.
