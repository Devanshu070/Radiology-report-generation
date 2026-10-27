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

## Approach 2: Biomedical Retrieval + QLoRA Fine-Tuning

The `hybrid_retrieval_qlora/` package implements the dense semantic retrieval, persistent vector cache, ChatML formatting, target-only completion masking, 4-bit NF4 QLoRA fine-tuning of `Qwen/Qwen2.5-3B-Instruct`, adapter inference, and clinically weighted RES evaluation for Approach 2.

### Retrieval & Cache Architecture

- **PubMedBERT Dense Embeddings:** Dense vectors are computed using `NeuML/pubmedbert-base-embeddings` (revision `b79526d6ef3645e0df4530322e266f24c829f5ef`), outputting unit-normalized 768-dimensional vectors.
- **Sliding-Window Chunk Pooling:** Dictations exceeding the 512-token sequence limit are windowed with 64-token overlap. Consecutive chunks from the same case discount the 64 duplicate tokens ($w_c = \max(N_c - 64, 0.0)$), ensuring repeated boundary tokens are not double-counted when pooling chunk vectors. Aggregated vectors are normalized to unit length.
- **Persistent `.npy` Caching:** Vectors are cached as individual `{cache_key}.npy` files with an accompanying `embedding_spec.json` manifest. Every cached vector is checked for strict 768-dimensional shape, finiteness, and non-zero magnitude; incompatible or corrupted files are rejected.
- **Leakage Prevention & Self-Masking:** The candidate pool strictly admits labeled training examples. Training queries are masked from retrieving themselves, while identical dictation texts from separate cases remain eligible.
- **Hybrid Linear Ensemble:** Ranks candidates by:
  $$\text{Score} = 0.60 \cdot \text{DenseSim} + 0.20 \cdot \text{TF-IDF} + 0.10 \cdot \text{ModalityMatch} + 0.10 \cdot \text{BodyPartMatch}$$

### ChatML Formatting & Target-Only Completion Masking

- **Qwen ChatML Standard:** Formats multi-turn chat examples using `<|im_start|>role\ncontent<|im_end|>\n` turns, with generation prompt `<|im_start|>assistant\n`.
- **Demonstration Budgeting:** Lowest-ranked demonstrations are pruned whole from the end if the example exceeds sequence limits (`max_seq_length = 6144`), preserving complete clinical reports without truncation.
- **Loss Masking Invariant:** Loss is applied strictly to assistant target completion tokens (`completion_mask = 1`, `labels = input_ids`). System instructions, demonstration turns, current case inputs, assistant generation headers, and padding tokens are masked (`completion_mask = 0`, `labels = -100`).
- **Offline Validation:** Public unit tests execute offline without downloading model weights using mock ChatML tokenizers.

### 4-bit NF4 QLoRA Fine-Tuning & RES Evaluation

- **Model Specification:** `Qwen/Qwen2.5-3B-Instruct` loaded in 4-bit NormalFloat4 (NF4) with double quantization.
- **Adapter Parameters:** PEFT LoRA with rank $r=16$, $\alpha=32$, dropout $0.05$, adapting all linear projection layers.
- **SFT Training:** TRL `SFTTrainer` configured for completion-only loss, paged AdamW 8-bit optimizer, cosine learning rate schedule (`1e-4`), gradient accumulation steps $8$, and checkpointing.
- **Inference & Evaluation:** Greedy continuation generation (`do_sample=False`, `max_new_tokens=2048`) scored via the canonical clinically weighted RES metric in [`evaluation/res.py`](evaluation/res.py).
- **Interactive Notebook:** The full runnable pipeline is demonstrated in [`hybrid_retrieval_qlora/notebook.ipynb`](hybrid_retrieval_qlora/notebook.ipynb).

### Running Retrieval & Cache Management Locally

Build or warm the `.npy` vector cache from local CSV data:

```bash
python -m hybrid_retrieval_qlora.cli build-cache \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/
```

Inspect and validate cached `.npy` vectors and manifest compatibility:

```bash
python -m hybrid_retrieval_qlora.cli inspect-cache \
  --cache-dir artifacts/pubmedbert_cache/
```

Run hybrid retrieval to inspect demonstration selection for evaluation cases:

```bash
python -m hybrid_retrieval_qlora.cli retrieve \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/ \
  --top-k 5 \
  --num-queries 3
```
