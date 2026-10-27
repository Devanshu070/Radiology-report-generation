# Hybrid Biomedical Retrieval + QLoRA Fine-Tuning

This package implements the end-to-end pipeline for **Approach 2 (Retrieval-Assisted QLoRA Fine-Tuning)**: dense semantic retrieval, sliding-window chunk pooling, persistent vector caching, ChatML formatting, target-only completion masking, 4-bit NF4 QLoRA fine-tuning of `Qwen/Qwen2.5-3B-Instruct`, adapter inference, and clinically weighted RES evaluation.

---

## Architecture & Data Contracts

### 1. PubMedBERT Semantic Embeddings
Dense retrieval uses `NeuML/pubmedbert-base-embeddings` (pinned revision `b79526d6ef3645e0df4530322e266f24c829f5ef`), producing 768-dimensional normalized biomedical vectors.

### 2. Sliding-Window Chunk Pooling
Clinical radiology dictations regularly exceed PubMedBERT's 512-token limit. Rather than truncating diagnostic text:
1. **Sliding Window:** Long text is tokenized into chunks of length $L = 512$ with stride $S = 448$, producing an overlap of $O = 64$ tokens between consecutive chunks of the same sample.
2. **Overlap Discount Weighting:** Each chunk $c$ has content token count $N_c = \sum m_{c, j} - 2$ (excluding `[CLS]` and `[SEP]`).
   - For chunk 0: $w_0 = N_0$
   - For chunk $c > 0$ (same case): $w_c = \max(N_c - 64, 0.0)$ to avoid double-counting tokens shared across the window boundary.
3. **Unit Normalization per Chunk:** Each chunk embedding $\mathbf{v}_c \in \mathbb{R}^{768}$ is normalized: $\hat{\mathbf{v}}_c = \frac{\mathbf{v}_c}{\|\mathbf{v}_c\|_2}$.
4. **Weighted Case Aggregation:** $\mathbf{u}_i = \sum_{c \in \text{chunks}(i)} w_c \hat{\mathbf{v}}_c$.
5. **Final Case Normalization:** $\mathbf{e}_i = \frac{\mathbf{u}_i}{\max(\|\mathbf{u}_i\|_2, 10^{-12})}$.

### 3. Persistent Content-Addressed `.npy` Cache
- Individual vectors are saved atomically as `{cache_key}.npy` files in `artifacts/pubmedbert_cache/`.
- Cache key is the SHA-256 digest of `{model, revision, chunk_size, overlap, pooling, dimension, text}`.
- An accompanying `embedding_spec.json` records directory configuration and verifies compatibility before any read/write.
- Every vector file is validated upon loading: shape must be exactly `(768,)`, values must be finite, and norm must be non-zero.
- Incompatible or corrupted vector files are rejected.

### 4. Hybrid Retrieval & Leakage Prevention
- **Candidate Pool Isolation:** Candidates are strictly filtered from labeled training cases (`manifest.train_ids`). Validation and test cases can never enter the demonstration pool.
- **Self-Retrieval Prevention:** When running retrieval for training queries, self-matches (`query_case_id == candidate_case_id`) are masked with $-\infty$. Distinct cases with identical dictations remain eligible.
- **Ensemble Scoring:**
  $$\text{Score} = 0.60 \cdot \text{DenseSim} + 0.20 \cdot \text{TF-IDF} + 0.10 \cdot \text{ModalityMatch} + 0.10 \cdot \text{BodyPartMatch}$$

### 5. ChatML SFT Formatting & Whole-Demonstration Budgeting
- **Model Standard:** Aligned with `Qwen/Qwen2.5-3B-Instruct` ChatML convention (`<|im_start|>role\ncontent<|im_end|>\n`).
- **Demonstration Budgeting:** Formats multi-turn chat examples (system prompt $\to$ retrieved demonstration turns $\to$ current case input $\to$ reference target report).
- **No Text Truncation:** If a sequence exceeds the configured sequence length (`max_seq_length = 6144` or `max_input_tokens = 5120`), lowest-ranked demonstration pairs are pruned whole from the end. No clinical findings or reports are truncated.

### 6. Target-Only Completion Masking
- **Boundary Precision:** Only the assistant target completion tokens (and trailing `<|im_end|>`) receive loss (`completion_mask = 1`, `labels = input_ids`).
- **Prompt Masking:** System instructions, retrieved demonstration turns, current case inputs, and the assistant generation prompt header (`<|im_start|>assistant\n`) are masked from loss (`completion_mask = 0`, `labels = -100`).
- **Padding Invariant:** Right-padding tokens have `attention_mask = 0`, `completion_mask = 0`, and `labels = -100`.
- **Length Alignment:** `input_ids`, `attention_mask`, `completion_mask`, and `labels` arrays strictly maintain equal length.

### 7. 4-bit NF4 QLoRA Fine-Tuning
- **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (pinned revision `main`).
- **Quantization:** 4-bit NormalFloat4 (NF4) with double quantization and bfloat16/float16 compute dtype via `BitsAndBytesConfig`.
- **PEFT LoRA:** Rank $r=16$, $\alpha=32$, dropout $0.05$, targeting `all-linear` projection modules (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
- **SFT Training:** TRL `SFTTrainer` with `completion_only_loss=True`, paged AdamW 8-bit optimizer, cosine learning rate scheduler (`lr = 1e-4`, warmup ratio $0.03$), gradient accumulation steps $8$, and gradient checkpointing.
- **Artifacts:** Checkpoints and run configurations (`run_config.json`, `adapter/`) persist strictly under ignored `artifacts/runs/`.

### 8. Adapter Report Generation & RES Evaluation
- **Greedy Inference:** Deterministic continuation generation (`do_sample=False`, `max_new_tokens=2048`) appending generated tokens to case context.
- **RES Evaluation:** Scored via the canonical clinically weighted `evaluation/res.py` metric:
  $$\text{RES}_{\text{proxy}} = 0.65 \cdot \text{Findings}_{\text{proxy}} + 0.35 \cdot \text{Impression}_{\text{proxy}}$$

---

## Local Usage & Commands

### 1. Build or Warm Embedding Cache
Precompute `.npy` vector caches for clinical dictations:

```bash
python -m hybrid_retrieval_qlora.cli build-cache \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/
```

### 2. Inspect Cache Integrity
Verify cached vector shapes, finiteness, and parameter compatibility:

```bash
python -m hybrid_retrieval_qlora.cli inspect-cache \
  --cache-dir artifacts/pubmedbert_cache/
```

### 3. Run Hybrid Retrieval Inspection
Retrieve demonstrations for query cases without data leakage:

```bash
python -m hybrid_retrieval_qlora.cli retrieve \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/ \
  --top-k 5 \
  --num-queries 3
```

### 4. Interactive Pipeline Notebook
Run the end-to-end interactive workflow in Jupyter:

```bash
jupyter notebook hybrid_retrieval_qlora/notebook.ipynb
```

---

## Hardware Requirements & Offline Testing

- **Fine-Tuning:** Requires an NVIDIA GPU supporting CUDA and bfloat16/float16 (e.g. A100, RTX 3090/4090).
- **Training Dependencies:** Install the optional QLoRA stack with `python -m pip install -r requirements-finetune.txt` from the repository root. The default `requirements.txt` remains suitable for retrieval and public unit tests without installing GPU training packages.
- **Public CI & Unit Testing:** All unit tests in `tests/test_qlora_training.py` execute offline with synthetic fixtures and zero model weight downloads using `MockQwenTokenizer` and injectable mock model architectures:

```bash
python -m unittest discover tests
```
