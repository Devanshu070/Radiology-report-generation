# Hybrid Biomedical Retrieval & Embedding Cache

This package implements the dense semantic retrieval, sliding-window chunk pooling, persistent vector caching, and hybrid ranking engine for **Approach 2 (Retrieval-Assisted QLoRA Fine-Tuning)**.

> [!NOTE]
> This branch implements **only** the biomedical retrieval and embedding cache foundation. QLoRA adapter training, completion-loss masking, and SFT tokenization are isolated on the subsequent training branch (`hybrid-retrieval-qlora/qlora-sft-training`).

---

## Architecture & Mathematical Pooling Formulation

### PubMedBERT Semantic Embeddings
Dense retrieval uses `NeuML/pubmedbert-base-embeddings` (pinned revision `b79526d6ef3645e0df4530322e266f24c829f5ef`), producing 768-dimensional normalized biomedical vectors.

### Sliding-Window Chunk Pooling
Clinical radiology dictations regularly exceed PubMedBERT's 512-token limit. Rather than truncating diagnostic text:
1. **Sliding Window:** Long text is tokenized into chunks of length $L = 512$ with stride $S = 448$, producing an overlap of $O = 64$ tokens between consecutive chunks of the same sample.
2. **Overlap Discount Weighting:** Each chunk $c$ has content token count $N_c = \sum m_{c, j} - 2$ (excluding `[CLS]` and `[SEP]`).
   - For chunk 0: $w_0 = N_0$
   - For chunk $c > 0$ (same case): $w_c = \max(N_c - 64, 0.0)$ to avoid double-counting tokens shared across the window boundary.
3. **Unit Normalization per Chunk:** Each chunk embedding $\mathbf{v}_c \in \mathbb{R}^{768}$ is normalized: $\hat{\mathbf{v}}_c = \frac{\mathbf{v}_c}{\|\mathbf{v}_c\|_2}$.
4. **Weighted Case Aggregation:** $\mathbf{u}_i = \sum_{c \in \text{chunks}(i)} w_c \hat{\mathbf{v}}_c$.
5. **Final Case Normalization:** $\mathbf{e}_i = \frac{\mathbf{u}_i}{\max(\|\mathbf{u}_i\|_2, 10^{-12})}$.

### Persistent Content-Addressed `.npy` Cache
- Individual vectors are saved atomically as `{cache_key}.npy` files in `artifacts/pubmedbert_cache/`.
- Cache key is the SHA-256 digest of `{model, revision, chunk_size, overlap, pooling, dimension, text}`.
- An accompanying `embedding_spec.json` records directory configuration and verifies compatibility before any read/write.
- Every vector file is validated upon loading: shape must be exactly `(768,)`, values must be finite, and norm must be non-zero.
- Incompatible or corrupted vector files are rejected.

### Hybrid Retrieval & Leakage Prevention
- **Candidate Pool Isolation:** Candidates are strictly filtered from labeled training cases (`manifest.train_ids`). Validation and test cases can never enter the demonstration pool.
- **Self-Retrieval Prevention:** When running retrieval for training queries, self-matches (`query_case_id == candidate_case_id`) are masked with $-\infty$. Distinct cases with identical dictations remain eligible.
- **Ensemble Scoring:**
  $$\text{Score} = 0.60 \cdot \text{DenseSim} + 0.20 \cdot \text{TF-IDF} + 0.10 \cdot \text{ModalityMatch} + 0.10 \cdot \text{BodyPartMatch}$$

---

## Local Usage & Commands

### 1. Build or Prepopulate Cache
Extract dictations from a CSV and precompute `.npy` vector caches:

```bash
python -m hybrid_retrieval_qlora.cli build-cache \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/
```

### 2. Inspect and Validate Cache Integrity
Verify that cached `.npy` vectors are intact, have dimension 768, and match the active specification:

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

---

## Testing & Offline Injectability
All unit tests run offline without network access or downloading model weights:

```bash
python -m unittest discover tests
```

To run the optional live integration test with real PubMedBERT weight execution:

```bash
RUN_LOCAL_INTEGRATION=1 python -m unittest tests/test_pubmedbert_cache.py
```
