# Biomedical Retrieval and QLoRA

This guide describes the workflow implemented in `hybrid_retrieval_qlora/`: biomedical semantic retrieval, persistent embedding caches, Qwen chat formatting, target-only completion masking, QLoRA fine-tuning, adapter inference, and RES-style evaluation. This workflow owns its preprocessing; it does not use the in-context workflow's template-variable preparation.

## Retrieval and embedding cache

Dense retrieval uses `NeuML/pubmedbert-base-embeddings`, pinned to revision `b79526d6ef3645e0df4530322e266f24c829f5ef`, to produce normalized 768-dimensional vectors. Text exceeding PubMedBERT's 512-token limit is split into 512-token windows with a stride of 448 (64-token overlap). For later chunks, the overlap tokens are discounted from the chunk weight. The weighted chunk vectors are summed and the case vector is normalized.

Vectors are cached as individual `.npy` files, with an `embedding_spec.json` manifest describing model, revision, chunking, pooling, and dimensions. Cache keys include the source text and embedding configuration. Loaded vectors are checked for the expected 768-value shape, finite values, and nonzero magnitude; incompatible manifests and corrupt vectors are rejected.

The hybrid rank score is:

```text
score = 0.60 × dense_similarity
      + 0.20 × TF-IDF_similarity
      + 0.10 × modality_match
      + 0.10 × body_part_match
```

Candidate demonstrations must belong to the labeled training partition. Validation and test cases cannot be candidates. A training query cannot retrieve itself, but distinct cases with identical dictations may still match.

Build and inspect the local cache:

```bash
python -m hybrid_retrieval_qlora.cli build-cache \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/

python -m hybrid_retrieval_qlora.cli inspect-cache \
  --cache-dir artifacts/pubmedbert_cache/
```

Inspect a small retrieval sample:

```bash
python -m hybrid_retrieval_qlora.cli retrieve \
  --csv data/private/train.csv \
  --cache-dir artifacts/pubmedbert_cache/ \
  --top-k 5 \
  --num-queries 3
```

## SFT formatting and completion masking

Training examples use the Qwen ChatML structure:

```text
<|im_start|>role\ncontent<|im_end|>\n
```

The input sequence contains the system instruction, retrieved demonstrations, and current case. Only the target assistant report completion contributes to loss. Prompt, demonstration, generation-header, and padding positions are masked (`labels = -100`); target positions retain their token IDs as labels. The attention mask, completion mask, labels, and input IDs must remain length-aligned.

The configured maximum sequence length is 6144 tokens. If an example exceeds the budget, the lowest-ranked demonstration pairs are removed whole, from the end; clinical target text is not truncated to fit.

## QLoRA training configuration

The configured base model is `Qwen/Qwen2.5-3B-Instruct`. Training uses 4-bit NF4 quantization with double quantization and bfloat16 or float16 compute, depending on hardware. PEFT LoRA uses rank 16, alpha 32, dropout 0.05, and all linear projection layers. The SFT setup uses completion-masked loss, paged AdamW 8-bit, cosine scheduling, a learning rate of `1e-4`, warmup ratio `0.03`, gradient accumulation of 8, and gradient checkpointing.

Install the optional training dependencies from the repository root:

```bash
python -m pip install -r requirements-finetune.txt
```

Full fine-tuning requires a compatible NVIDIA CUDA GPU and sufficient VRAM. Hardware and library compatibility should be checked before starting a training run. Checkpoints and run metadata are written under ignored `artifacts/runs/`.

## Inference and evaluation

Adapter inference uses greedy generation (`do_sample=False`) with a configured limit of 2048 new tokens. Generated reports are scored through the shared evaluator at [`../evaluation/res.py`](../evaluation/res.py). The score is a development proxy and must not be treated as clinical validation.

The reported RES score for this workflow is **0.21**. See [the evaluation guide](evaluation.md) for the local proxy definition and limitations; only compare results produced with matching evaluation setups.

The interactive notebook demonstrates the workflow:

```bash
jupyter notebook hybrid_retrieval_qlora/notebook.ipynb
```

Public tests use synthetic fixtures and mocks; they do not download model weights or require private CSVs. Run them from the repository root with:

```bash
python -m unittest discover tests
```
