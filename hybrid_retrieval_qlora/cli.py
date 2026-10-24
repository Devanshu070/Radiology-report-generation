"""Command-line interface for PubMedBERT cache generation and retrieval inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_TOP_K,
    training_data_path,
)
from .preprocess.load_data import create_deterministic_splits, filter_training_pool, load_cases
from .retrieval.cache_manifest import EmbeddingCacheManager, EmbeddingSpec, validate_cached_vector
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.pubmedbert_embeddings import PubMedBERTEmbedder


def build_cache_command(args: argparse.Namespace) -> int:
    """Compute and persist PubMedBERT embeddings for cases in a CSV file."""
    csv_path = Path(args.csv) if args.csv else training_data_path()
    if not csv_path.is_file():
        print(f"Error: Dataset not found at {csv_path}", file=sys.stderr)
        return 1

    print(f"Loading cases from {csv_path}...")
    cases = load_cases(csv_path, require_report=False)
    dictations = [c.dictation for c in cases]
    unique_dictations = list(dict.fromkeys(dictations))
    print(f"Loaded {len(cases)} cases ({len(unique_dictations)} unique dictations).")

    spec = EmbeddingSpec(
        model=args.model,
        revision=args.revision,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    cache_mgr = EmbeddingCacheManager(args.cache_dir, spec=spec)
    embedder = PubMedBERTEmbedder(
        model_name=args.model,
        revision=args.revision,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )

    print(f"Target cache directory: {args.cache_dir}")
    print("Checking cache and encoding missing entries...")
    embeddings = cache_mgr.get_or_compute_embeddings(dictations, embedder, batch_size=args.batch_size)
    print(f"Successfully cached and verified {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")
    return 0


def inspect_cache_command(args: argparse.Namespace) -> int:
    """Validate and report statistics for a cached vector directory against active config."""
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        print(f"Error: Cache directory not found: {cache_dir}", file=sys.stderr)
        return 1

    spec_file = cache_dir / "embedding_spec.json"
    if not spec_file.is_file():
        print(f"Error: No embedding_spec.json found in {cache_dir}", file=sys.stderr)
        return 1

    try:
        stored_spec = json.loads(spec_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading spec file {spec_file}: {exc}", file=sys.stderr)
        return 1

    active_spec = EmbeddingSpec(
        model=args.model,
        revision=args.revision,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    active_dict = active_spec.model_dump()

    mismatches = {
        key: (stored_spec.get(key), active_dict[key])
        for key in active_dict
        if stored_spec.get(key) != active_dict[key]
    }

    if mismatches:
        print(f"Error: Cache specification at {cache_dir} is incompatible with active configuration:", file=sys.stderr)
        for key, (stored_val, active_val) in mismatches.items():
            print(f"  - {key}: stored={stored_val!r} vs active={active_val!r}", file=sys.stderr)
        return 1

    npy_files = list(cache_dir.glob("*.npy"))
    print(f"Cache Specification: {json.dumps(stored_spec, indent=2)}")
    print(f"Cache specification matches active configuration ({args.model} @ {args.revision}).")
    print(f"Found {len(npy_files)} .npy vector files.")

    valid_count = 0
    invalid_files = []
    expected_dim = active_spec.dimension

    for f in npy_files:
        try:
            validate_cached_vector(f, expected_dim=expected_dim)
            valid_count += 1
        except Exception as exc:
            invalid_files.append((f.name, str(exc)))

    print(f"Validated vectors: {valid_count} / {len(npy_files)} valid.")
    if invalid_files:
        print(f"Found {len(invalid_files)} corrupted/incompatible vector files:", file=sys.stderr)
        for name, err in invalid_files[:5]:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    """Demonstrate hybrid retrieval on a local clinical dataset."""
    csv_path = Path(args.csv) if args.csv else training_data_path()
    if not csv_path.is_file():
        print(f"Error: Dataset not found at {csv_path}", file=sys.stderr)
        return 1

    cases = load_cases(csv_path, require_report=True)
    manifest, train_cases, val_cases, test_cases = create_deterministic_splits(cases, seed=args.seed)
    training_pool = filter_training_pool(cases, manifest)

    spec = EmbeddingSpec(
        model=args.model,
        revision=args.revision,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    cache_mgr = EmbeddingCacheManager(args.cache_dir, spec=spec)
    embedder = PubMedBERTEmbedder(
        model_name=args.model,
        revision=args.revision,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    retriever = HybridRetriever(training_pool, manifest=manifest, top_k=args.top_k)

    queries = test_cases[: args.num_queries]
    print(f"Running hybrid retrieval for {len(queries)} evaluation query case(s)...")
    results = retriever.retrieve(
        queries,
        cache_manager=cache_mgr,
        embedder=embedder,
        top_k=args.top_k,
    )

    for res in results:
        print(f"\nQuery Case ID: {res.query_case_id}")
        for cand in res.candidates:
            print(
                f"  Rank {cand.rank}: candidate={cand.candidate_case_id} "
                f"score={cand.score:.4f} (dense={cand.embedding_similarity:.4f}, "
                f"tfidf={cand.keyword_similarity:.4f}, modality={cand.modality_match:.1f}, "
                f"body_part={cand.body_part_match:.1f})"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="python -m hybrid_retrieval_qlora.cli",
        description="Approach 2 PubMedBERT Embedding Cache and Hybrid Retrieval CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-cache
    build_p = subparsers.add_parser("build-cache", help="Generate and cache PubMedBERT embeddings")
    build_p.add_argument("--csv", type=str, default=None, help="Path to input CSV file")
    build_p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    build_p.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Hugging Face model")
    build_p.add_argument("--revision", type=str, default=DEFAULT_EMBEDDING_REVISION, help="Model revision")
    build_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk token length")
    build_p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap tokens")
    build_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")

    # inspect-cache
    inspect_p = subparsers.add_parser("inspect-cache", help="Inspect and validate an embedding cache directory")
    inspect_p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    inspect_p.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Expected Hugging Face model")
    inspect_p.add_argument("--revision", type=str, default=DEFAULT_EMBEDDING_REVISION, help="Expected model revision")
    inspect_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Expected chunk token length")
    inspect_p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Expected chunk overlap tokens")

    # retrieve
    ret_p = subparsers.add_parser("retrieve", help="Run hybrid retrieval for query cases")
    ret_p.add_argument("--csv", type=str, default=None, help="Path to input CSV file")
    ret_p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    ret_p.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Hugging Face model")
    ret_p.add_argument("--revision", type=str, default=DEFAULT_EMBEDDING_REVISION, help="Model revision")
    ret_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk token length")
    ret_p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap tokens")
    ret_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Demonstration count per query")
    ret_p.add_argument("--num-queries", type=int, default=3, help="Number of query cases to run")
    ret_p.add_argument("--seed", type=int, default=42, help="Partition split seed")

    args = parser.parse_args(argv)

    if args.command == "build-cache":
        return build_cache_command(args)
    if args.command == "inspect-cache":
        return inspect_cache_command(args)
    if args.command == "retrieve":
        return retrieve_command(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
