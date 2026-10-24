"""Public offline unit tests for PubMedBERT vector caching and compatibility validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_retrieval_qlora.cli import main as cli_main
from hybrid_retrieval_qlora.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EXPECTED_EMBEDDING_DIM,
)
from hybrid_retrieval_qlora.retrieval.cache_manifest import (
    EmbeddingCacheManager,
    EmbeddingSpec,
    IncompatibleCacheError,
    InvalidCacheVectorError,
    compute_cache_key,
    validate_cached_vector,
)
from hybrid_retrieval_qlora.retrieval.chunk_pooling import validate_embedding_matrix
from hybrid_retrieval_qlora.retrieval.pubmedbert_embeddings import PubMedBERTEmbedder


def make_synthetic_vector(dim: int = EXPECTED_EMBEDDING_DIM, seed: int = 42) -> np.ndarray:
    """Generate a reproducible normalized synthetic embedding vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


class PubMedBERTCacheKeyTests(unittest.TestCase):
    """Test deterministic content-addressed cache key generation."""

    def test_cache_key_is_deterministic(self):
        spec = EmbeddingSpec()
        text = "Small right apical pneumothorax without tension."
        key1 = compute_cache_key(text, spec)
        key2 = compute_cache_key(text, spec)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)

    def test_altering_text_or_spec_changes_key(self):
        spec1 = EmbeddingSpec()
        spec2 = EmbeddingSpec(chunk_size=256)
        spec3 = EmbeddingSpec(model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        text1 = "Case A"
        text2 = "Case B"

        self.assertNotEqual(compute_cache_key(text1, spec1), compute_cache_key(text2, spec1))
        self.assertNotEqual(compute_cache_key(text1, spec1), compute_cache_key(text1, spec2))
        self.assertNotEqual(compute_cache_key(text1, spec1), compute_cache_key(text1, spec3))


class CachedVectorValidationTests(unittest.TestCase):
    """Test strict .npy dimension and numerical validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_768_dim_vector_passes(self):
        valid_vec = make_synthetic_vector(768)
        vec_path = self.dir_path / "valid.npy"
        np.save(vec_path, valid_vec)

        loaded = validate_cached_vector(vec_path, expected_dim=768)
        self.assertEqual(loaded.shape, (768,))
        self.assertEqual(loaded.dtype, np.float32)
        self.assertTrue(np.allclose(loaded, valid_vec))

    def test_incompatible_dimension_is_rejected(self):
        invalid_dim_vec = make_synthetic_vector(384)
        vec_path = self.dir_path / "invalid_dim.npy"
        np.save(vec_path, invalid_dim_vec)

        with self.assertRaises(InvalidCacheVectorError) as ctx:
            validate_cached_vector(vec_path, expected_dim=768)
        self.assertIn("incompatible dimension", str(ctx.exception))

    def test_nan_or_inf_vector_is_rejected(self):
        vec = make_synthetic_vector(768)
        vec[10] = np.nan
        vec_path = self.dir_path / "nan.npy"
        np.save(vec_path, vec)

        with self.assertRaises(InvalidCacheVectorError) as ctx:
            validate_cached_vector(vec_path, expected_dim=768)
        self.assertIn("non-finite", str(ctx.exception))

    def test_zero_norm_vector_is_rejected(self):
        zero_vec = np.zeros(768, dtype=np.float32)
        vec_path = self.dir_path / "zero.npy"
        np.save(vec_path, zero_vec)

        with self.assertRaises(InvalidCacheVectorError) as ctx:
            validate_cached_vector(vec_path, expected_dim=768)
        self.assertIn("zero norm", str(ctx.exception))


class CacheManagerCompatibilityTests(unittest.TestCase):
    """Test cache manager directory spec tracking and reuse."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_spec_file_on_init(self):
        spec = EmbeddingSpec(chunk_size=512, overlap=64)
        EmbeddingCacheManager(self.cache_dir, spec=spec)

        spec_file = self.cache_dir / "embedding_spec.json"
        self.assertTrue(spec_file.is_file())
        data = json.loads(spec_file.read_text())
        self.assertEqual(data["dimension"], 768)
        self.assertEqual(data["chunk_size"], 512)

    def test_detects_incompatible_cache_spec(self):
        spec1 = EmbeddingSpec(chunk_size=512)
        EmbeddingCacheManager(self.cache_dir, spec=spec1)

        spec2 = EmbeddingSpec(chunk_size=256)
        with self.assertRaises(IncompatibleCacheError):
            EmbeddingCacheManager(self.cache_dir, spec=spec2)

    def test_reusable_cache_generation_with_mock_embedder(self):
        mgr = EmbeddingCacheManager(self.cache_dir)
        calls: list[list[str]] = []

        def mock_encoder(batch: list[str]) -> np.ndarray:
            calls.append(list(batch))
            rng = np.random.default_rng(len(batch))
            vecs = rng.standard_normal((len(batch), 768)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        texts = ["Text A", "Text B", "Text C"]

        # First call: computes all 3 texts
        out1 = mgr.get_or_compute_embeddings(texts, mock_encoder)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["Text A", "Text B", "Text C"])
        self.assertEqual(out1.shape, (3, 768))

        # Second call with identical texts: 100% cache hits, zero embedder calls
        out2 = mgr.get_or_compute_embeddings(texts, mock_encoder)
        self.assertEqual(len(calls), 1)  # No new calls
        self.assertTrue(np.allclose(out1, out2))

        # Third call with 2 existing texts + 1 new text: only encodes the 1 new text
        out3 = mgr.get_or_compute_embeddings(["Text A", "Text D", "Text B"], mock_encoder)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ["Text D"])
        self.assertEqual(out3.shape, (3, 768))
        self.assertTrue(np.allclose(out3[0], out1[0]))
        self.assertTrue(np.allclose(out3[2], out1[1]))

    def test_recovers_vector_from_extra_cache_dirs(self):
        extra_dir = self.cache_dir / "extra"
        extra_dir.mkdir()
        mgr_extra = EmbeddingCacheManager(extra_dir)

        vec = make_synthetic_vector(768, seed=1)
        text = "Extra cached dictation text"
        mgr_extra.save_vector(text, vec)

        # Primary cache directory is clean
        primary_dir = self.cache_dir / "primary"
        mgr_primary = EmbeddingCacheManager(primary_dir, extra_cache_dirs=[extra_dir])

        found = mgr_primary.lookup_vector(text)
        self.assertIsNotNone(found)
        self.assertTrue(np.allclose(found, vec))
        # Vector should now be copied into primary directory
        self.assertTrue((primary_dir / f"{compute_cache_key(text, EmbeddingSpec())}.npy").is_file())


class EmbedderInjectionTests(unittest.TestCase):
    """Test injectable embedder behavior offline without downloading weights."""

    def test_injectable_encoder_fn(self):
        def fake_fn(texts: list[str]) -> np.ndarray:
            rng = np.random.default_rng(len(texts))
            v = rng.standard_normal((len(texts), 768)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        embedder = PubMedBERTEmbedder(encoder_fn=fake_fn)
        result = embedder.encode(["Report 1", "Report 2"])
        self.assertEqual(result.shape, (2, 768))
        self.assertEqual(result.dtype, np.float32)
        # Verify unit normalization
        norms = np.linalg.norm(result, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))

    def test_injected_encoder_with_wrong_dimension_raises(self):
        def bad_fn(texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 512), dtype=np.float32)

        embedder = PubMedBERTEmbedder(encoder_fn=bad_fn)
        with self.assertRaises(ValueError) as ctx:
            embedder.encode(["Report 1"])
        self.assertIn("Incompatible embedding dimension", str(ctx.exception))


class CLICacheInspectionTests(unittest.TestCase):
    """Test CLI inspect-cache subcommand with synthetic cache data."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspect_cache_cli(self):
        mgr = EmbeddingCacheManager(self.cache_dir)
        vec = make_synthetic_vector(768)
        mgr.save_vector("Test text", vec)

        exit_code = cli_main(["inspect-cache", "--cache-dir", str(self.cache_dir)])
        self.assertEqual(exit_code, 0)

    def test_inspect_cache_cli_detects_incompatibility(self):
        mgr = EmbeddingCacheManager(self.cache_dir, spec=EmbeddingSpec(chunk_size=512))
        vec = make_synthetic_vector(768)
        mgr.save_vector("Test text", vec)

        # Inspecting with expected chunk size 256 must fail with exit code 1
        exit_code = cli_main(["inspect-cache", "--cache-dir", str(self.cache_dir), "--chunk-size", "256"])
        self.assertEqual(exit_code, 1)


class OptionalLocalIntegrationTests(unittest.TestCase):
    """Optional end-to-end integration tests requiring local model/data availability."""

    @unittest.skipUnless(
        os.environ.get("RUN_LOCAL_INTEGRATION") == "1",
        "Set RUN_LOCAL_INTEGRATION=1 to run end-to-end PubMedBERT model download/execution.",
    )
    def test_live_pubmedbert_encoding(self):
        embedder = PubMedBERTEmbedder()
        texts = [
            "Normal chest examination. Lungs are well expanded and clear.",
            "Bilateral lower lobe consolidations suspicious for multifocal pneumonia.",
        ]
        matrix = embedder.encode(texts)
        self.assertEqual(matrix.shape, (2, 768))
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
