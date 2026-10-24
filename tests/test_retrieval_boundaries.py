"""Public offline unit tests for chunk pooling and hybrid retrieval leakage boundaries."""

from __future__ import annotations

import unittest

import numpy as np

from hybrid_retrieval_qlora.config import EXPECTED_EMBEDDING_DIM
from hybrid_retrieval_qlora.retrieval.chunk_pooling import (
    aggregate_chunk_vectors,
    compute_chunk_weights,
    validate_embedding_matrix,
)
from hybrid_retrieval_qlora.retrieval.hybrid_retriever import HybridRetriever
from hybrid_retrieval_qlora.schemas import ClinicalCase, SplitManifest


def make_case(
    case_id: str,
    *,
    modality: str = "CT",
    body_part: str = "Chest",
    dictation: str = "No acute cardiopulmonary disease.",
    report: str | None = "FINDINGS: Normal. IMPRESSION: Clear.",
) -> ClinicalCase:
    return ClinicalCase(
        case_id=case_id,
        modality=modality,
        body_part=body_part,
        study_description="Routine scan",
        patient_age_band="60-70",
        patient_sex="F",
        template_content="FINDINGS:\nIMPRESSION:",
        dictation=dictation,
        report=report,
    )


class ChunkPoolingMathTests(unittest.TestCase):
    """Test chunk token weighting, overlap subtraction, and aggregation math."""

    def test_single_chunk_weights(self):
        # 1 chunk, 100 active tokens, 2 special tokens ([CLS], [SEP])
        masks = [[1] * 100]
        owners = [0]
        weights = compute_chunk_weights(masks, owners, num_special_tokens=2, chunk_overlap=64)
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0], 98.0)

    def test_multi_chunk_overlap_discount(self):
        # Sample 0 has 2 chunks: chunk 0 (512 tokens), chunk 1 (200 tokens)
        # Sample 1 has 1 chunk: chunk 2 (150 tokens)
        masks = [
            [1] * 512,
            [1] * 200,
            [1] * 150,
        ]
        owners = [0, 0, 1]
        weights = compute_chunk_weights(masks, owners, num_special_tokens=2, chunk_overlap=64)

        self.assertEqual(len(weights), 3)
        # Chunk 0: 512 - 2 = 510
        self.assertEqual(weights[0], 510.0)
        # Chunk 1: (200 - 2) - 64 = 134 (discounted for overlap with chunk 0 of same sample)
        self.assertEqual(weights[1], 134.0)
        # Chunk 2: 150 - 2 = 148 (first chunk of sample 1, NO discount)
        self.assertEqual(weights[2], 148.0)

    def test_aggregation_and_normalization(self):
        # 2 samples, 3 chunks total (Sample 0 has 2 chunks, Sample 1 has 1 chunk)
        rng = np.random.default_rng(42)
        chunk_vecs = rng.standard_normal((3, 768)).astype(np.float32)
        weights = np.array([510.0, 134.0, 148.0], dtype=np.float32)
        owners = np.array([0, 0, 1], dtype=np.int64)

        pooled = aggregate_chunk_vectors(
            chunk_vecs,
            weights,
            owners,
            num_samples=2,
            expected_dim=768,
        )

        self.assertEqual(pooled.shape, (2, 768))
        self.assertEqual(pooled.dtype, np.float32)

        # Both output vectors must have unit norm
        norms = np.linalg.norm(pooled, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

        # Expected sample 0 vector direction: (w0 * v0_unit + w1 * v1_unit)
        v0_unit = chunk_vecs[0] / np.linalg.norm(chunk_vecs[0])
        v1_unit = chunk_vecs[1] / np.linalg.norm(chunk_vecs[1])
        expected_s0 = weights[0] * v0_unit + weights[1] * v1_unit
        expected_s0 /= np.linalg.norm(expected_s0)
        self.assertTrue(np.allclose(pooled[0], expected_s0, atol=1e-5))


class RetrievalBoundaryAndLeakageTests(unittest.TestCase):
    """Test retrieval candidate boundaries, self-retrieval prevention, and hybrid scoring."""

    def setUp(self):
        self.c1 = make_case("case-01", modality="CT", body_part="Chest", dictation="Pneumonia in right lower lobe.")
        self.c2 = make_case("case-02", modality="CT", body_part="Chest", dictation="Bilateral pleural effusion.")
        self.c3 = make_case("case-03", modality="MRI", body_part="Brain", dictation="Acute infarction left MCA.")
        self.val_case = make_case("val-01", modality="CT", body_part="Chest", dictation="Pneumonia.")
        self.test_case = make_case("test-01", modality="CT", body_part="Chest", dictation="Pneumonia.")

        self.manifest = SplitManifest(
            seed=42,
            assignments={
                "case-01": "train",
                "case-02": "train",
                "case-03": "train",
                "val-01": "validation",
                "test-01": "test",
            },
        )

    def test_prevents_val_or_test_leakage_into_candidates(self):
        # Adding a validation case to candidate pool must raise ValueError
        leaked_pool = [self.c1, self.c2, self.val_case]
        with self.assertRaises(ValueError) as ctx:
            HybridRetriever(leaked_pool, manifest=self.manifest)
        self.assertIn("Data leakage detected", str(ctx.exception))

    def test_prevents_unlabeled_cases_in_candidate_pool(self):
        unlabeled = make_case("case-unlabeled", report=None)
        manifest = SplitManifest(
            seed=42,
            assignments={"case-unlabeled": "train"},
        )
        with self.assertRaises(ValueError) as ctx:
            HybridRetriever([unlabeled], manifest=manifest)
        self.assertIn("lacks a labeled reference report", str(ctx.exception))

    def test_requires_split_manifest_and_rejects_missing(self):
        with self.assertRaises(TypeError):
            HybridRetriever([self.c1])  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            HybridRetriever([self.c1], manifest=None)  # type: ignore[arg-type]

    def test_prevents_self_retrieval_for_training_queries(self):
        train_pool = [self.c1, self.c2, self.c3]
        retriever = HybridRetriever(train_pool, manifest=self.manifest, top_k=2)

        # Construct synthetic unit vectors
        rng = np.random.default_rng(10)
        vecs = rng.standard_normal((3, 768)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        # Querying with case-01 (a training case)
        results = retriever.retrieve(
            [self.c1],
            query_embeddings=vecs[:1],
            candidate_embeddings=vecs,
            top_k=2,
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        # case-01 must NOT appear in its own demonstration candidates!
        retrieved_ids = [cand.candidate_case_id for cand in res.candidates]
        self.assertNotIn("case-01", retrieved_ids)
        self.assertIn("case-02", retrieved_ids)

    def test_allows_identical_dictation_from_different_case(self):
        # If two different training cases have identical dictation, candidate case is permitted
        twin_case = make_case("case-twin", modality="CT", body_part="Chest", dictation=self.c1.dictation)
        manifest = SplitManifest(
            seed=42,
            assignments={
                "case-01": "train",
                "case-02": "train",
                "case-twin": "train",
            },
        )
        pool = [self.c1, self.c2, twin_case]
        retriever = HybridRetriever(pool, manifest=manifest, top_k=2)

        rng = np.random.default_rng(10)
        vecs = rng.standard_normal((3, 768)).astype(np.float32)
        # Twin case shares identical vector with case-01
        vecs[2] = vecs[0]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        results = retriever.retrieve(
            [self.c1],
            query_embeddings=vecs[:1],
            candidate_embeddings=vecs,
            top_k=2,
        )

        retrieved_ids = [cand.candidate_case_id for cand in results[0].candidates]
        self.assertNotIn("case-01", retrieved_ids)  # Self is blocked
        self.assertIn("case-twin", retrieved_ids)  # Twin with distinct ID is permitted

    def test_hybrid_scoring_components(self):
        pool = [self.c1, self.c2, self.c3]
        retriever = HybridRetriever(
            pool,
            manifest=self.manifest,
            weights={
                "embedding_similarity": 0.60,
                "keyword_similarity": 0.20,
                "modality_match": 0.10,
                "body_part_match": 0.10,
            },
            top_k=3,
        )

        # Orthogonal synthetic vectors for predictable dot products
        candidate_vecs = np.zeros((3, 768), dtype=np.float32)
        candidate_vecs[0, 0] = 1.0  # case-01
        candidate_vecs[1, 1] = 1.0  # case-02
        candidate_vecs[2, 2] = 1.0  # case-03

        # Query aligns with case-01
        query_vec = np.zeros((1, 768), dtype=np.float32)
        query_vec[0, 0] = 1.0  # exact dense match with case-01

        query = make_case(
            "test-query",
            modality="CT",
            body_part="Chest",
            dictation="Pneumonia in right lower lobe.",
        )

        results = retriever.retrieve(
            [query],
            query_embeddings=query_vec,
            candidate_embeddings=candidate_vecs,
            top_k=3,
        )

        candidates = results[0].candidates
        # case-01 should be rank 1 (dense=1.0, tfidf=1.0, modality=1.0, body_part=1.0 -> score=1.0)
        self.assertEqual(candidates[0].candidate_case_id, "case-01")
        self.assertAlmostEqual(candidates[0].embedding_similarity, 1.0, places=4)
        self.assertAlmostEqual(candidates[0].keyword_similarity, 1.0, places=4)
        self.assertEqual(candidates[0].modality_match, 1.0)
        self.assertEqual(candidates[0].body_part_match, 1.0)
        self.assertAlmostEqual(candidates[0].score, 1.0, places=4)

        # case-03 has different anatomy (Brain) and modality (MRI), and orthogonal embedding
        c3_cand = next(c for c in candidates if c.candidate_case_id == "case-03")
        self.assertEqual(c3_cand.modality_match, 0.0)
        self.assertEqual(c3_cand.body_part_match, 0.0)


if __name__ == "__main__":
    unittest.main()
