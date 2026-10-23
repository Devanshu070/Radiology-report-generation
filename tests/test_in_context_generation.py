"""Synthetic tests for Approach 1 retrieval, generation, caching, and review."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from in_context_rag.generation.groq_client import GroqGenerator, ProviderError
from in_context_rag.generation.request_builder import build_generation_request
from in_context_rag.generation.response_cache import ResponseCache, request_fingerprint
from in_context_rag.generation.retries import retry_async
from in_context_rag.pipeline import assemble_report, evaluate_results, generate_reports, retrieve_examples
from in_context_rag.preprocess.split import create_split_manifest
from in_context_rag.preprocess.template_sections import to_prompt_case
from in_context_rag.retrieval.cache import (
    embedding_cache_path,
    embedding_fingerprint,
    load_embedding_cache,
    save_embedding_cache,
)
from in_context_rag.retrieval.few_shot_selection import select_few_shot_examples
from in_context_rag.review.html_reviewer import build_review_html
from in_context_rag.schemas import CaseRecord, FewShotPair, GeneratedSections


def make_case(index: int, *, content: str | None = None) -> CaseRecord:
    dictation = content or f"Small right pleural effusion case {index}."
    report = (
        "FINDINGS:\nLUNGS: Lungs are clear.\nPLEURA: Small right pleural effusion."
        "\n\nIMPRESSION:\n1. Small right pleural effusion."
    )
    return CaseRecord(
        case_id=f"case-{index:02d}",
        modality="X-Ray",
        body_part="Chest",
        study_description="Portable chest radiograph",
        patient_age_band="Adult",
        patient_sex="Unspecified",
        template_content=(
            "FINDINGS:\nLUNGS: Lungs are clear.\nPLEURA: No pleural effusion."
            "\n\nIMPRESSION:\nNo acute cardiopulmonary finding."
        ),
        dictation=dictation,
        report=report,
    )


def split_cases(count: int = 30):
    cases = [make_case(index) for index in range(count)]
    return cases, create_split_manifest(cases, seed=11)


class RateLimitError(Exception):
    status_code = 429
    response = SimpleNamespace(headers={"retry-after": "90"})


class BadRequestError(Exception):
    status_code = 400


class JSONValidationError(Exception):
    status_code = 400
    body = {"error": {"code": "json_validate_failed", "type": "invalid_request_error"}}


class JSONValidationError(Exception):
    status_code = 400
    body = {"error": {"code": "json_validate_failed", "type": "invalid_request_error"}}


class PayloadTooLargeProviderError(ProviderError):
    def __init__(self):
        super().__init__("request too large", status_code=413)


class RetrievalTests(unittest.TestCase):
    def test_few_shot_selector_returns_at_most_five_training_examples(self):
        cases, manifest = split_cases()
        candidates = [case for case in cases if case.case_id in manifest.train_ids]
        queries = [case for case in cases if case.case_id in manifest.test_ids]
        training_vectors = np.ones((len(candidates), 4), dtype=np.float32)
        query_vectors = np.ones((len(queries), 4), dtype=np.float32)
        selected = select_few_shot_examples(
            cases,
            queries,
            manifest,
            top_k=5,
            embedding_matrices=(training_vectors, query_vectors),
        )
        self.assertEqual(set(selected), {query.case_id for query in queries})
        for query_id, examples in selected.items():
            self.assertLessEqual(len(examples), 5)
            self.assertNotIn(query_id, {example.case_id for example in examples})
            self.assertTrue({example.case_id for example in examples}.issubset(manifest.train_ids))

    def test_tied_scores_have_stable_case_id_order(self):
        cases = [make_case(index, content="shared dictation") for index in range(30)]
        manifest = create_split_manifest(cases, seed=11)
        candidates = [case for case in cases if case.case_id in manifest.train_ids]
        queries = [case for case in cases if case.case_id in manifest.test_ids]
        vectors = np.ones((len(candidates), 2), dtype=np.float32)
        selected = select_few_shot_examples(
            cases, queries[:1], manifest, top_k=5,
            embedding_matrices=(vectors, np.ones((1, 2), dtype=np.float32)),
        )[queries[0].case_id]
        ids = [example.case_id for example in selected]
        self.assertEqual(ids, sorted(ids))

    def test_selector_rejects_invalid_k_and_incompatible_vectors(self):
        cases, manifest = split_cases()
        query = next(case for case in cases if case.case_id in manifest.test_ids)
        candidates = [case for case in cases if case.case_id in manifest.train_ids]
        with self.assertRaisesRegex(ValueError, "top_k"):
            select_few_shot_examples(cases, [query], manifest, top_k=6)
        with self.assertRaisesRegex(ValueError, "Embedding matrices"):
            select_few_shot_examples(
                cases, [query], manifest,
                embedding_matrices=(np.ones((len(candidates), 2)), np.ones((1, 3))),
            )

    def test_embedding_cache_fingerprint_tracks_inputs_and_settings(self):
        texts = ["synthetic dictation"]
        fingerprint = embedding_fingerprint(
            texts, ["query"], model_name="test-model", model_revision="abc",
            max_seq_length=128, chunk_overlap=16,
        )
        self.assertNotEqual(fingerprint, embedding_fingerprint(
            ["changed"], ["query"], model_name="test-model", model_revision="abc",
            max_seq_length=128, chunk_overlap=16,
        ))
        self.assertNotEqual(fingerprint, embedding_fingerprint(
            texts, ["query"], model_name="test-model", model_revision="def",
            max_seq_length=128, chunk_overlap=16,
        ))
        training = np.eye(2, dtype=np.float32)
        queries = np.ones((1, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = embedding_cache_path(directory, fingerprint)
            save_embedding_cache(path, fingerprint, training, queries)
            loaded = load_embedding_cache(path, fingerprint)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded[0], training)
            np.testing.assert_array_equal(loaded[1], queries)
            self.assertIsNone(load_embedding_cache(path, "stale-fingerprint"))


class RequestAndGenerationTests(unittest.IsolatedAsyncioTestCase):
    def test_request_builder_uses_separated_template_and_schema_contract(self):
        case = to_prompt_case(make_case(1))
        demonstration = FewShotPair(
            case_id="demo",
            modality="X-Ray",
            body_part="Chest",
            dictation="Synthetic dictation.",
            template_content="FINDINGS:\nLUNGS: Clear.\nPLEURA: No effusion.\nIMPRESSION:\nNormal.",
            report="FINDINGS:\nLUNGS: Clear.\nPLEURA: Small effusion.\nIMPRESSION:\n1. Small effusion.",
        )
        payload = build_generation_request(case, [demonstration], model="test-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        prompt = json.loads(payload["messages"][1]["content"])
        self.assertEqual(prompt["CURRENT_TEMPLATE"]["findings"].splitlines()[0], "LUNGS: Lungs are clear.")
        self.assertEqual(prompt["DEMONSTRATIONS"][0]["target"]["impression"], "1. Small effusion.")
        self.assertEqual(prompt["DEMONSTRATIONS"][0]["case"]["modality"], "X-Ray")
        self.assertNotIn("FINDINGS:", prompt["CURRENT_TEMPLATE"]["findings"])

    def test_gpt_oss_request_excludes_separate_reasoning_payload(self):
        payload = build_generation_request(to_prompt_case(make_case(2)), [], model="openai/gpt-oss-120b")
        self.assertFalse(payload["include_reasoning"])
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["required"],
            ["findings", "impression"],
        )
        self.assertIs(response_format["json_schema"]["schema"]["additionalProperties"], False)

    def test_report_assembly_preserves_template_field_labels_and_order(self):
        template = to_prompt_case(make_case(1))
        sections = GeneratedSections(
            findings="LUNGS: Lungs are clear.\nPLEURA: Small right pleural effusion.",
            impression="1. Small right pleural effusion.",
        )
        report = assemble_report(template, sections)
        self.assertIn("LUNGS: Lungs are clear.", report)
        self.assertIn("PLEURA: Small right pleural effusion.", report)
        with self.assertRaisesRegex(ValueError, "field labels or order"):
            assemble_report(template, GeneratedSections(findings="PLEURA: effusion", impression="Finding."))

    async def test_retry_policy_respects_bounded_retry_after_and_does_not_retry_400(self):
        attempts = 0
        waits: list[float] = []

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RateLimitError("rate limited")
            return "ok"

        async def fake_sleep(delay: float):
            waits.append(delay)

        result = await retry_async(operation, max_attempts=3, max_delay_seconds=2, sleep=fake_sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(waits, [2, 2])

        bad_attempts = 0

        async def bad_operation():
            nonlocal bad_attempts
            bad_attempts += 1
            raise BadRequestError("invalid")

        with self.assertRaises(BadRequestError):
            await retry_async(bad_operation, max_attempts=4, sleep=fake_sleep)
        self.assertEqual(bad_attempts, 1)

        structured_attempts = 0

        async def structured_operation():
            nonlocal structured_attempts
            structured_attempts += 1
            raise JSONValidationError("invalid generated JSON")

        with self.assertRaises(JSONValidationError):
            await retry_async(structured_operation, max_attempts=3, sleep=fake_sleep)
        self.assertEqual(structured_attempts, 1)

    async def test_groq_client_repairs_specific_json_validation_failure(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"findings":"Clear lungs.","impression":"No acute finding."}')
        )])
        create = AsyncMock(side_effect=[JSONValidationError("bad JSON"), completion])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        generator = GroqGenerator(
            api_key="test-secret", client=client, max_structured_retries=1
        )
        result = await generator.generate({
            "model": "mock",
            "messages": [{"role": "user", "content": "synthetic case"}],
        })
        self.assertEqual(result.findings, "Clear lungs.")
        self.assertEqual(create.await_count, 2)
        repaired_payload = create.await_args_list[1].kwargs
        self.assertIn("failed structured-output validation", repaired_payload["messages"][-1]["content"])

    async def test_groq_client_validates_json_with_mocked_sdk(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"findings":"Lungs clear.","impression":"Normal."}')
        )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=completion))))
        generator = GroqGenerator(api_key="test-secret", client=client)
        result = await generator.generate({"model": "mock", "messages": []})
        self.assertEqual(result.findings, "Lungs clear.")
        client.chat.completions.create.assert_awaited_once()

    async def test_groq_client_redacts_key_from_provider_errors(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(side_effect=RuntimeError("request failed with test-secret"))
        )))
        generator = GroqGenerator(api_key="test-secret", client=client, max_attempts=1)
        with self.assertRaises(ProviderError) as raised:
            await generator.generate({"model": "mock", "messages": []})
        self.assertNotIn("test-secret", str(raised.exception))

    async def test_groq_client_loads_numbered_key_pool_and_rotates_per_request(self):
        class FakeClient:
            def __init__(self, api_key, **kwargs):
                self.api_key = api_key
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=(
                    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"findings":"Synthetic normal finding.","impression":"Normal."}'
                    ))])
                ))))

        created = []

        def factory(**kwargs):
            client = FakeClient(**kwargs)
            created.append(client)
            return client

        with patch("in_context_rag.generation.groq_client.load_project_environment"), patch.dict(
            "os.environ", {"GROQ_API_KEY_2": "second-test-key", "GROQ_API_KEY_1": "first-test-key"}, clear=True
        ):
            generator = GroqGenerator(client_factory=factory)
            await generator.generate({"model": "mock", "messages": []})
            await generator.generate({"model": "mock", "messages": []})

        self.assertEqual([client.api_key for client in created], ["first-test-key", "second-test-key"])
        created[0].chat.completions.create.assert_awaited_once()
        created[1].chat.completions.create.assert_awaited_once()

    async def test_generation_respects_training_only_selections_and_scores_afterward(self):
        cases, manifest = split_cases()
        queries = [case for case in cases if case.case_id in manifest.test_ids]
        train_ids = sorted(manifest.train_ids)
        class FakeGenerator:
            async def generate(self, payload):
                return GeneratedSections(
                    findings="LUNGS: Lungs are clear.\nPLEURA: No pleural effusion.",
                    impression="No acute cardiopulmonary finding.",
                )

        result = await generate_reports(
            cases,
            manifest,
            {query.case_id: [train_ids[0]] for query in queries},
            FakeGenerator(),
            query_split="test",
        )
        self.assertTrue(all(item.status == "success" for item in result))
        self.assertTrue(all(item.example_ids == [train_ids[0]] for item in result))
        evaluated = evaluate_results(cases, result)
        self.assertTrue(all("res_proxy" in item for item in evaluated))

        leaked = await generate_reports(
            cases,
            manifest,
            {query.case_id: [next(iter(manifest.validation_ids))] for query in queries},
            FakeGenerator(),
            query_split="test",
        )
        self.assertEqual(leaked[0].status, "failed")
        self.assertIn("outside the training split", leaked[0].error)

    async def test_generation_reduces_examples_on_provider_payload_limit(self):
        cases, manifest = split_cases()
        queries = [case for case in cases if case.case_id in manifest.test_ids]
        example_ids = sorted(manifest.train_ids)[:5]

        class LargePayloadOnceGenerator:
            def __init__(self):
                self.example_counts: list[int] = []

            async def generate(self, payload):
                request = json.loads(payload["messages"][1]["content"])
                self.example_counts.append(len(request["DEMONSTRATIONS"]))
                if len(self.example_counts) <= 2:
                    raise PayloadTooLargeProviderError()
                return GeneratedSections(
                    findings="LUNGS: Lungs are clear.\nPLEURA: No pleural effusion.",
                    impression="No acute cardiopulmonary finding.",
                )

        fake = LargePayloadOnceGenerator()
        results = await generate_reports(
            cases,
            manifest,
            {query.case_id: example_ids for query in queries},
            fake,
            query_split="test",
            concurrency=1,
            use_cache=False,
        )
        self.assertEqual(fake.example_counts[:3], [5, 4, 3])
        self.assertEqual(len(results[0].example_ids), 3)
        self.assertEqual(results[0].status, "success")

    async def test_successful_response_cache_is_content_addressed_and_validated(self):
        payload = {"model": "mock", "messages": [{"role": "user", "content": "synthetic"}]}
        fingerprint = request_fingerprint(payload)
        sections = GeneratedSections(findings="Normal.", impression="No acute finding.")
        with tempfile.TemporaryDirectory() as directory:
            cache = ResponseCache(directory)
            self.assertIsNone(cache.get(fingerprint))
            cache.put(fingerprint, sections)
            self.assertEqual(cache.get(fingerprint), sections)
            path = Path(directory) / f"{fingerprint}.json"
            path.write_text('{"fingerprint":"wrong","response":{"findings":"x","impression":"y"}}')
            self.assertIsNone(cache.get(fingerprint))


class ReviewTests(unittest.TestCase):
    def test_html_reviewer_escapes_case_and_model_content(self):
        case = make_case(1, content="Dictation <script>alert(1)</script>")
        result = {
            "case_id": case.case_id,
            "status": "success",
            "generated_report": "<img src=x onerror=alert(1)>",
            "res_proxy": 0.25,
            "error": "<script>bad</script>",
        }
        html = build_review_html([result], [case], title="Review <unsafe>")
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img src=x", html)

    def test_html_reviewer_rejects_unknown_case_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            build_review_html([{"case_id": "missing"}], [make_case(1)])


if __name__ == "__main__":
    unittest.main()
