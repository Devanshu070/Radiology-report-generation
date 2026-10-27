"""Pydantic data models for Approach 2 retrieval and split integrity."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClinicalCase(BaseModel):
    """Clinical case record representation for hybrid retrieval.

    ``report`` is required for training demonstrations and target supervision,
    but may be omitted for unannotated inference inputs.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    body_part: str = Field(min_length=1)
    study_description: str = ""
    patient_age_band: str = ""
    patient_sex: str = ""
    template_content: str = Field(min_length=1)
    dictation: str = Field(min_length=1)
    report: str | None = None


class RetrievalCandidate(BaseModel):
    """Scored training candidate retrieved for a specific query case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query_case_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    candidate_case_id: str = Field(min_length=1)
    score: float
    embedding_similarity: float
    keyword_similarity: float
    modality_match: float
    body_part_match: float


class RetrievalResult(BaseModel):
    """Container for retrieval candidates matching a query case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query_case_id: str = Field(min_length=1)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)


class SplitManifest(BaseModel):
    """Partition assignment mapping each case to train, validation, or test.

    Ensures zero data leakage by enforcing candidate demonstration pools
    exclusively from labeled training partitions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    seed: int
    assignments: dict[str, Literal["train", "validation", "test"]]

    @property
    def train_ids(self) -> set[str]:
        """Set of case IDs designated as training examples."""
        return {case_id for case_id, split in self.assignments.items() if split == "train"}

    @property
    def validation_ids(self) -> set[str]:
        """Set of case IDs designated as validation examples."""
        return {case_id for case_id, split in self.assignments.items() if split == "validation"}

    @property
    def test_ids(self) -> set[str]:
        """Set of case IDs designated as test examples."""
        return {case_id for case_id, split in self.assignments.items() if split == "test"}


class ChatMessage(BaseModel):
    """A single turn in a multi-turn chat conversation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class FormattedSFTExample(BaseModel):
    """An SFT training or evaluation example structured into prompt and target completion."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    prompt_messages: list[ChatMessage] = Field(min_length=1)
    completion_message: ChatMessage | None = None
    demonstration_ids: list[str] = Field(default_factory=list)

    @property
    def all_messages(self) -> list[ChatMessage]:
        if self.completion_message is not None:
            return [*self.prompt_messages, self.completion_message]
        return list(self.prompt_messages)


class TokenizedSFTRecord(BaseModel):
    """Aligned token arrays for supervised fine-tuning with masked loss."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    input_ids: list[int]
    attention_mask: list[int]
    completion_mask: list[int]
    labels: list[int]


class GenerationResult(BaseModel):
    """Output from adapter-based or baseline report generation."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    generated_report: str = ""
    status: Literal["success", "failed"] = "success"
    error: str = ""
    generated_tokens: int = 0


class EvaluationScore(BaseModel):
    """Case-level scoring results computed with the canonical RES metric."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    status: str = "success"
    generated_report: str = ""
    expected_report: str = ""
    res_proxy: float = Field(ge=0.0, le=1.0)
    findings_proxy: float = Field(ge=0.0, le=1.0)
    impression_proxy: float = Field(ge=0.0, le=1.0)
    error: str = ""
    scoring_error: str = ""


class EvaluationSummary(BaseModel):
    """Aggregate evaluation metrics across a dataset."""

    model_config = ConfigDict(extra="forbid")

    variant: str
    total_cases: int
    successful_cases: int
    mean_res_proxy: float
    mean_findings_proxy: float
    mean_impression_proxy: float
