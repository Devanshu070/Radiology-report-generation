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
