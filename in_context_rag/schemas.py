"""Validated records exchanged by the in-context preprocessing pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseRecord(BaseModel):
    """One input case; ``report`` is absent for unlabeled inference rows."""

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


class TemplateSections(BaseModel):
    """The two top-level sections needed to construct a report prompt."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    findings: str
    impression: str


class PromptCase(BaseModel):
    """Prompt-ready inputs, with template sections separated from metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    body_part: str = Field(min_length=1)
    study_description: str = ""
    patient_age_band: str = ""
    patient_sex: str = ""
    template_findings: str
    template_impression: str
    dictation: str = Field(min_length=1)


class FewShotPair(BaseModel):
    """A labeled demonstration pair eligible for later retrieval selection."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    modality: str = ""
    body_part: str = ""
    study_description: str = ""
    patient_age_band: str = ""
    patient_sex: str = ""
    dictation: str = Field(min_length=1)
    template_content: str = Field(min_length=1)
    report: str = Field(min_length=1)


class GeneratedSections(BaseModel):
    """Validated model output before it is assembled into a full report."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    findings: str = Field(min_length=1)
    impression: str = Field(min_length=1)


__all__ = [
    "CaseRecord",
    "FewShotPair",
    "GeneratedSections",
    "PromptCase",
    "SplitManifest",
    "TemplateSections",
]


class SplitManifest(BaseModel):
    """Deterministic, disjoint data partitions keyed by case ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    assignments: dict[str, Literal["train", "validation", "test"]]

    @model_validator(mode="after")
    def require_nonempty_partitions(self) -> "SplitManifest":
        if not self.assignments:
            raise ValueError("A split manifest must contain at least one case.")
        if set(self.assignments.values()) != {"train", "validation", "test"}:
            raise ValueError("Train, validation, and test partitions must all be nonempty.")
        return self

    @property
    def train_ids(self) -> frozenset[str]:
        return frozenset(case_id for case_id, split in self.assignments.items() if split == "train")

    @property
    def validation_ids(self) -> frozenset[str]:
        return frozenset(case_id for case_id, split in self.assignments.items() if split == "validation")

    @property
    def test_ids(self) -> frozenset[str]:
        return frozenset(case_id for case_id, split in self.assignments.items() if split == "test")
