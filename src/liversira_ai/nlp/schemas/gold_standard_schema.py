"""Human Gold Standard Schema.

Gold labels answer one question: *what does the report state?* They are built
from independent human annotation plus adjudication, never copied from the
source-derived labels. Unlike ``GroundTruthLabels``, a positive study may have
fields the report does not document; those are ``None`` ("not stated").
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ground_truth_schema import Attenuation


class GoldMethod(str, Enum):
    """How the final gold value was reached."""

    CONSENSUS = "consensus"  # two independent annotators agreed on every field
    CLINICIAN_ADJUDICATION = "clinician_adjudication"  # disagreement resolved by a clinician
    SINGLE_REVIEW = "single_review"  # one attributable reviewer (legacy workflow)


class GoldLargestLesion(BaseModel):
    """Largest liver lesion as stated in the report; ``None`` means not stated."""

    model_config = ConfigDict(extra="forbid")

    location_segments: Optional[list[int]] = Field(
        default=None, description="Couinaud segments 1-8 stated for the largest lesion"
    )
    diameter_cm: Optional[float] = Field(
        default=None, gt=0.0, description="Largest stated dimension of the largest lesion, in cm"
    )
    attenuation: Optional[Attenuation] = Field(
        default=None, description="Attenuation stated for the largest lesion"
    )

    @model_validator(mode="after")
    def validate_segments(self) -> "GoldLargestLesion":
        if self.location_segments is not None:
            if not self.location_segments:
                raise ValueError("location_segments must be None or non-empty")
            for segment in self.location_segments:
                if not 1 <= segment <= 8:
                    raise ValueError(f"Couinaud segment must be between 1 and 8, got {segment}")
            object.__setattr__(self, "location_segments", sorted(set(self.location_segments)))
        return self


class GoldLabels(BaseModel):
    """Report-level gold labels for the liver lesion extraction task."""

    model_config = ConfigDict(extra="forbid")

    has_lesion: bool
    lesion_count: Optional[int] = Field(
        default=None, ge=0, description="Stated lesion count; None if a lesion exists but no count is stated"
    )
    largest_lesion: Optional[GoldLargestLesion] = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "GoldLabels":
        if not self.has_lesion:
            if self.lesion_count not in (0, None) or self.largest_lesion is not None:
                raise ValueError("A negative gold label must have count 0 and no largest lesion")
            object.__setattr__(self, "lesion_count", 0)
        else:
            if self.lesion_count is not None and self.lesion_count <= 0:
                raise ValueError("A positive gold label needs a positive count or None")
            if self.largest_lesion is None:
                object.__setattr__(self, "largest_lesion", GoldLargestLesion())
        return self


class GoldProvenance(BaseModel):
    """Who produced the gold label and how it relates to the source label."""

    model_config = ConfigDict(extra="forbid")

    method: GoldMethod
    annotators: list[str] = Field(..., min_length=1)
    adjudicator: Optional[str] = None
    reviewed_at: str
    notes: str = ""
    evidence_quote: Optional[str] = Field(
        default=None, description="Verbatim report text supporting the presence decision"
    )
    source_label_agreement: dict[str, bool] = Field(
        default_factory=dict,
        description="Per field: does the gold value equal the source-derived label?",
    )


class GoldStandardRecord(BaseModel):
    """One human gold-standard study."""

    model_config = ConfigDict(extra="forbid")

    id: str
    gold: GoldLabels
    provenance: GoldProvenance
