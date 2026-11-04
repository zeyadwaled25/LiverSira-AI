"""Model Inference & Evidence Schema.

Defines the output produced by clinical NLP models when parsing unstructured CT reports.
Supports Explainable AI (XAI) with evidence text spans, character offsets,
handling of negative and ambiguous findings (Ternary logic: True/False/None),
and calibration confidence scores.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputReference(BaseModel):
    """Reference to the source input field and normalization version."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        default="narrative_clean",
        description="Text field consumed by the model"
    )
    text_version: str = Field(
        default="1.0",
        description="Version identifier of preprocessing pipeline"
    )


class StudyMetadata(BaseModel):
    """Inferred or confirmed imaging modality."""
    model_config = ConfigDict(extra="forbid")

    modality: str = Field(default="CT", description="Imaging modality (e.g. CT, MRI)")


class EvidenceSpan(BaseModel):
    """Exact text evidence backing an extracted clinical entity."""
    model_config = ConfigDict(extra="forbid")

    target_field: str = Field(
        ...,
        description="JSON path of the supported field (e.g. 'liver_lesions.lesion_count')"
    )
    text: str = Field(
        ...,
        description="Exact verbatim substring found in the input report"
    )
    start: int = Field(..., ge=0, description="Start character offset in narrative_clean")
    end: int = Field(..., ge=0, description="End character offset in narrative_clean")

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError(
                f"Invalid span offset: end ({self.end}) must be greater than start ({self.start})"
            )
        return self


class MissingOrUnclearField(BaseModel):
    """Documentation of clinical features that are absent, ambiguous, or contested."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Target field that could not be deterministically extracted")
    reason: str = Field(
        ...,
        description="Explanation: 'not_mentioned', 'ambiguous', 'conflicting_terms', or 'poor_visualization'"
    )


class LargestLesionInference(BaseModel):
    """Model predictions for the index (largest) liver lesion."""
    model_config = ConfigDict(extra="forbid")

    location_segments: Optional[list[int]] = Field(
        default=None,
        description="Predicted Couinaud liver segments (1 to 8)"
    )
    reported_dimensions_cm: Optional[list[float]] = Field(
        default=None,
        description="All dimensions mentioned in report in cm (e.g. [3.1, 1.6])"
    )
    derived_max_diameter_cm: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Maximum diameter derived as max(reported_dimensions_cm)"
    )
    attenuation: Optional[str] = Field(
        default=None,
        description="Extracted attenuation descriptor"
    )

    @model_validator(mode="after")
    def auto_derive_max_diameter(self) -> "LargestLesionInference":
        if self.reported_dimensions_cm and self.derived_max_diameter_cm is None:
            object.__setattr__(
                self,
                "derived_max_diameter_cm",
                max(self.reported_dimensions_cm)
            )
        return self


class LiverLesionsInference(BaseModel):
    """Extracted liver findings following Ternary Logic (True / False / None)."""
    model_config = ConfigDict(extra="forbid")

    has_lesion: Optional[bool] = Field(
        default=None,
        description="True if confirmed, False if explicitly negated, None if omitted/unclear"
    )
    lesion_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of lesions detected, 0 if negated, None if unquantified"
    )
    largest_lesion: Optional[LargestLesionInference] = Field(
        default=None,
        description="Detailed features of largest lesion, or None if no lesion / unmentioned"
    )


class ConfidenceScores(BaseModel):
    """Model calibrated probabilities [0.0 - 1.0] for individual predictions."""
    model_config = ConfigDict(extra="allow")

    has_lesion: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lesion_count: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    location_segments: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reported_dimensions_cm: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    attenuation: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InferenceRecord(BaseModel):
    """Complete prediction object returned by the clinical NLP inference engine."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="BDMAP study identifier")
    input_reference: InputReference = Field(default_factory=InputReference)
    study: StudyMetadata = Field(default_factory=StudyMetadata)
    liver_lesions: LiverLesionsInference
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    missing_or_unclear: list[MissingOrUnclearField] = Field(default_factory=list)
    confidence: Optional[ConfidenceScores] = None

