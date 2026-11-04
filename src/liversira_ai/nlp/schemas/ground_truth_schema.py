"""Ground Truth / Benchmark Schema.

Defines the reference target labels for training and evaluating clinical NLP models.
Enforces strict medical consistency constraints:
- has_lesion == False <==> lesion_count == 0 and largest_lesion is None
- has_lesion == True <==> lesion_count > 0 and largest_lesion is present
- location_segments Couinaud numbers must be in [1..8]
- attenuation must be one of the three verified radiological classes
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Attenuation(str, Enum):
    """Radiological density relative to normal liver parenchyma."""
    HYPO = "hypoattenuating"
    ISO = "isoattenuating"
    HYPER = "hyperattenuating"


class ValidationStatus(str, Enum):
    """Review and verification status of ground truth labels."""
    PENDING = "pending"
    PARTIALLY_VERIFIED = "partially_verified"
    VERIFIED = "verified"
    NEEDS_ADJUDICATION = "needs_adjudication"


class LargestLesionGT(BaseModel):
    """Ground truth features of the largest liver lesion."""
    model_config = ConfigDict(extra="forbid")

    location_segments: Optional[list[int]] = Field(
        default=None,
        description="Couinaud anatomical liver segments (1 to 8), or None if unmentioned"
    )
    source_diameter_cm: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Largest diameter in cm from source record"
    )
    attenuation: Optional[Attenuation] = Field(
        default=None,
        description="Attenuation category relative to liver parenchyma"
    )

    @model_validator(mode="after")
    def validate_and_deduplicate_segments(self) -> "LargestLesionGT":
        if self.location_segments is not None:
            for s in self.location_segments:
                if not (1 <= s <= 8):
                    raise ValueError(f"Couinaud segment must be between 1 and 8, got {s}")
            # Store sorted and unique segment integers
            object.__setattr__(
                self,
                "location_segments",
                sorted(list(set(self.location_segments)))
            )
        return self


class GroundTruthLabels(BaseModel):
    """Target labels for the liver lesion extraction task."""
    model_config = ConfigDict(extra="forbid")

    has_lesion: bool = Field(..., description="True if one or more liver lesions exist")
    lesion_count: int = Field(..., ge=0, description="Total count of liver lesions")
    largest_lesion: Optional[LargestLesionGT] = Field(
        default=None,
        description="Features of largest lesion; must be None if has_lesion is False"
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "GroundTruthLabels":
        if not self.has_lesion:
            if self.lesion_count != 0:
                raise ValueError(
                    f"Inconsistent state: has_lesion is False but lesion_count is {self.lesion_count} (must be 0)."
                )
            if self.largest_lesion is not None:
                raise ValueError(
                    "Inconsistent state: has_lesion is False but largest_lesion is populated (must be None)."
                )
        else:
            if self.lesion_count <= 0:
                raise ValueError(
                    f"Inconsistent state: has_lesion is True but lesion_count is {self.lesion_count} (must be > 0)."
                )
            # largest_lesion must be provided when lesions exist
            if self.largest_lesion is None:
                raise ValueError(
                    "Inconsistent state: has_lesion is True but largest_lesion object is None."
                )
        return self


class ValidationMetadata(BaseModel):
    """Provenance of human review and adjudication."""
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Current verification tier"
    )
    verified_fields: list[str] = Field(
        default_factory=list,
        description="List of fields confirmed against narrative text"
    )
    unverified_fields: list[str] = Field(
        default_factory=list,
        description="List of fields still pending adjudication"
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="Auditor observations (e.g. data discrepancies or ambiguous text)"
    )


class GroundTruthRecord(BaseModel):
    """A benchmark ground truth record linked to a study by BDMAP ID."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Primary identifier matching Master Record")
    ground_truth: GroundTruthLabels
    validation: ValidationMetadata = Field(default_factory=ValidationMetadata)

