"""Master Medical Record Schema (Data Storage Tier).

Preserves all 37 source dataset columns, imaging metadata, multi-organ findings
(liver, spleen, pancreas, kidneys, colon), and all report variants.
"""

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class MasterSource(BaseModel):
    """Source provenance and raw column capture."""
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="AbdomenAtlas3.0MiniWithMeta", description="Source dataset name")
    schema_version: str = Field(default="1.0", description="Master schema version")
    raw_columns: dict[str, Any] = Field(
        ...,
        description="Verbatim dictionary of all 37 columns from the source CSV"
    )


class PatientMeta(BaseModel):
    """Demographic metadata."""
    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = Field(default=None, ge=0, description="Patient age at study")
    sex: Optional[str] = Field(default=None, description="Patient biological sex ('M'/'F')")


class MasterMeta(BaseModel):
    """Technical and acquisition metadata for the CT study."""
    model_config = ConfigDict(extra="forbid")

    scanner: Optional[str] = Field(default=None, description="CT scanner manufacturer/model")
    contrast: Optional[str] = Field(default=None, description="Contrast administration details")
    patient: PatientMeta
    study_spacing: Optional[list[float]] = Field(
        default=None,
        description="Voxel spacing [x, y, z] in mm"
    )
    shape: Optional[list[int]] = Field(
        default=None,
        description="Volume matrix dimensions [dim_x, dim_y, dim_z]"
    )


class OrganFindings(BaseModel):
    """Multi-organ segmented measurements and lesion metadata."""
    model_config = ConfigDict(extra="allow")

    liver: dict[str, Any] = Field(..., description="Liver volume and liver lesion metrics")
    spleen: dict[str, Any] = Field(..., description="Spleen volume")
    pancreas: dict[str, Any] = Field(..., description="Pancreas volume and lesion metrics")
    kidneys: dict[str, Any] = Field(..., description="Kidney volumes (left, right, total) and lesion metrics")
    colon: Optional[dict[str, Any]] = Field(default=None, description="Colon volume and lesion metrics")


class MasterReports(BaseModel):
    """All textual records associated with the CT study."""
    model_config = ConfigDict(extra="forbid")

    narrative_original: Optional[str] = Field(
        default=None,
        description="Original unedited radiologist narrative report"
    )
    narrative_clean: Optional[str] = Field(
        default=None,
        description="Whitespace-normalized narrative report (only authorized input for NLP)"
    )
    structured_report: Optional[str] = Field(
        default=None,
        description="Source structured report (isolated in Master Record to prevent data leakage)"
    )
    fusion_structured_report: Optional[str] = Field(
        default=None,
        description="Source fusion structured report (isolated from NLP input)"
    )
    fusion_narrative_report: Optional[str] = Field(
        default=None,
        description="Source fusion narrative report (isolated from NLP input)"
    )
    radiologist_note: Optional[str] = Field(
        default=None,
        description="Additional radiologist notes, if documented"
    )


class MasterRecord(BaseModel):
    """Complete Master Medical Record for a single study."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Primary identifier, e.g. BDMAP_00000001")
    source: MasterSource
    meta: MasterMeta
    organ_findings: OrganFindings
    reports: MasterReports

