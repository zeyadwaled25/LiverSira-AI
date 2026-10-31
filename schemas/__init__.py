"""LiverSira-AI Pydantic Schemas Package.

Exports schemas for:
- MasterRecord (Storage Tier)
- GroundTruthRecord, GroundTruthLabels, LargestLesionGT, Attenuation, ValidationStatus (Evaluation Tier)
- InferenceRecord, LiverLesionsInference, EvidenceSpan, MissingOrUnclearField (Inference Tier)
"""

from schemas.master_schema import (
    MasterMeta,
    MasterRecord,
    MasterReports,
    MasterSource,
    OrganFindings,
    PatientMeta,
)
from schemas.ground_truth_schema import (
    Attenuation,
    GroundTruthLabels,
    GroundTruthRecord,
    LargestLesionGT,
    ValidationMetadata,
    ValidationStatus,
)
from schemas.inference_schema import (
    ConfidenceScores,
    EvidenceSpan,
    InferenceRecord,
    InputReference,
    LargestLesionInference,
    LiverLesionsInference,
    MissingOrUnclearField,
    StudyMetadata,
)

__all__ = [
    # Master
    "MasterRecord",
    "MasterSource",
    "MasterMeta",
    "PatientMeta",
    "OrganFindings",
    "MasterReports",
    # Ground Truth
    "GroundTruthRecord",
    "GroundTruthLabels",
    "LargestLesionGT",
    "Attenuation",
    "ValidationStatus",
    "ValidationMetadata",
    # Inference
    "InferenceRecord",
    "LiverLesionsInference",
    "LargestLesionInference",
    "EvidenceSpan",
    "MissingOrUnclearField",
    "ConfidenceScores",
    "InputReference",
    "StudyMetadata",
]

