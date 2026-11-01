"""Unit tests for LiverSira-AI Pydantic schemas."""

import pytest
from pydantic import ValidationError

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
    EvidenceSpan,
    InferenceRecord,
    LargestLesionInference,
    LiverLesionsInference,
    MissingOrUnclearField,
)


# ── Ground Truth Schema Tests ──────────────────────────────────────────


def test_ground_truth_zero_lesions():
    """Zero lesions requires lesion_count == 0 and largest_lesion is None."""
    gt = GroundTruthRecord(
        id="BDMAP_00000002",
        ground_truth=GroundTruthLabels(
            has_lesion=False,
            lesion_count=0,
            largest_lesion=None,
        ),
        validation=ValidationMetadata(
            status=ValidationStatus.PENDING,
            unverified_fields=["has_lesion", "lesion_count"],
        ),
    )
    assert gt.ground_truth.has_lesion is False
    assert gt.ground_truth.lesion_count == 0
    assert gt.ground_truth.largest_lesion is None


def test_ground_truth_inconsistent_negation():
    """has_lesion=False with count > 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        GroundTruthLabels(
            has_lesion=False,
            lesion_count=3,
            largest_lesion=None,
        )


def test_ground_truth_inconsistent_positive():
    """has_lesion=True with count == 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        GroundTruthLabels(
            has_lesion=True,
            lesion_count=0,
            largest_lesion=None,
        )


def test_ground_truth_positive_lesions():
    """Positive lesion record with validated Couinaud segments and attenuation."""
    gt = GroundTruthRecord(
        id="BDMAP_00000001",
        ground_truth=GroundTruthLabels(
            has_lesion=True,
            lesion_count=25,
            largest_lesion=LargestLesionGT(
                location_segments=[5, 2, 5],  # Duplicates should be deduplicated & sorted -> [2, 5]
                source_diameter_cm=3.1,
                attenuation=Attenuation.HYPO,
            ),
        ),
    )
    assert gt.ground_truth.largest_lesion.location_segments == [2, 5]
    assert gt.ground_truth.largest_lesion.attenuation == Attenuation.HYPO
    assert gt.validation.status == ValidationStatus.PENDING


def test_ground_truth_invalid_segment():
    """Segments outside [1, 8] must be rejected."""
    with pytest.raises(ValidationError):
        LargestLesionGT(location_segments=[9])

    with pytest.raises(ValidationError):
        LargestLesionGT(location_segments=[0])


# ── Inference Schema Tests ─────────────────────────────────────────────


def test_inference_auto_derive_max_diameter():
    """derived_max_diameter_cm is automatically computed from max(reported_dimensions_cm)."""
    lesion = LargestLesionInference(
        reported_dimensions_cm=[3.1, 1.6],
        location_segments=[2],
        attenuation="hypoattenuating",
    )
    assert lesion.derived_max_diameter_cm == 3.1


def test_evidence_span_offsets():
    """EvidenceSpan requires end > start."""
    with pytest.raises(ValidationError):
        EvidenceSpan(
            target_field="liver_lesions.lesion_count",
            text="invalid",
            start=10,
            end=5,
        )


def test_inference_record_ternary_logic():
    """Supports True, False, and None (not mentioned) for has_lesion."""
    # Not mentioned / ambiguous
    inf_unclear = InferenceRecord(
        id="BDMAP_00000099",
        liver_lesions=LiverLesionsInference(
            has_lesion=None,
            lesion_count=None,
            largest_lesion=None,
        ),
        missing_or_unclear=[
            MissingOrUnclearField(field="liver_lesions.has_lesion", reason="not_mentioned")
        ],
    )
    assert inf_unclear.liver_lesions.has_lesion is None
    assert len(inf_unclear.missing_or_unclear) == 1


# ── Master Record Schema Tests ─────────────────────────────────────────


def test_master_record_serialization():
    """MasterRecord serializes to JSON with all 37 raw columns preserved."""
    rec = MasterRecord(
        id="BDMAP_00000001",
        source=MasterSource(
            dataset="AbdomenAtlas3.0MiniWithMeta",
            schema_version="1.0",
            raw_columns={"column_1": "test", "column_37": 123},
        ),
        meta=MasterMeta(
            scanner="GE",
            contrast="IV",
            patient=PatientMeta(age=55.0, sex="F"),
            study_spacing=[0.8, 0.8, 2.5],
            shape=[512, 512, 100],
        ),
        organ_findings=OrganFindings(
            liver={"volume_cc": 1500.0, "lesion_count": 0},
            spleen={"volume_cc": 120.0},
            pancreas={"volume_cc": 70.0},
            kidneys={"total_volume_cc": 300.0},
        ),
        reports=MasterReports(
            narrative_original="Normal liver.",
            narrative_clean="Normal liver.",
            structured_report="Report text",
        ),
    )
    json_str = rec.model_dump_json()
    assert "BDMAP_00000001" in json_str
    assert "raw_columns" in json_str

