"""Reusable claim-to-proof contracts for the MRI Guardian Layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from icsd.core.serialise import hash_canonical_json


@dataclass(frozen=True)
class AIClaimEnvelope:
    """Explicit AI-side claim over reconstruction, mask, and volume outputs."""

    case_id: str
    study_uid: str
    series_uid: str
    patient_pseudonym: str
    ai_model_id: str
    raw_kspace_hash: str
    voxel_spacing_mm: list[float]
    ai_reconstruction_hash: str
    ai_mask_hash: str
    claimed_tumor_volume_mm3: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "AIClaimEnvelope",
            "interface_version": "2026.03",
            "claim_scope": "reconstruction_segmentation_volume",
            "case_id": self.case_id,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "patient_pseudonym": self.patient_pseudonym,
            "ai_model_id": self.ai_model_id,
            "raw_kspace_hash": self.raw_kspace_hash,
            "voxel_spacing_mm": self.voxel_spacing_mm,
            "claimed_outputs": {
                "ai_reconstruction_hash": self.ai_reconstruction_hash,
                "ai_mask_hash": self.ai_mask_hash,
                "claimed_tumor_volume_mm3": self.claimed_tumor_volume_mm3,
            },
        }


@dataclass(frozen=True)
class MaskDerivationPolicy:
    """Deterministic policy that maps a verified reconstruction to a mask."""

    policy_name: str
    policy_version: str
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "MaskDerivationPolicy",
            "interface_version": self.policy_version,
            "policy_name": self.policy_name,
            "policy_scope": "verified_mask_from_raw_derived_reconstruction",
            "algorithm": "threshold_on_raw_derived_reconstruction",
            "deterministic": True,
            "input_semantics": "raw_derived_reconstruction",
            "output_semantics": "verified_mask",
            "parameters": {
                "comparison": "greater_than_or_equal",
                "threshold": self.threshold,
            },
        }


@dataclass(frozen=True)
class VerifiedReconstructionTrace:
    """Proof trace from raw sensor data to the verified reconstruction outputs."""

    case_id: str
    raw_kspace_hash: str
    derived_reconstruction_hash: str
    forward_projection_hash: str
    verified_mask_hash: str
    computed_tumor_volume_mm3: float
    claim_envelope_hash: str
    mask_derivation_policy_hash: str
    max_measurement_abs_error: float
    relative_measurement_l2_error: float
    max_imaginary_leakage: float
    max_ai_abs_error: float
    mask_voxel_delta: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "VerifiedReconstructionTrace",
            "interface_version": "2026.03",
            "trace_scope": "raw_kspace_to_verified_reconstruction_mask_volume",
            "case_id": self.case_id,
            "claim_envelope_hash": self.claim_envelope_hash,
            "mask_derivation_policy_hash": self.mask_derivation_policy_hash,
            "raw_kspace_hash": self.raw_kspace_hash,
            "derived_reconstruction_hash": self.derived_reconstruction_hash,
            "forward_projection_hash": self.forward_projection_hash,
            "verified_mask_hash": self.verified_mask_hash,
            "computed_tumor_volume_mm3": self.computed_tumor_volume_mm3,
            "measurement_consistency": {
                "max_abs_error": self.max_measurement_abs_error,
                "relative_l2_error": self.relative_measurement_l2_error,
                "max_imaginary_leakage": self.max_imaginary_leakage,
            },
            "claim_comparison": {
                "max_ai_abs_error": self.max_ai_abs_error,
                "mask_voxel_delta": self.mask_voxel_delta,
            },
        }


def build_ai_claim_envelope(
    *,
    state: dict[str, Any],
    raw_kspace_hash: str,
    ai_reconstruction_hash: str,
    ai_mask_hash: str,
    claimed_tumor_volume_mm3: float,
) -> dict[str, Any]:
    return AIClaimEnvelope(
        case_id=str(state["case_id"]),
        study_uid=str(state["study_uid"]),
        series_uid=str(state["series_uid"]),
        patient_pseudonym=str(state["patient_pseudonym"]),
        ai_model_id=str(state["ai_model_id"]),
        raw_kspace_hash=raw_kspace_hash,
        voxel_spacing_mm=[float(value) for value in state["voxel_spacing_mm"]],
        ai_reconstruction_hash=ai_reconstruction_hash,
        ai_mask_hash=ai_mask_hash,
        claimed_tumor_volume_mm3=float(claimed_tumor_volume_mm3),
    ).to_dict()


def build_mask_derivation_policy(
    *,
    policy_name: str,
    policy_version: str,
    threshold: float,
) -> dict[str, Any]:
    return MaskDerivationPolicy(
        policy_name=policy_name,
        policy_version=policy_version,
        threshold=float(threshold),
    ).to_dict()


def build_verified_reconstruction_trace(
    *,
    case_id: str,
    raw_kspace_hash: str,
    derived_reconstruction_hash: str,
    forward_projection_hash: str,
    verified_mask_hash: str,
    computed_tumor_volume_mm3: float,
    claim_envelope: dict[str, Any],
    mask_derivation_policy: dict[str, Any],
    max_measurement_abs_error: float,
    relative_measurement_l2_error: float,
    max_imaginary_leakage: float,
    max_ai_abs_error: float,
    mask_voxel_delta: int,
) -> dict[str, Any]:
    return VerifiedReconstructionTrace(
        case_id=case_id,
        raw_kspace_hash=raw_kspace_hash,
        derived_reconstruction_hash=derived_reconstruction_hash,
        forward_projection_hash=forward_projection_hash,
        verified_mask_hash=verified_mask_hash,
        computed_tumor_volume_mm3=float(computed_tumor_volume_mm3),
        claim_envelope_hash=hash_canonical_json(claim_envelope),
        mask_derivation_policy_hash=hash_canonical_json(mask_derivation_policy),
        max_measurement_abs_error=float(max_measurement_abs_error),
        relative_measurement_l2_error=float(relative_measurement_l2_error),
        max_imaginary_leakage=float(max_imaginary_leakage),
        max_ai_abs_error=float(max_ai_abs_error),
        mask_voxel_delta=int(mask_voxel_delta),
    ).to_dict()
