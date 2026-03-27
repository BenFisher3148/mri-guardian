"""Reusable claim-to-proof contracts for the MRI Guardian Layer."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from icsd.core.serialise import hash_canonical_json

from .mask_engine import THRESHOLD_ALGORITHM, resolve_mask_policy
from .profiles import (
    build_verification_profile_descriptor,
    get_verification_profile,
    list_verification_profiles,
    resolve_profile_name,
    verification_profile_descriptor_from_dict,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_AI_STACK_IDENTITY_FIELDS = (
    "model_version_id",
    "weights_sha256",
    "preprocessing_pipeline_sha256",
    "postprocessing_pipeline_sha256",
    "input_normalization",
    "sequence_model_compatibility",
)
_INPUT_NORMALIZATION_FIELDS = ("scheme", "parameters")
_SEQUENCE_MODEL_COMPATIBILITY_FIELDS = ("compatibility_type", "declared_value")
_TIMESTAMP_REPO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_CURRENT_TIMESTAMP_PROVENANCE = frozenset({"trusted_study_metadata", "missing", "untrusted"})
_PRIOR_TIMESTAMP_PROVENANCE = frozenset(
    {"verified_bundle", "trusted_study_metadata", "missing", "untrusted"}
)


def _require_object_shape(
    value: Any,
    *,
    path: str,
    required_fields: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    payload = dict(value)
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"{path} is missing required fields: {missing}.")
    unexpected_fields = sorted(
        field for field in payload if field not in set(required_fields)
    )
    if unexpected_fields:
        unexpected = ", ".join(unexpected_fields)
        raise ValueError(f"{path} contains unsupported fields: {unexpected}.")
    return payload


def _require_non_empty_string(value: Any, *, path: str) -> str:
    if type(value) is not str or value == "":
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _require_trimmed_non_empty_string(value: Any, *, path: str) -> str:
    text = _require_non_empty_string(value, path=path)
    if text != text.strip():
        raise ValueError(
            f"{path} must not include leading or trailing whitespace."
        )
    return text


def _require_sha256_hex(value: Any, *, path: str) -> str:
    digest = _require_trimmed_non_empty_string(value, path=path)
    if digest.startswith("0x"):
        raise ValueError(f"{path} must not include a 0x prefix.")
    if not _SHA256_HEX_RE.fullmatch(digest):
        raise ValueError(f"{path} must be a lowercase SHA-256 hex digest.")
    return digest


def _canonicalise_json_like(value: Any, *, path: str) -> Any:
    if type(value) is str:
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite float values.")
        return value
    if type(value) is list:
        return [
            _canonicalise_json_like(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} must contain only string keys.")
            payload[key] = _canonicalise_json_like(item, path=f"{path}.{key}")
        return payload
    if value is None:
        raise ValueError(f"{path} must not contain null values.")
    raise ValueError(
        f"{path} contains unsupported value type {type(value).__name__!r}."
    )


def _round_guardian_float(value: float) -> float:
    if abs(value) < 1e-12:
        return 0.0
    return round(float(value), 9)


def _canonicalise_repo_timestamp(value: Any, *, path: str) -> str:
    text = _require_trimmed_non_empty_string(value, path=path)
    try:
        parsed = datetime.strptime(text, _TIMESTAMP_REPO_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"{path} must use canonical UTC format YYYY-MM-DDTHH:MM:SSZ."
        ) from exc
    return parsed.strftime(_TIMESTAMP_REPO_FORMAT)


def canonicalise_timestamp_provenance(
    value: Any,
    *,
    path: str,
    allow_verified_bundle: bool,
) -> str:
    provenance = _require_trimmed_non_empty_string(value, path=path)
    allowed = _PRIOR_TIMESTAMP_PROVENANCE if allow_verified_bundle else _CURRENT_TIMESTAMP_PROVENANCE
    if provenance not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"{path} must be one of: {allowed_list}.")
    return provenance


def canonicalise_study_acquired_at(
    value: Any,
    *,
    path: str,
) -> str | None:
    if value is None:
        return None
    return _canonicalise_repo_timestamp(value, path=path)


def canonicalise_spatial_geometry(
    value: Any,
    *,
    path: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _require_object_shape(
        value,
        path=path,
        required_fields=("frame_of_reference_uid", "affine_ijk_to_mm"),
    )
    frame_uid = _require_trimmed_non_empty_string(
        payload["frame_of_reference_uid"],
        path=f"{path}.frame_of_reference_uid",
    )
    affine = payload["affine_ijk_to_mm"]
    if not isinstance(affine, list) or len(affine) != 4:
        raise ValueError(f"{path}.affine_ijk_to_mm must be a 4x4 numeric matrix.")
    canonical_affine: list[list[float]] = []
    for row_index, row in enumerate(affine):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{path}.affine_ijk_to_mm must be a 4x4 numeric matrix.")
        canonical_row: list[float] = []
        for column_index, item in enumerate(row):
            if type(item) not in {int, float} or not math.isfinite(float(item)):
                raise ValueError(
                    f"{path}.affine_ijk_to_mm[{row_index}][{column_index}] "
                    "must be a finite number."
                )
            canonical_row.append(_round_guardian_float(float(item)))
        canonical_affine.append(canonical_row)
    return {
        "frame_of_reference_uid": frame_uid,
        "affine_ijk_to_mm": canonical_affine,
    }


def _sequence_family_vocabulary() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                get_verification_profile(profile_name).sequence_family
                for profile_name in list_verification_profiles()
            }
        )
    )


def _expected_sequence_family_from_state(state: Mapping[str, Any]) -> str:
    profile_descriptor = state.get("verification_profile_descriptor")
    if isinstance(profile_descriptor, Mapping):
        try:
            return verification_profile_descriptor_from_dict(
                dict(profile_descriptor)
            ).sequence_family
        except ValueError:
            profile_name = profile_descriptor.get("profile_name")
            if isinstance(profile_name, str) and profile_name.strip():
                return get_verification_profile(profile_name.strip()).sequence_family
    resolved_profile_name = resolve_profile_name(profile_name=None, state=dict(state))
    return get_verification_profile(resolved_profile_name).sequence_family


def canonicalise_ai_stack_identity(state: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = state.get("ai_stack_identity")
    if payload is None:
        return None
    ai_stack_identity = _require_object_shape(
        payload,
        path="ai_stack_identity",
        required_fields=_AI_STACK_IDENTITY_FIELDS,
    )
    input_normalization = _require_object_shape(
        ai_stack_identity["input_normalization"],
        path="ai_stack_identity.input_normalization",
        required_fields=_INPUT_NORMALIZATION_FIELDS,
    )
    sequence_model_compatibility = _require_object_shape(
        ai_stack_identity["sequence_model_compatibility"],
        path="ai_stack_identity.sequence_model_compatibility",
        required_fields=_SEQUENCE_MODEL_COMPATIBILITY_FIELDS,
    )
    compatibility_type = _require_trimmed_non_empty_string(
        sequence_model_compatibility["compatibility_type"],
        path="ai_stack_identity.sequence_model_compatibility.compatibility_type",
    )
    declared_value = _require_trimmed_non_empty_string(
        sequence_model_compatibility["declared_value"],
        path="ai_stack_identity.sequence_model_compatibility.declared_value",
    )
    if compatibility_type not in {"exact_sequence_id", "sequence_family"}:
        raise ValueError(
            "ai_stack_identity.sequence_model_compatibility.compatibility_type "
            "must be 'exact_sequence_id' or 'sequence_family'."
        )
    if compatibility_type == "exact_sequence_id":
        sequence_id = _require_non_empty_string(
            state.get("sequence_id"),
            path="state.sequence_id",
        )
        if declared_value != sequence_id:
            raise ValueError(
                "ai_stack_identity.sequence_model_compatibility.declared_value "
                "must exactly match state.sequence_id."
            )
    else:
        allowed_values = _sequence_family_vocabulary()
        expected_sequence_family = _expected_sequence_family_from_state(state)
        if declared_value not in allowed_values:
            raise ValueError(
                "ai_stack_identity.sequence_model_compatibility.declared_value "
                "must be a known verification-profile sequence_family."
            )
        if declared_value != expected_sequence_family:
            raise ValueError(
                "ai_stack_identity.sequence_model_compatibility.declared_value "
                "must match the active case/profile sequence_family."
            )
    return {
        "model_version_id": _require_trimmed_non_empty_string(
            ai_stack_identity["model_version_id"],
            path="ai_stack_identity.model_version_id",
        ),
        "weights_sha256": _require_sha256_hex(
            ai_stack_identity["weights_sha256"],
            path="ai_stack_identity.weights_sha256",
        ),
        "preprocessing_pipeline_sha256": _require_sha256_hex(
            ai_stack_identity["preprocessing_pipeline_sha256"],
            path="ai_stack_identity.preprocessing_pipeline_sha256",
        ),
        "postprocessing_pipeline_sha256": _require_sha256_hex(
            ai_stack_identity["postprocessing_pipeline_sha256"],
            path="ai_stack_identity.postprocessing_pipeline_sha256",
        ),
        "input_normalization": {
            "scheme": _require_trimmed_non_empty_string(
                input_normalization["scheme"],
                path="ai_stack_identity.input_normalization.scheme",
            ),
            "parameters": _canonicalise_json_like(
                input_normalization["parameters"],
                path="ai_stack_identity.input_normalization.parameters",
            ),
        },
        "sequence_model_compatibility": {
            "compatibility_type": compatibility_type,
            "declared_value": declared_value,
        },
    }


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
    ai_stack_identity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface_type": "AIClaimEnvelope",
            "interface_version": "2026.03",
            "claim_scope": "reconstruction_segmentation_volume",
            "case_id": self.case_id,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "patient_pseudonym": self.patient_pseudonym,
            "ai_model_id": self.ai_model_id,
        }
        if self.ai_stack_identity is not None:
            payload["ai_stack_identity"] = deepcopy(self.ai_stack_identity)
        payload["raw_kspace_hash"] = self.raw_kspace_hash
        payload["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        payload["claimed_outputs"] = {
            "ai_reconstruction_hash": self.ai_reconstruction_hash,
            "ai_mask_hash": self.ai_mask_hash,
            "claimed_tumor_volume_mm3": self.claimed_tumor_volume_mm3,
        }
        return payload


@dataclass(frozen=True)
class MaskDerivationPolicy:
    """Deterministic policy that maps a verified reconstruction to a mask."""

    profile_name: str
    profile_revision: str
    profile_digest: str
    icsd_policy_id: str
    icsd_policy_version: str
    algorithm: str
    parameters: dict[str, Any]
    deterministic: bool
    input_semantics: str
    output_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "MaskDerivationPolicy",
            "interface_version": self.icsd_policy_version,
            "profile_name": self.profile_name,
            "profile_revision": self.profile_revision,
            "profile_digest": self.profile_digest,
            "icsd_policy_id": self.icsd_policy_id,
            "icsd_policy_version": self.icsd_policy_version,
            "policy_scope": "verified_mask_from_raw_derived_reconstruction",
            "algorithm": self.algorithm,
            "deterministic": self.deterministic,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class VerificationTolerancePolicy:
    """Explicit tolerance contract for local and global verification metrics."""

    profile_name: str
    profile_revision: str
    profile_digest: str
    icsd_policy_id: str
    icsd_policy_version: str
    guard_band_fraction: float
    spatial_per_voxel_abs_error: float
    spatial_global_relative_l2_error: float
    frequency_per_coefficient_abs_error: float
    frequency_global_relative_l2_error: float
    imaginary_leakage_abs_error: float
    volume_absolute_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "VerificationTolerancePolicy",
            "interface_version": self.icsd_policy_version,
            "profile_name": self.profile_name,
            "profile_revision": self.profile_revision,
            "profile_digest": self.profile_digest,
            "icsd_policy_id": self.icsd_policy_id,
            "icsd_policy_version": self.icsd_policy_version,
            "policy_scope": "attack_aware_numeric_tolerances",
            "guard_band_fraction": self.guard_band_fraction,
            "rules": {
                "ai_reconstruction_match": {
                    "domain": "spatial_image_domain",
                    "attack_surface": (
                        "Distributed low-amplitude voxel perturbations are checked with a "
                        "global norm in addition to a per-voxel maximum."
                    ),
                    "metrics": [
                        {
                            "metric": "per_voxel_abs_error_max",
                            "scope": "per_voxel",
                            "threshold": self.spatial_per_voxel_abs_error,
                        },
                        {
                            "metric": "global_relative_l2_error",
                            "scope": "global_norm",
                            "threshold": self.spatial_global_relative_l2_error,
                        },
                    ],
                },
                "measurement_consistency": {
                    "domain": "frequency_domain",
                    "attack_surface": (
                        "Distributed low-amplitude k-space perturbations are checked with a "
                        "global norm in addition to a per-frequency maximum."
                    ),
                    "metrics": [
                        {
                            "metric": "per_frequency_complex_abs_error_max",
                            "scope": "per_frequency",
                            "threshold": self.frequency_per_coefficient_abs_error,
                        },
                        {
                            "metric": "global_relative_l2_error",
                            "scope": "global_norm",
                            "threshold": self.frequency_global_relative_l2_error,
                        },
                        {
                            "metric": "spatial_imaginary_leakage_max",
                            "scope": "per_voxel",
                            "threshold": self.imaginary_leakage_abs_error,
                        },
                    ],
                },
                "tumor_volume_proof": {
                    "domain": "scalar_volume",
                    "attack_surface": "Scalar claim delta is checked as an absolute error.",
                    "metrics": [
                        {
                            "metric": "absolute_error",
                            "scope": "global_scalar",
                            "threshold": self.volume_absolute_error,
                        }
                    ],
                },
                "mask_verification": {
                    "domain": "binary_mask",
                    "attack_surface": "No numeric tolerance is allowed; the mask must match exactly.",
                    "metrics": [
                        {
                            "metric": "voxel_delta",
                            "scope": "exact_match",
                            "threshold": 0.0,
                        }
                    ],
                },
            },
        }


@dataclass(frozen=True)
class VerifiedReconstructionTrace:
    """Proof trace from raw sensor data to the verified reconstruction outputs."""

    case_id: str
    profile_name: str
    profile_revision: str
    profile_digest: str
    icsd_policy_id: str
    icsd_policy_version: str
    raw_kspace_hash: str
    derived_reconstruction_hash: str
    forward_projection_hash: str
    verified_mask_hash: str
    computed_tumor_volume_mm3: float
    claim_envelope_hash: str
    mask_derivation_policy_hash: str
    tolerance_policy_hash: str
    max_measurement_abs_error: float
    relative_measurement_l2_error: float
    max_imaginary_leakage: float
    max_ai_abs_error: float
    relative_ai_l2_error: float
    mask_voxel_delta: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "VerifiedReconstructionTrace",
            "interface_version": "2026.03",
            "trace_scope": "raw_kspace_to_verified_reconstruction_mask_volume",
            "case_id": self.case_id,
            "profile_name": self.profile_name,
            "profile_revision": self.profile_revision,
            "profile_digest": self.profile_digest,
            "icsd_policy_id": self.icsd_policy_id,
            "icsd_policy_version": self.icsd_policy_version,
            "claim_envelope_hash": self.claim_envelope_hash,
            "mask_derivation_policy_hash": self.mask_derivation_policy_hash,
            "tolerance_policy_hash": self.tolerance_policy_hash,
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
                "relative_l2_error": self.relative_ai_l2_error,
                "mask_voxel_delta": self.mask_voxel_delta,
            },
        }


@dataclass(frozen=True)
class CrossSeriesVerificationPolicy:
    """Study-level policy that selects the blocking cross-series gate."""

    case_id: str
    primary_series_id: str
    supporting_series_ids: tuple[str, ...]
    required_series_ids: tuple[str, ...]
    geometry_rule: str
    corroboration_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "CrossSeriesVerificationPolicy",
            "interface_version": "2026.03",
            "case_id": self.case_id,
            "primary_series_id": self.primary_series_id,
            "supporting_series_ids": list(self.supporting_series_ids),
            "required_series_ids": list(self.required_series_ids),
            "geometry_rule": self.geometry_rule,
            "corroboration_rule": self.corroboration_rule,
        }


@dataclass(frozen=True)
class CrossSeriesVerificationTrace:
    """Deterministic trace for study-level geometry and corroboration gating."""

    case_id: str
    policy_hash: str
    geometry_verification: dict[str, Any]
    lesion_corroboration: dict[str, Any]
    merged_release_evaluation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "CrossSeriesVerificationTrace",
            "interface_version": "2026.03",
            "case_id": self.case_id,
            "policy_hash": self.policy_hash,
            "geometry_verification": dict(self.geometry_verification),
            "lesion_corroboration": dict(self.lesion_corroboration),
            "merged_release_evaluation": dict(self.merged_release_evaluation),
        }


@dataclass(frozen=True)
class MultiSeriesVerificationProfileDescriptorSet:
    """Ordered wrapper that binds per-series runtime descriptors for one study."""

    case_id: str
    series_descriptors: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "MultiSeriesVerificationProfileDescriptorSet",
            "interface_version": "2026.03",
            "case_id": self.case_id,
            "series_descriptors": [dict(item) for item in self.series_descriptors],
        }


@dataclass(frozen=True)
class PriorVerifiedContext:
    """Compact canonical prior-summary contract for temporal comparison."""

    prior_bundle_sha256: str
    prior_case_id: str
    prior_study_uid: str
    prior_series_uid: str
    prior_released_state_hash: str
    prior_verification_report_hash: str
    prior_profile: dict[str, Any]
    prior_release: dict[str, Any]
    study_acquired_at: str | None
    timestamp_provenance: str
    verified_mask: list[list[list[int]]]
    verified_mask_hash: str
    verified_volume_mm3: float
    verified_mask_voxel_count: int
    voxel_spacing_mm: list[float]
    spatial_geometry: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "PriorVerifiedContext",
            "interface_version": "2026.03",
            "prior_bundle_sha256": self.prior_bundle_sha256,
            "prior_case_id": self.prior_case_id,
            "prior_study_uid": self.prior_study_uid,
            "prior_series_uid": self.prior_series_uid,
            "prior_released_state_hash": self.prior_released_state_hash,
            "prior_verification_report_hash": self.prior_verification_report_hash,
            "prior_profile": deepcopy(self.prior_profile),
            "prior_release": deepcopy(self.prior_release),
            "study_acquired_at": self.study_acquired_at,
            "timestamp_provenance": self.timestamp_provenance,
            "verified_mask": deepcopy(self.verified_mask),
            "verified_mask_hash": self.verified_mask_hash,
            "verified_volume_mm3": self.verified_volume_mm3,
            "verified_mask_voxel_count": self.verified_mask_voxel_count,
            "voxel_spacing_mm": list(self.voxel_spacing_mm),
            "spatial_geometry": deepcopy(self.spatial_geometry),
        }


@dataclass(frozen=True)
class TemporalConsistencyTrace:
    """Deterministic temporal comparison trace for Level 3 longitudinal review."""

    case_id: str
    profile_name: str
    profile_revision: str
    profile_digest: str
    comparison_status: str
    comparison_scope: str
    prior_admissibility: dict[str, Any]
    chronology_assessment: dict[str, Any]
    geometry_assessment: dict[str, Any]
    computed_metrics: dict[str, Any]
    flags: tuple[str, ...]
    skipped_metrics: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_type": "TemporalConsistencyTrace",
            "interface_version": "2026.03",
            "case_id": self.case_id,
            "profile_name": self.profile_name,
            "profile_revision": self.profile_revision,
            "profile_digest": self.profile_digest,
            "comparison_status": self.comparison_status,
            "comparison_scope": self.comparison_scope,
            "prior_admissibility": deepcopy(self.prior_admissibility),
            "chronology_assessment": deepcopy(self.chronology_assessment),
            "geometry_assessment": deepcopy(self.geometry_assessment),
            "computed_metrics": deepcopy(self.computed_metrics),
            "flags": list(self.flags),
            "skipped_metrics": [deepcopy(item) for item in self.skipped_metrics],
        }


def canonicalise_prior_verified_context(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    prior = _require_object_shape(
        payload,
        path="prior_verified_context",
        required_fields=(
            "prior_bundle_sha256",
            "prior_case_id",
            "prior_study_uid",
            "prior_series_uid",
            "prior_released_state_hash",
            "prior_verification_report_hash",
            "prior_profile",
            "prior_release",
            "study_acquired_at",
            "timestamp_provenance",
            "verified_mask",
            "verified_mask_hash",
            "verified_volume_mm3",
            "verified_mask_voxel_count",
            "voxel_spacing_mm",
            "spatial_geometry",
        ),
    )
    prior_profile = _require_object_shape(
        prior["prior_profile"],
        path="prior_verified_context.prior_profile",
        required_fields=(
            "profile_name",
            "profile_revision",
            "profile_digest",
            "sequence_family",
        ),
    )
    prior_release = _require_object_shape(
        prior["prior_release"],
        path="prior_verified_context.prior_release",
        required_fields=(
            "release_status",
            "final_disposition",
            "decision_class",
        ),
    )
    verified_mask_array = np.asarray(prior["verified_mask"], dtype=np.int64)
    if verified_mask_array.ndim != 3 or verified_mask_array.size == 0:
        raise ValueError(
            "prior_verified_context.verified_mask must be a non-empty 3D binary tensor."
        )
    if not np.isin(verified_mask_array, (0, 1)).all():
        raise ValueError(
            "prior_verified_context.verified_mask must contain only 0/1 values."
        )
    verified_mask = verified_mask_array.astype(np.int64).tolist()
    verified_mask_hash = hash_canonical_json(verified_mask)
    observed_mask_hash = _require_trimmed_non_empty_string(
        prior["verified_mask_hash"],
        path="prior_verified_context.verified_mask_hash",
    )
    if observed_mask_hash != verified_mask_hash:
        raise ValueError(
            "prior_verified_context.verified_mask_hash must match verified_mask."
        )
    voxel_spacing_raw = prior["voxel_spacing_mm"]
    if not isinstance(voxel_spacing_raw, list) or len(voxel_spacing_raw) != 3:
        raise ValueError(
            "prior_verified_context.voxel_spacing_mm must be a three-item numeric list."
        )
    voxel_spacing_mm: list[float] = []
    for index, item in enumerate(voxel_spacing_raw):
        if type(item) not in {int, float} or not math.isfinite(float(item)) or float(item) <= 0:
            raise ValueError(
                f"prior_verified_context.voxel_spacing_mm[{index}] must be a positive finite number."
            )
        voxel_spacing_mm.append(_round_guardian_float(float(item)))
    verified_mask_voxel_count = int(prior["verified_mask_voxel_count"])
    if verified_mask_voxel_count != int(np.count_nonzero(verified_mask_array)):
        raise ValueError(
            "prior_verified_context.verified_mask_voxel_count must match verified_mask."
        )
    verified_volume_mm3 = _round_guardian_float(float(prior["verified_volume_mm3"]))
    expected_volume = _round_guardian_float(
        verified_mask_voxel_count * float(np.prod(np.asarray(voxel_spacing_mm, dtype=np.float64)))
    )
    if verified_volume_mm3 != expected_volume:
        raise ValueError(
            "prior_verified_context.verified_volume_mm3 must match voxel count and spacing."
        )
    return PriorVerifiedContext(
        prior_bundle_sha256=_require_sha256_hex(
            prior["prior_bundle_sha256"],
            path="prior_verified_context.prior_bundle_sha256",
        ),
        prior_case_id=_require_trimmed_non_empty_string(
            prior["prior_case_id"],
            path="prior_verified_context.prior_case_id",
        ),
        prior_study_uid=_require_trimmed_non_empty_string(
            prior["prior_study_uid"],
            path="prior_verified_context.prior_study_uid",
        ),
        prior_series_uid=_require_trimmed_non_empty_string(
            prior["prior_series_uid"],
            path="prior_verified_context.prior_series_uid",
        ),
        prior_released_state_hash=_require_sha256_hex(
            prior["prior_released_state_hash"],
            path="prior_verified_context.prior_released_state_hash",
        ),
        prior_verification_report_hash=_require_sha256_hex(
            prior["prior_verification_report_hash"],
            path="prior_verified_context.prior_verification_report_hash",
        ),
        prior_profile={
            "profile_name": _require_trimmed_non_empty_string(
                prior_profile["profile_name"],
                path="prior_verified_context.prior_profile.profile_name",
            ),
            "profile_revision": _require_trimmed_non_empty_string(
                prior_profile["profile_revision"],
                path="prior_verified_context.prior_profile.profile_revision",
            ),
            "profile_digest": _require_trimmed_non_empty_string(
                prior_profile["profile_digest"],
                path="prior_verified_context.prior_profile.profile_digest",
            ),
            "sequence_family": _require_trimmed_non_empty_string(
                prior_profile["sequence_family"],
                path="prior_verified_context.prior_profile.sequence_family",
            ),
        },
        prior_release={
            "release_status": _require_trimmed_non_empty_string(
                prior_release["release_status"],
                path="prior_verified_context.prior_release.release_status",
            ),
            "final_disposition": _require_trimmed_non_empty_string(
                prior_release["final_disposition"],
                path="prior_verified_context.prior_release.final_disposition",
            ),
            "decision_class": _require_trimmed_non_empty_string(
                prior_release["decision_class"],
                path="prior_verified_context.prior_release.decision_class",
            ),
        },
        study_acquired_at=canonicalise_study_acquired_at(
            prior["study_acquired_at"],
            path="prior_verified_context.study_acquired_at",
        ),
        timestamp_provenance=canonicalise_timestamp_provenance(
            prior["timestamp_provenance"],
            path="prior_verified_context.timestamp_provenance",
            allow_verified_bundle=True,
        ),
        verified_mask=verified_mask,
        verified_mask_hash=verified_mask_hash,
        verified_volume_mm3=verified_volume_mm3,
        verified_mask_voxel_count=verified_mask_voxel_count,
        voxel_spacing_mm=voxel_spacing_mm,
        spatial_geometry=canonicalise_spatial_geometry(
            prior["spatial_geometry"],
            path="prior_verified_context.spatial_geometry",
        ),
    ).to_dict()


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
        ai_stack_identity=canonicalise_ai_stack_identity(state),
    ).to_dict()


def build_mask_derivation_policy(
    *,
    profile_name: str,
    profile_revision: str,
    profile_digest: str,
    icsd_policy_id: str,
    icsd_policy_version: str,
    mask_rules: dict[str, Any] | None = None,
    algorithm: str | None = None,
    parameters: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if mask_rules is not None and any(
        value is not None for value in (algorithm, parameters, threshold)
    ):
        raise ValueError(
            "build_mask_derivation_policy accepts either mask_rules or the "
            "compatibility algorithm/parameters/threshold inputs, not both."
        )
    if mask_rules is None:
        compatibility_parameters = dict(parameters or {})
        if threshold is not None:
            compatibility_parameters["threshold"] = threshold
        mask_rules = {
            "algorithm": algorithm or THRESHOLD_ALGORITHM,
            "parameters": compatibility_parameters,
        }
    resolved_policy = resolve_mask_policy(mask_rules)
    return MaskDerivationPolicy(
        profile_name=profile_name,
        profile_revision=profile_revision,
        profile_digest=profile_digest,
        icsd_policy_id=icsd_policy_id,
        icsd_policy_version=icsd_policy_version,
        algorithm=resolved_policy.algorithm,
        parameters=dict(resolved_policy.parameters),
        deterministic=bool(resolved_policy.deterministic),
        input_semantics=str(resolved_policy.input_semantics),
        output_semantics=str(resolved_policy.output_semantics),
    ).to_dict()


def build_verification_tolerance_policy(
    *,
    profile_name: str,
    profile_revision: str,
    profile_digest: str,
    icsd_policy_id: str,
    icsd_policy_version: str,
    guard_band_fraction: float,
    spatial_per_voxel_abs_error: float,
    spatial_global_relative_l2_error: float,
    frequency_per_coefficient_abs_error: float,
    frequency_global_relative_l2_error: float,
    imaginary_leakage_abs_error: float,
    volume_absolute_error: float,
) -> dict[str, Any]:
    return VerificationTolerancePolicy(
        profile_name=profile_name,
        profile_revision=profile_revision,
        profile_digest=profile_digest,
        icsd_policy_id=icsd_policy_id,
        icsd_policy_version=icsd_policy_version,
        guard_band_fraction=float(guard_band_fraction),
        spatial_per_voxel_abs_error=float(spatial_per_voxel_abs_error),
        spatial_global_relative_l2_error=float(spatial_global_relative_l2_error),
        frequency_per_coefficient_abs_error=float(frequency_per_coefficient_abs_error),
        frequency_global_relative_l2_error=float(frequency_global_relative_l2_error),
        imaginary_leakage_abs_error=float(imaginary_leakage_abs_error),
        volume_absolute_error=float(volume_absolute_error),
    ).to_dict()


def build_verified_reconstruction_trace(
    *,
    case_id: str,
    profile_name: str,
    profile_revision: str,
    profile_digest: str,
    icsd_policy_id: str,
    icsd_policy_version: str,
    raw_kspace_hash: str,
    derived_reconstruction_hash: str,
    forward_projection_hash: str,
    verified_mask_hash: str,
    computed_tumor_volume_mm3: float,
    claim_envelope: dict[str, Any],
    mask_derivation_policy: dict[str, Any],
    tolerance_policy: dict[str, Any],
    max_measurement_abs_error: float,
    relative_measurement_l2_error: float,
    max_imaginary_leakage: float,
    max_ai_abs_error: float,
    relative_ai_l2_error: float,
    mask_voxel_delta: int,
) -> dict[str, Any]:
    return VerifiedReconstructionTrace(
        case_id=case_id,
        profile_name=profile_name,
        profile_revision=profile_revision,
        profile_digest=profile_digest,
        icsd_policy_id=icsd_policy_id,
        icsd_policy_version=icsd_policy_version,
        raw_kspace_hash=raw_kspace_hash,
        derived_reconstruction_hash=derived_reconstruction_hash,
        forward_projection_hash=forward_projection_hash,
        verified_mask_hash=verified_mask_hash,
        computed_tumor_volume_mm3=float(computed_tumor_volume_mm3),
        claim_envelope_hash=hash_canonical_json(claim_envelope),
        mask_derivation_policy_hash=hash_canonical_json(mask_derivation_policy),
        tolerance_policy_hash=hash_canonical_json(tolerance_policy),
        max_measurement_abs_error=float(max_measurement_abs_error),
        relative_measurement_l2_error=float(relative_measurement_l2_error),
        max_imaginary_leakage=float(max_imaginary_leakage),
        max_ai_abs_error=float(max_ai_abs_error),
        relative_ai_l2_error=float(relative_ai_l2_error),
        mask_voxel_delta=int(mask_voxel_delta),
    ).to_dict()


def build_verification_profile_contract(profile_name: str) -> dict[str, Any]:
    """Return the canonical public descriptor for the active runtime profile."""
    return build_verification_profile_descriptor(profile_name)


def build_cross_series_verification_policy(
    *,
    case_id: str,
    primary_series_id: str,
    supporting_series_ids: list[str] | tuple[str, ...],
    required_series_ids: list[str] | tuple[str, ...],
    geometry_rule: str,
    corroboration_rule: str,
) -> dict[str, Any]:
    return CrossSeriesVerificationPolicy(
        case_id=case_id,
        primary_series_id=primary_series_id,
        supporting_series_ids=tuple(str(value) for value in supporting_series_ids),
        required_series_ids=tuple(str(value) for value in required_series_ids),
        geometry_rule=str(geometry_rule),
        corroboration_rule=str(corroboration_rule),
    ).to_dict()


def build_cross_series_verification_trace(
    *,
    case_id: str,
    policy: dict[str, Any],
    geometry_verification: dict[str, Any],
    lesion_corroboration: dict[str, Any],
    merged_release_evaluation: dict[str, Any],
) -> dict[str, Any]:
    return CrossSeriesVerificationTrace(
        case_id=str(case_id),
        policy_hash=hash_canonical_json(policy),
        geometry_verification=dict(geometry_verification),
        lesion_corroboration=dict(lesion_corroboration),
        merged_release_evaluation=dict(merged_release_evaluation),
    ).to_dict()


def build_multi_series_verification_profile_descriptor_set(
    *,
    case_id: str,
    series_descriptors: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return MultiSeriesVerificationProfileDescriptorSet(
        case_id=str(case_id),
        series_descriptors=tuple(dict(item) for item in series_descriptors),
    ).to_dict()


def build_prior_verified_context(
    prior_context: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = canonicalise_prior_verified_context(prior_context)
    if canonical is None:
        raise ValueError("prior_context must not be null.")
    return canonical


def build_temporal_consistency_trace(
    *,
    case_id: str,
    profile_name: str,
    profile_revision: str,
    profile_digest: str,
    comparison_status: str,
    comparison_scope: str,
    prior_admissibility: Mapping[str, Any],
    chronology_assessment: Mapping[str, Any],
    geometry_assessment: Mapping[str, Any],
    computed_metrics: Mapping[str, Any],
    flags: list[str] | tuple[str, ...],
    skipped_metrics: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return TemporalConsistencyTrace(
        case_id=str(case_id),
        profile_name=str(profile_name),
        profile_revision=str(profile_revision),
        profile_digest=str(profile_digest),
        comparison_status=str(comparison_status),
        comparison_scope=str(comparison_scope),
        prior_admissibility=dict(prior_admissibility),
        chronology_assessment=dict(chronology_assessment),
        geometry_assessment=dict(geometry_assessment),
        computed_metrics=dict(computed_metrics),
        flags=tuple(str(item) for item in flags),
        skipped_metrics=tuple(dict(item) for item in skipped_metrics),
    ).to_dict()
