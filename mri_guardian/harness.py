"""ICSD-bound MRI release harness that blocks invented pixels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from icsd.adapters.numba_invariants import NumbaBatchFieldInvariant
from icsd.core.errors import InvariantFailure
from icsd.core.evidence import AuthoritySource, EvidenceBundle
from icsd.core.invariants import Invariant, InvariantProfile, InvariantProfileRegistry, InvariantSet
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json
from icsd.core.transition import Transition, TransitionResult

from .transparency import JsonlHashChainTransparencyLog

MRI_GUARDIAN_POLICY_NAME = "mri_guardian_release"
MRI_GUARDIAN_POLICY = "policy/mri-guardian"
MRI_GUARDIAN_POLICY_VERSION = "2026.03"
RELEASE_STATUS_PENDING = "pending_verification"
RELEASE_STATUS_RELEASED = "released_to_radiologist"
_RECONSTRUCTION_TOLERANCE = 1e-9
_IMAGINARY_LEAK_TOLERANCE = 1e-9
_VOLUME_TOLERANCE = 1e-9
_REQUIRED_AUTHORITY_SOURCE_TYPES = frozenset(
    {"scanner_manifest", "voxel_geometry", "model_manifest"}
)


class MriGuardianHarness:
    """Release MRI AI outputs only when raw-data derivation is provable."""

    def __init__(self, *, transparency_log_path: Path | str) -> None:
        self._transition = _build_release_transition(
            transparency_log_path=Path(transparency_log_path),
        )

    def release_case(
        self,
        *,
        state: Mapping[str, Any],
        actor: str,
        authority_sources: Iterable[AuthoritySource | Mapping[str, Any]],
        timestamp: str | None = None,
    ) -> TransitionResult:
        return self._transition.apply(
            state=state,
            entity_type="mri_case",
            entity_id=str(state["case_id"]),
            actor=actor,
            authority_sources=authority_sources,
            profile=MRI_GUARDIAN_POLICY_NAME,
            policy_version=MRI_GUARDIAN_POLICY_VERSION,
            timestamp=timestamp,
        )

    def verify_evidence(
        self,
        evidence: EvidenceBundle | Mapping[str, Any],
    ) -> bool:
        return self._transition.verify_transparency(evidence).verified


def build_demo_case() -> dict[str, Any]:
    """Return a deterministic happy-path MRI case."""
    reconstruction = np.asarray(
        [
            [[0.0, 1.0], [0.0, 4.0]],
            [[0.0, 3.0], [2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    tumor_mask = np.asarray(
        [
            [[0, 0], [0, 1]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.int64,
    )
    voxel_spacing_mm = np.asarray([0.8, 0.8, 2.5], dtype=np.float64)
    raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
    tumor_volume_mm3 = _round_float(int(tumor_mask.sum()) * float(np.prod(voxel_spacing_mm)))
    return {
        "case_id": "mri-case-0001",
        "patient_pseudonym": "anon-0001",
        "study_uid": "study-0001",
        "series_uid": "series-0001",
        "scanner_id": "scanner-7t-alpha",
        "sequence_id": "t1-post-contrast",
        "ai_model_id": "raw-mri-ai-v4",
        "release_status": RELEASE_STATUS_PENDING,
        "raw_real": _float_array_to_list(raw_kspace.real),
        "raw_imag": _float_array_to_list(raw_kspace.imag),
        "ai_reconstruction": _float_array_to_list(reconstruction),
        "tumor_mask": tumor_mask.tolist(),
        "voxel_spacing_mm": _float_array_to_list(voxel_spacing_mm),
        "tumor_volume_mm3": tumor_volume_mm3,
        "verified_manifest": None,
        "workstation_payload": None,
    }


def build_demo_authority_sources(*, case_id: str) -> tuple[AuthoritySource, ...]:
    """Return authority anchors required by the demo policy profile."""
    return (
        AuthoritySource(
            source_type="scanner_manifest",
            source_id="scanner-7t-alpha",
            reference="scanner://scanner-7t-alpha/calibration/2026-03-01",
            details={"manufacturer": "ExampleMRI", "field_strength_t": 7.0},
        ),
        AuthoritySource(
            source_type="voxel_geometry",
            source_id=f"{case_id}-geometry",
            reference=f"geometry://{case_id}/voxel-grid",
            details={"spacing_mm": [0.8, 0.8, 2.5], "shape": [2, 2, 2]},
        ),
        AuthoritySource(
            source_type="model_manifest",
            source_id="raw-mri-ai-v4",
            reference="model://raw-mri-ai-v4/reconstruction-segmentation",
            details={"task": "reconstruction-and-volume-estimation"},
        ),
    )


def _build_release_transition(*, transparency_log_path: Path) -> Transition:
    invariants = InvariantSet(
        (
            Invariant(
                name="required_strings_present",
                check=_check_required_strings,
            ),
            Invariant(
                name="release_status_known",
                check=_check_release_status,
            ),
            Invariant(
                name="raw_real_numeric",
                check=NumbaBatchFieldInvariant(
                    field="raw_real",
                    name="raw_real_numeric",
                    expected_ndim=3,
                    require_finite=True,
                    allow_empty=False,
                ),
            ),
            Invariant(
                name="raw_imag_numeric",
                check=NumbaBatchFieldInvariant(
                    field="raw_imag",
                    name="raw_imag_numeric",
                    expected_ndim=3,
                    require_finite=True,
                    allow_empty=False,
                ),
            ),
            Invariant(
                name="ai_reconstruction_numeric",
                check=NumbaBatchFieldInvariant(
                    field="ai_reconstruction",
                    name="ai_reconstruction_numeric",
                    expected_ndim=3,
                    require_finite=True,
                    allow_empty=False,
                ),
            ),
            Invariant(
                name="tumor_mask_numeric",
                check=NumbaBatchFieldInvariant(
                    field="tumor_mask",
                    name="tumor_mask_numeric",
                    expected_ndim=3,
                    min_value=0,
                    max_value=1,
                    require_finite=True,
                    allow_empty=False,
                ),
            ),
            Invariant(
                name="geometry_consistent",
                check=_check_geometry_consistent,
            ),
            Invariant(
                name="voxel_spacing_positive",
                check=_check_voxel_spacing_positive,
            ),
            Invariant(
                name="release_payloads_guarded",
                check=_check_release_payload_contract,
            ),
            Invariant(
                name="released_state_matches_mathematical_trace",
                check=_check_released_state_matches_trace,
            ),
        )
    )
    profile = InvariantProfile(
        name=MRI_GUARDIAN_POLICY_NAME,
        policy=MRI_GUARDIAN_POLICY,
        policy_version=MRI_GUARDIAN_POLICY_VERSION,
        invariants=invariants,
    )
    return Transition(
        name="release_verified_mri_estimate",
        mutate=_release_to_radiologist,
        profile_registry=InvariantProfileRegistry((profile,)),
        default_profile=MRI_GUARDIAN_POLICY_NAME,
        authority_validator=_validate_authority_sources,
        transparency_log=JsonlHashChainTransparencyLog(transparency_log_path),
        verify_transparency_after_append=True,
    )


def _release_to_radiologist(state: dict[str, Any]) -> None:
    analysis = _analyse_case(state)
    state["verified_manifest"] = _build_verified_manifest(state, analysis)
    state["workstation_payload"] = _build_workstation_payload(state, analysis)
    state["release_status"] = RELEASE_STATUS_RELEASED


def _validate_authority_sources(
    authority: str,
    authority_sources: tuple[AuthoritySource, ...],
) -> None:
    seen_types = {source.source_type for source in authority_sources}
    missing_types = sorted(_REQUIRED_AUTHORITY_SOURCE_TYPES - seen_types)
    if missing_types:
        missing = ", ".join(missing_types)
        msg = (
            f"authority validation failed for {authority!r}: "
            f"missing authority source types {missing}."
        )
        raise ValueError(msg)


def _check_required_strings(state: Mapping[str, Any]) -> InvariantFailure | None:
    required_fields = (
        "case_id",
        "patient_pseudonym",
        "study_uid",
        "series_uid",
        "scanner_id",
        "sequence_id",
        "ai_model_id",
    )
    for field_name in required_fields:
        value = state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return InvariantFailure(
                invariant="required_strings_present",
                code="missing_string_field",
                message=f"Field {field_name!r} must be a non-empty string.",
                details={"field": field_name},
            )
    return None


def _check_release_status(state: Mapping[str, Any]) -> InvariantFailure | None:
    value = state.get("release_status")
    if value in {RELEASE_STATUS_PENDING, RELEASE_STATUS_RELEASED}:
        return None
    return InvariantFailure(
        invariant="release_status_known",
        code="invalid_release_status",
        message="release_status must be a known MRI guardian state.",
        details={"release_status": value},
    )


def _check_geometry_consistent(state: Mapping[str, Any]) -> InvariantFailure | None:
    arrays, failure = _extract_arrays(state)
    if failure is not None:
        return failure
    raw_real = arrays["raw_real"]
    raw_imag = arrays["raw_imag"]
    ai_reconstruction = arrays["ai_reconstruction"]
    tumor_mask = arrays["tumor_mask"]
    if raw_real.shape != raw_imag.shape:
        return InvariantFailure(
            invariant="geometry_consistent",
            code="raw_component_shape_mismatch",
            message="raw_real and raw_imag must have identical shapes.",
            details={
                "raw_real_shape": list(raw_real.shape),
                "raw_imag_shape": list(raw_imag.shape),
            },
        )
    if ai_reconstruction.shape != tumor_mask.shape:
        return InvariantFailure(
            invariant="geometry_consistent",
            code="ai_mask_shape_mismatch",
            message="ai_reconstruction and tumor_mask must share the same voxel grid.",
            details={
                "ai_reconstruction_shape": list(ai_reconstruction.shape),
                "tumor_mask_shape": list(tumor_mask.shape),
            },
        )
    if raw_real.shape != ai_reconstruction.shape:
        return InvariantFailure(
            invariant="geometry_consistent",
            code="reconstruction_grid_mismatch",
            message="AI reconstruction grid must match the raw MRI k-space grid.",
            details={
                "raw_shape": list(raw_real.shape),
                "ai_reconstruction_shape": list(ai_reconstruction.shape),
            },
        )
    return None


def _check_voxel_spacing_positive(state: Mapping[str, Any]) -> InvariantFailure | None:
    spacing = state.get("voxel_spacing_mm")
    if not isinstance(spacing, list) or len(spacing) != 3:
        return InvariantFailure(
            invariant="voxel_spacing_positive",
            code="invalid_spacing_shape",
            message="voxel_spacing_mm must be a 3-element list.",
            details={"voxel_spacing_mm": spacing},
        )
    if any(not isinstance(value, (int, float)) or value <= 0 for value in spacing):
        return InvariantFailure(
            invariant="voxel_spacing_positive",
            code="non_positive_spacing",
            message="voxel_spacing_mm values must all be positive numbers.",
            details={"voxel_spacing_mm": spacing},
        )
    return None


def _check_release_payload_contract(state: Mapping[str, Any]) -> InvariantFailure | None:
    status = state.get("release_status")
    manifest = state.get("verified_manifest")
    workstation_payload = state.get("workstation_payload")
    if status == RELEASE_STATUS_PENDING:
        if manifest is None and workstation_payload is None:
            return None
        return InvariantFailure(
            invariant="release_payloads_guarded",
            code="pending_state_contains_release_material",
            message="Pending cases must not expose verified_manifest or workstation_payload.",
        )
    if isinstance(manifest, Mapping) and isinstance(workstation_payload, Mapping):
        return None
    return InvariantFailure(
        invariant="release_payloads_guarded",
        code="released_state_missing_material",
        message="Released cases must contain verified_manifest and workstation_payload.",
    )


def _check_released_state_matches_trace(
    state: Mapping[str, Any],
) -> tuple[InvariantFailure, ...] | None:
    if state.get("release_status") != RELEASE_STATUS_RELEASED:
        return None
    try:
        analysis = _analyse_case(state)
    except ValueError as exc:
        return (
            InvariantFailure(
                invariant="released_state_matches_mathematical_trace",
                code="analysis_failed",
                message=str(exc),
            ),
        )
    failures: list[InvariantFailure] = []
    manifest = state.get("verified_manifest")
    expected_manifest = _build_verified_manifest(state, analysis)
    if manifest != expected_manifest:
        failures.append(
            InvariantFailure(
                invariant="released_state_matches_mathematical_trace",
                code="manifest_mismatch",
                message="verified_manifest does not match the expected mathematical trace.",
                details={
                    "expected_hash": hash_canonical_json(expected_manifest),
                    "actual_hash": hash_canonical_json(manifest),
                },
            )
        )
    else:
        proof_sections = (
            "measurement_consistency",
            "ai_reconstruction_match",
            "tumor_volume_proof",
            "no_invented_pixels",
        )
        for section_name in proof_sections:
            section = expected_manifest[section_name]
            if section["verified"]:
                continue
            failures.append(
                InvariantFailure(
                    invariant="released_state_matches_mathematical_trace",
                    code="release_proof_failed",
                    message=(
                        "Released cases are only representable when every proof section "
                        f"verifies successfully; {section_name!r} failed."
                    ),
                    details={"section": section_name, "section_payload": section},
                )
            )
    workstation_payload = state.get("workstation_payload")
    expected_workstation_payload = _build_workstation_payload(state, analysis)
    if workstation_payload != expected_workstation_payload:
        failures.append(
            InvariantFailure(
                invariant="released_state_matches_mathematical_trace",
                code="workstation_payload_mismatch",
                message=(
                    "workstation_payload does not match the harness-derived reconstruction "
                    "and mask overlay."
                ),
                details={
                    "expected_hash": hash_canonical_json(expected_workstation_payload),
                    "actual_hash": hash_canonical_json(workstation_payload),
                },
            )
        )
    return tuple(failures) or None


def _analyse_case(state: Mapping[str, Any]) -> dict[str, Any]:
    arrays, failure = _extract_arrays(state)
    if failure is not None:
        raise ValueError(failure.message)
    raw_real = arrays["raw_real"]
    raw_imag = arrays["raw_imag"]
    ai_reconstruction = arrays["ai_reconstruction"]
    tumor_mask = arrays["tumor_mask"]
    raw_kspace = raw_real + (1j * raw_imag)
    inverse_fft = np.fft.ifft2(raw_kspace, axes=(-2, -1))
    imaginary_leakage = _max_abs(inverse_fft.imag)
    derived_reconstruction = inverse_fft.real
    reprojection = np.fft.fft2(derived_reconstruction, axes=(-2, -1))
    measurement_residual = reprojection - raw_kspace
    ai_residual = ai_reconstruction - derived_reconstruction
    voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    tumor_voxel_count = int(np.count_nonzero(tumor_mask))
    computed_tumor_volume_mm3 = _round_float(tumor_voxel_count * voxel_volume_mm3)
    derived_reconstruction_list = _float_array_to_list(derived_reconstruction)
    tumor_mask_list = np.asarray(tumor_mask, dtype=np.int64).tolist()
    forward_projection_pairs = _complex_array_to_pairs(reprojection)
    raw_hash = hash_canonical_json(
        {
            "raw_real": _float_array_to_list(raw_real),
            "raw_imag": _float_array_to_list(raw_imag),
        }
    )
    derived_reconstruction_hash = hash_canonical_json(derived_reconstruction_list)
    tumor_mask_hash = hash_canonical_json(tumor_mask_list)
    return {
        "raw_hash": raw_hash,
        "derived_reconstruction": derived_reconstruction_list,
        "derived_reconstruction_hash": derived_reconstruction_hash,
        "tumor_mask": tumor_mask_list,
        "tumor_mask_hash": tumor_mask_hash,
        "forward_projection_hash": hash_canonical_json(forward_projection_pairs),
        "max_measurement_abs_error": _max_abs(measurement_residual),
        "relative_measurement_l2_error": _relative_l2(measurement_residual, raw_kspace),
        "max_ai_abs_error": _max_abs(ai_residual),
        "max_imaginary_leakage": imaginary_leakage,
        "voxel_volume_mm3": _round_float(voxel_volume_mm3),
        "tumor_voxel_count": tumor_voxel_count,
        "computed_tumor_volume_mm3": computed_tumor_volume_mm3,
    }


def _build_verified_manifest(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
    consistency_verified = (
        analysis["max_measurement_abs_error"] <= _RECONSTRUCTION_TOLERANCE
        and analysis["max_imaginary_leakage"] <= _IMAGINARY_LEAK_TOLERANCE
    )
    ai_match_verified = analysis["max_ai_abs_error"] <= _RECONSTRUCTION_TOLERANCE
    volume_verified = abs(claimed_volume - analysis["computed_tumor_volume_mm3"]) <= _VOLUME_TOLERANCE
    return {
        "policy_name": MRI_GUARDIAN_POLICY_NAME,
        "policy_version": MRI_GUARDIAN_POLICY_VERSION,
        "raw_hash": analysis["raw_hash"],
        "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
        "tumor_mask_hash": analysis["tumor_mask_hash"],
        "reconstruction_algorithm": "numpy.ifft2_per_slice.real",
        "measurement_consistency": {
            "verified": consistency_verified,
            "max_abs_error": analysis["max_measurement_abs_error"],
            "relative_l2_error": analysis["relative_measurement_l2_error"],
            "max_imaginary_leakage": analysis["max_imaginary_leakage"],
            "tolerance": _RECONSTRUCTION_TOLERANCE,
            "forward_projection_hash": analysis["forward_projection_hash"],
        },
        "ai_reconstruction_match": {
            "verified": ai_match_verified,
            "max_abs_error": analysis["max_ai_abs_error"],
            "tolerance": _RECONSTRUCTION_TOLERANCE,
        },
        "tumor_volume_proof": {
            "verified": volume_verified,
            "claimed_mm3": claimed_volume,
            "computed_mm3": analysis["computed_tumor_volume_mm3"],
            "voxel_count": analysis["tumor_voxel_count"],
            "voxel_volume_mm3": analysis["voxel_volume_mm3"],
        },
        "no_invented_pixels": {
            "verified": consistency_verified and ai_match_verified,
            "pixel_source": "harness_reconstructed_from_raw_kspace",
            "ai_render_passthrough_allowed": False,
        },
    }


def _build_workstation_payload(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": state["case_id"],
        "study_uid": state["study_uid"],
        "series_uid": state["series_uid"],
        "scanner_id": state["scanner_id"],
        "sequence_id": state["sequence_id"],
        "patient_pseudonym": state["patient_pseudonym"],
        "pixel_source": "harness_reconstructed_from_raw_kspace",
        "overlay_mode": "binary_mask_on_measured_grid",
        "reconstruction_hash": analysis["derived_reconstruction_hash"],
        "tumor_mask_hash": analysis["tumor_mask_hash"],
        "tumor_volume_mm3": analysis["computed_tumor_volume_mm3"],
        "reconstruction": analysis["derived_reconstruction"],
        "tumor_mask": analysis["tumor_mask"],
    }


def _extract_arrays(
    state: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], InvariantFailure | None]:
    field_types = {
        "raw_real": np.float64,
        "raw_imag": np.float64,
        "ai_reconstruction": np.float64,
        "tumor_mask": np.float64,
    }
    arrays: dict[str, np.ndarray] = {}
    for field_name, dtype in field_types.items():
        value = state.get(field_name)
        try:
            arrays[field_name] = np.asarray(value, dtype=dtype)
        except Exception as exc:
            return {}, InvariantFailure(
                invariant="geometry_consistent",
                code="array_cast_failed",
                message=f"Unable to cast field {field_name!r} into a numeric array.",
                details={"field": field_name, "error_type": type(exc).__name__},
            )
    return arrays, None


def _float_array_to_list(values: np.ndarray | list[float] | list[list[float]]) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return _round_float(float(array))
    return [_float_array_to_list(item) for item in array]


def _complex_array_to_pairs(values: np.ndarray) -> Any:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 0:
        return [_round_float(float(array.real)), _round_float(float(array.imag))]
    return [_complex_array_to_pairs(item) for item in array]


def _round_float(value: float) -> float:
    if abs(value) < 1e-12:
        return 0.0
    return round(float(value), 9)


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return _round_float(float(np.max(np.abs(array))))


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(reference.ravel()))
    if reference_norm == 0:
        return 0.0
    return _round_float(float(np.linalg.norm(delta.ravel()) / reference_norm))


def render_release_bundle_json(result: TransitionResult) -> str:
    """Return the released workstation payload in canonical JSON."""
    return canonical_json_dumps(result.after_state["workstation_payload"])
