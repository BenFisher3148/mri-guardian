"""ICSD-bound MRI verification layer that blocks invented pixels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

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
_VERIFIED_MASK_THRESHOLD = 2.5
_CLINICAL_PLAUSIBILITY_MAX_MASK_FRACTION = 0.35
_CLINICAL_PLAUSIBILITY_MIN_CONTRAST_DELTA = 0.75
_CLINICAL_PLAUSIBILITY_MIN_SLICE_SPAN = 2
_REQUIRED_AUTHORITY_SOURCE_TYPES = frozenset(
    {"scanner_manifest", "voxel_geometry", "model_manifest"}
)


class MriGuardianHarness:
    """Verify MRI AI outputs before forwarding them downstream."""

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
        transition_id: str | UUID | None = None,
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
            transition_id=transition_id,
        )

    def verify_evidence(
        self,
        evidence: EvidenceBundle | Mapping[str, Any],
    ) -> bool:
        return self._transition.verify_transparency(evidence).verified


def build_case_verification_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic verification envelope for one MRI case."""
    analysis = _analyse_case(state)
    manifest = _build_verified_manifest(state, analysis)
    reasons = _build_failure_reasons(state, analysis, manifest)
    tiers = manifest["verification_tiers"]
    return {
        "analysis": analysis,
        "manifest": manifest,
        "verification_tiers": tiers,
        "failure_reasons": reasons,
        "release_eligible": not reasons
        and tiers["level_1"]["status"] == "passed"
        and tiers["level_2"]["status"] == "passed",
    }


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
        report = build_case_verification_report(state)
    except ValueError as exc:
        return (
            InvariantFailure(
                invariant="released_state_matches_mathematical_trace",
                code="analysis_failed",
                message=str(exc),
            ),
        )
    analysis = report["analysis"]
    failures: list[InvariantFailure] = []
    manifest = state.get("verified_manifest")
    expected_manifest = report["manifest"]
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
        for section_name in _release_gate_sections():
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
    ai_mask = np.asarray(arrays["tumor_mask"], dtype=np.int64)
    raw_kspace = raw_real + (1j * raw_imag)
    inverse_fft = np.fft.ifft2(raw_kspace, axes=(-2, -1))
    imaginary_leakage = _max_abs(inverse_fft.imag)
    derived_reconstruction = inverse_fft.real
    reprojection = np.fft.fft2(derived_reconstruction, axes=(-2, -1))
    measurement_residual = reprojection - raw_kspace
    ai_residual = ai_reconstruction - derived_reconstruction
    reconstruction_mismatch_mask = (
        np.abs(ai_reconstruction - derived_reconstruction) > _RECONSTRUCTION_TOLERANCE
    )
    verified_mask = _derive_verified_mask(derived_reconstruction)
    mask_false_positive = (ai_mask == 1) & (verified_mask == 0)
    mask_false_negative = (ai_mask == 0) & (verified_mask == 1)
    mask_mismatch_mask = mask_false_positive | mask_false_negative
    voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    ai_mask_voxel_count = int(np.count_nonzero(ai_mask))
    verified_mask_voxel_count = int(np.count_nonzero(verified_mask))
    computed_tumor_volume_mm3 = _round_float(verified_mask_voxel_count * voxel_volume_mm3)
    derived_reconstruction_list = _float_array_to_list(derived_reconstruction)
    ai_mask_list = ai_mask.tolist()
    verified_mask_list = verified_mask.astype(np.int64).tolist()
    forward_projection_pairs = _complex_array_to_pairs(reprojection)
    raw_hash = hash_canonical_json(
        {
            "raw_real": _float_array_to_list(raw_real),
            "raw_imag": _float_array_to_list(raw_imag),
        }
    )
    derived_reconstruction_hash = hash_canonical_json(derived_reconstruction_list)
    ai_mask_hash = hash_canonical_json(ai_mask_list)
    verified_mask_hash = hash_canonical_json(verified_mask_list)
    return {
        "raw_hash": raw_hash,
        "derived_reconstruction": derived_reconstruction_list,
        "derived_reconstruction_hash": derived_reconstruction_hash,
        "ai_mask": ai_mask_list,
        "ai_mask_hash": ai_mask_hash,
        "ai_mask_voxel_count": ai_mask_voxel_count,
        "verified_mask": verified_mask_list,
        "verified_mask_hash": verified_mask_hash,
        "verified_mask_voxel_count": verified_mask_voxel_count,
        "mask_voxel_delta": int(np.count_nonzero(mask_mismatch_mask)),
        "mask_false_positive_count": int(np.count_nonzero(mask_false_positive)),
        "mask_false_negative_count": int(np.count_nonzero(mask_false_negative)),
        "rejected_region_count": int(np.count_nonzero(mask_false_positive)),
        "forward_projection_hash": hash_canonical_json(forward_projection_pairs),
        "max_measurement_abs_error": _max_abs(measurement_residual),
        "relative_measurement_l2_error": _relative_l2(measurement_residual, raw_kspace),
        "max_ai_abs_error": _max_abs(ai_residual),
        "reconstruction_mismatch_voxel_count": int(np.count_nonzero(reconstruction_mismatch_mask)),
        "max_imaginary_leakage": imaginary_leakage,
        "voxel_volume_mm3": _round_float(voxel_volume_mm3),
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
    mask_verified = analysis["mask_voxel_delta"] == 0
    volume_verified = abs(claimed_volume - analysis["computed_tumor_volume_mm3"]) <= _VOLUME_TOLERANCE
    clinical_plausibility = _build_clinical_plausibility_manifest(analysis)
    manifest = {
        "policy_name": MRI_GUARDIAN_POLICY_NAME,
        "policy_version": MRI_GUARDIAN_POLICY_VERSION,
        "raw_hash": analysis["raw_hash"],
        "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
        "ai_mask_hash": analysis["ai_mask_hash"],
        "verified_mask_hash": analysis["verified_mask_hash"],
        "reconstruction_algorithm": "numpy.ifft2_per_slice.real",
        "mask_verification": {
            "verified": mask_verified,
            "algorithm": "threshold_on_raw_derived_reconstruction",
            "threshold": _VERIFIED_MASK_THRESHOLD,
            "ai_voxel_count": analysis["ai_mask_voxel_count"],
            "verified_voxel_count": analysis["verified_mask_voxel_count"],
            "voxel_delta": analysis["mask_voxel_delta"],
            "false_positive_count": analysis["mask_false_positive_count"],
            "false_negative_count": analysis["mask_false_negative_count"],
        },
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
            "voxel_count": analysis["verified_mask_voxel_count"],
            "voxel_volume_mm3": analysis["voxel_volume_mm3"],
        },
        "no_invented_pixels": {
            "verified": consistency_verified and ai_match_verified,
            "pixel_source": "harness_reconstructed_from_raw_kspace",
            "ai_render_passthrough_allowed": False,
        },
        "clinical_plausibility": clinical_plausibility,
    }
    manifest["verification_tiers"] = _build_verification_tiers(state, manifest)
    return manifest


def _build_clinical_plausibility_manifest(
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    derived_reconstruction = np.asarray(analysis["derived_reconstruction"], dtype=np.float64)
    verified_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    total_voxels = int(verified_mask.size)
    verified_voxels = int(analysis["verified_mask_voxel_count"])
    mask_fraction = 0.0
    if total_voxels:
        mask_fraction = verified_voxels / total_voxels
    heuristics = [
        _heuristic_entry(
            code="mask_fraction_guardrail",
            label="Verified occupancy fraction",
            status=(
                "warning"
                if mask_fraction > _CLINICAL_PLAUSIBILITY_MAX_MASK_FRACTION
                else "passed"
            ),
            message=(
                "The verified mask occupies an unusually large share of the voxel grid "
                f"({mask_fraction:.1%} vs advisory cap {_CLINICAL_PLAUSIBILITY_MAX_MASK_FRACTION:.0%})."
                if mask_fraction > _CLINICAL_PLAUSIBILITY_MAX_MASK_FRACTION
                else (
                    "The verified mask occupies a bounded share of the voxel grid "
                    f"({mask_fraction:.1%})."
                )
            ),
            observed=_round_float(mask_fraction),
            threshold=_round_float(_CLINICAL_PLAUSIBILITY_MAX_MASK_FRACTION),
        ),
    ]

    if verified_voxels and verified_voxels < total_voxels:
        inside_mean = float(np.mean(derived_reconstruction[verified_mask == 1]))
        outside_mean = float(np.mean(derived_reconstruction[verified_mask == 0]))
        contrast_delta = inside_mean - outside_mean
        heuristics.append(
            _heuristic_entry(
                code="inside_outside_contrast",
                label="Inside/outside contrast delta",
                status=(
                    "warning"
                    if contrast_delta < _CLINICAL_PLAUSIBILITY_MIN_CONTRAST_DELTA
                    else "passed"
                ),
                message=(
                    "The verified lesion is not meaningfully brighter than the surrounding "
                    f"background ({_round_float(contrast_delta)} vs advisory floor "
                    f"{_round_float(_CLINICAL_PLAUSIBILITY_MIN_CONTRAST_DELTA)})."
                    if contrast_delta < _CLINICAL_PLAUSIBILITY_MIN_CONTRAST_DELTA
                    else (
                        "The verified lesion remains visually separable from the "
                        f"surrounding background (delta = {_round_float(contrast_delta)})."
                    )
                ),
                observed=_round_float(contrast_delta),
                threshold=_round_float(_CLINICAL_PLAUSIBILITY_MIN_CONTRAST_DELTA),
            )
        )
    else:
        heuristics.append(
            _heuristic_entry(
                code="inside_outside_contrast",
                label="Inside/outside contrast delta",
                status="not_evaluated",
                message=(
                    "Contrast heuristics were not evaluated because the verified mask "
                    "does not leave both lesion and background voxels to compare."
                ),
            )
        )

    slice_span = int(np.count_nonzero(np.any(verified_mask == 1, axis=(1, 2))))
    heuristics.append(
        _heuristic_entry(
            code="slice_span",
            label="Slice span",
            status=(
                "warning"
                if verified_voxels and slice_span < _CLINICAL_PLAUSIBILITY_MIN_SLICE_SPAN
                else "passed"
            ),
            message=(
                "The verified lesion only appears on "
                f"{slice_span} slice(s), below the advisory floor of "
                f"{_CLINICAL_PLAUSIBILITY_MIN_SLICE_SPAN}."
                if verified_voxels and slice_span < _CLINICAL_PLAUSIBILITY_MIN_SLICE_SPAN
                else f"The verified lesion spans {slice_span} slice(s)."
            ),
            observed=slice_span,
            threshold=_CLINICAL_PLAUSIBILITY_MIN_SLICE_SPAN,
        )
    )

    warnings = [
        {
            "code": entry["code"],
            "label": entry["label"],
            "message": entry["message"],
        }
        for entry in heuristics
        if entry["status"] == "warning"
    ]
    status = "warning" if warnings else "passed"
    return {
        "advisory_only": True,
        "status": status,
        "summary": (
            f"{len(warnings)} advisory clinical plausibility warning(s) detected."
            if warnings
            else "No advisory clinical plausibility warnings were detected."
        ),
        "warning_count": len(warnings),
        "warnings": warnings,
        "heuristics": heuristics,
    }


def _build_verification_tiers(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    structural_checks = _build_structural_validity_checks(state)
    structural_status = _tier_status_from_checks(structural_checks)
    physical_checks = _build_physical_consistency_checks(manifest)
    physical_status = _tier_status_from_checks(physical_checks)
    clinical_section = manifest["clinical_plausibility"]
    clinical_checks = [
        {
            "code": entry["code"],
            "label": entry["label"],
            "status": entry["status"],
            "message": entry["message"],
        }
        for entry in clinical_section["heuristics"]
    ]
    return {
        "level_1": {
            "level": 1,
            "title": "Level 1: Structural Validity",
            "subtitle": "ICSD envelope",
            "blocking": True,
            "status": structural_status,
            "summary": (
                "ICSD structural checks passed for identifiers, numeric fields, voxel geometry, "
                "and release payload shape."
                if structural_status == "passed"
                else "ICSD structural checks failed before mathematical verification can be trusted."
            ),
            "checks": structural_checks,
        },
        "level_2": {
            "level": 2,
            "title": "Level 2: Physical Consistency",
            "subtitle": "k-space reconstruction",
            "blocking": True,
            "status": physical_status,
            "summary": (
                "Raw k-space, deterministic reconstruction, verified mask, and tumour volume "
                "all agree."
                if physical_status == "passed"
                else "Raw-derived physical consistency failed; the AI claim diverges from measured reality."
            ),
            "checks": physical_checks,
        },
        "level_3": {
            "level": 3,
            "title": "Level 3: Clinical Plausibility",
            "subtitle": "Optional heuristics",
            "blocking": False,
            "status": clinical_section["status"],
            "summary": clinical_section["summary"],
            "checks": clinical_checks,
        },
    }


def _build_structural_validity_checks(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        _invariant_check_result(
            code="required_strings_present",
            label="Required identifiers",
            failure=_check_required_strings(state),
            success_message="Required identifiers are present and non-empty.",
        ),
        _invariant_check_result(
            code="release_status_known",
            label="Release status",
            failure=_check_release_status(state),
            success_message="release_status is a known guardian state.",
        ),
    ]
    checks.extend(_numeric_field_checks(state))
    checks.extend(
        [
            _invariant_check_result(
                code="geometry_consistent",
                label="Voxel grid geometry",
                failure=_check_geometry_consistent(state),
                success_message="Raw k-space, AI reconstruction, and tumour mask share one voxel grid.",
            ),
            _invariant_check_result(
                code="voxel_spacing_positive",
                label="Voxel spacing",
                failure=_check_voxel_spacing_positive(state),
                success_message="voxel_spacing_mm is positive and shaped for a 3D study.",
            ),
            _invariant_check_result(
                code="release_payloads_guarded",
                label="Release payload contract",
                failure=_check_release_payload_contract(state),
                success_message="The state shape matches its pending or released contract.",
            ),
        ]
    )
    return checks


def _numeric_field_checks(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = []
    field_specs = (
        ("raw_real", "Raw real component", False),
        ("raw_imag", "Raw imaginary component", False),
        ("ai_reconstruction", "AI reconstruction tensor", False),
        ("tumor_mask", "Tumour mask tensor", True),
    )
    for field_name, label, binary_only in field_specs:
        checks.append(
            _numeric_field_check(
                state=state,
                field_name=field_name,
                label=label,
                binary_only=binary_only,
            )
        )
    return checks


def _numeric_field_check(
    *,
    state: Mapping[str, Any],
    field_name: str,
    label: str,
    binary_only: bool,
) -> dict[str, Any]:
    value = state.get(field_name)
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        return {
            "code": field_name,
            "label": label,
            "status": "failed",
            "message": (
                f"{field_name!r} could not be cast into a numeric array "
                f"({type(exc).__name__})."
            ),
        }
    if array.ndim != 3:
        return {
            "code": field_name,
            "label": label,
            "status": "failed",
            "message": f"{field_name!r} must be a non-empty 3D numeric tensor.",
        }
    if array.size == 0:
        return {
            "code": field_name,
            "label": label,
            "status": "failed",
            "message": f"{field_name!r} must not be empty.",
        }
    if not np.isfinite(array).all():
        return {
            "code": field_name,
            "label": label,
            "status": "failed",
            "message": f"{field_name!r} contains non-finite values.",
        }
    if binary_only and not np.isin(array, (0.0, 1.0)).all():
        return {
            "code": field_name,
            "label": label,
            "status": "failed",
            "message": f"{field_name!r} must contain only 0/1 mask values.",
        }
    return {
        "code": field_name,
        "label": label,
        "status": "passed",
        "message": f"{field_name!r} is a finite 3D tensor.",
    }


def _build_physical_consistency_checks(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _physical_check_result(
            code="measurement_consistency",
            label="Raw k-space re-projection",
            verified=manifest["measurement_consistency"]["verified"],
            success_message=(
                "The deterministic reconstruction reprojects back to the supplied raw k-space."
            ),
            failure_message=(
                "The deterministic reconstruction does not cleanly reproject back to the "
                "supplied raw k-space."
            ),
        ),
        _physical_check_result(
            code="ai_reconstruction_match",
            label="AI reconstruction parity",
            verified=manifest["ai_reconstruction_match"]["verified"],
            success_message=(
                "The AI reconstruction matches the deterministic reconstruction on the measured grid."
            ),
            failure_message=(
                "The AI reconstruction diverges from the deterministic reconstruction."
            ),
        ),
        _physical_check_result(
            code="mask_verification",
            label="Verified mask parity",
            verified=manifest["mask_verification"]["verified"],
            success_message="The AI tumour mask matches the deterministic verified mask.",
            failure_message=(
                "The AI tumour mask diverges from the deterministic verified mask."
            ),
        ),
        _physical_check_result(
            code="tumor_volume_proof",
            label="Tumour volume proof",
            verified=manifest["tumor_volume_proof"]["verified"],
            success_message=(
                "The submitted tumour volume equals the verified voxel count times voxel spacing."
            ),
            failure_message=(
                "The submitted tumour volume does not match the verified voxel geometry."
            ),
        ),
        _physical_check_result(
            code="no_invented_pixels",
            label="Invented pixel guard",
            verified=manifest["no_invented_pixels"]["verified"],
            success_message="No AI-supplied pixels bypass the raw-derived verification-layer path.",
            failure_message=(
                "The downstream PACS/workstation path is no longer fully explained "
                "by raw-derived pixels."
            ),
        ),
    ]


def _physical_check_result(
    *,
    code: str,
    label: str,
    verified: bool,
    success_message: str,
    failure_message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "status": "passed" if verified else "failed",
        "message": success_message if verified else failure_message,
    }


def _invariant_check_result(
    *,
    code: str,
    label: str,
    failure: InvariantFailure | None,
    success_message: str,
) -> dict[str, Any]:
    if failure is None:
        return {
            "code": code,
            "label": label,
            "status": "passed",
            "message": success_message,
        }
    return {
        "code": code,
        "label": label,
        "status": "failed",
        "message": failure.message,
    }


def _tier_status_from_checks(checks: list[dict[str, Any]]) -> str:
    for check in checks:
        if check["status"] == "failed":
            return "failed"
    return "passed"


def _heuristic_entry(
    *,
    code: str,
    label: str,
    status: str,
    message: str,
    observed: float | int | None = None,
    threshold: float | int | None = None,
) -> dict[str, Any]:
    entry = {
        "code": code,
        "label": label,
        "status": status,
        "message": message,
    }
    if observed is not None:
        entry["observed"] = observed
    if threshold is not None:
        entry["threshold"] = threshold
    return entry


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
        "overlay_mode": "policy_verified_mask_on_measured_grid",
        "reconstruction_hash": analysis["derived_reconstruction_hash"],
        "verified_mask_hash": analysis["verified_mask_hash"],
        "tumor_volume_mm3": analysis["computed_tumor_volume_mm3"],
        "reconstruction": analysis["derived_reconstruction"],
        "verified_mask": analysis["verified_mask"],
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


def _release_gate_sections() -> tuple[str, ...]:
    return (
        "measurement_consistency",
        "ai_reconstruction_match",
        "mask_verification",
        "tumor_volume_proof",
        "no_invented_pixels",
    )


def _build_failure_reasons(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    mask_delta = int(analysis["mask_voxel_delta"])
    if mask_delta:
        reasons.append(
            {
                "code": "mask_voxel_mismatch",
                "summary": f"mask_voxel_mismatch (delta = {mask_delta} voxels)",
                "delta": mask_delta,
                "message": (
                    "The AI tumour mask does not match the policy-derived verified mask "
                    "computed from the raw-derived reconstruction."
                ),
                "details": {
                    "false_positive_count": int(analysis["mask_false_positive_count"]),
                    "false_negative_count": int(analysis["mask_false_negative_count"]),
                },
            }
        )
    intensity_delta = int(analysis["reconstruction_mismatch_voxel_count"])
    if intensity_delta:
        reasons.append(
            {
                "code": "intensity_manipulation_detected",
                "summary": (
                    "intensity_manipulation_detected "
                    f"(delta = {intensity_delta} voxels; max_abs_error = {analysis['max_ai_abs_error']})"
                ),
                "delta": intensity_delta,
                "message": (
                    "The AI reconstruction no longer matches the deterministic reconstruction "
                    "recomputed from raw k-space."
                ),
                "details": {
                    "max_abs_error": analysis["max_ai_abs_error"],
                },
            }
        )
    if not manifest["tumor_volume_proof"]["verified"]:
        claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
        verified_volume = manifest["tumor_volume_proof"]["computed_mm3"]
        reasons.append(
            {
                "code": "tumor_volume_mismatch",
                "summary": (
                    "tumor_volume_mismatch "
                    f"(claimed = {claimed_volume} mm3; verified = {verified_volume} mm3)"
                ),
                "delta": _round_float(abs(claimed_volume - verified_volume)),
                "message": "The submitted tumour volume does not match the verified mask volume.",
                "details": {
                    "claimed_mm3": claimed_volume,
                    "verified_mm3": verified_volume,
                },
            }
        )
    if not manifest["measurement_consistency"]["verified"]:
        reasons.append(
            {
                "code": "raw_measurement_inconsistency",
                "summary": (
                    "raw_measurement_inconsistency "
                    f"(max |Δ| = {analysis['max_measurement_abs_error']})"
                ),
                "delta": analysis["max_measurement_abs_error"],
                "message": (
                    "The deterministic reconstruction cannot be cleanly reprojected back "
                    "to the supplied raw measurements."
                ),
                "details": {
                    "relative_l2_error": analysis["relative_measurement_l2_error"],
                },
            }
        )
    return reasons


def _derive_verified_mask(derived_reconstruction: np.ndarray) -> np.ndarray:
    reconstruction = np.asarray(derived_reconstruction, dtype=np.float64)
    return (reconstruction >= _VERIFIED_MASK_THRESHOLD).astype(np.int64)


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
