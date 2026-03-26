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
from icsd.core.invariants import Invariant, InvariantSet
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json
from icsd.core.transition import Transition, TransitionResult

from .contracts import (
    build_ai_claim_envelope,
    build_mask_derivation_policy,
    build_verification_profile_contract,
    build_verification_tolerance_policy,
    build_verified_reconstruction_trace,
)
from .profiles import (
    DECISION_CLASS_TRIAGE_ONLY,
    DEFAULT_PROFILE_NAME,
    HEURISTIC_BLOCKING,
    build_authority_requirement_map,
    build_profile_registry,
    default_icsd_policy_version,
    get_verification_profile,
    resolve_profile_name,
)
from .transparency import JsonlHashChainTransparencyLog

MRI_GUARDIAN_POLICY_NAME = DEFAULT_PROFILE_NAME
MRI_GUARDIAN_POLICY_VERSION = default_icsd_policy_version()
RELEASE_STATUS_PENDING = "pending_verification"
RELEASE_STATUS_RELEASED = "released_to_radiologist"
_AUTHORITY_REQUIREMENT_MAP = build_authority_requirement_map()


class MriGuardianHarness:
    """Verify MRI AI outputs before forwarding them downstream."""

    def __init__(
        self,
        *,
        transparency_log_path: Path | str,
        default_profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> None:
        self._default_profile_name = get_verification_profile(default_profile_name).profile_name
        self._transition = _build_release_transition(
            transparency_log_path=Path(transparency_log_path),
            default_profile_name=self._default_profile_name,
        )

    def release_case(
        self,
        *,
        state: Mapping[str, Any],
        actor: str,
        authority_sources: Iterable[AuthoritySource | Mapping[str, Any]],
        profile_name: str | None = None,
        timestamp: str | None = None,
        transition_id: str | UUID | None = None,
    ) -> TransitionResult:
        resolved_profile_name = resolve_profile_name(profile_name=profile_name, state=dict(state))
        descriptor = get_verification_profile(resolved_profile_name)
        if not descriptor.release_semantics["final_release_allowed"]:
            raise ValueError(
                f"Runtime profile {descriptor.profile_name!r} is non-final and cannot release a case."
            )
        working_state = dict(state)
        working_state["verification_profile_descriptor"] = build_verification_profile_contract(
            descriptor.profile_name
        )
        return self._transition.apply(
            state=working_state,
            entity_type="mri_case",
            entity_id=str(working_state["case_id"]),
            actor=actor,
            authority_sources=authority_sources,
            profile=descriptor.profile_name,
            policy_version=descriptor.icsd_policy_version,
            timestamp=timestamp,
            transition_id=transition_id,
        )

    def verify_evidence(
        self,
        evidence: EvidenceBundle | Mapping[str, Any],
    ) -> bool:
        return self._transition.verify_transparency(evidence).verified

def build_case_verification_report(
    state: Mapping[str, Any],
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Return the deterministic verification envelope for one MRI case."""
    resolved_profile_name = resolve_profile_name(profile_name=profile_name, state=dict(state))
    descriptor = get_verification_profile(resolved_profile_name)
    profile_contract = build_verification_profile_contract(descriptor.profile_name)
    analysis = _analyse_case(state, descriptor)
    contracts = _build_claim_to_proof_contracts(state, analysis, descriptor)
    manifest = _build_verified_manifest(state, analysis, contracts, descriptor)
    reasons = _build_failure_reasons(state, analysis, manifest)
    tiers = manifest["verification_tiers"]
    decision_summary = manifest["decision_summary"]
    return {
        "profile": profile_contract,
        "analysis": analysis,
        "contracts": contracts,
        "manifest": manifest,
        "verification_tiers": tiers,
        "failure_reasons": reasons,
        "decision_summary": decision_summary,
        "release_eligible": bool(decision_summary["final_release_allowed"]),
    }


def build_demo_case(*, profile_name: str = DEFAULT_PROFILE_NAME) -> dict[str, Any]:
    """Return a deterministic happy-path MRI case."""
    descriptor = get_verification_profile(profile_name)
    reconstruction = np.asarray(
        [
            [[0.0, 1.0], [0.0, 4.0]],
            [[0.0, 3.0], [2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    tumor_mask = (reconstruction >= descriptor.mask_threshold).astype(np.int64)
    voxel_spacing_mm = np.asarray([0.8, 0.8, 2.5], dtype=np.float64)
    raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
    tumor_volume_mm3 = _round_float(int(tumor_mask.sum()) * float(np.prod(voxel_spacing_mm)))
    return {
        "case_id": "mri-case-0001",
        "patient_pseudonym": "anon-0001",
        "study_uid": "study-0001",
        "series_uid": "series-0001",
        "scanner_id": "scanner-7t-alpha",
        "sequence_id": _demo_sequence_id(descriptor.profile_name),
        "ai_model_id": "raw-mri-ai-v4",
        "release_status": RELEASE_STATUS_PENDING,
        "raw_real": _float_array_to_list(raw_kspace.real),
        "raw_imag": _float_array_to_list(raw_kspace.imag),
        "ai_reconstruction": _float_array_to_list(reconstruction),
        "tumor_mask": tumor_mask.tolist(),
        "voxel_spacing_mm": _float_array_to_list(voxel_spacing_mm),
        "tumor_volume_mm3": tumor_volume_mm3,
        "verification_profile_descriptor": build_verification_profile_contract(
            descriptor.profile_name
        ),
        "verified_manifest": None,
        "workstation_payload": None,
    }


def build_demo_authority_sources(
    *,
    case_id: str,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> tuple[AuthoritySource, ...]:
    """Return authority anchors required by the selected runtime profile."""
    descriptor = get_verification_profile(profile_name)
    sources = [
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
    ]
    if "sequence_manifest" in descriptor.required_authority_source_types:
        sources.append(
            AuthoritySource(
                source_type="sequence_manifest",
                source_id=f"{case_id}-sequence",
                reference=f"sequence://{case_id}/{descriptor.sequence_family}",
                details={
                    "sequence_family": descriptor.sequence_family,
                    "clinical_mode": descriptor.clinical_mode,
                },
            )
        )
    if "contrast_protocol" in descriptor.required_authority_source_types:
        sources.append(
            AuthoritySource(
                source_type="contrast_protocol",
                source_id=f"{case_id}-contrast",
                reference=f"contrast://{case_id}/protocol",
                details={"agent": "gadobutrol", "phase": "post_contrast"},
            )
        )
    if "research_protocol" in descriptor.required_authority_source_types:
        sources.append(
            AuthoritySource(
                source_type="research_protocol",
                source_id=f"{case_id}-research",
                reference=f"research://{case_id}/strict-protocol",
                details={"study_arm": "strict_reproducibility"},
            )
        )
    return tuple(sources)


def _demo_sequence_id(profile_name: str) -> str:
    if profile_name == "brain_t1_baseline":
        return "t1-pre-contrast"
    if profile_name == "brain_t2_baseline":
        return "t2-weighted"
    if profile_name == "contrast_enhanced_review":
        return "t1-post-contrast"
    if profile_name == "strict_research_mode":
        return "research-brain-mri"
    return "triage-brain-mri"


def _build_release_transition(
    *,
    transparency_log_path: Path,
    default_profile_name: str,
) -> Transition:
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
    return Transition(
        name="release_verified_mri_estimate",
        mutate=_release_to_radiologist,
        profile_registry=build_profile_registry(invariants),
        default_profile=default_profile_name,
        authority_validator=_validate_authority_sources,
        transparency_log=JsonlHashChainTransparencyLog(transparency_log_path),
        verify_transparency_after_append=True,
    )


def _release_to_radiologist(state: dict[str, Any]) -> None:
    descriptor = get_verification_profile(
        resolve_profile_name(profile_name=None, state=state)
    )
    state["verification_profile_descriptor"] = build_verification_profile_contract(
        descriptor.profile_name
    )
    analysis = _analyse_case(state, descriptor)
    contracts = _build_claim_to_proof_contracts(state, analysis, descriptor)
    state["verified_manifest"] = _build_verified_manifest(state, analysis, contracts, descriptor)
    state["workstation_payload"] = _build_workstation_payload(state, analysis, descriptor)
    state["release_status"] = RELEASE_STATUS_RELEASED


def _validate_authority_sources(
    authority: str,
    authority_sources: tuple[AuthoritySource, ...],
) -> None:
    seen_types = {source.source_type for source in authority_sources}
    required_types = frozenset(_AUTHORITY_REQUIREMENT_MAP.get(authority, ()))
    missing_types = sorted(required_types - seen_types)
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
    descriptor = get_verification_profile(
        resolve_profile_name(profile_name=None, state=dict(state))
    )
    expected_workstation_payload = _build_workstation_payload(state, analysis, descriptor)
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


def _analyse_case(
    state: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
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
        np.abs(ai_reconstruction - derived_reconstruction)
        > descriptor.tolerance("spatial_per_voxel_abs_error")
    )
    verified_mask = _derive_verified_mask(derived_reconstruction, descriptor)
    mask_false_positive = (ai_mask == 1) & (verified_mask == 0)
    mask_false_negative = (ai_mask == 0) & (verified_mask == 1)
    mask_mismatch_mask = mask_false_positive | mask_false_negative
    voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    ai_mask_voxel_count = int(np.count_nonzero(ai_mask))
    verified_mask_voxel_count = int(np.count_nonzero(verified_mask))
    computed_tumor_volume_mm3 = _round_float(verified_mask_voxel_count * voxel_volume_mm3)
    derived_reconstruction_list = _float_array_to_list(derived_reconstruction)
    ai_reconstruction_list = _float_array_to_list(ai_reconstruction)
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
    ai_reconstruction_hash = hash_canonical_json(ai_reconstruction_list)
    ai_mask_hash = hash_canonical_json(ai_mask_list)
    verified_mask_hash = hash_canonical_json(verified_mask_list)
    return {
        "profile_name": descriptor.profile_name,
        "profile_revision": descriptor.profile_revision,
        "profile_digest": descriptor.profile_digest(),
        "icsd_policy_id": descriptor.icsd_policy_id,
        "icsd_policy_version": descriptor.icsd_policy_version,
        "raw_hash": raw_hash,
        "ai_reconstruction_hash": ai_reconstruction_hash,
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
        "relative_ai_l2_error": _relative_l2(ai_residual, derived_reconstruction),
        "reconstruction_mismatch_voxel_count": int(np.count_nonzero(reconstruction_mismatch_mask)),
        "max_imaginary_leakage": imaginary_leakage,
        "voxel_volume_mm3": _round_float(voxel_volume_mm3),
        "computed_tumor_volume_mm3": computed_tumor_volume_mm3,
    }


def _build_verified_manifest(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    contracts: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
    measurement_metrics = {
        "per_frequency_complex_abs_error": _tolerance_metric(
            metric="per_frequency_complex_abs_error_max",
            domain="frequency_domain",
            scope="per_frequency",
            observed=analysis["max_measurement_abs_error"],
            threshold=descriptor.tolerance("frequency_per_coefficient_abs_error"),
            guard_band_fraction=descriptor.guard_band_fraction,
        ),
        "global_relative_l2_error": _tolerance_metric(
            metric="global_relative_l2_error",
            domain="frequency_domain",
            scope="global_norm",
            observed=analysis["relative_measurement_l2_error"],
            threshold=descriptor.tolerance("frequency_global_relative_l2_error"),
            guard_band_fraction=descriptor.guard_band_fraction,
        ),
        "spatial_imaginary_leakage": _tolerance_metric(
            metric="spatial_imaginary_leakage_max",
            domain="spatial_image_domain",
            scope="per_voxel",
            observed=analysis["max_imaginary_leakage"],
            threshold=descriptor.tolerance("imaginary_leakage_abs_error"),
            guard_band_fraction=descriptor.guard_band_fraction,
        ),
    }
    ai_match_metrics = {
        "per_voxel_abs_error": _tolerance_metric(
            metric="per_voxel_abs_error_max",
            domain="spatial_image_domain",
            scope="per_voxel",
            observed=analysis["max_ai_abs_error"],
            threshold=descriptor.tolerance("spatial_per_voxel_abs_error"),
            guard_band_fraction=descriptor.guard_band_fraction,
        ),
        "global_relative_l2_error": _tolerance_metric(
            metric="global_relative_l2_error",
            domain="spatial_image_domain",
            scope="global_norm",
            observed=analysis["relative_ai_l2_error"],
            threshold=descriptor.tolerance("spatial_global_relative_l2_error"),
            guard_band_fraction=descriptor.guard_band_fraction,
        ),
    }
    volume_metric = _tolerance_metric(
        metric="absolute_error",
        domain="scalar_volume",
        scope="global_scalar",
        observed=abs(claimed_volume - analysis["computed_tumor_volume_mm3"]),
        threshold=descriptor.tolerance("volume_absolute_error"),
        guard_band_fraction=descriptor.guard_band_fraction,
    )
    consistency_verified = all(
        metric["within_tolerance"] for metric in measurement_metrics.values()
    )
    ai_match_verified = all(metric["within_tolerance"] for metric in ai_match_metrics.values())
    mask_verified = analysis["mask_voxel_delta"] == 0
    volume_verified = volume_metric["within_tolerance"]
    clinical_plausibility = _build_clinical_plausibility_manifest(analysis, descriptor)
    heuristic_blocked = bool(clinical_plausibility["blocking_failures"])
    profile_policy_blocked = not descriptor.release_semantics["final_release_allowed"]
    proof_status = (
        "passed"
        if consistency_verified and ai_match_verified and mask_verified and volume_verified
        else "failed"
    )
    profile_policy_status = (
        "blocked" if heuristic_blocked or profile_policy_blocked else "passed"
    )
    block_category = "none"
    if proof_status != "passed":
        block_category = "proof_failure"
    elif heuristic_blocked:
        block_category = "heuristic_block"
    elif profile_policy_blocked:
        block_category = "profile_policy_block"
    final_release_allowed = (
        proof_status == "passed"
        and not heuristic_blocked
        and bool(descriptor.release_semantics["final_release_allowed"])
    )
    decision_class = "blocked"
    if proof_status == "passed" and not heuristic_blocked:
        decision_class = str(descriptor.release_semantics["success_decision_class"])
    decision_summary = {
        "proof_status": proof_status,
        "profile_policy_status": profile_policy_status,
        "decision_class": decision_class,
        "block_category": block_category,
        "final_release_allowed": final_release_allowed,
        "operator_review_required": clinical_plausibility["operator_review_required"],
    }
    manifest = {
        "profile_name": descriptor.profile_name,
        "profile_revision": descriptor.profile_revision,
        "profile_digest": descriptor.profile_digest(),
        "icsd_policy_id": descriptor.icsd_policy_id,
        "icsd_policy_version": descriptor.icsd_policy_version,
        "raw_hash": analysis["raw_hash"],
        "verification_profile_descriptor": contracts["verification_profile_descriptor"],
        "ai_claim_envelope": contracts["ai_claim_envelope"],
        "verified_reconstruction_trace": contracts["verified_reconstruction_trace"],
        "mask_derivation_policy": contracts["mask_derivation_policy"],
        "verification_tolerance_policy": contracts["verification_tolerance_policy"],
        "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
        "ai_mask_hash": analysis["ai_mask_hash"],
        "verified_mask_hash": analysis["verified_mask_hash"],
        "reconstruction_algorithm": "numpy.ifft2_per_slice.real",
        "performance_expectations": contracts["verification_profile_descriptor"][
            "performance_expectations"
        ],
        "mask_verification": {
            "verified": mask_verified,
            "algorithm": contracts["mask_derivation_policy"]["algorithm"],
            "threshold": contracts["mask_derivation_policy"]["parameters"]["threshold"],
            "ai_voxel_count": analysis["ai_mask_voxel_count"],
            "verified_voxel_count": analysis["verified_mask_voxel_count"],
            "voxel_delta": analysis["mask_voxel_delta"],
            "false_positive_count": analysis["mask_false_positive_count"],
            "false_negative_count": analysis["mask_false_negative_count"],
        },
        "measurement_consistency": {
            "verified": consistency_verified,
            "domain": "frequency_domain",
            "attack_surface": (
                "Distributed low-amplitude k-space perturbations are checked with a "
                "global norm in addition to the per-frequency maximum."
            ),
            "max_abs_error": analysis["max_measurement_abs_error"],
            "relative_l2_error": analysis["relative_measurement_l2_error"],
            "max_imaginary_leakage": analysis["max_imaginary_leakage"],
            "max_abs_tolerance": descriptor.tolerance("frequency_per_coefficient_abs_error"),
            "relative_l2_tolerance": descriptor.tolerance(
                "frequency_global_relative_l2_error"
            ),
            "imaginary_leakage_tolerance": descriptor.tolerance(
                "imaginary_leakage_abs_error"
            ),
            "guard_band_fraction": descriptor.guard_band_fraction,
            "metrics": measurement_metrics,
            "forward_projection_hash": analysis["forward_projection_hash"],
        },
        "ai_reconstruction_match": {
            "verified": ai_match_verified,
            "domain": "spatial_image_domain",
            "attack_surface": (
                "Distributed low-amplitude voxel perturbations are checked with a "
                "global norm in addition to the per-voxel maximum."
            ),
            "max_abs_error": analysis["max_ai_abs_error"],
            "relative_l2_error": analysis["relative_ai_l2_error"],
            "max_abs_tolerance": descriptor.tolerance("spatial_per_voxel_abs_error"),
            "relative_l2_tolerance": descriptor.tolerance(
                "spatial_global_relative_l2_error"
            ),
            "guard_band_fraction": descriptor.guard_band_fraction,
            "metrics": ai_match_metrics,
        },
        "tumor_volume_proof": {
            "verified": volume_verified,
            "claimed_mm3": claimed_volume,
            "computed_mm3": analysis["computed_tumor_volume_mm3"],
            "voxel_count": analysis["verified_mask_voxel_count"],
            "voxel_volume_mm3": analysis["voxel_volume_mm3"],
            "tolerance_metric": volume_metric,
        },
        "no_invented_pixels": {
            "verified": consistency_verified and ai_match_verified,
            "pixel_source": "harness_reconstructed_from_raw_kspace",
            "ai_render_passthrough_allowed": False,
        },
        "clinical_plausibility": clinical_plausibility,
        "decision_summary": decision_summary,
    }
    manifest["verification_tiers"] = _build_verification_tiers(state, manifest, descriptor)
    return manifest


def _build_clinical_plausibility_manifest(
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    derived_reconstruction = np.asarray(analysis["derived_reconstruction"], dtype=np.float64)
    verified_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    total_voxels = int(verified_mask.size)
    verified_voxels = int(analysis["verified_mask_voxel_count"])
    mask_fraction = 0.0
    if total_voxels:
        mask_fraction = verified_voxels / total_voxels
    mask_rule = descriptor.heuristic("mask_fraction_guardrail")
    heuristics = [
        _evaluate_heuristic_rule(
            code="mask_fraction_guardrail",
            rule=mask_rule,
            observed=mask_fraction,
            fail_message=(
                "The verified mask occupies an unusually large share of the voxel grid "
                f"({mask_fraction:.1%} vs configured ceiling {mask_rule['threshold']:.0%})."
            ),
            pass_message=(
                "The verified mask occupies a bounded share of the voxel grid "
                f"({mask_fraction:.1%})."
            ),
        ),
    ]

    if verified_voxels and verified_voxels < total_voxels:
        inside_mean = float(np.mean(derived_reconstruction[verified_mask == 1]))
        outside_mean = float(np.mean(derived_reconstruction[verified_mask == 0]))
        contrast_delta = inside_mean - outside_mean
        contrast_rule = descriptor.heuristic("inside_outside_contrast")
        heuristics.append(
            _evaluate_heuristic_rule(
                code="inside_outside_contrast",
                rule=contrast_rule,
                observed=contrast_delta,
                fail_message=(
                    "The verified lesion is not meaningfully brighter than the surrounding "
                    f"background ({_round_float(contrast_delta)} vs configured floor "
                    f"{_round_float(float(contrast_rule['threshold']))})."
                ),
                pass_message=(
                    "The verified lesion remains visually separable from the surrounding "
                    f"background (delta = {_round_float(contrast_delta)})."
                ),
            )
        )
    else:
        contrast_rule = descriptor.heuristic("inside_outside_contrast")
        heuristics.append(
            _heuristic_entry(
                code="inside_outside_contrast",
                label="Inside/outside contrast delta",
                severity=contrast_rule["severity"],
                status="not_evaluated",
                message=(
                    "Contrast heuristics were not evaluated because the verified mask "
                    "does not leave both lesion and background voxels to compare."
                ),
            )
        )

    slice_span = int(np.count_nonzero(np.any(verified_mask == 1, axis=(1, 2))))
    slice_rule = descriptor.heuristic("slice_span")
    heuristics.append(
        _evaluate_heuristic_rule(
            code="slice_span",
            rule=slice_rule,
            observed=float(slice_span),
            fail_message=(
                "The verified lesion only appears on "
                f"{slice_span} slice(s), below the configured floor of "
                f"{_round_float(float(slice_rule['threshold']))}."
            ),
            pass_message=f"The verified lesion spans {slice_span} slice(s).",
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
    advisories = [
        {
            "code": entry["code"],
            "label": entry["label"],
            "message": entry["message"],
        }
        for entry in heuristics
        if entry["status"] == "advisory"
    ]
    blocking_failures = [
        {
            "code": entry["code"],
            "label": entry["label"],
            "message": entry["message"],
        }
        for entry in heuristics
        if entry["status"] == "failed"
    ]
    status = "passed"
    if blocking_failures:
        status = "failed"
    elif warnings:
        status = "warning"
    elif advisories:
        status = "advisory"
    return {
        "blocking_configured": any(
            rule["severity"] == HEURISTIC_BLOCKING
            for rule in descriptor.heuristic_rules.values()
        ),
        "status": status,
        "summary": _clinical_plausibility_summary(
            warnings=warnings,
            advisories=advisories,
            blocking_failures=blocking_failures,
        ),
        "warning_count": len(warnings),
        "advisory_count": len(advisories),
        "blocking_failure_count": len(blocking_failures),
        "operator_review_required": bool(warnings)
        and bool(descriptor.release_semantics["requires_operator_review_on_warning"]),
        "warnings": warnings,
        "advisories": advisories,
        "blocking_failures": blocking_failures,
        "heuristics": heuristics,
    }


def _build_verification_tiers(
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    descriptor: Any,
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
            "subtitle": "Runtime profile heuristics",
            "blocking": clinical_section["blocking_configured"],
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
    severity: str,
    status: str,
    message: str,
    observed: float | int | None = None,
    threshold: float | int | None = None,
) -> dict[str, Any]:
    entry = {
        "code": code,
        "label": label,
        "severity": severity,
        "status": status,
        "message": message,
    }
    if observed is not None:
        entry["observed"] = observed
    if threshold is not None:
        entry["threshold"] = threshold
    return entry


def _evaluate_heuristic_rule(
    *,
    code: str,
    rule: Mapping[str, Any],
    observed: float,
    fail_message: str,
    pass_message: str,
) -> dict[str, Any]:
    label = str(rule["label"])
    severity = str(rule["severity"])
    threshold = float(rule["threshold"])
    if not bool(rule.get("enabled", True)) or severity == "disabled":
        return _heuristic_entry(
            code=code,
            label=label,
            severity=severity,
            status="not_evaluated",
            message="This heuristic is disabled for the active runtime profile.",
        )
    comparison = str(rule["comparison"])
    passed = _heuristic_comparison_passed(
        comparison=comparison,
        observed=float(observed),
        threshold=threshold,
    )
    if passed:
        status = "passed"
    elif severity == "blocking":
        status = "failed"
    elif severity == "warning":
        status = "warning"
    else:
        status = "advisory"
    return _heuristic_entry(
        code=code,
        label=label,
        severity=severity,
        status=status,
        message=pass_message if passed else fail_message,
        observed=_round_float(float(observed)),
        threshold=_round_float(threshold),
    )


def _heuristic_comparison_passed(
    *,
    comparison: str,
    observed: float,
    threshold: float,
) -> bool:
    if comparison == "less_than_or_equal":
        return observed <= threshold
    if comparison == "greater_than_or_equal":
        return observed >= threshold
    raise ValueError(f"Unsupported heuristic comparison {comparison!r}.")


def _clinical_plausibility_summary(
    *,
    warnings: list[dict[str, Any]],
    advisories: list[dict[str, Any]],
    blocking_failures: list[dict[str, Any]],
) -> str:
    if blocking_failures:
        return f"{len(blocking_failures)} blocking clinical heuristic failure(s) detected."
    if warnings:
        return f"{len(warnings)} warning-level clinical heuristic finding(s) detected."
    if advisories:
        return f"{len(advisories)} advisory clinical heuristic finding(s) detected."
    return "No clinical heuristic findings were detected."


def _build_workstation_payload(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    return {
        "case_id": state["case_id"],
        "study_uid": state["study_uid"],
        "series_uid": state["series_uid"],
        "scanner_id": state["scanner_id"],
        "sequence_id": state["sequence_id"],
        "patient_pseudonym": state["patient_pseudonym"],
        "profile_name": descriptor.profile_name,
        "profile_revision": descriptor.profile_revision,
        "profile_digest": descriptor.profile_digest(),
        "pixel_source": "harness_reconstructed_from_raw_kspace",
        "overlay_mode": "profile_verified_mask_on_measured_grid",
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
                "category": "proof_failure",
                "summary": f"mask_voxel_mismatch (delta = {mask_delta} voxels)",
                "delta": mask_delta,
                "message": (
                    "The AI tumour mask does not match the profile-derived verified mask "
                    "computed from the raw-derived reconstruction."
                ),
                "details": {
                    "false_positive_count": int(analysis["mask_false_positive_count"]),
                    "false_negative_count": int(analysis["mask_false_negative_count"]),
                },
            }
        )
    intensity_delta = int(analysis["reconstruction_mismatch_voxel_count"])
    if not manifest["ai_reconstruction_match"]["verified"]:
        if intensity_delta:
            summary = (
                "intensity_manipulation_detected "
                f"(delta = {intensity_delta} voxels; max_abs_error = "
                f"{_format_metric_for_summary(analysis['max_ai_abs_error'])})"
            )
        else:
            summary = (
                "intensity_manipulation_detected "
                f"(relative_l2_error = {_format_metric_for_summary(analysis['relative_ai_l2_error'])}; "
                f"per_voxel_max_abs_error = {_format_metric_for_summary(analysis['max_ai_abs_error'])})"
            )
        reasons.append(
            {
                "code": "intensity_manipulation_detected",
                "category": "proof_failure",
                "summary": summary,
                "delta": intensity_delta,
                "message": (
                    "The AI reconstruction no longer matches the deterministic reconstruction "
                    "recomputed from raw k-space under the explicit per-voxel and global-norm "
                    "tolerance contract."
                ),
                "details": {
                    "max_abs_error": analysis["max_ai_abs_error"],
                    "relative_l2_error": analysis["relative_ai_l2_error"],
                    "metrics": manifest["ai_reconstruction_match"]["metrics"],
                },
            }
        )
    if not manifest["tumor_volume_proof"]["verified"]:
        claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
        verified_volume = manifest["tumor_volume_proof"]["computed_mm3"]
        reasons.append(
            {
                "code": "tumor_volume_mismatch",
                "category": "proof_failure",
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
                "category": "proof_failure",
                "summary": (
                    "raw_measurement_inconsistency "
                    f"(max |Δ| = {_format_metric_for_summary(analysis['max_measurement_abs_error'])}; "
                    f"relative_l2_error = {_format_metric_for_summary(analysis['relative_measurement_l2_error'])})"
                ),
                "delta": analysis["max_measurement_abs_error"],
                "message": (
                    "The deterministic reconstruction cannot be cleanly reprojected back "
                    "to the supplied raw measurements under the explicit per-frequency and "
                    "global-norm tolerance policy."
                ),
                "details": {
                    "relative_l2_error": analysis["relative_measurement_l2_error"],
                    "max_imaginary_leakage": analysis["max_imaginary_leakage"],
                    "metrics": manifest["measurement_consistency"]["metrics"],
                },
            }
        )
    for entry in manifest["clinical_plausibility"]["blocking_failures"]:
        reasons.append(
            {
                "code": entry["code"],
                "category": "heuristic_block",
                "summary": f"{entry['code']} (blocking heuristic)",
                "delta": None,
                "message": entry["message"],
                "details": {"source": "clinical_plausibility"},
            }
        )
    if not manifest["decision_summary"]["final_release_allowed"] and manifest["decision_summary"][
        "decision_class"
    ] == DECISION_CLASS_TRIAGE_ONLY:
        reasons.append(
            {
                "code": "non_final_profile_selected",
                "category": "profile_policy_block",
                "summary": (
                    "non_final_profile_selected "
                    f"(decision_class = {manifest['decision_summary']['decision_class']})"
                ),
                "delta": None,
                "message": (
                    "The selected runtime profile is triage-only and cannot produce a final "
                    "released state or portable evidence bundle."
                ),
                "details": {
                    "profile_name": manifest["profile_name"],
                    "decision_class": manifest["decision_summary"]["decision_class"],
                },
            }
        )
    return reasons


def _derive_verified_mask(
    derived_reconstruction: np.ndarray,
    descriptor: Any,
) -> np.ndarray:
    reconstruction = np.asarray(derived_reconstruction, dtype=np.float64)
    return (reconstruction >= descriptor.mask_threshold).astype(np.int64)


def _tolerance_metric(
    *,
    metric: str,
    domain: str,
    scope: str,
    observed: float,
    threshold: float,
    guard_band_fraction: float,
) -> dict[str, Any]:
    observed_value = _round_metric_float(float(observed))
    threshold_value = _round_metric_float(float(threshold))
    guard_band_threshold = _round_metric_float(
        threshold_value * float(guard_band_fraction)
    )
    if observed_value > threshold_value:
        boundary_status = "exceeded"
    elif observed_value >= guard_band_threshold:
        boundary_status = "near_boundary"
    else:
        boundary_status = "clear"
    return {
        "metric": metric,
        "domain": domain,
        "scope": scope,
        "observed": observed_value,
        "threshold": threshold_value,
        "guard_band_fraction": _round_metric_float(float(guard_band_fraction)),
        "guard_band_threshold": guard_band_threshold,
        "within_tolerance": observed_value <= threshold_value,
        "boundary_status": boundary_status,
        "margin_to_threshold": _round_metric_float(threshold_value - observed_value),
    }


def _build_claim_to_proof_contracts(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    ai_claim_envelope = build_ai_claim_envelope(
        state=dict(state),
        raw_kspace_hash=analysis["raw_hash"],
        ai_reconstruction_hash=analysis["ai_reconstruction_hash"],
        ai_mask_hash=analysis["ai_mask_hash"],
        claimed_tumor_volume_mm3=_round_float(float(state["tumor_volume_mm3"])),
    )
    verification_profile_descriptor = build_verification_profile_contract(
        descriptor.profile_name
    )
    mask_derivation_policy = build_mask_derivation_policy(
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        threshold=descriptor.mask_threshold,
    )
    verification_tolerance_policy = build_verification_tolerance_policy(
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        guard_band_fraction=descriptor.guard_band_fraction,
        spatial_per_voxel_abs_error=descriptor.tolerance("spatial_per_voxel_abs_error"),
        spatial_global_relative_l2_error=descriptor.tolerance(
            "spatial_global_relative_l2_error"
        ),
        frequency_per_coefficient_abs_error=descriptor.tolerance(
            "frequency_per_coefficient_abs_error"
        ),
        frequency_global_relative_l2_error=descriptor.tolerance(
            "frequency_global_relative_l2_error"
        ),
        imaginary_leakage_abs_error=descriptor.tolerance("imaginary_leakage_abs_error"),
        volume_absolute_error=descriptor.tolerance("volume_absolute_error"),
    )
    verified_reconstruction_trace = build_verified_reconstruction_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        raw_kspace_hash=analysis["raw_hash"],
        derived_reconstruction_hash=analysis["derived_reconstruction_hash"],
        forward_projection_hash=analysis["forward_projection_hash"],
        verified_mask_hash=analysis["verified_mask_hash"],
        computed_tumor_volume_mm3=analysis["computed_tumor_volume_mm3"],
        claim_envelope=ai_claim_envelope,
        mask_derivation_policy=mask_derivation_policy,
        tolerance_policy=verification_tolerance_policy,
        max_measurement_abs_error=analysis["max_measurement_abs_error"],
        relative_measurement_l2_error=analysis["relative_measurement_l2_error"],
        max_imaginary_leakage=analysis["max_imaginary_leakage"],
        max_ai_abs_error=analysis["max_ai_abs_error"],
        relative_ai_l2_error=analysis["relative_ai_l2_error"],
        mask_voxel_delta=analysis["mask_voxel_delta"],
    )
    return {
        "verification_profile_descriptor": verification_profile_descriptor,
        "ai_claim_envelope": ai_claim_envelope,
        "verified_reconstruction_trace": verified_reconstruction_trace,
        "mask_derivation_policy": mask_derivation_policy,
        "verification_tolerance_policy": verification_tolerance_policy,
    }


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


def _round_metric_float(value: float) -> float:
    if abs(value) < 1e-15:
        return 0.0
    return round(float(value), 15)


def _format_metric_for_summary(value: float) -> str:
    metric = _round_metric_float(float(value))
    if metric == 0.0:
        return "0.0"
    if abs(metric) < 1e-6:
        return f"{metric:.3e}"
    return str(_round_float(metric))


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return _round_metric_float(float(np.max(np.abs(array))))


def _relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(reference.ravel()))
    if reference_norm == 0:
        return 0.0
    return _round_metric_float(float(np.linalg.norm(delta.ravel()) / reference_norm))


def render_release_bundle_json(result: TransitionResult) -> str:
    """Return the released workstation payload in canonical JSON."""
    return canonical_json_dumps(result.after_state["workstation_payload"])
