"""ICSD-bound MRI verification layer that blocks invented pixels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime
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
    build_cross_series_verification_policy,
    build_cross_series_verification_trace,
    build_longitudinal_registration_trace,
    build_longitudinal_selection_trace,
    build_mask_derivation_policy,
    build_measurement_integrity_proof,
    build_multi_series_verification_profile_descriptor_set,
    build_prior_verified_context,
    build_proof_core_evidence_trace,
    build_reconstruction_parity_proof,
    build_segmentation_policy_proof,
    build_study_temporal_aggregation_trace,
    build_temporal_consistency_trace,
    build_temporal_trajectory_trace,
    build_verification_profile_contract,
    build_verification_tolerance_policy,
    build_volume_computation_proof,
    canonicalise_prior_verified_context,
    canonicalise_prior_verified_contexts,
    canonicalise_ai_stack_identity,
    canonicalise_spatial_geometry,
    canonicalise_study_acquired_at,
    canonicalise_timestamp_provenance,
)
from .mask_engine import (
    CONNECTED_COMPONENT_ALGORITHM,
    MORPHOLOGY_ALGORITHM,
    SLICE_CONSISTENCY_ALGORITHM,
    THRESHOLD_ALGORITHM,
    derive_verified_mask,
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
    verification_profile_descriptor_from_dict,
)
from .transparency import JsonlHashChainTransparencyLog

MRI_GUARDIAN_POLICY_NAME = DEFAULT_PROFILE_NAME
MRI_GUARDIAN_POLICY_VERSION = default_icsd_policy_version()
RELEASE_STATUS_PENDING = "pending_verification"
RELEASE_STATUS_RELEASED = "released_to_radiologist"
MULTI_SERIES_CASE_KIND = "multi_series_v2"
MULTI_SERIES_GEOMETRY_RULE = "same_voxel_grid_required"
MULTI_SERIES_CORROBORATION_RULE = (
    "positive_lesion_requires_primary_supporting_exact_overlap"
)
_AUTHORITY_REQUIREMENT_MAP = build_authority_requirement_map()

PIPELINE_PHASE_INTAKE_VALIDATION = "intake_validation"
PIPELINE_PHASE_STRUCTURAL_VALIDITY = "structural_validity"
PIPELINE_PHASE_PROOF_CORE = "proof_core"
PIPELINE_PHASE_CLINICAL_PLAUSIBILITY = "clinical_plausibility"
PIPELINE_PHASE_POLICY_EVALUATION = "policy_evaluation"
PIPELINE_PHASE_MULTI_SERIES_AGGREGATION = "multi_series_aggregation"
PIPELINE_PHASE_RELEASE_CONTROL = "release_control"

PROOF_STAGE_MEASUREMENT_INTEGRITY = "MeasurementIntegrityProof"
PROOF_STAGE_RECONSTRUCTION_PARITY = "ReconstructionParityProof"
PROOF_STAGE_SEGMENTATION_POLICY = "SegmentationPolicyProof"
PROOF_STAGE_VOLUME_COMPUTATION = "VolumeComputationProof"
_PROOF_STAGE_ORDER = (
    PROOF_STAGE_MEASUREMENT_INTEGRITY,
    PROOF_STAGE_RECONSTRUCTION_PARITY,
    PROOF_STAGE_SEGMENTATION_POLICY,
    PROOF_STAGE_VOLUME_COMPUTATION,
)
_PROOF_STAGE_TITLES = {
    PROOF_STAGE_MEASUREMENT_INTEGRITY: "Measurement Integrity",
    PROOF_STAGE_RECONSTRUCTION_PARITY: "Reconstruction Parity",
    PROOF_STAGE_SEGMENTATION_POLICY: "Segmentation Policy",
    PROOF_STAGE_VOLUME_COMPUTATION: "Volume Computation",
}

OPERATIONAL_SUBPHASE_PRE_CHECK = "pre_check"
OPERATIONAL_SUBPHASE_FULL_PROOF = "full_proof"
_OPERATIONAL_SUBPHASE_SCHEMA = "derived_operational_subphases_v1"
_OPERATIONAL_SUBPHASE_ORDER = (
    OPERATIONAL_SUBPHASE_PRE_CHECK,
    OPERATIONAL_SUBPHASE_FULL_PROOF,
)

OPERATIONAL_CHECK_STRUCTURE_VALIDATION = "structure_validation"
OPERATIONAL_CHECK_HASH_CHECKS = "hash_checks"
OPERATIONAL_CHECK_SHAPE_CONSISTENCY = "shape_consistency"
OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION = "quick_anomaly_detection"
_PRE_CHECK_CHECK_ORDER = (
    OPERATIONAL_CHECK_STRUCTURE_VALIDATION,
    OPERATIONAL_CHECK_HASH_CHECKS,
    OPERATIONAL_CHECK_SHAPE_CONSISTENCY,
    OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION,
)
_PRE_CHECK_BLOCKING_CHECKS = (
    OPERATIONAL_CHECK_STRUCTURE_VALIDATION,
    OPERATIONAL_CHECK_HASH_CHECKS,
    OPERATIONAL_CHECK_SHAPE_CONSISTENCY,
)
_PRE_CHECK_ALLOWED_STATUSES = frozenset({"passed", "failed", "advisory"})
_FULL_PROOF_ALLOWED_STATUSES = frozenset({"passed", "failed", "not_evaluated"})
_PRE_CHECK_CHECK_ALLOWED_STATUSES = {
    OPERATIONAL_CHECK_STRUCTURE_VALIDATION: frozenset({"passed", "failed"}),
    OPERATIONAL_CHECK_HASH_CHECKS: frozenset({"passed", "failed", "not_evaluated"}),
    OPERATIONAL_CHECK_SHAPE_CONSISTENCY: frozenset({"passed", "failed", "not_evaluated"}),
    OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION: frozenset(
        {"passed", "advisory", "not_evaluated"}
    ),
}
_FULL_PROOF_STAGE_ALLOWED_STATUSES = frozenset({"passed", "failed", "not_evaluated"})

EVALUATION_STAGE_INTAKE_VALIDATION = PIPELINE_PHASE_INTAKE_VALIDATION
EVALUATION_STAGE_STRUCTURAL_VALIDITY = PIPELINE_PHASE_STRUCTURAL_VALIDITY
EVALUATION_STAGE_PHYSICAL_CONSISTENCY = PIPELINE_PHASE_PROOF_CORE
EVALUATION_STAGE_CLINICAL_PLAUSIBILITY = PIPELINE_PHASE_CLINICAL_PLAUSIBILITY
EVALUATION_STAGE_POLICY_EVALUATION = PIPELINE_PHASE_POLICY_EVALUATION
EVALUATION_STAGE_MULTI_SERIES_AGGREGATION = PIPELINE_PHASE_MULTI_SERIES_AGGREGATION
EVALUATION_STAGE_RELEASE_CONTROL = PIPELINE_PHASE_RELEASE_CONTROL

DECISION_CLASS_VERIFIED_CLEAN = "verified_clean"
DECISION_CLASS_VERIFIED_BOUNDARY_HUGGING = "verified_boundary_hugging"
DECISION_CLASS_VERIFIED_WITH_ADVISORY = "verified_with_advisory"
DECISION_CLASS_BLOCKED_STRUCTURAL = "blocked_structural"
DECISION_CLASS_BLOCKED_PHYSICAL = "blocked_physical"
DECISION_CLASS_BLOCKED_POLICY = "blocked_policy"

DECISION_SEVERITY_INFO = "info"
DECISION_SEVERITY_WARNING = "warning"
DECISION_SEVERITY_ERROR = "error"
DECISION_SEVERITY_CRITICAL = "critical"

DECISION_REASON_BAND_PRIMARY = "primary"
DECISION_REASON_BAND_SAME_STAGE = "same_stage"
DECISION_REASON_BAND_SECONDARY_MODIFIER = "secondary_modifier"
DECISION_REASON_BAND_ADVISORY_INFO = "advisory_info"

STRUCTURAL_FAILURE_CATEGORY = "structural_failure"

_DECISION_REASON_BAND_ORDER = {
    DECISION_REASON_BAND_PRIMARY: 0,
    DECISION_REASON_BAND_SAME_STAGE: 1,
    DECISION_REASON_BAND_SECONDARY_MODIFIER: 2,
    DECISION_REASON_BAND_ADVISORY_INFO: 3,
}

TEMPORAL_STATUS_NOT_EVALUATED = "not_evaluated"
TEMPORAL_STATUS_ALIGNED = "aligned"
TEMPORAL_STATUS_DEGRADED = "degraded"
TEMPORAL_STATUS_UNSUITABLE = "unsuitable"

TEMPORAL_SCOPE_NOT_EVALUATED = "not_evaluated"
TEMPORAL_SCOPE_FULL = "full"
TEMPORAL_SCOPE_VOLUME_ONLY = "volume_only"
TEMPORAL_SCOPE_NONE = "none"

TEMPORAL_SPATIAL_MODE_EXACT = "exact"
TEMPORAL_SPATIAL_MODE_REGISTERED = "registered"
TEMPORAL_SPATIAL_MODE_VOLUME_ONLY = "volume_only"
TEMPORAL_SPATIAL_MODE_NONE = "none"

TEMPORAL_PAIR_PROVENANCE_VERIFIED_BUNDLE = "verified_bundle_pair"
TEMPORAL_PAIR_PROVENANCE_TRUSTED_METADATA = "trusted_metadata_pair"
TEMPORAL_PAIR_PROVENANCE_MIXED = "mixed_provenance"
TEMPORAL_PAIR_PROVENANCE_MISSING = "timestamp_missing"
TEMPORAL_PAIR_PROVENANCE_UNTRUSTED = "timestamp_untrusted"

TEMPORAL_GEOMETRY_EXACT_MATCH = "exact_match"
TEMPORAL_GEOMETRY_REGISTERED_MATCH = "registered_match"
TEMPORAL_GEOMETRY_VOLUME_ONLY = "volume_basis_only"
TEMPORAL_GEOMETRY_INSUFFICIENT = "insufficient_geometry_trust"
TEMPORAL_GEOMETRY_REASON_EXACT_MATCH = "geometry_exact_match"
TEMPORAL_GEOMETRY_REASON_REGISTERED = "geometry_registered_match"
TEMPORAL_GEOMETRY_REASON_NOT_EXACT = "geometry_not_exactly_comparable"
TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT = "geometry_not_sufficient_for_volume_only"

TEMPORAL_PRIOR_ADMISSIBLE = "admissible"
TEMPORAL_PRIOR_INADMISSIBLE = "inadmissible"

TEMPORAL_REASON_TEMPORAL_PRIOR_UNSUITABLE = "temporal_prior_unsuitable"
TEMPORAL_REASON_INTERVAL_DEGRADED = "temporal_interval_degraded"
TEMPORAL_REASON_GEOMETRY_LIMITED = "temporal_geometry_limited"
TEMPORAL_REASON_TIMESTAMP_REVERSED = "temporal_timestamp_reversed"
TEMPORAL_REASON_VOLUME_WARNING = "temporal_volume_change_warning"
TEMPORAL_REASON_RATE_WARNING = "temporal_growth_rate_warning"
TEMPORAL_REASON_IOU_WARNING = "temporal_intersection_over_union_warning"
TEMPORAL_REASON_CENTROID_WARNING = "temporal_centroid_shift_warning"
TEMPORAL_REASON_TRAJECTORY_ACCELERATION = "temporal_trajectory_acceleration"
TEMPORAL_REASON_TRAJECTORY_INFLECTION = "temporal_trajectory_inflection"
TEMPORAL_REASON_TRAJECTORY_RECURRENCE = "temporal_trajectory_recurrence_after_zero"
TEMPORAL_REASON_TRAJECTORY_COMPONENT_CHANGE = "temporal_component_count_trajectory_change"

TEMPORAL_FLAG_COMPONENT_COUNT_CHANGED = "component_count_changed"
TEMPORAL_FLAG_RATE_NOT_EVALUATED = "temporal_rate_not_evaluated"
TEMPORAL_FLAG_TRAJECTORY_ACCELERATION = "trajectory_acceleration_detected"
TEMPORAL_FLAG_TRAJECTORY_INFLECTION = "trajectory_inflection_detected"
TEMPORAL_FLAG_TRAJECTORY_RECURRENCE = "trajectory_recurrence_after_zero"
TEMPORAL_FLAG_TRAJECTORY_COMPONENT_CHANGE = "trajectory_component_count_changed"

TEMPORAL_REGISTRATION_STATUS_NOT_EVALUATED = "not_evaluated"
TEMPORAL_REGISTRATION_STATUS_USED = "used"
TEMPORAL_REGISTRATION_STATUS_REJECTED = "rejected"
TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE = "not_applicable"

TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT = "no_prior_context"
TEMPORAL_SKIPPED_PRIOR_BUNDLE_IDENTITY_MISSING = "prior_bundle_identity_missing"
TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED = "prior_not_released"
TEMPORAL_SKIPPED_SEQUENCE_FAMILY_MISMATCH = "sequence_family_mismatch"
TEMPORAL_SKIPPED_TIMESTAMP_MISSING = "timestamp_missing"
TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED = "timestamp_untrusted"
TEMPORAL_SKIPPED_MIXED_PROVENANCE = "mixed_provenance"
TEMPORAL_SKIPPED_TIMESTAMP_REVERSED = "timestamp_reversed"
TEMPORAL_SKIPPED_ZERO_INTERVAL = "zero_interval_days"
TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX = "interval_exceeds_max_days"
TEMPORAL_SKIPPED_GEOMETRY_NOT_EXACT = "geometry_not_exactly_comparable"
TEMPORAL_SKIPPED_GEOMETRY_NOT_SUFFICIENT = "geometry_not_sufficient_for_volume_only"
TEMPORAL_SKIPPED_ZERO_PRIOR_VOLUME = "zero_prior_volume"
TEMPORAL_SKIPPED_EMPTY_PRIOR_MASK = "empty_prior_mask"
TEMPORAL_SKIPPED_EMPTY_CURRENT_MASK = "empty_current_mask"
TEMPORAL_SKIPPED_BOTH_MASKS_EMPTY = "both_masks_empty"
TEMPORAL_SKIPPED_NO_SELECTED_PRIORS = "no_selected_priors"
TEMPORAL_SKIPPED_NO_USABLE_INTERVALS = "no_usable_trajectory_intervals"
TEMPORAL_SKIPPED_OLDEST_SELECTED_PRIOR_ZERO = "oldest_selected_prior_zero_volume"


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
        self._multi_series_transition = _build_multi_series_release_transition(
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

    def release_multi_series_case(
        self,
        *,
        state: Mapping[str, Any],
        actor: str,
        authority_sources_by_series_id: Mapping[
            str,
            Iterable[AuthoritySource | Mapping[str, Any]],
        ],
        timestamp: str | None = None,
        transition_id: str | UUID | None = None,
    ) -> TransitionResult:
        working_state = deepcopy(dict(state))
        if working_state.get("case_kind") != MULTI_SERIES_CASE_KIND:
            raise ValueError(
                "release_multi_series_case requires a state with "
                f"case_kind={MULTI_SERIES_CASE_KIND!r}."
            )
        context = _build_multi_series_context(working_state)
        _validate_multi_series_authority_sources_by_series_id(
            state=working_state,
            authority_sources_by_series_id=authority_sources_by_series_id,
        )
        flattened_sources = _flatten_multi_series_authority_sources(
            authority_sources_by_series_id=authority_sources_by_series_id,
            series_order=context["series_order"],
        )
        primary_descriptor = _resolve_primary_series_descriptor(working_state)
        return self._multi_series_transition.apply(
            state=working_state,
            entity_type="mri_study",
            entity_id=str(working_state["case_id"]),
            actor=actor,
            authority_sources=flattened_sources,
            profile=primary_descriptor.profile_name,
            policy_version=primary_descriptor.icsd_policy_version,
            timestamp=timestamp,
            transition_id=transition_id,
        )

def build_case_verification_report(
    state: Mapping[str, Any],
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Return the deterministic verification envelope for one MRI case."""
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return build_multi_series_case_verification_report(state)
    explicit_profile_name = isinstance(profile_name, str) and bool(profile_name.strip())
    resolved_profile_name = resolve_profile_name(profile_name=profile_name, state=dict(state))
    descriptor, structural_report = _prepare_single_series_report(
        state,
        profile_name=resolved_profile_name,
        explicit_profile_name=explicit_profile_name,
    )
    if structural_report is not None:
        return structural_report
    if descriptor is None:
        raise RuntimeError("Single-series report preparation did not resolve a descriptor.")
    profile_contract = build_verification_profile_contract(descriptor.profile_name)
    analysis = _analyse_case(state, descriptor)
    temporal_consistency = _build_temporal_consistency_section_v2(
        state,
        analysis,
        descriptor,
    )
    contracts = _build_claim_to_proof_contracts(
        state,
        analysis,
        descriptor,
        temporal_consistency=temporal_consistency,
    )
    manifest = _build_verified_manifest(
        state,
        analysis,
        contracts,
        descriptor,
        temporal_consistency=temporal_consistency,
    )
    contracts = _materialize_proof_core_contracts(
        state=state,
        analysis=analysis,
        manifest=manifest,
        contracts=contracts,
        descriptor=descriptor,
    )
    manifest["proof_core_evidence_trace"] = contracts["proof_core_evidence_trace"]
    reasons = _build_failure_reasons(state, analysis, manifest)
    verification_pipeline = manifest["verification_pipeline"]
    decision_summary = manifest["decision_summary"]
    report = {
        "profile": profile_contract,
        "analysis": analysis,
        "contracts": contracts,
        "manifest": manifest,
        "verification_pipeline": verification_pipeline,
        "failure_reasons": reasons,
        "decision_summary": decision_summary,
        "release_eligible": bool(decision_summary["final_release_allowed"]),
    }
    decision = _build_single_series_decision(report)
    report["decision"] = decision
    report["manifest"]["decision"] = deepcopy(decision)
    return report


def build_multi_series_case_verification_report(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic verification envelope for one study-shaped Guardian case."""
    context = _build_multi_series_context(state)
    series_reports = [_build_multi_series_series_report(context, series_id) for series_id in context["series_order"]]
    report = _build_multi_series_study_report(context, series_reports)
    return report


def build_demo_case(*, profile_name: str = DEFAULT_PROFILE_NAME) -> dict[str, Any]:
    """Return a deterministic happy-path MRI case."""
    descriptor = get_verification_profile(profile_name)
    reconstruction = _build_demo_reconstruction(descriptor)
    tumor_mask = _derive_verified_mask(reconstruction, descriptor)
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


def build_multi_series_demo_case() -> dict[str, Any]:
    """Return a deterministic multi-series study with primary, supporting, and auxiliary series."""
    case_id = "mri-study-0001"
    patient_pseudonym = "anon-0001"
    study_uid = "study-0001"
    scanner_id = "scanner-7t-alpha"
    demo_shape = (2, 2, 2)
    series_specs = (
        ("series-primary-t1", "t1-primary", "brain_t1_baseline", ((0, 0, 0), (0, 0, 1))),
        ("series-supporting-t2", "t2-supporting", "brain_t2_baseline", ((0, 0, 0),)),
        ("series-supporting-flair", "flair-supporting", "brain_t1_baseline", ((1, 1, 1),)),
        ("series-auxiliary-derived", "derived-auxiliary", "brain_t1_component_review", ((1, 1, 1),)),
    )
    series: list[dict[str, Any]] = []
    for series_uid, series_id, profile_name, hot_voxels in series_specs:
        descriptor = get_verification_profile(profile_name)
        seeded_reconstruction = _reconstruction_with_hot_voxels(
            demo_shape,
            hot_voxels,
            hot_value=4.0 if descriptor.mask_algorithm == THRESHOLD_ALGORITHM else 40.0,
        )
        voxel_spacing_mm = np.asarray([0.8, 0.8, 2.5], dtype=np.float64)
        raw_kspace = np.fft.fft2(seeded_reconstruction, axes=(-2, -1))
        reconstruction = np.fft.ifft2(raw_kspace, axes=(-2, -1)).real
        tumor_mask = _derive_verified_mask(reconstruction, descriptor)
        tumor_volume_mm3 = _round_float(
            int(tumor_mask.sum()) * float(np.prod(voxel_spacing_mm))
        )
        series.append(
            {
                "series_id": series_id,
                "series_uid": series_uid,
                "sequence_id": _demo_sequence_id(descriptor.profile_name),
                "ai_model_id": "raw-mri-ai-v4",
                "raw_real": _float_array_to_list(raw_kspace.real),
                "raw_imag": _float_array_to_list(raw_kspace.imag),
                "ai_reconstruction": _float_array_to_list(reconstruction),
                "tumor_mask": tumor_mask.tolist(),
                "voxel_spacing_mm": _float_array_to_list(voxel_spacing_mm),
                "tumor_volume_mm3": tumor_volume_mm3,
                "verification_profile_descriptor": build_verification_profile_contract(
                    descriptor.profile_name
                ),
            }
        )
    return {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "case_id": case_id,
        "patient_pseudonym": patient_pseudonym,
        "study_uid": study_uid,
        "scanner_id": scanner_id,
        "release_status": RELEASE_STATUS_PENDING,
        "series": series,
        "cross_series_policy": {
            "primary_series_id": "t1-primary",
            "supporting_series_ids": ["t2-supporting", "flair-supporting"],
            "required_series_ids": ["t1-primary", "t2-supporting", "flair-supporting"],
            "geometry_rule": MULTI_SERIES_GEOMETRY_RULE,
            "corroboration_rule": MULTI_SERIES_CORROBORATION_RULE,
        },
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
    demo_shape = list(_build_demo_reconstruction(descriptor).shape)
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
            details={"spacing_mm": [0.8, 0.8, 2.5], "shape": demo_shape},
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


def build_multi_series_demo_authority_sources() -> dict[str, tuple[AuthoritySource, ...]]:
    """Return authority anchors for each demo study series."""
    state = build_multi_series_demo_case()
    sources: dict[str, tuple[AuthoritySource, ...]] = {}
    for series_entry in state["series"]:
        descriptor = verification_profile_descriptor_from_state(series_entry)
        base_sources = build_demo_authority_sources(
            case_id=f"{state['case_id']}-{series_entry['series_id']}",
            profile_name=descriptor.profile_name,
        )
        sources[str(series_entry["series_id"])] = tuple(
            _series_scoped_authority_source(series_entry["series_id"], source)
            for source in base_sources
        )
    return sources


def verification_profile_descriptor_from_state(state: Mapping[str, Any]) -> Any:
    """Resolve an embedded profile descriptor payload or fall back to the registry."""
    payload = state.get("verification_profile_descriptor")
    if isinstance(payload, Mapping):
        return verification_profile_descriptor_from_dict(dict(payload))
    return get_verification_profile(resolve_profile_name(profile_name=None, state=dict(state)))


def _normalise_hashable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_hashable_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, np.ndarray):
        return _normalise_hashable_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_normalise_hashable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_input_state_hash(state: Mapping[str, Any]) -> str | None:
    try:
        return hash_canonical_json(_normalise_hashable_value(dict(state)))
    except Exception:
        return None


def _default_success_modifiers() -> dict[str, Any]:
    return {
        "near_proof_threshold": False,
        "has_non_blocking_advisory": False,
        "near_boundary_metrics": [],
        "advisory_reason_codes": [],
        "near_boundary_series_ids": [],
        "advisory_series_ids": [],
    }


def _decision_lineage_entry(
    pipeline_phase: str,
    outcome: str,
    *,
    proof_stage: str | None = None,
) -> dict[str, str]:
    entry = {"pipeline_phase": pipeline_phase, "outcome": outcome}
    if proof_stage is not None:
        entry["proof_stage"] = proof_stage
    return entry


def _decision_reason(
    *,
    code: str,
    pipeline_phase: str,
    severity: str,
    message: str,
    band: str,
    proof_stage: str | None = None,
) -> dict[str, Any]:
    item = {
        "code": code,
        "pipeline_phase": pipeline_phase,
        "severity": severity,
        "message": message,
        "_band": band,
    }
    if proof_stage is not None:
        item["proof_stage"] = proof_stage
    return item


def _decision_reason_sort_key(reason: Mapping[str, Any]) -> tuple[int, str, str, str, str]:
    return (
        _DECISION_REASON_BAND_ORDER[str(reason["_band"])],
        str(reason["code"]),
        str(reason["pipeline_phase"]),
        str(reason.get("proof_stage", "")),
        str(reason["message"]),
    )


def _strip_decision_reason_metadata(reason: Mapping[str, Any]) -> dict[str, str]:
    item = {
        "code": str(reason["code"]),
        "pipeline_phase": str(reason["pipeline_phase"]),
        "severity": str(reason["severity"]),
        "message": str(reason["message"]),
    }
    if "proof_stage" in reason:
        item["proof_stage"] = str(reason["proof_stage"])
    return item


def _ordered_unique_reason_codes(reasons: list[dict[str, str]]) -> list[str]:
    ordered: list[str] = []
    for reason in reasons:
        code = str(reason["code"])
        if code not in ordered:
            ordered.append(code)
    return ordered


def _build_decision(
    *,
    final_disposition: str,
    decision_class: str,
    severity: str,
    pipeline_phase: str,
    reasons: list[dict[str, Any]],
    success_modifiers: dict[str, Any],
    decision_lineage: list[dict[str, str]],
    proof_stage: str | None = None,
) -> dict[str, Any]:
    ordered_reasons = [
        _strip_decision_reason_metadata(reason)
        for reason in sorted(reasons, key=_decision_reason_sort_key)
    ]
    decision = {
        "final_disposition": final_disposition,
        "decision_class": decision_class,
        "severity": severity,
        "pipeline_phase": pipeline_phase,
        "reason_codes": _ordered_unique_reason_codes(ordered_reasons),
        "decision_reasons": ordered_reasons,
        "success_modifiers": deepcopy(success_modifiers),
        "decision_lineage": deepcopy(decision_lineage),
    }
    if proof_stage is not None:
        decision["proof_stage"] = proof_stage
    return decision


def _resolve_report_descriptor(
    state: Mapping[str, Any],
    *,
    profile_name: str,
    explicit_profile_name: bool,
) -> tuple[Any, dict[str, Any] | None]:
    try:
        descriptor = get_verification_profile(profile_name)
    except ValueError as exc:
        if explicit_profile_name:
            raise
        descriptor = get_verification_profile(DEFAULT_PROFILE_NAME)
        return descriptor, _decision_reason(
            code="unknown_runtime_profile",
            pipeline_phase=EVALUATION_STAGE_INTAKE_VALIDATION,
            severity=DECISION_SEVERITY_ERROR,
            message=str(exc),
            band=DECISION_REASON_BAND_PRIMARY,
        )
    if explicit_profile_name:
        return descriptor, None
    payload = state.get("verification_profile_descriptor")
    if not isinstance(payload, Mapping):
        return descriptor, None
    try:
        return verification_profile_descriptor_from_dict(dict(payload)), None
    except ValueError as exc:
        return descriptor, _decision_reason(
            code="invalid_profile_descriptor",
            pipeline_phase=EVALUATION_STAGE_INTAKE_VALIDATION,
            severity=DECISION_SEVERITY_ERROR,
            message=str(exc),
            band=DECISION_REASON_BAND_PRIMARY,
        )


def _structural_profile_contract(descriptor: Any) -> dict[str, Any]:
    return build_verification_profile_contract(descriptor.profile_name)


def _build_structural_verification_pipeline(
    state: Mapping[str, Any],
    structural_checks: list[dict[str, Any]],
    *,
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intake_blocked: bool,
) -> dict[str, Any]:
    structural_status = _tier_status_from_checks(structural_checks)
    proof_core_summary = (
        "Intake validation blocked proof-core construction before proof stages could be evaluated."
        if intake_blocked
        else "Structural validity failed before proof-core stages could be evaluated."
    )
    proof_stages = [
        _proof_stage_result(
            proof_stage=proof_stage,
            title=_proof_stage_title(proof_stage),
            verified=None,
            summary=proof_core_summary,
        )
        for proof_stage in _PROOF_STAGE_ORDER
    ]
    payload = {
        "schema": "canonical_assurance_flow",
        "ordering_semantics": (
            "Canonical assurance-flow schema and evidence-ordering model. "
            "It does not mandate runtime execution topology."
        ),
        "phase_order": [
            PIPELINE_PHASE_INTAKE_VALIDATION,
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            PIPELINE_PHASE_PROOF_CORE,
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
            PIPELINE_PHASE_POLICY_EVALUATION,
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            PIPELINE_PHASE_RELEASE_CONTROL,
        ],
        "phases": {
            PIPELINE_PHASE_INTAKE_VALIDATION: _phase_entry(
                title="Intake Validation",
                status="failed" if intake_blocked else "passed",
                summary=(
                    "Runtime-profile intake blocked report preparation."
                    if intake_blocked
                    else "Input intake completed."
                ),
                blocking=True,
            ),
            PIPELINE_PHASE_STRUCTURAL_VALIDITY: _phase_entry(
                title="Structural Validity",
                status=structural_status,
                summary=(
                    "ICSD structural checks passed for identifiers, numeric fields, voxel geometry, "
                    "and release payload shape."
                    if structural_status == "passed"
                    else "ICSD structural checks failed before proof-core results could be trusted."
                ),
                blocking=True,
                checks=structural_checks,
            ),
            PIPELINE_PHASE_PROOF_CORE: _phase_entry(
                title="Proof Core",
                status="not_evaluated",
                summary=proof_core_summary,
                blocking=True,
                proof_stages=proof_stages,
            ),
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY: _phase_entry(
                title="Clinical Plausibility",
                status="not_evaluated",
                summary="Clinical plausibility was not evaluated because proof-core construction did not run.",
                blocking=False,
                checks=[],
            ),
            PIPELINE_PHASE_POLICY_EVALUATION: _phase_entry(
                title="Policy Evaluation",
                status="not_evaluated",
                summary="Policy evaluation was not reached.",
                blocking=True,
            ),
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION: _phase_entry(
                title="Multi-Series Aggregation",
                status="not_applicable",
                summary="Single-series reports do not execute study-level aggregation.",
                blocking=True,
            ),
            PIPELINE_PHASE_RELEASE_CONTROL: _phase_entry(
                title="Release Control",
                status="blocked",
                summary="Release control blocked downstream release.",
                blocking=True,
            ),
        },
    }
    payload["operational_subphases"] = _build_operational_subphases(
        state=state,
        analysis=analysis,
        manifest=manifest,
        structural_checks=structural_checks,
        intake_blocked=intake_blocked,
        proof_stages=proof_stages,
    )
    return payload


def _build_minimal_structural_analysis(state: Mapping[str, Any]) -> dict[str, Any]:
    case_id = state.get("case_id")
    return {
        "case_id": case_id if isinstance(case_id, str) and case_id.strip() else None,
        "input_state_hash": _safe_input_state_hash(state),
        "raw_hash": None,
        "ai_reconstruction_hash": None,
        "derived_reconstruction_hash": None,
        "ai_mask_hash": None,
        "verified_mask_hash": None,
        "computed_tumor_volume_mm3": None,
        "mask_voxel_delta": 0,
        "rejected_region_count": 0,
        "reconstruction_mismatch_voxel_count": 0,
        "verified_mask_voxel_count": 0,
        "max_ai_abs_error": None,
        "max_measurement_abs_error": None,
    }


def _build_structural_failure_reasons(
    decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "code": reason["code"],
            "category": STRUCTURAL_FAILURE_CATEGORY,
            "summary": reason["message"],
            "delta": None,
            "message": reason["message"],
            "details": {
                "pipeline_phase": reason["pipeline_phase"],
                **(
                    {"proof_stage": reason["proof_stage"]}
                    if "proof_stage" in reason
                    else {}
                ),
            },
        }
        for reason in decision["decision_reasons"]
    ]


def _build_structural_block_lineage(
    structural_checks: list[dict[str, Any]],
    *,
    pipeline_phase: str,
) -> list[dict[str, str]]:
    structural_outcome = _tier_status_from_checks(structural_checks)
    intake_outcome = "failed" if pipeline_phase == EVALUATION_STAGE_INTAKE_VALIDATION else "passed"
    return [
        _decision_lineage_entry(PIPELINE_PHASE_INTAKE_VALIDATION, intake_outcome),
        _decision_lineage_entry(
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            "not_evaluated"
            if pipeline_phase == EVALUATION_STAGE_INTAKE_VALIDATION and structural_outcome == "passed"
            else structural_outcome,
        ),
        *[
            _decision_lineage_entry(
                PIPELINE_PHASE_PROOF_CORE,
                "not_evaluated",
                proof_stage=proof_stage,
            )
            for proof_stage in _PROOF_STAGE_ORDER
        ],
        _decision_lineage_entry(PIPELINE_PHASE_CLINICAL_PLAUSIBILITY, "not_evaluated"),
        _decision_lineage_entry(PIPELINE_PHASE_POLICY_EVALUATION, "not_evaluated"),
        _decision_lineage_entry(PIPELINE_PHASE_MULTI_SERIES_AGGREGATION, "not_applicable"),
        _decision_lineage_entry(PIPELINE_PHASE_RELEASE_CONTROL, "blocked"),
    ]


def _build_structural_block_report(
    state: Mapping[str, Any],
    *,
    descriptor: Any,
    structural_checks: list[dict[str, Any]],
    pipeline_phase: str,
    primary_reason: dict[str, Any],
    secondary_reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = [primary_reason, *secondary_reasons]
    decision_summary = {
        "proof_status": "failed",
        "profile_policy_status": "blocked",
        "decision_class": "blocked",
        "block_category": STRUCTURAL_FAILURE_CATEGORY,
        "final_release_allowed": False,
        "operator_review_required": False,
    }
    analysis = _build_minimal_structural_analysis(state)
    profile_contract = _structural_profile_contract(descriptor)
    manifest = {
        "profile_name": profile_contract["profile_name"],
        "profile_revision": profile_contract["profile_revision"],
        "profile_digest": profile_contract["profile_digest"],
        "icsd_policy_id": profile_contract["icsd_policy_id"],
        "icsd_policy_version": profile_contract["icsd_policy_version"],
        "verification_profile_descriptor": profile_contract,
        "performance_expectations": profile_contract["performance_expectations"],
        "input_state_hash": analysis["input_state_hash"],
        "decision_summary": deepcopy(decision_summary),
    }
    verification_pipeline = _build_structural_verification_pipeline(
        state,
        structural_checks,
        analysis=analysis,
        manifest=manifest,
        intake_blocked=pipeline_phase == EVALUATION_STAGE_INTAKE_VALIDATION,
    )
    decision = _build_decision(
        final_disposition="blocked",
        decision_class=DECISION_CLASS_BLOCKED_STRUCTURAL,
        severity=DECISION_SEVERITY_ERROR,
        pipeline_phase=pipeline_phase,
        reasons=reasons,
        success_modifiers=_default_success_modifiers(),
        decision_lineage=_build_structural_block_lineage(
            structural_checks,
            pipeline_phase=pipeline_phase,
        ),
    )
    manifest["decision"] = deepcopy(decision)
    manifest["verification_pipeline"] = deepcopy(verification_pipeline)
    failure_reasons = _build_structural_failure_reasons(decision)
    return {
        "profile": profile_contract,
        "analysis": analysis,
        "contracts": {
            "verification_profile_descriptor": profile_contract,
        },
        "manifest": manifest,
        "verification_pipeline": verification_pipeline,
        "failure_reasons": failure_reasons,
        "decision_summary": decision_summary,
        "decision": decision,
        "release_eligible": False,
    }


def _prepare_single_series_report(
    state: Mapping[str, Any],
    *,
    profile_name: str,
    explicit_profile_name: bool,
) -> tuple[Any | None, dict[str, Any] | None]:
    descriptor, intake_reason = _resolve_report_descriptor(
        state,
        profile_name=profile_name,
        explicit_profile_name=explicit_profile_name,
    )
    structural_checks = _build_structural_validity_checks(state)
    if intake_reason is not None:
        failed_structural_checks = [
            _decision_reason(
                code=str(check["code"]),
                pipeline_phase=EVALUATION_STAGE_STRUCTURAL_VALIDITY,
                severity=DECISION_SEVERITY_ERROR,
                message=str(check["message"]),
                band=DECISION_REASON_BAND_SECONDARY_MODIFIER,
            )
            for check in structural_checks
            if check["status"] == "failed"
        ]
        return descriptor, _build_structural_block_report(
            state,
            descriptor=descriptor,
            structural_checks=structural_checks,
            pipeline_phase=EVALUATION_STAGE_INTAKE_VALIDATION,
            primary_reason=intake_reason,
            secondary_reasons=failed_structural_checks,
        )
    failed_checks = [check for check in structural_checks if check["status"] == "failed"]
    if failed_checks:
        primary_check = failed_checks[0]
        secondary_checks = sorted(failed_checks[1:], key=lambda item: str(item["code"]))
        return descriptor, _build_structural_block_report(
            state,
            descriptor=descriptor,
            structural_checks=structural_checks,
            pipeline_phase=EVALUATION_STAGE_STRUCTURAL_VALIDITY,
            primary_reason=_decision_reason(
                code=str(primary_check["code"]),
                pipeline_phase=EVALUATION_STAGE_STRUCTURAL_VALIDITY,
                severity=DECISION_SEVERITY_ERROR,
                message=str(primary_check["message"]),
                band=DECISION_REASON_BAND_PRIMARY,
            ),
            secondary_reasons=[
                _decision_reason(
                    code=str(check["code"]),
                    pipeline_phase=EVALUATION_STAGE_STRUCTURAL_VALIDITY,
                    severity=DECISION_SEVERITY_ERROR,
                    message=str(check["message"]),
                    band=DECISION_REASON_BAND_SAME_STAGE,
                )
                for check in secondary_checks
            ],
        )
    return descriptor, None


def _reconstruction_with_hot_voxels(
    shape: tuple[int, int, int],
    hot_voxels: tuple[tuple[int, int, int], ...],
    *,
    hot_value: float,
) -> np.ndarray:
    reconstruction = np.zeros(shape, dtype=np.float64)
    for z_index, y_index, x_index in hot_voxels:
        reconstruction[z_index, y_index, x_index] = hot_value
    return reconstruction


def _series_scoped_authority_source(
    series_id: str,
    source: AuthoritySource | Mapping[str, Any],
) -> AuthoritySource:
    if isinstance(source, AuthoritySource):
        details = dict(source.details)
        details["series_id"] = str(series_id)
        return AuthoritySource(
            source_type=str(source.source_type),
            source_id=str(source.source_id),
            reference=str(source.reference),
            details=details,
        )
    details = dict(source.get("details", {}))
    details["series_id"] = str(series_id)
    return AuthoritySource(
        source_type=str(source["source_type"]),
        source_id=str(source["source_id"]),
        reference=str(source["reference"]),
        details=details,
    )


def _build_multi_series_context(state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("case_kind") != MULTI_SERIES_CASE_KIND:
        raise ValueError(
            f"Multi-series cases must declare case_kind={MULTI_SERIES_CASE_KIND!r}."
        )
    for field_name in ("case_id", "patient_pseudonym", "study_uid", "scanner_id"):
        value = state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field {field_name!r} must be a non-empty string.")
    release_status = state.get("release_status")
    if release_status not in {RELEASE_STATUS_PENDING, RELEASE_STATUS_RELEASED}:
        raise ValueError("release_status must be a known MRI guardian state.")
    series_entries = state.get("series")
    if not isinstance(series_entries, list) or not series_entries:
        raise ValueError("Multi-series cases must contain a non-empty series list.")
    series_by_id: dict[str, dict[str, Any]] = {}
    series_order: list[str] = []
    duplicate_series_ids: set[str] = set()
    for entry in series_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each multi-series entry must be a mapping.")
        series_id = entry.get("series_id")
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("Each multi-series entry must contain a non-empty series_id.")
        normalized_id = series_id.strip()
        if normalized_id in series_by_id:
            duplicate_series_ids.add(normalized_id)
            continue
        series_by_id[normalized_id] = dict(entry)
        series_order.append(normalized_id)
    if duplicate_series_ids:
        duplicate_list = ", ".join(sorted(duplicate_series_ids))
        raise ValueError(
            "Multi-series study shape is invalid: duplicate payload series_id values are "
            f"not allowed ({duplicate_list})."
        )
    policy = state.get("cross_series_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Multi-series cases must contain a cross_series_policy mapping.")
    primary_series_id = policy.get("primary_series_id")
    if not isinstance(primary_series_id, str) or not primary_series_id.strip():
        raise ValueError("cross_series_policy.primary_series_id must be a non-empty string.")
    supporting_series_ids = policy.get("supporting_series_ids")
    if not isinstance(supporting_series_ids, list) or not supporting_series_ids:
        raise ValueError(
            "cross_series_policy.supporting_series_ids must be a non-empty list for "
            f"{MULTI_SERIES_CASE_KIND!r}."
        )
    required_series_ids = policy.get("required_series_ids")
    if not isinstance(required_series_ids, list) or not required_series_ids:
        raise ValueError("cross_series_policy.required_series_ids must be a non-empty list.")
    if any(
        not isinstance(value, str) or not value.strip() for value in supporting_series_ids
    ):
        raise ValueError(
            "cross_series_policy.supporting_series_ids entries must be non-empty strings."
        )
    if any(
        not isinstance(value, str) or not value.strip() for value in required_series_ids
    ):
        raise ValueError(
            "cross_series_policy.required_series_ids entries must be non-empty strings."
        )
    declared_supporting = tuple(value.strip() for value in supporting_series_ids)
    declared_required = tuple(value.strip() for value in required_series_ids)
    if len(set(declared_supporting)) != len(declared_supporting):
        raise ValueError("cross_series_policy.supporting_series_ids must be unique.")
    if len(set(declared_required)) != len(declared_required):
        raise ValueError("cross_series_policy.required_series_ids must be unique.")
    primary_series_id = str(primary_series_id).strip()
    if primary_series_id in set(declared_supporting):
        raise ValueError(
            "cross_series_policy.supporting_series_ids must not include primary_series_id."
        )
    if declared_required.count(primary_series_id) != 1:
        raise ValueError(
            "cross_series_policy.primary_series_id must appear exactly once in required_series_ids."
        )
    for series_id in declared_supporting:
        if declared_required.count(series_id) != 1:
            raise ValueError(
                "Each supporting series_id must appear exactly once in required_series_ids."
            )
    if set(declared_required) != ({primary_series_id} | set(declared_supporting)):
        raise ValueError(
            "For this milestone, required_series_ids must resolve to exactly the primary "
            "series plus every supporting series."
        )
    if policy.get("geometry_rule") != MULTI_SERIES_GEOMETRY_RULE:
        raise ValueError(
            "cross_series_policy.geometry_rule must be "
            f"{MULTI_SERIES_GEOMETRY_RULE!r}."
        )
    if policy.get("corroboration_rule") != MULTI_SERIES_CORROBORATION_RULE:
        raise ValueError(
            "cross_series_policy.corroboration_rule must be "
            f"{MULTI_SERIES_CORROBORATION_RULE!r}."
        )
    canonical_required = tuple(
        series_id for series_id in series_order if series_id in set(declared_required)
    )
    canonical_supporting = tuple(
        series_id for series_id in series_order if series_id in set(declared_supporting)
    )
    missing_required = tuple(
        series_id for series_id in declared_required if series_id not in series_by_id
    )
    auxiliary_series_ids = tuple(
        series_id for series_id in series_order if series_id not in set(canonical_required)
    )
    policy_contract = build_cross_series_verification_policy(
        case_id=str(state["case_id"]),
        primary_series_id=primary_series_id,
        supporting_series_ids=list(canonical_supporting),
        required_series_ids=list(canonical_required),
        geometry_rule=MULTI_SERIES_GEOMETRY_RULE,
        corroboration_rule=MULTI_SERIES_CORROBORATION_RULE,
    )
    prior_verified_contexts_by_series = state.get("prior_verified_contexts_by_series")
    if prior_verified_contexts_by_series is None:
        canonical_prior_verified_contexts_by_series: dict[str, Any] = {}
    elif not isinstance(prior_verified_contexts_by_series, Mapping):
        raise ValueError("prior_verified_contexts_by_series must be an object keyed by series_id.")
    else:
        canonical_prior_verified_contexts_by_series = {}
        for series_id, payload in prior_verified_contexts_by_series.items():
            if not isinstance(series_id, str) or not series_id.strip():
                raise ValueError("prior_verified_contexts_by_series keys must be non-empty strings.")
            canonical_prior_verified_contexts_by_series[series_id.strip()] = deepcopy(payload)
    prior_verified_context_by_series = state.get("prior_verified_context_by_series")
    if prior_verified_context_by_series is None:
        canonical_prior_verified_context_by_series: dict[str, Any] = {}
    elif not isinstance(prior_verified_context_by_series, Mapping):
        raise ValueError("prior_verified_context_by_series must be an object keyed by series_id.")
    else:
        canonical_prior_verified_context_by_series = {}
        for series_id, payload in prior_verified_context_by_series.items():
            if not isinstance(series_id, str) or not series_id.strip():
                raise ValueError("prior_verified_context_by_series keys must be non-empty strings.")
            canonical_prior_verified_context_by_series[series_id.strip()] = deepcopy(payload)
    return {
        "case_id": str(state["case_id"]),
        "patient_pseudonym": str(state["patient_pseudonym"]),
        "study_uid": str(state["study_uid"]),
        "scanner_id": str(state["scanner_id"]),
        "release_status": str(release_status),
        "series_entries": [series_by_id[series_id] for series_id in series_order],
        "series_by_id": series_by_id,
        "series_order": tuple(series_order),
        "primary_series_id": primary_series_id,
        "supporting_series_ids": canonical_supporting,
        "required_series_ids": canonical_required,
        "auxiliary_series_ids": auxiliary_series_ids,
        "missing_required_series_ids": missing_required,
        "cross_series_policy": policy_contract,
        "prior_verified_contexts_by_series": canonical_prior_verified_contexts_by_series,
        "prior_verified_context_by_series": canonical_prior_verified_context_by_series,
        "state": dict(state),
    }


def _build_multi_series_series_state(
    context: Mapping[str, Any],
    series_id: str,
) -> dict[str, Any]:
    entry = dict(context["series_by_id"][series_id])
    payload = {
        "case_id": context["case_id"],
        "patient_pseudonym": context["patient_pseudonym"],
        "study_uid": context["study_uid"],
        "series_uid": entry["series_uid"],
        "scanner_id": context["scanner_id"],
        "sequence_id": entry["sequence_id"],
        "ai_model_id": entry["ai_model_id"],
        "release_status": RELEASE_STATUS_PENDING,
        "raw_real": deepcopy(entry["raw_real"]),
        "raw_imag": deepcopy(entry["raw_imag"]),
        "ai_reconstruction": deepcopy(entry["ai_reconstruction"]),
        "tumor_mask": deepcopy(entry["tumor_mask"]),
        "voxel_spacing_mm": deepcopy(entry["voxel_spacing_mm"]),
        "tumor_volume_mm3": entry["tumor_volume_mm3"],
        "verification_profile_descriptor": deepcopy(entry["verification_profile_descriptor"]),
        "verified_manifest": None,
        "workstation_payload": None,
    }
    if "ai_stack_identity" in entry:
        payload["ai_stack_identity"] = deepcopy(entry["ai_stack_identity"])
    for optional_field in ("study_acquired_at", "study_timestamp_provenance", "spatial_geometry"):
        if optional_field in entry:
            payload[optional_field] = deepcopy(entry[optional_field])
    if series_id in context["prior_verified_contexts_by_series"]:
        payload["prior_verified_contexts"] = deepcopy(
            context["prior_verified_contexts_by_series"][series_id]
        )
    elif series_id in context["prior_verified_context_by_series"]:
        payload["prior_verified_context"] = deepcopy(
            context["prior_verified_context_by_series"][series_id]
        )
    return payload


def _canonical_voxel_spacing_tuple(
    values: list[float] | tuple[float, ...],
) -> tuple[float, float, float]:
    spacing = tuple(_round_float(float(value)) for value in values)
    if len(spacing) != 3:
        raise ValueError("voxel_spacing_mm must contain exactly three elements.")
    return spacing  # type: ignore[return-value]


def _grid_signature_for_series_report(series_report: Mapping[str, Any]) -> dict[str, Any]:
    analysis = series_report["analysis"]
    if analysis.get("derived_reconstruction") is not None:
        shape = tuple(np.asarray(analysis["derived_reconstruction"]).shape)
    else:
        shape = tuple(np.asarray(series_report["series_state"].get("ai_reconstruction", [])).shape)
    spacing = _canonical_voxel_spacing_tuple(series_report["series_state"]["voxel_spacing_mm"])
    payload = {
        "shape": list(shape),
        "canonical_voxel_spacing_mm": list(spacing),
    }
    return {
        "shape": list(shape),
        "canonical_voxel_spacing_mm": list(spacing),
        "signature": hash_canonical_json(payload),
    }


def _build_multi_series_series_report(
    context: Mapping[str, Any],
    series_id: str,
) -> dict[str, Any]:
    series_state = _build_multi_series_series_state(context, series_id)
    single_report = build_case_verification_report(series_state)
    resolved_role = "auxiliary"
    if series_id == context["primary_series_id"]:
        resolved_role = "primary"
    elif series_id in set(context["supporting_series_ids"]):
        resolved_role = "supporting"
    gating_status = "required" if series_id in set(context["required_series_ids"]) else "auxiliary"
    descriptor = verification_profile_descriptor_from_dict(dict(single_report["profile"]))
    expected_payload = None
    if single_report["release_eligible"]:
        expected_payload = _build_workstation_payload(
            series_state,
            single_report["analysis"],
            descriptor,
        )
    primary_reason = "all_blocking_checks_passed"
    if single_report["failure_reasons"]:
        primary_reason = str(single_report["failure_reasons"][0]["summary"])
    elif single_report["decision_summary"]["decision_class"] == DECISION_CLASS_TRIAGE_ONLY:
        primary_reason = "proof_passed_but_profile_is_triage_only"
    elif single_report["decision_summary"]["operator_review_required"]:
        primary_reason = "all_blocking_checks_passed; clinical_plausibility_review_recommended"
    report = {
        **single_report,
        "series_id": series_id,
        "series_uid": series_state["series_uid"],
        "sequence_id": series_state["sequence_id"],
        "series_state": series_state,
        "resolved_role": resolved_role,
        "gating_status": gating_status,
        "series_proof_status": single_report["decision_summary"]["proof_status"],
        "series_release_eligible": bool(single_report["release_eligible"]),
        "downstream_release_allowed": False,
        "expected_workstation_payload": expected_payload,
        "primary_reason_summary": primary_reason,
    }
    report["grid_signature"] = _grid_signature_for_series_report(report)
    return report


def _build_multi_series_geometry_verification(
    *,
    required_series_ids: tuple[str, ...],
    missing_required_series_ids: tuple[str, ...],
    series_reports: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    signatures = [
        {
            "series_id": series_id,
            "resolved_role": series_reports[series_id]["resolved_role"],
            "grid_signature": deepcopy(series_reports[series_id]["grid_signature"]),
        }
        for series_id in required_series_ids
    ]
    if missing_required_series_ids:
        return {
            "evaluated": False,
            "verified": False,
            "rule": MULTI_SERIES_GEOMETRY_RULE,
            "required_series": signatures,
            "missing_required_series_ids": list(missing_required_series_ids),
            "mismatched_series_ids": [],
            "message": "Required-series geometry was not evaluated because required series are missing.",
        }
    if not signatures:
        return {
            "evaluated": False,
            "verified": False,
            "rule": MULTI_SERIES_GEOMETRY_RULE,
            "required_series": [],
            "missing_required_series_ids": [],
            "mismatched_series_ids": [],
            "message": "Required-series geometry was not evaluated because no required series resolved.",
        }
    reference_signature = signatures[0]["grid_signature"]["signature"]
    mismatched = [
        item["series_id"]
        for item in signatures[1:]
        if item["grid_signature"]["signature"] != reference_signature
    ]
    return {
        "evaluated": True,
        "verified": not mismatched,
        "rule": MULTI_SERIES_GEOMETRY_RULE,
        "required_series": signatures,
        "reference_series_id": signatures[0]["series_id"],
        "missing_required_series_ids": [],
        "mismatched_series_ids": mismatched,
        "message": (
            "Required-series geometry is canonicalized onto one exact voxel grid."
            if not mismatched
            else "Required-series geometry does not share one canonical voxel grid."
        ),
    }


def _build_multi_series_lesion_corroboration(
    *,
    primary_series_id: str,
    supporting_series_ids: tuple[str, ...],
    required_series_ids: tuple[str, ...],
    missing_required_series_ids: tuple[str, ...],
    geometry_verification: Mapping[str, Any],
    series_reports: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    required_counts = {
        series_id: int(series_reports[series_id]["analysis"]["verified_mask_voxel_count"])
        for series_id in required_series_ids
    }
    if missing_required_series_ids or not geometry_verification["verified"]:
        return {
            "evaluated": False,
            "verified": False,
            "rule": MULTI_SERIES_CORROBORATION_RULE,
            "primary_series_id": primary_series_id,
            "supporting_series_ids": list(supporting_series_ids),
            "required_series_ids": list(required_series_ids),
            "required_verified_mask_voxel_counts": required_counts,
            "supporting_series": [],
            "message": "Positive-lesion corroboration was not evaluated because required-series geometry did not verify.",
        }
    if any(
        series_reports[series_id]["analysis"].get("verified_mask") is None
        for series_id in required_series_ids
    ):
        return {
            "evaluated": False,
            "verified": False,
            "rule": MULTI_SERIES_CORROBORATION_RULE,
            "primary_series_id": primary_series_id,
            "supporting_series_ids": list(supporting_series_ids),
            "required_series_ids": list(required_series_ids),
            "required_verified_mask_voxel_counts": required_counts,
            "supporting_series": [],
            "message": (
                "Positive-lesion corroboration was not evaluated because at least one "
                "required series did not produce a verified mask."
            ),
        }
    primary_mask = np.asarray(
        series_reports[primary_series_id]["analysis"]["verified_mask"],
        dtype=np.int64,
    )
    primary_count = int(np.count_nonzero(primary_mask))
    supporting_entries: list[dict[str, Any]] = []
    overlapping_supporters = 0
    supporting_positive_without_primary = False
    for series_id in supporting_series_ids:
        supporting_mask = np.asarray(
            series_reports[series_id]["analysis"]["verified_mask"],
            dtype=np.int64,
        )
        supporting_count = int(np.count_nonzero(supporting_mask))
        intersection_count = int(np.count_nonzero((primary_mask == 1) & (supporting_mask == 1)))
        if primary_count == 0 and supporting_count > 0:
            supporting_positive_without_primary = True
        if intersection_count > 0:
            overlapping_supporters += 1
        supporting_entries.append(
            {
                "series_id": series_id,
                "verified_mask_voxel_count": supporting_count,
                "intersection_voxel_count": intersection_count,
                "overlaps_primary": bool(intersection_count > 0),
            }
        )
    if all(count == 0 for count in required_counts.values()):
        verified = True
        message = "Every required verified mask is empty, so positive-lesion corroboration passes."
    elif primary_count == 0 and supporting_positive_without_primary:
        verified = False
        message = (
            "The primary verified mask is empty while at least one required supporting "
            "series contains a positive verified lesion."
        )
    elif primary_count > 0 and overlapping_supporters > 0:
        verified = True
        message = (
            "The primary verified lesion is corroborated by at least one supporting series "
            "with exact coordinate overlap."
        )
    else:
        verified = False
        message = (
            "The primary verified lesion is not corroborated by any supporting series with "
            "exact coordinate overlap."
        )
    return {
        "evaluated": True,
        "verified": verified,
        "rule": MULTI_SERIES_CORROBORATION_RULE,
        "primary_series_id": primary_series_id,
        "supporting_series_ids": list(supporting_series_ids),
        "required_series_ids": list(required_series_ids),
        "primary_verified_mask_voxel_count": primary_count,
        "required_verified_mask_voxel_counts": required_counts,
        "supporting_series": supporting_entries,
        "message": message,
    }


def _build_multi_series_series_manifest(series_report: Mapping[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(series_report["manifest"])
    manifest.update(
        {
            "series_id": series_report["series_id"],
            "series_uid": series_report["series_uid"],
            "sequence_id": series_report["sequence_id"],
            "resolved_role": series_report["resolved_role"],
            "gating_status": series_report["gating_status"],
            "series_proof_status": series_report["series_proof_status"],
            "downstream_release_allowed": series_report["downstream_release_allowed"],
            "primary_reason_summary": series_report["primary_reason_summary"],
            "grid_signature": deepcopy(series_report["grid_signature"]),
            "decision": deepcopy(series_report["decision"]),
        }
    )
    return manifest


def _build_multi_series_failure_reasons(
    *,
    series_reports: list[dict[str, Any]],
    geometry_verification: Mapping[str, Any],
    lesion_corroboration: Mapping[str, Any],
    merged_release_evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    missing_required = list(merged_release_evaluation["missing_required_series_ids"])
    if missing_required:
        reasons.append(
            {
                "code": "missing_required_series",
                "category": "study_shape_failure",
                "summary": "missing_required_series",
                "delta": len(missing_required),
                "message": (
                    "The cross-series policy references required series_id values that are "
                    "not present in the study payload."
                ),
                "details": {"missing_required_series_ids": missing_required},
            }
        )
    required_not_release_eligible = list(
        merged_release_evaluation["required_series_not_release_eligible"]
    )
    if required_not_release_eligible:
        reasons.append(
            {
                "code": "required_series_not_release_eligible",
                "category": "per_series_gate_failure",
                "summary": "required_series_not_release_eligible",
                "delta": len(required_not_release_eligible),
                "message": (
                    "At least one required series failed its individual proof or profile gate."
                ),
                "details": {
                    "series_ids": required_not_release_eligible,
                    "series_summaries": {
                        report["series_id"]: report["primary_reason_summary"]
                        for report in series_reports
                        if report["series_id"] in set(required_not_release_eligible)
                    },
                },
            }
        )
    mismatched_series = list(geometry_verification.get("mismatched_series_ids", []))
    if mismatched_series:
        reasons.append(
            {
                "code": "required_series_geometry_mismatch",
                "category": "cross_series_failure",
                "summary": "required_series_geometry_mismatch",
                "delta": len(mismatched_series),
                "message": (
                    "Required series do not share one canonical voxel grid after spacing "
                    "canonicalization."
                ),
                "details": {"mismatched_series_ids": mismatched_series},
            }
        )
    if lesion_corroboration.get("evaluated") and not lesion_corroboration.get("verified"):
        reasons.append(
            {
                "code": "positive_lesion_corroboration_failed",
                "category": "cross_series_failure",
                "summary": "positive_lesion_corroboration_failed",
                "delta": None,
                "message": str(lesion_corroboration["message"]),
                "details": {
                    "primary_series_id": lesion_corroboration["primary_series_id"],
                    "supporting_series": lesion_corroboration["supporting_series"],
                },
            }
        )
    return reasons


def _build_multi_series_verification_pipeline(
    *,
    series_reports: list[dict[str, Any]],
    missing_required_series_ids: tuple[str, ...],
    geometry_verification: Mapping[str, Any],
    lesion_corroboration: Mapping[str, Any],
    final_release_allowed: bool,
) -> dict[str, Any]:
    required_reports = [report for report in series_reports if report["gating_status"] == "required"]
    required_proof_phases = [
        report["verification_pipeline"]["phases"][PIPELINE_PHASE_PROOF_CORE]
        for report in required_reports
    ]
    required_passed = all(phase["status"] == "passed" for phase in required_proof_phases)
    policy_blocked = any(
        item["decision"]["decision_class"] == DECISION_CLASS_BLOCKED_POLICY
        for item in required_reports
    )
    clinical_statuses = [
        item["verification_pipeline"]["phases"][PIPELINE_PHASE_CLINICAL_PLAUSIBILITY]["status"]
        for item in required_reports
    ]
    clinical_status = "passed"
    if any(status == "failed" for status in clinical_statuses):
        clinical_status = "failed"
    elif any(status == "warning" for status in clinical_statuses):
        clinical_status = "warning"
    elif any(status == "advisory" for status in clinical_statuses):
        clinical_status = "advisory"
    auxiliary_failures = [
        report["series_id"]
        for report in series_reports
        if report["gating_status"] == "auxiliary" and not report["series_release_eligible"]
    ]
    return {
        "schema": "canonical_assurance_flow",
        "ordering_semantics": (
            "Canonical assurance-flow schema and evidence-ordering model. "
            "It does not mandate runtime execution topology."
        ),
        "phase_order": [
            PIPELINE_PHASE_INTAKE_VALIDATION,
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            PIPELINE_PHASE_PROOF_CORE,
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
            PIPELINE_PHASE_POLICY_EVALUATION,
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            PIPELINE_PHASE_RELEASE_CONTROL,
        ],
        "phases": {
            PIPELINE_PHASE_INTAKE_VALIDATION: _phase_entry(
                title="Intake Validation",
                status="passed",
                summary="Study intake completed without blocking descriptor faults.",
                blocking=True,
            ),
            PIPELINE_PHASE_STRUCTURAL_VALIDITY: _phase_entry(
                title="Structural Validity",
                status="passed" if not missing_required_series_ids else "failed",
                summary=(
                    "Study-level payload shape and required-series references are structurally coherent."
                    if not missing_required_series_ids
                    else "The study-level policy references required series that are missing from the payload."
                ),
                blocking=True,
            ),
            PIPELINE_PHASE_PROOF_CORE: _phase_entry(
                title="Proof Core",
                status="passed" if required_passed else "failed",
                summary=(
                    "Every required series passed its own proof core."
                    if required_passed
                    else "At least one required series did not pass its own proof core."
                ),
                blocking=True,
                checks=[
                    {
                        "code": "required_series_proof",
                        "label": "Required series proof core",
                        "status": "passed" if required_passed else "failed",
                        "message": (
                            "Every required series passed its own proof core."
                            if required_passed
                            else "At least one required series did not pass its own proof core."
                        ),
                    }
                ],
            ),
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY: _phase_entry(
                title="Clinical Plausibility",
                status=clinical_status,
                summary=(
                    "Required-series clinical plausibility findings were aggregated from per-series reports."
                ),
                blocking=True,
            ),
            PIPELINE_PHASE_POLICY_EVALUATION: _phase_entry(
                title="Policy Evaluation",
                status="blocked" if policy_blocked else "passed",
                summary=(
                    "A required series carried a policy-level block."
                    if policy_blocked
                    else "No required series carried a policy-level block."
                ),
                blocking=True,
            ),
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION: _phase_entry(
                title="Multi-Series Aggregation",
                status="passed" if final_release_allowed else "failed",
                summary=(
                    "Study aggregation verified required-series geometry, corroboration, and merged release."
                    if final_release_allowed
                    else "Study aggregation did not verify required-series geometry, corroboration, or merged release."
                ),
                blocking=True,
                checks=[
                    {
                        "code": "cross_series_geometry",
                        "label": "Required-series geometry",
                        "status": "passed" if geometry_verification["verified"] else "failed",
                        "message": str(geometry_verification["message"]),
                    },
                    {
                        "code": "lesion_corroboration",
                        "label": "Positive-lesion corroboration",
                        "status": "passed" if lesion_corroboration["verified"] else "failed",
                        "message": str(lesion_corroboration["message"]),
                    },
                    {
                        "code": "merged_release",
                        "label": "Merged release",
                        "status": "passed" if final_release_allowed else "failed",
                        "message": (
                            "The study passed the merged release gate."
                            if final_release_allowed
                            else "The study did not pass the merged release gate."
                        ),
                    },
                ],
            ),
            PIPELINE_PHASE_RELEASE_CONTROL: _phase_entry(
                title="Release Control",
                status="released" if final_release_allowed else "blocked",
                summary=(
                    "The study passed release control."
                    if final_release_allowed
                    else "The study did not pass release control."
                ),
                blocking=True,
            ),
        },
        "auxiliary_failures": auxiliary_failures,
    }


def _build_study_temporal_aggregation_section(
    context: Mapping[str, Any],
    series_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    required_series_ids = tuple(context["required_series_ids"])
    auxiliary_series_ids = tuple(context["auxiliary_series_ids"])
    per_series: list[dict[str, Any]] = []
    top_flags: set[str] = set()
    top_reason_codes: set[str] = set()
    required_series_without_usable_temporal: list[str] = []
    temporal_blocked_required_series: list[str] = []
    required_series_aligned: list[bool] = []
    any_temporal_evaluated = False
    primary_report = next(
        report for report in series_reports if report["series_id"] == context["primary_series_id"]
    )
    allow_blocking = bool(
        primary_report["profile"]["temporal_rules"]["gating_rules"]["allow_blocking"]
    )
    for report in series_reports:
        temporal_trace = report["contracts"].get("temporal_consistency_trace")
        trajectory_trace = report["contracts"].get("temporal_trajectory_trace")
        clinical_plausibility = report.get("manifest", {}).get("clinical_plausibility", {})
        clinical_heuristics = clinical_plausibility.get("heuristics", [])
        blocking_failures = clinical_plausibility.get("blocking_failures", [])
        comparison_status = "not_evaluated"
        comparison_scope = "not_evaluated"
        spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_NONE
        if isinstance(temporal_trace, Mapping):
            comparison_status = str(temporal_trace["comparison_status"])
            comparison_scope = str(temporal_trace["comparison_scope"])
            spatial_comparison_mode = str(temporal_trace["spatial_comparison_mode"])
            top_flags.update(str(item) for item in temporal_trace["flags"])
        if isinstance(trajectory_trace, Mapping):
            top_flags.update(str(item) for item in trajectory_trace["flags"])
        temporal_reason_codes = sorted(
            {
                str(item.get("code"))
                for item in clinical_heuristics
                if isinstance(item, Mapping) and str(item.get("code", "")).startswith("temporal_")
            }
        )
        top_reason_codes.update(temporal_reason_codes)
        usable_temporal = comparison_scope in {TEMPORAL_SCOPE_FULL, TEMPORAL_SCOPE_VOLUME_ONLY} and comparison_status in {
            TEMPORAL_STATUS_ALIGNED,
            TEMPORAL_STATUS_DEGRADED,
        }
        any_temporal_evaluated = any_temporal_evaluated or comparison_status != TEMPORAL_STATUS_NOT_EVALUATED
        if report["gating_status"] == "required":
            required_series_aligned.append(
                comparison_status == TEMPORAL_STATUS_ALIGNED and comparison_scope == TEMPORAL_SCOPE_FULL
            )
            if not usable_temporal:
                required_series_without_usable_temporal.append(report["series_id"])
            if any(
                str(item.get("code", "")).startswith("temporal_")
                for item in blocking_failures
            ):
                temporal_blocked_required_series.append(report["series_id"])
        per_series.append(
            {
                "series_id": report["series_id"],
                "resolved_role": report["resolved_role"],
                "gating_status": report["gating_status"],
                "comparison_status": comparison_status,
                "comparison_scope": comparison_scope,
                "spatial_comparison_mode": spatial_comparison_mode,
                "trajectory_class": (
                    None
                    if not isinstance(trajectory_trace, Mapping)
                    else trajectory_trace["trajectory_class"]
                ),
                "usable_temporal_result": usable_temporal,
                "top_flags": (
                    []
                    if not isinstance(temporal_trace, Mapping)
                    else list(temporal_trace["flags"])
                ),
                "top_reason_codes": temporal_reason_codes,
            }
        )
    all_required_series_temporal_usable = not required_series_without_usable_temporal
    if not any_temporal_evaluated:
        final_temporal_disposition = "not_evaluated"
    elif allow_blocking and all_required_series_temporal_usable and temporal_blocked_required_series:
        final_temporal_disposition = "blocked"
    elif required_series_aligned and all(required_series_aligned):
        final_temporal_disposition = "aligned"
    else:
        final_temporal_disposition = "advisory"
    trace = build_study_temporal_aggregation_trace(
        case_id=str(context["case_id"]),
        required_series_ids=list(required_series_ids),
        auxiliary_series_ids=list(auxiliary_series_ids),
        per_series=per_series,
        final_temporal_disposition=final_temporal_disposition,
        top_flags=sorted(top_flags),
        top_reason_codes=sorted(top_reason_codes),
        gating_summary={
            "allow_blocking": allow_blocking,
            "all_required_series_temporal_usable": all_required_series_temporal_usable,
            "required_series_without_usable_temporal": required_series_without_usable_temporal,
            "temporal_blocked_required_series": temporal_blocked_required_series,
            "any_temporal_evaluated": any_temporal_evaluated,
        },
    )
    return {"trace": trace}


def _build_multi_series_study_report(
    context: Mapping[str, Any],
    series_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    report_by_series_id = {report["series_id"]: report for report in series_reports}
    required_series_ids = tuple(context["required_series_ids"])
    auxiliary_series_ids = tuple(context["auxiliary_series_ids"])
    missing_required = tuple(context["missing_required_series_ids"])
    required_not_release_eligible = [
        series_id
        for series_id in required_series_ids
        if not report_by_series_id[series_id]["series_release_eligible"]
    ]
    geometry_verification = _build_multi_series_geometry_verification(
        required_series_ids=required_series_ids,
        missing_required_series_ids=missing_required,
        series_reports=report_by_series_id,
    )
    lesion_corroboration = _build_multi_series_lesion_corroboration(
        primary_series_id=context["primary_series_id"],
        supporting_series_ids=tuple(context["supporting_series_ids"]),
        required_series_ids=required_series_ids,
        missing_required_series_ids=missing_required,
        geometry_verification=geometry_verification,
        series_reports=report_by_series_id,
    )
    final_release_allowed = (
        not missing_required
        and not required_not_release_eligible
        and bool(geometry_verification["verified"])
        and bool(lesion_corroboration["verified"])
    )
    updated_series_reports: list[dict[str, Any]] = []
    released_series_ids: list[str] = []
    withheld_series_ids: list[str] = []
    for report in series_reports:
        downstream_release_allowed = bool(
            final_release_allowed and report["series_release_eligible"]
        )
        updated_report = {**report, "downstream_release_allowed": downstream_release_allowed}
        updated_series_reports.append(updated_report)
        if downstream_release_allowed:
            released_series_ids.append(report["series_id"])
        else:
            withheld_series_ids.append(report["series_id"])
    study_temporal_aggregation = _build_study_temporal_aggregation_section(
        context,
        updated_series_reports,
    )
    block_category = "none"
    if missing_required:
        block_category = "missing_required_series"
    elif required_not_release_eligible:
        block_category = "required_series_not_release_eligible"
    elif not geometry_verification["verified"]:
        block_category = "required_series_geometry_mismatch"
    elif not lesion_corroboration["verified"]:
        block_category = "positive_lesion_corroboration_failed"
    merged_release_evaluation = {
        "verified": final_release_allowed,
        "required_series_ids": list(required_series_ids),
        "auxiliary_series_ids": list(auxiliary_series_ids),
        "missing_required_series_ids": list(missing_required),
        "required_series_not_release_eligible": list(required_not_release_eligible),
        "released_series_ids": released_series_ids,
        "withheld_series_ids": withheld_series_ids,
        "block_category": block_category,
        "final_release_allowed": final_release_allowed,
    }
    cross_series_trace = build_cross_series_verification_trace(
        case_id=context["case_id"],
        policy=context["cross_series_policy"],
        geometry_verification=geometry_verification,
        lesion_corroboration=lesion_corroboration,
        merged_release_evaluation=merged_release_evaluation,
    )
    verification_profile_descriptor = build_multi_series_verification_profile_descriptor_set(
        case_id=context["case_id"],
        series_descriptors=[
            {
                "series_id": report["series_id"],
                "resolved_role": report["resolved_role"],
                "gating_status": report["gating_status"],
                "verification_profile_descriptor": deepcopy(
                    report["contracts"]["verification_profile_descriptor"]
                ),
            }
            for report in updated_series_reports
        ],
    )
    failure_reasons = _build_multi_series_failure_reasons(
        series_reports=updated_series_reports,
        geometry_verification=geometry_verification,
        lesion_corroboration=lesion_corroboration,
        merged_release_evaluation=merged_release_evaluation,
    )
    decision_summary = {
        "proof_status": "passed" if final_release_allowed else "failed",
        "profile_policy_status": "passed" if final_release_allowed else "blocked",
        "decision_class": "final_release_candidate" if final_release_allowed else "blocked",
        "block_category": block_category,
        "final_release_allowed": final_release_allowed,
        "operator_review_required": any(
            report["decision_summary"]["operator_review_required"]
            for report in updated_series_reports
            if report["downstream_release_allowed"]
        ),
    }
    verification_pipeline = _build_multi_series_verification_pipeline(
        series_reports=updated_series_reports,
        missing_required_series_ids=missing_required,
        geometry_verification=geometry_verification,
        lesion_corroboration=lesion_corroboration,
        final_release_allowed=final_release_allowed,
    )
    manifest = {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "case_id": context["case_id"],
        "patient_pseudonym": context["patient_pseudonym"],
        "study_uid": context["study_uid"],
        "scanner_id": context["scanner_id"],
        "verification_profile_descriptor": verification_profile_descriptor,
        "cross_series_policy": context["cross_series_policy"],
        "cross_series_verification": cross_series_trace,
        "study_temporal_aggregation": study_temporal_aggregation["trace"],
        "series_manifests": [
            _build_multi_series_series_manifest(report) for report in updated_series_reports
        ],
        "decision_summary": decision_summary,
        "verification_pipeline": verification_pipeline,
    }
    report = {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "profile": verification_profile_descriptor,
        "analysis": {
            "required_series_ids": list(required_series_ids),
            "supporting_series_ids": list(context["supporting_series_ids"]),
            "auxiliary_series_ids": list(auxiliary_series_ids),
            "missing_required_series_ids": list(missing_required),
            "released_series_ids": released_series_ids,
            "withheld_series_ids": withheld_series_ids,
            "series": [
                {
                    "series_id": report["series_id"],
                    "resolved_role": report["resolved_role"],
                    "gating_status": report["gating_status"],
                    "series_proof_status": report["series_proof_status"],
                    "series_release_eligible": report["series_release_eligible"],
                    "downstream_release_allowed": report["downstream_release_allowed"],
                    "grid_signature": deepcopy(report["grid_signature"]),
                    "verified_mask_voxel_count": int(
                        report["analysis"]["verified_mask_voxel_count"]
                    ),
                    "computed_tumor_volume_mm3": report["analysis"]["computed_tumor_volume_mm3"],
                }
                for report in updated_series_reports
            ],
            "geometry_verification": geometry_verification,
            "lesion_corroboration": lesion_corroboration,
            "study_temporal_aggregation": study_temporal_aggregation["trace"],
        },
        "contracts": {
            "verification_profile_descriptor": verification_profile_descriptor,
            "cross_series_verification_policy": context["cross_series_policy"],
            "cross_series_verification_trace": cross_series_trace,
            "study_temporal_aggregation_trace": study_temporal_aggregation["trace"],
        },
        "series_reports": updated_series_reports,
        "manifest": manifest,
        "verification_pipeline": verification_pipeline,
        "failure_reasons": failure_reasons,
        "decision_summary": decision_summary,
        "release_eligible": final_release_allowed,
    }
    decision = _build_multi_series_decision(report)
    report["decision"] = decision
    report["manifest"]["decision"] = deepcopy(decision)
    return report


def _build_multi_series_workstation_payload(
    state: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "case_id": state["case_id"],
        "study_uid": state["study_uid"],
        "scanner_id": state["scanner_id"],
        "patient_pseudonym": state["patient_pseudonym"],
        "cross_series_policy": deepcopy(report["contracts"]["cross_series_verification_policy"]),
        "decision_summary": deepcopy(report["decision_summary"]),
        "series_payloads": [
            {
                "series_id": series_report["series_id"],
                "series_uid": series_report["series_uid"],
                "sequence_id": series_report["sequence_id"],
                "resolved_role": series_report["resolved_role"],
                "gating_status": series_report["gating_status"],
                "series_proof_status": series_report["series_proof_status"],
                "downstream_release_allowed": True,
                "workstation_payload": deepcopy(series_report["expected_workstation_payload"]),
            }
            for series_report in report["series_reports"]
            if series_report["downstream_release_allowed"]
        ],
        "withheld_series": [
            {
                "series_id": series_report["series_id"],
                "series_uid": series_report["series_uid"],
                "sequence_id": series_report["sequence_id"],
                "resolved_role": series_report["resolved_role"],
                "gating_status": series_report["gating_status"],
                "series_proof_status": series_report["series_proof_status"],
                "downstream_release_allowed": False,
                "withheld_reason": series_report["primary_reason_summary"],
            }
            for series_report in report["series_reports"]
            if not series_report["downstream_release_allowed"]
        ],
    }


def _resolve_primary_series_descriptor(state: Mapping[str, Any]) -> Any:
    context = _build_multi_series_context(state)
    primary_series_id = context["primary_series_id"]
    if primary_series_id in context["series_by_id"]:
        return verification_profile_descriptor_from_state(context["series_by_id"][primary_series_id])
    return verification_profile_descriptor_from_state(context["series_entries"][0])


def _flatten_multi_series_authority_sources(
    *,
    authority_sources_by_series_id: Mapping[str, Iterable[AuthoritySource | Mapping[str, Any]]],
    series_order: tuple[str, ...],
) -> tuple[AuthoritySource, ...]:
    flattened: list[AuthoritySource] = []
    for series_id in series_order:
        sources = authority_sources_by_series_id[series_id]
        flattened.extend(
            _series_scoped_authority_source(series_id, source) for source in sources
        )
    return tuple(flattened)


def _validate_multi_series_authority_sources_by_series_id(
    *,
    state: Mapping[str, Any],
    authority_sources_by_series_id: Mapping[str, Iterable[AuthoritySource | Mapping[str, Any]]],
) -> None:
    context = _build_multi_series_context(state)
    expected_series_ids = set(context["series_order"])
    provided_series_ids = {str(series_id) for series_id in authority_sources_by_series_id}
    missing_series_ids = sorted(expected_series_ids - provided_series_ids)
    extra_series_ids = sorted(provided_series_ids - expected_series_ids)
    if missing_series_ids:
        raise ValueError(
            "authority_sources_by_series_id is missing entries for payload series: "
            + ", ".join(missing_series_ids)
        )
    if extra_series_ids:
        raise ValueError(
            "authority_sources_by_series_id contains unknown payload series: "
            + ", ".join(extra_series_ids)
        )
    for series_id in context["series_order"]:
        descriptor = verification_profile_descriptor_from_state(context["series_by_id"][series_id])
        resolved_sources = tuple(
            _series_scoped_authority_source(series_id, source)
            for source in authority_sources_by_series_id[series_id]
        )
        seen_types = {source.source_type for source in resolved_sources}
        missing_types = sorted(set(descriptor.required_authority_source_types) - seen_types)
        if missing_types:
            raise ValueError(
                f"authority validation failed for series {series_id!r}: missing authority "
                f"source types {', '.join(missing_types)}."
            )
        if any(source.details.get("series_id") != series_id for source in resolved_sources):
            raise ValueError(
                f"authority validation failed for series {series_id!r}: sources must stay tagged "
                "to the owning series_id."
            )


def _build_demo_reconstruction(descriptor: Any) -> np.ndarray:
    if descriptor.mask_algorithm == MORPHOLOGY_ALGORITHM:
        reconstruction = np.zeros((8, 8, 8), dtype=np.float64)
        reconstruction[2:5, :, :] = 4.0
        return reconstruction
    if descriptor.mask_algorithm == CONNECTED_COMPONENT_ALGORITHM:
        reconstruction = np.zeros((8, 8, 8), dtype=np.float64)
        # Use a higher deterministic signal floor here so the rounded FFT/IFFT demo
        # path stays inside the existing global reconstruction tolerance.
        reconstruction[2:5, 2:4, 2:4] = 40.0
        reconstruction[5, 5:7, 5:7] = 40.0
        return reconstruction
    if descriptor.mask_algorithm == SLICE_CONSISTENCY_ALGORITHM:
        reconstruction = np.zeros((8, 8, 8), dtype=np.float64)
        # Keep one lesion footprint on consecutive slices so the slice-consistency
        # policy has an accepted run, plus one detached single-slice island that
        # the policy rejects while the threshold seed would keep.
        reconstruction[2:4, 2:4, 2:4] = 40.0
        reconstruction[6, 5:7, 5:7] = 40.0
        return reconstruction
    return np.asarray(
        [
            [[0.0, 1.0], [0.0, 4.0]],
            [[0.0, 3.0], [2.0, 0.0]],
        ],
        dtype=np.float64,
    )


def _demo_sequence_id(profile_name: str) -> str:
    if profile_name in {
        "brain_t1_baseline",
        "brain_t1_morphology_review",
        "brain_t1_component_review",
        "brain_t1_slice_review",
    }:
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
                name="ai_stack_identity_valid",
                check=_check_ai_stack_identity,
            ),
            Invariant(
                name="study_acquired_at_valid",
                check=_check_study_acquired_at,
            ),
            Invariant(
                name="study_timestamp_provenance_valid",
                check=_check_study_timestamp_provenance,
            ),
            Invariant(
                name="spatial_geometry_valid",
                check=_check_spatial_geometry,
            ),
            Invariant(
                name="prior_verified_context_valid",
                check=_check_prior_verified_context,
            ),
            Invariant(
                name="prior_verified_contexts_valid",
                check=_check_prior_verified_contexts,
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


def _build_multi_series_release_transition(
    *,
    transparency_log_path: Path,
    default_profile_name: str,
) -> Transition:
    invariants = InvariantSet(
        (
            Invariant(
                name="multi_series_required_strings_present",
                check=_check_multi_series_required_strings,
            ),
            Invariant(
                name="release_status_known",
                check=_check_release_status,
            ),
            Invariant(
                name="multi_series_study_shape_valid",
                check=_check_multi_series_study_shape,
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
        name="release_verified_multi_series_mri_study",
        mutate=_release_multi_series_to_radiologist,
        profile_registry=build_profile_registry(invariants),
        default_profile=default_profile_name,
        authority_validator=_validate_multi_series_flattened_authority_sources,
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
    report = build_case_verification_report(state, profile_name=descriptor.profile_name)
    state["verified_manifest"] = deepcopy(report["manifest"])
    state["workstation_payload"] = _build_workstation_payload(
        state,
        report["analysis"],
        descriptor,
    )
    state["release_status"] = RELEASE_STATUS_RELEASED


def _release_multi_series_to_radiologist(state: dict[str, Any]) -> None:
    report = build_multi_series_case_verification_report(state)
    state["verified_manifest"] = deepcopy(report["manifest"])
    state["workstation_payload"] = _build_multi_series_workstation_payload(state, report)
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


def _validate_multi_series_flattened_authority_sources(
    authority: str,
    authority_sources: tuple[AuthoritySource, ...],
) -> None:
    del authority
    for source in authority_sources:
        if source.details.get("series_id"):
            continue
        raise ValueError(
            "Multi-series authority sources must carry series_id in their details block."
        )


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


def _check_multi_series_required_strings(state: Mapping[str, Any]) -> InvariantFailure | None:
    required_fields = (
        "case_id",
        "patient_pseudonym",
        "study_uid",
        "scanner_id",
    )
    for field_name in required_fields:
        value = state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return InvariantFailure(
                invariant="multi_series_required_strings_present",
                code="missing_string_field",
                message=f"Field {field_name!r} must be a non-empty string.",
                details={"field": field_name},
            )
    return None


def _check_multi_series_study_shape(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        _build_multi_series_context(state)
    except ValueError as exc:
        return InvariantFailure(
            invariant="multi_series_study_shape_valid",
            code="invalid_multi_series_study_shape",
            message=str(exc),
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


def _check_ai_stack_identity(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_ai_stack_identity(state)
    except ValueError as exc:
        return InvariantFailure(
            invariant="ai_stack_identity_valid",
            code="invalid_ai_stack_identity",
            message=str(exc),
            details={"field": "ai_stack_identity"},
        )
    return None


def _check_study_acquired_at(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_study_acquired_at(
            state.get("study_acquired_at"),
            path="state.study_acquired_at",
        )
    except ValueError as exc:
        return InvariantFailure(
            invariant="study_acquired_at_valid",
            code="invalid_study_acquired_at",
            message=str(exc),
            details={"field": "study_acquired_at"},
        )
    return None


def _check_study_timestamp_provenance(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_timestamp_provenance(
            state.get("study_timestamp_provenance", "missing"),
            path="state.study_timestamp_provenance",
            allow_verified_bundle=False,
        )
    except ValueError as exc:
        return InvariantFailure(
            invariant="study_timestamp_provenance_valid",
            code="invalid_study_timestamp_provenance",
            message=str(exc),
            details={"field": "study_timestamp_provenance"},
        )
    return None


def _check_spatial_geometry(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_spatial_geometry(
            state.get("spatial_geometry"),
            path="state.spatial_geometry",
        )
    except ValueError as exc:
        return InvariantFailure(
            invariant="spatial_geometry_valid",
            code="invalid_spatial_geometry",
            message=str(exc),
            details={"field": "spatial_geometry"},
        )
    return None


def _check_prior_verified_context(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_prior_verified_context(state.get("prior_verified_context"))
    except ValueError as exc:
        return InvariantFailure(
            invariant="prior_verified_context_valid",
            code="invalid_prior_verified_context",
            message=str(exc),
            details={"field": "prior_verified_context"},
        )
    return None


def _check_prior_verified_contexts(state: Mapping[str, Any]) -> InvariantFailure | None:
    try:
        canonicalise_prior_verified_contexts(state.get("prior_verified_contexts"))
    except ValueError as exc:
        return InvariantFailure(
            invariant="prior_verified_contexts_valid",
            code="invalid_prior_verified_contexts",
            message=str(exc),
            details={"field": "prior_verified_contexts"},
        )
    return None


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
    elif state.get("case_kind") != MULTI_SERIES_CASE_KIND:
        analysis = report["analysis"]
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
    elif not report["release_eligible"]:
        failures.append(
            InvariantFailure(
                invariant="released_state_matches_mathematical_trace",
                code="release_proof_failed",
                message=(
                    "Released multi-series cases are only representable when the merged "
                    "study gate verifies successfully."
                ),
                details={"decision_summary": report["decision_summary"]},
            )
        )
    workstation_payload = state.get("workstation_payload")
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        expected_workstation_payload = _build_multi_series_workstation_payload(state, report)
    else:
        descriptor = get_verification_profile(
            resolve_profile_name(profile_name=None, state=dict(state))
        )
        expected_workstation_payload = _build_workstation_payload(
            state,
            report["analysis"],
            descriptor,
        )
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
    study_acquired_at = canonicalise_study_acquired_at(
        state.get("study_acquired_at"),
        path="state.study_acquired_at",
    )
    study_timestamp_provenance = canonicalise_timestamp_provenance(
        state.get("study_timestamp_provenance", "missing"),
        path="state.study_timestamp_provenance",
        allow_verified_bundle=False,
    )
    spatial_geometry = canonicalise_spatial_geometry(
        state.get("spatial_geometry"),
        path="state.spatial_geometry",
    )
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
        "verified_mask_empty": verified_mask_voxel_count == 0,
        "verified_mask_component_count": _count_face_connected_components(verified_mask),
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
        "voxel_spacing_mm": _float_array_to_list(voxel_spacing),
        "voxel_volume_mm3": _round_float(voxel_volume_mm3),
        "computed_tumor_volume_mm3": computed_tumor_volume_mm3,
        "study_acquired_at": study_acquired_at,
        "study_timestamp_provenance": study_timestamp_provenance,
        "spatial_geometry": spatial_geometry,
    }


def _count_face_connected_components(mask: np.ndarray) -> int:
    binary_mask = np.asarray(mask, dtype=np.int64) == 1
    if not np.any(binary_mask):
        return 0
    visited = np.zeros(binary_mask.shape, dtype=bool)
    neighbours = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    component_count = 0
    for seed in np.argwhere(binary_mask):
        z_index, y_index, x_index = (int(value) for value in seed)
        if visited[z_index, y_index, x_index]:
            continue
        component_count += 1
        stack = [(z_index, y_index, x_index)]
        visited[z_index, y_index, x_index] = True
        while stack:
            current_z, current_y, current_x = stack.pop()
            for delta_z, delta_y, delta_x in neighbours:
                next_z = current_z + delta_z
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if (
                    next_z < 0
                    or next_y < 0
                    or next_x < 0
                    or next_z >= binary_mask.shape[0]
                    or next_y >= binary_mask.shape[1]
                    or next_x >= binary_mask.shape[2]
                ):
                    continue
                if not binary_mask[next_z, next_y, next_x] or visited[next_z, next_y, next_x]:
                    continue
                visited[next_z, next_y, next_x] = True
                stack.append((next_z, next_y, next_x))
    return component_count


def _temporal_pair_provenance(
    current_provenance: str,
    prior_provenance: str,
) -> str:
    if "untrusted" in {current_provenance, prior_provenance}:
        return TEMPORAL_PAIR_PROVENANCE_UNTRUSTED
    if "missing" in {current_provenance, prior_provenance}:
        return TEMPORAL_PAIR_PROVENANCE_MISSING
    if current_provenance == prior_provenance == "verified_bundle":
        return TEMPORAL_PAIR_PROVENANCE_VERIFIED_BUNDLE
    if current_provenance == prior_provenance == "trusted_study_metadata":
        return TEMPORAL_PAIR_PROVENANCE_TRUSTED_METADATA
    return TEMPORAL_PAIR_PROVENANCE_MIXED


def _temporal_metric_direction(value: float | None) -> str:
    if value is None:
        return "not_evaluated"
    if value == 0.0:
        return "stable_band"
    return "increase" if value > 0 else "decrease"


def _temporal_skipped_metric(metric: str, reason_code: str) -> dict[str, str]:
    return {"metric": metric, "reason_code": reason_code}


def _temporal_prior_identity_fields_present(prior_context: Mapping[str, Any]) -> bool:
    for field_name in (
        "prior_bundle_sha256",
        "prior_case_id",
        "prior_study_uid",
        "prior_series_uid",
        "prior_released_state_hash",
        "prior_verification_report_hash",
    ):
        if not str(prior_context.get(field_name, "")):
            return False
    return True


def _temporal_exact_geometry_match(
    analysis: Mapping[str, Any],
    prior_context: Mapping[str, Any],
) -> bool:
    current_geometry = analysis["spatial_geometry"]
    prior_geometry = prior_context["spatial_geometry"]
    if current_geometry is None or prior_geometry is None:
        return False
    current_shape = tuple(np.asarray(analysis["verified_mask"], dtype=np.int64).shape)
    prior_shape = tuple(np.asarray(prior_context["verified_mask"], dtype=np.int64).shape)
    return (
        current_shape == prior_shape
        and list(analysis["voxel_spacing_mm"]) == list(prior_context["voxel_spacing_mm"])
        and current_geometry["frame_of_reference_uid"] == prior_geometry["frame_of_reference_uid"]
        and current_geometry["affine_ijk_to_mm"] == prior_geometry["affine_ijk_to_mm"]
    )


def _temporal_scalar_basis_trusted(
    analysis: Mapping[str, Any],
    prior_context: Mapping[str, Any],
) -> bool:
    current_spacing = analysis.get("voxel_spacing_mm")
    prior_spacing = prior_context.get("voxel_spacing_mm")
    if not (
        isinstance(current_spacing, list)
        and len(current_spacing) == 3
        and isinstance(prior_spacing, list)
        and len(prior_spacing) == 3
    ):
        return False
    if any(
        type(value) not in {int, float} or float(value) <= 0.0
        for value in [*current_spacing, *prior_spacing]
    ):
        return False
    current_voxel_count = analysis.get("verified_mask_voxel_count")
    prior_voxel_count = prior_context.get("verified_mask_voxel_count")
    if not isinstance(current_voxel_count, int) or not isinstance(prior_voxel_count, int):
        return False
    current_volume = analysis.get("computed_tumor_volume_mm3")
    prior_volume = prior_context.get("verified_volume_mm3")
    if type(current_volume) not in {int, float} or type(prior_volume) not in {int, float}:
        return False
    expected_current_volume = _round_float(
        current_voxel_count
        * float(np.prod(np.asarray(current_spacing, dtype=np.float64)))
    )
    expected_prior_volume = _round_float(
        prior_voxel_count
        * float(np.prod(np.asarray(prior_spacing, dtype=np.float64)))
    )
    return (
        _round_float(float(current_volume)) == expected_current_volume
        and _round_float(float(prior_volume)) == expected_prior_volume
    )


def _mask_centroid_mm(
    mask: np.ndarray,
    affine_ijk_to_mm: list[list[float]],
) -> np.ndarray | None:
    positive_voxels = np.argwhere(np.asarray(mask, dtype=np.int64) == 1)
    if positive_voxels.size == 0:
        return None
    affine = np.asarray(affine_ijk_to_mm, dtype=np.float64)
    homogeneous = np.concatenate(
        [positive_voxels.astype(np.float64), np.ones((positive_voxels.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (affine @ homogeneous.T).T[:, :3]
    return np.mean(transformed, axis=0)


def _temporal_parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")


def _temporal_pair_is_fully_trusted(pair_provenance: str) -> bool:
    return pair_provenance in {
        TEMPORAL_PAIR_PROVENANCE_VERIFIED_BUNDLE,
        TEMPORAL_PAIR_PROVENANCE_TRUSTED_METADATA,
    }


def _temporal_sequence_is_compatible(
    prior_context: Mapping[str, Any],
    descriptor: Any,
) -> bool:
    prior_sequence_family = str(prior_context["prior_profile"]["sequence_family"])
    compatibility_rules = dict(descriptor.temporal_rules.get("sequence_compatibility_rules", {}))
    if bool(descriptor.temporal_rules.get("same_sequence_family_required", True)):
        return prior_sequence_family == descriptor.sequence_family
    allowed = compatibility_rules.get("allowed_prior_sequence_families", [])
    if not isinstance(allowed, list):
        return False
    return prior_sequence_family in {str(item) for item in allowed}


def _temporal_prior_admissibility(
    prior_context: Mapping[str, Any],
    descriptor: Any,
) -> tuple[str, str]:
    if not _temporal_prior_identity_fields_present(prior_context):
        return TEMPORAL_PRIOR_INADMISSIBLE, TEMPORAL_SKIPPED_PRIOR_BUNDLE_IDENTITY_MISSING
    prior_release = dict(prior_context["prior_release"])
    if prior_release["release_status"] != RELEASE_STATUS_RELEASED:
        return TEMPORAL_PRIOR_INADMISSIBLE, TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    if prior_release["final_disposition"] != "released":
        return TEMPORAL_PRIOR_INADMISSIBLE, TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    if prior_release["decision_class"] not in {
        DECISION_CLASS_VERIFIED_CLEAN,
        DECISION_CLASS_VERIFIED_BOUNDARY_HUGGING,
        DECISION_CLASS_VERIFIED_WITH_ADVISORY,
    }:
        return TEMPORAL_PRIOR_INADMISSIBLE, TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    if not _temporal_sequence_is_compatible(prior_context, descriptor):
        return TEMPORAL_PRIOR_INADMISSIBLE, TEMPORAL_SKIPPED_SEQUENCE_FAMILY_MISMATCH
    return TEMPORAL_PRIOR_ADMISSIBLE, "admissible"


def _temporal_prior_sort_key(
    prior_context: Mapping[str, Any],
) -> tuple[bool, str, str, str]:
    timestamp = prior_context.get("study_acquired_at")
    return (
        timestamp is None,
        "" if timestamp is None else str(timestamp),
        str(prior_context["prior_bundle_sha256"]),
        str(prior_context["prior_case_id"]),
    )


def _temporal_collect_prior_contexts(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if state.get("prior_verified_contexts") is not None:
        return canonicalise_prior_verified_contexts(state.get("prior_verified_contexts")) or []
    prior_payload = state.get("prior_verified_context")
    if prior_payload is None:
        return []
    return [build_prior_verified_context(dict(prior_payload))]


def _temporal_shift_slices(length: int, offset: int) -> tuple[slice, slice] | None:
    if offset >= 0:
        src_start = 0
        src_end = length - offset
        dst_start = offset
        dst_end = length
    else:
        src_start = -offset
        src_end = length
        dst_start = 0
        dst_end = length + offset
    if src_end <= src_start or dst_end <= dst_start:
        return None
    return slice(src_start, src_end), slice(dst_start, dst_end)


def _temporal_translate_mask(mask: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
    translated = np.zeros_like(mask, dtype=np.int64)
    z_slices = _temporal_shift_slices(mask.shape[0], int(shift_zyx[0]))
    y_slices = _temporal_shift_slices(mask.shape[1], int(shift_zyx[1]))
    x_slices = _temporal_shift_slices(mask.shape[2], int(shift_zyx[2]))
    if z_slices is None or y_slices is None or x_slices is None:
        return translated
    src_z, dst_z = z_slices
    src_y, dst_y = y_slices
    src_x, dst_x = x_slices
    translated[dst_z, dst_y, dst_x] = np.asarray(mask, dtype=np.int64)[src_z, src_y, src_x]
    return translated


def _temporal_mask_iou(
    current_mask: np.ndarray,
    prior_mask: np.ndarray,
) -> float | None:
    union_voxel_count = int(np.count_nonzero((current_mask == 1) | (prior_mask == 1)))
    if union_voxel_count == 0:
        return None
    return _round_float(
        int(np.count_nonzero((current_mask == 1) & (prior_mask == 1))) / union_voxel_count
    )


def _build_longitudinal_selection_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    single_prior_alias_only = (
        state.get("prior_verified_contexts") is None
        and state.get("prior_verified_context") is not None
    )
    supplied_priors = sorted(_temporal_collect_prior_contexts(state), key=_temporal_prior_sort_key)
    current_dt = _temporal_parse_timestamp(analysis["study_acquired_at"])
    max_selected_priors = int(descriptor.temporal_rules["selection_rules"]["max_selected_priors"])
    degraded_interval_max_days = float(descriptor.temporal_rules["degraded_interval_max_days"])
    fallback_enabled = bool(
        descriptor.temporal_rules["selection_rules"].get("primary_fallback_enabled", False)
    )
    admissible_priors: list[dict[str, Any]] = []
    entry_index: dict[str, dict[str, Any]] = {}
    first_inadmissible_reason_code: str | None = None
    for prior_context in supplied_priors:
        admissibility_status, admissibility_reason_code = _temporal_prior_admissibility(
            prior_context,
            descriptor,
        )
        prior_dt = _temporal_parse_timestamp(prior_context["study_acquired_at"])
        earlier_than_current: bool | None = None
        within_interval_window: bool | None = None
        interval_days_to_current: float | None = None
        if current_dt is not None and prior_dt is not None:
            delta_seconds = (current_dt - prior_dt).total_seconds()
            earlier_than_current = delta_seconds > 0
            if delta_seconds > 0:
                interval_days_to_current = _round_float(delta_seconds / 86400.0)
                within_interval_window = interval_days_to_current <= degraded_interval_max_days
            else:
                within_interval_window = False
        if admissibility_status == TEMPORAL_PRIOR_ADMISSIBLE:
            admissible_priors.append(prior_context)
        elif first_inadmissible_reason_code is None:
            first_inadmissible_reason_code = admissibility_reason_code
        entry_index[str(prior_context["prior_bundle_sha256"])] = {
            "prior_bundle_sha256": str(prior_context["prior_bundle_sha256"]),
            "prior_case_id": str(prior_context["prior_case_id"]),
            "study_acquired_at": prior_context["study_acquired_at"],
            "timestamp_provenance": str(prior_context["timestamp_provenance"]),
            "sequence_family": str(prior_context["prior_profile"]["sequence_family"]),
            "admissibility_status": admissibility_status,
            "admissibility_reason_code": admissibility_reason_code,
            "earlier_than_current": earlier_than_current,
            "within_interval_window": within_interval_window,
            "interval_days_to_current": interval_days_to_current,
            "eligible_for_selection": False,
            "selected_for_trajectory": False,
            "selected_as_primary": False,
        }
    selected_candidates: list[dict[str, Any]] = []
    if current_dt is not None:
        for prior_context in admissible_priors:
            prior_dt = _temporal_parse_timestamp(prior_context["study_acquired_at"])
            if prior_dt is None:
                continue
            delta_seconds = (current_dt - prior_dt).total_seconds()
            if delta_seconds <= 0:
                continue
            interval_days = _round_float(delta_seconds / 86400.0)
            if interval_days > degraded_interval_max_days:
                continue
            entry_index[str(prior_context["prior_bundle_sha256"])]["eligible_for_selection"] = True
            selected_candidates.append(prior_context)
    selected_priors = selected_candidates[-max_selected_priors:]
    primary_prior = selected_priors[-1] if selected_priors else None
    primary_selection_mode = "trajectory_most_recent" if primary_prior is not None else "none"
    primary_unavailable_reason_code = TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT
    if supplied_priors:
        primary_unavailable_reason_code = (
            first_inadmissible_reason_code
            if not admissible_priors and first_inadmissible_reason_code is not None
            else TEMPORAL_SKIPPED_NO_SELECTED_PRIORS
        )
    if primary_prior is None and fallback_enabled and admissible_priors:
        if current_dt is None:
            primary_prior = admissible_priors[-1]
            primary_selection_mode = "fallback_missing_current_chronology"
        elif all(
            _temporal_parse_timestamp(item["study_acquired_at"]) is None for item in admissible_priors
        ):
            primary_prior = admissible_priors[-1]
            primary_selection_mode = "fallback_missing_prior_chronology"
        elif single_prior_alias_only and len(admissible_priors) == 1:
            primary_prior = admissible_priors[0]
            primary_selection_mode = "single_prior_alias"
    selected_hashes = {str(item["prior_bundle_sha256"]) for item in selected_priors}
    primary_hash = None if primary_prior is None else str(primary_prior["prior_bundle_sha256"])
    entries: list[dict[str, Any]] = []
    for prior_context in supplied_priors:
        prior_hash = str(prior_context["prior_bundle_sha256"])
        entry = dict(entry_index[prior_hash])
        entry["selected_for_trajectory"] = prior_hash in selected_hashes
        entry["selected_as_primary"] = primary_hash == prior_hash
        entries.append(entry)
    selection_trace = build_longitudinal_selection_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        supplied_prior_count=len(supplied_priors),
        selected_prior_count=len(selected_priors),
        primary_prior_bundle_sha256=primary_hash,
        primary_selection_mode=primary_selection_mode,
        entries=entries,
    )
    return {
        "supplied_priors": supplied_priors,
        "selected_priors": selected_priors,
        "primary_prior": primary_prior,
        "primary_selection_mode": primary_selection_mode,
        "primary_unavailable_reason_code": (
            primary_unavailable_reason_code if primary_prior is None else "admissible"
        ),
        "trace": selection_trace,
    }


def _build_temporal_trajectory_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    selected_priors: list[dict[str, Any]],
) -> dict[str, Any]:
    current_volume = _round_float(float(analysis["computed_tumor_volume_mm3"]))
    current_component_count = int(analysis["verified_mask_component_count"])
    aggregate_metrics = {
        "oldest_selected_prior_volume_mm3": None,
        "current_volume_mm3": current_volume,
        "aggregate_signed_volume_delta_mm3": None,
        "aggregate_signed_volume_change_percent": None,
        "aggregate_signed_volume_slope_mm3_per_day": None,
        "aggregate_signed_percent_change_per_30d": None,
        "current_component_count": current_component_count,
    }
    if not selected_priors:
        trace = build_temporal_trajectory_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            selected_prior_count=0,
            usable_interval_count=0,
            trajectory_class="not_evaluated",
            aggregate_metrics=aggregate_metrics,
            interval_summaries=[],
            flags=[],
            skipped_metrics=[
                _temporal_skipped_metric(
                    "aggregate_signed_volume_slope_mm3_per_day",
                    TEMPORAL_SKIPPED_NO_SELECTED_PRIORS,
                ),
                _temporal_skipped_metric(
                    "aggregate_signed_percent_change_per_30d",
                    TEMPORAL_SKIPPED_NO_SELECTED_PRIORS,
                ),
            ],
        )
        return {"trace": trace, "heuristics": []}
    timeline = [
        {
            "kind": "prior",
            "label": str(item["prior_case_id"]),
            "bundle_sha256": str(item["prior_bundle_sha256"]),
            "study_acquired_at": item["study_acquired_at"],
            "timestamp_provenance": str(item["timestamp_provenance"]),
            "verified_volume_mm3": _round_float(float(item["verified_volume_mm3"])),
            "component_count": _count_face_connected_components(
                np.asarray(item["verified_mask"], dtype=np.int64)
            ),
        }
        for item in selected_priors
    ]
    timeline.append(
        {
            "kind": "current",
            "label": str(state["case_id"]),
            "bundle_sha256": None,
            "study_acquired_at": analysis["study_acquired_at"],
            "timestamp_provenance": str(analysis["study_timestamp_provenance"]),
            "verified_volume_mm3": current_volume,
            "component_count": current_component_count,
        }
    )
    interval_summaries: list[dict[str, Any]] = []
    usable_deltas: list[float] = []
    usable_days: list[float] = []
    usable_slopes: list[float] = []
    skipped_metrics: list[dict[str, str]] = []
    flags: list[str] = []
    for index in range(len(timeline) - 1):
        previous_entry = timeline[index]
        next_entry = timeline[index + 1]
        pair_provenance = _temporal_pair_provenance(
            str(next_entry["timestamp_provenance"]),
            str(previous_entry["timestamp_provenance"]),
        )
        previous_dt = _temporal_parse_timestamp(previous_entry["study_acquired_at"])
        next_dt = _temporal_parse_timestamp(next_entry["study_acquired_at"])
        interval_days: float | None = None
        signed_volume_slope_mm3_per_day: float | None = None
        signed_volume_change_percent: float | None = None
        usable_for_trend = False
        skip_reason_code: str | None = None
        if previous_dt is None or next_dt is None:
            skip_reason_code = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
        else:
            delta_seconds = (next_dt - previous_dt).total_seconds()
            if delta_seconds < 0:
                skip_reason_code = TEMPORAL_SKIPPED_TIMESTAMP_REVERSED
            elif delta_seconds == 0:
                interval_days = 0.0
                skip_reason_code = TEMPORAL_SKIPPED_ZERO_INTERVAL
            else:
                interval_days = _round_float(delta_seconds / 86400.0)
                if _temporal_pair_is_fully_trusted(pair_provenance):
                    usable_for_trend = True
                elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_MIXED:
                    skip_reason_code = TEMPORAL_SKIPPED_MIXED_PROVENANCE
                elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_UNTRUSTED:
                    skip_reason_code = TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED
                else:
                    skip_reason_code = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
        signed_volume_delta_mm3 = _round_float(
            float(next_entry["verified_volume_mm3"]) - float(previous_entry["verified_volume_mm3"])
        )
        if usable_for_trend and interval_days is not None and interval_days > 0.0:
            signed_volume_slope_mm3_per_day = _round_float(
                signed_volume_delta_mm3 / float(interval_days)
            )
            usable_deltas.append(signed_volume_delta_mm3)
            usable_days.append(float(interval_days))
            usable_slopes.append(float(signed_volume_slope_mm3_per_day))
        if float(previous_entry["verified_volume_mm3"]) > 0.0:
            signed_volume_change_percent = _round_float(
                (signed_volume_delta_mm3 / float(previous_entry["verified_volume_mm3"])) * 100.0
            )
        interval_summaries.append(
            {
                "from_label": previous_entry["label"],
                "to_label": next_entry["label"],
                "from_bundle_sha256": previous_entry["bundle_sha256"],
                "to_bundle_sha256": next_entry["bundle_sha256"],
                "pair_provenance": pair_provenance,
                "interval_days": interval_days,
                "signed_volume_delta_mm3": signed_volume_delta_mm3,
                "signed_volume_change_percent": signed_volume_change_percent,
                "signed_volume_slope_mm3_per_day": signed_volume_slope_mm3_per_day,
                "from_component_count": int(previous_entry["component_count"]),
                "to_component_count": int(next_entry["component_count"]),
                "usable_for_trend": usable_for_trend,
                "skip_reason_code": skip_reason_code,
            }
        )
    oldest_selected_prior = selected_priors[0]
    oldest_selected_prior_volume = _round_float(float(oldest_selected_prior["verified_volume_mm3"]))
    aggregate_metrics["oldest_selected_prior_volume_mm3"] = oldest_selected_prior_volume
    aggregate_metrics["aggregate_signed_volume_delta_mm3"] = _round_float(
        current_volume - oldest_selected_prior_volume
    )
    if oldest_selected_prior_volume > 0.0:
        aggregate_metrics["aggregate_signed_volume_change_percent"] = _round_float(
            (
                float(aggregate_metrics["aggregate_signed_volume_delta_mm3"])
                / oldest_selected_prior_volume
            )
            * 100.0
        )
    else:
        skipped_metrics.append(
            _temporal_skipped_metric(
                "aggregate_signed_percent_change_per_30d",
                TEMPORAL_SKIPPED_OLDEST_SELECTED_PRIOR_ZERO,
            )
        )
    usable_interval_count = len(usable_slopes)
    if usable_interval_count:
        total_usable_days = sum(usable_days)
        aggregate_metrics["aggregate_signed_volume_slope_mm3_per_day"] = _round_float(
            sum(usable_deltas) / total_usable_days
        )
        if (
            aggregate_metrics["aggregate_signed_volume_change_percent"] is not None
            and usable_interval_count == len(interval_summaries)
            and total_usable_days > 0.0
        ):
            aggregate_metrics["aggregate_signed_percent_change_per_30d"] = _round_float(
                float(aggregate_metrics["aggregate_signed_volume_change_percent"])
                * 30.0
                / total_usable_days
            )
    else:
        skipped_metrics.append(
            _temporal_skipped_metric(
                "aggregate_signed_volume_slope_mm3_per_day",
                TEMPORAL_SKIPPED_NO_USABLE_INTERVALS,
            )
        )
        if aggregate_metrics["aggregate_signed_volume_change_percent"] is not None:
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "aggregate_signed_percent_change_per_30d",
                    TEMPORAL_SKIPPED_NO_USABLE_INTERVALS,
                )
            )
    trajectory_class = "not_evaluated"
    if usable_deltas:
        if all(delta == 0.0 for delta in usable_deltas):
            trajectory_class = "monotonic_stable"
        elif all(delta >= 0.0 for delta in usable_deltas) and any(delta > 0.0 for delta in usable_deltas):
            trajectory_class = "monotonic_growth"
        elif all(delta <= 0.0 for delta in usable_deltas) and any(delta < 0.0 for delta in usable_deltas):
            trajectory_class = "monotonic_shrink"
        else:
            trajectory_class = "oscillating"
    component_counts = [
        _count_face_connected_components(np.asarray(item["verified_mask"], dtype=np.int64))
        for item in selected_priors
    ]
    component_counts.append(current_component_count)
    if len(set(component_counts)) > 1:
        flags.append(TEMPORAL_FLAG_TRAJECTORY_COMPONENT_CHANGE)
    if any(_round_float(float(item["verified_volume_mm3"])) == 0.0 for item in selected_priors) and current_volume > 0.0:
        flags.append(TEMPORAL_FLAG_TRAJECTORY_RECURRENCE)
    if usable_interval_count >= int(
        descriptor.temporal_rules["trajectory_rules"]["min_intervals_for_acceleration"]
    ):
        slope_signs = [1 if slope > 0.0 else -1 if slope < 0.0 else 0 for slope in usable_slopes]
        non_zero_signs = [sign for sign in slope_signs if sign != 0]
        if any(
            non_zero_signs[index] != non_zero_signs[index - 1]
            for index in range(1, len(non_zero_signs))
        ):
            flags.append(TEMPORAL_FLAG_TRAJECTORY_INFLECTION)
        if non_zero_signs and len(set(non_zero_signs)) == 1 and any(
            abs(usable_slopes[index]) > abs(usable_slopes[index - 1])
            for index in range(1, len(usable_slopes))
        ):
            flags.append(TEMPORAL_FLAG_TRAJECTORY_ACCELERATION)
    trace = build_temporal_trajectory_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        selected_prior_count=len(selected_priors),
        usable_interval_count=usable_interval_count,
        trajectory_class=trajectory_class,
        aggregate_metrics=aggregate_metrics,
        interval_summaries=interval_summaries,
        flags=sorted(set(flags)),
        skipped_metrics=sorted(
            skipped_metrics,
            key=lambda item: (str(item["metric"]), str(item["reason_code"])),
        ),
    )
    heuristics: list[dict[str, Any]] = []
    if TEMPORAL_FLAG_TRAJECTORY_ACCELERATION in flags:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TRAJECTORY_ACCELERATION,
                label="Trajectory acceleration",
                severity="advisory",
                status="advisory",
                message="The selected prior trajectory shows increasing magnitude of change across usable intervals.",
            )
        )
    if TEMPORAL_FLAG_TRAJECTORY_INFLECTION in flags:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TRAJECTORY_INFLECTION,
                label="Trajectory inflection",
                severity="advisory",
                status="advisory",
                message="The selected prior trajectory changes directional sign across usable intervals.",
            )
        )
    if TEMPORAL_FLAG_TRAJECTORY_RECURRENCE in flags:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TRAJECTORY_RECURRENCE,
                label="Recurrence after zero-volume state",
                severity="warning",
                status="warning",
                message="A selected prior recorded zero verified volume before a later positive verified volume reappeared.",
            )
        )
    if TEMPORAL_FLAG_TRAJECTORY_COMPONENT_CHANGE in flags:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TRAJECTORY_COMPONENT_CHANGE,
                label="Component-count trajectory change",
                severity="advisory",
                status="advisory",
                message="Component counts vary across the selected longitudinal trajectory.",
            )
        )
    return {"trace": trace, "heuristics": heuristics}


def _build_longitudinal_registration_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    primary_prior: Mapping[str, Any] | None,
    prior_admissibility_status: str,
) -> dict[str, Any]:
    registration_rules = dict(descriptor.temporal_rules.get("registration_rules", {}))
    algorithm = {
        "name": str(registration_rules.get("algorithm", "disabled")),
        "max_translation_voxels": int(registration_rules.get("max_translation_voxels", 0)),
    }
    if primary_prior is None:
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=None,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_EVALUATED,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={
                "attempted": False,
                "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
                "registration_prerequisites_met": False,
                "exact_iou_before_registration": None,
                "best_iou_after_registration": None,
                "candidate_count": 0,
                "spatial_use_allowed": False,
                "blocking_use_allowed": False,
                "minimum_iou_for_spatial_use": _round_float(
                    float(registration_rules.get("minimum_iou_for_spatial_use", 0.0))
                ),
                "minimum_iou_for_blocking": _round_float(
                    float(registration_rules.get("minimum_iou_for_blocking", 0.0))
                ),
            },
        )
        return {"trace": trace, "transformed_prior_mask": None}
    prior_bundle_sha256 = str(primary_prior["prior_bundle_sha256"])
    current_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    prior_mask = np.asarray(primary_prior["verified_mask"], dtype=np.int64)
    minimum_iou_for_spatial_use = _round_float(
        float(registration_rules.get("minimum_iou_for_spatial_use", 0.0))
    )
    minimum_iou_for_blocking = _round_float(
        float(registration_rules.get("minimum_iou_for_blocking", 0.0))
    )
    quality = {
        "attempted": False,
        "reason_code": "registration_not_required",
        "registration_prerequisites_met": False,
        "exact_iou_before_registration": _temporal_mask_iou(current_mask, prior_mask),
        "best_iou_after_registration": None,
        "candidate_count": 0,
        "spatial_use_allowed": False,
        "blocking_use_allowed": False,
        "minimum_iou_for_spatial_use": minimum_iou_for_spatial_use,
        "minimum_iou_for_blocking": minimum_iou_for_blocking,
    }
    if prior_admissibility_status != TEMPORAL_PRIOR_ADMISSIBLE:
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=prior_bundle_sha256,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={**quality, "reason_code": TEMPORAL_REASON_TEMPORAL_PRIOR_UNSUITABLE},
        )
        return {"trace": trace, "transformed_prior_mask": None}
    if not bool(registration_rules.get("enabled", False)):
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=prior_bundle_sha256,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={**quality, "reason_code": "registration_disabled"},
        )
        return {"trace": trace, "transformed_prior_mask": None}
    if _temporal_exact_geometry_match(analysis, primary_prior):
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=prior_bundle_sha256,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={**quality, "reason_code": "exact_geometry_available"},
        )
        return {"trace": trace, "transformed_prior_mask": None}
    current_shape = tuple(current_mask.shape)
    prior_shape = tuple(prior_mask.shape)
    same_spacing = list(analysis["voxel_spacing_mm"]) == list(primary_prior["voxel_spacing_mm"])
    if (
        current_shape != prior_shape
        or not same_spacing
        or analysis.get("spatial_geometry") is None
        or primary_prior.get("spatial_geometry") is None
        or not _temporal_scalar_basis_trusted(analysis, primary_prior)
    ):
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=prior_bundle_sha256,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={**quality, "reason_code": TEMPORAL_SKIPPED_GEOMETRY_NOT_SUFFICIENT},
        )
        return {"trace": trace, "transformed_prior_mask": None}
    if not np.any(current_mask == 1) or not np.any(prior_mask == 1):
        empty_reason = (
            TEMPORAL_SKIPPED_BOTH_MASKS_EMPTY
            if not np.any(current_mask == 1) and not np.any(prior_mask == 1)
            else TEMPORAL_SKIPPED_EMPTY_CURRENT_MASK
            if not np.any(current_mask == 1)
            else TEMPORAL_SKIPPED_EMPTY_PRIOR_MASK
        )
        trace = build_longitudinal_registration_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            prior_bundle_sha256=prior_bundle_sha256,
            status=TEMPORAL_REGISTRATION_STATUS_NOT_APPLICABLE,
            blocking_admissible=False,
            algorithm=algorithm,
            transform=None,
            quality={**quality, "reason_code": empty_reason},
        )
        return {"trace": trace, "transformed_prior_mask": None}
    max_translation_voxels = int(registration_rules["max_translation_voxels"])
    best_shift = (0, 0, 0)
    best_iou = -1.0
    best_translated_mask = prior_mask.copy()
    candidate_count = 0
    for delta_z in range(-max_translation_voxels, max_translation_voxels + 1):
        for delta_y in range(-max_translation_voxels, max_translation_voxels + 1):
            for delta_x in range(-max_translation_voxels, max_translation_voxels + 1):
                candidate_count += 1
                shift = (delta_z, delta_y, delta_x)
                translated_mask = _temporal_translate_mask(prior_mask, shift)
                iou = _temporal_mask_iou(current_mask, translated_mask)
                candidate_iou = -1.0 if iou is None else float(iou)
                current_best_l1 = sum(abs(value) for value in best_shift)
                candidate_l1 = sum(abs(value) for value in shift)
                if (
                    candidate_iou > best_iou
                    or (
                        candidate_iou == best_iou
                        and (
                            candidate_l1 < current_best_l1
                            or (candidate_l1 == current_best_l1 and shift < best_shift)
                        )
                    )
                ):
                    best_iou = candidate_iou
                    best_shift = shift
                    best_translated_mask = translated_mask
    best_iou_value = None if best_iou < 0.0 else _round_float(best_iou)
    spatial_use_allowed = best_iou_value is not None and best_iou_value >= minimum_iou_for_spatial_use
    blocking_use_allowed = best_iou_value is not None and best_iou_value >= minimum_iou_for_blocking
    transform_payload = {
        "type": "integer_translation_zyx_voxels",
        "translation_zyx_voxels": [int(best_shift[0]), int(best_shift[1]), int(best_shift[2])],
        "transform_hash": hash_canonical_json(
            {
                "type": "integer_translation_zyx_voxels",
                "translation_zyx_voxels": [int(best_shift[0]), int(best_shift[1]), int(best_shift[2])],
            }
        ),
    }
    quality = {
        **quality,
        "attempted": True,
        "reason_code": (
            "registration_quality_passed"
            if spatial_use_allowed
            else "registration_quality_below_spatial_threshold"
        ),
        "registration_prerequisites_met": True,
        "best_iou_after_registration": best_iou_value,
        "candidate_count": candidate_count,
        "spatial_use_allowed": spatial_use_allowed,
        "blocking_use_allowed": blocking_use_allowed,
    }
    trace = build_longitudinal_registration_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        prior_bundle_sha256=prior_bundle_sha256,
        status=(
            TEMPORAL_REGISTRATION_STATUS_USED
            if spatial_use_allowed
            else TEMPORAL_REGISTRATION_STATUS_REJECTED
        ),
        blocking_admissible=bool(blocking_use_allowed and spatial_use_allowed),
        algorithm=algorithm,
        transform=transform_payload,
        quality=quality,
    )
    return {
        "trace": trace,
        "transformed_prior_mask": best_translated_mask if spatial_use_allowed else None,
    }


def _build_temporal_consistency_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    prior_payload = state.get("prior_verified_context")
    if prior_payload is None:
        trace = build_temporal_consistency_trace(
            case_id=str(state["case_id"]),
            profile_name=descriptor.profile_name,
            profile_revision=descriptor.profile_revision,
            profile_digest=descriptor.profile_digest(),
            comparison_status=TEMPORAL_STATUS_NOT_EVALUATED,
            comparison_scope=TEMPORAL_SCOPE_NOT_EVALUATED,
            prior_admissibility={
                "status": TEMPORAL_PRIOR_INADMISSIBLE,
                "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
            },
            chronology_assessment={
                "pair_provenance": TEMPORAL_PAIR_PROVENANCE_MISSING,
                "current_timestamp_provenance": analysis["study_timestamp_provenance"],
                "prior_timestamp_provenance": "missing",
                "current_study_acquired_at": analysis["study_acquired_at"],
                "prior_study_acquired_at": None,
                "interval_days": None,
                "ordered": False,
                "time_metrics_available": False,
                "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
            },
            geometry_assessment={
                "comparability_class": TEMPORAL_GEOMETRY_INSUFFICIENT,
                "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
            },
            computed_metrics={
                "signed_volume_delta_mm3": None,
                "absolute_volume_delta_mm3": None,
                "signed_volume_change_percent": None,
                "signed_normalized_percent_change_per_30d": None,
                "volume_change_direction": "not_evaluated",
                "normalized_rate_direction": "not_evaluated",
                "intersection_over_union": None,
                "global_centroid_shift_mm": None,
                "prior_component_count": None,
                "current_component_count": int(analysis["verified_mask_component_count"]),
                "component_count_delta": None,
            },
            flags=[],
            skipped_metrics=[
                _temporal_skipped_metric("signed_volume_delta_mm3", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
                _temporal_skipped_metric(
                    "absolute_volume_delta_mm3", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT
                ),
                _temporal_skipped_metric(
                    "signed_volume_change_percent", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT
                ),
                _temporal_skipped_metric(
                    "signed_normalized_percent_change_per_30d",
                    TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
                ),
                _temporal_skipped_metric("intersection_over_union", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
                _temporal_skipped_metric("global_centroid_shift_mm", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
            ],
        )
        return {
            "prior_verified_context": None,
            "trace": trace,
            "heuristics": [],
            "summary": "No prior verified context was supplied; longitudinal verification was not evaluated.",
        }

    prior_context = build_prior_verified_context(dict(prior_payload))
    prior_admissibility_status = TEMPORAL_PRIOR_ADMISSIBLE
    prior_admissibility_reason = "admissible"
    if not _temporal_prior_identity_fields_present(prior_context):
        prior_admissibility_status = TEMPORAL_PRIOR_INADMISSIBLE
        prior_admissibility_reason = TEMPORAL_SKIPPED_PRIOR_BUNDLE_IDENTITY_MISSING
    elif prior_context["prior_release"]["release_status"] != RELEASE_STATUS_RELEASED:
        prior_admissibility_status = TEMPORAL_PRIOR_INADMISSIBLE
        prior_admissibility_reason = TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    elif prior_context["prior_release"]["final_disposition"] != "released":
        prior_admissibility_status = TEMPORAL_PRIOR_INADMISSIBLE
        prior_admissibility_reason = TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    elif prior_context["prior_release"]["decision_class"] not in {
        DECISION_CLASS_VERIFIED_CLEAN,
        DECISION_CLASS_VERIFIED_BOUNDARY_HUGGING,
        DECISION_CLASS_VERIFIED_WITH_ADVISORY,
    }:
        prior_admissibility_status = TEMPORAL_PRIOR_INADMISSIBLE
        prior_admissibility_reason = TEMPORAL_SKIPPED_PRIOR_NOT_RELEASED
    elif prior_context["prior_profile"]["sequence_family"] != descriptor.sequence_family:
        prior_admissibility_status = TEMPORAL_PRIOR_INADMISSIBLE
        prior_admissibility_reason = TEMPORAL_SKIPPED_SEQUENCE_FAMILY_MISMATCH

    pair_provenance = _temporal_pair_provenance(
        str(analysis["study_timestamp_provenance"]),
        str(prior_context["timestamp_provenance"]),
    )
    chronology_reason = "chronology_available"
    interval_days: float | None = None
    ordered = False
    time_metrics_available = False
    current_timestamp = analysis["study_acquired_at"]
    prior_timestamp = prior_context["study_acquired_at"]
    if pair_provenance == TEMPORAL_PAIR_PROVENANCE_UNTRUSTED:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED
    elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_MISSING:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
    elif current_timestamp is None or prior_timestamp is None:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
    else:
        current_dt = datetime.strptime(str(current_timestamp), "%Y-%m-%dT%H:%M:%SZ")
        prior_dt = datetime.strptime(str(prior_timestamp), "%Y-%m-%dT%H:%M:%SZ")
        delta_seconds = (current_dt - prior_dt).total_seconds()
        if delta_seconds < 0:
            chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_REVERSED
        elif delta_seconds == 0:
            chronology_reason = TEMPORAL_SKIPPED_ZERO_INTERVAL
            interval_days = 0.0
            ordered = True
        else:
            ordered = True
            interval_days = _round_float(delta_seconds / 86400.0)
            if interval_days > float(descriptor.temporal_rules["degraded_interval_max_days"]):
                chronology_reason = TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX
            elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_MIXED:
                chronology_reason = TEMPORAL_SKIPPED_MIXED_PROVENANCE
                time_metrics_available = True
            elif interval_days > float(descriptor.temporal_rules["aligned_interval_max_days"]):
                chronology_reason = "interval_within_degraded_window"
                time_metrics_available = True
            else:
                chronology_reason = "interval_within_aligned_window"
                time_metrics_available = True

    geometry_class = TEMPORAL_GEOMETRY_INSUFFICIENT
    geometry_reason = TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT
    if prior_admissibility_status == TEMPORAL_PRIOR_ADMISSIBLE:
        if _temporal_exact_geometry_match(analysis, prior_context):
            geometry_class = TEMPORAL_GEOMETRY_EXACT_MATCH
            geometry_reason = TEMPORAL_GEOMETRY_REASON_EXACT_MATCH
        elif _temporal_scalar_basis_trusted(analysis, prior_context):
            geometry_class = TEMPORAL_GEOMETRY_VOLUME_ONLY
            geometry_reason = TEMPORAL_GEOMETRY_REASON_NOT_EXACT

    comparison_scope = {
        TEMPORAL_GEOMETRY_EXACT_MATCH: TEMPORAL_SCOPE_FULL,
        TEMPORAL_GEOMETRY_VOLUME_ONLY: TEMPORAL_SCOPE_VOLUME_ONLY,
        TEMPORAL_GEOMETRY_INSUFFICIENT: TEMPORAL_SCOPE_NONE,
    }[geometry_class]
    comparison_status = TEMPORAL_STATUS_ALIGNED
    if prior_admissibility_status == TEMPORAL_PRIOR_INADMISSIBLE:
        comparison_status = TEMPORAL_STATUS_UNSUITABLE
        comparison_scope = TEMPORAL_SCOPE_NONE
    elif chronology_reason in {
        TEMPORAL_SKIPPED_TIMESTAMP_REVERSED,
        TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX,
    }:
        comparison_status = TEMPORAL_STATUS_UNSUITABLE
        comparison_scope = TEMPORAL_SCOPE_NONE
    elif comparison_scope == TEMPORAL_SCOPE_NONE:
        comparison_status = TEMPORAL_STATUS_UNSUITABLE
    elif comparison_scope == TEMPORAL_SCOPE_VOLUME_ONLY:
        comparison_status = TEMPORAL_STATUS_DEGRADED
    elif pair_provenance in {
        TEMPORAL_PAIR_PROVENANCE_MISSING,
        TEMPORAL_PAIR_PROVENANCE_UNTRUSTED,
        TEMPORAL_PAIR_PROVENANCE_MIXED,
    } or chronology_reason in {
        TEMPORAL_SKIPPED_ZERO_INTERVAL,
        "interval_within_degraded_window",
    }:
        comparison_status = TEMPORAL_STATUS_DEGRADED

    current_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    prior_mask = np.asarray(prior_context["verified_mask"], dtype=np.int64)
    signed_volume_delta_mm3: float | None = None
    absolute_volume_delta_mm3: float | None = None
    signed_volume_change_percent: float | None = None
    signed_normalized_percent_change_per_30d: float | None = None
    intersection_over_union: float | None = None
    global_centroid_shift_mm: float | None = None
    skipped_metrics: list[dict[str, str]] = []
    flags: list[str] = []
    if comparison_scope == TEMPORAL_SCOPE_NONE:
        blocking_reason = prior_admissibility_reason
        if chronology_reason == TEMPORAL_SKIPPED_TIMESTAMP_REVERSED:
            blocking_reason = TEMPORAL_SKIPPED_TIMESTAMP_REVERSED
        elif chronology_reason == TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX:
            blocking_reason = TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX
        elif geometry_reason == TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT:
            blocking_reason = TEMPORAL_SKIPPED_GEOMETRY_NOT_SUFFICIENT
        for metric_name in (
            "signed_volume_delta_mm3",
            "absolute_volume_delta_mm3",
            "signed_volume_change_percent",
            "signed_normalized_percent_change_per_30d",
            "intersection_over_union",
            "global_centroid_shift_mm",
        ):
            skipped_metrics.append(_temporal_skipped_metric(metric_name, str(blocking_reason)))
    else:
        signed_volume_delta_mm3 = _round_float(
            float(analysis["computed_tumor_volume_mm3"]) - float(prior_context["verified_volume_mm3"])
        )
        absolute_volume_delta_mm3 = _round_float(abs(float(signed_volume_delta_mm3)))
        if float(prior_context["verified_volume_mm3"]) > 0.0:
            signed_volume_change_percent = _round_float(
                (
                    signed_volume_delta_mm3 / float(prior_context["verified_volume_mm3"])
                )
                * 100.0
            )
            if time_metrics_available and interval_days is not None and interval_days > 0.0:
                signed_normalized_percent_change_per_30d = _round_float(
                    signed_volume_change_percent * 30.0 / float(interval_days)
                )
            else:
                rate_skip_reason = chronology_reason
                if rate_skip_reason in {
                    TEMPORAL_SKIPPED_TIMESTAMP_MISSING,
                    TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED,
                    TEMPORAL_SKIPPED_MIXED_PROVENANCE,
                    TEMPORAL_SKIPPED_ZERO_INTERVAL,
                }:
                    flags.append(TEMPORAL_FLAG_RATE_NOT_EVALUATED)
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "signed_normalized_percent_change_per_30d",
                        str(rate_skip_reason),
                    )
                )
        else:
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "signed_volume_change_percent",
                    TEMPORAL_SKIPPED_ZERO_PRIOR_VOLUME,
                )
            )
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "signed_normalized_percent_change_per_30d",
                    TEMPORAL_SKIPPED_ZERO_PRIOR_VOLUME,
                )
            )
        if comparison_scope == TEMPORAL_SCOPE_FULL:
            union_voxel_count = int(np.count_nonzero((current_mask == 1) | (prior_mask == 1)))
            if union_voxel_count == 0:
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "intersection_over_union",
                        TEMPORAL_SKIPPED_BOTH_MASKS_EMPTY,
                    )
                )
            else:
                intersection_over_union = _round_float(
                    int(np.count_nonzero((current_mask == 1) & (prior_mask == 1)))
                    / union_voxel_count
                )
            current_centroid = None
            prior_centroid = None
            if not np.any(prior_mask == 1):
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "global_centroid_shift_mm",
                        TEMPORAL_SKIPPED_EMPTY_PRIOR_MASK,
                    )
                )
            elif not np.any(current_mask == 1):
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "global_centroid_shift_mm",
                        TEMPORAL_SKIPPED_EMPTY_CURRENT_MASK,
                    )
                )
            else:
                current_centroid = _mask_centroid_mm(
                    current_mask,
                    analysis["spatial_geometry"]["affine_ijk_to_mm"],
                )
                prior_centroid = _mask_centroid_mm(
                    prior_mask,
                    prior_context["spatial_geometry"]["affine_ijk_to_mm"],
                )
                if current_centroid is not None and prior_centroid is not None:
                    global_centroid_shift_mm = _round_float(
                        float(np.linalg.norm(current_centroid - prior_centroid))
                    )
        else:
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "intersection_over_union",
                    TEMPORAL_SKIPPED_GEOMETRY_NOT_EXACT,
                )
            )
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "global_centroid_shift_mm",
                    TEMPORAL_SKIPPED_GEOMETRY_NOT_EXACT,
                )
            )

    prior_component_count = _count_face_connected_components(prior_mask)
    current_component_count = int(analysis["verified_mask_component_count"])
    component_count_delta = abs(current_component_count - prior_component_count)
    if component_count_delta:
        flags.append(TEMPORAL_FLAG_COMPONENT_COUNT_CHANGED)

    computed_metrics = {
        "signed_volume_delta_mm3": signed_volume_delta_mm3,
        "absolute_volume_delta_mm3": absolute_volume_delta_mm3,
        "signed_volume_change_percent": signed_volume_change_percent,
        "signed_normalized_percent_change_per_30d": signed_normalized_percent_change_per_30d,
        "volume_change_direction": _temporal_metric_direction(signed_volume_change_percent),
        "normalized_rate_direction": _temporal_metric_direction(
            signed_normalized_percent_change_per_30d
        ),
        "intersection_over_union": intersection_over_union,
        "global_centroid_shift_mm": global_centroid_shift_mm,
        "prior_component_count": prior_component_count,
        "current_component_count": current_component_count,
        "component_count_delta": component_count_delta,
    }

    temporal_rules = descriptor.temporal_rules["metrics"]
    heuristics: list[dict[str, Any]] = []
    if prior_admissibility_status == TEMPORAL_PRIOR_INADMISSIBLE:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TEMPORAL_PRIOR_UNSUITABLE,
                label="Prior admissibility",
                severity="warning",
                status="warning",
                message=(
                    "The supplied prior context is inadmissible for longitudinal comparison "
                    f"({prior_admissibility_reason})."
                ),
            )
        )
    if chronology_reason == TEMPORAL_SKIPPED_TIMESTAMP_REVERSED:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TIMESTAMP_REVERSED,
                label="Chronology ordering",
                severity="warning",
                status="warning",
                message="The supplied prior timestamp is later than the current case timestamp.",
            )
        )
    if comparison_scope == TEMPORAL_SCOPE_VOLUME_ONLY:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_GEOMETRY_LIMITED,
                label="Geometry comparability",
                severity="advisory",
                status="advisory",
                message="Geometry is not exactly comparable; only scalar volume comparison ran.",
            )
        )
    if comparison_status == TEMPORAL_STATUS_DEGRADED and chronology_reason in {
        TEMPORAL_SKIPPED_TIMESTAMP_MISSING,
        TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED,
        TEMPORAL_SKIPPED_MIXED_PROVENANCE,
        TEMPORAL_SKIPPED_ZERO_INTERVAL,
        "interval_within_degraded_window",
    }:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_INTERVAL_DEGRADED,
                label="Chronology quality",
                severity="advisory",
                status="advisory",
                message=(
                    "Chronology quality is degraded, so time-dependent interpretation is limited."
                ),
            )
        )
    if (
        signed_volume_change_percent is not None
        and abs(float(signed_volume_change_percent))
        >= float(temporal_rules["signed_volume_change_percent"]["warning_abs_threshold"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_VOLUME_WARNING,
                label="Signed volume change percent",
                severity="warning",
                status="warning",
                message=(
                    "Absolute signed volume change exceeded the configured warning threshold "
                    f"({signed_volume_change_percent}%)."
                ),
                observed=signed_volume_change_percent,
                threshold=float(
                    temporal_rules["signed_volume_change_percent"]["warning_abs_threshold"]
                ),
            )
        )
    elif (
        signed_volume_change_percent is not None
        and abs(float(signed_volume_change_percent))
        >= float(temporal_rules["signed_volume_change_percent"]["advisory_abs_threshold"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_VOLUME_WARNING,
                label="Signed volume change percent",
                severity="advisory",
                status="advisory",
                message=(
                    "Absolute signed volume change exceeded the configured advisory threshold "
                    f"({signed_volume_change_percent}%)."
                ),
                observed=signed_volume_change_percent,
                threshold=float(
                    temporal_rules["signed_volume_change_percent"]["advisory_abs_threshold"]
                ),
            )
        )
    if (
        signed_normalized_percent_change_per_30d is not None
        and abs(float(signed_normalized_percent_change_per_30d))
        >= float(
            temporal_rules["signed_normalized_percent_change_per_30d"][
                "warning_abs_threshold"
            ]
        )
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_RATE_WARNING,
                label="Signed normalized percent change per 30d",
                severity="warning",
                status="warning",
                message=(
                    "Absolute signed normalized rate exceeded the configured warning threshold "
                    f"({signed_normalized_percent_change_per_30d}%)."
                ),
                observed=signed_normalized_percent_change_per_30d,
                threshold=float(
                    temporal_rules["signed_normalized_percent_change_per_30d"][
                        "warning_abs_threshold"
                    ]
                ),
            )
        )
    elif (
        signed_normalized_percent_change_per_30d is not None
        and abs(float(signed_normalized_percent_change_per_30d))
        >= float(
            temporal_rules["signed_normalized_percent_change_per_30d"][
                "advisory_abs_threshold"
            ]
        )
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_RATE_WARNING,
                label="Signed normalized percent change per 30d",
                severity="advisory",
                status="advisory",
                message=(
                    "Absolute signed normalized rate exceeded the configured advisory threshold "
                    f"({signed_normalized_percent_change_per_30d}%)."
                ),
                observed=signed_normalized_percent_change_per_30d,
                threshold=float(
                    temporal_rules["signed_normalized_percent_change_per_30d"][
                        "advisory_abs_threshold"
                    ]
                ),
            )
        )
    if (
        intersection_over_union is not None
        and float(intersection_over_union)
        < float(temporal_rules["intersection_over_union"]["warning_floor"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_IOU_WARNING,
                label="Intersection over union",
                severity="warning",
                status="warning",
                message=(
                    "Spatial overlap fell below the configured warning floor "
                    f"({intersection_over_union})."
                ),
                observed=intersection_over_union,
                threshold=float(temporal_rules["intersection_over_union"]["warning_floor"]),
            )
        )
    elif (
        intersection_over_union is not None
        and float(intersection_over_union)
        < float(temporal_rules["intersection_over_union"]["advisory_floor"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_IOU_WARNING,
                label="Intersection over union",
                severity="advisory",
                status="advisory",
                message=(
                    "Spatial overlap fell below the configured advisory floor "
                    f"({intersection_over_union})."
                ),
                observed=intersection_over_union,
                threshold=float(temporal_rules["intersection_over_union"]["advisory_floor"]),
            )
        )
    if (
        global_centroid_shift_mm is not None
        and float(global_centroid_shift_mm)
        > float(temporal_rules["global_centroid_shift_mm"]["warning_threshold"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_CENTROID_WARNING,
                label="Global centroid shift",
                severity="warning",
                status="warning",
                message=(
                    "Centroid shift exceeded the configured warning threshold "
                    f"({global_centroid_shift_mm} mm)."
                ),
                observed=global_centroid_shift_mm,
                threshold=float(
                    temporal_rules["global_centroid_shift_mm"]["warning_threshold"]
                ),
            )
        )
    elif (
        global_centroid_shift_mm is not None
        and float(global_centroid_shift_mm)
        > float(temporal_rules["global_centroid_shift_mm"]["advisory_threshold"])
    ):
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_CENTROID_WARNING,
                label="Global centroid shift",
                severity="advisory",
                status="advisory",
                message=(
                    "Centroid shift exceeded the configured advisory threshold "
                    f"({global_centroid_shift_mm} mm)."
                ),
                observed=global_centroid_shift_mm,
                threshold=float(
                    temporal_rules["global_centroid_shift_mm"]["advisory_threshold"]
                ),
            )
        )

    trace = build_temporal_consistency_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        comparison_status=comparison_status,
        comparison_scope=comparison_scope,
        prior_admissibility={
            "status": prior_admissibility_status,
            "reason_code": prior_admissibility_reason,
        },
        chronology_assessment={
            "pair_provenance": pair_provenance,
            "current_timestamp_provenance": analysis["study_timestamp_provenance"],
            "prior_timestamp_provenance": prior_context["timestamp_provenance"],
            "current_study_acquired_at": current_timestamp,
            "prior_study_acquired_at": prior_timestamp,
            "interval_days": interval_days,
            "ordered": ordered,
            "time_metrics_available": time_metrics_available,
            "reason_code": chronology_reason,
        },
        geometry_assessment={
            "comparability_class": geometry_class,
            "reason_code": geometry_reason,
        },
        computed_metrics=computed_metrics,
        flags=sorted(set(flags)),
        skipped_metrics=sorted(
            skipped_metrics,
            key=lambda item: (str(item["metric"]), str(item["reason_code"])),
        ),
    )
    summary = {
        TEMPORAL_STATUS_ALIGNED: "Longitudinal comparison ran with full aligned temporal context.",
        TEMPORAL_STATUS_DEGRADED: "Longitudinal comparison ran with degraded temporal context.",
        TEMPORAL_STATUS_UNSUITABLE: "Longitudinal comparison was suppressed by v1 temporal policy.",
    }.get(comparison_status, "Longitudinal comparison was not evaluated.")
    return {
        "prior_verified_context": prior_context,
        "trace": trace,
        "heuristics": heuristics,
        "summary": summary,
    }


def _build_temporal_not_evaluated_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    selection_trace: Mapping[str, Any],
    trajectory_trace: Mapping[str, Any],
    trajectory_heuristics: list[dict[str, Any]],
) -> dict[str, Any]:
    registration_rules = descriptor.temporal_rules["registration_rules"]
    trace = build_temporal_consistency_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        comparison_status=TEMPORAL_STATUS_NOT_EVALUATED,
        comparison_scope=TEMPORAL_SCOPE_NOT_EVALUATED,
        spatial_comparison_mode=TEMPORAL_SPATIAL_MODE_NONE,
        prior_admissibility={
            "status": TEMPORAL_PRIOR_INADMISSIBLE,
            "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
        },
        chronology_assessment={
            "pair_provenance": TEMPORAL_PAIR_PROVENANCE_MISSING,
            "current_timestamp_provenance": analysis["study_timestamp_provenance"],
            "prior_timestamp_provenance": "missing",
            "current_study_acquired_at": analysis["study_acquired_at"],
            "prior_study_acquired_at": None,
            "interval_days": None,
            "ordered": False,
            "time_metrics_available": False,
            "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
        },
        geometry_assessment={
            "comparability_class": TEMPORAL_GEOMETRY_INSUFFICIENT,
            "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
        },
        computed_metrics={
            "signed_volume_delta_mm3": None,
            "absolute_volume_delta_mm3": None,
            "signed_volume_change_percent": None,
            "signed_normalized_percent_change_per_30d": None,
            "volume_change_direction": "not_evaluated",
            "normalized_rate_direction": "not_evaluated",
            "intersection_over_union": None,
            "global_centroid_shift_mm": None,
            "prior_component_count": None,
            "current_component_count": int(analysis["verified_mask_component_count"]),
            "component_count_delta": None,
        },
        flags=[],
        skipped_metrics=[
            _temporal_skipped_metric("signed_volume_delta_mm3", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
            _temporal_skipped_metric("absolute_volume_delta_mm3", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
            _temporal_skipped_metric("signed_volume_change_percent", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
            _temporal_skipped_metric(
                "signed_normalized_percent_change_per_30d",
                TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
            ),
            _temporal_skipped_metric("intersection_over_union", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
            _temporal_skipped_metric("global_centroid_shift_mm", TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT),
        ],
    )
    registration_trace = build_longitudinal_registration_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        prior_bundle_sha256=None,
        status=TEMPORAL_REGISTRATION_STATUS_NOT_EVALUATED,
        blocking_admissible=False,
        algorithm={
            "name": str(registration_rules["algorithm"]),
            "max_translation_voxels": int(registration_rules["max_translation_voxels"]),
        },
        transform=None,
        quality={
            "attempted": False,
            "reason_code": TEMPORAL_SKIPPED_NO_PRIOR_CONTEXT,
            "registration_prerequisites_met": False,
            "exact_iou_before_registration": None,
            "best_iou_after_registration": None,
            "candidate_count": 0,
            "spatial_use_allowed": False,
            "blocking_use_allowed": False,
            "minimum_iou_for_spatial_use": _round_float(
                float(registration_rules["minimum_iou_for_spatial_use"])
            ),
            "minimum_iou_for_blocking": _round_float(
                float(registration_rules["minimum_iou_for_blocking"])
            ),
        },
    )
    return {
        "prior_verified_contexts": None,
        "prior_verified_context": None,
        "selection_trace": selection_trace,
        "trajectory_trace": trajectory_trace,
        "registration_trace": registration_trace,
        "trace": trace,
        "heuristics": list(trajectory_heuristics),
        "summary": "No prior verified context was supplied; longitudinal verification was not evaluated.",
    }


def _build_temporal_no_primary_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    supplied_priors: list[dict[str, Any]],
    selection_section: Mapping[str, Any],
    trajectory_trace: Mapping[str, Any],
    trajectory_heuristics: list[dict[str, Any]],
    registration_trace: Mapping[str, Any],
) -> dict[str, Any]:
    prior_admissibility_status = (
        TEMPORAL_PRIOR_ADMISSIBLE
        if any(
            _temporal_prior_admissibility(item, descriptor)[0] == TEMPORAL_PRIOR_ADMISSIBLE
            for item in supplied_priors
        )
        else TEMPORAL_PRIOR_INADMISSIBLE
    )
    prior_admissibility_reason = str(selection_section["primary_unavailable_reason_code"])
    trace = build_temporal_consistency_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        comparison_status=TEMPORAL_STATUS_UNSUITABLE,
        comparison_scope=TEMPORAL_SCOPE_NONE,
        spatial_comparison_mode=TEMPORAL_SPATIAL_MODE_NONE,
        prior_admissibility={
            "status": prior_admissibility_status,
            "reason_code": prior_admissibility_reason,
        },
        chronology_assessment={
            "pair_provenance": TEMPORAL_PAIR_PROVENANCE_MISSING,
            "current_timestamp_provenance": analysis["study_timestamp_provenance"],
            "prior_timestamp_provenance": "missing",
            "current_study_acquired_at": analysis["study_acquired_at"],
            "prior_study_acquired_at": None,
            "interval_days": None,
            "ordered": False,
            "time_metrics_available": False,
            "reason_code": prior_admissibility_reason,
        },
        geometry_assessment={
            "comparability_class": TEMPORAL_GEOMETRY_INSUFFICIENT,
            "reason_code": TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT,
        },
        computed_metrics={
            "signed_volume_delta_mm3": None,
            "absolute_volume_delta_mm3": None,
            "signed_volume_change_percent": None,
            "signed_normalized_percent_change_per_30d": None,
            "volume_change_direction": "not_evaluated",
            "normalized_rate_direction": "not_evaluated",
            "intersection_over_union": None,
            "global_centroid_shift_mm": None,
            "prior_component_count": None,
            "current_component_count": int(analysis["verified_mask_component_count"]),
            "component_count_delta": None,
        },
        flags=[],
        skipped_metrics=[
            _temporal_skipped_metric("signed_volume_delta_mm3", prior_admissibility_reason),
            _temporal_skipped_metric("absolute_volume_delta_mm3", prior_admissibility_reason),
            _temporal_skipped_metric("signed_volume_change_percent", prior_admissibility_reason),
            _temporal_skipped_metric("signed_normalized_percent_change_per_30d", prior_admissibility_reason),
            _temporal_skipped_metric("intersection_over_union", prior_admissibility_reason),
            _temporal_skipped_metric("global_centroid_shift_mm", prior_admissibility_reason),
        ],
    )
    heuristics = list(trajectory_heuristics)
    if prior_admissibility_status == TEMPORAL_PRIOR_INADMISSIBLE:
        heuristics.insert(
            0,
            _heuristic_entry(
                code=TEMPORAL_REASON_TEMPORAL_PRIOR_UNSUITABLE,
                label="Prior admissibility",
                severity="warning",
                status="warning",
                message=(
                    "Supplied prior contexts are inadmissible for longitudinal comparison "
                    f"({prior_admissibility_reason})."
                ),
            )
        )
    return {
        "prior_verified_contexts": supplied_priors,
        "prior_verified_context": None,
        "selection_trace": selection_section["trace"],
        "trajectory_trace": trajectory_trace,
        "registration_trace": registration_trace,
        "trace": trace,
        "heuristics": heuristics,
        "summary": "Supplied prior contexts did not yield a usable primary comparator under the configured temporal policy.",
    }


def _build_temporal_primary_section(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    supplied_priors: list[dict[str, Any]],
    primary_prior: Mapping[str, Any],
    selection_trace: Mapping[str, Any],
    trajectory_trace: Mapping[str, Any],
    trajectory_heuristics: list[dict[str, Any]],
    registration_trace: Mapping[str, Any],
    transformed_prior_mask: np.ndarray | None,
) -> dict[str, Any]:
    pair_provenance = _temporal_pair_provenance(
        str(analysis["study_timestamp_provenance"]),
        str(primary_prior["timestamp_provenance"]),
    )
    chronology_reason = "chronology_available"
    interval_days: float | None = None
    ordered = False
    time_metrics_available = False
    current_timestamp = analysis["study_acquired_at"]
    prior_timestamp = primary_prior["study_acquired_at"]
    current_dt = _temporal_parse_timestamp(current_timestamp)
    prior_dt = _temporal_parse_timestamp(prior_timestamp)
    if pair_provenance == TEMPORAL_PAIR_PROVENANCE_UNTRUSTED:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED
    elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_MISSING:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
    elif current_dt is None or prior_dt is None:
        chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_MISSING
    else:
        delta_seconds = (current_dt - prior_dt).total_seconds()
        if delta_seconds < 0:
            chronology_reason = TEMPORAL_SKIPPED_TIMESTAMP_REVERSED
        elif delta_seconds == 0:
            chronology_reason = TEMPORAL_SKIPPED_ZERO_INTERVAL
            interval_days = 0.0
            ordered = True
        else:
            ordered = True
            interval_days = _round_float(delta_seconds / 86400.0)
            if interval_days > float(descriptor.temporal_rules["degraded_interval_max_days"]):
                chronology_reason = TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX
            elif pair_provenance == TEMPORAL_PAIR_PROVENANCE_MIXED:
                chronology_reason = TEMPORAL_SKIPPED_MIXED_PROVENANCE
                time_metrics_available = True
            elif interval_days > float(descriptor.temporal_rules["aligned_interval_max_days"]):
                chronology_reason = "interval_within_degraded_window"
                time_metrics_available = True
            else:
                chronology_reason = "interval_within_aligned_window"
                time_metrics_available = True
    geometry_class = TEMPORAL_GEOMETRY_INSUFFICIENT
    geometry_reason = TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT
    spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_NONE
    if _temporal_exact_geometry_match(analysis, primary_prior):
        geometry_class = TEMPORAL_GEOMETRY_EXACT_MATCH
        geometry_reason = TEMPORAL_GEOMETRY_REASON_EXACT_MATCH
        spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_EXACT
    elif registration_trace["status"] == TEMPORAL_REGISTRATION_STATUS_USED:
        geometry_class = TEMPORAL_GEOMETRY_REGISTERED_MATCH
        geometry_reason = TEMPORAL_GEOMETRY_REASON_REGISTERED
        spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_REGISTERED
    elif _temporal_scalar_basis_trusted(analysis, primary_prior):
        geometry_class = TEMPORAL_GEOMETRY_VOLUME_ONLY
        geometry_reason = TEMPORAL_GEOMETRY_REASON_NOT_EXACT
        spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_VOLUME_ONLY
    comparison_scope = {
        TEMPORAL_GEOMETRY_EXACT_MATCH: TEMPORAL_SCOPE_FULL,
        TEMPORAL_GEOMETRY_REGISTERED_MATCH: TEMPORAL_SCOPE_FULL,
        TEMPORAL_GEOMETRY_VOLUME_ONLY: TEMPORAL_SCOPE_VOLUME_ONLY,
        TEMPORAL_GEOMETRY_INSUFFICIENT: TEMPORAL_SCOPE_NONE,
    }[geometry_class]
    comparison_status = TEMPORAL_STATUS_ALIGNED
    if chronology_reason in {TEMPORAL_SKIPPED_TIMESTAMP_REVERSED, TEMPORAL_SKIPPED_INTERVAL_EXCEEDS_MAX}:
        comparison_status = TEMPORAL_STATUS_UNSUITABLE
        comparison_scope = TEMPORAL_SCOPE_NONE
        spatial_comparison_mode = TEMPORAL_SPATIAL_MODE_NONE
    elif comparison_scope == TEMPORAL_SCOPE_NONE:
        comparison_status = TEMPORAL_STATUS_UNSUITABLE
    elif spatial_comparison_mode in {TEMPORAL_SPATIAL_MODE_REGISTERED, TEMPORAL_SPATIAL_MODE_VOLUME_ONLY}:
        comparison_status = TEMPORAL_STATUS_DEGRADED
    elif pair_provenance in {
        TEMPORAL_PAIR_PROVENANCE_MISSING,
        TEMPORAL_PAIR_PROVENANCE_UNTRUSTED,
        TEMPORAL_PAIR_PROVENANCE_MIXED,
    } or chronology_reason in {TEMPORAL_SKIPPED_ZERO_INTERVAL, "interval_within_degraded_window"}:
        comparison_status = TEMPORAL_STATUS_DEGRADED
    current_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    prior_mask = np.asarray(primary_prior["verified_mask"], dtype=np.int64)
    spatial_prior_mask = prior_mask if transformed_prior_mask is None else transformed_prior_mask
    spatial_prior_affine = (
        analysis["spatial_geometry"]["affine_ijk_to_mm"]
        if spatial_comparison_mode == TEMPORAL_SPATIAL_MODE_REGISTERED
        else primary_prior["spatial_geometry"]["affine_ijk_to_mm"]
    )
    signed_volume_delta_mm3: float | None = None
    absolute_volume_delta_mm3: float | None = None
    signed_volume_change_percent: float | None = None
    signed_normalized_percent_change_per_30d: float | None = None
    intersection_over_union: float | None = None
    global_centroid_shift_mm: float | None = None
    skipped_metrics: list[dict[str, str]] = []
    flags: list[str] = []
    if comparison_scope == TEMPORAL_SCOPE_NONE:
        blocking_reason = chronology_reason
        if geometry_reason == TEMPORAL_GEOMETRY_REASON_NOT_SUFFICIENT:
            blocking_reason = TEMPORAL_SKIPPED_GEOMETRY_NOT_SUFFICIENT
        for metric_name in (
            "signed_volume_delta_mm3",
            "absolute_volume_delta_mm3",
            "signed_volume_change_percent",
            "signed_normalized_percent_change_per_30d",
            "intersection_over_union",
            "global_centroid_shift_mm",
        ):
            skipped_metrics.append(_temporal_skipped_metric(metric_name, str(blocking_reason)))
    else:
        signed_volume_delta_mm3 = _round_float(
            float(analysis["computed_tumor_volume_mm3"]) - float(primary_prior["verified_volume_mm3"])
        )
        absolute_volume_delta_mm3 = _round_float(abs(float(signed_volume_delta_mm3)))
        if float(primary_prior["verified_volume_mm3"]) > 0.0:
            signed_volume_change_percent = _round_float(
                (signed_volume_delta_mm3 / float(primary_prior["verified_volume_mm3"])) * 100.0
            )
            if time_metrics_available and interval_days is not None and interval_days > 0.0:
                signed_normalized_percent_change_per_30d = _round_float(
                    signed_volume_change_percent * 30.0 / float(interval_days)
                )
            else:
                if chronology_reason in {
                    TEMPORAL_SKIPPED_TIMESTAMP_MISSING,
                    TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED,
                    TEMPORAL_SKIPPED_MIXED_PROVENANCE,
                    TEMPORAL_SKIPPED_ZERO_INTERVAL,
                }:
                    flags.append(TEMPORAL_FLAG_RATE_NOT_EVALUATED)
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "signed_normalized_percent_change_per_30d",
                        str(chronology_reason),
                    )
                )
        else:
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "signed_volume_change_percent",
                    TEMPORAL_SKIPPED_ZERO_PRIOR_VOLUME,
                )
            )
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "signed_normalized_percent_change_per_30d",
                    TEMPORAL_SKIPPED_ZERO_PRIOR_VOLUME,
                )
            )
        if comparison_scope == TEMPORAL_SCOPE_FULL:
            intersection_over_union = _temporal_mask_iou(current_mask, spatial_prior_mask)
            if intersection_over_union is None:
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "intersection_over_union",
                        TEMPORAL_SKIPPED_BOTH_MASKS_EMPTY,
                    )
                )
            if not np.any(spatial_prior_mask == 1):
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "global_centroid_shift_mm",
                        TEMPORAL_SKIPPED_EMPTY_PRIOR_MASK,
                    )
                )
            elif not np.any(current_mask == 1):
                skipped_metrics.append(
                    _temporal_skipped_metric(
                        "global_centroid_shift_mm",
                        TEMPORAL_SKIPPED_EMPTY_CURRENT_MASK,
                    )
                )
            else:
                current_centroid = _mask_centroid_mm(
                    current_mask,
                    analysis["spatial_geometry"]["affine_ijk_to_mm"],
                )
                prior_centroid = _mask_centroid_mm(
                    spatial_prior_mask,
                    spatial_prior_affine,
                )
                if current_centroid is not None and prior_centroid is not None:
                    global_centroid_shift_mm = _round_float(
                        float(np.linalg.norm(current_centroid - prior_centroid))
                    )
        else:
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "intersection_over_union",
                    TEMPORAL_SKIPPED_GEOMETRY_NOT_EXACT,
                )
            )
            skipped_metrics.append(
                _temporal_skipped_metric(
                    "global_centroid_shift_mm",
                    TEMPORAL_SKIPPED_GEOMETRY_NOT_EXACT,
                )
            )
    prior_component_count = _count_face_connected_components(prior_mask)
    current_component_count = int(analysis["verified_mask_component_count"])
    component_count_delta = abs(current_component_count - prior_component_count)
    if component_count_delta:
        flags.append(TEMPORAL_FLAG_COMPONENT_COUNT_CHANGED)
    computed_metrics = {
        "signed_volume_delta_mm3": signed_volume_delta_mm3,
        "absolute_volume_delta_mm3": absolute_volume_delta_mm3,
        "signed_volume_change_percent": signed_volume_change_percent,
        "signed_normalized_percent_change_per_30d": signed_normalized_percent_change_per_30d,
        "volume_change_direction": _temporal_metric_direction(signed_volume_change_percent),
        "normalized_rate_direction": _temporal_metric_direction(signed_normalized_percent_change_per_30d),
        "intersection_over_union": intersection_over_union,
        "global_centroid_shift_mm": global_centroid_shift_mm,
        "prior_component_count": prior_component_count,
        "current_component_count": current_component_count,
        "component_count_delta": component_count_delta,
    }
    metric_rules = descriptor.temporal_rules["metrics"]
    heuristics: list[dict[str, Any]] = list(trajectory_heuristics)
    if comparison_scope == TEMPORAL_SCOPE_VOLUME_ONLY:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_GEOMETRY_LIMITED,
                label="Geometry comparability",
                severity="advisory",
                status="advisory",
                message="Geometry was not spatially comparable enough for overlap metrics, so only scalar volume comparison ran.",
            )
        )
    if chronology_reason == TEMPORAL_SKIPPED_TIMESTAMP_REVERSED:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_TIMESTAMP_REVERSED,
                label="Chronology ordering",
                severity="warning",
                status="warning",
                message="The selected prior timestamp is later than the current case timestamp.",
            )
        )
    if comparison_status == TEMPORAL_STATUS_DEGRADED and chronology_reason in {
        TEMPORAL_SKIPPED_TIMESTAMP_MISSING,
        TEMPORAL_SKIPPED_TIMESTAMP_UNTRUSTED,
        TEMPORAL_SKIPPED_MIXED_PROVENANCE,
        TEMPORAL_SKIPPED_ZERO_INTERVAL,
        "interval_within_degraded_window",
    }:
        heuristics.append(
            _heuristic_entry(
                code=TEMPORAL_REASON_INTERVAL_DEGRADED,
                label="Chronology quality",
                severity="advisory",
                status="advisory",
                message="Chronology quality is degraded, so time-dependent interpretation is limited.",
            )
        )
    for code, label, metric_name, high_rule, low_rule in (
        (TEMPORAL_REASON_VOLUME_WARNING, "Signed volume change percent", "signed_volume_change_percent", metric_rules["signed_volume_change_percent"]["warning_abs_threshold"], metric_rules["signed_volume_change_percent"]["advisory_abs_threshold"]),
        (TEMPORAL_REASON_RATE_WARNING, "Signed normalized percent change per 30d", "signed_normalized_percent_change_per_30d", metric_rules["signed_normalized_percent_change_per_30d"]["warning_abs_threshold"], metric_rules["signed_normalized_percent_change_per_30d"]["advisory_abs_threshold"]),
    ):
        value = computed_metrics[metric_name]
        if value is None:
            continue
        if abs(float(value)) >= float(high_rule):
            heuristics.append(_heuristic_entry(code=code, label=label, severity="warning", status="warning", message=f"Absolute {label.lower()} exceeded the configured warning threshold ({value}).", observed=value, threshold=float(high_rule)))
        elif abs(float(value)) >= float(low_rule):
            heuristics.append(_heuristic_entry(code=code, label=label, severity="advisory", status="advisory", message=f"Absolute {label.lower()} exceeded the configured advisory threshold ({value}).", observed=value, threshold=float(low_rule)))
    if intersection_over_union is not None:
        if float(intersection_over_union) < float(metric_rules["intersection_over_union"]["warning_floor"]):
            heuristics.append(_heuristic_entry(code=TEMPORAL_REASON_IOU_WARNING, label="Intersection over union", severity="warning", status="warning", message=f"Spatial overlap fell below the configured warning floor ({intersection_over_union}).", observed=intersection_over_union, threshold=float(metric_rules["intersection_over_union"]["warning_floor"])))
        elif float(intersection_over_union) < float(metric_rules["intersection_over_union"]["advisory_floor"]):
            heuristics.append(_heuristic_entry(code=TEMPORAL_REASON_IOU_WARNING, label="Intersection over union", severity="advisory", status="advisory", message=f"Spatial overlap fell below the configured advisory floor ({intersection_over_union}).", observed=intersection_over_union, threshold=float(metric_rules["intersection_over_union"]["advisory_floor"])))
    if global_centroid_shift_mm is not None:
        if float(global_centroid_shift_mm) > float(metric_rules["global_centroid_shift_mm"]["warning_threshold"]):
            heuristics.append(_heuristic_entry(code=TEMPORAL_REASON_CENTROID_WARNING, label="Global centroid shift", severity="warning", status="warning", message=f"Centroid shift exceeded the configured warning threshold ({global_centroid_shift_mm} mm).", observed=global_centroid_shift_mm, threshold=float(metric_rules["global_centroid_shift_mm"]["warning_threshold"])))
        elif float(global_centroid_shift_mm) > float(metric_rules["global_centroid_shift_mm"]["advisory_threshold"]):
            heuristics.append(_heuristic_entry(code=TEMPORAL_REASON_CENTROID_WARNING, label="Global centroid shift", severity="advisory", status="advisory", message=f"Centroid shift exceeded the configured advisory threshold ({global_centroid_shift_mm} mm).", observed=global_centroid_shift_mm, threshold=float(metric_rules["global_centroid_shift_mm"]["advisory_threshold"])))
    chronology_assessment = {
        "pair_provenance": pair_provenance,
        "current_timestamp_provenance": analysis["study_timestamp_provenance"],
        "prior_timestamp_provenance": primary_prior["timestamp_provenance"],
        "current_study_acquired_at": current_timestamp,
        "prior_study_acquired_at": prior_timestamp,
        "interval_days": interval_days,
        "ordered": ordered,
        "time_metrics_available": time_metrics_available,
        "reason_code": chronology_reason,
    }
    geometry_assessment = {"comparability_class": geometry_class, "reason_code": geometry_reason}
    blocking_rules = descriptor.temporal_rules["gating_rules"]
    if (
        bool(blocking_rules.get("allow_blocking", False))
        and ordered
        and interval_days is not None
        and interval_days > 0.0
        and _temporal_pair_is_fully_trusted(pair_provenance)
        and chronology_reason in {"interval_within_aligned_window", "interval_within_degraded_window"}
        and (
            spatial_comparison_mode != TEMPORAL_SPATIAL_MODE_REGISTERED
            or bool(registration_trace["blocking_admissible"])
        )
    ):
        blocking_codes = {str(item) for item in blocking_rules.get("blocking_trigger_codes", [])}
        heuristics = [
            {
                **entry,
                "severity": HEURISTIC_BLOCKING,
                "status": "failed",
                "message": f"{entry['message']} Temporal blocking was enabled for this profile and the gating prerequisites were satisfied.",
            }
            if entry["status"] == "warning" and entry["code"] in blocking_codes
            else entry
            for entry in heuristics
        ]
    trace = build_temporal_consistency_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        comparison_status=comparison_status,
        comparison_scope=comparison_scope,
        spatial_comparison_mode=spatial_comparison_mode,
        prior_admissibility={"status": TEMPORAL_PRIOR_ADMISSIBLE, "reason_code": "admissible"},
        chronology_assessment=chronology_assessment,
        geometry_assessment=geometry_assessment,
        computed_metrics=computed_metrics,
        flags=sorted(set(flags)),
        skipped_metrics=sorted(skipped_metrics, key=lambda item: (str(item["metric"]), str(item["reason_code"]))),
    )
    summary = {
        TEMPORAL_STATUS_ALIGNED: "Longitudinal comparison ran with full aligned temporal context.",
        TEMPORAL_STATUS_DEGRADED: "Longitudinal comparison ran with degraded temporal context.",
        TEMPORAL_STATUS_UNSUITABLE: "Longitudinal comparison was suppressed by temporal policy.",
        TEMPORAL_STATUS_NOT_EVALUATED: "Longitudinal comparison was not evaluated.",
    }[comparison_status]
    return {
        "prior_verified_contexts": supplied_priors,
        "prior_verified_context": primary_prior,
        "selection_trace": selection_trace,
        "trajectory_trace": trajectory_trace,
        "registration_trace": registration_trace,
        "trace": trace,
        "heuristics": heuristics,
        "summary": summary,
    }


def _build_temporal_consistency_section_v2(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    selection_section = _build_longitudinal_selection_section(state, analysis, descriptor)
    trajectory_section = _build_temporal_trajectory_section(
        state,
        analysis,
        descriptor,
        selected_priors=selection_section["selected_priors"],
    )
    if not selection_section["supplied_priors"]:
        return _build_temporal_not_evaluated_section(
            state,
            analysis,
            descriptor,
            selection_trace=selection_section["trace"],
            trajectory_trace=trajectory_section["trace"],
            trajectory_heuristics=list(trajectory_section["heuristics"]),
        )
    registration_section = _build_longitudinal_registration_section(
        state,
        analysis,
        descriptor,
        primary_prior=selection_section["primary_prior"],
        prior_admissibility_status=(
            TEMPORAL_PRIOR_ADMISSIBLE
            if selection_section["primary_prior"] is not None
            else TEMPORAL_PRIOR_INADMISSIBLE
        ),
    )
    if selection_section["primary_prior"] is None:
        return _build_temporal_no_primary_section(
            state,
            analysis,
            descriptor,
            supplied_priors=selection_section["supplied_priors"],
            selection_section=selection_section,
            trajectory_trace=trajectory_section["trace"],
            trajectory_heuristics=list(trajectory_section["heuristics"]),
            registration_trace=registration_section["trace"],
        )
    return _build_temporal_primary_section(
        state,
        analysis,
        descriptor,
        supplied_priors=selection_section["supplied_priors"],
        primary_prior=selection_section["primary_prior"],
        selection_trace=selection_section["trace"],
        trajectory_trace=trajectory_section["trace"],
        trajectory_heuristics=list(trajectory_section["heuristics"]),
        registration_trace=registration_section["trace"],
        transformed_prior_mask=registration_section["transformed_prior_mask"],
    )


def _build_verified_manifest(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    contracts: Mapping[str, Any],
    descriptor: Any,
    *,
    temporal_consistency: Mapping[str, Any],
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
    clinical_plausibility = _build_clinical_plausibility_manifest(
        analysis,
        descriptor,
        temporal_consistency=temporal_consistency,
    )
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
    proof_stage_verification = {
        PROOF_STAGE_MEASUREMENT_INTEGRITY: consistency_verified,
        PROOF_STAGE_RECONSTRUCTION_PARITY: ai_match_verified,
        PROOF_STAGE_SEGMENTATION_POLICY: mask_verified,
        PROOF_STAGE_VOLUME_COMPUTATION: volume_verified,
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
        "mask_derivation_policy": contracts["mask_derivation_policy"],
        "verification_tolerance_policy": contracts["verification_tolerance_policy"],
        "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
        "ai_mask_hash": analysis["ai_mask_hash"],
        "verified_mask_hash": analysis["verified_mask_hash"],
        "reconstruction_algorithm": "numpy.ifft2_per_slice.real",
        "performance_expectations": contracts["verification_profile_descriptor"][
            "performance_expectations"
        ],
        "measurement_integrity_proof": {
            "verified": consistency_verified,
            "domain": "frequency_domain",
            "proof_scope": "derived_reconstruction_measurement_domain_anchoring",
            "attack_surface": (
                "Distributed low-amplitude k-space perturbations are checked with a "
                "global norm in addition to the per-frequency maximum."
            ),
            "raw_kspace_hash": analysis["raw_hash"],
            "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
            "forward_projection_hash": analysis["forward_projection_hash"],
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
        },
        "reconstruction_parity_proof": {
            "verified": ai_match_verified,
            "domain": "spatial_image_domain",
            "proof_scope": "ai_reconstruction_to_derived_reconstruction_parity",
            "attack_surface": (
                "Distributed low-amplitude voxel perturbations are checked with a "
                "global norm in addition to the per-voxel maximum."
            ),
            "ai_reconstruction_hash": analysis["ai_reconstruction_hash"],
            "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
            "max_abs_error": analysis["max_ai_abs_error"],
            "relative_l2_error": analysis["relative_ai_l2_error"],
            "max_abs_tolerance": descriptor.tolerance("spatial_per_voxel_abs_error"),
            "relative_l2_tolerance": descriptor.tolerance(
                "spatial_global_relative_l2_error"
            ),
            "guard_band_fraction": descriptor.guard_band_fraction,
            "metrics": ai_match_metrics,
        },
        "segmentation_policy_proof": _build_segmentation_policy_proof_manifest(
            analysis=analysis,
            verified=mask_verified,
            mask_derivation_policy=contracts["mask_derivation_policy"],
        ),
        "volume_computation_proof": {
            "verified": volume_verified,
            "proof_scope": "verified_mask_geometry_scalar_volume_check",
            "claimed_mm3": claimed_volume,
            "computed_mm3": analysis["computed_tumor_volume_mm3"],
            "voxel_count": analysis["verified_mask_voxel_count"],
            "voxel_volume_mm3": analysis["voxel_volume_mm3"],
            "verified_mask_hash": analysis["verified_mask_hash"],
            "tolerance_metric": volume_metric,
        },
        "proof_core_summary": {
            "verified": proof_status == "passed",
            "no_invented_pixels": consistency_verified and ai_match_verified,
            "canonical_stage_order": list(_PROOF_STAGE_ORDER),
            "earliest_failing_proof_stage": _earliest_failing_proof_stage(
                proof_stage_verification
            ),
        },
        "clinical_plausibility": clinical_plausibility,
        "decision_summary": decision_summary,
    }
    manifest["verification_pipeline"] = _build_verification_pipeline(
        state,
        analysis,
        manifest,
    )
    return manifest


def _build_clinical_plausibility_manifest(
    analysis: Mapping[str, Any],
    descriptor: Any,
    *,
    temporal_consistency: Mapping[str, Any],
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
    heuristics.extend(list(temporal_consistency["heuristics"]))

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
        "longitudinal_selection": deepcopy(temporal_consistency["selection_trace"]),
        "temporal_trajectory": deepcopy(temporal_consistency["trajectory_trace"]),
        "longitudinal_registration": deepcopy(temporal_consistency["registration_trace"]),
        "temporal_consistency": deepcopy(temporal_consistency["trace"]),
    }


def _phase_entry(
    *,
    title: str,
    status: str,
    summary: str,
    blocking: bool,
    checks: list[dict[str, Any]] | None = None,
    proof_stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "title": title,
        "status": status,
        "summary": summary,
        "blocking": blocking,
    }
    if checks is not None:
        payload["checks"] = checks
    if proof_stages is not None:
        payload["proof_stage_order"] = list(_PROOF_STAGE_ORDER)
        payload["proof_stages"] = proof_stages
    return payload


def _operational_check_entry(
    *,
    code: str,
    label: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "status": status,
        "message": message,
    }


def _proof_stage_title(proof_stage: str) -> str:
    return _PROOF_STAGE_TITLES[proof_stage]


def _proof_stage_summary(
    *,
    proof_stage: str,
    status: str,
) -> str:
    if status == "not_evaluated":
        return (
            f"{_proof_stage_title(proof_stage)} was not evaluated because canonical proof-core "
            "was not entered."
        )
    if proof_stage == PROOF_STAGE_MEASUREMENT_INTEGRITY:
        return (
            "The derived reconstruction remained anchored to the supplied raw measurements "
            "within policy."
            if status == "passed"
            else "The derived reconstruction could not be justified against the supplied raw "
            "measurements within policy."
        )
    if proof_stage == PROOF_STAGE_RECONSTRUCTION_PARITY:
        return (
            "The AI reconstruction matched the derived reconstruction within policy."
            if status == "passed"
            else "The AI reconstruction diverged from the derived reconstruction beyond policy."
        )
    if proof_stage == PROOF_STAGE_SEGMENTATION_POLICY:
        return (
            "The AI mask matched the policy-derived verified mask."
            if status == "passed"
            else "The AI mask diverged from the policy-derived verified mask."
        )
    return (
        "The claimed scalar volume matched the verified mask geometry within policy."
        if status == "passed"
        else "The claimed scalar volume did not match the verified mask geometry within policy."
    )


def _proof_stage_result(
    *,
    proof_stage: str,
    title: str,
    verified: bool | None,
    summary: str,
) -> dict[str, Any]:
    if verified is None:
        status = "not_evaluated"
    else:
        status = "passed" if verified else "failed"
    return {
        "proof_stage": proof_stage,
        "title": title,
        "status": status,
        "summary": summary,
    }


def _earliest_failing_proof_stage(stage_verification: Mapping[str, bool]) -> str | None:
    for proof_stage in _PROOF_STAGE_ORDER:
        if not bool(stage_verification.get(proof_stage, False)):
            return proof_stage
    return None


def _build_proof_core_stage_results(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    measurement = manifest["measurement_integrity_proof"]
    reconstruction = manifest["reconstruction_parity_proof"]
    segmentation = manifest["segmentation_policy_proof"]
    volume = manifest["volume_computation_proof"]
    return [
        _proof_stage_result(
            proof_stage=PROOF_STAGE_MEASUREMENT_INTEGRITY,
            title=_proof_stage_title(PROOF_STAGE_MEASUREMENT_INTEGRITY),
            verified=bool(measurement["verified"]),
            summary=_proof_stage_summary(
                proof_stage=PROOF_STAGE_MEASUREMENT_INTEGRITY,
                status="passed" if measurement["verified"] else "failed",
            ),
        ),
        _proof_stage_result(
            proof_stage=PROOF_STAGE_RECONSTRUCTION_PARITY,
            title=_proof_stage_title(PROOF_STAGE_RECONSTRUCTION_PARITY),
            verified=bool(reconstruction["verified"]),
            summary=_proof_stage_summary(
                proof_stage=PROOF_STAGE_RECONSTRUCTION_PARITY,
                status="passed" if reconstruction["verified"] else "failed",
            ),
        ),
        _proof_stage_result(
            proof_stage=PROOF_STAGE_SEGMENTATION_POLICY,
            title=_proof_stage_title(PROOF_STAGE_SEGMENTATION_POLICY),
            verified=bool(segmentation["verified"]),
            summary=_proof_stage_summary(
                proof_stage=PROOF_STAGE_SEGMENTATION_POLICY,
                status="passed" if segmentation["verified"] else "failed",
            ),
        ),
        _proof_stage_result(
            proof_stage=PROOF_STAGE_VOLUME_COMPUTATION,
            title=_proof_stage_title(PROOF_STAGE_VOLUME_COMPUTATION),
            verified=bool(volume["verified"]),
            summary=_proof_stage_summary(
                proof_stage=PROOF_STAGE_VOLUME_COMPUTATION,
                status="passed" if volume["verified"] else "failed",
            ),
        ),
    ]


def _build_operational_subphases(
    *,
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    structural_checks: list[dict[str, Any]],
    intake_blocked: bool,
    proof_stages: list[dict[str, Any]],
) -> dict[str, Any]:
    pre_check_checks = _build_pre_check_checks(
        state=state,
        analysis=analysis,
        manifest=manifest,
        structural_checks=structural_checks,
        intake_blocked=intake_blocked,
    )
    pre_check_status = _derive_pre_check_status(pre_check_checks)
    full_proof_stages = _build_full_proof_operational_stage_results(proof_stages)
    full_proof_status = _derive_full_proof_status(full_proof_stages)
    payload = {
        "schema": _OPERATIONAL_SUBPHASE_SCHEMA,
        "ordering_semantics": (
            "Derived operational subphase view over the canonical assurance flow. "
            "A non-passing subphase is release-preventing within operational flow, "
            "but canonical decision ownership remains with decision.pipeline_phase "
            "and optional decision.proof_stage."
        ),
        "subphase_order": list(_OPERATIONAL_SUBPHASE_ORDER),
        "subphases": {
            OPERATIONAL_SUBPHASE_PRE_CHECK: _phase_entry(
                title="Pre-Check",
                status=pre_check_status,
                summary=_pre_check_summary(pre_check_checks, pre_check_status),
                blocking=True,
                checks=pre_check_checks,
            ),
            OPERATIONAL_SUBPHASE_FULL_PROOF: _phase_entry(
                title="Full Proof",
                status=full_proof_status,
                summary=_full_proof_summary(full_proof_status),
                blocking=True,
                proof_stages=full_proof_stages,
            ),
        },
    }
    _validate_operational_subphases(payload)
    return payload


def _build_pre_check_checks(
    *,
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    structural_checks: list[dict[str, Any]],
    intake_blocked: bool,
) -> list[dict[str, Any]]:
    return [
        _build_structure_validation_check(
            structural_checks=structural_checks,
            intake_blocked=intake_blocked,
        ),
        _build_hash_surface_consistency_check(
            analysis=analysis,
            manifest=manifest,
        ),
        _build_shape_consistency_check(structural_checks),
        _build_quick_anomaly_detection_check(
            state=state,
            structural_checks=structural_checks,
        ),
    ]


def _build_structure_validation_check(
    *,
    structural_checks: list[dict[str, Any]],
    intake_blocked: bool,
) -> dict[str, Any]:
    structural_status = _tier_status_from_checks(structural_checks)
    if intake_blocked:
        status = "failed"
    else:
        status = "passed" if structural_status == "passed" else "failed"
    message = {
        "passed": "Input intake and basic structural admissibility passed during pre-check.",
        "failed": (
            "Input intake or basic structural admissibility blocked operational pre-check."
            if intake_blocked
            else "Basic structural admissibility failed during operational pre-check."
        ),
    }[status]
    return _operational_check_entry(
        code=OPERATIONAL_CHECK_STRUCTURE_VALIDATION,
        label="Structure validation",
        status=status,
        message=message,
    )


def _build_hash_surface_consistency_check(
    *,
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    comparable_groups = 0
    mismatch_detected = False
    for hash_values in _hash_surface_consistency_groups(analysis=analysis, manifest=manifest):
        present_values = [value for value in hash_values if isinstance(value, str) and value]
        if len(present_values) < 2:
            continue
        comparable_groups += 1
        if len(set(present_values)) != 1:
            mismatch_detected = True
            break
    if mismatch_detected:
        status = "failed"
    elif comparable_groups:
        status = "passed"
    else:
        status = "not_evaluated"
    return _operational_check_entry(
        code=OPERATIONAL_CHECK_HASH_CHECKS,
        label="Hash-surface consistency",
        status=status,
        message={
            "passed": (
                "Already-materialized deterministic hash surfaces were consistent across "
                "reusable report contracts."
            ),
            "failed": (
                "Already-materialized deterministic hash surfaces disagreed across reusable "
                "report contracts."
            ),
            "not_evaluated": (
                "No reusable deterministic hash-surface comparison set was available during "
                "pre-check."
            ),
        }[status],
    )


def _hash_surface_consistency_groups(
    *,
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[tuple[str | None, ...], ...]:
    ai_claim = manifest.get("ai_claim_envelope", {})
    measurement = manifest.get("measurement_integrity_proof", {})
    reconstruction = manifest.get("reconstruction_parity_proof", {})
    segmentation = manifest.get("segmentation_policy_proof", {})
    volume = manifest.get("volume_computation_proof", {})
    return (
        (
            _string_or_none(analysis.get("input_state_hash")),
            _string_or_none(manifest.get("input_state_hash")),
        ),
        (
            _string_or_none(analysis.get("raw_hash")),
            _string_or_none(manifest.get("raw_hash")),
            _string_or_none(_mapping_get(ai_claim, "raw_kspace_hash")),
            _string_or_none(_mapping_get(measurement, "raw_kspace_hash")),
        ),
        (
            _string_or_none(analysis.get("ai_reconstruction_hash")),
            _string_or_none(_mapping_get(ai_claim, "ai_reconstruction_hash")),
            _string_or_none(_mapping_get(reconstruction, "ai_reconstruction_hash")),
        ),
        (
            _string_or_none(analysis.get("derived_reconstruction_hash")),
            _string_or_none(manifest.get("derived_reconstruction_hash")),
            _string_or_none(_mapping_get(measurement, "derived_reconstruction_hash")),
            _string_or_none(_mapping_get(reconstruction, "derived_reconstruction_hash")),
        ),
        (
            _string_or_none(analysis.get("ai_mask_hash")),
            _string_or_none(manifest.get("ai_mask_hash")),
            _string_or_none(_mapping_get(ai_claim, "ai_mask_hash")),
            _string_or_none(_mapping_get(segmentation, "ai_mask_hash")),
        ),
        (
            _string_or_none(analysis.get("verified_mask_hash")),
            _string_or_none(manifest.get("verified_mask_hash")),
            _string_or_none(_mapping_get(segmentation, "verified_mask_hash")),
            _string_or_none(_mapping_get(volume, "verified_mask_hash")),
        ),
    )


def _build_shape_consistency_check(
    structural_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    shape_check_codes = (
        "raw_real",
        "raw_imag",
        "ai_reconstruction",
        "tumor_mask",
        "geometry_consistent",
        "voxel_spacing_positive",
        "release_payloads_guarded",
    )
    structural_index = {
        str(item["code"]): str(item["status"])
        for item in structural_checks
        if isinstance(item, Mapping)
    }
    selected_statuses = [structural_index.get(code) for code in shape_check_codes]
    if any(status is None for status in selected_statuses):
        status = "not_evaluated"
    elif any(status == "failed" for status in selected_statuses):
        status = "failed"
    else:
        status = "passed"
    return _operational_check_entry(
        code=OPERATIONAL_CHECK_SHAPE_CONSISTENCY,
        label="Shape consistency",
        status=status,
        message={
            "passed": (
                "Numeric tensors, shared-grid geometry, voxel spacing, and release-shape "
                "constraints remained consistent during pre-check."
            ),
            "failed": (
                "Numeric tensor, shared-grid geometry, voxel spacing, or release-shape "
                "consistency failed during pre-check."
            ),
            "not_evaluated": "Shape-consistency screening was not fully materialized during pre-check.",
        }[status],
    )


def _build_quick_anomaly_detection_check(
    *,
    state: Mapping[str, Any],
    structural_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    structural_index = {
        str(item["code"]): str(item["status"])
        for item in structural_checks
        if isinstance(item, Mapping)
    }
    if structural_index.get("invalid_ai_stack_identity") != "passed":
        status = "not_evaluated"
    else:
        ai_stack_identity = state.get("ai_stack_identity")
        compatibility = (
            ai_stack_identity.get("sequence_model_compatibility")
            if isinstance(ai_stack_identity, Mapping)
            else None
        )
        compatibility_type = (
            compatibility.get("compatibility_type")
            if isinstance(compatibility, Mapping)
            else None
        )
        if compatibility_type == "sequence_family":
            status = "advisory"
        elif compatibility_type == "exact_sequence_id":
            status = "passed"
        else:
            status = "not_evaluated"
    return _operational_check_entry(
        code=OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION,
        label="Quick anomaly detection",
        status=status,
        message={
            "passed": "Low-cost pre-check screening found no approved intake-adjacent anomaly signal.",
            "advisory": (
                "Low-cost pre-check screening recorded a sequence-family AI stack compatibility "
                "declaration for operator awareness."
            ),
            "not_evaluated": (
                "No approved low-cost intake-adjacent anomaly screening rule was applicable "
                "before full proof."
            ),
        }[status],
    )


def _derive_pre_check_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(item["code"]): str(item["status"]) for item in checks}
    if any(statuses.get(code) == "failed" for code in _PRE_CHECK_BLOCKING_CHECKS):
        return "failed"
    if statuses.get(OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION) == "advisory" and all(
        statuses.get(code) == "passed" for code in _PRE_CHECK_BLOCKING_CHECKS
    ):
        return "advisory"
    return "passed"


def _pre_check_summary(
    checks: list[dict[str, Any]],
    status: str,
) -> str:
    statuses = {str(item["code"]): str(item["status"]) for item in checks}
    if status == "failed":
        if statuses[OPERATIONAL_CHECK_STRUCTURE_VALIDATION] == "failed":
            return "Pre-check failed because intake or structural admissibility did not pass."
        if statuses[OPERATIONAL_CHECK_HASH_CHECKS] == "failed":
            return "Pre-check failed because reusable hash-surface consistency did not pass."
        return "Pre-check failed because shape consistency did not pass."
    if status == "advisory":
        return (
            "Pre-check passed its blocking checks and recorded a non-blocking "
            "intake-adjacent anomaly signal."
        )
    if statuses[OPERATIONAL_CHECK_QUICK_ANOMALY_DETECTION] == "passed":
        return "Pre-check passed its blocking checks and approved low-cost anomaly screening."
    return "Pre-check passed its blocking checks without an applicable anomaly-screening rule."


def _build_full_proof_operational_stage_results(
    proof_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_index = {
        str(item["proof_stage"]): str(item["status"])
        for item in proof_stages
        if isinstance(item, Mapping)
    }
    return [
        {
            "proof_stage": proof_stage,
            "title": _proof_stage_title(proof_stage),
            "status": stage_index.get(proof_stage, "not_evaluated"),
            "summary": _proof_stage_summary(
                proof_stage=proof_stage,
                status=stage_index.get(proof_stage, "not_evaluated"),
            ),
        }
        for proof_stage in _PROOF_STAGE_ORDER
    ]


def _derive_full_proof_status(proof_stages: list[dict[str, Any]]) -> str:
    statuses = [str(item["status"]) for item in proof_stages]
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "passed" for status in statuses):
        return "passed"
    if all(status == "not_evaluated" for status in statuses):
        return "not_evaluated"
    raise ValueError(
        "full_proof.proof_stages must be all passed, contain a failed stage, or be all not_evaluated."
    )


def _full_proof_summary(status: str) -> str:
    return {
        "passed": "Full proof passed because every canonical proof-core stage passed.",
        "failed": "Full proof failed because at least one canonical proof-core stage did not pass.",
        "not_evaluated": (
            "Full proof was not evaluated because canonical proof-core was not entered."
        ),
    }[status]


def _validate_operational_subphases(payload: Mapping[str, Any]) -> None:
    if str(payload.get("schema")) != _OPERATIONAL_SUBPHASE_SCHEMA:
        raise ValueError("operational_subphases.schema must be the frozen derived schema identifier.")
    if list(payload.get("subphase_order", [])) != list(_OPERATIONAL_SUBPHASE_ORDER):
        raise ValueError("operational_subphases.subphase_order must match the frozen order.")
    subphases = payload.get("subphases")
    if not isinstance(subphases, Mapping):
        raise ValueError("operational_subphases.subphases must be a mapping.")
    if sorted(subphases.keys()) != sorted(_OPERATIONAL_SUBPHASE_ORDER):
        raise ValueError("operational_subphases must emit only the frozen subphase vocabulary.")

    pre_check = subphases[OPERATIONAL_SUBPHASE_PRE_CHECK]
    if not isinstance(pre_check, Mapping):
        raise ValueError("operational_subphases.pre_check must be a mapping.")
    if pre_check.get("blocking") is not True:
        raise ValueError("operational_subphases.pre_check.blocking must be true.")
    pre_check_status = str(pre_check.get("status"))
    if pre_check_status not in _PRE_CHECK_ALLOWED_STATUSES:
        raise ValueError("operational_subphases.pre_check.status is invalid.")
    pre_check_checks = pre_check.get("checks")
    if not isinstance(pre_check_checks, list):
        raise ValueError("operational_subphases.pre_check.checks must be a list.")
    check_codes = [str(item.get("code")) for item in pre_check_checks if isinstance(item, Mapping)]
    if check_codes != list(_PRE_CHECK_CHECK_ORDER):
        raise ValueError(
            "operational_subphases.pre_check.checks must follow the frozen check vocabulary and order."
        )
    for item in pre_check_checks:
        if not isinstance(item, Mapping):
            raise ValueError("operational_subphases.pre_check.checks entries must be mappings.")
        code = str(item["code"])
        status = str(item.get("status"))
        if status not in _PRE_CHECK_CHECK_ALLOWED_STATUSES[code]:
            raise ValueError(f"operational_subphases.pre_check.{code} has an invalid status.")
        if type(item.get("label")) is not str or not str(item["label"]).strip():
            raise ValueError("operational_subphases.pre_check labels must be non-empty strings.")
        if type(item.get("message")) is not str or not str(item["message"]).strip():
            raise ValueError("operational_subphases.pre_check messages must be non-empty strings.")
    derived_pre_check_status = _derive_pre_check_status(list(pre_check_checks))
    if pre_check_status != derived_pre_check_status:
        raise ValueError("operational_subphases.pre_check.status must match its frozen derivation.")
    blocking_statuses = {
        str(item["code"]): str(item["status"])
        for item in pre_check_checks
        if str(item["code"]) in _PRE_CHECK_BLOCKING_CHECKS
    }
    if pre_check_status == "advisory" and any(
        blocking_statuses.get(code) != "passed" for code in _PRE_CHECK_BLOCKING_CHECKS
    ):
        raise ValueError(
            "operational_subphases.pre_check cannot be advisory unless every blocking check passed."
        )

    full_proof = subphases[OPERATIONAL_SUBPHASE_FULL_PROOF]
    if not isinstance(full_proof, Mapping):
        raise ValueError("operational_subphases.full_proof must be a mapping.")
    if full_proof.get("blocking") is not True:
        raise ValueError("operational_subphases.full_proof.blocking must be true.")
    full_proof_status = str(full_proof.get("status"))
    if full_proof_status not in _FULL_PROOF_ALLOWED_STATUSES:
        raise ValueError("operational_subphases.full_proof.status is invalid.")
    if list(full_proof.get("proof_stage_order", [])) != list(_PROOF_STAGE_ORDER):
        raise ValueError("operational_subphases.full_proof.proof_stage_order must be canonical.")
    full_proof_stages = full_proof.get("proof_stages")
    if not isinstance(full_proof_stages, list) or len(full_proof_stages) != len(_PROOF_STAGE_ORDER):
        raise ValueError(
            "operational_subphases.full_proof.proof_stages must materialize every canonical proof stage."
        )
    stage_names = [str(item.get("proof_stage")) for item in full_proof_stages if isinstance(item, Mapping)]
    if stage_names != list(_PROOF_STAGE_ORDER):
        raise ValueError(
            "operational_subphases.full_proof.proof_stages must use canonical proof-stage identifiers."
        )
    for item in full_proof_stages:
        if not isinstance(item, Mapping):
            raise ValueError("operational_subphases.full_proof.proof_stages entries must be mappings.")
        status = str(item.get("status"))
        if status not in _FULL_PROOF_STAGE_ALLOWED_STATUSES:
            raise ValueError("operational_subphases.full_proof proof-stage status is invalid.")
        if type(item.get("title")) is not str or not str(item["title"]).strip():
            raise ValueError("operational_subphases.full_proof proof-stage titles must be non-empty.")
        if type(item.get("summary")) is not str or not str(item["summary"]).strip():
            raise ValueError("operational_subphases.full_proof proof-stage summaries must be non-empty.")
    if full_proof_status != _derive_full_proof_status(list(full_proof_stages)):
        raise ValueError("operational_subphases.full_proof.status must match its frozen derivation.")


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping_get(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _build_verification_pipeline(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    structural_checks = _build_structural_validity_checks(state)
    structural_status = _tier_status_from_checks(structural_checks)
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
    proof_stages = _build_proof_core_stage_results(manifest)
    proof_core_status = _tier_status_from_checks(proof_stages)
    profile_policy_status = manifest["decision_summary"]["profile_policy_status"]
    payload = {
        "schema": "canonical_assurance_flow",
        "ordering_semantics": (
            "Canonical assurance-flow schema and evidence-ordering model. "
            "It does not mandate runtime execution topology."
        ),
        "phase_order": [
            PIPELINE_PHASE_INTAKE_VALIDATION,
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            PIPELINE_PHASE_PROOF_CORE,
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
            PIPELINE_PHASE_POLICY_EVALUATION,
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            PIPELINE_PHASE_RELEASE_CONTROL,
        ],
        "phases": {
            PIPELINE_PHASE_INTAKE_VALIDATION: _phase_entry(
                title="Intake Validation",
                status="passed",
                summary="Profile selection and input intake completed without blocking intake faults.",
                blocking=True,
            ),
            PIPELINE_PHASE_STRUCTURAL_VALIDITY: _phase_entry(
                title="Structural Validity",
                status=structural_status,
                summary=(
                    "ICSD structural checks passed for identifiers, numeric fields, voxel geometry, "
                    "and release payload shape."
                    if structural_status == "passed"
                    else "ICSD structural checks failed before proof-core results could be trusted."
                ),
                blocking=True,
                checks=structural_checks,
            ),
            PIPELINE_PHASE_PROOF_CORE: _phase_entry(
                title="Proof Core",
                status=proof_core_status,
                summary=(
                    "Measurement anchoring, reconstruction parity, segmentation policy, and volume computation all verified."
                    if proof_core_status == "passed"
                    else "At least one proof-core stage failed."
                ),
                blocking=True,
                proof_stages=proof_stages,
            ),
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY: _phase_entry(
                title="Clinical Plausibility",
                status=str(clinical_section["status"]),
                summary=str(clinical_section["summary"]),
                blocking=bool(clinical_section["blocking_configured"]),
                checks=clinical_checks,
            ),
            PIPELINE_PHASE_POLICY_EVALUATION: _phase_entry(
                title="Policy Evaluation",
                status="blocked" if profile_policy_status == "blocked" else "passed",
                summary=(
                    "A configured non-proof policy blocked final release."
                    if profile_policy_status == "blocked"
                    else "No non-proof policy blocked final release."
                ),
                blocking=True,
            ),
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION: _phase_entry(
                title="Multi-Series Aggregation",
                status="not_applicable",
                summary="Single-series reports do not execute study-level aggregation.",
                blocking=True,
            ),
            PIPELINE_PHASE_RELEASE_CONTROL: _phase_entry(
                title="Release Control",
                status="released" if manifest["decision_summary"]["final_release_allowed"] else "blocked",
                summary=(
                    "The case passed release control."
                    if manifest["decision_summary"]["final_release_allowed"]
                    else "The case did not pass release control."
                ),
                blocking=True,
            ),
        },
    }
    payload["operational_subphases"] = _build_operational_subphases(
        state=state,
        analysis=analysis,
        manifest=manifest,
        structural_checks=structural_checks,
        intake_blocked=False,
        proof_stages=proof_stages,
    )
    return payload


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
        _invariant_check_result(
            code="invalid_ai_stack_identity",
            label="Declared AI stack identity",
            failure=_check_ai_stack_identity(state),
            success_message=(
                "Declared AI stack identity is absent or canonically valid for the active case context."
            ),
        ),
        _invariant_check_result(
            code="study_acquired_at_valid",
            label="Study acquisition timestamp",
            failure=_check_study_acquired_at(state),
            success_message="study_acquired_at is absent or uses the canonical UTC timestamp format.",
        ),
        _invariant_check_result(
            code="study_timestamp_provenance_valid",
            label="Study timestamp provenance",
            failure=_check_study_timestamp_provenance(state),
            success_message="study_timestamp_provenance is absent or uses an approved current-case provenance class.",
        ),
        _invariant_check_result(
            code="spatial_geometry_valid",
            label="Spatial geometry metadata",
            failure=_check_spatial_geometry(state),
            success_message="spatial_geometry is absent or canonically valid for temporal comparison.",
        ),
        _invariant_check_result(
            code="prior_verified_context_valid",
            label="Prior verified context",
            failure=_check_prior_verified_context(state),
            success_message="prior_verified_context is absent or canonically valid for longitudinal comparison.",
        ),
        _invariant_check_result(
            code="prior_verified_contexts_valid",
            label="Prior verified context set",
            failure=_check_prior_verified_contexts(state),
            success_message="prior_verified_contexts is absent or canonically valid for longitudinal comparison.",
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


def _build_proof_core_checks(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _physical_check_result(
            code="measurement_integrity_proof",
            label="Raw k-space re-projection",
            verified=manifest["measurement_integrity_proof"]["verified"],
            success_message=(
                "The deterministic reconstruction reprojects back to the supplied raw k-space."
            ),
            failure_message=(
                "The deterministic reconstruction does not cleanly reproject back to the "
                "supplied raw k-space."
            ),
        ),
        _physical_check_result(
            code="reconstruction_parity_proof",
            label="AI reconstruction parity",
            verified=manifest["reconstruction_parity_proof"]["verified"],
            success_message=(
                "The AI reconstruction matches the deterministic reconstruction on the measured grid."
            ),
            failure_message=(
                "The AI reconstruction diverges from the deterministic reconstruction."
            ),
        ),
        _physical_check_result(
            code="segmentation_policy_proof",
            label="Verified mask parity",
            verified=manifest["segmentation_policy_proof"]["verified"],
            success_message="The AI tumour mask matches the deterministic verified mask.",
            failure_message=(
                "The AI tumour mask diverges from the deterministic verified mask."
            ),
        ),
        _physical_check_result(
            code="volume_computation_proof",
            label="Tumour volume proof",
            verified=manifest["volume_computation_proof"]["verified"],
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
            verified=manifest["proof_core_summary"]["no_invented_pixels"],
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
        "measurement_integrity_proof",
        "reconstruction_parity_proof",
        "segmentation_policy_proof",
        "volume_computation_proof",
    )


def _build_failure_reasons(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if not manifest["measurement_integrity_proof"]["verified"]:
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
                    "The derived reconstruction could not be justified against the supplied raw "
                    "measurements under the explicit per-frequency and global-norm tolerance policy."
                ),
                "details": {
                    "pipeline_phase": PIPELINE_PHASE_PROOF_CORE,
                    "proof_stage": PROOF_STAGE_MEASUREMENT_INTEGRITY,
                    "relative_l2_error": analysis["relative_measurement_l2_error"],
                    "max_imaginary_leakage": analysis["max_imaginary_leakage"],
                    "metrics": manifest["measurement_integrity_proof"]["metrics"],
                },
            }
        )
    intensity_delta = int(analysis["reconstruction_mismatch_voxel_count"])
    if not manifest["reconstruction_parity_proof"]["verified"]:
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
                    "pipeline_phase": PIPELINE_PHASE_PROOF_CORE,
                    "proof_stage": PROOF_STAGE_RECONSTRUCTION_PARITY,
                    "max_abs_error": analysis["max_ai_abs_error"],
                    "relative_l2_error": analysis["relative_ai_l2_error"],
                    "metrics": manifest["reconstruction_parity_proof"]["metrics"],
                },
            }
        )
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
                    "pipeline_phase": PIPELINE_PHASE_PROOF_CORE,
                    "proof_stage": PROOF_STAGE_SEGMENTATION_POLICY,
                    "false_positive_count": int(analysis["mask_false_positive_count"]),
                    "false_negative_count": int(analysis["mask_false_negative_count"]),
                },
            }
        )
    if not manifest["volume_computation_proof"]["verified"]:
        claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
        verified_volume = manifest["volume_computation_proof"]["computed_mm3"]
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
                    "pipeline_phase": PIPELINE_PHASE_PROOF_CORE,
                    "proof_stage": PROOF_STAGE_VOLUME_COMPUTATION,
                    "claimed_mm3": claimed_volume,
                    "verified_mm3": verified_volume,
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
                "details": {
                    "pipeline_phase": PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
                    "source": "clinical_plausibility",
                },
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
                    "pipeline_phase": PIPELINE_PHASE_POLICY_EVALUATION,
                    "profile_name": manifest["profile_name"],
                    "decision_class": manifest["decision_summary"]["decision_class"],
                },
            }
        )
    return reasons


def _decision_reasons_from_failure_reasons(
    failure_reasons: list[dict[str, Any]],
    *,
    pipeline_phase: str,
    severity: str,
    primary_code: str | None = None,
    primary_message: str | None = None,
    proof_stage: str | None = None,
) -> list[dict[str, Any]]:
    if not failure_reasons and primary_code is None:
        return []
    reasons: list[dict[str, Any]] = []
    if primary_code is not None and primary_message is not None:
        reasons.append(
            _decision_reason(
                code=primary_code,
                pipeline_phase=pipeline_phase,
                severity=severity,
                message=primary_message,
                band=DECISION_REASON_BAND_PRIMARY,
                proof_stage=proof_stage,
            )
        )
        supporting = sorted(failure_reasons, key=lambda item: str(item["code"]))
    else:
        primary = failure_reasons[0]
        reasons.append(
            _decision_reason(
                code=str(primary["code"]),
                pipeline_phase=str(
                    primary.get("details", {}).get("pipeline_phase", pipeline_phase)
                ),
                severity=severity,
                message=str(primary["message"]),
                band=DECISION_REASON_BAND_PRIMARY,
                proof_stage=str(primary.get("details", {}).get("proof_stage", proof_stage))
                if primary.get("details", {}).get("proof_stage", proof_stage) is not None
                else None,
            )
        )
        supporting = sorted(failure_reasons[1:], key=lambda item: str(item["code"]))
    reasons.extend(
        _decision_reason(
            code=str(item["code"]),
            pipeline_phase=str(item.get("details", {}).get("pipeline_phase", pipeline_phase)),
            severity=severity,
            message=str(item["message"]),
            band=DECISION_REASON_BAND_SAME_STAGE,
            proof_stage=str(item.get("details", {}).get("proof_stage", proof_stage))
            if item.get("details", {}).get("proof_stage", proof_stage) is not None
            else None,
        )
        for item in supporting
    )
    return reasons


def _collect_near_boundary_metric_reasons(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    metric_specs = (
        (
            "measurement_integrity_proof",
            "per_frequency_complex_abs_error",
            "measurement_per_frequency_complex_abs_error_near_boundary",
            "The frequency-domain per-coefficient absolute error remained within tolerance but inside the configured guard band.",
            PROOF_STAGE_MEASUREMENT_INTEGRITY,
        ),
        (
            "measurement_integrity_proof",
            "global_relative_l2_error",
            "measurement_global_relative_l2_error_near_boundary",
            "The frequency-domain global relative L2 error remained within tolerance but inside the configured guard band.",
            PROOF_STAGE_MEASUREMENT_INTEGRITY,
        ),
        (
            "measurement_integrity_proof",
            "spatial_imaginary_leakage",
            "measurement_spatial_imaginary_leakage_near_boundary",
            "The inverse-FFT imaginary leakage remained within tolerance but inside the configured guard band.",
            PROOF_STAGE_MEASUREMENT_INTEGRITY,
        ),
        (
            "reconstruction_parity_proof",
            "per_voxel_abs_error",
            "ai_reconstruction_per_voxel_abs_error_near_boundary",
            "The AI reconstruction per-voxel absolute error remained within tolerance but inside the configured guard band.",
            PROOF_STAGE_RECONSTRUCTION_PARITY,
        ),
        (
            "reconstruction_parity_proof",
            "global_relative_l2_error",
            "ai_reconstruction_global_relative_l2_error_near_boundary",
            "The AI reconstruction global relative L2 error remained within tolerance but inside the configured guard band.",
            PROOF_STAGE_RECONSTRUCTION_PARITY,
        ),
    )
    reasons: list[dict[str, str]] = []
    for section_name, metric_name, code, message, proof_stage in metric_specs:
        metric = manifest[section_name]["metrics"][metric_name]
        if metric["within_tolerance"] and metric["boundary_status"] == "near_boundary":
            reasons.append({"code": code, "message": message, "proof_stage": proof_stage})
    volume_metric = manifest["volume_computation_proof"]["tolerance_metric"]
    if volume_metric["within_tolerance"] and volume_metric["boundary_status"] == "near_boundary":
        reasons.append(
            {
                "code": "tumor_volume_absolute_error_near_boundary",
                "message": "The tumour-volume absolute error remained within tolerance but inside the configured guard band.",
                "proof_stage": PROOF_STAGE_VOLUME_COMPUTATION,
            }
        )
    return reasons


def _collect_non_blocking_clinical_reasons(
    manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    for entry in manifest["clinical_plausibility"]["heuristics"]:
        if entry["status"] not in {"warning", "advisory"}:
            continue
        reasons.append(
            {
                "code": str(entry["code"]),
                "message": str(entry["message"]),
                "severity": (
                    DECISION_SEVERITY_WARNING
                    if entry["status"] == "warning"
                    else DECISION_SEVERITY_INFO
                ),
            }
        )
    return reasons


def _build_single_series_decision_lineage(report: Mapping[str, Any]) -> list[dict[str, str]]:
    phases = report["verification_pipeline"]["phases"]
    return [
        _decision_lineage_entry(
            PIPELINE_PHASE_INTAKE_VALIDATION,
            str(phases[PIPELINE_PHASE_INTAKE_VALIDATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            str(phases[PIPELINE_PHASE_STRUCTURAL_VALIDITY]["status"]),
        ),
        *[
            _decision_lineage_entry(
                PIPELINE_PHASE_PROOF_CORE,
                str(stage["status"]),
                proof_stage=str(stage["proof_stage"]),
            )
            for stage in phases[PIPELINE_PHASE_PROOF_CORE]["proof_stages"]
        ],
        _decision_lineage_entry(
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
            str(phases[PIPELINE_PHASE_CLINICAL_PLAUSIBILITY]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_POLICY_EVALUATION,
            str(phases[PIPELINE_PHASE_POLICY_EVALUATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            str(phases[PIPELINE_PHASE_MULTI_SERIES_AGGREGATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_RELEASE_CONTROL,
            str(phases[PIPELINE_PHASE_RELEASE_CONTROL]["status"]),
        ),
    ]


def _build_single_series_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    manifest = report["manifest"]
    failure_reasons = list(report["failure_reasons"])
    decision_summary = report["decision_summary"]
    boundary_reasons = _collect_near_boundary_metric_reasons(manifest)
    advisory_reasons = _collect_non_blocking_clinical_reasons(manifest)
    success_modifiers = _default_success_modifiers()
    success_modifiers["near_proof_threshold"] = bool(boundary_reasons)
    success_modifiers["has_non_blocking_advisory"] = bool(advisory_reasons)
    success_modifiers["near_boundary_metrics"] = [item["code"] for item in boundary_reasons]
    success_modifiers["advisory_reason_codes"] = [item["code"] for item in advisory_reasons]
    ordered_boundary_reasons = sorted(
        boundary_reasons,
        key=lambda entry: (
            _PROOF_STAGE_ORDER.index(str(entry["proof_stage"])),
            str(entry["code"]),
        ),
    )
    primary_boundary_stage = (
        None if not ordered_boundary_reasons else str(ordered_boundary_reasons[0]["proof_stage"])
    )

    if report["release_eligible"]:
        if advisory_reasons:
            reasons = [
                _decision_reason(
                    code="clinical_plausibility_non_blocking_finding",
                    pipeline_phase=PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
                    severity=DECISION_SEVERITY_WARNING,
                    message=manifest["clinical_plausibility"]["summary"],
                    band=DECISION_REASON_BAND_PRIMARY,
                )
            ]
            reasons.extend(
                _decision_reason(
                    code=item["code"],
                    pipeline_phase=PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
                    severity=item["severity"],
                    message=item["message"],
                    band=DECISION_REASON_BAND_SAME_STAGE,
                )
                for item in sorted(advisory_reasons, key=lambda entry: entry["code"])
            )
            reasons.extend(
                _decision_reason(
                    code=item["code"],
                    pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
                    severity=DECISION_SEVERITY_WARNING,
                    message=item["message"],
                    band=DECISION_REASON_BAND_SECONDARY_MODIFIER,
                    proof_stage=str(item["proof_stage"]),
                )
                for item in ordered_boundary_reasons
            )
            return _build_decision(
                final_disposition="released",
                decision_class=DECISION_CLASS_VERIFIED_WITH_ADVISORY,
                severity=DECISION_SEVERITY_WARNING,
                pipeline_phase=PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
                reasons=reasons,
                success_modifiers=success_modifiers,
                decision_lineage=_build_single_series_decision_lineage(report),
            )
        if boundary_reasons:
            reasons = [
                _decision_reason(
                    code="proof_metric_near_boundary",
                    pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        "The deterministic proof passed, but at least one eligible proof metric "
                        "fell within the configured guard band of a failure threshold."
                    ),
                    band=DECISION_REASON_BAND_PRIMARY,
                    proof_stage=primary_boundary_stage,
                )
            ]
            reasons.extend(
                _decision_reason(
                    code=item["code"],
                    pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
                    severity=DECISION_SEVERITY_WARNING,
                    message=item["message"],
                    band=DECISION_REASON_BAND_SAME_STAGE,
                    proof_stage=str(item["proof_stage"]),
                )
                for item in ordered_boundary_reasons
            )
            return _build_decision(
                final_disposition="released",
                decision_class=DECISION_CLASS_VERIFIED_BOUNDARY_HUGGING,
                severity=DECISION_SEVERITY_WARNING,
                pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
                reasons=reasons,
                success_modifiers=success_modifiers,
                decision_lineage=_build_single_series_decision_lineage(report),
                proof_stage=primary_boundary_stage,
            )
        return _build_decision(
            final_disposition="released",
            decision_class=DECISION_CLASS_VERIFIED_CLEAN,
            severity=DECISION_SEVERITY_INFO,
            pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
            reasons=[
                _decision_reason(
                    code="final_release_authorized",
                    pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
                    severity=DECISION_SEVERITY_INFO,
                    message=(
                        "All structural, physical, clinical, and release-control checks passed "
                        "without advisory findings or near-boundary proof metrics."
                    ),
                    band=DECISION_REASON_BAND_PRIMARY,
                )
            ],
            success_modifiers=success_modifiers,
            decision_lineage=_build_single_series_decision_lineage(report),
        )

    if failure_reasons and failure_reasons[0]["category"] == STRUCTURAL_FAILURE_CATEGORY:
        primary_phase = str(
            failure_reasons[0].get("details", {}).get(
                "pipeline_phase",
                PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            )
        )
        return _build_decision(
            final_disposition="blocked",
            decision_class=DECISION_CLASS_BLOCKED_STRUCTURAL,
            severity=DECISION_SEVERITY_ERROR,
            pipeline_phase=primary_phase,
            reasons=[
                _decision_reason(
                    code=str(item["code"]),
                    pipeline_phase=str(
                        item.get("details", {}).get(
                            "pipeline_phase",
                            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
                        )
                    ),
                    severity=DECISION_SEVERITY_ERROR,
                    message=str(item["message"]),
                    band=(
                        DECISION_REASON_BAND_PRIMARY
                        if index == 0
                        else DECISION_REASON_BAND_SAME_STAGE
                    ),
                )
                for index, item in enumerate(failure_reasons)
            ],
            success_modifiers=success_modifiers,
            decision_lineage=_build_single_series_decision_lineage(report),
        )

    if decision_summary["proof_status"] == "failed":
        physical_reasons = [
            reason for reason in failure_reasons if reason["category"] == "proof_failure"
        ]
        earliest_failing_proof_stage = str(
            physical_reasons[0]["details"]["proof_stage"]
        )
        return _build_decision(
            final_disposition="blocked",
            decision_class=DECISION_CLASS_BLOCKED_PHYSICAL,
            severity=DECISION_SEVERITY_CRITICAL,
            pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
            reasons=_decision_reasons_from_failure_reasons(
                physical_reasons,
                pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
                severity=DECISION_SEVERITY_CRITICAL,
            ),
            success_modifiers=success_modifiers,
            decision_lineage=_build_single_series_decision_lineage(report),
            proof_stage=earliest_failing_proof_stage,
        )

    if decision_summary["decision_class"] == DECISION_CLASS_TRIAGE_ONLY:
        reasons = _decision_reasons_from_failure_reasons(
            [
                reason
                for reason in failure_reasons
                if reason["category"] == "profile_policy_block"
            ],
            pipeline_phase=PIPELINE_PHASE_POLICY_EVALUATION,
            severity=DECISION_SEVERITY_ERROR,
        )
    else:
        heuristic_block_reasons = [
            reason
            for reason in failure_reasons
            if reason["category"] == "heuristic_block"
        ]
        reasons = [
            _decision_reason(
                code="policy_blocked_by_clinical_plausibility",
                pipeline_phase=PIPELINE_PHASE_POLICY_EVALUATION,
                severity=DECISION_SEVERITY_ERROR,
                message=(
                    "Structure and proof passed, but a configured blocking clinical policy prevented final release."
                ),
                band=DECISION_REASON_BAND_PRIMARY,
            )
        ]
        reasons.extend(
            _decision_reason(
                code=str(item["code"]),
                pipeline_phase=PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
                severity=DECISION_SEVERITY_ERROR,
                message=str(item["message"]),
                band=DECISION_REASON_BAND_SECONDARY_MODIFIER,
            )
            for item in sorted(heuristic_block_reasons, key=lambda entry: str(entry["code"]))
        )
    return _build_decision(
        final_disposition="blocked",
        decision_class=DECISION_CLASS_BLOCKED_POLICY,
        severity=DECISION_SEVERITY_ERROR,
        pipeline_phase=PIPELINE_PHASE_POLICY_EVALUATION,
        reasons=reasons,
        success_modifiers=success_modifiers,
        decision_lineage=_build_single_series_decision_lineage(report),
    )


def _decision_block_precedence(decision_class: str) -> int:
    order = {
        DECISION_CLASS_BLOCKED_STRUCTURAL: 0,
        DECISION_CLASS_BLOCKED_PHYSICAL: 1,
        DECISION_CLASS_BLOCKED_POLICY: 2,
    }
    return order.get(decision_class, 3)


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_evaluated"
    priority = {"failed": 0, "warning": 1, "advisory": 2, "passed": 3, "not_evaluated": 4}
    return min(statuses, key=lambda item: priority.get(item, 5))


def _build_multi_series_decision_lineage(report: Mapping[str, Any]) -> list[dict[str, str]]:
    phases = report["verification_pipeline"]["phases"]
    return [
        _decision_lineage_entry(
            PIPELINE_PHASE_INTAKE_VALIDATION,
            str(phases[PIPELINE_PHASE_INTAKE_VALIDATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_STRUCTURAL_VALIDITY,
            str(phases[PIPELINE_PHASE_STRUCTURAL_VALIDITY]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_PROOF_CORE,
            str(phases[PIPELINE_PHASE_PROOF_CORE]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
            str(phases[PIPELINE_PHASE_CLINICAL_PLAUSIBILITY]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_POLICY_EVALUATION,
            str(phases[PIPELINE_PHASE_POLICY_EVALUATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            str(phases[PIPELINE_PHASE_MULTI_SERIES_AGGREGATION]["status"]),
        ),
        _decision_lineage_entry(
            PIPELINE_PHASE_RELEASE_CONTROL,
            str(phases[PIPELINE_PHASE_RELEASE_CONTROL]["status"]),
        ),
    ]


def _build_multi_series_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    downstream_reports = [
        item for item in report["series_reports"] if item["downstream_release_allowed"]
    ]
    success_modifiers = _default_success_modifiers()
    success_modifiers["near_boundary_series_ids"] = [
        item["series_id"]
        for item in downstream_reports
        if item["decision"]["success_modifiers"]["near_proof_threshold"]
    ]
    success_modifiers["advisory_series_ids"] = [
        item["series_id"]
        for item in downstream_reports
        if item["decision"]["success_modifiers"]["has_non_blocking_advisory"]
    ]
    success_modifiers["near_proof_threshold"] = bool(
        success_modifiers["near_boundary_series_ids"]
    )
    success_modifiers["has_non_blocking_advisory"] = bool(
        success_modifiers["advisory_series_ids"]
    )
    success_modifiers["near_boundary_metrics"] = [
        f"{item['series_id']}:{metric_code}"
        for item in downstream_reports
        for metric_code in item["decision"]["success_modifiers"]["near_boundary_metrics"]
    ]
    success_modifiers["advisory_reason_codes"] = [
        f"{item['series_id']}:{reason_code}"
        for item in downstream_reports
        for reason_code in item["decision"]["success_modifiers"]["advisory_reason_codes"]
    ]

    if report["release_eligible"]:
        if success_modifiers["has_non_blocking_advisory"]:
            reasons = [
                _decision_reason(
                    code="study_contains_non_blocking_advisory",
                    pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        "At least one downstream-released series carried non-blocking advisory or warning findings."
                    ),
                    band=DECISION_REASON_BAND_PRIMARY,
                )
            ]
            reasons.extend(
                _decision_reason(
                    code=f"downstream_series_{series_id}_with_advisory",
                    pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        f"Downstream-released series {series_id!r} carries non-blocking advisory findings."
                    ),
                    band=DECISION_REASON_BAND_SAME_STAGE,
                )
                for series_id in sorted(success_modifiers["advisory_series_ids"])
            )
            reasons.extend(
                _decision_reason(
                    code=f"downstream_series_{series_id}_near_boundary",
                    pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        f"Downstream-released series {series_id!r} also passed with guard-band boundary proximity."
                    ),
                    band=DECISION_REASON_BAND_SECONDARY_MODIFIER,
                )
                for series_id in sorted(success_modifiers["near_boundary_series_ids"])
            )
            return _build_decision(
                final_disposition="released",
                decision_class=DECISION_CLASS_VERIFIED_WITH_ADVISORY,
                severity=DECISION_SEVERITY_WARNING,
                pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                reasons=reasons,
                success_modifiers=success_modifiers,
                decision_lineage=_build_multi_series_decision_lineage(report),
            )
        if success_modifiers["near_proof_threshold"]:
            reasons = [
                _decision_reason(
                    code="study_contains_near_boundary_series",
                    pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        "At least one downstream-released series passed inside the configured proof guard band."
                    ),
                    band=DECISION_REASON_BAND_PRIMARY,
                )
            ]
            reasons.extend(
                _decision_reason(
                    code=f"downstream_series_{series_id}_near_boundary",
                    pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                    severity=DECISION_SEVERITY_WARNING,
                    message=(
                        f"Downstream-released series {series_id!r} passed with guard-band boundary proximity."
                    ),
                    band=DECISION_REASON_BAND_SAME_STAGE,
                )
                for series_id in sorted(success_modifiers["near_boundary_series_ids"])
            )
            return _build_decision(
                final_disposition="released",
                decision_class=DECISION_CLASS_VERIFIED_BOUNDARY_HUGGING,
                severity=DECISION_SEVERITY_WARNING,
                pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                reasons=reasons,
                success_modifiers=success_modifiers,
                decision_lineage=_build_multi_series_decision_lineage(report),
            )
        return _build_decision(
            final_disposition="released",
            decision_class=DECISION_CLASS_VERIFIED_CLEAN,
            severity=DECISION_SEVERITY_INFO,
            pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
            reasons=[
                _decision_reason(
                    code="study_final_release_authorized",
                    pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
                    severity=DECISION_SEVERITY_INFO,
                    message=(
                        "Every release-gating series passed, study-level aggregation verified, and no downstream-released series carried advisory or guard-band boundary signals."
                    ),
                    band=DECISION_REASON_BAND_PRIMARY,
                )
            ],
            success_modifiers=success_modifiers,
            decision_lineage=_build_multi_series_decision_lineage(report),
        )

    block_category = str(report["decision_summary"]["block_category"])
    matching_reason = next(
        (
            item
            for item in report["failure_reasons"]
            if str(item["code"]) == block_category
        ),
        report["failure_reasons"][0],
    )
    required_reports = {
        item["series_id"]: item
        for item in report["series_reports"]
        if item["gating_status"] == "required"
    }

    if block_category in {"missing_required_series", "required_series_geometry_mismatch"}:
        decision_class = DECISION_CLASS_BLOCKED_STRUCTURAL
        severity = DECISION_SEVERITY_ERROR
    elif block_category == "positive_lesion_corroboration_failed":
        decision_class = DECISION_CLASS_BLOCKED_PHYSICAL
        severity = DECISION_SEVERITY_CRITICAL
    elif block_category == "required_series_not_release_eligible":
        inherited_class = min(
            (
                required_reports[series_id]["decision"]["decision_class"]
                for series_id in report["analysis"]["required_series_ids"]
                if not required_reports[series_id]["series_release_eligible"]
            ),
            key=_decision_block_precedence,
        )
        decision_class = inherited_class
        severity = {
            DECISION_CLASS_BLOCKED_STRUCTURAL: DECISION_SEVERITY_ERROR,
            DECISION_CLASS_BLOCKED_PHYSICAL: DECISION_SEVERITY_CRITICAL,
            DECISION_CLASS_BLOCKED_POLICY: DECISION_SEVERITY_ERROR,
        }[decision_class]
    else:
        decision_class = DECISION_CLASS_BLOCKED_POLICY
        severity = DECISION_SEVERITY_ERROR

    reasons = [
        _decision_reason(
            code=str(matching_reason["code"]),
            pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
            severity=severity,
            message=str(matching_reason["message"]),
            band=DECISION_REASON_BAND_PRIMARY,
        )
    ]
    if block_category == "required_series_not_release_eligible":
        blocked_series_ids = sorted(
            series_id
            for series_id in report["analysis"]["required_series_ids"]
            if not required_reports[series_id]["series_release_eligible"]
        )
        reasons.extend(
            _decision_reason(
                code=f"required_series_{series_id}_{required_reports[series_id]['decision']['decision_class']}",
                pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                severity=severity,
                message=(
                    f"Required series {series_id!r} carried child decision class "
                    f"{required_reports[series_id]['decision']['decision_class']}."
                ),
                band=DECISION_REASON_BAND_SAME_STAGE,
            )
            for series_id in blocked_series_ids
        )
    else:
        reasons.extend(
            _decision_reason(
                code=str(item["code"]),
                pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
                severity=severity,
                message=str(item["message"]),
                band=DECISION_REASON_BAND_SAME_STAGE,
            )
            for item in sorted(report["failure_reasons"][1:], key=lambda entry: str(entry["code"]))
        )
    return _build_decision(
        final_disposition="blocked",
        decision_class=decision_class,
        severity=severity,
        pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
        reasons=reasons,
        success_modifiers=success_modifiers,
        decision_lineage=_build_multi_series_decision_lineage(report),
    )


def _derive_verified_mask(
    derived_reconstruction: np.ndarray,
    descriptor: Any,
) -> np.ndarray:
    execution = derive_verified_mask(derived_reconstruction, descriptor.mask_rules)
    return execution.verified_mask.astype(np.int64)


def _build_segmentation_policy_proof_manifest(
    *,
    analysis: Mapping[str, Any],
    verified: bool,
    mask_derivation_policy: Mapping[str, Any],
) -> dict[str, Any]:
    policy = {
        "algorithm": str(mask_derivation_policy["algorithm"]),
        "parameters": dict(mask_derivation_policy["parameters"]),
    }
    comparison = {
        "verified": bool(verified),
        "ai_voxel_count": int(analysis["ai_mask_voxel_count"]),
        "verified_voxel_count": int(analysis["verified_mask_voxel_count"]),
        "voxel_delta": int(analysis["mask_voxel_delta"]),
        "false_positive_count": int(analysis["mask_false_positive_count"]),
        "false_negative_count": int(analysis["mask_false_negative_count"]),
    }
    payload = {
        "verified": bool(verified),
        "policy": policy,
        "comparison": comparison,
        "algorithm": policy["algorithm"],
        "ai_voxel_count": comparison["ai_voxel_count"],
        "verified_voxel_count": comparison["verified_voxel_count"],
        "voxel_delta": comparison["voxel_delta"],
        "false_positive_count": comparison["false_positive_count"],
        "false_negative_count": comparison["false_negative_count"],
    }
    if "threshold" in policy["parameters"]:
        payload["threshold"] = policy["parameters"]["threshold"]
    return payload


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
    *,
    temporal_consistency: Mapping[str, Any],
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
        mask_rules=descriptor.mask_rules,
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
    prior_verified_context = temporal_consistency["prior_verified_context"]
    return {
        "verification_profile_descriptor": verification_profile_descriptor,
        "ai_claim_envelope": ai_claim_envelope,
        "mask_derivation_policy": mask_derivation_policy,
        "verification_tolerance_policy": verification_tolerance_policy,
        "prior_verified_contexts": temporal_consistency["prior_verified_contexts"],
        "prior_verified_context": prior_verified_context,
        "longitudinal_selection_trace": temporal_consistency["selection_trace"],
        "temporal_trajectory_trace": temporal_consistency["trajectory_trace"],
        "longitudinal_registration_trace": temporal_consistency["registration_trace"],
        "temporal_consistency_trace": temporal_consistency["trace"],
    }


def _materialize_proof_core_contracts(
    *,
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contracts: Mapping[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    measurement_integrity_proof = build_measurement_integrity_proof(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        claim_envelope=contracts["ai_claim_envelope"],
        tolerance_policy=contracts["verification_tolerance_policy"],
        raw_kspace_hash=analysis["raw_hash"],
        derived_reconstruction_hash=analysis["derived_reconstruction_hash"],
        forward_projection_hash=analysis["forward_projection_hash"],
        verified=manifest["measurement_integrity_proof"]["verified"],
        metrics=manifest["measurement_integrity_proof"]["metrics"],
    )
    reconstruction_parity_proof = build_reconstruction_parity_proof(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        claim_envelope=contracts["ai_claim_envelope"],
        tolerance_policy=contracts["verification_tolerance_policy"],
        ai_reconstruction_hash=analysis["ai_reconstruction_hash"],
        derived_reconstruction_hash=analysis["derived_reconstruction_hash"],
        verified=manifest["reconstruction_parity_proof"]["verified"],
        metrics=manifest["reconstruction_parity_proof"]["metrics"],
    )
    segmentation_policy_proof = build_segmentation_policy_proof(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        claim_envelope=contracts["ai_claim_envelope"],
        mask_derivation_policy=contracts["mask_derivation_policy"],
        ai_mask_hash=analysis["ai_mask_hash"],
        verified_mask_hash=analysis["verified_mask_hash"],
        verified=manifest["segmentation_policy_proof"]["verified"],
        comparison=manifest["segmentation_policy_proof"]["comparison"],
    )
    volume_computation_proof = build_volume_computation_proof(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        claim_envelope=contracts["ai_claim_envelope"],
        verified_mask_hash=analysis["verified_mask_hash"],
        verified=manifest["volume_computation_proof"]["verified"],
        claimed_tumor_volume_mm3=manifest["volume_computation_proof"]["claimed_mm3"],
        computed_tumor_volume_mm3=manifest["volume_computation_proof"]["computed_mm3"],
        verified_mask_voxel_count=manifest["volume_computation_proof"]["voxel_count"],
        voxel_volume_mm3=manifest["volume_computation_proof"]["voxel_volume_mm3"],
        tolerance_metric=manifest["volume_computation_proof"]["tolerance_metric"],
    )
    proof_core_evidence_trace = build_proof_core_evidence_trace(
        case_id=str(state["case_id"]),
        profile_name=descriptor.profile_name,
        profile_revision=descriptor.profile_revision,
        profile_digest=descriptor.profile_digest(),
        icsd_policy_id=descriptor.icsd_policy_id,
        icsd_policy_version=descriptor.icsd_policy_version,
        verification_profile_descriptor=contracts["verification_profile_descriptor"],
        ai_claim_envelope=contracts["ai_claim_envelope"],
        measurement_integrity_proof=measurement_integrity_proof,
        reconstruction_parity_proof=reconstruction_parity_proof,
        segmentation_policy_proof=segmentation_policy_proof,
        volume_computation_proof=volume_computation_proof,
    )
    return {
        **contracts,
        "measurement_integrity_proof": measurement_integrity_proof,
        "reconstruction_parity_proof": reconstruction_parity_proof,
        "segmentation_policy_proof": segmentation_policy_proof,
        "volume_computation_proof": volume_computation_proof,
        "proof_core_evidence_trace": proof_core_evidence_trace,
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
