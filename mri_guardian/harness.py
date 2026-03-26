"""ICSD-bound MRI verification layer that blocks invented pixels."""

from __future__ import annotations

from copy import deepcopy
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
    build_cross_series_verification_policy,
    build_cross_series_verification_trace,
    build_mask_derivation_policy,
    build_multi_series_verification_profile_descriptor_set,
    build_verification_profile_contract,
    build_verification_tolerance_policy,
    build_verified_reconstruction_trace,
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
        "state": dict(state),
    }


def _build_multi_series_series_state(
    context: Mapping[str, Any],
    series_id: str,
) -> dict[str, Any]:
    entry = dict(context["series_by_id"][series_id])
    return {
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


def _canonical_voxel_spacing_tuple(
    values: list[float] | tuple[float, ...],
) -> tuple[float, float, float]:
    spacing = tuple(_round_float(float(value)) for value in values)
    if len(spacing) != 3:
        raise ValueError("voxel_spacing_mm must contain exactly three elements.")
    return spacing  # type: ignore[return-value]


def _grid_signature_for_series_report(series_report: Mapping[str, Any]) -> dict[str, Any]:
    shape = tuple(np.asarray(series_report["analysis"]["derived_reconstruction"]).shape)
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
    descriptor = verification_profile_descriptor_from_state(series_state)
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


def _build_multi_series_verification_tiers(
    *,
    series_reports: list[dict[str, Any]],
    missing_required_series_ids: tuple[str, ...],
    geometry_verification: Mapping[str, Any],
    lesion_corroboration: Mapping[str, Any],
    final_release_allowed: bool,
) -> dict[str, Any]:
    required_reports = [report for report in series_reports if report["gating_status"] == "required"]
    required_passed = all(report["series_release_eligible"] for report in required_reports)
    auxiliary_failures = [
        report["series_id"]
        for report in series_reports
        if report["gating_status"] == "auxiliary" and not report["series_release_eligible"]
    ]
    return {
        "study_shape": {
            "title": "Study Shape",
            "status": "passed" if not missing_required_series_ids else "failed",
            "summary": (
                "Study-level payload shape and required-series references are structurally coherent."
                if not missing_required_series_ids
                else "The study-level policy references required series that are missing from the payload."
            ),
        },
        "series_proof": {
            "title": "Per-Series Proof",
            "status": "passed" if required_passed else "failed",
            "summary": (
                "Every required series passed its own proof and profile gate."
                if required_passed
                else "At least one required series failed its own proof or profile gate."
            ),
            "auxiliary_failures": auxiliary_failures,
        },
        "cross_series_consistency": {
            "title": "Cross-Series Consistency",
            "status": (
                "passed"
                if geometry_verification["verified"] and lesion_corroboration["verified"]
                else "failed"
            ),
            "summary": (
                "Required-series geometry and positive-lesion corroboration both verified."
                if geometry_verification["verified"] and lesion_corroboration["verified"]
                else "Required-series geometry or positive-lesion corroboration did not verify."
            ),
        },
        "merged_release": {
            "title": "Merged Release",
            "status": "passed" if final_release_allowed else "failed",
            "summary": (
                "The study passed the merged release gate."
                if final_release_allowed
                else "The study did not pass the merged release gate."
            ),
        },
    }


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
    verification_tiers = _build_multi_series_verification_tiers(
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
        "series_manifests": [
            _build_multi_series_series_manifest(report) for report in updated_series_reports
        ],
        "decision_summary": decision_summary,
        "verification_tiers": verification_tiers,
    }
    return {
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
        },
        "contracts": {
            "verification_profile_descriptor": verification_profile_descriptor,
            "cross_series_verification_policy": context["cross_series_policy"],
            "cross_series_verification_trace": cross_series_trace,
        },
        "series_reports": updated_series_reports,
        "manifest": manifest,
        "verification_tiers": verification_tiers,
        "failure_reasons": failure_reasons,
        "decision_summary": decision_summary,
        "release_eligible": final_release_allowed,
    }


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
    analysis = _analyse_case(state, descriptor)
    contracts = _build_claim_to_proof_contracts(state, analysis, descriptor)
    state["verified_manifest"] = _build_verified_manifest(state, analysis, contracts, descriptor)
    state["workstation_payload"] = _build_workstation_payload(state, analysis, descriptor)
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
        "mask_verification": _build_mask_verification_manifest(
            analysis=analysis,
            verified=mask_verified,
            mask_derivation_policy=contracts["mask_derivation_policy"],
        ),
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
    execution = derive_verified_mask(derived_reconstruction, descriptor.mask_rules)
    return execution.verified_mask.astype(np.int64)


def _build_mask_verification_manifest(
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
