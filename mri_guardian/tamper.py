"""Deterministic tamper-mode scenarios and verification threat corpus builders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .contracts import build_verification_profile_contract
from .harness import (
    MULTI_SERIES_CASE_KIND,
    _derive_verified_mask,
    _float_array_to_list,
    _round_float,
    build_multi_series_demo_case,
)
from .mask_engine import (
    CONNECTED_COMPONENT_ALGORITHM,
    SLICE_CONSISTENCY_ALGORITHM,
    derive_verified_mask,
)
from .profiles import (
    DEFAULT_PROFILE_NAME,
    get_verification_profile,
    verification_profile_descriptor_from_dict,
)

THREAT_CORPUS_INTERFACE_TYPE = "VerificationThreatCorpus"
THREAT_CORPUS_REVISION = "v1"
THREAT_CORPUS_SEVERITY_NOTE = (
    "Corpus hazard severity is a qualitative documentation and comparison field. "
    "It is not a patient-specific runtime severity and not a formal clinical risk score."
)
THREAT_CORPUS_NORMATIVE_EXPECTATION_NOTE = (
    "Normative invariants must remain stable unless the scenario meaning changes."
)
THREAT_CORPUS_OBSERVATIONAL_EXPECTATION_NOTE = (
    "Observational expectations are asserted only where intentionally durable."
)

THREAT_CORPUS_STATUS_ACTIVE = "active"
THREAT_CORPUS_STATUS_PROFILE_SCOPED = "profile_scoped"
THREAT_CORPUS_STATUS_DEFERRED = "deferred"
_THREAT_CORPUS_STATUSES = {
    THREAT_CORPUS_STATUS_ACTIVE,
    THREAT_CORPUS_STATUS_PROFILE_SCOPED,
    THREAT_CORPUS_STATUS_DEFERRED,
}

THREAT_ROUTE_STRUCTURAL = "structural"
THREAT_ROUTE_PHYSICAL = "physical"
THREAT_ROUTE_POLICY = "policy"
_THREAT_FAILURE_ROUTES = {
    THREAT_ROUTE_STRUCTURAL,
    THREAT_ROUTE_PHYSICAL,
    THREAT_ROUTE_POLICY,
}

SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY = "reconstruction_integrity"
SCENARIO_FAMILY_MASK_SEMANTICS = "mask_semantics"
SCENARIO_FAMILY_QUANTITATIVE_CLAIMS = "quantitative_claims"
SCENARIO_FAMILY_TEMPORAL_CONTEXT = "temporal_context"
SCENARIO_FAMILY_MULTI_SERIES_CORROBORATION = "multi_series_corroboration"
SCENARIO_FAMILY_PROFILE_SCOPED_POLICY = "profile_scoped_policy"
_SCENARIO_FAMILIES = {
    SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
    SCENARIO_FAMILY_MASK_SEMANTICS,
    SCENARIO_FAMILY_QUANTITATIVE_CLAIMS,
    SCENARIO_FAMILY_TEMPORAL_CONTEXT,
    SCENARIO_FAMILY_MULTI_SERIES_CORROBORATION,
    SCENARIO_FAMILY_PROFILE_SCOPED_POLICY,
}

SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION = "single_series_reconstruction"
SCENARIO_SURFACE_SINGLE_SERIES_MASK = "single_series_mask"
SCENARIO_SURFACE_SINGLE_SERIES_QUANTIFICATION = "single_series_quantification"
SCENARIO_SURFACE_MULTI_SERIES_CORROBORATION = "multi_series_corroboration"
SCENARIO_SURFACE_STATE_REUSE = "state_reuse"
SCENARIO_SURFACE_PROFILE_SCOPED_MASK_POLICY = "profile_scoped_mask_policy"
_SCENARIO_SURFACES = {
    SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
    SCENARIO_SURFACE_SINGLE_SERIES_MASK,
    SCENARIO_SURFACE_SINGLE_SERIES_QUANTIFICATION,
    SCENARIO_SURFACE_MULTI_SERIES_CORROBORATION,
    SCENARIO_SURFACE_STATE_REUSE,
    SCENARIO_SURFACE_PROFILE_SCOPED_MASK_POLICY,
}

HAZARD_SEVERITY_MINOR = "minor"
HAZARD_SEVERITY_MODERATE = "moderate"
HAZARD_SEVERITY_MAJOR = "major"
HAZARD_SEVERITY_CRITICAL = "critical"
_HAZARD_SEVERITIES = {
    HAZARD_SEVERITY_MINOR,
    HAZARD_SEVERITY_MODERATE,
    HAZARD_SEVERITY_MAJOR,
    HAZARD_SEVERITY_CRITICAL,
}

CONTROL_ROLE_PREVENTATIVE = "preventative"
CONTROL_ROLE_DETECTIVE = "detective"
CONTROL_ROLE_GATING = "gating"
CONTROL_ROLE_EVIDENTIARY = "evidentiary"
_CONTROL_ROLES = {
    CONTROL_ROLE_PREVENTATIVE,
    CONTROL_ROLE_DETECTIVE,
    CONTROL_ROLE_GATING,
    CONTROL_ROLE_EVIDENTIARY,
}

CONTROL_SURFACE_TRACE = "trace"
CONTROL_SURFACE_POLICY = "policy"
CONTROL_SURFACE_GATE = "gate"
CONTROL_SURFACE_COMPARISON = "comparison"
_CONTROL_SURFACE_TYPES = {
    CONTROL_SURFACE_TRACE,
    CONTROL_SURFACE_POLICY,
    CONTROL_SURFACE_GATE,
    CONTROL_SURFACE_COMPARISON,
}

_COMPATIBILITY_TAMPER_SLUGS = (
    "fake_tumour",
    "shifted_mask",
    "intensity_manipulation",
)
_MASK_REASON_FAMILY = "mask_voxel_mismatch"
_INTENSITY_REASON_FAMILY = "intensity_manipulation_detected"
_QUANT_REASON_FAMILY = "tumor_volume_mismatch"
_CORROBORATION_REASON_FAMILY = "positive_lesion_corroboration_failed"

_TAMPER_VOXEL_SPACING_MM = np.asarray([0.8, 0.8, 1.8], dtype=np.float64)


def build_tamper_mode_base_case(
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> dict[str, Any]:
    """Return a larger deterministic case suitable for visual tamper demonstrations."""
    descriptor = get_verification_profile(profile_name)
    source_reconstruction = _build_tamper_reconstruction(descriptor)
    raw_kspace = np.fft.fft2(source_reconstruction, axes=(-2, -1))
    raw_real = _float_array_to_list(raw_kspace.real)
    raw_imag = _float_array_to_list(raw_kspace.imag)
    stored_raw_kspace = np.asarray(raw_real, dtype=np.float64) + (
        1j * np.asarray(raw_imag, dtype=np.float64)
    )
    reconstruction = np.fft.ifft2(stored_raw_kspace, axes=(-2, -1)).real
    verified_mask = _derive_verified_mask(reconstruction, descriptor)
    tumor_volume_mm3 = _round_float(
        int(np.count_nonzero(verified_mask)) * float(np.prod(_TAMPER_VOXEL_SPACING_MM))
    )
    return {
        "case_id": "mri-case-tamper-base",
        "patient_pseudonym": "anon-tamper-0001",
        "study_uid": "study-tamper-0001",
        "series_uid": "series-tamper-0001",
        "scanner_id": "scanner-7t-alpha",
        "sequence_id": "t1-post-contrast",
        "ai_model_id": "raw-mri-ai-v4",
        "release_status": "pending_verification",
        "raw_real": raw_real,
        "raw_imag": raw_imag,
        "ai_reconstruction": reconstruction.astype(np.float64).tolist(),
        "tumor_mask": verified_mask.astype(np.int64).tolist(),
        "voxel_spacing_mm": _float_array_to_list(_TAMPER_VOXEL_SPACING_MM),
        "tumor_volume_mm3": tumor_volume_mm3,
        "verification_profile_descriptor": build_verification_profile_contract(
            descriptor.profile_name
        ),
        "verified_manifest": None,
        "workstation_payload": None,
    }


def _build_tamper_reconstruction(descriptor: Any) -> np.ndarray:
    z_index, y_index, x_index = np.indices((6, 24, 24))
    reconstruction = (0.15 * z_index + 0.04 * y_index + 0.02 * x_index).astype(
        np.float64
    )
    tumour_body = (
        ((z_index - 3.0) / 1.4) ** 2
        + ((y_index - 12.0) / 4.6) ** 2
        + ((x_index - 12.0) / 5.3) ** 2
        <= 1.0
    )
    reconstruction += tumour_body * 3.2
    if descriptor.mask_algorithm in {
        CONNECTED_COMPONENT_ALGORITHM,
        SLICE_CONSISTENCY_ALGORITHM,
    }:
        reconstruction[0, 1:3, 20:22] = 3.4
    return reconstruction


def _control(
    control_id: str,
    control_role: str,
    *,
    control_surface_type: str | None = None,
) -> dict[str, str]:
    if control_role not in _CONTROL_ROLES:
        raise ValueError(f"Unsupported control role: {control_role!r}")
    item = {
        "control_id": str(control_id),
        "control_role": str(control_role),
    }
    if control_surface_type is not None:
        if control_surface_type not in _CONTROL_SURFACE_TYPES:
            raise ValueError(
                f"Unsupported control surface type: {control_surface_type!r}"
            )
        item["control_surface_type"] = str(control_surface_type)
    return item


def _mask_semantic_controls() -> list[dict[str, str]]:
    return [
        _control(
            "MaskDerivationPolicy",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_POLICY,
        ),
        _control(
            "manifest.mask_verification.comparison",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_COMPARISON,
        ),
        _control(
            "VerifiedReconstructionTrace",
            CONTROL_ROLE_EVIDENTIARY,
            control_surface_type=CONTROL_SURFACE_TRACE,
        ),
        _control(
            "release_control",
            CONTROL_ROLE_GATING,
            control_surface_type=CONTROL_SURFACE_GATE,
        ),
    ]


def _reconstruction_integrity_controls() -> list[dict[str, str]]:
    return [
        _control(
            "VerificationTolerancePolicy",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_POLICY,
        ),
        _control(
            "manifest.ai_reconstruction_match",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_COMPARISON,
        ),
        _control(
            "VerifiedReconstructionTrace",
            CONTROL_ROLE_EVIDENTIARY,
            control_surface_type=CONTROL_SURFACE_TRACE,
        ),
        _control(
            "release_control",
            CONTROL_ROLE_GATING,
            control_surface_type=CONTROL_SURFACE_GATE,
        ),
    ]


def _quantitative_claim_controls() -> list[dict[str, str]]:
    return [
        _control(
            "manifest.tumor_volume_proof",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_COMPARISON,
        ),
        _control(
            "manifest.mask_verification.comparison",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_COMPARISON,
        ),
        _control(
            "VerifiedReconstructionTrace",
            CONTROL_ROLE_EVIDENTIARY,
            control_surface_type=CONTROL_SURFACE_TRACE,
        ),
        _control(
            "release_control",
            CONTROL_ROLE_GATING,
            control_surface_type=CONTROL_SURFACE_GATE,
        ),
    ]


def _multi_series_corroboration_controls() -> list[dict[str, str]]:
    return [
        _control(
            "manifest.multi_series_aggregation.geometry_verification",
            CONTROL_ROLE_PREVENTATIVE,
            control_surface_type=CONTROL_SURFACE_COMPARISON,
        ),
        _control(
            "cross_series_verification_trace.lesion_corroboration",
            CONTROL_ROLE_DETECTIVE,
            control_surface_type=CONTROL_SURFACE_TRACE,
        ),
        _control(
            "StudyTemporalAggregationTrace",
            CONTROL_ROLE_EVIDENTIARY,
            control_surface_type=CONTROL_SURFACE_TRACE,
        ),
        _control(
            "release_control",
            CONTROL_ROLE_GATING,
            control_surface_type=CONTROL_SURFACE_GATE,
        ),
    ]


def _rewrite_case_for_profile(
    state: dict[str, Any],
    reconstruction: np.ndarray,
) -> None:
    descriptor = verification_profile_descriptor_from_dict(
        dict(state["verification_profile_descriptor"])
    )
    raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
    derived_reconstruction = np.fft.ifft2(raw_kspace, axes=(-2, -1)).real
    verified_mask = derive_verified_mask(
        derived_reconstruction,
        descriptor.mask_rules,
    ).verified_mask
    voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
    state["raw_real"] = raw_kspace.real.tolist()
    state["raw_imag"] = raw_kspace.imag.tolist()
    state["ai_reconstruction"] = derived_reconstruction.tolist()
    state["tumor_mask"] = verified_mask.astype(np.int64).tolist()
    state["tumor_volume_mm3"] = round(
        float(np.count_nonzero(verified_mask) * np.prod(voxel_spacing)),
        9,
    )


def _series_entry(state: Mapping[str, Any], series_id: str) -> dict[str, Any]:
    for entry in state["series"]:
        if entry["series_id"] == series_id:
            return entry
    raise ValueError(f"Series {series_id!r} not found.")


def _rewrite_series_for_profile(
    state: dict[str, Any],
    series_id: str,
    reconstruction: np.ndarray,
) -> None:
    series_entry = _series_entry(state, series_id)
    descriptor = verification_profile_descriptor_from_dict(
        dict(series_entry["verification_profile_descriptor"])
    )
    raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
    derived_reconstruction = np.fft.ifft2(raw_kspace, axes=(-2, -1)).real
    verified_mask = derive_verified_mask(
        derived_reconstruction,
        descriptor.mask_rules,
    ).verified_mask
    voxel_spacing = np.asarray(series_entry["voxel_spacing_mm"], dtype=np.float64)
    series_entry["raw_real"] = raw_kspace.real.tolist()
    series_entry["raw_imag"] = raw_kspace.imag.tolist()
    series_entry["ai_reconstruction"] = derived_reconstruction.tolist()
    series_entry["tumor_mask"] = verified_mask.astype(np.int64).tolist()
    series_entry["tumor_volume_mm3"] = round(
        float(np.count_nonzero(verified_mask) * np.prod(voxel_spacing)),
        9,
    )


def _hot_voxel_reconstruction(
    shape: tuple[int, int, int],
    hot_voxels: tuple[tuple[int, int, int], ...],
    *,
    hot_value: float = 4.0,
) -> np.ndarray:
    reconstruction = np.zeros(shape, dtype=np.float64)
    for z_index, y_index, x_index in hot_voxels:
        reconstruction[z_index, y_index, x_index] = hot_value
    return reconstruction


def _structured_ringing_delta(shape: tuple[int, int, int]) -> np.ndarray:
    z_index, y_index, x_index = np.indices(shape)
    envelope = np.exp(-(((y_index - 12.0) ** 2) + ((x_index - 12.0) ** 2)) / 72.0)
    return 0.22 * np.cos((x_index - 12.0) * np.pi / 3.0) * envelope


def _mixed_frequency_delta(shape: tuple[int, int, int]) -> np.ndarray:
    z_index, y_index, x_index = np.indices(shape)
    low = 0.09 * np.sin((y_index + 1.0) * np.pi / 6.0)
    high = 0.08 * np.sign(np.cos((x_index + z_index + 1.0) * np.pi))
    return low + high


def _shift_mask(mask: np.ndarray, delta_zyx: tuple[int, int, int]) -> np.ndarray:
    delta_z, delta_y, delta_x = delta_zyx
    shifted = np.zeros_like(mask, dtype=np.int64)
    src_z = slice(max(0, -delta_z), mask.shape[0] - max(0, delta_z))
    dst_z = slice(max(0, delta_z), mask.shape[0] - max(0, -delta_z))
    src_y = slice(max(0, -delta_y), mask.shape[1] - max(0, delta_y))
    dst_y = slice(max(0, delta_y), mask.shape[1] - max(0, -delta_y))
    src_x = slice(max(0, -delta_x), mask.shape[2] - max(0, delta_x))
    dst_x = slice(max(0, delta_x), mask.shape[2] - max(0, -delta_x))
    shifted[dst_z, dst_y, dst_x] = mask[src_z, src_y, src_x]
    return shifted


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    eroded = np.asarray(mask, dtype=np.int64).copy()
    eroded[:, :, 16] = 0
    return eroded


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    dilated = np.asarray(mask, dtype=np.int64).copy()
    dilated[1:5, 8:16, 17] = 1
    return dilated


def _state_profile_name(state: Mapping[str, Any]) -> str:
    descriptor = verification_profile_descriptor_from_dict(
        dict(state["verification_profile_descriptor"])
    )
    return str(descriptor.profile_name)


def _build_replay_source_mask(profile_name: str) -> np.ndarray:
    replay_state = build_tamper_mode_base_case(profile_name=profile_name)
    source = np.asarray(replay_state["ai_reconstruction"], dtype=np.float64)
    shifted = np.zeros_like(source)
    shifted[:, :, 1:] = source[:, :, :-1]
    replay_descriptor = verification_profile_descriptor_from_dict(
        dict(replay_state["verification_profile_descriptor"])
    )
    return derive_verified_mask(
        shifted,
        replay_descriptor.mask_rules,
    ).verified_mask.astype(np.int64)


def _threat_entry(
    *,
    corpus_id: str,
    slug: str,
    title: str,
    description: str,
    scenario_family: str,
    scenario_surface: str,
    corpus_status: str,
    technical_failure_mechanism: str,
    clinical_hazard: str,
    downstream_effect: str,
    hazard_severity: str,
    mitigation_mapping: list[dict[str, str]],
    determinism_contract: dict[str, Any],
    expected_decision_class: str,
    expected_failure_route: str,
    state: dict[str, Any],
    evidence_tests: tuple[str, ...],
    evidence_traces: tuple[str, ...],
    expected_block_category: str | None = None,
    expected_primary_reason_family: str | None = None,
    provenance_scope: str | None = None,
) -> dict[str, Any]:
    entry = {
        "corpus_id": str(corpus_id),
        "slug": str(slug),
        "title": str(title),
        "description": str(description),
        "scenario_family": str(scenario_family),
        "scenario_surface": str(scenario_surface),
        "corpus_status": str(corpus_status),
        "technical_failure_mechanism": str(technical_failure_mechanism),
        "clinical_hazard": str(clinical_hazard),
        "downstream_effect": str(downstream_effect),
        "hazard_severity": str(hazard_severity),
        "mitigation_mapping": mitigation_mapping,
        "determinism_contract": determinism_contract,
        "normative_invariants": {
            "expected_decision_class": str(expected_decision_class),
            "expected_failure_route": str(expected_failure_route),
        },
        "observational_expectations": {},
        "evidence": {
            "tests": list(evidence_tests),
            "trace_surfaces": list(evidence_traces),
        },
        "state": state,
        "profile_name": _state_profile_name(state)
        if state.get("case_kind") != MULTI_SERIES_CASE_KIND
        else None,
    }
    if expected_block_category is not None:
        entry["observational_expectations"]["expected_block_category"] = str(
            expected_block_category
        )
    if expected_primary_reason_family is not None:
        entry["observational_expectations"]["expected_primary_reason_family"] = str(
            expected_primary_reason_family
        )
    if provenance_scope is not None:
        entry["provenance_scope"] = str(provenance_scope)
    _validate_threat_entry(entry)
    return entry


def _validate_threat_entry(entry: Mapping[str, Any]) -> None:
    if entry["scenario_family"] not in _SCENARIO_FAMILIES:
        raise ValueError(f"Unsupported scenario family: {entry['scenario_family']!r}")
    if entry["scenario_surface"] not in _SCENARIO_SURFACES:
        raise ValueError(
            f"Unsupported scenario surface: {entry['scenario_surface']!r}"
        )
    if entry["corpus_status"] not in _THREAT_CORPUS_STATUSES:
        raise ValueError(f"Unsupported corpus status: {entry['corpus_status']!r}")
    if entry["hazard_severity"] not in _HAZARD_SEVERITIES:
        raise ValueError(f"Unsupported hazard severity: {entry['hazard_severity']!r}")
    invariants = entry["normative_invariants"]
    if invariants["expected_failure_route"] not in _THREAT_FAILURE_ROUTES:
        raise ValueError(
            f"Unsupported failure route: {invariants['expected_failure_route']!r}"
        )
    mitigation_mapping = list(entry["mitigation_mapping"])
    if not mitigation_mapping:
        raise ValueError("Threat corpus entry is missing mitigation mapping.")
    has_detect_or_prevent = any(
        item["control_role"] in {CONTROL_ROLE_DETECTIVE, CONTROL_ROLE_PREVENTATIVE}
        for item in mitigation_mapping
    )
    has_gate_or_evidence = any(
        item["control_role"] in {CONTROL_ROLE_GATING, CONTROL_ROLE_EVIDENTIARY}
        for item in mitigation_mapping
    )
    if entry["corpus_status"] == THREAT_CORPUS_STATUS_ACTIVE and (
        not has_detect_or_prevent or not has_gate_or_evidence
    ):
        raise ValueError(
            "Active scenarios must include both detection/prevention and "
            "gating/evidentiary controls."
        )
    contract = dict(entry["determinism_contract"])
    for required_key in ("builder_id", "fixed_inputs", "seed_strategy"):
        if required_key not in contract:
            raise ValueError(
                f"Determinism contract missing required field: {required_key!r}"
            )


def _build_fake_tumour_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    mask[0:2, 0:9, 5:24] = 1
    state["tumor_mask"] = mask.tolist()
    state["case_id"] = "mri-case-tamper-fake-tumour"
    return _threat_entry(
        corpus_id="VFTC-001",
        slug="fake_tumour",
        title="Tamper Mode: Fake Tumour Injection",
        description=(
            "Inject unsupported tumour voxels into a cold background region without "
            "touching raw k-space."
        ),
        scenario_family=SCENARIO_FAMILY_MASK_SEMANTICS,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_MASK,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="unsupported_mask_voxel_injection",
        clinical_hazard="false_positive_finding",
        downstream_effect="Unnecessary escalation; possible invasive follow-up.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "single_series.fake_tumour",
            "fixed_inputs": {"z": [0, 2], "y": [0, 9], "x": [5, 24]},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"mask_patch_voxels": 342},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="proof_failure",
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=("test_rejects_ai_mask_with_fake_tumour_voxel",),
        evidence_traces=(
            "manifest.mask_verification.comparison",
            "VerifiedReconstructionTrace",
        ),
    )


def _build_shifted_mask_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    state["tumor_mask"] = _shift_mask(mask, (0, 0, 1)).tolist()
    state["case_id"] = "mri-case-tamper-shifted-mask"
    return _threat_entry(
        corpus_id="VFTC-002",
        slug="shifted_mask",
        title="Tamper Mode: Slightly Shifted Mask",
        description="Shift the tumour mask one column to the right.",
        scenario_family=SCENARIO_FAMILY_MASK_SEMANTICS,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_MASK,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="spatial_mask_shift",
        clinical_hazard="spatial_mislocalised_finding",
        downstream_effect="Mislocalised review attention and avoidable follow-up.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "single_series.shifted_mask",
            "fixed_inputs": {"delta_zyx": [0, 0, 1]},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"translation_voxels": [0, 0, 1]},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="proof_failure",
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=(
            "test_tamper_mode_scenarios_surface_expected_primary_failures",
        ),
        evidence_traces=(
            "manifest.mask_verification.comparison",
            "VerifiedReconstructionTrace",
        ),
    )


def _build_intensity_manipulation_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction[0:3, 0:7, 16:23] += 0.55
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-intensity"
    return _threat_entry(
        corpus_id="VFTC-003",
        slug="intensity_manipulation",
        title="Tamper Mode: Intensity Manipulation",
        description="Alter a background intensity patch while raw measurements stay fixed.",
        scenario_family=SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="local_intensity_patch_mutation",
        clinical_hazard="image_integrity_distortion",
        downstream_effect="False positive or false negative interpretation burden.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_reconstruction_integrity_controls(),
        determinism_contract={
            "builder_id": "single_series.intensity_manipulation",
            "fixed_inputs": {"z": [0, 3], "y": [0, 7], "x": [16, 23], "delta": 0.55},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"mutated_voxels": 147},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="proof_failure",
        expected_primary_reason_family=_INTENSITY_REASON_FAMILY,
        state=state,
        evidence_tests=("test_rejects_ai_reconstruction_with_invented_pixels",),
        evidence_traces=(
            "manifest.ai_reconstruction_match",
            "VerifiedReconstructionTrace",
        ),
    )


def _build_subtle_distributed_perturbation_entry(
    base_case: dict[str, Any],
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction += 9e-10
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-subtle-distributed"
    return _threat_entry(
        corpus_id="VFTC-004",
        slug="subtle_distributed_perturbation",
        title="Tamper Mode: Subtle Distributed Perturbation",
        description="Apply a low-amplitude distributed perturbation across the full image.",
        scenario_family=SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="distributed_low_amplitude_voxel_perturbation",
        clinical_hazard="image_integrity_distortion",
        downstream_effect="Boundary-hugging corruption that may look visually benign.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_reconstruction_integrity_controls(),
        determinism_contract={
            "builder_id": "single_series.subtle_distributed_perturbation",
            "fixed_inputs": {"global_delta": 9e-10},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"applied_to": "all_voxels"},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="proof_failure",
        expected_primary_reason_family=_INTENSITY_REASON_FAMILY,
        state=state,
        evidence_tests=(
            "test_rejects_distributed_low_amplitude_attack_by_global_l2_bound",
        ),
        evidence_traces=(
            "manifest.ai_reconstruction_match.metrics.global_relative_l2_error",
            "decision.success_modifiers.near_proof_threshold",
        ),
    )


def _build_structured_ringing_artefact_entry(
    base_case: dict[str, Any],
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction += _structured_ringing_delta(reconstruction.shape)
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-structured-ringing"
    return _threat_entry(
        corpus_id="VFTC-005",
        slug="structured_ringing_artefact",
        title="Tamper Mode: Structured Ringing Artefact",
        description="Inject a deterministic ringing pattern without altering raw input.",
        scenario_family=SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="ringing_pattern_injection",
        clinical_hazard="image_integrity_distortion",
        downstream_effect="Spurious haloing around anatomy can bias human review.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_reconstruction_integrity_controls(),
        determinism_contract={
            "builder_id": "single_series.structured_ringing_artefact",
            "fixed_inputs": {"amplitude": 0.22, "period_x": 3.0},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"shape": list(reconstruction.shape)},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        state=state,
        evidence_tests=("test_tamper_mode_scenarios_surface_expected_primary_failures",),
        evidence_traces=("manifest.ai_reconstruction_match",),
    )


def _build_slice_local_anomaly_injection_entry(
    base_case: dict[str, Any],
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction[1, 9:15, 9:15] += 0.85
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-slice-local-anomaly"
    return _threat_entry(
        corpus_id="VFTC-006",
        slug="slice_local_anomaly_injection",
        title="Tamper Mode: Slice-Local Anomaly Injection",
        description="Inject a localised bright anomaly into a single slice.",
        scenario_family=SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="slice_scoped_signal_injection",
        clinical_hazard="false_positive_finding",
        downstream_effect="False lesion cue on one slice can trigger avoidable escalation.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_reconstruction_integrity_controls(),
        determinism_contract={
            "builder_id": "single_series.slice_local_anomaly_injection",
            "fixed_inputs": {"slice_index": 1, "y": [9, 15], "x": [9, 15], "delta": 0.85},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"mutated_voxels": 36},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        state=state,
        evidence_tests=("test_rejects_ai_reconstruction_with_invented_pixels",),
        evidence_traces=("manifest.ai_reconstruction_match",),
    )


def _build_mask_erosion_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    state["tumor_mask"] = _erode_mask(mask).tolist()
    state["case_id"] = "mri-case-tamper-mask-erosion"
    return _threat_entry(
        corpus_id="VFTC-007",
        slug="mask_erosion",
        title="Tamper Mode: Mask Erosion",
        description="Remove a deterministic lesion face from the asserted mask.",
        scenario_family=SCENARIO_FAMILY_MASK_SEMANTICS,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_MASK,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="mask_undersegmentation_mutation",
        clinical_hazard="false_negative_finding",
        downstream_effect="Under-call risk and delayed follow-up.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "single_series.mask_erosion",
            "fixed_inputs": {"x_plane": 16},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"operation": "zero_x_plane"},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=("test_rejects_ai_mask_with_fake_tumour_voxel",),
        evidence_traces=("manifest.mask_verification.comparison",),
    )


def _build_mask_dilation_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    state["tumor_mask"] = _dilate_mask(mask).tolist()
    state["case_id"] = "mri-case-tamper-mask-dilation"
    return _threat_entry(
        corpus_id="VFTC-008",
        slug="mask_dilation",
        title="Tamper Mode: Mask Dilation",
        description="Extend the asserted lesion by a deterministic one-voxel face.",
        scenario_family=SCENARIO_FAMILY_MASK_SEMANTICS,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_MASK,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="mask_oversegmentation_mutation",
        clinical_hazard="false_positive_finding",
        downstream_effect="Over-call risk and avoidable diagnostic escalation.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "single_series.mask_dilation",
            "fixed_inputs": {"z": [1, 5], "y": [8, 16], "x": 17},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"operation": "set_face_positive"},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=("test_rejects_ai_mask_with_fake_tumour_voxel",),
        evidence_traces=("manifest.mask_verification.comparison",),
    )


def _build_volume_rounding_attack_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    state["tumor_volume_mm3"] = _round_float(float(state["tumor_volume_mm3"]) + 0.5)
    state["case_id"] = "mri-case-tamper-volume-rounding"
    return _threat_entry(
        corpus_id="VFTC-009",
        slug="volume_rounding_attack",
        title="Tamper Mode: Volume Rounding Attack",
        description="Alter only the claimed volume scalar while reconstruction and mask stay fixed.",
        scenario_family=SCENARIO_FAMILY_QUANTITATIVE_CLAIMS,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_QUANTIFICATION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="scalar_volume_rounding_mismatch",
        clinical_hazard="quantitative_misstatement",
        downstream_effect="Misleading trend framing and inappropriate follow-up planning.",
        hazard_severity=HAZARD_SEVERITY_MODERATE,
        mitigation_mapping=_quantitative_claim_controls(),
        determinism_contract={
            "builder_id": "single_series.volume_rounding_attack",
            "fixed_inputs": {"volume_delta_mm3": 0.5},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"scalar_only": True},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="proof_failure",
        expected_primary_reason_family=_QUANT_REASON_FAMILY,
        state=state,
        evidence_tests=("test_rejects_incorrect_tumor_volume_claim",),
        evidence_traces=("manifest.tumor_volume_proof", "VerifiedReconstructionTrace"),
    )


def _build_mixed_frequency_perturbation_entry(
    base_case: dict[str, Any],
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction += _mixed_frequency_delta(reconstruction.shape)
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-mixed-frequency"
    return _threat_entry(
        corpus_id="VFTC-010",
        slug="mixed_frequency_perturbation",
        title="Tamper Mode: Mixed-Frequency Perturbation",
        description="Combine low-frequency drift with a deterministic high-frequency perturbation.",
        scenario_family=SCENARIO_FAMILY_RECONSTRUCTION_INTEGRITY,
        scenario_surface=SCENARIO_SURFACE_SINGLE_SERIES_RECONSTRUCTION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="mixed_domain_frequency_perturbation",
        clinical_hazard="image_integrity_distortion",
        downstream_effect="Hybrid artefacts can escape purely local visual heuristics.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_reconstruction_integrity_controls(),
        determinism_contract={
            "builder_id": "single_series.mixed_frequency_perturbation",
            "fixed_inputs": {"low_amplitude": 0.09, "high_amplitude": 0.08},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"shape": list(reconstruction.shape)},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        state=state,
        evidence_tests=("test_rejects_distributed_low_amplitude_attack_by_global_l2_bound",),
        evidence_traces=("manifest.ai_reconstruction_match",),
    )


def _build_stale_mask_replay_entry(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    profile_name = _state_profile_name(state)
    state["tumor_mask"] = _build_replay_source_mask(profile_name).tolist()
    state["case_id"] = "mri-case-tamper-stale-mask-replay"
    return _threat_entry(
        corpus_id="VFTC-011",
        slug="stale_mask_replay",
        title="Tamper Mode: Stale Mask Replay",
        description="Reuse a syntactically valid mask from a different valid state.",
        scenario_family=SCENARIO_FAMILY_TEMPORAL_CONTEXT,
        scenario_surface=SCENARIO_SURFACE_STATE_REUSE,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="stale_mask_semantic_reuse",
        clinical_hazard="stale_context_driven_misstatement",
        downstream_effect="Out-of-date lesion state can misframe present review and follow-up.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "state_reuse.stale_mask_replay",
            "fixed_inputs": {"profile_name": profile_name},
            "seed_strategy": "no_rng",
            "replay_source_id": "single_series.shifted_replay_source",
            "coordinate_contract": {"replay_shift_zyx": [0, 0, 1]},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        provenance_scope=(
            "State-level semantic mismatch only. Not manifest tampering, bundle replay "
            "corruption, transport fraud, or provenance fraud."
        ),
        state=state,
        evidence_tests=("test_rejects_ai_mask_with_fake_tumour_voxel",),
        evidence_traces=("manifest.mask_verification.comparison",),
    )


def _build_cross_series_misregistration_entry() -> dict[str, Any]:
    state = build_multi_series_demo_case()
    support_shape = np.asarray(_series_entry(state, "t2-supporting")["ai_reconstruction"]).shape
    _rewrite_series_for_profile(
        state,
        "t2-supporting",
        _hot_voxel_reconstruction(support_shape, ((1, 1, 0),)),
    )
    state["case_id"] = "mri-study-tamper-cross-series-misregistration"
    return _threat_entry(
        corpus_id="VFTC-012",
        slug="cross_series_misregistration",
        title="Tamper Mode: Cross-Series Misregistration",
        description=(
            "Preserve admissible series metadata while shifting corroborative lesion "
            "location so geometry passes and corroboration fails."
        ),
        scenario_family=SCENARIO_FAMILY_MULTI_SERIES_CORROBORATION,
        scenario_surface=SCENARIO_SURFACE_MULTI_SERIES_CORROBORATION,
        corpus_status=THREAT_CORPUS_STATUS_ACTIVE,
        technical_failure_mechanism="cross_series_spatial_offset_without_geometry_mismatch",
        clinical_hazard="false_corroborative_confidence",
        downstream_effect="Study-level confidence can be distorted by misaligned support.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_multi_series_corroboration_controls(),
        determinism_contract={
            "builder_id": "multi_series.cross_series_misregistration",
            "fixed_inputs": {"shifted_series_id": "t2-supporting"},
            "seed_strategy": "no_rng",
            "cross_series_offset": {"series_id": "t2-supporting", "delta_zyx": [1, 1, 0]},
            "coordinate_contract": {
                "preserve_geometry_metadata": True,
                "preserve_required_series_admissibility": True,
            },
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_block_category="positive_lesion_corroboration_failed",
        expected_primary_reason_family=_CORROBORATION_REASON_FAMILY,
        state=state,
        evidence_tests=(
            "test_multi_series_primary_positive_without_any_supporting_overlap_blocks_release",
        ),
        evidence_traces=(
            "analysis.lesion_corroboration",
            "contracts.cross_series_verification_trace",
        ),
    )


def _build_component_micro_island_entry() -> dict[str, Any]:
    base_case = build_tamper_mode_base_case(profile_name="brain_t1_component_review")
    state = deepcopy(base_case)
    descriptor = get_verification_profile("brain_t1_component_review")
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    threshold_mask = (reconstruction >= descriptor.mask_threshold).astype(np.int64)
    state["tumor_mask"] = threshold_mask.tolist()
    state["case_id"] = "mri-case-tamper-component-micro-island"
    return _threat_entry(
        corpus_id="VFTC-013",
        slug="component_micro_island",
        title="Tamper Mode: Detached Micro-Island Restored",
        description="Restore a threshold-positive micro-island rejected by connected-component policy.",
        scenario_family=SCENARIO_FAMILY_PROFILE_SCOPED_POLICY,
        scenario_surface=SCENARIO_SURFACE_PROFILE_SCOPED_MASK_POLICY,
        corpus_status=THREAT_CORPUS_STATUS_PROFILE_SCOPED,
        technical_failure_mechanism="disconnected_component_restoration",
        clinical_hazard="false_positive_finding",
        downstream_effect="Detached positives can inflate lesion footprint.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "profile_scoped.component_micro_island",
            "fixed_inputs": {"profile_name": "brain_t1_component_review"},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"policy_algorithm": CONNECTED_COMPONENT_ALGORITHM},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=("test_component_profile_tamper_mode_surfaces_micro_island_mismatch",),
        evidence_traces=("contracts.mask_derivation_policy",),
    )


def _build_slice_single_slice_island_entry() -> dict[str, Any]:
    base_case = build_tamper_mode_base_case(profile_name="brain_t1_slice_review")
    state = deepcopy(base_case)
    descriptor = get_verification_profile("brain_t1_slice_review")
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    threshold_mask = (reconstruction >= descriptor.mask_threshold).astype(np.int64)
    state["tumor_mask"] = threshold_mask.tolist()
    state["case_id"] = "mri-case-tamper-slice-single-slice-island"
    return _threat_entry(
        corpus_id="VFTC-014",
        slug="slice_single_slice_island",
        title="Tamper Mode: Single-Slice Island Restored",
        description="Restore a single-slice island rejected by slice-consistency policy.",
        scenario_family=SCENARIO_FAMILY_PROFILE_SCOPED_POLICY,
        scenario_surface=SCENARIO_SURFACE_PROFILE_SCOPED_MASK_POLICY,
        corpus_status=THREAT_CORPUS_STATUS_PROFILE_SCOPED,
        technical_failure_mechanism="single_slice_component_restoration",
        clinical_hazard="false_positive_finding",
        downstream_effect="Transient single-slice positives can trigger avoidable review.",
        hazard_severity=HAZARD_SEVERITY_MAJOR,
        mitigation_mapping=_mask_semantic_controls(),
        determinism_contract={
            "builder_id": "profile_scoped.slice_single_slice_island",
            "fixed_inputs": {"profile_name": "brain_t1_slice_review"},
            "seed_strategy": "no_rng",
            "coordinate_contract": {"policy_algorithm": SLICE_CONSISTENCY_ALGORITHM},
        },
        expected_decision_class="blocked_physical",
        expected_failure_route=THREAT_ROUTE_PHYSICAL,
        expected_primary_reason_family=_MASK_REASON_FAMILY,
        state=state,
        evidence_tests=("test_slice_profile_tamper_mode_surfaces_single_slice_island_mismatch",),
        evidence_traces=("contracts.mask_derivation_policy",),
    )


def build_verification_threat_corpus(
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> dict[str, Any]:
    base_case = build_tamper_mode_base_case(profile_name=profile_name)
    entries = (
        _build_fake_tumour_entry(base_case),
        _build_shifted_mask_entry(base_case),
        _build_intensity_manipulation_entry(base_case),
        _build_subtle_distributed_perturbation_entry(base_case),
        _build_structured_ringing_artefact_entry(base_case),
        _build_slice_local_anomaly_injection_entry(base_case),
        _build_mask_erosion_entry(base_case),
        _build_mask_dilation_entry(base_case),
        _build_volume_rounding_attack_entry(base_case),
        _build_mixed_frequency_perturbation_entry(base_case),
        _build_stale_mask_replay_entry(base_case),
        _build_cross_series_misregistration_entry(),
        _build_component_micro_island_entry(),
        _build_slice_single_slice_island_entry(),
    )
    return {
        "interface_type": THREAT_CORPUS_INTERFACE_TYPE,
        "revision": THREAT_CORPUS_REVISION,
        "profile_name": profile_name,
        "hazard_severity_note": THREAT_CORPUS_SEVERITY_NOTE,
        "expectation_model": {
            "normative": THREAT_CORPUS_NORMATIVE_EXPECTATION_NOTE,
            "observational": THREAT_CORPUS_OBSERVATIONAL_EXPECTATION_NOTE,
        },
        "entries": list(entries),
    }


def build_tamper_scenarios(
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> tuple[dict[str, Any], ...]:
    descriptor = get_verification_profile(profile_name)
    corpus_entries = build_verification_threat_corpus(profile_name=profile_name)["entries"]
    selected = [entry for entry in corpus_entries if entry["slug"] in _COMPATIBILITY_TAMPER_SLUGS]
    if descriptor.mask_algorithm == CONNECTED_COMPONENT_ALGORITHM:
        selected.extend(
            entry for entry in corpus_entries if entry["slug"] == "component_micro_island"
        )
    if descriptor.mask_algorithm == SLICE_CONSISTENCY_ALGORITHM:
        selected.extend(
            entry for entry in corpus_entries if entry["slug"] == "slice_single_slice_island"
        )
    return tuple(selected)
