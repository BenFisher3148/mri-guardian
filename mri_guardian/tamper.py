"""Deterministic tamper-mode scenarios for the MRI guardian demo."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .contracts import build_verification_profile_contract
from .harness import _derive_verified_mask, _float_array_to_list, _round_float
from .mask_engine import CONNECTED_COMPONENT_ALGORITHM, SLICE_CONSISTENCY_ALGORITHM
from .profiles import DEFAULT_PROFILE_NAME, get_verification_profile

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


def build_tamper_scenarios(
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> tuple[dict[str, Any], ...]:
    """Return deterministic tamper scenarios with expected top-level failure reasons."""
    descriptor = get_verification_profile(profile_name)
    base_case = build_tamper_mode_base_case(profile_name=profile_name)
    scenarios: list[dict[str, Any]] = [
        _build_fake_tumour_scenario(base_case),
        _build_shifted_mask_scenario(base_case),
        _build_intensity_manipulation_scenario(base_case),
    ]
    if descriptor.mask_algorithm == CONNECTED_COMPONENT_ALGORITHM:
        scenarios.append(_build_component_micro_island_scenario(base_case, descriptor))
    if descriptor.mask_algorithm == SLICE_CONSISTENCY_ALGORITHM:
        scenarios.append(_build_slice_single_slice_island_scenario(base_case, descriptor))
    return tuple(scenarios)


def _build_fake_tumour_scenario(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    mask[0:2, 0:9, 5:24] = 1
    state["tumor_mask"] = mask.tolist()
    state["case_id"] = "mri-case-tamper-fake-tumour"
    return {
        "slug": "fake_tumour",
        "title": "Tamper Mode: Fake Tumour Injection",
        "description": (
            "Inject 342 unsupported tumour voxels into a cold background region "
            "without touching raw k-space."
        ),
        "state": state,
        "expected_primary_reason": "mask_voxel_mismatch",
        "expected_delta": 342,
    }


def _build_shifted_mask_scenario(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    mask = np.asarray(state["tumor_mask"], dtype=np.int64)
    shifted = np.zeros_like(mask)
    shifted[:, :, 1:] = mask[:, :, :-1]
    state["tumor_mask"] = shifted.tolist()
    state["case_id"] = "mri-case-tamper-shifted-mask"
    return {
        "slug": "shifted_mask",
        "title": "Tamper Mode: Slightly Shifted Mask",
        "description": (
            "Shift the tumour mask one column to the right while leaving the raw-derived "
            "reconstruction untouched."
        ),
        "state": state,
        "expected_primary_reason": "mask_voxel_mismatch",
        "expected_delta": 46,
    }


def _build_intensity_manipulation_scenario(base_case: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    reconstruction[0:3, 0:7, 16:23] += 0.55
    state["ai_reconstruction"] = _float_array_to_list(reconstruction)
    state["case_id"] = "mri-case-tamper-intensity"
    return {
        "slug": "intensity_manipulation",
        "title": "Tamper Mode: Intensity Manipulation",
        "description": (
            "Alter a background intensity patch in the AI image while leaving the raw "
            "sensor measurements unchanged."
        ),
        "state": state,
        "expected_primary_reason": "intensity_manipulation_detected",
        "expected_delta": 147,
    }


def _build_component_micro_island_scenario(
    base_case: dict[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    threshold_seed = reconstruction >= descriptor.mask_threshold
    threshold_mask = threshold_seed.astype(np.int64)
    verified_mask = np.asarray(base_case["tumor_mask"], dtype=np.int64)
    state["tumor_mask"] = threshold_mask.tolist()
    state["case_id"] = "mri-case-tamper-component-micro-island"
    expected_delta = int(np.count_nonzero(threshold_mask != verified_mask))
    return {
        "slug": "component_micro_island",
        "title": "Tamper Mode: Detached Micro-Island Restored",
        "description": (
            "Restore a detached threshold-positive micro-island that the connected-component "
            "policy deterministically rejects."
        ),
        "state": state,
        "expected_primary_reason": "mask_voxel_mismatch",
        "expected_delta": expected_delta,
    }


def _build_slice_single_slice_island_scenario(
    base_case: dict[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    state = deepcopy(base_case)
    reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
    threshold_mask = (reconstruction >= descriptor.mask_threshold).astype(np.int64)
    verified_mask = np.asarray(base_case["tumor_mask"], dtype=np.int64)
    state["tumor_mask"] = threshold_mask.tolist()
    state["case_id"] = "mri-case-tamper-slice-single-slice-island"
    expected_delta = int(np.count_nonzero(threshold_mask != verified_mask))
    return {
        "slug": "slice_single_slice_island",
        "title": "Tamper Mode: Single-Slice Island Restored",
        "description": (
            "Restore a detached single-slice threshold-positive island that the "
            "slice-consistency policy deterministically rejects."
        ),
        "state": state,
        "expected_primary_reason": "mask_voxel_mismatch",
        "expected_delta": expected_delta,
    }


def _build_tamper_reconstruction(descriptor: Any) -> np.ndarray:
    z_index, y_index, x_index = np.indices((6, 24, 24))
    reconstruction = (
        0.15 * z_index
        + 0.04 * y_index
        + 0.02 * x_index
    ).astype(np.float64)
    tumour_body = (
        ((z_index - 3.0) / 1.4) ** 2
        + ((y_index - 12.0) / 4.6) ** 2
        + ((x_index - 12.0) / 5.3) ** 2
        <= 1.0
    )
    reconstruction += tumour_body * 3.2
    if descriptor.mask_algorithm == CONNECTED_COMPONENT_ALGORITHM:
        reconstruction[0, 1:3, 20:22] = 3.4
    if descriptor.mask_algorithm == SLICE_CONSISTENCY_ALGORITHM:
        reconstruction[0, 1:3, 20:22] = 3.4
    return reconstruction
