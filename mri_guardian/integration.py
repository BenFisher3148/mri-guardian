"""Mock DICOM and PACS envelopes around the MRI Guardian Layer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from icsd.core.serialise import hash_canonical_json
from icsd.core.transition import TransitionResult

from .harness import build_case_verification_report

_INTEGRATION_SCHEMA_VERSION = "0.1"
_STATE_FIELDS = (
    "case_id",
    "patient_pseudonym",
    "study_uid",
    "series_uid",
    "scanner_id",
    "sequence_id",
    "ai_model_id",
    "release_status",
    "raw_real",
    "raw_imag",
    "ai_reconstruction",
    "tumor_mask",
    "voxel_spacing_mm",
    "tumor_volume_mm3",
    "verified_manifest",
    "workstation_payload",
)


def build_mock_dicom_input(
    state: dict[str, Any],
    *,
    source_ae_title: str = "MRI-AI-SCANNER",
    destination_ae_title: str = "MRI-GUARDIAN-LAYER",
) -> dict[str, Any]:
    """Return a mocked DICOM ingress envelope for the Guardian Layer."""
    report = build_case_verification_report(state)
    analysis = report["analysis"]
    contracts = report["contracts"]
    claim_envelope = contracts["ai_claim_envelope"]
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "kind": "mock_dicom_input",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": {
            "case_id": state["case_id"],
            "study_uid": state["study_uid"],
            "series_uid": state["series_uid"],
            "patient_pseudonym": state["patient_pseudonym"],
        },
        "dicom_tags": {
            "Modality": "MR",
            "PatientID": state["patient_pseudonym"],
            "StudyInstanceUID": state["study_uid"],
            "SeriesInstanceUID": state["series_uid"],
            "SeriesDescription": "AI reconstruction candidate for Guardian Layer verification",
            "ProtocolName": state["sequence_id"],
            "ManufacturerModelName": state["scanner_id"],
        },
        "objects": [
            {
                "role": "raw_kspace",
                "sop_class": "MR Raw Data Storage (mock)",
                "hash": analysis["raw_hash"],
                "members": ["raw_real", "raw_imag"],
            },
            {
                "role": "ai_reconstruction",
                "sop_class": "MR Image Storage (mock)",
                "hash": claim_envelope["claimed_outputs"]["ai_reconstruction_hash"],
            },
            {
                "role": "ai_segmentation",
                "sop_class": "Segmentation Storage (mock)",
                "hash": claim_envelope["claimed_outputs"]["ai_mask_hash"],
            },
            {
                "role": "ai_volume_claim",
                "sop_class": "Comprehensive SR Storage (mock)",
                "hash": hash_canonical_json(claim_envelope["claimed_outputs"]),
            },
        ],
        "ai_claim_envelope": claim_envelope,
        "guardian_case": {field: deepcopy(state[field]) for field in _STATE_FIELDS},
    }


def restore_case_state_from_dicom_input(dicom_input: dict[str, Any]) -> dict[str, Any]:
    """Return the Guardian Layer case state embedded in a mocked DICOM envelope."""
    if dicom_input.get("kind") != "mock_dicom_input":
        raise ValueError("Expected a mocked DICOM input envelope.")
    case_state = dicom_input.get("guardian_case")
    if not isinstance(case_state, dict):
        raise ValueError("Mocked DICOM input is missing guardian_case.")
    missing_fields = [field for field in _STATE_FIELDS if field not in case_state]
    if missing_fields:
        raise ValueError(
            "Mocked DICOM input is missing required Guardian Layer fields: "
            + ", ".join(missing_fields)
        )
    return {field: deepcopy(case_state[field]) for field in _STATE_FIELDS}


def build_mock_pacs_submission(
    result: TransitionResult,
    *,
    source_ae_title: str = "MRI-GUARDIAN-LAYER",
    destination_ae_title: str = "NHS-RAD-WS-PACS",
    bundle_reference: str | None = None,
) -> dict[str, Any]:
    """Return a mocked PACS handoff envelope for an accepted case."""
    state = result.after_state
    report = build_case_verification_report(state)
    contracts = report["contracts"]
    if not report["release_eligible"]:
        raise ValueError("Mock PACS submission requires a release-eligible case.")
    workstation_payload = state["workstation_payload"]
    evidence_hash = hash_canonical_json(result.evidence.as_dict())
    manifest_hash = hash_canonical_json(state["verified_manifest"])
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "kind": "mock_pacs_submission",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": {
            "case_id": state["case_id"],
            "study_uid": state["study_uid"],
            "source_series_uid": state["series_uid"],
        },
        "submission_status": "released",
        "series": [
            {
                "role": "verified_reconstruction",
                "series_uid": f"{state['series_uid']}-guardian-reconstruction",
                "series_description": "Guardian Layer Verified Reconstruction",
                "sop_class": "MR Image Storage (mock)",
                "hash": workstation_payload["reconstruction_hash"],
            },
            {
                "role": "verified_segmentation",
                "series_uid": f"{state['series_uid']}-guardian-segmentation",
                "series_description": "Guardian Layer Verified Tumour Mask",
                "sop_class": "Segmentation Storage (mock)",
                "hash": workstation_payload["verified_mask_hash"],
            },
            {
                "role": "verified_manifest",
                "series_uid": f"{state['series_uid']}-guardian-manifest",
                "series_description": "Guardian Layer Verification Manifest",
                "sop_class": "Comprehensive SR Storage (mock)",
                "hash": manifest_hash,
            },
            {
                "role": "icsd_evidence",
                "series_uid": f"{state['series_uid']}-guardian-evidence",
                "series_description": "Guardian Layer ICSD Evidence",
                "sop_class": "Encapsulated JSON Storage (mock)",
                "hash": evidence_hash,
            },
        ],
        "verification_summary": {
            "guardian_mode": "verification_proxy",
            "decision": "released",
            "primary_reason": _primary_reason_summary(report),
            "tier_statuses": _tier_statuses(report),
            "pixel_source": workstation_payload["pixel_source"],
            "overlay_mode": workstation_payload["overlay_mode"],
            "verified_tumor_volume_mm3": workstation_payload["tumor_volume_mm3"],
            "bundle_reference": bundle_reference,
            "evidence_hash": evidence_hash,
        },
        "proof_contracts": {
            "verified_reconstruction_trace": contracts["verified_reconstruction_trace"],
            "mask_derivation_policy": contracts["mask_derivation_policy"],
        },
    }


def build_mock_pacs_block_notice(
    state: dict[str, Any],
    *,
    source_ae_title: str = "MRI-GUARDIAN-LAYER",
    destination_ae_title: str = "NHS-RAD-WS-PACS",
) -> dict[str, Any]:
    """Return a mocked PACS block notice for a rejected case."""
    report = build_case_verification_report(state)
    contracts = report["contracts"]
    if report["release_eligible"]:
        raise ValueError("Mock PACS block notice requires a rejected case.")
    analysis = report["analysis"]
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "kind": "mock_pacs_block_notice",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": {
            "case_id": state["case_id"],
            "study_uid": state["study_uid"],
            "source_series_uid": state["series_uid"],
        },
        "submission_status": "blocked",
        "blocked_reason": _primary_reason_summary(report),
        "failure_reasons": report["failure_reasons"],
        "verification_summary": {
            "guardian_mode": "verification_proxy",
            "decision": "blocked",
            "tier_statuses": _tier_statuses(report),
            "verified_tumor_volume_mm3": analysis["computed_tumor_volume_mm3"],
        },
        "proof_contracts": {
            "verified_reconstruction_trace": contracts["verified_reconstruction_trace"],
            "mask_derivation_policy": contracts["mask_derivation_policy"],
        },
        "withheld_series": [
            {
                "role": "verified_reconstruction",
                "series_description": "Guardian Layer Verified Reconstruction",
                "expected_hash": analysis["derived_reconstruction_hash"],
            },
            {
                "role": "verified_segmentation",
                "series_description": "Guardian Layer Verified Tumour Mask",
                "expected_hash": analysis["verified_mask_hash"],
            },
            {
                "role": "verification_manifest",
                "series_description": "Guardian Layer Verification Manifest",
                "expected_hash": hash_canonical_json(report["manifest"]),
            },
        ],
    }


def build_mock_integration_trace(
    *,
    dicom_input: dict[str, Any],
    pacs_output: dict[str, Any],
) -> dict[str, Any]:
    """Return a mocked boundary trace from DICOM ingress to PACS egress."""
    state = restore_case_state_from_dicom_input(dicom_input)
    report = build_case_verification_report(state)
    pacs_kind = pacs_output.get("kind")
    if pacs_kind not in {"mock_pacs_submission", "mock_pacs_block_notice"}:
        raise ValueError("Unsupported PACS output envelope.")
    egress_objects = pacs_output.get("series") or pacs_output.get("withheld_series") or []
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "kind": "mock_integration_trace",
        "case_id": state["case_id"],
        "pipeline": [
            {
                "stage": "dicom_ingress",
                "label": "DICOM Input",
                "transport": dicom_input["transport"],
                "ae_title": dicom_input["endpoint"]["source_ae_title"],
                "object_count": len(dicom_input["objects"]),
                "study_uid": state["study_uid"],
                "series_uid": state["series_uid"],
            },
            {
                "stage": "guardian_layer",
                "label": "MRI Guardian Layer",
                "mode": "verification_proxy",
                "decision": pacs_output["submission_status"],
                "primary_reason": _primary_reason_summary(report),
                "tier_statuses": _tier_statuses(report),
            },
            {
                "stage": "pacs_egress",
                "label": (
                    "PACS Output"
                    if pacs_output["submission_status"] == "released"
                    else "PACS Output Blocked"
                ),
                "transport": pacs_output["transport"],
                "ae_title": pacs_output["endpoint"]["destination_ae_title"],
                "artifact_kind": pacs_kind,
                "object_count": len(egress_objects),
            },
        ],
        "hashes": {
            "dicom_input_hash": hash_canonical_json(dicom_input),
            "pacs_output_hash": hash_canonical_json(pacs_output),
        },
    }


def _tier_statuses(report: dict[str, Any]) -> dict[str, str]:
    return {key: tier["status"] for key, tier in report["verification_tiers"].items()}


def _primary_reason_summary(report: dict[str, Any]) -> str:
    reasons = report["failure_reasons"]
    if reasons:
        return reasons[0]["summary"]
    if report["verification_tiers"]["level_3"]["status"] == "warning":
        return "all_blocking_checks_passed; clinical_plausibility_review_recommended"
    return "all_blocking_checks_passed"
