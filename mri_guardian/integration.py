"""Mock DICOM and PACS envelopes around the MRI Guardian Layer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from icsd.core.serialise import hash_canonical_json
from icsd.core.transition import TransitionResult

from .harness import MULTI_SERIES_CASE_KIND, build_case_verification_report

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
    "verification_profile_descriptor",
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
    integration_narrative = build_guardian_integration_narrative(
        current_guardian_decision="pending_verification",
    )
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        objects: list[dict[str, Any]] = []
        for series_report in report["series_reports"]:
            claim_envelope = series_report["contracts"]["ai_claim_envelope"]
            objects.extend(
                [
                    {
                        "series_id": series_report["series_id"],
                        "series_uid": series_report["series_uid"],
                        "role": "raw_kspace",
                        "sop_class": "MR Raw Data Storage (mock)",
                        "hash": series_report["analysis"]["raw_hash"],
                        "members": ["raw_real", "raw_imag"],
                    },
                    {
                        "series_id": series_report["series_id"],
                        "series_uid": series_report["series_uid"],
                        "role": "ai_reconstruction",
                        "sop_class": "MR Image Storage (mock)",
                        "hash": claim_envelope["claimed_outputs"]["ai_reconstruction_hash"],
                    },
                    {
                        "series_id": series_report["series_id"],
                        "series_uid": series_report["series_uid"],
                        "role": "ai_segmentation",
                        "sop_class": "Segmentation Storage (mock)",
                        "hash": claim_envelope["claimed_outputs"]["ai_mask_hash"],
                    },
                    {
                        "series_id": series_report["series_id"],
                        "series_uid": series_report["series_uid"],
                        "role": "ai_volume_claim",
                        "sop_class": "Comprehensive SR Storage (mock)",
                        "hash": hash_canonical_json(claim_envelope["claimed_outputs"]),
                    },
                ]
            )
        return {
            "schema_version": _INTEGRATION_SCHEMA_VERSION,
            "kind": "mock_dicom_input",
            "transport": "dicom-c-store",
            "endpoint": {
                "source_ae_title": source_ae_title,
                "destination_ae_title": destination_ae_title,
            },
            "study": {
                "case_kind": MULTI_SERIES_CASE_KIND,
                "case_id": state["case_id"],
                "study_uid": state["study_uid"],
                "patient_pseudonym": state["patient_pseudonym"],
                "series_count": len(state["series"]),
            },
            "dicom_tags": {
                "Modality": "MR",
                "PatientID": state["patient_pseudonym"],
                "StudyInstanceUID": state["study_uid"],
                "ManufacturerModelName": state["scanner_id"],
                "NumberOfStudyRelatedSeries": len(state["series"]),
            },
            "objects": objects,
            "cross_series_policy": report["contracts"]["cross_series_verification_policy"],
            "verification_profile_descriptor": report["contracts"]["verification_profile_descriptor"],
            "integration_narrative": integration_narrative,
            "guardian_case": deepcopy(state),
        }
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
        "verification_profile_descriptor": report["profile"],
        "integration_narrative": integration_narrative,
        "guardian_case": {field: deepcopy(state[field]) for field in _STATE_FIELDS},
    }


def restore_case_state_from_dicom_input(dicom_input: dict[str, Any]) -> dict[str, Any]:
    """Return the Guardian Layer case state embedded in a mocked DICOM envelope."""
    if dicom_input.get("kind") != "mock_dicom_input":
        raise ValueError("Expected a mocked DICOM input envelope.")
    case_state = dicom_input.get("guardian_case")
    if not isinstance(case_state, dict):
        raise ValueError("Mocked DICOM input is missing guardian_case.")
    if case_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return deepcopy(case_state)
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
    if not report["release_eligible"]:
        raise ValueError("Mock PACS submission requires a release-eligible case.")
    contracts = report["contracts"]
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        workstation_payload = state["workstation_payload"]
        evidence_hash = hash_canonical_json(result.evidence.as_dict())
        manifest_hash = hash_canonical_json(state["verified_manifest"])
        integration_narrative = build_guardian_integration_narrative(
            current_guardian_decision="released",
        )
        released_series = []
        for series_payload in workstation_payload["series_payloads"]:
            payload = series_payload["workstation_payload"]
            released_series.extend(
                [
                    {
                        "series_id": series_payload["series_id"],
                        "role": "verified_reconstruction",
                        "series_uid": f"{series_payload['series_uid']}-guardian-reconstruction",
                        "series_description": "Guardian Layer Verified Reconstruction",
                        "sop_class": "MR Image Storage (mock)",
                        "hash": payload["reconstruction_hash"],
                    },
                    {
                        "series_id": series_payload["series_id"],
                        "role": "verified_segmentation",
                        "series_uid": f"{series_payload['series_uid']}-guardian-segmentation",
                        "series_description": "Guardian Layer Verified Tumour Mask",
                        "sop_class": "Segmentation Storage (mock)",
                        "hash": payload["verified_mask_hash"],
                    },
                ]
            )
        released_series.extend(
            [
                {
                    "role": "verified_manifest",
                    "series_uid": f"{state['study_uid']}-guardian-study-manifest",
                    "series_description": "Guardian Layer Study Verification Manifest",
                    "sop_class": "Comprehensive SR Storage (mock)",
                    "hash": manifest_hash,
                },
                {
                    "role": "icsd_evidence",
                    "series_uid": f"{state['study_uid']}-guardian-study-evidence",
                    "series_description": "Guardian Layer Study ICSD Evidence",
                    "sop_class": "Encapsulated JSON Storage (mock)",
                    "hash": evidence_hash,
                },
            ]
        )
        return {
            "schema_version": _INTEGRATION_SCHEMA_VERSION,
            "kind": "mock_pacs_submission",
            "transport": "dicom-c-store",
            "endpoint": {
                "source_ae_title": source_ae_title,
                "destination_ae_title": destination_ae_title,
            },
            "study": {
                "case_kind": MULTI_SERIES_CASE_KIND,
                "case_id": state["case_id"],
                "study_uid": state["study_uid"],
                "source_series_ids": [item["series_id"] for item in report["series_reports"]],
            },
            "submission_status": "released",
            "series": released_series,
            "verification_summary": {
                "guardian_mode": "verification_proxy",
                "case_kind": MULTI_SERIES_CASE_KIND,
                "decision": "released",
                "decision_class": report["decision_summary"]["decision_class"],
                "primary_reason": _primary_reason_summary(report),
                "tier_statuses": _tier_statuses(report),
                "proof_status": report["decision_summary"]["proof_status"],
                "profile_policy_status": report["decision_summary"]["profile_policy_status"],
                "released_series_ids": report["analysis"]["released_series_ids"],
                "withheld_series_ids": report["analysis"]["withheld_series_ids"],
                "bundle_reference": bundle_reference,
                "evidence_hash": evidence_hash,
                "verification_profile_descriptor_hash": hash_canonical_json(
                    contracts["verification_profile_descriptor"]
                ),
            },
            "integration_narrative": integration_narrative,
            "proof_contracts": {
                "verification_profile_descriptor": contracts["verification_profile_descriptor"],
                "cross_series_verification_policy": contracts["cross_series_verification_policy"],
                "cross_series_verification_trace": contracts["cross_series_verification_trace"],
            },
        }
    workstation_payload = state["workstation_payload"]
    evidence_hash = hash_canonical_json(result.evidence.as_dict())
    manifest_hash = hash_canonical_json(state["verified_manifest"])
    integration_narrative = build_guardian_integration_narrative(
        current_guardian_decision="released",
    )
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
            "decision_class": report["decision_summary"]["decision_class"],
            "primary_reason": _primary_reason_summary(report),
            "tier_statuses": _tier_statuses(report),
            "proof_status": report["decision_summary"]["proof_status"],
            "profile_policy_status": report["decision_summary"]["profile_policy_status"],
            "pixel_source": workstation_payload["pixel_source"],
            "overlay_mode": workstation_payload["overlay_mode"],
            "verified_tumor_volume_mm3": workstation_payload["tumor_volume_mm3"],
            "bundle_reference": bundle_reference,
            "evidence_hash": evidence_hash,
            "profile_name": report["manifest"]["profile_name"],
            "profile_digest": report["manifest"]["profile_digest"],
        },
        "integration_narrative": integration_narrative,
        "proof_contracts": {
            "verification_profile_descriptor": contracts["verification_profile_descriptor"],
            "verified_reconstruction_trace": contracts["verified_reconstruction_trace"],
            "mask_derivation_policy": contracts["mask_derivation_policy"],
            "verification_tolerance_policy": contracts["verification_tolerance_policy"],
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
    integration_narrative = build_guardian_integration_narrative(
        current_guardian_decision="blocked",
    )
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return {
            "schema_version": _INTEGRATION_SCHEMA_VERSION,
            "kind": "mock_pacs_block_notice",
            "transport": "dicom-c-store",
            "endpoint": {
                "source_ae_title": source_ae_title,
                "destination_ae_title": destination_ae_title,
            },
            "study": {
                "case_kind": MULTI_SERIES_CASE_KIND,
                "case_id": state["case_id"],
                "study_uid": state["study_uid"],
                "source_series_ids": [item["series_id"] for item in report["series_reports"]],
            },
            "submission_status": "blocked",
            "blocked_reason": _primary_reason_summary(report),
            "failure_reasons": report["failure_reasons"],
            "verification_summary": {
                "guardian_mode": "verification_proxy",
                "case_kind": MULTI_SERIES_CASE_KIND,
                "decision": "blocked",
                "decision_class": report["decision_summary"]["decision_class"],
                "tier_statuses": _tier_statuses(report),
                "proof_status": report["decision_summary"]["proof_status"],
                "profile_policy_status": report["decision_summary"]["profile_policy_status"],
                "released_series_ids": report["analysis"]["released_series_ids"],
                "withheld_series_ids": report["analysis"]["withheld_series_ids"],
                "verification_profile_descriptor_hash": hash_canonical_json(
                    contracts["verification_profile_descriptor"]
                ),
            },
            "integration_narrative": integration_narrative,
            "proof_contracts": {
                "verification_profile_descriptor": contracts["verification_profile_descriptor"],
                "cross_series_verification_policy": contracts["cross_series_verification_policy"],
                "cross_series_verification_trace": contracts["cross_series_verification_trace"],
            },
            "withheld_series": [
                {
                    "series_id": item["series_id"],
                    "resolved_role": item["resolved_role"],
                    "gating_status": item["gating_status"],
                    "series_proof_status": item["series_proof_status"],
                    "downstream_release_allowed": item["downstream_release_allowed"],
                }
                for item in report["analysis"]["series"]
            ],
        }
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
            "decision_class": report["decision_summary"]["decision_class"],
            "tier_statuses": _tier_statuses(report),
            "proof_status": report["decision_summary"]["proof_status"],
            "profile_policy_status": report["decision_summary"]["profile_policy_status"],
            "verified_tumor_volume_mm3": analysis["computed_tumor_volume_mm3"],
            "profile_name": report["manifest"]["profile_name"],
            "profile_digest": report["manifest"]["profile_digest"],
        },
        "integration_narrative": integration_narrative,
        "proof_contracts": {
            "verification_profile_descriptor": contracts["verification_profile_descriptor"],
            "verified_reconstruction_trace": contracts["verified_reconstruction_trace"],
            "mask_derivation_policy": contracts["mask_derivation_policy"],
            "verification_tolerance_policy": contracts["verification_tolerance_policy"],
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
    integration_narrative = build_guardian_integration_narrative(
        current_guardian_decision=str(pacs_output["submission_status"]),
    )
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return {
            "schema_version": _INTEGRATION_SCHEMA_VERSION,
            "kind": "mock_integration_trace",
            "case_id": state["case_id"],
            "integration_narrative": integration_narrative,
            "pipeline": [
                {
                    "stage": "dicom_ingress",
                    "label": "DICOM Input",
                    "transport": dicom_input["transport"],
                    "ae_title": dicom_input["endpoint"]["source_ae_title"],
                    "object_count": len(dicom_input["objects"]),
                    "study_uid": state["study_uid"],
                    "series_count": len(state["series"]),
                },
                {
                    "stage": "guardian_layer",
                    "label": "MRI Guardian Layer",
                    "mode": "verification_proxy",
                    "case_kind": MULTI_SERIES_CASE_KIND,
                    "decision": pacs_output["submission_status"],
                    "decision_class": report["decision_summary"]["decision_class"],
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
    return {
        "schema_version": _INTEGRATION_SCHEMA_VERSION,
        "kind": "mock_integration_trace",
        "case_id": state["case_id"],
        "integration_narrative": integration_narrative,
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
                "decision_class": report["decision_summary"]["decision_class"],
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


def build_guardian_integration_narrative(
    *,
    current_guardian_decision: str,
) -> dict[str, Any]:
    """Return a deterministic hospital-integration narrative for the Guardian Layer."""
    return {
        "interface_type": "GuardianIntegrationNarrative",
        "interface_version": _INTEGRATION_SCHEMA_VERSION,
        "guardian_role": "verification_proxy",
        "dicom_compliance_story": {
            "ingress_transport": "dicom-c-store",
            "ingress_objects": [
                "MR Raw Data Storage (mock)",
                "MR Image Storage (mock)",
                "Segmentation Storage (mock)",
                "Comprehensive SR Storage (mock)",
            ],
            "egress_transport": "dicom-c-store",
            "egress_objects": [
                "MR Image Storage (mock)",
                "Segmentation Storage (mock)",
                "Comprehensive SR Storage (mock)",
                "Encapsulated JSON Storage (mock)",
            ],
            "story": (
                "The Guardian Layer receives DICOM-wrapped MRI evidence and AI candidate "
                "objects, preserves study and patient context, and emits only verified "
                "derived series back onto the hospital imaging network."
            ),
        },
        "hospital_placement": {
            "position": "between_reconstruction_ai_and_pacs",
            "upstream_systems": [
                "MRI scanner",
                "Raw MRI reconstruction AI",
            ],
            "guardian_layer": "verification proxy AE on the imaging network",
            "downstream_systems": [
                "PACS or VNA",
                "radiologist workstation",
            ],
            "story": (
                "This sits after the reconstruction AI and before PACS/workstation import, "
                "so it behaves like a non-invasive gateway rather than a new clinical system."
            ),
        },
        "failure_mode": {
            "mode": "fail_closed",
            "decision_scope": "guardian_verified_handoff",
            "current_guardian_decision": current_guardian_decision,
            "release_rule": (
                "Verified series are released only when the blocking Guardian Layer proof "
                "passes and the selected runtime profile allows final release."
            ),
            "on_verification_failure": (
                "No verified reconstruction, segmentation, manifest, or evidence series are "
                "forwarded to PACS/workstation from the Guardian path."
            ),
            "on_service_unavailable": (
                "The Guardian path does not auto-pass AI output through. Cases remain pending "
                "or require an explicit external fallback workflow outside this proxy."
            ),
        },
    }


def _tier_statuses(report: dict[str, Any]) -> dict[str, str]:
    return {key: tier["status"] for key, tier in report["verification_tiers"].items()}


def _primary_reason_summary(report: dict[str, Any]) -> str:
    reasons = report["failure_reasons"]
    if reasons:
        return reasons[0]["summary"]
    if report["decision_summary"]["decision_class"] == "triage_only":
        return "proof_passed_but_profile_is_triage_only"
    if report["decision_summary"]["operator_review_required"]:
        return "all_blocking_checks_passed; clinical_plausibility_review_recommended"
    return "all_blocking_checks_passed"
