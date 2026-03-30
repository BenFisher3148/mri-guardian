"""Portable MRI Guardian Layer bundle builder and deterministic replay verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from icsd.core.evidence import (
    _transparency_log_payload_from_mapping,
    validate_evidence_payload,
)
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json
from icsd.core.transition import TransitionResult

from .harness import (
    MULTI_SERIES_CASE_KIND,
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    _check_released_state_matches_trace,
    build_case_verification_report,
)
from .profiles import (
    get_verification_profile,
    verification_profile_descriptor_from_dict,
)

MRI_GUARD_BUNDLE_EXTENSION = ".mri-guard-bundle"
_BUNDLE_SCHEMA_VERSION = "0.3"
_BUNDLE_MEDIA_TYPE = "application/vnd.mri-guard.bundle+zip"
_FIXED_ZIP_TIMESTAMP = (2026, 3, 26, 0, 0, 0)
_BUNDLE_MANIFEST_NAME = "bundle.json"
_SIGNATURES_NAME = "signatures.json"
_EVIDENCE_NAME = "evidence.json"
_INPUT_STATE_NAME = "input_state.json"
_RELEASED_STATE_NAME = "released_state.json"
_WORKSTATION_PAYLOAD_NAME = "workstation_payload.json"
_VERIFICATION_REPORT_NAME = "verification_report.json"
_PROFILE_DESCRIPTOR_NAME = "verification_profile_descriptor.json"
_TRANSCRIPT_NAME = "transcript.json"
_REPLAY_DESCRIPTOR_NAME = "replay.json"
_TRANSPARENCY_LOG_NAME = "transparency-log.jsonl"
_PRIOR_VERIFIED_CONTEXTS_NAME = "prior_verified_contexts.json"
_PRIOR_VERIFIED_CONTEXT_NAME = "prior_verified_context.json"
_LONGITUDINAL_SELECTION_TRACE_NAME = "longitudinal_selection_trace.json"
_TEMPORAL_TRAJECTORY_TRACE_NAME = "temporal_trajectory_trace.json"
_LONGITUDINAL_REGISTRATION_TRACE_NAME = "longitudinal_registration_trace.json"
_TEMPORAL_CONSISTENCY_TRACE_NAME = "temporal_consistency_trace.json"
_STUDY_TEMPORAL_AGGREGATION_TRACE_NAME = "study_temporal_aggregation_trace.json"
_REQUIRED_BUNDLE_MEMBERS = (
    _BUNDLE_MANIFEST_NAME,
    _SIGNATURES_NAME,
    _EVIDENCE_NAME,
    _INPUT_STATE_NAME,
    _RELEASED_STATE_NAME,
    _WORKSTATION_PAYLOAD_NAME,
    _VERIFICATION_REPORT_NAME,
    _PROFILE_DESCRIPTOR_NAME,
    _TRANSCRIPT_NAME,
    _REPLAY_DESCRIPTOR_NAME,
    _TRANSPARENCY_LOG_NAME,
)
_OPTIONAL_BUNDLE_MEMBERS = (
    _PRIOR_VERIFIED_CONTEXTS_NAME,
    _PRIOR_VERIFIED_CONTEXT_NAME,
    _LONGITUDINAL_SELECTION_TRACE_NAME,
    _TEMPORAL_TRAJECTORY_TRACE_NAME,
    _LONGITUDINAL_REGISTRATION_TRACE_NAME,
    _TEMPORAL_CONSISTENCY_TRACE_NAME,
    _STUDY_TEMPORAL_AGGREGATION_TRACE_NAME,
)


def _pipeline_phase_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value["status"])
        for key, value in report["verification_pipeline"]["phases"].items()
    }


def _proof_stage_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    proof_core = report["verification_pipeline"]["phases"].get("proof_core", {})
    return {
        str(item["proof_stage"]): str(item["status"])
        for item in proof_core.get("proof_stages", [])
    }


def write_portable_evidence_bundle(
    *,
    result: TransitionResult,
    transparency_log_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Write a deterministic .mri-guard-bundle archive and return its metadata."""
    before_state = result.before_state
    state = result.after_state
    _ensure_released_state(state)
    evidence_payload = result.evidence.as_dict()
    evidence_failures = validate_evidence_payload(evidence_payload)
    if evidence_failures:
        msg = f"ICSD evidence payload is invalid: {evidence_failures!r}"
        raise ValueError(msg)
    anchored_evidence_payload = result.evidence.transparency_log_payload()
    report = build_case_verification_report(state)
    if not report["release_eligible"]:
        raise ValueError("Portable bundles can only be written for release-eligible cases.")
    if (
        state.get("case_kind") != MULTI_SERIES_CASE_KIND
        and not report["profile"]["release_semantics"]["portable_bundle_allowed"]
    ):
        raise ValueError(
            "Portable bundles are not allowed for the selected runtime profile."
        )

    log_snapshot = _read_transparency_log_snapshot(transparency_log_path)
    receipt_entry = _resolve_receipt_entry(
        evidence_payload=evidence_payload,
        anchored_evidence_payload=anchored_evidence_payload,
        log_entries=log_snapshot["entries"],
    )
    transcript = _build_bundle_transcript(
        state=state,
        report=report,
        evidence_payload=evidence_payload,
        receipt_entry=receipt_entry,
        checkpoint=log_snapshot["checkpoint"],
    )
    replay_descriptor = _build_replay_descriptor(
        before_state=before_state,
        state=state,
        report=report,
        evidence_payload=evidence_payload,
        anchored_evidence_payload=anchored_evidence_payload,
        log_snapshot=log_snapshot,
        receipt_entry=receipt_entry,
    )
    bundle_payload = _build_bundle_manifest(
        state=state,
        report=report,
        evidence_payload=evidence_payload,
        anchored_evidence_payload=anchored_evidence_payload,
        transcript=transcript,
        replay_descriptor=replay_descriptor,
        log_snapshot=log_snapshot,
        receipt_entry=receipt_entry,
    )
    members = {
        _BUNDLE_MANIFEST_NAME: _json_bytes(bundle_payload),
        _EVIDENCE_NAME: _json_bytes(evidence_payload),
        _INPUT_STATE_NAME: _json_bytes(before_state),
        _RELEASED_STATE_NAME: _json_bytes(state),
        _WORKSTATION_PAYLOAD_NAME: _json_bytes(state["workstation_payload"]),
        _VERIFICATION_REPORT_NAME: _json_bytes(report),
        _PROFILE_DESCRIPTOR_NAME: _json_bytes(report["contracts"]["verification_profile_descriptor"]),
        _TRANSCRIPT_NAME: _json_bytes(transcript),
        _REPLAY_DESCRIPTOR_NAME: _json_bytes(replay_descriptor),
        _TRANSPARENCY_LOG_NAME: log_snapshot["bytes"],
    }
    members.update(_build_optional_bundle_members(report))
    signatures = _build_signature_catalog(members)
    members[_SIGNATURES_NAME] = _json_bytes(signatures)

    bundle_path = Path(output_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, mode="w") as archive:
        for member_name in _ordered_bundle_member_names(members):
            _write_bundle_member(archive, member_name, members[member_name])

    return {
        "path": str(bundle_path),
        "bundle": bundle_payload,
        "replay": replay_descriptor,
        "signatures": signatures,
    }


def verify_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Verify a bundle archive or an extracted bundle.json replay package."""
    return verify_portable_evidence_bundle(bundle_path)


def verify_portable_evidence_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Verify a bundle and deterministically replay its release proof."""
    path = Path(bundle_path)
    failures: list[str] = []
    replay_verification = {
        "verified": False,
        "failures": ["deterministic replay was not attempted"],
        "details": {},
    }
    source_kind = "unknown"
    try:
        source = _load_bundle_source(path)
        source_kind = source["source_kind"]
        payloads = source["payloads"]
        member_bytes = source["member_bytes"]
        bundle_bytes = source["bundle_bytes"]

        bundle_payload = payloads[_BUNDLE_MANIFEST_NAME]
        evidence_payload = payloads[_EVIDENCE_NAME]
        input_state = payloads[_INPUT_STATE_NAME]
        state = payloads[_RELEASED_STATE_NAME]
        report = payloads[_VERIFICATION_REPORT_NAME]
        profile_descriptor = payloads[_PROFILE_DESCRIPTOR_NAME]
        transcript = payloads[_TRANSCRIPT_NAME]
        replay_descriptor = payloads[_REPLAY_DESCRIPTOR_NAME]
        signatures = payloads[_SIGNATURES_NAME]
        anchored_evidence_payload = _transparency_log_payload_from_mapping(evidence_payload)
        if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
            descriptor_failure = _validate_multi_series_profile_descriptor_set(profile_descriptor)
            if descriptor_failure is not None:
                failures.append(descriptor_failure)
        else:
            try:
                verification_profile_descriptor_from_dict(profile_descriptor)
            except ValueError as exc:
                failures.append(f"verification_profile_descriptor.json is invalid: {exc}")

        evidence_failures = validate_evidence_payload(evidence_payload)
        if evidence_failures:
            failures.append(f"evidence.json failed ICSD validation: {evidence_failures!r}")

        try:
            _ensure_released_state(state)
        except ValueError as exc:
            failures.append(str(exc))

        expected_report = build_case_verification_report(state)
        if report != expected_report:
            failures.append("verification_report.json does not match the released state's math trace.")
        if profile_descriptor != expected_report["contracts"]["verification_profile_descriptor"]:
            failures.append(
                "verification_profile_descriptor.json does not match the released state's runtime profile."
            )
        if not expected_report["release_eligible"]:
            failures.append("released_state.json is not release-eligible.")

        expected_optional_members = _build_optional_bundle_members(expected_report)
        for member_name in expected_optional_members:
            if member_name not in member_bytes:
                failures.append(f"Bundle is missing expected optional member {member_name!r}.")

        trace_failures = _check_released_state_matches_trace(state)
        if trace_failures:
            failures.extend(failure.message for failure in trace_failures)

        log_snapshot = {
            "bytes": member_bytes[_TRANSPARENCY_LOG_NAME],
            "entries": _parse_transparency_entries(member_bytes[_TRANSPARENCY_LOG_NAME]),
            "checkpoint": None,
        }
        chain_error = _verify_transparency_entries(log_snapshot["entries"])
        if chain_error is not None:
            failures.append(chain_error)
        else:
            log_snapshot["checkpoint"] = (
                log_snapshot["entries"][-1]["entry_hash"] if log_snapshot["entries"] else None
            )

        receipt_entry = None
        try:
            receipt_entry = _resolve_receipt_entry(
                evidence_payload=evidence_payload,
                anchored_evidence_payload=anchored_evidence_payload,
                log_entries=log_snapshot["entries"],
            )
        except ValueError as exc:
            failures.append(str(exc))

        if receipt_entry is not None:
            expected_transcript = _build_bundle_transcript(
                state=state,
                report=expected_report,
                evidence_payload=evidence_payload,
                receipt_entry=receipt_entry,
                checkpoint=log_snapshot["checkpoint"],
            )
            if transcript != expected_transcript:
                failures.append(
                    "transcript.json does not match the deterministic verification transcript."
                )
            expected_bundle = _build_bundle_manifest(
                state=state,
                report=expected_report,
                evidence_payload=evidence_payload,
                anchored_evidence_payload=anchored_evidence_payload,
                transcript=expected_transcript,
                replay_descriptor=replay_descriptor,
                log_snapshot=log_snapshot,
                receipt_entry=receipt_entry,
            )
            if bundle_payload != expected_bundle:
                failures.append("bundle.json does not match the deterministic bundle manifest.")

            replay_verification = _verify_deterministic_replay(
                input_state=input_state,
                expected_state=state,
                expected_report=expected_report,
                expected_evidence_payload=evidence_payload,
                expected_profile_descriptor=profile_descriptor,
                expected_member_bytes=member_bytes,
                expected_bundle_bytes=bundle_bytes,
                replay_descriptor=replay_descriptor,
                log_snapshot=log_snapshot,
                receipt_entry=receipt_entry,
                scratch_root=path.parent,
            )
            failures.extend(replay_verification["failures"])

        expected_signatures = _build_signature_catalog(
            {
                name: member_bytes[name]
                for name in _ordered_bundle_member_names(member_bytes)
                if name != _SIGNATURES_NAME
            }
        )
        if signatures != expected_signatures:
            failures.append("signatures.json does not match the bundled member contents.")
    except Exception as exc:
        return {
            "verified": False,
            "bundle_path": str(path),
            "failures": [str(exc)],
            "details": {"source_kind": source_kind},
        }

    return {
        "verified": not failures,
        "bundle_path": str(path),
        "failures": failures,
        "details": {
            "case_id": state.get("case_id"),
            "bundle_type": bundle_payload.get("bundle_type"),
            "receipt_reference": bundle_payload.get("transparency", {}).get("receipt_reference"),
            "evidence_hash": bundle_payload.get("hashes", {}).get("evidence_hash"),
            "replay_verified": replay_verification["verified"],
            "replayed_bundle_sha256": replay_verification["details"].get(
                "replayed_bundle_sha256"
            ),
            "source_kind": source_kind,
        },
    }


def _ensure_released_state(state: Mapping[str, Any]) -> None:
    if state.get("release_status") != RELEASE_STATUS_RELEASED:
        raise ValueError("Portable bundles require a released MRI guardian state.")


def _build_bundle_manifest(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    transcript: Mapping[str, Any],
    replay_descriptor: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_bundle_manifest(
            state=state,
            report=report,
            evidence_payload=evidence_payload,
            anchored_evidence_payload=anchored_evidence_payload,
            transcript=transcript,
            replay_descriptor=replay_descriptor,
            log_snapshot=log_snapshot,
            receipt_entry=receipt_entry,
        )
    analysis = report["analysis"]
    contracts = report["contracts"]
    members = list(_REQUIRED_BUNDLE_MEMBERS)
    if (
        contracts.get("prior_verified_context") is not None
        or contracts.get("prior_verified_contexts") is not None
    ):
        members.extend(_OPTIONAL_BUNDLE_MEMBERS)
    hashes = {
        "verification_profile_descriptor_hash": hash_canonical_json(
            contracts["verification_profile_descriptor"]
        ),
        "raw_kspace_hash": analysis["raw_hash"],
        "ai_claim_envelope_hash": hash_canonical_json(contracts["ai_claim_envelope"]),
        "mask_derivation_policy_hash": hash_canonical_json(
            contracts["mask_derivation_policy"]
        ),
        "verification_tolerance_policy_hash": hash_canonical_json(
            contracts["verification_tolerance_policy"]
        ),
        "proof_stage_hashes": dict(contracts["proof_core_evidence_trace"]["proof_stage_hashes"]),
        "proof_core_evidence_trace_hash": hash_canonical_json(
            contracts["proof_core_evidence_trace"]
        ),
        "temporal_consistency_trace_hash": hash_canonical_json(
            contracts["temporal_consistency_trace"]
        ),
        "reconstruction_hash": analysis["derived_reconstruction_hash"],
        "ai_mask_hash": analysis["ai_mask_hash"],
        "verified_mask_hash": analysis["verified_mask_hash"],
        "input_state_hash": replay_descriptor["expected_hashes"]["input_state_hash"],
        "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
        "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
        "released_state_hash": hash_canonical_json(state),
        "evidence_hash": hash_canonical_json(evidence_payload),
        "anchored_evidence_hash": hash_canonical_json(anchored_evidence_payload),
        "verification_report_hash": hash_canonical_json(report),
        "transcript_hash": hash_canonical_json(transcript),
        "transparency_log_hash": _sha256_hex(log_snapshot["bytes"]),
        "transparency_prefix_hash": replay_descriptor["expected_hashes"][
            "transparency_prefix_hash"
        ],
    }
    prior_verified_context = contracts.get("prior_verified_context")
    prior_verified_contexts = contracts.get("prior_verified_contexts")
    if prior_verified_contexts is not None:
        hashes["prior_verified_contexts_hash"] = hash_canonical_json(prior_verified_contexts)
    if prior_verified_context is not None:
        hashes["prior_verified_context_hash"] = hash_canonical_json(prior_verified_context)
    for key, hash_name in (
        ("longitudinal_selection_trace", "longitudinal_selection_trace_hash"),
        ("temporal_trajectory_trace", "temporal_trajectory_trace_hash"),
        ("longitudinal_registration_trace", "longitudinal_registration_trace_hash"),
    ):
        if contracts.get(key) is not None:
            hashes[hash_name] = hash_canonical_json(contracts[key])
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "bundle_type": "mri_guard_bundle",
        "bundle_media_type": _BUNDLE_MEDIA_TYPE,
        "case_id": state["case_id"],
        "entity_type": evidence_payload["entity_type"],
        "entity_id": evidence_payload["entity_id"],
        "generated_at": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "profile_name": report["profile"]["profile_name"],
        "profile_revision": report["profile"]["profile_revision"],
        "profile_digest": report["profile"]["profile_digest"],
        "icsd_policy_id": report["profile"]["icsd_policy_id"],
        "icsd_policy_version": report["profile"]["icsd_policy_version"],
        "verdict": "This scan is verifiably correct.",
        "hashes": hashes,
        "verification": {
            "release_eligible": bool(report["release_eligible"]),
            "decision": report["decision"],
            "decision_summary": report["decision_summary"],
            "failure_reasons": report["failure_reasons"],
            "pipeline_phase_statuses": _pipeline_phase_statuses(report),
            "proof_stage_statuses": _proof_stage_statuses(report),
        },
        "deterministic_replay": {
            "supported": True,
            "mode": replay_descriptor["replay_mode"],
            "input_member": _INPUT_STATE_NAME,
            "replay_member": _REPLAY_DESCRIPTOR_NAME,
            "profile_descriptor_member": _PROFILE_DESCRIPTOR_NAME,
            "transition_id": replay_descriptor["transition_id"],
        },
        "transparency": {
            "receipt_reference": _transparency_receipt(evidence_payload)["reference"],
            "entry_hash": receipt_entry["entry_hash"],
            "previous_entry_hash": receipt_entry["previous_entry_hash"],
            "checkpoint": log_snapshot["checkpoint"],
            "entry_index": receipt_entry["index"],
            "log_length": len(log_snapshot["entries"]),
        },
        "members": members,
        "signatures_member": _SIGNATURES_NAME,
    }


def _build_bundle_transcript(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
    checkpoint: str | None,
 ) -> dict[str, Any]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_bundle_transcript(
            state=state,
            report=report,
            evidence_payload=evidence_payload,
            receipt_entry=receipt_entry,
            checkpoint=checkpoint,
        )
    analysis = report["analysis"]
    manifest = report["manifest"]
    verification_pipeline = report["verification_pipeline"]
    proof_core = verification_pipeline["phases"]["proof_core"]
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "transcript_type": "portable_mri_guard_transcript",
        "case_id": state["case_id"],
        "timestamp": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "statement": "This scan is verifiably correct.",
        "events": [
            {
                "stage": "profile",
                "summary": "Resolved the Guardian runtime profile and fixed its canonical digest.",
                "details": {
                    "profile_name": report["profile"]["profile_name"],
                    "profile_revision": report["profile"]["profile_revision"],
                    "profile_digest": report["profile"]["profile_digest"],
                    "icsd_policy_id": report["profile"]["icsd_policy_id"],
                    "icsd_policy_version": report["profile"]["icsd_policy_version"],
                },
            },
            {
                "stage": "capture",
                "summary": "Captured the raw MRI measurements and fixed their canonical hash.",
                "details": {
                    "raw_kspace_hash": analysis["raw_hash"],
                    "voxel_spacing_mm": state["voxel_spacing_mm"],
                },
            },
            {
                "stage": "measurement_integrity_proof",
                "summary": (
                    "Materialized the measurement-domain proof that anchors the derived "
                    "reconstruction to the supplied raw measurements within policy."
                ),
                "details": {
                    "reconstruction_hash": analysis["derived_reconstruction_hash"],
                    "forward_projection_hash": analysis["forward_projection_hash"],
                    "max_measurement_abs_error": analysis["max_measurement_abs_error"],
                    "relative_measurement_l2_error": analysis["relative_measurement_l2_error"],
                    "max_imaginary_leakage": analysis["max_imaginary_leakage"],
                },
            },
            {
                "stage": "reconstruction_parity_proof",
                "summary": (
                    "Materialized the reconstruction-parity proof between the AI image "
                    "and the raw-derived reconstruction."
                ),
                "details": {
                    "ai_reconstruction_hash": analysis["ai_reconstruction_hash"],
                    "derived_reconstruction_hash": analysis["derived_reconstruction_hash"],
                    "max_ai_abs_error": analysis["max_ai_abs_error"],
                    "relative_ai_l2_error": analysis["relative_ai_l2_error"],
                },
            },
            {
                "stage": "segmentation_policy_proof",
                "summary": (
                    "Materialized the segmentation-policy proof between the AI mask and "
                    "the policy-derived verified mask."
                ),
                "details": {
                    "ai_mask_hash": analysis["ai_mask_hash"],
                    "verified_mask_hash": analysis["verified_mask_hash"],
                    "mask_voxel_delta": analysis["mask_voxel_delta"],
                },
            },
            {
                "stage": "volume_computation_proof",
                "summary": (
                    "Materialized the volume-computation proof from the verified mask, "
                    "voxel geometry, and claimed scalar quantity."
                ),
                "details": {
                    "claimed_tumor_volume_mm3": manifest["volume_computation_proof"][
                        "claimed_mm3"
                    ],
                    "verified_tumor_volume_mm3": manifest["volume_computation_proof"][
                        "computed_mm3"
                    ],
                    "verified_mask_voxel_count": manifest["volume_computation_proof"][
                        "voxel_count"
                    ],
                    "voxel_volume_mm3": manifest["volume_computation_proof"][
                        "voxel_volume_mm3"
                    ],
                },
            },
            {
                "stage": "proof_core_evidence_trace",
                "summary": (
                    "Aggregated the already-materialized proof-stage artefacts into the "
                    "strictly derived proof-core evidence trace."
                ),
                "details": dict(report["contracts"]["proof_core_evidence_trace"]),
            },
            {
                "stage": "verification_pipeline",
                "summary": (
                    "Recorded the canonical assurance-flow schema and evidence-ordering "
                    "model for this case."
                ),
                "details": {
                    "ordering_semantics": verification_pipeline["ordering_semantics"],
                    "pipeline_phase_statuses": _pipeline_phase_statuses(report),
                    "proof_stage_statuses": _proof_stage_statuses(report),
                    "proof_core_status": proof_core["status"],
                },
            },
            {
                "stage": "release",
                "summary": "Passed only the raw-derived reconstruction and verified mask through the verification layer to the PACS/workstation boundary.",
                "details": {
                    "release_status": state["release_status"],
                    "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
                    "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
                },
            },
            {
                "stage": "transparency",
                "summary": "Anchored the ICSD evidence envelope into the append-only transparency hash chain.",
                "details": {
                    "receipt_reference": _transparency_receipt(evidence_payload)["reference"],
                    "entry_hash": receipt_entry["entry_hash"],
                    "previous_entry_hash": receipt_entry["previous_entry_hash"],
                    "checkpoint": checkpoint,
                },
            },
        ],
    }


def _build_replay_descriptor(
    *,
    before_state: Mapping[str, Any],
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_replay_descriptor(
            before_state=before_state,
            state=state,
            report=report,
            evidence_payload=evidence_payload,
            anchored_evidence_payload=anchored_evidence_payload,
            log_snapshot=log_snapshot,
            receipt_entry=receipt_entry,
        )
    prefix_entries = log_snapshot["entries"][: receipt_entry["index"]]
    prefix_bytes = _serialise_transparency_entries(prefix_entries)
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "replay_type": "deterministic_mri_guard_replay",
        "replay_mode": "seeded_transparency_prefix",
        "statement": "Same replay input produces identical output and identical hashes.",
        "entity_type": evidence_payload["entity_type"],
        "entity_id": evidence_payload["entity_id"],
        "authority": evidence_payload["authority"],
        "actor": evidence_payload["actor"],
        "timestamp": evidence_payload["timestamp"],
        "transition_id": evidence_payload["transition_id"],
        "profile_name": report["profile"]["profile_name"],
        "profile_revision": report["profile"]["profile_revision"],
        "profile_digest": report["profile"]["profile_digest"],
        "icsd_policy_id": report["profile"]["icsd_policy_id"],
        "icsd_policy_version": report["profile"]["icsd_policy_version"],
        "authority_sources": evidence_payload["authority_sources"],
        "transparency_prefix_entry_count": receipt_entry["index"],
        "expected_hashes": {
            "input_state_hash": hash_canonical_json(before_state),
            "released_state_hash": hash_canonical_json(state),
            "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
            "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
            "verification_profile_descriptor_hash": hash_canonical_json(report["profile"]),
            "evidence_hash": hash_canonical_json(evidence_payload),
            "anchored_evidence_hash": hash_canonical_json(anchored_evidence_payload),
            "verification_report_hash": hash_canonical_json(report),
            "transparency_prefix_hash": _sha256_hex(prefix_bytes),
        },
    }


def _build_signature_catalog(member_bytes: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "signature_type": "sha256_integrity_catalog",
        "members": [
            {
                "path": member_name,
                "algorithm": "sha256",
                "value": _sha256_hex(payload),
            }
            for member_name, payload in sorted(member_bytes.items())
        ],
    }


def _build_single_series_optional_bundle_members(
    report: Mapping[str, Any],
) -> dict[str, bytes]:
    if report.get("analysis", {}).get("series") is not None:
        return {}
    contracts = report["contracts"]
    prior_verified_context = contracts.get("prior_verified_context")
    prior_verified_contexts = contracts.get("prior_verified_contexts")
    if prior_verified_context is None and prior_verified_contexts is None:
        return {}
    members = {
        _LONGITUDINAL_SELECTION_TRACE_NAME: _json_bytes(
            contracts["longitudinal_selection_trace"]
        ),
        _TEMPORAL_TRAJECTORY_TRACE_NAME: _json_bytes(
            contracts["temporal_trajectory_trace"]
        ),
        _LONGITUDINAL_REGISTRATION_TRACE_NAME: _json_bytes(
            contracts["longitudinal_registration_trace"]
        ),
        _TEMPORAL_CONSISTENCY_TRACE_NAME: _json_bytes(
            contracts["temporal_consistency_trace"]
        ),
    }
    if prior_verified_contexts is not None:
        members[_PRIOR_VERIFIED_CONTEXTS_NAME] = _json_bytes(prior_verified_contexts)
    if prior_verified_context is not None:
        members[_PRIOR_VERIFIED_CONTEXT_NAME] = _json_bytes(prior_verified_context)
    return members


def _build_multi_series_optional_bundle_members(
    report: Mapping[str, Any],
) -> dict[str, bytes]:
    if report.get("analysis", {}).get("series") is None:
        return {}
    trace = report["contracts"].get("study_temporal_aggregation_trace")
    if trace is None:
        return {}
    return {
        _STUDY_TEMPORAL_AGGREGATION_TRACE_NAME: _json_bytes(trace),
    }


def _build_optional_bundle_members(
    report: Mapping[str, Any],
) -> dict[str, bytes]:
    members = dict(_build_single_series_optional_bundle_members(report))
    members.update(_build_multi_series_optional_bundle_members(report))
    return members


def _ordered_bundle_member_names(member_names: Mapping[str, Any] | set[str] | tuple[str, ...]) -> list[str]:
    names = set(member_names)
    ordered = list(_REQUIRED_BUNDLE_MEMBERS)
    ordered.extend(name for name in _OPTIONAL_BUNDLE_MEMBERS if name in names)
    return ordered


def _load_bundle_source(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _read_extracted_bundle_members(path)
    return _read_bundle_archive_members(path)


def _read_bundle_archive_members(path: Path) -> dict[str, Any]:
    member_bytes: dict[str, bytes] = {}
    payloads: dict[str, Any] = {}
    bundle_bytes = path.read_bytes()
    with ZipFile(path, mode="r") as archive:
        names = set(archive.namelist())
        missing_members = [name for name in _REQUIRED_BUNDLE_MEMBERS if name not in names]
        if missing_members:
            missing = ", ".join(missing_members)
            raise ValueError(f"Bundle is missing required members: {missing}.")
        for member_name in _ordered_bundle_member_names(names):
            payload = archive.read(member_name)
            member_bytes[member_name] = payload
            if member_name.endswith(".json"):
                payloads[member_name] = json.loads(payload.decode("utf-8"))
    return {
        "source_kind": "archive",
        "bundle_bytes": bundle_bytes,
        "member_bytes": member_bytes,
        "payloads": payloads,
    }


def _read_extracted_bundle_members(bundle_json_path: Path) -> dict[str, Any]:
    if bundle_json_path.name != _BUNDLE_MANIFEST_NAME:
        msg = f"verify_bundle expects {_BUNDLE_MANIFEST_NAME!r} when pointed at an extracted JSON bundle."
        raise ValueError(msg)
    root = bundle_json_path.parent
    member_bytes: dict[str, bytes] = {}
    payloads: dict[str, Any] = {}
    for member_name in _ordered_bundle_member_names(
        _REQUIRED_BUNDLE_MEMBERS
        + tuple(
            member_name
            for member_name in _OPTIONAL_BUNDLE_MEMBERS
            if (root / member_name).exists()
        )
    ):
        member_path = root / member_name
        if not member_path.exists():
            msg = f"Extracted bundle is missing required member {member_name!r}."
            raise ValueError(msg)
        payload = member_path.read_bytes()
        member_bytes[member_name] = payload
        if member_name.endswith(".json"):
            payloads[member_name] = json.loads(payload.decode("utf-8"))
    return {
        "source_kind": "directory",
        "bundle_bytes": None,
        "member_bytes": member_bytes,
        "payloads": payloads,
    }


def _verify_deterministic_replay(
    *,
    input_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    expected_evidence_payload: Mapping[str, Any],
    expected_profile_descriptor: Mapping[str, Any],
    expected_member_bytes: Mapping[str, bytes],
    expected_bundle_bytes: bytes | None,
    replay_descriptor: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    if input_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _verify_multi_series_deterministic_replay(
            input_state=input_state,
            expected_state=expected_state,
            expected_report=expected_report,
            expected_evidence_payload=expected_evidence_payload,
            expected_profile_descriptor=expected_profile_descriptor,
            expected_member_bytes=expected_member_bytes,
            expected_bundle_bytes=expected_bundle_bytes,
            replay_descriptor=replay_descriptor,
            log_snapshot=log_snapshot,
            receipt_entry=receipt_entry,
            scratch_root=scratch_root,
        )
    failures: list[str] = []
    expected_hashes = dict(replay_descriptor["expected_hashes"])
    observed_hashes = {
        "input_state_hash": hash_canonical_json(input_state),
        "released_state_hash": hash_canonical_json(expected_state),
        "workstation_payload_hash": hash_canonical_json(expected_state["workstation_payload"]),
        "verified_manifest_hash": hash_canonical_json(expected_state["verified_manifest"]),
        "verification_profile_descriptor_hash": hash_canonical_json(expected_profile_descriptor),
        "evidence_hash": hash_canonical_json(expected_evidence_payload),
        "anchored_evidence_hash": hash_canonical_json(
            _transparency_log_payload_from_mapping(expected_evidence_payload)
        ),
        "verification_report_hash": hash_canonical_json(expected_report),
        "transparency_prefix_hash": _sha256_hex(
            _serialise_transparency_entries(log_snapshot["entries"][: receipt_entry["index"]])
        ),
    }
    for hash_name, observed_value in observed_hashes.items():
        if expected_hashes.get(hash_name) == observed_value:
            continue
        failures.append(
            f"replay.json expected {hash_name}={expected_hashes.get(hash_name)!r}, got {observed_value!r}."
        )

    replay_bundle_sha256 = None
    if failures:
        return {
            "verified": False,
            "failures": failures,
            "details": {"replayed_bundle_sha256": replay_bundle_sha256},
        }

    prefix_entries = log_snapshot["entries"][: receipt_entry["index"]]
    prefix_bytes = _serialise_transparency_entries(prefix_entries)
    temp_root = scratch_root / (
        f".mri_guard_replay_{_sha256_hex(_json_bytes(replay_descriptor))[:12]}"
    )
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime_profile = get_verification_profile(str(replay_descriptor["profile_name"]))
        runtime_profile_payload = runtime_profile.to_dict()
        if runtime_profile.profile_revision != replay_descriptor["profile_revision"]:
            failures.append(
                "embedded_profile_revision_mismatch: replay descriptor profile_revision "
                "does not match the runtime registry."
            )
        if runtime_profile.profile_digest() != replay_descriptor["profile_digest"]:
            failures.append(
                "embedded_profile_digest_mismatch: replay descriptor digest does not "
                "match the runtime registry."
            )
        if runtime_profile.icsd_policy_id != replay_descriptor["icsd_policy_id"]:
            failures.append(
                "embedded_profile_policy_id_mismatch: replay descriptor ICSD policy id "
                "does not match the runtime registry."
            )
        if runtime_profile.icsd_policy_version != replay_descriptor["icsd_policy_version"]:
            failures.append(
                "embedded_profile_policy_version_mismatch: replay descriptor ICSD policy version "
                "does not match the runtime registry."
            )
        if runtime_profile_payload != expected_profile_descriptor:
            failures.append(
                "embedded_profile_descriptor_mismatch: bundled profile descriptor does not "
                "match the runtime registry."
            )
        if failures:
            return {
                "verified": False,
                "failures": failures,
                "details": {"replayed_bundle_sha256": replay_bundle_sha256},
            }
        temp_log_path = temp_root / _TRANSPARENCY_LOG_NAME
        temp_log_path.write_bytes(prefix_bytes)
        harness = MriGuardianHarness(
            transparency_log_path=temp_log_path,
            default_profile_name=str(replay_descriptor["profile_name"]),
        )
        replay_result = harness.release_case(
            state=input_state,
            actor=str(replay_descriptor["actor"]),
            authority_sources=replay_descriptor["authority_sources"],
            profile_name=str(replay_descriptor["profile_name"]),
            timestamp=str(replay_descriptor["timestamp"]),
            transition_id=str(replay_descriptor["transition_id"]),
        )
        replay_evidence_payload = replay_result.evidence.as_dict()
        if replay_result.after_state != expected_state:
            failures.append(
                "Deterministic replay did not reproduce the released_state.json payload."
            )
        if replay_evidence_payload != expected_evidence_payload:
            failures.append("Deterministic replay did not reproduce the evidence.json payload.")
        replay_report = build_case_verification_report(replay_result.after_state)
        if replay_report != expected_report:
            failures.append(
                "Deterministic replay did not reproduce the verification_report.json payload."
            )
        replay_profile_descriptor = replay_report["contracts"]["verification_profile_descriptor"]
        if replay_profile_descriptor != expected_profile_descriptor:
            failures.append(
                "Deterministic replay did not reproduce the verification_profile_descriptor.json payload."
            )

        replay_bundle_path = temp_root / f"{expected_state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=replay_result,
            transparency_log_path=temp_log_path,
            output_path=replay_bundle_path,
        )
        replay_source = _load_bundle_source(replay_bundle_path)
        for member_name in _ordered_bundle_member_names(expected_member_bytes):
            if replay_source["member_bytes"][member_name] == expected_member_bytes[member_name]:
                continue
            failures.append(
                f"Deterministic replay changed bundle member {member_name!r}."
            )
        replay_bundle_bytes = replay_bundle_path.read_bytes()
        replay_bundle_sha256 = _sha256_hex(replay_bundle_bytes)
        if expected_bundle_bytes is not None and replay_bundle_bytes != expected_bundle_bytes:
            failures.append("Deterministic replay did not reproduce the original bundle archive bytes.")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "verified": not failures,
        "failures": failures,
        "details": {"replayed_bundle_sha256": replay_bundle_sha256},
    }


def _validate_multi_series_profile_descriptor_set(payload: Mapping[str, Any]) -> str | None:
    if payload.get("interface_type") != "MultiSeriesVerificationProfileDescriptorSet":
        return (
            "verification_profile_descriptor.json is invalid: expected "
            "'MultiSeriesVerificationProfileDescriptorSet'."
        )
    series_descriptors = payload.get("series_descriptors")
    if not isinstance(series_descriptors, list) or not series_descriptors:
        return (
            "verification_profile_descriptor.json is invalid: series_descriptors must be a "
            "non-empty list."
        )
    for item in series_descriptors:
        if not isinstance(item, Mapping):
            return (
                "verification_profile_descriptor.json is invalid: each series descriptor "
                "entry must be a mapping."
            )
        profile_descriptor = item.get("verification_profile_descriptor")
        if not isinstance(profile_descriptor, Mapping):
            return (
                "verification_profile_descriptor.json is invalid: each series descriptor "
                "entry must contain verification_profile_descriptor."
            )
        try:
            verification_profile_descriptor_from_dict(dict(profile_descriptor))
        except ValueError as exc:
            return f"verification_profile_descriptor.json is invalid: {exc}"
    return None


def _group_authority_sources_by_series_id(
    authority_sources: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in authority_sources:
        details = dict(source.get("details", {}))
        series_id = details.get("series_id")
        if not isinstance(series_id, str) or not series_id:
            raise ValueError("Multi-series authority source is missing details.series_id.")
        grouped.setdefault(series_id, []).append(dict(source))
    return grouped


def _build_multi_series_bundle_manifest(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    transcript: Mapping[str, Any],
    replay_descriptor: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = report["contracts"]
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "bundle_type": "mri_guard_bundle_multi_series_v2",
        "bundle_media_type": _BUNDLE_MEDIA_TYPE,
        "case_kind": MULTI_SERIES_CASE_KIND,
        "case_id": state["case_id"],
        "entity_type": evidence_payload["entity_type"],
        "entity_id": evidence_payload["entity_id"],
        "generated_at": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "verdict": "This study is verifiably correct.",
        "hashes": {
            "verification_profile_descriptor_hash": hash_canonical_json(
                contracts["verification_profile_descriptor"]
            ),
            "cross_series_verification_policy_hash": hash_canonical_json(
                contracts["cross_series_verification_policy"]
            ),
            "cross_series_verification_trace_hash": hash_canonical_json(
                contracts["cross_series_verification_trace"]
            ),
            "study_temporal_aggregation_trace_hash": hash_canonical_json(
                contracts["study_temporal_aggregation_trace"]
            ),
            "series_manifest_hashes": [
                {
                    "series_id": item["series_id"],
                    "hash": hash_canonical_json(item),
                }
                for item in report["manifest"]["series_manifests"]
            ],
            "input_state_hash": replay_descriptor["expected_hashes"]["input_state_hash"],
            "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
            "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
            "released_state_hash": hash_canonical_json(state),
            "evidence_hash": hash_canonical_json(evidence_payload),
            "anchored_evidence_hash": hash_canonical_json(anchored_evidence_payload),
            "verification_report_hash": hash_canonical_json(report),
            "transcript_hash": hash_canonical_json(transcript),
            "transparency_log_hash": _sha256_hex(log_snapshot["bytes"]),
            "transparency_prefix_hash": replay_descriptor["expected_hashes"][
                "transparency_prefix_hash"
            ],
        },
        "verification": {
            "release_eligible": bool(report["release_eligible"]),
            "decision": report["decision"],
            "decision_summary": report["decision_summary"],
            "failure_reasons": report["failure_reasons"],
            "pipeline_phase_statuses": _pipeline_phase_statuses(report),
            "released_series_ids": report["analysis"]["released_series_ids"],
            "withheld_series_ids": report["analysis"]["withheld_series_ids"],
        },
        "deterministic_replay": {
            "supported": True,
            "mode": replay_descriptor["replay_mode"],
            "input_member": _INPUT_STATE_NAME,
            "replay_member": _REPLAY_DESCRIPTOR_NAME,
            "profile_descriptor_member": _PROFILE_DESCRIPTOR_NAME,
            "transition_id": replay_descriptor["transition_id"],
        },
        "transparency": {
            "receipt_reference": _transparency_receipt(evidence_payload)["reference"],
            "entry_hash": receipt_entry["entry_hash"],
            "previous_entry_hash": receipt_entry["previous_entry_hash"],
            "checkpoint": log_snapshot["checkpoint"],
            "entry_index": receipt_entry["index"],
            "log_length": len(log_snapshot["entries"]),
        },
        "members": list(_REQUIRED_BUNDLE_MEMBERS),
        "signatures_member": _SIGNATURES_NAME,
    }


def _build_multi_series_bundle_transcript(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
    checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "transcript_type": "portable_mri_guard_multi_series_transcript",
        "case_id": state["case_id"],
        "timestamp": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "statement": "This study is verifiably correct.",
        "events": [
            {
                "stage": "profile_set",
                "summary": "Resolved the ordered study profile descriptor set.",
                "details": {
                    "descriptor_hash": hash_canonical_json(
                        report["contracts"]["verification_profile_descriptor"]
                    ),
                    "series_count": len(report["series_reports"]),
                },
            },
            {
                "stage": "per_series_proof",
                "summary": "Verified every payload series independently.",
                "details": {
                    "series": [
                        {
                            "series_id": item["series_id"],
                            "resolved_role": item["resolved_role"],
                            "gating_status": item["gating_status"],
                            "series_proof_status": item["series_proof_status"],
                        }
                        for item in report["series_reports"]
                    ],
                },
            },
            {
                "stage": "cross_series_consistency",
                "summary": "Evaluated required-series geometry and positive-lesion corroboration.",
                "details": {
                    "geometry_verification": report["contracts"]["cross_series_verification_trace"][
                        "geometry_verification"
                    ],
                    "lesion_corroboration": report["contracts"]["cross_series_verification_trace"][
                        "lesion_corroboration"
                    ],
                },
            },
            {
                "stage": "release",
                "summary": "Released only the study series that remained downstream-allowed after the merged gate.",
                "details": {
                    "release_status": state["release_status"],
                    "released_series_ids": report["analysis"]["released_series_ids"],
                    "withheld_series_ids": report["analysis"]["withheld_series_ids"],
                    "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
                    "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
                },
            },
            {
                "stage": "transparency",
                "summary": "Anchored the ICSD evidence envelope into the append-only transparency hash chain.",
                "details": {
                    "receipt_reference": _transparency_receipt(evidence_payload)["reference"],
                    "entry_hash": receipt_entry["entry_hash"],
                    "previous_entry_hash": receipt_entry["previous_entry_hash"],
                    "checkpoint": checkpoint,
                },
            },
        ],
    }


def _build_multi_series_replay_descriptor(
    *,
    before_state: Mapping[str, Any],
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
) -> dict[str, Any]:
    prefix_entries = log_snapshot["entries"][: receipt_entry["index"]]
    prefix_bytes = _serialise_transparency_entries(prefix_entries)
    primary_series_report = next(
        item for item in report["series_reports"] if item["resolved_role"] == "primary"
    )
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "case_kind": MULTI_SERIES_CASE_KIND,
        "replay_type": "deterministic_mri_guard_multi_series_replay",
        "replay_mode": "seeded_transparency_prefix",
        "statement": "Same replay input produces identical multi-series output and identical hashes.",
        "entity_type": evidence_payload["entity_type"],
        "entity_id": evidence_payload["entity_id"],
        "authority": evidence_payload["authority"],
        "actor": evidence_payload["actor"],
        "timestamp": evidence_payload["timestamp"],
        "transition_id": evidence_payload["transition_id"],
        "default_profile_name": primary_series_report["profile"]["profile_name"],
        "authority_sources_by_series_id": _group_authority_sources_by_series_id(
            list(evidence_payload["authority_sources"])
        ),
        "transparency_prefix_entry_count": receipt_entry["index"],
        "expected_hashes": {
            "input_state_hash": hash_canonical_json(before_state),
            "released_state_hash": hash_canonical_json(state),
            "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
            "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
            "verification_profile_descriptor_hash": hash_canonical_json(
                report["contracts"]["verification_profile_descriptor"]
            ),
            "evidence_hash": hash_canonical_json(evidence_payload),
            "anchored_evidence_hash": hash_canonical_json(anchored_evidence_payload),
            "verification_report_hash": hash_canonical_json(report),
            "transparency_prefix_hash": _sha256_hex(prefix_bytes),
        },
    }


def _verify_multi_series_deterministic_replay(
    *,
    input_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    expected_evidence_payload: Mapping[str, Any],
    expected_profile_descriptor: Mapping[str, Any],
    expected_member_bytes: Mapping[str, bytes],
    expected_bundle_bytes: bytes | None,
    replay_descriptor: Mapping[str, Any],
    log_snapshot: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_hashes = dict(replay_descriptor["expected_hashes"])
    observed_hashes = {
        "input_state_hash": hash_canonical_json(input_state),
        "released_state_hash": hash_canonical_json(expected_state),
        "workstation_payload_hash": hash_canonical_json(expected_state["workstation_payload"]),
        "verified_manifest_hash": hash_canonical_json(expected_state["verified_manifest"]),
        "verification_profile_descriptor_hash": hash_canonical_json(expected_profile_descriptor),
        "evidence_hash": hash_canonical_json(expected_evidence_payload),
        "anchored_evidence_hash": hash_canonical_json(
            _transparency_log_payload_from_mapping(expected_evidence_payload)
        ),
        "verification_report_hash": hash_canonical_json(expected_report),
        "transparency_prefix_hash": _sha256_hex(
            _serialise_transparency_entries(log_snapshot["entries"][: receipt_entry["index"]])
        ),
    }
    for hash_name, observed_value in observed_hashes.items():
        if expected_hashes.get(hash_name) == observed_value:
            continue
        failures.append(
            f"replay.json expected {hash_name}={expected_hashes.get(hash_name)!r}, got {observed_value!r}."
        )

    descriptor_failure = _validate_multi_series_profile_descriptor_set(expected_profile_descriptor)
    if descriptor_failure is not None:
        failures.append(descriptor_failure)

    replay_bundle_sha256 = None
    if failures:
        return {
            "verified": False,
            "failures": failures,
            "details": {"replayed_bundle_sha256": replay_bundle_sha256},
        }

    prefix_entries = log_snapshot["entries"][: receipt_entry["index"]]
    prefix_bytes = _serialise_transparency_entries(prefix_entries)
    temp_root = scratch_root / (
        f".mri_guard_replay_{_sha256_hex(_json_bytes(replay_descriptor))[:12]}"
    )
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        temp_log_path = temp_root / _TRANSPARENCY_LOG_NAME
        temp_log_path.write_bytes(prefix_bytes)
        harness = MriGuardianHarness(
            transparency_log_path=temp_log_path,
            default_profile_name=str(replay_descriptor["default_profile_name"]),
        )
        replay_result = harness.release_multi_series_case(
            state=input_state,
            actor=str(replay_descriptor["actor"]),
            authority_sources_by_series_id=replay_descriptor["authority_sources_by_series_id"],
            timestamp=str(replay_descriptor["timestamp"]),
            transition_id=str(replay_descriptor["transition_id"]),
        )
        replay_evidence_payload = replay_result.evidence.as_dict()
        if replay_result.after_state != expected_state:
            failures.append(
                "Deterministic replay did not reproduce the released_state.json payload."
            )
        if replay_evidence_payload != expected_evidence_payload:
            failures.append("Deterministic replay did not reproduce the evidence.json payload.")
        replay_report = build_case_verification_report(replay_result.after_state)
        if replay_report != expected_report:
            failures.append(
                "Deterministic replay did not reproduce the verification_report.json payload."
            )
        replay_profile_descriptor = replay_report["contracts"]["verification_profile_descriptor"]
        if replay_profile_descriptor != expected_profile_descriptor:
            failures.append(
                "Deterministic replay did not reproduce the verification_profile_descriptor.json payload."
            )

        replay_bundle_path = temp_root / f"{expected_state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=replay_result,
            transparency_log_path=temp_log_path,
            output_path=replay_bundle_path,
        )
        replay_source = _load_bundle_source(replay_bundle_path)
        for member_name in _REQUIRED_BUNDLE_MEMBERS:
            if replay_source["member_bytes"][member_name] == expected_member_bytes[member_name]:
                continue
            failures.append(
                f"Deterministic replay changed bundle member {member_name!r}."
            )
        replay_bundle_bytes = replay_bundle_path.read_bytes()
        replay_bundle_sha256 = _sha256_hex(replay_bundle_bytes)
        if expected_bundle_bytes is not None and replay_bundle_bytes != expected_bundle_bytes:
            failures.append("Deterministic replay did not reproduce the original bundle archive bytes.")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "verified": not failures,
        "failures": failures,
        "details": {"replayed_bundle_sha256": replay_bundle_sha256},
    }


def _read_transparency_log_snapshot(path: Path | str) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.exists():
        msg = f"Transparency log snapshot not found: {log_path}"
        raise ValueError(msg)
    payload = log_path.read_bytes()
    entries = _parse_transparency_entries(payload)
    chain_error = _verify_transparency_entries(entries)
    if chain_error is not None:
        raise ValueError(chain_error)
    return {
        "bytes": payload,
        "entries": entries,
        "checkpoint": entries[-1]["entry_hash"] if entries else None,
    }


def _parse_transparency_entries(payload: bytes) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            msg = f"Transparency log line {line_number} is not a JSON object."
            raise ValueError(msg)
        entries.append(entry)
    return entries


def _verify_transparency_entries(entries: list[dict[str, Any]]) -> str | None:
    previous_entry_hash = None
    for expected_index, entry in enumerate(entries):
        if entry.get("index") != expected_index:
            return "Bundled transparency log index sequence is not contiguous."
        if entry.get("previous_entry_hash") != previous_entry_hash:
            return "Bundled transparency log previous_entry_hash chain is broken."
        candidate = {
            "index": entry.get("index"),
            "previous_entry_hash": entry.get("previous_entry_hash"),
            "evidence_hash": entry.get("evidence_hash"),
        }
        expected_entry_hash = hash_canonical_json(candidate)
        if entry.get("entry_hash") != expected_entry_hash:
            return "Bundled transparency log entry_hash does not match canonical contents."
        previous_entry_hash = expected_entry_hash
    return None


def _resolve_receipt_entry(
    *,
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    log_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = _transparency_receipt(evidence_payload)
    expected_reference = receipt["reference"]
    expected_evidence_hash = hash_canonical_json(anchored_evidence_payload)
    for entry in log_entries:
        if _transparency_reference(entry) != expected_reference:
            continue
        if entry["evidence_hash"] != expected_evidence_hash:
            msg = "Transparency receipt reference exists, but evidence_hash does not match."
            raise ValueError(msg)
        return entry
    msg = f"Transparency receipt {expected_reference!r} is not present in the bundled log."
    raise ValueError(msg)


def _transparency_receipt(evidence_payload: Mapping[str, Any]) -> dict[str, Any]:
    for provenance in evidence_payload["provenance"]:
        source = provenance.get("source", "")
        if isinstance(source, str) and source.startswith("transparency-log/"):
            return provenance
    raise ValueError("Evidence payload does not contain a transparency-log provenance receipt.")


def _transparency_reference(entry: Mapping[str, Any]) -> str:
    return f"{entry['index']}:{entry['entry_hash']}"


def _serialise_transparency_entries(entries: list[dict[str, Any]]) -> bytes:
    if not entries:
        return b""
    lines = [canonical_json_dumps(entry) for entry in entries]
    return f"{'\n'.join(lines)}\n".encode("utf-8")


def _json_bytes(payload: Mapping[str, Any] | list[Any]) -> bytes:
    return f"{canonical_json_dumps(payload)}\n".encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle_member(archive: ZipFile, member_name: str, payload: bytes) -> None:
    info = ZipInfo(filename=member_name, date_time=_FIXED_ZIP_TIMESTAMP)
    info.compress_type = ZIP_STORED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)
