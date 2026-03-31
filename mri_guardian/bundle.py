"""Guardian Bundle V1 writer and deterministic replay verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from icsd.core.evidence import _transparency_log_payload_from_mapping, validate_evidence_payload
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json
from icsd.core.transition import TransitionResult

from .harness import (
    MULTI_SERIES_CASE_KIND,
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    _check_released_state_matches_trace,
    build_case_verification_report,
)
from .journal import build_decision_journal, build_transparency_receipt_context
from .profiles import get_verification_profile

GUARDIAN_BUNDLE_EXTENSION = ".guardian-bundle"
GUARDIAN_BUNDLE_TYPE = "dpa-guardian-bundle"
GUARDIAN_BUNDLE_SCHEMA_VERSION = "1"

_FIXED_ZIP_TIMESTAMP = (2026, 3, 26, 0, 0, 0)
_BUNDLE_MANIFEST_NAME = "bundle.json"
_INPUT_STATE_NAME = "input_state.json"
_RELEASED_STATE_NAME = "released_state.json"
_WORKSTATION_PAYLOAD_NAME = "workstation_payload.json"
_VERIFICATION_REPORT_NAME = "verification_report.json"
_DECISION_JOURNAL_NAME = "decision_journal.json"
_EVIDENCE_NAME = "evidence.json"
_TRANSPARENCY_PREFIX_NAME = "transparency-prefix.jsonl"
_JSON_MEDIA_TYPE = "application/json"
_BUNDLE_REDACTION_SENTINEL = "__INTEGRITY_MANIFEST_REDACTED__"
_ATTACHMENTS_KEY = "attachments"
_REQUIRED_TOP_LEVEL_KEYS = (
    "bundle_type",
    "bundle_schema_version",
    "claim",
    "proof_context",
    "replay_inputs",
    "release_artifacts",
    "integrity_manifest",
    "transparency_anchor",
)
_RELEASE_ARTIFACT_ORDER = (
    ("evidence", _EVIDENCE_NAME),
    ("released_state", _RELEASED_STATE_NAME),
    ("workstation_payload", _WORKSTATION_PAYLOAD_NAME),
    ("verification_report", _VERIFICATION_REPORT_NAME),
    ("decision_journal", _DECISION_JOURNAL_NAME),
)
_PROOF_CRITICAL_MEMBERS = (
    _INPUT_STATE_NAME,
    _RELEASED_STATE_NAME,
    _WORKSTATION_PAYLOAD_NAME,
    _VERIFICATION_REPORT_NAME,
    _DECISION_JOURNAL_NAME,
    _EVIDENCE_NAME,
    _TRANSPARENCY_PREFIX_NAME,
)


def write_portable_evidence_bundle(
    *,
    result: TransitionResult,
    transparency_log_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Write a deterministic Guardian Bundle V1 archive and return its metadata."""
    before_state = result.before_state
    state = result.after_state
    _ensure_released_state(state)
    evidence_payload = result.evidence.as_dict()
    evidence_failures = validate_evidence_payload(evidence_payload)
    if evidence_failures:
        raise ValueError(f"ICSD evidence payload is invalid: {evidence_failures!r}")
    report = build_case_verification_report(state)
    if not report["release_eligible"]:
        raise ValueError("Guardian bundles can only be written for release-eligible cases.")
    if (
        state.get("case_kind") != MULTI_SERIES_CASE_KIND
        and not report["profile"]["release_semantics"]["portable_bundle_allowed"]
    ):
        raise ValueError("Guardian bundles are not allowed for the selected runtime profile.")

    anchored_evidence_payload = _transparency_log_payload_from_mapping(evidence_payload)
    log_snapshot = _read_transparency_log_snapshot(transparency_log_path)
    receipt_entry = _resolve_receipt_entry(
        evidence_payload=evidence_payload,
        anchored_evidence_payload=anchored_evidence_payload,
        log_entries=log_snapshot["entries"],
    )
    transparency_prefix_bytes = _serialise_transparency_entries(
        log_snapshot["entries"][: int(receipt_entry["index"])]
    )
    decision_journal = build_decision_journal(
        report,
        transparency_receipt_context=_minimal_transparency_receipt_context(
            evidence_payload=evidence_payload,
            receipt_entry=receipt_entry,
        ),
    )
    bundle_payload = _build_bundle_payload(
        state=state,
        report=report,
        evidence_payload=evidence_payload,
        decision_journal=decision_journal,
        receipt_entry=receipt_entry,
    )
    member_bytes = {
        _INPUT_STATE_NAME: _json_bytes(before_state),
        _RELEASED_STATE_NAME: _json_bytes(state),
        _WORKSTATION_PAYLOAD_NAME: _json_bytes(state["workstation_payload"]),
        _VERIFICATION_REPORT_NAME: _json_bytes(report),
        _DECISION_JOURNAL_NAME: _json_bytes(decision_journal),
        _EVIDENCE_NAME: _json_bytes(evidence_payload),
        _TRANSPARENCY_PREFIX_NAME: transparency_prefix_bytes,
    }
    bundle_payload["integrity_manifest"] = _build_integrity_manifest(
        bundle_payload=bundle_payload,
        member_bytes=member_bytes,
        attachments=None,
    )
    member_bytes[_BUNDLE_MANIFEST_NAME] = _json_bytes(bundle_payload)
    bundle_path = Path(output_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, mode="w") as archive:
        for member_name in _declared_member_order(bundle_payload):
            _write_bundle_member(archive, member_name, member_bytes[member_name])
    return {
        "path": str(bundle_path),
        "bundle": bundle_payload,
        "decision_journal": decision_journal,
    }


def verify_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Verify a bundle archive or an extracted bundle.json package."""
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
        bundle_payload = source["bundle_payload"]
        member_names = list(source["member_names"])
        member_bytes = dict(source["member_bytes"])
        payloads = dict(source["payloads"])
        failures.extend(
            _verify_bundle_structure_and_integrity(
                bundle_payload=bundle_payload,
                member_names=member_names,
                member_bytes=member_bytes,
                payloads=payloads,
                source_kind=source_kind,
            )
        )
        input_state = _require_member_payload(payloads, _INPUT_STATE_NAME)
        state = _require_member_payload(payloads, _RELEASED_STATE_NAME)
        report = _require_member_payload(payloads, _VERIFICATION_REPORT_NAME)
        decision_journal = _require_member_payload(payloads, _DECISION_JOURNAL_NAME)
        evidence_payload = _require_member_payload(payloads, _EVIDENCE_NAME)
        anchored_evidence_payload = _transparency_log_payload_from_mapping(evidence_payload)
        receipt_entry = _receipt_entry_from_bundle(bundle_payload)
        transparency_prefix_bytes = member_bytes.get(_TRANSPARENCY_PREFIX_NAME, b"")
        transparency_prefix_entries = _parse_transparency_entries(transparency_prefix_bytes)
        evidence_failures = validate_evidence_payload(evidence_payload)
        if evidence_failures:
            failures.append(f"evidence.json failed ICSD validation: {evidence_failures!r}")
        _ensure_released_state(state)
        expected_report = build_case_verification_report(state)
        if report != expected_report:
            failures.append("verification_report.json does not match the released state's math trace.")
        trace_failures = _check_released_state_matches_trace(state)
        if trace_failures:
            failures.extend(failure.message for failure in trace_failures)
        if receipt_entry is None:
            failures.append("bundle.json transparency_anchor is missing anchored receipt metadata.")
        else:
            failures.extend(
                _verify_transparency_anchor_against_prefix(
                    evidence_payload=evidence_payload,
                    anchored_evidence_payload=anchored_evidence_payload,
                    transparency_prefix_entries=transparency_prefix_entries,
                    receipt_entry=receipt_entry,
                )
            )
        expected_decision_journal = build_decision_journal(
            expected_report,
            transparency_receipt_context=(
                None
                if receipt_entry is None
                else _minimal_transparency_receipt_context(
                    evidence_payload=evidence_payload,
                    receipt_entry=receipt_entry,
                )
            ),
        )
        if decision_journal != expected_decision_journal:
            failures.append("decision_journal.json does not match the deterministic decision journal.")
        expected_bundle = _build_bundle_payload(
            state=state,
            report=expected_report,
            evidence_payload=evidence_payload,
            decision_journal=expected_decision_journal,
            receipt_entry=receipt_entry,
            attachments=bundle_payload.get(_ATTACHMENTS_KEY),
        )
        if bundle_payload.get("claim") != expected_bundle["claim"]:
            failures.append("bundle.json claim does not match the deterministic claim surface.")
        if bundle_payload.get("proof_context") != expected_bundle["proof_context"]:
            failures.append("bundle.json proof_context does not match the deterministic proof surface.")
        if bundle_payload.get("replay_inputs") != expected_bundle["replay_inputs"]:
            failures.append("bundle.json replay_inputs does not match the deterministic replay seed surface.")
        if bundle_payload.get("release_artifacts") != expected_bundle["release_artifacts"]:
            failures.append("bundle.json release_artifacts does not match the canonical artifact descriptor table.")
        replay_verification = _verify_deterministic_replay(
            input_state=input_state,
            expected_state=state,
            expected_report=expected_report,
            expected_decision_journal=expected_decision_journal,
            expected_evidence_payload=evidence_payload,
            bundle_payload=bundle_payload,
            transparency_prefix_bytes=transparency_prefix_bytes,
            receipt_entry=receipt_entry,
            scratch_root=path.parent,
        )
        failures.extend(replay_verification["failures"])
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
            "case_id": payloads.get(_RELEASED_STATE_NAME, {}).get("case_id"),
            "bundle_type": bundle_payload.get("bundle_type"),
            "replay_verified": replay_verification["verified"],
            "replayed_bundle_sha256": replay_verification["details"].get("replayed_bundle_sha256"),
            "source_kind": source_kind,
        },
    }


def _build_bundle_payload(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    decision_journal: Mapping[str, Any],
    receipt_entry: Mapping[str, Any] | None,
    attachments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bundle_type": GUARDIAN_BUNDLE_TYPE,
        "bundle_schema_version": GUARDIAN_BUNDLE_SCHEMA_VERSION,
        "claim": _build_claim(state=state, report=report),
        "proof_context": _build_proof_context(report),
        "replay_inputs": _build_replay_inputs(state=state, report=report, evidence_payload=evidence_payload),
        "release_artifacts": _build_release_artifacts(),
        "integrity_manifest": {},
        "transparency_anchor": _build_transparency_anchor(
            evidence_payload=evidence_payload,
            receipt_entry=receipt_entry,
        ),
    }
    if attachments:
        payload[_ATTACHMENTS_KEY] = _normalise_attachments(attachments)
    return payload


def _build_claim(*, state: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_claim(state=state, report=report)
    return _build_single_series_claim(report["contracts"]["ai_claim_envelope"])


def _build_single_series_claim(ai_claim_envelope: Mapping[str, Any]) -> dict[str, Any]:
    ai_identity = {"ai_model_id": ai_claim_envelope["ai_model_id"]}
    if ai_claim_envelope.get("ai_stack_identity") is not None:
        ai_identity["ai_stack_identity"] = deepcopy(ai_claim_envelope["ai_stack_identity"])
    return {
        "claim_scope": ai_claim_envelope["claim_scope"],
        "case_id": ai_claim_envelope["case_id"],
        "case_kind": "single_series_v1",
        "input_identity": {
            "study_uid": ai_claim_envelope["study_uid"],
            "series_uid": ai_claim_envelope["series_uid"],
            "patient_pseudonym": ai_claim_envelope["patient_pseudonym"],
            "raw_kspace_hash": ai_claim_envelope["raw_kspace_hash"],
            "voxel_spacing_mm": list(ai_claim_envelope["voxel_spacing_mm"]),
        },
        "ai_identity": ai_identity,
        "declared_outputs": deepcopy(ai_claim_envelope["claimed_outputs"]),
    }


def _build_multi_series_claim(*, state: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    input_identity = {
        "study_uid": state["study_uid"],
        "patient_pseudonym": state["patient_pseudonym"],
    }
    scanner_id = state.get("scanner_id")
    if isinstance(scanner_id, str) and scanner_id:
        input_identity["scanner_id"] = scanner_id
    return {
        "claim_scope": "multi_series_study_verification",
        "case_id": state["case_id"],
        "case_kind": MULTI_SERIES_CASE_KIND,
        "input_identity": input_identity,
        "required_series_ids": list(report["analysis"]["required_series_ids"]),
        "supporting_series_ids": list(report["analysis"]["supporting_series_ids"]),
        "series_claims": [
            {
                "series_id": series_report["series_id"],
                "series_uid": series_report["series_uid"],
                "claim": _build_single_series_claim(series_report["contracts"]["ai_claim_envelope"]),
            }
            for series_report in report["series_reports"]
        ],
    }


def _build_proof_context(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_proof_context(report)
    return _build_single_series_proof_context(report)


def _build_single_series_proof_context(report: Mapping[str, Any]) -> dict[str, Any]:
    contracts = report["contracts"]
    derived_traces: dict[str, Any] = {
        "proof_core_evidence_trace": deepcopy(contracts["proof_core_evidence_trace"]),
    }
    if _single_series_temporal_context_present(contracts):
        derived_traces.update(
            {
                "longitudinal_selection_trace": deepcopy(contracts["longitudinal_selection_trace"]),
                "temporal_trajectory_trace": deepcopy(contracts["temporal_trajectory_trace"]),
                "longitudinal_registration_trace": deepcopy(contracts["longitudinal_registration_trace"]),
                "temporal_consistency_trace": deepcopy(contracts["temporal_consistency_trace"]),
            }
        )
    return {
        "profile_identity": {
            "profile_name": report["profile"]["profile_name"],
            "profile_revision": report["profile"]["profile_revision"],
            "profile_digest": report["profile"]["profile_digest"],
        },
        "policy_identity": {
            "icsd_policy_id": report["profile"]["icsd_policy_id"],
            "icsd_policy_version": report["profile"]["icsd_policy_version"],
            "verification_profile_descriptor_hash": hash_canonical_json(contracts["verification_profile_descriptor"]),
            "mask_derivation_policy_hash": hash_canonical_json(contracts["mask_derivation_policy"]),
            "verification_tolerance_policy_hash": hash_canonical_json(contracts["verification_tolerance_policy"]),
        },
        "tolerance_definitions": deepcopy(contracts["verification_tolerance_policy"]["rules"]),
        "proof_outputs": {
            "measurement_integrity_proof": deepcopy(contracts["measurement_integrity_proof"]),
            "reconstruction_parity_proof": deepcopy(contracts["reconstruction_parity_proof"]),
            "segmentation_policy_proof": deepcopy(contracts["segmentation_policy_proof"]),
            "volume_computation_proof": deepcopy(contracts["volume_computation_proof"]),
        },
        "derived_traces": derived_traces,
    }


def _build_multi_series_proof_context(report: Mapping[str, Any]) -> dict[str, Any]:
    contracts = report["contracts"]
    return {
        "profile_identity": {
            "case_kind": MULTI_SERIES_CASE_KIND,
            "interface_type": report["profile"]["interface_type"],
            "interface_version": report["profile"]["interface_version"],
            "verification_profile_descriptor_hash": hash_canonical_json(contracts["verification_profile_descriptor"]),
        },
        "policy_identity": {
            "cross_series_verification_policy_hash": hash_canonical_json(contracts["cross_series_verification_policy"]),
            "series_policy_hashes": [
                {
                    "series_id": series_report["series_id"],
                    "mask_derivation_policy_hash": hash_canonical_json(series_report["contracts"]["mask_derivation_policy"]),
                    "verification_tolerance_policy_hash": hash_canonical_json(series_report["contracts"]["verification_tolerance_policy"]),
                }
                for series_report in report["series_reports"]
            ],
        },
        "tolerance_definitions": {
            "series": [
                {
                    "series_id": series_report["series_id"],
                    "rules": deepcopy(series_report["contracts"]["verification_tolerance_policy"]["rules"]),
                }
                for series_report in report["series_reports"]
            ]
        },
        "proof_outputs": {
            "series": [
                {
                    "series_id": series_report["series_id"],
                    "series_uid": series_report["series_uid"],
                    "resolved_role": series_report["resolved_role"],
                    "gating_status": series_report["gating_status"],
                    "proof_stage_outputs": {
                        "measurement_integrity_proof": deepcopy(series_report["contracts"]["measurement_integrity_proof"]),
                        "reconstruction_parity_proof": deepcopy(series_report["contracts"]["reconstruction_parity_proof"]),
                        "segmentation_policy_proof": deepcopy(series_report["contracts"]["segmentation_policy_proof"]),
                        "volume_computation_proof": deepcopy(series_report["contracts"]["volume_computation_proof"]),
                    },
                }
                for series_report in report["series_reports"]
            ]
        },
        "derived_traces": {
            "series": [_build_series_trace_surface(series_report) for series_report in report["series_reports"]],
            "study": _build_multi_series_study_traces(contracts),
        },
    }


def _build_series_trace_surface(series_report: Mapping[str, Any]) -> dict[str, Any]:
    contracts = series_report["contracts"]
    payload: dict[str, Any] = {
        "series_id": series_report["series_id"],
        "series_uid": series_report["series_uid"],
        "resolved_role": series_report["resolved_role"],
        "gating_status": series_report["gating_status"],
        "proof_core_evidence_trace": deepcopy(contracts["proof_core_evidence_trace"]),
    }
    if _single_series_temporal_context_present(contracts):
        payload["longitudinal_selection_trace"] = deepcopy(contracts["longitudinal_selection_trace"])
        payload["temporal_trajectory_trace"] = deepcopy(contracts["temporal_trajectory_trace"])
        payload["longitudinal_registration_trace"] = deepcopy(contracts["longitudinal_registration_trace"])
        payload["temporal_consistency_trace"] = deepcopy(contracts["temporal_consistency_trace"])
    return payload


def _build_multi_series_study_traces(contracts: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cross_series_verification_trace": deepcopy(contracts["cross_series_verification_trace"]),
    }
    if contracts.get("study_temporal_aggregation_trace") is not None:
        payload["study_temporal_aggregation_trace"] = deepcopy(contracts["study_temporal_aggregation_trace"])
    return payload


def _build_replay_inputs(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_replay_inputs(report=report, evidence_payload=evidence_payload)
    return _build_single_series_replay_inputs(report=report, evidence_payload=evidence_payload)


def _build_single_series_replay_inputs(
    *,
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = report["contracts"]
    replay_contracts: dict[str, Any] = {
        "verification_profile_descriptor": deepcopy(contracts["verification_profile_descriptor"]),
        "mask_derivation_policy": deepcopy(contracts["mask_derivation_policy"]),
        "verification_tolerance_policy": deepcopy(contracts["verification_tolerance_policy"]),
    }
    if contracts.get("prior_verified_contexts") is not None:
        replay_contracts["prior_verified_contexts"] = deepcopy(contracts["prior_verified_contexts"])
    if contracts.get("prior_verified_context") is not None:
        replay_contracts["prior_verified_context"] = deepcopy(contracts["prior_verified_context"])
    return {
        "replay_mode": "seeded_transparency_prefix",
        "input_member": _INPUT_STATE_NAME,
        "transparency_prefix_member": _TRANSPARENCY_PREFIX_NAME,
        "actor": evidence_payload["actor"],
        "timestamp": evidence_payload["timestamp"],
        "transition_id": evidence_payload["transition_id"],
        "profile_name": report["profile"]["profile_name"],
        "authority_sources": deepcopy(evidence_payload["authority_sources"]),
        "contracts": replay_contracts,
    }


def _build_multi_series_replay_inputs(
    *,
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    primary_series_report = next(item for item in report["series_reports"] if item["resolved_role"] == "primary")
    series_contracts: list[dict[str, Any]] = []
    for series_report in report["series_reports"]:
        contracts = series_report["contracts"]
        payload: dict[str, Any] = {
            "series_id": series_report["series_id"],
            "verification_profile_descriptor": deepcopy(contracts["verification_profile_descriptor"]),
            "mask_derivation_policy": deepcopy(contracts["mask_derivation_policy"]),
            "verification_tolerance_policy": deepcopy(contracts["verification_tolerance_policy"]),
        }
        if contracts.get("prior_verified_contexts") is not None:
            payload["prior_verified_contexts"] = deepcopy(contracts["prior_verified_contexts"])
        if contracts.get("prior_verified_context") is not None:
            payload["prior_verified_context"] = deepcopy(contracts["prior_verified_context"])
        series_contracts.append(payload)
    return {
        "replay_mode": "seeded_transparency_prefix",
        "input_member": _INPUT_STATE_NAME,
        "transparency_prefix_member": _TRANSPARENCY_PREFIX_NAME,
        "actor": evidence_payload["actor"],
        "timestamp": evidence_payload["timestamp"],
        "transition_id": evidence_payload["transition_id"],
        "default_profile_name": primary_series_report["profile"]["profile_name"],
        "authority_sources": deepcopy(evidence_payload["authority_sources"]),
        "contracts": {
            "verification_profile_descriptor": deepcopy(report["contracts"]["verification_profile_descriptor"]),
            "cross_series_verification_policy": deepcopy(report["contracts"]["cross_series_verification_policy"]),
            "series_contracts": series_contracts,
        },
    }


def _build_release_artifacts() -> dict[str, Any]:
    return {
        "artifacts": [
            {"artifact_id": artifact_id, "member": member_name, "media_type": _JSON_MEDIA_TYPE}
            for artifact_id, member_name in _RELEASE_ARTIFACT_ORDER
        ]
    }


def _build_transparency_anchor(
    *,
    evidence_payload: Mapping[str, Any],
    receipt_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if receipt_entry is None:
        return {"status": "not_anchored"}
    return {
        "status": "anchored",
        "receipt_reference": _transparency_receipt(evidence_payload)["reference"],
        "anchor_id": _transparency_reference(receipt_entry),
        "entry_hash": receipt_entry["entry_hash"],
        "previous_entry_hash": receipt_entry.get("previous_entry_hash"),
        "entry_index": int(receipt_entry["index"]),
    }


def _build_integrity_manifest(
    *,
    bundle_payload: Mapping[str, Any],
    member_bytes: Mapping[str, bytes],
    attachments: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    member_order = [_BUNDLE_MANIFEST_NAME, *_PROOF_CRITICAL_MEMBERS]
    attachment_members = [str(item["member"]) for item in _normalise_attachments(attachments or [])]
    member_order.extend(attachment_members)
    member_hashes = _member_hash_entries(
        bundle_payload=bundle_payload,
        member_bytes=member_bytes,
        member_order=member_order,
    )
    return {
        "member_order": member_order,
        "member_hashes": member_hashes,
        "proof_critical_members": list(_PROOF_CRITICAL_MEMBERS),
        "bundle_descriptor_hash": _bundle_descriptor_hash(bundle_payload),
        "bundle_hash": hash_canonical_json(member_hashes),
    }


def _verify_bundle_structure_and_integrity(
    *,
    bundle_payload: Mapping[str, Any],
    member_names: Sequence[str],
    member_bytes: Mapping[str, bytes],
    payloads: Mapping[str, Any],
    source_kind: str,
) -> list[str]:
    failures: list[str] = []
    expected_keys = set(_REQUIRED_TOP_LEVEL_KEYS)
    if _ATTACHMENTS_KEY in bundle_payload:
        expected_keys.add(_ATTACHMENTS_KEY)
    missing = sorted(key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in bundle_payload)
    unexpected = sorted(key for key in bundle_payload if key not in expected_keys)
    if missing:
        failures.append(f"bundle.json is missing required top-level fields: {', '.join(missing)}.")
    if unexpected:
        failures.append(f"bundle.json contains unsupported top-level fields: {', '.join(unexpected)}.")
    if bundle_payload.get("bundle_type") != GUARDIAN_BUNDLE_TYPE:
        failures.append(f"bundle.json bundle_type must be {GUARDIAN_BUNDLE_TYPE!r}.")
    if bundle_payload.get("bundle_schema_version") != GUARDIAN_BUNDLE_SCHEMA_VERSION:
        failures.append(f"bundle.json bundle_schema_version must be {GUARDIAN_BUNDLE_SCHEMA_VERSION!r}.")
    if member_bytes.get(_BUNDLE_MANIFEST_NAME) != _json_bytes(bundle_payload):
        failures.append("bundle.json is not canonically encoded.")

    integrity_manifest = bundle_payload.get("integrity_manifest")
    if not isinstance(integrity_manifest, Mapping):
        failures.append("bundle.json integrity_manifest must be an object.")
        return failures
    declared_member_order = _declared_member_order(bundle_payload)
    if not declared_member_order:
        failures.append("bundle.json integrity_manifest.member_order must be a non-empty list.")
        return failures
    if declared_member_order[0] != _BUNDLE_MANIFEST_NAME:
        failures.append("bundle.json integrity_manifest.member_order must start with bundle.json.")
    if source_kind == "archive" and list(member_names) != declared_member_order:
        failures.append("Archive member order does not match bundle.json integrity_manifest.member_order.")
    if source_kind == "directory" and set(member_names) != set(declared_member_order):
        failures.append("Extracted bundle members do not match bundle.json integrity_manifest.member_order.")

    declared_member_set = set(declared_member_order)
    for member_name in declared_member_order:
        if member_name not in member_bytes:
            failures.append(f"Bundle is missing declared member {member_name!r}.")
    undeclared = sorted(name for name in member_names if name not in declared_member_set)
    if undeclared:
        failures.append(f"Bundle contains undeclared members: {', '.join(undeclared)}.")

    if integrity_manifest.get("proof_critical_members") != list(_PROOF_CRITICAL_MEMBERS):
        failures.append("bundle.json integrity_manifest.proof_critical_members does not match Guardian Bundle V1.")
    attachments = _normalise_attachments(bundle_payload.get(_ATTACHMENTS_KEY, []))
    attachment_members = {str(item["member"]) for item in attachments}
    if attachment_members.intersection(_PROOF_CRITICAL_MEMBERS):
        failures.append("bundle.json attachments must never be proof-critical in V1.")
    for attachment_member in attachment_members:
        if attachment_member not in declared_member_set:
            failures.append(f"bundle.json attachments references undeclared member {attachment_member!r}.")

    if bundle_payload.get("release_artifacts") != _build_release_artifacts():
        failures.append("bundle.json release_artifacts does not match the canonical artifact descriptor table.")
    replay_inputs = bundle_payload.get("replay_inputs")
    if isinstance(replay_inputs, Mapping):
        if replay_inputs.get("input_member") != _INPUT_STATE_NAME:
            failures.append("bundle.json replay_inputs.input_member must be input_state.json.")
        if replay_inputs.get("transparency_prefix_member") != _TRANSPARENCY_PREFIX_NAME:
            failures.append("bundle.json replay_inputs.transparency_prefix_member must be transparency-prefix.jsonl.")

    transparency_anchor = bundle_payload.get("transparency_anchor")
    if not isinstance(transparency_anchor, Mapping):
        failures.append("bundle.json transparency_anchor must be an object.")
    elif transparency_anchor.get("status") not in {"anchored", "not_anchored"}:
        failures.append("bundle.json transparency_anchor.status must be 'anchored' or 'not_anchored'.")

    actual_member_hashes = _member_hash_entries(
        bundle_payload=bundle_payload,
        member_bytes=member_bytes,
        member_order=declared_member_order,
    )
    if integrity_manifest.get("member_hashes") != actual_member_hashes:
        failures.append("bundle.json integrity_manifest.member_hashes does not match the declared bundle members.")
    if integrity_manifest.get("bundle_descriptor_hash") != _bundle_descriptor_hash(bundle_payload):
        failures.append("bundle.json integrity_manifest.bundle_descriptor_hash does not match the redacted bundle descriptor.")
    if integrity_manifest.get("bundle_hash") != hash_canonical_json(actual_member_hashes):
        failures.append("bundle.json integrity_manifest.bundle_hash does not match the ordered member hash list.")

    transparency_prefix_bytes = member_bytes.get(_TRANSPARENCY_PREFIX_NAME, b"")
    try:
        prefix_entries = _parse_transparency_entries(transparency_prefix_bytes)
    except ValueError as exc:
        failures.append(str(exc))
    else:
        chain_error = _verify_transparency_entries(prefix_entries)
        if chain_error is not None:
            failures.append(chain_error)
        elif transparency_prefix_bytes != _serialise_transparency_entries(prefix_entries):
            failures.append("transparency-prefix.jsonl is not canonically encoded.")

    for member_name, payload in payloads.items():
        if member_name == _BUNDLE_MANIFEST_NAME:
            continue
        if member_name.endswith(".json") and member_bytes.get(member_name) != _json_bytes(payload):
            failures.append(f"{member_name} is not canonically encoded.")
    return failures


def _verify_deterministic_replay(
    *,
    input_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    expected_report: Mapping[str, Any],
    expected_decision_journal: Mapping[str, Any],
    expected_evidence_payload: Mapping[str, Any],
    bundle_payload: Mapping[str, Any],
    transparency_prefix_bytes: bytes,
    receipt_entry: Mapping[str, Any] | None,
    scratch_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    replayed_bundle_sha256 = None
    if receipt_entry is None:
        failures.append("deterministic replay requires an anchored transparency receipt.")
        return {"verified": False, "failures": failures, "details": {"replayed_bundle_sha256": replayed_bundle_sha256}}

    replay_inputs = bundle_payload["replay_inputs"]
    expected_replay_inputs = _build_replay_inputs(
        state=expected_state,
        report=expected_report,
        evidence_payload=expected_evidence_payload,
    )
    if replay_inputs != expected_replay_inputs:
        failures.append("bundle.json replay_inputs does not match the deterministic replay seed surface.")
    failures.extend(
        _verify_runtime_profile_registry(
            replay_inputs=replay_inputs,
            expected_replay_inputs=expected_replay_inputs,
        )
    )
    if failures:
        return {"verified": False, "failures": failures, "details": {"replayed_bundle_sha256": replayed_bundle_sha256}}

    temp_root = scratch_root / f".guardian_bundle_replay_{_sha256_hex(_json_bytes(replay_inputs))[:12]}"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        temp_log_path = temp_root / _TRANSPARENCY_PREFIX_NAME
        temp_log_path.write_bytes(transparency_prefix_bytes)
        harness = MriGuardianHarness(
            transparency_log_path=temp_log_path,
            default_profile_name=str(replay_inputs.get("profile_name", replay_inputs.get("default_profile_name", ""))),
        )
        if input_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
            replay_result = harness.release_multi_series_case(
                state=input_state,
                actor=str(replay_inputs["actor"]),
                authority_sources_by_series_id=_group_authority_sources_by_series_id(list(replay_inputs["authority_sources"])),
                timestamp=str(replay_inputs["timestamp"]),
                transition_id=str(replay_inputs["transition_id"]),
            )
        else:
            replay_result = harness.release_case(
                state=input_state,
                actor=str(replay_inputs["actor"]),
                authority_sources=list(replay_inputs["authority_sources"]),
                profile_name=str(replay_inputs["profile_name"]),
                timestamp=str(replay_inputs["timestamp"]),
                transition_id=str(replay_inputs["transition_id"]),
            )
        replay_evidence_payload = replay_result.evidence.as_dict()
        if replay_result.after_state != expected_state:
            failures.append("Deterministic replay did not reproduce the released_state.json payload.")
        if replay_evidence_payload != expected_evidence_payload:
            failures.append("Deterministic replay did not reproduce the evidence.json payload.")
        replay_report = build_case_verification_report(replay_result.after_state)
        if replay_report != expected_report:
            failures.append("Deterministic replay did not reproduce the verification_report.json payload.")
        replay_decision_journal = build_decision_journal(
            replay_report,
            transparency_receipt_context=_minimal_transparency_receipt_context(
                evidence_payload=replay_evidence_payload,
                receipt_entry=_resolve_receipt_entry(
                    evidence_payload=replay_evidence_payload,
                    anchored_evidence_payload=_transparency_log_payload_from_mapping(replay_evidence_payload),
                    log_entries=_parse_transparency_entries(temp_log_path.read_bytes()),
                ),
            ),
        )
        if replay_decision_journal != expected_decision_journal:
            failures.append("Deterministic replay did not reproduce the decision_journal.json payload.")
        replay_bundle_path = temp_root / f"{expected_state['case_id']}{GUARDIAN_BUNDLE_EXTENSION}"
        replay_metadata = write_portable_evidence_bundle(
            result=replay_result,
            transparency_log_path=temp_log_path,
            output_path=replay_bundle_path,
        )
        replayed_bundle_sha256 = _sha256_hex(replay_bundle_path.read_bytes())
        if replay_metadata["bundle"]["claim"] != bundle_payload["claim"]:
            failures.append("Deterministic replay did not reproduce the bundle.json claim surface.")
        if replay_metadata["bundle"]["proof_context"] != bundle_payload["proof_context"]:
            failures.append("Deterministic replay did not reproduce the bundle.json proof_context surface.")
        if replay_metadata["bundle"]["replay_inputs"] != bundle_payload["replay_inputs"]:
            failures.append("Deterministic replay did not reproduce the bundle.json replay_inputs surface.")
        if replay_metadata["bundle"]["release_artifacts"] != bundle_payload["release_artifacts"]:
            failures.append("Deterministic replay did not reproduce the bundle.json release_artifacts surface.")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return {"verified": not failures, "failures": failures, "details": {"replayed_bundle_sha256": replayed_bundle_sha256}}


def _verify_runtime_profile_registry(
    *,
    replay_inputs: Mapping[str, Any],
    expected_replay_inputs: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    contracts = replay_inputs["contracts"]
    if "profile_name" in replay_inputs:
        profile_name = str(replay_inputs["profile_name"])
        runtime_profile_payload = get_verification_profile(profile_name).to_dict()
        if contracts["verification_profile_descriptor"] != runtime_profile_payload:
            failures.append("embedded_profile_descriptor_mismatch: bundled profile descriptor does not match the runtime registry.")
        if runtime_profile_payload != expected_replay_inputs["contracts"]["verification_profile_descriptor"]:
            failures.append("embedded_profile_digest_mismatch: bundled profile descriptor no longer matches the runtime registry.")
        return failures
    for item in contracts.get("series_contracts", []):
        profile_descriptor = item["verification_profile_descriptor"]
        profile_name = str(profile_descriptor["profile_name"])
        runtime_profile_payload = get_verification_profile(profile_name).to_dict()
        if profile_descriptor != runtime_profile_payload:
            failures.append(
                f"embedded_profile_descriptor_mismatch: series {item['series_id']!r} profile descriptor does not match the runtime registry."
            )
    return failures


def _load_bundle_source(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _read_extracted_bundle_members(path)
    return _read_bundle_archive_members(path)


def _read_bundle_archive_members(path: Path) -> dict[str, Any]:
    member_names: list[str] = []
    member_bytes: dict[str, bytes] = {}
    payloads: dict[str, Any] = {}
    with ZipFile(path, mode="r") as archive:
        for info in archive.infolist():
            member_names.append(info.filename)
            payload = archive.read(info.filename)
            member_bytes[info.filename] = payload
            if info.filename.endswith(".json"):
                payloads[info.filename] = json.loads(payload.decode("utf-8"))
    if _BUNDLE_MANIFEST_NAME not in payloads:
        raise ValueError("Bundle archive is missing bundle.json.")
    return {
        "source_kind": "archive",
        "member_names": member_names,
        "member_bytes": member_bytes,
        "payloads": payloads,
        "bundle_payload": payloads[_BUNDLE_MANIFEST_NAME],
    }


def _read_extracted_bundle_members(bundle_json_path: Path) -> dict[str, Any]:
    if bundle_json_path.name != _BUNDLE_MANIFEST_NAME:
        raise ValueError(f"verify_bundle expects {_BUNDLE_MANIFEST_NAME!r} when pointed at an extracted JSON bundle.")
    root = bundle_json_path.parent
    bundle_payload = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    member_names = sorted(path.name for path in root.iterdir() if path.is_file())
    member_bytes: dict[str, bytes] = {}
    payloads: dict[str, Any] = {_BUNDLE_MANIFEST_NAME: bundle_payload}
    for member_name in member_names:
        payload = (root / member_name).read_bytes()
        member_bytes[member_name] = payload
        if member_name.endswith(".json") and member_name != _BUNDLE_MANIFEST_NAME:
            payloads[member_name] = json.loads(payload.decode("utf-8"))
    return {
        "source_kind": "directory",
        "member_names": member_names,
        "member_bytes": member_bytes,
        "payloads": payloads,
        "bundle_payload": bundle_payload,
    }


def _is_minimal_extracted_bundle_root(root: Path) -> bool:
    """Return whether ``root`` has the direct-child layout of an extracted Guardian Bundle V1."""
    if not root.is_dir():
        return False
    required_members = {_BUNDLE_MANIFEST_NAME, *_PROOF_CRITICAL_MEMBERS}
    present_members = {path.name for path in root.iterdir() if path.is_file()}
    return required_members.issubset(present_members)


def _declared_member_order(bundle_payload: Mapping[str, Any]) -> list[str]:
    member_order = bundle_payload.get("integrity_manifest", {}).get("member_order", [])
    if not isinstance(member_order, list):
        return []
    names: list[str] = []
    for item in member_order:
        if type(item) is not str or item == "":
            return []
        names.append(item)
    return names


def _member_hash_entries(
    *,
    bundle_payload: Mapping[str, Any],
    member_bytes: Mapping[str, bytes],
    member_order: Sequence[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for member_name in member_order:
        value = _bundle_descriptor_hash(bundle_payload) if member_name == _BUNDLE_MANIFEST_NAME else _sha256_hex(member_bytes.get(member_name, b""))
        entries.append({"member": member_name, "algorithm": "sha256", "value": value})
    return entries


def _bundle_descriptor_hash(bundle_payload: Mapping[str, Any]) -> str:
    redacted = deepcopy(dict(bundle_payload))
    redacted["integrity_manifest"] = _BUNDLE_REDACTION_SENTINEL
    return hash_canonical_json(redacted)


def _single_series_temporal_context_present(contracts: Mapping[str, Any]) -> bool:
    return contracts.get("prior_verified_context") is not None or contracts.get("prior_verified_contexts") is not None


def _require_member_payload(payloads: Mapping[str, Any], member_name: str) -> dict[str, Any]:
    payload = payloads.get(member_name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Bundle member {member_name!r} is missing or is not a JSON object.")
    return dict(payload)


def _normalise_attachments(attachments: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if attachments is None:
        return []
    payload: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for index, item in enumerate(attachments):
        if not isinstance(item, Mapping):
            raise ValueError(f"attachments[{index}] must be an object.")
        member = item.get("member")
        media_type = item.get("media_type")
        if type(member) is not str or member == "":
            raise ValueError(f"attachments[{index}].member must be a non-empty string.")
        if type(media_type) is not str or media_type == "":
            raise ValueError(f"attachments[{index}].media_type must be a non-empty string.")
        if member in seen_members:
            raise ValueError(f"attachments contains duplicate member {member!r}.")
        payload.append({"member": member, "media_type": media_type})
        seen_members.add(member)
    return payload


def _minimal_transparency_receipt_context(
    *,
    evidence_payload: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
) -> dict[str, Any]:
    return build_transparency_receipt_context(
        reference=_transparency_receipt(evidence_payload)["reference"],
        entry_hash=str(receipt_entry["entry_hash"]),
        previous_entry_hash=receipt_entry.get("previous_entry_hash"),
        checkpoint=str(receipt_entry["entry_hash"]),
        entry_index=int(receipt_entry["index"]),
        log_length=int(receipt_entry["index"]) + 1,
    )


def _receipt_entry_from_bundle(bundle_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    transparency_anchor = bundle_payload.get("transparency_anchor")
    if not isinstance(transparency_anchor, Mapping):
        return None
    if transparency_anchor.get("status") != "anchored":
        return None
    if type(transparency_anchor.get("entry_hash")) is not str:
        return None
    if type(transparency_anchor.get("entry_index")) is not int:
        return None
    previous_entry_hash = transparency_anchor.get("previous_entry_hash")
    if previous_entry_hash is not None and type(previous_entry_hash) is not str:
        return None
    return {
        "entry_hash": transparency_anchor["entry_hash"],
        "index": int(transparency_anchor["entry_index"]),
        "previous_entry_hash": previous_entry_hash,
    }


def _verify_transparency_anchor_against_prefix(
    *,
    evidence_payload: Mapping[str, Any],
    anchored_evidence_payload: Mapping[str, Any],
    transparency_prefix_entries: list[dict[str, Any]],
    receipt_entry: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    prefix_chain_error = _verify_transparency_entries(transparency_prefix_entries)
    if prefix_chain_error is not None:
        failures.append(prefix_chain_error)
        return failures
    expected_previous = None if not transparency_prefix_entries else transparency_prefix_entries[-1]["entry_hash"]
    if int(receipt_entry["index"]) != len(transparency_prefix_entries):
        failures.append("bundle.json transparency_anchor.entry_index does not match transparency-prefix.jsonl length.")
    if receipt_entry.get("previous_entry_hash") != expected_previous:
        failures.append("bundle.json transparency_anchor.previous_entry_hash does not match transparency-prefix.jsonl.")
    if _transparency_receipt(evidence_payload)["reference"] != _transparency_reference(receipt_entry):
        failures.append("bundle.json transparency_anchor does not match evidence.json provenance receipt.")
    candidate = {
        "index": int(receipt_entry["index"]),
        "previous_entry_hash": receipt_entry.get("previous_entry_hash"),
        "evidence_hash": hash_canonical_json(anchored_evidence_payload),
    }
    if receipt_entry["entry_hash"] != hash_canonical_json(candidate):
        failures.append("bundle.json transparency_anchor.entry_hash does not match the canonical anchored evidence hash.")
    return failures


def _ensure_released_state(state: Mapping[str, Any]) -> None:
    if state.get("release_status") != RELEASE_STATUS_RELEASED:
        raise ValueError("Guardian bundles require a released guardian state.")


def _group_authority_sources_by_series_id(authority_sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in authority_sources:
        details = dict(source.get("details", {}))
        series_id = details.get("series_id")
        if not isinstance(series_id, str) or not series_id:
            raise ValueError("Multi-series authority source is missing details.series_id.")
        grouped.setdefault(series_id, []).append(dict(source))
    return grouped


def _read_transparency_log_snapshot(path: Path | str) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.exists():
        raise ValueError(f"Transparency log snapshot not found: {log_path}")
    payload = log_path.read_bytes()
    entries = _parse_transparency_entries(payload)
    chain_error = _verify_transparency_entries(entries)
    if chain_error is not None:
        raise ValueError(chain_error)
    return {"bytes": payload, "entries": entries}


def _parse_transparency_entries(payload: bytes) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError(f"Transparency log line {line_number} is not a JSON object.")
        entries.append(entry)
    return entries


def _verify_transparency_entries(entries: list[dict[str, Any]]) -> str | None:
    previous_entry_hash = None
    for expected_index, entry in enumerate(entries):
        if entry.get("index") != expected_index:
            return "Bundled transparency prefix index sequence is not contiguous."
        if entry.get("previous_entry_hash") != previous_entry_hash:
            return "Bundled transparency prefix previous_entry_hash chain is broken."
        candidate = {
            "index": entry.get("index"),
            "previous_entry_hash": entry.get("previous_entry_hash"),
            "evidence_hash": entry.get("evidence_hash"),
        }
        expected_entry_hash = hash_canonical_json(candidate)
        if entry.get("entry_hash") != expected_entry_hash:
            return "Bundled transparency prefix entry_hash does not match canonical contents."
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
            raise ValueError("Transparency receipt reference exists, but evidence_hash does not match.")
        return entry
    raise ValueError(f"Transparency receipt {expected_reference!r} is not present in the transparency log.")


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
    return f"{'\n'.join(canonical_json_dumps(entry) for entry in entries)}\n".encode("utf-8")


def _json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return f"{canonical_json_dumps(payload)}\n".encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle_member(archive: ZipFile, member_name: str, payload: bytes) -> None:
    info = ZipInfo(filename=member_name, date_time=_FIXED_ZIP_TIMESTAMP)
    info.compress_type = ZIP_STORED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)
