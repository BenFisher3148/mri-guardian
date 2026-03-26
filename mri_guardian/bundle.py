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
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    _check_released_state_matches_trace,
    build_case_verification_report,
)

MRI_GUARD_BUNDLE_EXTENSION = ".mri-guard-bundle"
_BUNDLE_SCHEMA_VERSION = "0.2"
_BUNDLE_MEDIA_TYPE = "application/vnd.mri-guard.bundle+zip"
_FIXED_ZIP_TIMESTAMP = (2026, 3, 26, 0, 0, 0)
_BUNDLE_MANIFEST_NAME = "bundle.json"
_SIGNATURES_NAME = "signatures.json"
_EVIDENCE_NAME = "evidence.json"
_INPUT_STATE_NAME = "input_state.json"
_RELEASED_STATE_NAME = "released_state.json"
_WORKSTATION_PAYLOAD_NAME = "workstation_payload.json"
_VERIFICATION_REPORT_NAME = "verification_report.json"
_TRANSCRIPT_NAME = "transcript.json"
_REPLAY_DESCRIPTOR_NAME = "replay.json"
_TRANSPARENCY_LOG_NAME = "transparency-log.jsonl"
_REQUIRED_BUNDLE_MEMBERS = (
    _BUNDLE_MANIFEST_NAME,
    _SIGNATURES_NAME,
    _EVIDENCE_NAME,
    _INPUT_STATE_NAME,
    _RELEASED_STATE_NAME,
    _WORKSTATION_PAYLOAD_NAME,
    _VERIFICATION_REPORT_NAME,
    _TRANSCRIPT_NAME,
    _REPLAY_DESCRIPTOR_NAME,
    _TRANSPARENCY_LOG_NAME,
)


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
        _TRANSCRIPT_NAME: _json_bytes(transcript),
        _REPLAY_DESCRIPTOR_NAME: _json_bytes(replay_descriptor),
        _TRANSPARENCY_LOG_NAME: log_snapshot["bytes"],
    }
    signatures = _build_signature_catalog(members)
    members[_SIGNATURES_NAME] = _json_bytes(signatures)

    bundle_path = Path(output_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, mode="w") as archive:
        for member_name in _REQUIRED_BUNDLE_MEMBERS:
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
        transcript = payloads[_TRANSCRIPT_NAME]
        replay_descriptor = payloads[_REPLAY_DESCRIPTOR_NAME]
        signatures = payloads[_SIGNATURES_NAME]
        anchored_evidence_payload = _transparency_log_payload_from_mapping(evidence_payload)

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
        if not expected_report["release_eligible"]:
            failures.append("released_state.json is not release-eligible.")

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
                for name in _REQUIRED_BUNDLE_MEMBERS
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
    analysis = report["analysis"]
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "bundle_type": "mri_guard_bundle",
        "bundle_media_type": _BUNDLE_MEDIA_TYPE,
        "case_id": state["case_id"],
        "entity_type": evidence_payload["entity_type"],
        "entity_id": evidence_payload["entity_id"],
        "generated_at": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "verdict": "This scan is verifiably correct.",
        "hashes": {
            "raw_kspace_hash": analysis["raw_hash"],
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
        },
        "verification": {
            "release_eligible": bool(report["release_eligible"]),
            "failure_reasons": report["failure_reasons"],
            "tiers": {
                key: value["status"] for key, value in report["verification_tiers"].items()
            },
        },
        "deterministic_replay": {
            "supported": True,
            "mode": replay_descriptor["replay_mode"],
            "input_member": _INPUT_STATE_NAME,
            "replay_member": _REPLAY_DESCRIPTOR_NAME,
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


def _build_bundle_transcript(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    receipt_entry: Mapping[str, Any],
    checkpoint: str | None,
) -> dict[str, Any]:
    analysis = report["analysis"]
    manifest = report["manifest"]
    tiers = report["verification_tiers"]
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "transcript_type": "portable_mri_guard_transcript",
        "case_id": state["case_id"],
        "timestamp": evidence_payload["timestamp"],
        "actor": evidence_payload["actor"],
        "statement": "This scan is verifiably correct.",
        "events": [
            {
                "stage": "capture",
                "summary": "Captured the raw MRI measurements and fixed their canonical hash.",
                "details": {
                    "raw_kspace_hash": analysis["raw_hash"],
                    "voxel_spacing_mm": state["voxel_spacing_mm"],
                },
            },
            {
                "stage": "reconstruction",
                "summary": "Recomputed the Guardian Layer reconstruction directly from raw k-space.",
                "details": {
                    "reconstruction_hash": analysis["derived_reconstruction_hash"],
                    "forward_projection_hash": analysis["forward_projection_hash"],
                    "max_measurement_abs_error": analysis["max_measurement_abs_error"],
                    "max_imaginary_leakage": analysis["max_imaginary_leakage"],
                },
            },
            {
                "stage": "mask_and_volume",
                "summary": "Checked the AI tumour claim against the deterministic verified mask and voxel geometry.",
                "details": {
                    "ai_mask_hash": analysis["ai_mask_hash"],
                    "verified_mask_hash": analysis["verified_mask_hash"],
                    "mask_voxel_delta": analysis["mask_voxel_delta"],
                    "claimed_tumor_volume_mm3": manifest["tumor_volume_proof"]["claimed_mm3"],
                    "verified_tumor_volume_mm3": manifest["tumor_volume_proof"]["computed_mm3"],
                },
            },
            {
                "stage": "verification_tiers",
                "summary": "Applied Level 1 structural validity, Level 2 physical consistency, and Level 3 clinical plausibility.",
                "details": {
                    key: {
                        "title": tier["title"],
                        "status": tier["status"],
                        "summary": tier["summary"],
                    }
                    for key, tier in tiers.items()
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
        "authority_sources": evidence_payload["authority_sources"],
        "transparency_prefix_entry_count": receipt_entry["index"],
        "expected_hashes": {
            "input_state_hash": hash_canonical_json(before_state),
            "released_state_hash": hash_canonical_json(state),
            "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
            "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
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
        for member_name in _REQUIRED_BUNDLE_MEMBERS:
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
    for member_name in _REQUIRED_BUNDLE_MEMBERS:
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
        temp_log_path = temp_root / _TRANSPARENCY_LOG_NAME
        temp_log_path.write_bytes(prefix_bytes)
        harness = MriGuardianHarness(transparency_log_path=temp_log_path)
        replay_result = harness.release_case(
            state=input_state,
            actor=str(replay_descriptor["actor"]),
            authority_sources=replay_descriptor["authority_sources"],
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
