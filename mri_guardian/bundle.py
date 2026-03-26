"""Portable MRI guard bundle builder and offline verifier."""

from __future__ import annotations

import hashlib
import json
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
    _check_released_state_matches_trace,
    build_case_verification_report,
)

MRI_GUARD_BUNDLE_EXTENSION = ".mri-guard-bundle"
_BUNDLE_SCHEMA_VERSION = "0.1"
_BUNDLE_MEDIA_TYPE = "application/vnd.mri-guard.bundle+zip"
_FIXED_ZIP_TIMESTAMP = (2026, 3, 26, 0, 0, 0)
_REQUIRED_BUNDLE_MEMBERS = (
    "manifest.json",
    "signatures.json",
    "evidence.json",
    "released_state.json",
    "workstation_payload.json",
    "verification_report.json",
    "transcript.json",
    "transparency-log.jsonl",
)


def write_portable_evidence_bundle(
    *,
    result: TransitionResult,
    transparency_log_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Write a deterministic .mri-guard-bundle archive and return its metadata."""
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
    manifest = _build_bundle_manifest(
        state=state,
        report=report,
        evidence_payload=evidence_payload,
        anchored_evidence_payload=anchored_evidence_payload,
        transcript=transcript,
        log_snapshot=log_snapshot,
        receipt_entry=receipt_entry,
    )
    members = {
        "manifest.json": _json_bytes(manifest),
        "evidence.json": _json_bytes(evidence_payload),
        "released_state.json": _json_bytes(state),
        "workstation_payload.json": _json_bytes(state["workstation_payload"]),
        "verification_report.json": _json_bytes(report),
        "transcript.json": _json_bytes(transcript),
        "transparency-log.jsonl": log_snapshot["bytes"],
    }
    signatures = _build_signature_catalog(members)
    members["signatures.json"] = _json_bytes(signatures)

    bundle_path = Path(output_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, mode="w") as archive:
        for member_name in _REQUIRED_BUNDLE_MEMBERS:
            _write_bundle_member(archive, member_name, members[member_name])

    return {
        "path": str(bundle_path),
        "manifest": manifest,
        "signatures": signatures,
    }


def verify_portable_evidence_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Verify a portable .mri-guard-bundle without referring back to the running harness."""
    path = Path(bundle_path)
    failures: list[str] = []
    try:
        payloads, member_bytes = _read_bundle_members(path)
        evidence_payload = payloads["evidence.json"]
        state = payloads["released_state.json"]
        report = payloads["verification_report.json"]
        transcript = payloads["transcript.json"]
        manifest = payloads["manifest.json"]
        signatures = payloads["signatures.json"]
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
            "bytes": member_bytes["transparency-log.jsonl"],
            "entries": _parse_transparency_entries(member_bytes["transparency-log.jsonl"]),
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
            expected_manifest = _build_bundle_manifest(
                state=state,
                report=expected_report,
                evidence_payload=evidence_payload,
                anchored_evidence_payload=anchored_evidence_payload,
                transcript=expected_transcript,
                log_snapshot=log_snapshot,
                receipt_entry=receipt_entry,
            )
            if manifest != expected_manifest:
                failures.append("manifest.json does not match the deterministic bundle manifest.")

        expected_signatures = _build_signature_catalog(
            {
                name: member_bytes[name]
                for name in _REQUIRED_BUNDLE_MEMBERS
                if name != "signatures.json"
            }
        )
        if signatures != expected_signatures:
            failures.append("signatures.json does not match the bundled member contents.")
    except Exception as exc:
        return {
            "verified": False,
            "bundle_path": str(path),
            "failures": [str(exc)],
            "details": {},
        }

    return {
        "verified": not failures,
        "bundle_path": str(path),
        "failures": failures,
        "details": {
            "case_id": state.get("case_id"),
            "bundle_type": manifest.get("bundle_type"),
            "receipt_reference": manifest.get("transparency", {}).get("receipt_reference"),
            "evidence_hash": manifest.get("hashes", {}).get("evidence_hash"),
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
            "verified_manifest_hash": hash_canonical_json(state["verified_manifest"]),
            "workstation_payload_hash": hash_canonical_json(state["workstation_payload"]),
            "released_state_hash": hash_canonical_json(state),
            "evidence_hash": hash_canonical_json(evidence_payload),
            "anchored_evidence_hash": hash_canonical_json(anchored_evidence_payload),
            "verification_report_hash": hash_canonical_json(report),
            "transcript_hash": hash_canonical_json(transcript),
            "transparency_log_hash": _sha256_hex(log_snapshot["bytes"]),
        },
        "verification": {
            "release_eligible": bool(report["release_eligible"]),
            "failure_reasons": report["failure_reasons"],
            "tiers": {
                key: value["status"] for key, value in report["verification_tiers"].items()
            },
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
        "signatures_member": "signatures.json",
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
                "summary": "Recomputed the workstation reconstruction directly from raw k-space.",
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
                "summary": "Released only the raw-derived reconstruction and verified mask to the radiologist boundary.",
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


def _read_bundle_members(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    member_bytes: dict[str, bytes] = {}
    payloads: dict[str, Any] = {}
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
    return payloads, member_bytes


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
