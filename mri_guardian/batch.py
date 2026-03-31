"""Batch orchestration and queue semantics for the MRI Guardian Layer."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NotRequired, TypedDict
from uuid import UUID, uuid5

from icsd.core.errors import InvariantViolationError
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json

from .bundle import GUARDIAN_BUNDLE_EXTENSION, write_portable_evidence_bundle
from .contracts import build_verification_profile_contract
from .harness import (
    DEFAULT_PROFILE_NAME,
    EVALUATION_STAGE_INTAKE_VALIDATION,
    EVALUATION_STAGE_STRUCTURAL_VALIDITY,
    MULTI_SERIES_CASE_KIND,
    MriGuardianHarness,
    build_case_verification_report,
    render_release_bundle_json,
    verification_profile_descriptor_from_state,
)
from .journal import build_decision_journal
from .profiles import get_verification_profile

BATCH_SCHEMA_VERSION = "1"
BATCH_SUMMARY_SCHEMA_VERSION = "1"
BATCH_ITEM_RESULT_SCHEMA_VERSION = "1"
BATCH_ATTEMPT_RECORD_SCHEMA_VERSION = "1"
BATCH_VALIDATION_SCHEMA_VERSION = "1"

QUEUE_BUCKET_QUEUED = "queued"
QUEUE_BUCKET_PROCESSING = "processing"
QUEUE_BUCKET_COMPLETED = "completed"
QUEUE_BUCKET_QUARANTINE = "quarantine"
_QUEUE_BUCKETS = (
    QUEUE_BUCKET_QUEUED,
    QUEUE_BUCKET_PROCESSING,
    QUEUE_BUCKET_COMPLETED,
    QUEUE_BUCKET_QUARANTINE,
)

OUTCOME_RELEASED = "released"
OUTCOME_VERIFICATION_BLOCK = "verification_block"
OUTCOME_POLICY_BLOCK = "policy_block"
OUTCOME_PACKAGE_INVALID = "package_invalid"
OUTCOME_TRANSIENT_FAILURE_EXHAUSTED = "transient_failure_exhausted"
OUTCOME_WORKER_ERROR = "worker_error"
_FINAL_OUTCOME_ORDER = (
    OUTCOME_RELEASED,
    OUTCOME_VERIFICATION_BLOCK,
    OUTCOME_POLICY_BLOCK,
    OUTCOME_PACKAGE_INVALID,
    OUTCOME_TRANSIENT_FAILURE_EXHAUSTED,
    OUTCOME_WORKER_ERROR,
)
_RETRYABLE_ATTEMPT_OUTCOME = "retryable_failure"
_RETRYABLE_EXCEPTION_TYPES = (
    PermissionError,
    TimeoutError,
    BlockingIOError,
)
_PACKAGE_INVALID_PIPELINE_PHASES = frozenset(
    {
        EVALUATION_STAGE_INTAKE_VALIDATION,
        EVALUATION_STAGE_STRUCTURAL_VALIDITY,
    }
)

_BATCH_NAMESPACE = UUID("7a9d5d98-4ac7-4bde-9f08-d6d0dfb63fea")
_CASE_PACKAGE_INPUT_NAME = "input_state.json"
_SINGLE_SERIES_AUTHORITIES_NAME = "authority_sources.json"
_MULTI_SERIES_AUTHORITIES_NAME = "authority_sources_by_series_id.json"


class BatchQueueItem(TypedDict):
    """One queued case package."""

    item_id: str
    case_dir: str
    profile_override: NotRequired[str]


class BatchManifest(TypedDict):
    """Batch manifest schema."""

    batch_schema_version: str
    batch_id: str
    max_attempts: NotRequired[int]
    allowed_profile_overrides: NotRequired[list[str]]
    items: list[BatchQueueItem]


class BatchPolicySelection(TypedDict):
    """Resolved policy surface for one validated case."""

    mode: str
    profile_override: str | None
    override_applied: bool
    selected_profile_name: NotRequired[str]
    selected_profile_revision: NotRequired[str]
    selected_profile_digest: NotRequired[str]
    series_profiles: NotRequired[list[dict[str, str]]]


class BatchPackageValidation(TypedDict):
    """Pre-execution case-package validation record."""

    schema_version: str
    started_at: str
    completed_at: str
    valid: bool
    ready_for_execution: bool
    failures: list[str]
    normalized_input_digest: NotRequired[str]
    normalized_input_state_path: NotRequired[str]
    normalized_authority_path: NotRequired[str]
    execution_actor: NotRequired[str]
    execution_timestamp: NotRequired[str]
    transition_id: NotRequired[str]


class BatchAttemptRecord(TypedDict):
    """One execution attempt."""

    schema_version: str
    attempt_number: int
    started_at: str
    completed_at: str
    outcome: str
    retryable: bool
    exception_class: str | None
    artifact_root: str
    messages: list[str]


class BatchItemResult(TypedDict):
    """Final queue result for one item."""

    schema_version: str
    item_id: str
    manifest_index: int
    case_dir: str
    case_kind: str
    queue_path_name: str
    current_bucket: str
    final_bucket: str
    final_outcome: str
    entered_execution: bool
    profile_override: str | None
    override_applied: bool
    messages: list[str]
    package_validation: BatchPackageValidation
    resolved_policy: BatchPolicySelection | None
    attempts: list[BatchAttemptRecord]


class BatchSummary(TypedDict):
    """Final batch summary."""

    batch_summary_schema_version: str
    batch_id: str
    manifest_path: str | None
    output_root: str
    run_status: str
    started_at: str
    completed_at: str
    max_attempts: int
    allowed_profile_overrides: list[str]
    total_items: int
    released_items: int
    quarantined_items: int
    outcome_counts: dict[str, int]
    items: list[BatchItemResult]


@dataclass(frozen=True)
class _ValidatedPackage:
    item_id: str
    manifest_index: int
    case_dir: Path
    queue_path_name: str
    case_kind: str
    normalized_state: dict[str, Any]
    normalized_authority_name: str
    normalized_authority_payload: Any
    profile_override: str | None
    override_applied: bool
    normalized_input_digest: str
    execution_actor: str
    execution_timestamp: str
    transition_id: str


def run_batch(
    manifest: BatchManifest | Mapping[str, Any] | Path | str,
    output_dir: Path | str,
) -> BatchSummary:
    """Run one ordered Guardian batch and return the final summary."""
    manifest_payload, manifest_base_dir, manifest_path = _load_manifest_input(manifest)
    batch_manifest = _normalise_manifest_payload(
        manifest_payload,
        base_dir=manifest_base_dir,
    )
    output_root = _prepare_output_root(output_dir)
    _initialise_output_root(output_root)
    _write_json(output_root / "batch_manifest.json", batch_manifest)

    started_at = _utc_now()
    results: list[BatchItemResult] = []
    for manifest_index, item in enumerate(batch_manifest["items"]):
        queue_path_name = _queue_path_name(manifest_index, item["item_id"])
        queued_item_dir = output_root / QUEUE_BUCKET_QUEUED / queue_path_name
        queued_item_dir.mkdir(parents=True, exist_ok=False)
        _write_queue_item(
            queued_item_dir / "queue_item.json",
            manifest_index=manifest_index,
            queue_path_name=queue_path_name,
            item=item,
        )
        package_validation, validated_package, detected_case_kind = _validate_case_package(
            batch_id=batch_manifest["batch_id"],
            manifest_index=manifest_index,
            queue_path_name=queue_path_name,
            item=item,
            item_dir=queued_item_dir,
            allowed_profile_overrides=batch_manifest["allowed_profile_overrides"],
        )
        _write_json(queued_item_dir / "package_validation.json", package_validation)
        if validated_package is None:
            item_result = _package_invalid_result(
                item=item,
                manifest_index=manifest_index,
                queue_path_name=queue_path_name,
                case_kind=detected_case_kind,
                package_validation=package_validation,
            )
            final_item_dir = _move_item_directory(
                queued_item_dir,
                output_root / QUEUE_BUCKET_QUARANTINE / queue_path_name,
            )
            _write_json(final_item_dir / "queue_result.json", item_result)
            results.append(item_result)
            continue

        processing_item_dir = _move_item_directory(
            queued_item_dir,
            output_root / QUEUE_BUCKET_PROCESSING / queue_path_name,
        )
        item_result, final_bucket = _execute_validated_package(
            validated_package=validated_package,
            item=item,
            item_dir=processing_item_dir,
            max_attempts=batch_manifest["max_attempts"],
            package_validation=package_validation,
        )
        final_item_dir = _move_item_directory(
            processing_item_dir,
            output_root / final_bucket / queue_path_name,
        )
        _write_json(final_item_dir / "queue_result.json", item_result)
        results.append(item_result)

    completed_at = _utc_now()
    summary = _build_batch_summary(
        batch_id=batch_manifest["batch_id"],
        manifest_path=manifest_path,
        output_root=output_root,
        started_at=started_at,
        completed_at=completed_at,
        max_attempts=batch_manifest["max_attempts"],
        allowed_profile_overrides=batch_manifest["allowed_profile_overrides"],
        results=results,
    )
    _write_json(output_root / "batch_summary.json", summary)
    return summary


def _load_manifest_input(
    manifest: BatchManifest | Mapping[str, Any] | Path | str,
) -> tuple[Mapping[str, Any], Path, str | None]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Batch manifest must be a JSON object.")
        return dict(payload), manifest_path.parent, str(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping or a filesystem path.")
    return dict(manifest), Path.cwd(), None


def _normalise_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    base_dir: Path,
) -> BatchManifest:
    if manifest.get("batch_schema_version") != BATCH_SCHEMA_VERSION:
        raise ValueError(f"batch_schema_version must be {BATCH_SCHEMA_VERSION!r}.")
    batch_id = manifest.get("batch_id")
    if type(batch_id) is not str or not batch_id.strip():
        raise ValueError("batch_id must be a non-empty string.")
    max_attempts = manifest.get("max_attempts", 2)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be an integer >= 1.")
    allowed_raw = manifest.get("allowed_profile_overrides", [])
    if not isinstance(allowed_raw, list):
        raise ValueError("allowed_profile_overrides must be a list when present.")
    allowed_profile_overrides: list[str] = []
    for index, entry in enumerate(allowed_raw):
        if type(entry) is not str or not entry.strip():
            raise ValueError(
                f"allowed_profile_overrides[{index}] must be a non-empty string."
            )
        profile_name = entry.strip()
        get_verification_profile(profile_name)
        if profile_name in allowed_profile_overrides:
            raise ValueError(
                f"allowed_profile_overrides contains duplicate profile {profile_name!r}."
            )
        allowed_profile_overrides.append(profile_name)
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("items must be a non-empty list.")
    items: list[BatchQueueItem] = []
    seen_item_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"items[{index}] must be an object.")
        item_id = raw_item.get("item_id")
        if type(item_id) is not str or not item_id.strip():
            raise ValueError(f"items[{index}].item_id must be a non-empty string.")
        if item_id in seen_item_ids:
            raise ValueError(f"Duplicate item_id {item_id!r} is not allowed.")
        seen_item_ids.add(item_id)
        raw_case_dir = raw_item.get("case_dir")
        if type(raw_case_dir) is not str or not raw_case_dir.strip():
            raise ValueError(f"items[{index}].case_dir must be a non-empty string.")
        case_dir = _resolve_case_dir(base_dir=base_dir, raw_case_dir=raw_case_dir)
        item: BatchQueueItem = {
            "item_id": item_id,
            "case_dir": str(case_dir),
        }
        profile_override = raw_item.get("profile_override")
        if profile_override is not None:
            if type(profile_override) is not str or not profile_override.strip():
                raise ValueError(
                    f"items[{index}].profile_override must be a non-empty string when present."
                )
            item["profile_override"] = profile_override.strip()
        items.append(item)
    return {
        "batch_schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id.strip(),
        "max_attempts": max_attempts,
        "allowed_profile_overrides": allowed_profile_overrides,
        "items": items,
    }


def _resolve_case_dir(*, base_dir: Path, raw_case_dir: str) -> Path:
    case_dir = Path(raw_case_dir)
    if not case_dir.is_absolute():
        case_dir = base_dir / case_dir
    return case_dir.resolve()


def _prepare_output_root(output_dir: Path | str) -> Path:
    output_root = Path(output_dir)
    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing:
            raise ValueError(
                f"Batch output root must be absent or empty: {output_root}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    return output_root.resolve()


def _initialise_output_root(output_root: Path) -> None:
    for bucket in _QUEUE_BUCKETS:
        (output_root / bucket).mkdir(parents=True, exist_ok=False)


def _write_queue_item(
    path: Path,
    *,
    manifest_index: int,
    queue_path_name: str,
    item: BatchQueueItem,
) -> None:
    _write_json(
        path,
        {
            "item_id": item["item_id"],
            "manifest_index": manifest_index,
            "queue_path_name": queue_path_name,
            "case_dir": item["case_dir"],
            "profile_override": item.get("profile_override"),
        },
    )


def _validate_case_package(
    *,
    batch_id: str,
    manifest_index: int,
    queue_path_name: str,
    item: BatchQueueItem,
    item_dir: Path,
    allowed_profile_overrides: Sequence[str],
) -> tuple[BatchPackageValidation, _ValidatedPackage | None, str]:
    started_at = _utc_now()
    failures: list[str] = []
    package: _ValidatedPackage | None = None
    detected_case_kind = _case_kind_from_state_payload(None)
    case_dir = Path(item["case_dir"])
    try:
        normalized_state, normalized_authority_name, normalized_authority_payload = (
            _load_normalized_case_package(case_dir)
        )
        detected_case_kind = _case_kind_from_state_payload(normalized_state)
        profile_override = item.get("profile_override")
        override_applied = False
        if profile_override is not None:
            if normalized_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
                raise ValueError(
                    "profile_override is not supported for multi-series queue items in V1."
                )
            if profile_override not in allowed_profile_overrides:
                raise ValueError(
                    f"profile_override {profile_override!r} is not allow-listed for this batch."
                )
            get_verification_profile(profile_override)
            normalized_state["verification_profile_descriptor"] = (
                build_verification_profile_contract(profile_override)
            )
            override_applied = True
        _validate_authority_package(
            state=normalized_state,
            authority_name=normalized_authority_name,
            authority_payload=normalized_authority_payload,
        )
        package_validation_failure = _preflight_case_state_validation_failure(
            normalized_state
        )
        if package_validation_failure is not None:
            raise ValueError(package_validation_failure)
        normalized_input_digest = _hash_normalized_package(
            state=normalized_state,
            authority_name=normalized_authority_name,
            authority_payload=normalized_authority_payload,
        )
        execution_timestamp = _utc_now()
        execution_actor = f"guardian:batch:{batch_id}"
        transition_id = str(
            uuid5(
                _BATCH_NAMESPACE,
                f"{batch_id}:{item['item_id']}:{normalized_input_digest}",
            )
        )
        _write_json(item_dir / "normalized_input_state.json", normalized_state)
        _write_json(item_dir / normalized_authority_name, normalized_authority_payload)
        package = _ValidatedPackage(
            item_id=item["item_id"],
            manifest_index=manifest_index,
            case_dir=case_dir,
            queue_path_name=queue_path_name,
            case_kind=_case_kind_from_state_payload(normalized_state),
            normalized_state=normalized_state,
            normalized_authority_name=normalized_authority_name,
            normalized_authority_payload=normalized_authority_payload,
            profile_override=profile_override,
            override_applied=override_applied,
            normalized_input_digest=normalized_input_digest,
            execution_actor=execution_actor,
            execution_timestamp=execution_timestamp,
            transition_id=transition_id,
        )
    except ValueError as exc:
        failures.append(str(exc))
    completed_at = _utc_now()
    validation: BatchPackageValidation = {
        "schema_version": BATCH_VALIDATION_SCHEMA_VERSION,
        "started_at": started_at,
        "completed_at": completed_at,
        "valid": package is not None,
        "ready_for_execution": package is not None,
        "failures": failures,
    }
    if package is not None:
        validation["normalized_input_digest"] = package.normalized_input_digest
        validation["normalized_input_state_path"] = "normalized_input_state.json"
        validation["normalized_authority_path"] = package.normalized_authority_name
        validation["execution_actor"] = package.execution_actor
        validation["execution_timestamp"] = package.execution_timestamp
        validation["transition_id"] = package.transition_id
    return validation, package, detected_case_kind


def _load_normalized_case_package(
    case_dir: Path,
) -> tuple[dict[str, Any], str, Any]:
    if not case_dir.exists():
        raise ValueError(f"Case package directory not found: {case_dir}")
    if not case_dir.is_dir():
        raise ValueError(f"Case package path is not a directory: {case_dir}")
    input_path = case_dir / _CASE_PACKAGE_INPUT_NAME
    if not input_path.is_file():
        raise ValueError(
            f"Case package is missing required {_CASE_PACKAGE_INPUT_NAME!r}: {case_dir}"
        )
    authority_path = case_dir / _SINGLE_SERIES_AUTHORITIES_NAME
    authority_by_series_path = case_dir / _MULTI_SERIES_AUTHORITIES_NAME
    has_single = authority_path.is_file()
    has_multi = authority_by_series_path.is_file()
    if has_single == has_multi:
        raise ValueError(
            "Case package must contain exactly one authority source file: "
            f"{_SINGLE_SERIES_AUTHORITIES_NAME!r} or {_MULTI_SERIES_AUTHORITIES_NAME!r}."
        )
    state_payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(state_payload, Mapping):
        raise ValueError("input_state.json must contain a JSON object.")
    normalized_state = deepcopy(dict(state_payload))
    if has_single:
        authority_payload = json.loads(authority_path.read_text(encoding="utf-8"))
        return (
            normalized_state,
            _SINGLE_SERIES_AUTHORITIES_NAME,
            _normalise_single_series_authority_payload(authority_payload),
        )
    authority_payload = json.loads(authority_by_series_path.read_text(encoding="utf-8"))
    return (
        normalized_state,
        _MULTI_SERIES_AUTHORITIES_NAME,
        _normalise_multi_series_authority_payload(authority_payload),
    )


def _normalise_single_series_authority_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("authority_sources.json must contain a JSON array.")
    return [
        _normalise_authority_source(item, path=f"authority_sources[{index}]")
        for index, item in enumerate(payload)
    ]


def _normalise_multi_series_authority_payload(
    payload: Any,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise ValueError("authority_sources_by_series_id.json must contain a JSON object.")
    normalised: dict[str, list[dict[str, Any]]] = {}
    for raw_series_id, raw_sources in payload.items():
        if type(raw_series_id) is not str or not raw_series_id.strip():
            raise ValueError(
                "authority_sources_by_series_id.json keys must be non-empty strings."
            )
        if not isinstance(raw_sources, list):
            raise ValueError(
                f"authority_sources_by_series_id[{raw_series_id!r}] must contain a JSON array."
            )
        series_id = raw_series_id.strip()
        normalised[series_id] = [
            _normalise_authority_source(
                item,
                path=f"authority_sources_by_series_id[{series_id!r}][{index}]",
            )
            for index, item in enumerate(raw_sources)
        ]
    return normalised


def _normalise_authority_source(payload: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    normalised: dict[str, Any] = {}
    for field_name in ("source_type", "source_id", "reference"):
        value = payload.get(field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(f"{path}.{field_name} must be a non-empty string.")
        normalised[field_name] = value.strip()
    details = payload.get("details", {})
    if not isinstance(details, Mapping):
        raise ValueError(f"{path}.details must be an object when present.")
    normalised["details"] = deepcopy(dict(details))
    return normalised


def _validate_authority_package(
    *,
    state: Mapping[str, Any],
    authority_name: str,
    authority_payload: Any,
) -> None:
    case_kind = _case_kind_from_state_payload(state)
    if case_kind == MULTI_SERIES_CASE_KIND:
        if authority_name != _MULTI_SERIES_AUTHORITIES_NAME:
            raise ValueError(
                f"Multi-series packages must use {_MULTI_SERIES_AUTHORITIES_NAME!r}."
            )
        requirements = _try_multi_series_authority_requirements(state)
        if requirements is None:
            return
        if not isinstance(authority_payload, Mapping):
            raise ValueError("authority_sources_by_series_id.json must contain a JSON object.")
        expected_ids = [series_id for series_id, _ in requirements]
        missing_ids = [series_id for series_id in expected_ids if series_id not in authority_payload]
        extra_ids = [series_id for series_id in authority_payload if series_id not in set(expected_ids)]
        if missing_ids:
            raise ValueError(
                "authority_sources_by_series_id is missing entries for payload series: "
                + ", ".join(missing_ids)
            )
        if extra_ids:
            raise ValueError(
                "authority_sources_by_series_id contains unknown payload series: "
                + ", ".join(extra_ids)
            )
        for series_id, required_types in requirements:
            sources = authority_payload[series_id]
            seen_types = {str(source["source_type"]) for source in sources}
            missing_types = [item for item in required_types if item not in seen_types]
            if missing_types:
                raise ValueError(
                    f"authority validation failed for series {series_id!r}: missing authority "
                    f"source types {', '.join(missing_types)}."
                )
        return
    if authority_name != _SINGLE_SERIES_AUTHORITIES_NAME:
        raise ValueError(
            f"Single-series packages must use {_SINGLE_SERIES_AUTHORITIES_NAME!r}."
        )
    descriptor = _try_single_series_descriptor(state)
    if descriptor is None:
        return
    seen_types = {str(source["source_type"]) for source in authority_payload}
    missing_types = [
        item for item in descriptor.required_authority_source_types if item not in seen_types
    ]
    if missing_types:
        raise ValueError(
            "authority validation failed: missing authority source types "
            + ", ".join(missing_types)
            + "."
        )


def _preflight_case_state_validation_failure(
    state: Mapping[str, Any],
) -> str | None:
    try:
        report = build_case_verification_report(state)
    except ValueError as exc:
        return f"Case package failed pre-execution validation: {exc}"
    except Exception:
        return None
    if not _report_failed_pre_execution_validation(report):
        return None
    return _preflight_validation_message(report)


def _report_failed_pre_execution_validation(report: Mapping[str, Any]) -> bool:
    decision = _mapping_get(report, "decision")
    pipeline_phase = str(_mapping_get(decision, "pipeline_phase") or "")
    return pipeline_phase in _PACKAGE_INVALID_PIPELINE_PHASES


def _preflight_validation_message(report: Mapping[str, Any]) -> str:
    decision = _mapping_get(report, "decision")
    reasons = _mapping_get(decision, "decision_reasons")
    if isinstance(reasons, list):
        messages = [
            str(reason.get("message")).strip()
            for reason in reasons
            if isinstance(reason, Mapping)
            and isinstance(reason.get("message"), str)
            and str(reason.get("message")).strip()
        ]
        if messages:
            return (
                "Case package failed pre-execution validation: "
                + "; ".join(messages)
            )
    return (
        "Case package failed pre-execution validation during intake or "
        "structural validation."
    )


def _try_single_series_descriptor(state: Mapping[str, Any]) -> Any | None:
    try:
        return verification_profile_descriptor_from_state(state)
    except ValueError:
        return None


def _try_multi_series_authority_requirements(
    state: Mapping[str, Any],
) -> list[tuple[str, tuple[str, ...]]] | None:
    series_entries = state.get("series")
    if not isinstance(series_entries, list):
        return None
    seen_series_ids: set[str] = set()
    requirements: list[tuple[str, tuple[str, ...]]] = []
    for entry in series_entries:
        if not isinstance(entry, Mapping):
            return None
        series_id = entry.get("series_id")
        if type(series_id) is not str or not series_id.strip():
            return None
        normalised_series_id = series_id.strip()
        if normalised_series_id in seen_series_ids:
            return None
        seen_series_ids.add(normalised_series_id)
        try:
            descriptor = verification_profile_descriptor_from_state(entry)
        except ValueError:
            return None
        requirements.append(
            (
                normalised_series_id,
                tuple(str(item) for item in descriptor.required_authority_source_types),
            )
        )
    return requirements


def _hash_normalized_package(
    *,
    state: Mapping[str, Any],
    authority_name: str,
    authority_payload: Any,
) -> str:
    return hash_canonical_json(
        {
            "state": deepcopy(dict(state)),
            authority_name: deepcopy(authority_payload),
        }
    )


def _package_invalid_result(
    *,
    item: BatchQueueItem,
    manifest_index: int,
    queue_path_name: str,
    case_kind: str,
    package_validation: BatchPackageValidation,
) -> BatchItemResult:
    return {
        "schema_version": BATCH_ITEM_RESULT_SCHEMA_VERSION,
        "item_id": item["item_id"],
        "manifest_index": manifest_index,
        "case_dir": item["case_dir"],
        "case_kind": case_kind,
        "queue_path_name": queue_path_name,
        "current_bucket": QUEUE_BUCKET_QUARANTINE,
        "final_bucket": QUEUE_BUCKET_QUARANTINE,
        "final_outcome": OUTCOME_PACKAGE_INVALID,
        "entered_execution": False,
        "profile_override": item.get("profile_override"),
        "override_applied": False,
        "messages": list(package_validation["failures"]),
        "package_validation": package_validation,
        "resolved_policy": None,
        "attempts": [],
    }


def _execute_validated_package(
    *,
    validated_package: _ValidatedPackage,
    item: BatchQueueItem,
    item_dir: Path,
    max_attempts: int,
    package_validation: BatchPackageValidation,
) -> tuple[BatchItemResult, str]:
    attempts_root = item_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=False)
    attempts: list[BatchAttemptRecord] = []
    resolved_policy: BatchPolicySelection | None = None
    final_outcome = OUTCOME_WORKER_ERROR
    final_messages: list[str] = []
    final_bucket = QUEUE_BUCKET_QUARANTINE
    for attempt_number in range(1, max_attempts + 1):
        attempt_dir = attempts_root / f"attempt-{attempt_number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_record, resolved_policy, final_outcome, final_messages, should_retry = (
            _run_attempt(
                validated_package=validated_package,
                attempt_number=attempt_number,
                attempt_dir=attempt_dir,
                resolved_policy=resolved_policy,
                is_last_attempt=attempt_number == max_attempts,
            )
        )
        _write_json(attempt_dir / "attempt_record.json", attempt_record)
        attempts.append(attempt_record)
        if should_retry:
            continue
        final_bucket = (
            QUEUE_BUCKET_COMPLETED
            if final_outcome == OUTCOME_RELEASED
            else QUEUE_BUCKET_QUARANTINE
        )
        break
    item_result: BatchItemResult = {
        "schema_version": BATCH_ITEM_RESULT_SCHEMA_VERSION,
        "item_id": item["item_id"],
        "manifest_index": validated_package.manifest_index,
        "case_dir": item["case_dir"],
        "case_kind": validated_package.case_kind,
        "queue_path_name": validated_package.queue_path_name,
        "current_bucket": final_bucket,
        "final_bucket": final_bucket,
        "final_outcome": final_outcome,
        "entered_execution": True,
        "profile_override": validated_package.profile_override,
        "override_applied": validated_package.override_applied,
        "messages": final_messages,
        "package_validation": package_validation,
        "resolved_policy": resolved_policy,
        "attempts": attempts,
    }
    return item_result, final_bucket


def _run_attempt(
    *,
    validated_package: _ValidatedPackage,
    attempt_number: int,
    attempt_dir: Path,
    resolved_policy: BatchPolicySelection | None,
    is_last_attempt: bool,
) -> tuple[BatchAttemptRecord, BatchPolicySelection | None, str, list[str], bool]:
    started_at = _utc_now()
    messages: list[str] = []
    exception_class: str | None = None
    outcome = OUTCOME_WORKER_ERROR
    retryable = False
    should_retry = False
    try:
        report = build_case_verification_report(validated_package.normalized_state)
    except ValueError as exc:
        exception_class = exc.__class__.__name__
        messages.append(str(exc))
        outcome = OUTCOME_WORKER_ERROR
    except Exception as exc:  # pragma: no cover - exercised by tests via patching
        exception_class = exc.__class__.__name__
        messages.append(str(exc))
        if _is_retryable_execution_exception(exc):
            retryable = True
            if is_last_attempt:
                outcome = OUTCOME_TRANSIENT_FAILURE_EXHAUSTED
                should_retry = False
            else:
                outcome = _RETRYABLE_ATTEMPT_OUTCOME
                should_retry = True
        else:
            outcome = OUTCOME_WORKER_ERROR
    else:
        resolved_policy = _policy_selection_from_report(
            report,
            profile_override=validated_package.profile_override,
            override_applied=validated_package.override_applied,
        )
        try:
            if not report["release_eligible"]:
                outcome = _classify_non_release_report(report)
                messages.append(
                    "Case did not satisfy the canonical release gate and was withheld from automatic progression."
                )
                _write_blocked_attempt_artifacts(
                    attempt_dir=attempt_dir,
                    report=report,
                )
            else:
                _write_release_attempt_artifacts(
                    attempt_dir=attempt_dir,
                    validated_package=validated_package,
                )
                outcome = OUTCOME_RELEASED
                messages.append("Case released successfully through the Guardian batch worker.")
        except InvariantViolationError as exc:
            exception_class = exc.__class__.__name__
            outcome = OUTCOME_VERIFICATION_BLOCK
            messages.append(str(exc))
        except Exception as exc:  # pragma: no cover - exercised by tests via patching
            exception_class = exc.__class__.__name__
            messages.append(str(exc))
            if _is_retryable_execution_exception(exc):
                retryable = True
                if is_last_attempt:
                    outcome = OUTCOME_TRANSIENT_FAILURE_EXHAUSTED
                    should_retry = False
                else:
                    outcome = _RETRYABLE_ATTEMPT_OUTCOME
                    should_retry = True
            else:
                outcome = OUTCOME_WORKER_ERROR
    completed_at = _utc_now()
    record: BatchAttemptRecord = {
        "schema_version": BATCH_ATTEMPT_RECORD_SCHEMA_VERSION,
        "attempt_number": attempt_number,
        "started_at": started_at,
        "completed_at": completed_at,
        "outcome": outcome,
        "retryable": retryable,
        "exception_class": exception_class,
        "artifact_root": "artifacts",
        "messages": messages,
    }
    final_outcome = (
        OUTCOME_TRANSIENT_FAILURE_EXHAUSTED
        if outcome == _RETRYABLE_ATTEMPT_OUTCOME and is_last_attempt
        else outcome
    )
    return record, resolved_policy, final_outcome, messages, should_retry


def _write_blocked_attempt_artifacts(
    *,
    attempt_dir: Path,
    report: Mapping[str, Any],
) -> None:
    staged_root = _prepare_staged_artifact_root(attempt_dir)
    try:
        _write_json(staged_root / "verification_report.json", report)
        _write_json(staged_root / "decision_journal.json", build_decision_journal(report))
        _finalise_artifact_root(staged_root, attempt_dir / "artifacts")
    except Exception:
        _cleanup_staged_artifact_root(staged_root)
        raise


def _write_release_attempt_artifacts(
    *,
    attempt_dir: Path,
    validated_package: _ValidatedPackage,
) -> None:
    staged_root = _prepare_staged_artifact_root(attempt_dir)
    try:
        harness = MriGuardianHarness(
            transparency_log_path=staged_root / "transparency-log.jsonl",
            default_profile_name=DEFAULT_PROFILE_NAME,
        )
        if validated_package.case_kind == MULTI_SERIES_CASE_KIND:
            result = harness.release_multi_series_case(
                state=validated_package.normalized_state,
                actor=validated_package.execution_actor,
                authority_sources_by_series_id=validated_package.normalized_authority_payload,
                timestamp=validated_package.execution_timestamp,
                transition_id=validated_package.transition_id,
            )
        else:
            result = harness.release_case(
                state=validated_package.normalized_state,
                actor=validated_package.execution_actor,
                authority_sources=validated_package.normalized_authority_payload,
                profile_name=_single_series_profile_name(validated_package.normalized_state),
                timestamp=validated_package.execution_timestamp,
                transition_id=validated_package.transition_id,
            )
        report = build_case_verification_report(result.after_state)
        _write_json(staged_root / "released_state.json", result.after_state)
        _write_json(
            staged_root / "workstation_payload.json",
            json.loads(render_release_bundle_json(result)),
        )
        _write_json(staged_root / "evidence.json", result.evidence.as_dict())
        bundle_path = staged_root / f"{result.after_state['case_id']}{GUARDIAN_BUNDLE_EXTENSION}"
        bundle_metadata = write_portable_evidence_bundle(
            result=result,
            transparency_log_path=staged_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        _write_json(staged_root / "verification_report.json", report)
        _write_json(staged_root / "decision_journal.json", bundle_metadata["decision_journal"])
        _finalise_artifact_root(staged_root, attempt_dir / "artifacts")
    except Exception:
        _cleanup_staged_artifact_root(staged_root)
        raise


def _prepare_staged_artifact_root(attempt_dir: Path) -> Path:
    staged_root = attempt_dir / "artifacts.staging"
    if staged_root.exists():
        shutil.rmtree(staged_root, ignore_errors=True)
    staged_root.mkdir(parents=True, exist_ok=False)
    return staged_root


def _cleanup_staged_artifact_root(staged_root: Path) -> None:
    shutil.rmtree(staged_root, ignore_errors=True)


def _finalise_artifact_root(staged_root: Path, artifact_root: Path) -> None:
    staged_root.replace(artifact_root)


def _single_series_profile_name(state: Mapping[str, Any]) -> str:
    descriptor = verification_profile_descriptor_from_state(state)
    return str(descriptor.profile_name)


def _policy_selection_from_report(
    report: Mapping[str, Any],
    *,
    profile_override: str | None,
    override_applied: bool,
) -> BatchPolicySelection:
    if report.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return {
            "mode": "embedded_multi_series",
            "profile_override": profile_override,
            "override_applied": False,
            "series_profiles": [
                {
                    "series_id": str(series_report["series_id"]),
                    "profile_name": str(series_report["profile"]["profile_name"]),
                    "profile_revision": str(series_report["profile"]["profile_revision"]),
                    "profile_digest": str(series_report["profile"]["profile_digest"]),
                }
                for series_report in report.get("series_reports", [])
            ],
        }
    profile = report["profile"]
    return {
        "mode": "single_series",
        "profile_override": profile_override,
        "override_applied": override_applied,
        "selected_profile_name": str(profile["profile_name"]),
        "selected_profile_revision": str(profile["profile_revision"]),
        "selected_profile_digest": str(profile["profile_digest"]),
    }


def _classify_non_release_report(report: Mapping[str, Any]) -> str:
    decision = report.get("decision", {})
    decision_class = str(_mapping_get(decision, "decision_class") or "")
    if decision_class == "blocked_policy":
        return OUTCOME_POLICY_BLOCK
    if decision_class in {"blocked_structural", "blocked_physical"}:
        return OUTCOME_VERIFICATION_BLOCK
    raise ValueError(
        "Non-release report did not expose a supported canonical decision_class."
    )


def _is_retryable_execution_exception(exc: Exception) -> bool:
    return isinstance(exc, _RETRYABLE_EXCEPTION_TYPES)


def _move_item_directory(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(source), str(destination)))


def _build_batch_summary(
    *,
    batch_id: str,
    manifest_path: str | None,
    output_root: Path,
    started_at: str,
    completed_at: str,
    max_attempts: int,
    allowed_profile_overrides: Sequence[str],
    results: Sequence[BatchItemResult],
) -> BatchSummary:
    outcome_counts = {outcome: 0 for outcome in _FINAL_OUTCOME_ORDER}
    released_items = 0
    quarantined_items = 0
    for result in results:
        outcome_counts[result["final_outcome"]] += 1
        if result["final_outcome"] == OUTCOME_RELEASED:
            released_items += 1
        else:
            quarantined_items += 1
    return {
        "batch_summary_schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
        "batch_id": batch_id,
        "manifest_path": manifest_path,
        "output_root": str(output_root),
        "run_status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "max_attempts": max_attempts,
        "allowed_profile_overrides": list(allowed_profile_overrides),
        "total_items": len(results),
        "released_items": released_items,
        "quarantined_items": quarantined_items,
        "outcome_counts": outcome_counts,
        "items": list(results),
    }


def _queue_path_name(manifest_index: int, item_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip("-")
    if not safe_id:
        safe_id = "item"
    return f"{manifest_index:04d}-{safe_id}"


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.write_text(f"{canonical_json_dumps(payload)}\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _case_kind_from_state_payload(state: Mapping[str, Any] | None) -> str:
    if isinstance(state, Mapping) and state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return MULTI_SERIES_CASE_KIND
    return "single_series_v1"


def _mapping_get(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


__all__ = [
    "BATCH_ATTEMPT_RECORD_SCHEMA_VERSION",
    "BATCH_ITEM_RESULT_SCHEMA_VERSION",
    "BATCH_SCHEMA_VERSION",
    "BATCH_SUMMARY_SCHEMA_VERSION",
    "BATCH_VALIDATION_SCHEMA_VERSION",
    "BatchAttemptRecord",
    "BatchItemResult",
    "BatchManifest",
    "BatchPackageValidation",
    "BatchPolicySelection",
    "BatchQueueItem",
    "BatchSummary",
    "OUTCOME_PACKAGE_INVALID",
    "OUTCOME_POLICY_BLOCK",
    "OUTCOME_RELEASED",
    "OUTCOME_TRANSIENT_FAILURE_EXHAUSTED",
    "OUTCOME_VERIFICATION_BLOCK",
    "OUTCOME_WORKER_ERROR",
    "QUEUE_BUCKET_COMPLETED",
    "QUEUE_BUCKET_PROCESSING",
    "QUEUE_BUCKET_QUARANTINE",
    "QUEUE_BUCKET_QUEUED",
    "run_batch",
]
