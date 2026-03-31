"""Deterministic DICOM-shaped object modelling for mock Guardian interoperability."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from icsd.core.serialise import canonical_json_dumps

try:
    from pydicom.uid import UID
except Exception:  # pragma: no cover - optional dependency fallback for local execution
    UID = None

DICOM_MODEL_SCHEMA_VERSION = "0.2"
UID_NAMESPACE_VERSION = "1"
UID_ROOT = "1.2.826.0.1.3680043.10.54321"
GUARDIAN_EXTENSION_NAMESPACE = "guardian_v1"
OBJECT_KIND_VOCABULARY = (
    "study",
    "series",
    "instance",
    "document",
    "sidecar",
    "acquisition_provenance",
)
RELATIONSHIP_VOCABULARY = (
    "contains_series",
    "contains_instance",
    "derived_from",
    "references",
    "same_frame_of_reference",
    "blocked_output_for",
    "sidecar_for",
)
FRAME_OF_REFERENCE_STATUS_VOCABULARY = (
    "explicit_uid",
    "upstream_non_dicom_uid",
    "no_upstream_lineage",
    "not_applicable",
)
LINEAGE_COMPLETENESS_VOCABULARY = ("complete", "partial", "not_applicable")
SIDECAR_TYPE_ICSD_EVIDENCE_JSON = "icsd_evidence_json"

MR_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.4"
SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"
COMPREHENSIVE_SR_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.88.33"

_UID_COMPONENT_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_UID_KIND_CODES = {
    "study": "11",
    "series": "12",
    "instance": "13",
    "document": "14",
}
_RELATIONSHIP_RULES: dict[str, dict[str, Any]] = {
    "contains_series": {
        "directionality": "directional",
        "source_kinds": frozenset({"study"}),
        "target_kinds": frozenset({"series"}),
    },
    "contains_instance": {
        "directionality": "directional",
        "source_kinds": frozenset({"series"}),
        "target_kinds": frozenset({"instance", "document"}),
    },
    "derived_from": {
        "directionality": "directional",
        "source_kinds": frozenset({"series", "instance", "document"}),
        "target_kinds": frozenset({"series", "instance", "document"}),
    },
    "references": {
        "directionality": "directional",
        "source_kinds": frozenset({"series", "instance", "document"}),
        "target_kinds": frozenset({"series", "instance", "document", "acquisition_provenance"}),
    },
    "same_frame_of_reference": {
        "directionality": "symmetric",
        "source_kinds": frozenset({"series"}),
        "target_kinds": frozenset({"series"}),
    },
    "blocked_output_for": {
        "directionality": "directional",
        "source_kinds": frozenset({"series"}),
        "target_kinds": frozenset({"series"}),
    },
    "sidecar_for": {
        "directionality": "directional",
        "source_kinds": frozenset({"sidecar"}),
        "target_kinds": frozenset({"document"}),
    },
}
_EXTENSION_SCHEMA: dict[str, tuple[str, ...]] = {
    "study": ("source_case_id", "source_study_key", "case_kind"),
    "series": (
        "source_series_key",
        "source_series_uid",
        "resolved_role",
        "gating_status",
        "output_status",
        "downstream_release_allowed",
        "withheld_reason",
    ),
    "instance": ("source_series_key",),
    "document": ("source_series_key", "profile_name", "decision_class"),
    "sidecar": (),
    "acquisition_provenance": ("source_series_key",),
}


def deterministic_uid(*, object_kind: str, seed: Mapping[str, Any]) -> str:
    """Return a deterministic mock DICOM UID for a specific object kind."""
    if object_kind not in _UID_KIND_CODES:
        raise ValueError(f"Unsupported UID object_kind {object_kind!r}.")
    canonical_seed = {
        "schema_version": DICOM_MODEL_SCHEMA_VERSION,
        "uid_namespace_version": UID_NAMESPACE_VERSION,
        "object_kind": object_kind,
        "seed": _canonicalise_seed(seed, path="seed"),
    }
    decimal = str(
        int(
            hashlib.sha256(canonical_json_dumps(canonical_seed).encode("utf-8")).hexdigest(),
            16,
        )
    )
    prefix = f"{UID_ROOT}.{UID_NAMESPACE_VERSION}.{_UID_KIND_CODES[object_kind]}."
    available_digits = 64 - len(prefix)
    uid = f"{prefix}{decimal[:available_digits]}"
    _validate_uid(uid, path="uid")
    return uid


def build_study_object(
    *,
    study_instance_uid: str,
    patient_id: str,
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "object_kind": "study",
        "object_id": _object_id("study", study_instance_uid),
        "study_instance_uid": _validate_uid(
            study_instance_uid,
            path="study.study_instance_uid",
        ),
        "patient_id": _require_non_empty_trimmed_string(
            patient_id,
            path="study.patient_id",
        ),
        "extensions": _build_extensions("study", extensions),
    }


def build_series_object(
    *,
    series_instance_uid: str,
    modality: str,
    semantic_role: str,
    series_description: str,
    frame_of_reference_uid: str | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "object_kind": "series",
        "object_id": _object_id("series", series_instance_uid),
        "series_instance_uid": _validate_uid(
            series_instance_uid,
            path="series.series_instance_uid",
        ),
        "modality": _require_non_empty_trimmed_string(modality, path="series.modality"),
        "semantic_role": _require_non_empty_trimmed_string(
            semantic_role,
            path="series.semantic_role",
        ),
        "series_description": _require_non_empty_trimmed_string(
            series_description,
            path="series.series_description",
        ),
        "extensions": _build_extensions("series", extensions),
    }
    if frame_of_reference_uid is not None:
        payload["frame_of_reference_uid"] = _validate_uid(
            frame_of_reference_uid,
            path="series.frame_of_reference_uid",
        )
    return payload


def build_instance_object(
    *,
    sop_instance_uid: str,
    sop_class_uid: str,
    sop_class_name: str,
    semantic_role: str,
    content_hash: str,
    referenced_instance_uids: Sequence[str] | None = None,
    segmentation_metadata: Mapping[str, Any] | None = None,
    hash_algorithm: str = "sha256",
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "object_kind": "instance",
        "object_id": _object_id("instance", sop_instance_uid),
        "sop_instance_uid": _validate_uid(
            sop_instance_uid,
            path="instance.sop_instance_uid",
        ),
        "sop_class_uid": _validate_uid(
            sop_class_uid,
            path="instance.sop_class_uid",
        ),
        "sop_class_name": _require_non_empty_trimmed_string(
            sop_class_name,
            path="instance.sop_class_name",
        ),
        "semantic_role": _require_non_empty_trimmed_string(
            semantic_role,
            path="instance.semantic_role",
        ),
        "content_hash": _require_sha256_hex(content_hash, path="instance.content_hash"),
        "hash_algorithm": _require_non_empty_trimmed_string(
            hash_algorithm,
            path="instance.hash_algorithm",
        ),
        "extensions": _build_extensions("instance", extensions),
    }
    if referenced_instance_uids is not None:
        payload["referenced_instance_uids"] = [
            _validate_uid(item, path=f"instance.referenced_instance_uids[{index}]")
            for index, item in enumerate(referenced_instance_uids)
        ]
    if segmentation_metadata is not None:
        payload["segmentation_metadata"] = _canonicalise_segmentation_metadata(
            segmentation_metadata
        )
    return payload


def build_document_object(
    *,
    sop_instance_uid: str,
    semantic_role: str,
    document_kind: str,
    content_hash: str,
    referenced_instance_uids: Sequence[str],
    measurement_groups: Sequence[Mapping[str, Any]],
    observer_context: Mapping[str, Any],
    profile_name: str | None = None,
    hash_algorithm: str = "sha256",
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged_extensions = {} if extensions is None else dict(extensions)
    if profile_name is not None:
        merged_extensions["profile_name"] = profile_name
    return {
        "object_kind": "document",
        "object_id": _object_id("document", sop_instance_uid),
        "sop_instance_uid": _validate_uid(
            sop_instance_uid,
            path="document.sop_instance_uid",
        ),
        "sop_class_uid": _validate_uid(
            COMPREHENSIVE_SR_STORAGE_UID,
            path="document.sop_class_uid",
        ),
        "sop_class_name": "Comprehensive SR Storage",
        "semantic_role": _require_non_empty_trimmed_string(
            semantic_role,
            path="document.semantic_role",
        ),
        "document_kind": _require_non_empty_trimmed_string(
            document_kind,
            path="document.document_kind",
        ),
        "content_hash": _require_sha256_hex(content_hash, path="document.content_hash"),
        "hash_algorithm": _require_non_empty_trimmed_string(
            hash_algorithm,
            path="document.hash_algorithm",
        ),
        "referenced_instance_uids": [
            _validate_uid(item, path=f"document.referenced_instance_uids[{index}]")
            for index, item in enumerate(referenced_instance_uids)
        ],
        "measurement_groups": [
            _canonicalise_measurement_group(item, path=f"document.measurement_groups[{index}]")
            for index, item in enumerate(measurement_groups)
        ],
        "observer_context": _canonicalise_observer_context(observer_context),
        "extensions": _build_extensions("document", merged_extensions),
    }


def build_sidecar_object(
    *,
    sidecar_id: str,
    sidecar_type: str,
    media_type: str,
    hash_value: str,
    referenced_manifest_instance_uid: str,
    bundle_reference: str | None,
    bundle_member: str | None = None,
    hash_algorithm: str = "sha256",
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "object_kind": "sidecar",
        "object_id": _require_non_empty_trimmed_string(sidecar_id, path="sidecar.object_id"),
        "sidecar_id": _require_non_empty_trimmed_string(sidecar_id, path="sidecar.sidecar_id"),
        "sidecar_type": _require_non_empty_trimmed_string(
            sidecar_type,
            path="sidecar.sidecar_type",
        ),
        "media_type": _require_non_empty_trimmed_string(
            media_type,
            path="sidecar.media_type",
        ),
        "hash_algorithm": _require_non_empty_trimmed_string(
            hash_algorithm,
            path="sidecar.hash_algorithm",
        ),
        "hash_value": _require_sha256_hex(hash_value, path="sidecar.hash_value"),
        "referenced_manifest_instance_uid": _validate_uid(
            referenced_manifest_instance_uid,
            path="sidecar.referenced_manifest_instance_uid",
        ),
        "bundle_reference": _optional_trimmed_string(
            bundle_reference,
            path="sidecar.bundle_reference",
        ),
        "extensions": _build_extensions("sidecar", extensions),
    }
    if bundle_member is not None:
        payload["bundle_member"] = _require_non_empty_trimmed_string(
            bundle_member,
            path="sidecar.bundle_member",
        )
    return payload


def build_acquisition_provenance_object(
    *,
    object_id: str,
    provenance_type: str,
    raw_input_hash: str,
    source_members: Sequence[str],
    extensions: Mapping[str, Any] | None = None,
    hash_algorithm: str = "sha256",
) -> dict[str, Any]:
    return {
        "object_kind": "acquisition_provenance",
        "object_id": _require_non_empty_trimmed_string(
            object_id,
            path="acquisition_provenance.object_id",
        ),
        "provenance_type": _require_non_empty_trimmed_string(
            provenance_type,
            path="acquisition_provenance.provenance_type",
        ),
        "raw_input_hash": _require_sha256_hex(
            raw_input_hash,
            path="acquisition_provenance.raw_input_hash",
        ),
        "hash_algorithm": _require_non_empty_trimmed_string(
            hash_algorithm,
            path="acquisition_provenance.hash_algorithm",
        ),
        "source_members": [
            _require_non_empty_trimmed_string(
                item,
                path=f"acquisition_provenance.source_members[{index}]",
            )
            for index, item in enumerate(source_members)
        ],
        "extensions": _build_extensions("acquisition_provenance", extensions),
    }


def build_relationship(
    *,
    relationship_type: str,
    source_object_id: str,
    source_kind: str,
    target_object_id: str,
    target_kind: str,
) -> dict[str, Any]:
    if relationship_type not in _RELATIONSHIP_RULES:
        raise ValueError(f"Unsupported relationship_type {relationship_type!r}.")
    rule = _RELATIONSHIP_RULES[relationship_type]
    if source_kind not in rule["source_kinds"]:
        raise ValueError(
            f"{relationship_type} does not allow source_kind {source_kind!r}."
        )
    if target_kind not in rule["target_kinds"]:
        raise ValueError(
            f"{relationship_type} does not allow target_kind {target_kind!r}."
        )
    source_id = _require_non_empty_trimmed_string(
        source_object_id,
        path="relationship.source_object_id",
    )
    target_id = _require_non_empty_trimmed_string(
        target_object_id,
        path="relationship.target_object_id",
    )
    resolved_source_kind = _require_object_kind(
        source_kind,
        path="relationship.source_kind",
    )
    resolved_target_kind = _require_object_kind(
        target_kind,
        path="relationship.target_kind",
    )
    if rule["directionality"] == "symmetric" and target_id < source_id:
        source_id, target_id = target_id, source_id
        resolved_source_kind, resolved_target_kind = (
            resolved_target_kind,
            resolved_source_kind,
        )
    return {
        "relationship_type": relationship_type,
        "directionality": rule["directionality"],
        "source_object_id": source_id,
        "source_kind": resolved_source_kind,
        "target_object_id": target_id,
        "target_kind": resolved_target_kind,
    }


def build_object_registry(objects: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    registry: dict[str, str] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ValueError(f"object_registry[{index}] must be an object.")
        object_id = _require_non_empty_trimmed_string(
            item.get("object_id"),
            path=f"object_registry[{index}].object_id",
        )
        object_kind = _require_object_kind(
            item.get("object_kind"),
            path=f"object_registry[{index}].object_kind",
        )
        if object_id in registry:
            raise ValueError(f"Duplicate object_id {object_id!r}.")
        registry[object_id] = object_kind
    return registry


def build_sidecar_id(*, seed: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_json_dumps(_canonicalise_seed(seed, path="sidecar_seed")).encode("utf-8")
    ).hexdigest()
    return f"sidecar:{digest[:24]}"


def build_acquisition_provenance_id(*, seed: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_json_dumps(_canonicalise_seed(seed, path="acquisition_seed")).encode("utf-8")
    ).hexdigest()
    return f"acquisition:{digest[:24]}"


def sop_class_name(uid: str) -> str:
    if uid == MR_IMAGE_STORAGE_UID:
        return "MR Image Storage"
    if uid == SEGMENTATION_STORAGE_UID:
        return "Segmentation Storage"
    if uid == COMPREHENSIVE_SR_STORAGE_UID:
        return "Comprehensive SR Storage"
    if UID is not None:
        return str(UID(uid).name)
    return uid


def validate_standard_like_object(payload: Mapping[str, Any], *, object_kind: str) -> dict[str, Any]:
    return _validate_standard_like_object(dict(payload), object_kind=object_kind)


def validate_sidecar_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_sidecar_object(dict(payload))


def validate_acquisition_provenance_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_acquisition_provenance_object(dict(payload))


def validate_relationships(
    relationships: Sequence[Mapping[str, Any]],
    *,
    known_objects: Mapping[str, str],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    normalised: list[dict[str, Any]] = []
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, Mapping):
            raise ValueError(f"relationships[{index}] must be an object.")
        payload = build_relationship(
            relationship_type=str(relationship.get("relationship_type")),
            source_object_id=str(relationship.get("source_object_id")),
            source_kind=str(relationship.get("source_kind")),
            target_object_id=str(relationship.get("target_object_id")),
            target_kind=str(relationship.get("target_kind")),
        )
        if payload["source_object_id"] not in known_objects:
            raise ValueError(
                f"relationships[{index}] references unknown source_object_id {payload['source_object_id']!r}."
            )
        if payload["target_object_id"] not in known_objects:
            raise ValueError(
                f"relationships[{index}] references unknown target_object_id {payload['target_object_id']!r}."
            )
        if known_objects[payload["source_object_id"]] != payload["source_kind"]:
            raise ValueError(
                f"relationships[{index}] source_kind does not match object registry."
            )
        if known_objects[payload["target_object_id"]] != payload["target_kind"]:
            raise ValueError(
                f"relationships[{index}] target_kind does not match object registry."
            )
        signature = (
            payload["relationship_type"],
            payload["directionality"],
            payload["source_object_id"],
            payload["source_kind"],
            payload["target_object_id"],
            payload["target_kind"],
        )
        if signature in seen:
            raise ValueError(f"relationships[{index}] duplicates an existing relationship.")
        seen.add(signature)
        normalised.append(payload)
    return normalised


def _validate_standard_like_object(payload: dict[str, Any], *, object_kind: str) -> dict[str, Any]:
    expected_fields = {
        "study": {"object_kind", "object_id", "study_instance_uid", "patient_id", "extensions"},
        "series": {
            "object_kind",
            "object_id",
            "series_instance_uid",
            "modality",
            "semantic_role",
            "series_description",
            "frame_of_reference_uid",
            "extensions",
        },
        "instance": {
            "object_kind",
            "object_id",
            "sop_instance_uid",
            "sop_class_uid",
            "sop_class_name",
            "semantic_role",
            "content_hash",
            "hash_algorithm",
            "referenced_instance_uids",
            "segmentation_metadata",
            "extensions",
        },
        "document": {
            "object_kind",
            "object_id",
            "sop_instance_uid",
            "sop_class_uid",
            "sop_class_name",
            "semantic_role",
            "document_kind",
            "content_hash",
            "hash_algorithm",
            "observer_context",
            "referenced_instance_uids",
            "measurement_groups",
            "extensions",
        },
    }[object_kind]
    unexpected = sorted(key for key in payload if key not in expected_fields)
    if unexpected:
        raise ValueError(
            f"{object_kind} contains unsupported fields outside the Guardian extension container: {', '.join(unexpected)}."
        )
    payload["object_kind"] = _require_object_kind(
        payload.get("object_kind"),
        path=f"{object_kind}.object_kind",
    )
    if payload["object_kind"] != object_kind:
        raise ValueError(f"{object_kind}.object_kind must be {object_kind!r}.")
    payload["object_id"] = _require_non_empty_trimmed_string(
        payload.get("object_id"),
        path=f"{object_kind}.object_id",
    )
    if object_kind == "study":
        payload["study_instance_uid"] = _validate_uid(
            payload.get("study_instance_uid"),
            path="study.study_instance_uid",
        )
        payload["patient_id"] = _require_non_empty_trimmed_string(
            payload.get("patient_id"),
            path="study.patient_id",
        )
    elif object_kind == "series":
        payload["series_instance_uid"] = _validate_uid(
            payload.get("series_instance_uid"),
            path="series.series_instance_uid",
        )
        payload["modality"] = _require_non_empty_trimmed_string(
            payload.get("modality"),
            path="series.modality",
        )
        payload["semantic_role"] = _require_non_empty_trimmed_string(
            payload.get("semantic_role"),
            path="series.semantic_role",
        )
        payload["series_description"] = _require_non_empty_trimmed_string(
            payload.get("series_description"),
            path="series.series_description",
        )
        if "frame_of_reference_uid" in payload:
            payload["frame_of_reference_uid"] = _validate_uid(
                payload.get("frame_of_reference_uid"),
                path="series.frame_of_reference_uid",
            )
    elif object_kind == "instance":
        payload["sop_instance_uid"] = _validate_uid(
            payload.get("sop_instance_uid"),
            path="instance.sop_instance_uid",
        )
        payload["sop_class_uid"] = _validate_uid(
            payload.get("sop_class_uid"),
            path="instance.sop_class_uid",
        )
        payload["sop_class_name"] = _require_non_empty_trimmed_string(
            payload.get("sop_class_name"),
            path="instance.sop_class_name",
        )
        payload["semantic_role"] = _require_non_empty_trimmed_string(
            payload.get("semantic_role"),
            path="instance.semantic_role",
        )
        payload["content_hash"] = _require_sha256_hex(
            payload.get("content_hash"),
            path="instance.content_hash",
        )
        payload["hash_algorithm"] = _require_non_empty_trimmed_string(
            payload.get("hash_algorithm"),
            path="instance.hash_algorithm",
        )
        if "referenced_instance_uids" in payload:
            referenced_instance_uids = payload.get("referenced_instance_uids")
            if not isinstance(referenced_instance_uids, list):
                raise ValueError("instance.referenced_instance_uids must be a list.")
            payload["referenced_instance_uids"] = [
                _validate_uid(item, path=f"instance.referenced_instance_uids[{index}]")
                for index, item in enumerate(referenced_instance_uids)
            ]
        if "segmentation_metadata" in payload:
            payload["segmentation_metadata"] = _canonicalise_segmentation_metadata(
                payload.get("segmentation_metadata")
            )
    else:
        payload["sop_instance_uid"] = _validate_uid(
            payload.get("sop_instance_uid"),
            path="document.sop_instance_uid",
        )
        payload["sop_class_uid"] = _validate_uid(
            payload.get("sop_class_uid"),
            path="document.sop_class_uid",
        )
        payload["sop_class_name"] = _require_non_empty_trimmed_string(
            payload.get("sop_class_name"),
            path="document.sop_class_name",
        )
        payload["semantic_role"] = _require_non_empty_trimmed_string(
            payload.get("semantic_role"),
            path="document.semantic_role",
        )
        payload["document_kind"] = _require_non_empty_trimmed_string(
            payload.get("document_kind"),
            path="document.document_kind",
        )
        payload["content_hash"] = _require_sha256_hex(
            payload.get("content_hash"),
            path="document.content_hash",
        )
        payload["hash_algorithm"] = _require_non_empty_trimmed_string(
            payload.get("hash_algorithm"),
            path="document.hash_algorithm",
        )
        payload["observer_context"] = _canonicalise_observer_context(
            payload.get("observer_context")
        )
        referenced_instance_uids = payload.get("referenced_instance_uids")
        if not isinstance(referenced_instance_uids, list):
            raise ValueError("document.referenced_instance_uids must be a list.")
        payload["referenced_instance_uids"] = [
            _validate_uid(item, path=f"document.referenced_instance_uids[{index}]")
            for index, item in enumerate(referenced_instance_uids)
        ]
        measurement_groups = payload.get("measurement_groups")
        if not isinstance(measurement_groups, list):
            raise ValueError("document.measurement_groups must be a list.")
        payload["measurement_groups"] = [
            _canonicalise_measurement_group(item, path=f"document.measurement_groups[{index}]")
            for index, item in enumerate(measurement_groups)
        ]
    payload["extensions"] = _validate_extensions(
        object_kind,
        payload.get("extensions"),
    )
    return payload


def _validate_sidecar_object(payload: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "object_kind",
        "object_id",
        "sidecar_id",
        "sidecar_type",
        "media_type",
        "hash_algorithm",
        "hash_value",
        "referenced_manifest_instance_uid",
        "bundle_reference",
        "bundle_member",
        "extensions",
    }
    unexpected = sorted(key for key in payload if key not in expected_fields)
    if unexpected:
        raise ValueError(
            "sidecar contains unsupported fields outside the Guardian extension container: "
            + ", ".join(unexpected)
            + "."
        )
    payload["object_kind"] = _require_object_kind(payload.get("object_kind"), path="sidecar.object_kind")
    if payload["object_kind"] != "sidecar":
        raise ValueError("sidecar.object_kind must be 'sidecar'.")
    payload["object_id"] = _require_non_empty_trimmed_string(
        payload.get("object_id"),
        path="sidecar.object_id",
    )
    payload["sidecar_id"] = _require_non_empty_trimmed_string(
        payload.get("sidecar_id"),
        path="sidecar.sidecar_id",
    )
    if payload["object_id"] != payload["sidecar_id"]:
        raise ValueError("sidecar.object_id must match sidecar.sidecar_id.")
    payload["sidecar_type"] = _require_non_empty_trimmed_string(
        payload.get("sidecar_type"),
        path="sidecar.sidecar_type",
    )
    if payload["sidecar_type"] != SIDECAR_TYPE_ICSD_EVIDENCE_JSON:
        raise ValueError(
            f"sidecar.sidecar_type must be {SIDECAR_TYPE_ICSD_EVIDENCE_JSON!r}."
        )
    payload["media_type"] = _require_non_empty_trimmed_string(
        payload.get("media_type"),
        path="sidecar.media_type",
    )
    payload["hash_algorithm"] = _require_non_empty_trimmed_string(
        payload.get("hash_algorithm"),
        path="sidecar.hash_algorithm",
    )
    payload["hash_value"] = _require_sha256_hex(
        payload.get("hash_value"),
        path="sidecar.hash_value",
    )
    payload["referenced_manifest_instance_uid"] = _validate_uid(
        payload.get("referenced_manifest_instance_uid"),
        path="sidecar.referenced_manifest_instance_uid",
    )
    payload["bundle_reference"] = _optional_trimmed_string(
        payload.get("bundle_reference"),
        path="sidecar.bundle_reference",
    )
    if "bundle_member" in payload:
        payload["bundle_member"] = _require_non_empty_trimmed_string(
            payload.get("bundle_member"),
            path="sidecar.bundle_member",
        )
    payload["extensions"] = _validate_extensions("sidecar", payload.get("extensions"))
    return payload


def _validate_acquisition_provenance_object(payload: dict[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "object_kind",
        "object_id",
        "provenance_type",
        "raw_input_hash",
        "hash_algorithm",
        "source_members",
        "extensions",
    }
    unexpected = sorted(key for key in payload if key not in expected_fields)
    if unexpected:
        raise ValueError(
            "acquisition_provenance contains unsupported fields outside the Guardian extension container: "
            + ", ".join(unexpected)
            + "."
        )
    payload["object_kind"] = _require_object_kind(
        payload.get("object_kind"),
        path="acquisition_provenance.object_kind",
    )
    if payload["object_kind"] != "acquisition_provenance":
        raise ValueError("acquisition_provenance.object_kind must be 'acquisition_provenance'.")
    payload["object_id"] = _require_non_empty_trimmed_string(
        payload.get("object_id"),
        path="acquisition_provenance.object_id",
    )
    payload["provenance_type"] = _require_non_empty_trimmed_string(
        payload.get("provenance_type"),
        path="acquisition_provenance.provenance_type",
    )
    payload["raw_input_hash"] = _require_sha256_hex(
        payload.get("raw_input_hash"),
        path="acquisition_provenance.raw_input_hash",
    )
    payload["hash_algorithm"] = _require_non_empty_trimmed_string(
        payload.get("hash_algorithm"),
        path="acquisition_provenance.hash_algorithm",
    )
    source_members = payload.get("source_members")
    if not isinstance(source_members, list) or not source_members:
        raise ValueError("acquisition_provenance.source_members must be a non-empty list.")
    payload["source_members"] = [
        _require_non_empty_trimmed_string(
            item,
            path=f"acquisition_provenance.source_members[{index}]",
        )
        for index, item in enumerate(source_members)
    ]
    payload["extensions"] = _validate_extensions(
        "acquisition_provenance",
        payload.get("extensions"),
    )
    return payload


def _build_extensions(object_kind: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {} if value is None else dict(value)
    allowed_fields = set(_EXTENSION_SCHEMA[object_kind])
    unexpected = sorted(key for key in payload if key not in allowed_fields)
    if unexpected:
        raise ValueError(
            f"{object_kind}.extensions.{GUARDIAN_EXTENSION_NAMESPACE} contains unsupported fields: "
            + ", ".join(unexpected)
            + "."
        )
    normalised: dict[str, Any] = {}
    for key, item in payload.items():
        if type(item) is bool:
            normalised[key] = item
        elif item is None:
            normalised[key] = None
        else:
            normalised[key] = _require_non_empty_trimmed_string(
                item,
                path=f"{object_kind}.extensions.{GUARDIAN_EXTENSION_NAMESPACE}.{key}",
            )
    return {GUARDIAN_EXTENSION_NAMESPACE: normalised}


def _validate_extensions(object_kind: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{object_kind}.extensions must be an object.")
    payload = dict(value)
    unexpected_namespaces = sorted(key for key in payload if key != GUARDIAN_EXTENSION_NAMESPACE)
    if unexpected_namespaces:
        raise ValueError(
            f"{object_kind}.extensions supports only {GUARDIAN_EXTENSION_NAMESPACE!r}; "
            f"found {', '.join(unexpected_namespaces)}."
        )
    extension_payload = payload.get(GUARDIAN_EXTENSION_NAMESPACE, {})
    return _build_extensions(object_kind, extension_payload)


def _canonicalise_observer_context(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("document.observer_context must be an object.")
    payload = dict(value)
    expected = {"observer_type", "observer_name"}
    unexpected = sorted(key for key in payload if key not in expected)
    missing = sorted(key for key in expected if key not in payload)
    if missing:
        raise ValueError(
            "document.observer_context is missing required fields: "
            + ", ".join(missing)
            + "."
        )
    if unexpected:
        raise ValueError(
            "document.observer_context contains unsupported fields: "
            + ", ".join(unexpected)
            + "."
        )
    return {
        "observer_type": _require_non_empty_trimmed_string(
            payload["observer_type"],
            path="document.observer_context.observer_type",
        ),
        "observer_name": _require_non_empty_trimmed_string(
            payload["observer_name"],
            path="document.observer_context.observer_name",
        ),
    }


def _canonicalise_measurement_group(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    payload = dict(value)
    expected = {"measurement_name", "value", "units"}
    unexpected = sorted(key for key in payload if key not in expected)
    missing = sorted(key for key in expected if key not in payload)
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}.")
    if unexpected:
        raise ValueError(f"{path} contains unsupported fields: {', '.join(unexpected)}.")
    value_field = payload["value"]
    if type(value_field) not in {int, float}:
        raise ValueError(f"{path}.value must be numeric.")
    return {
        "measurement_name": _require_non_empty_trimmed_string(
            payload["measurement_name"],
            path=f"{path}.measurement_name",
        ),
        "value": float(value_field),
        "units": _require_non_empty_trimmed_string(
            payload["units"],
            path=f"{path}.units",
        ),
    }


def _canonicalise_segmentation_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("instance.segmentation_metadata must be an object.")
    payload = dict(value)
    expected = {
        "segment_number",
        "segment_label",
        "algorithm_type",
        "algorithm_name",
    }
    unexpected = sorted(key for key in payload if key not in expected)
    missing = sorted(key for key in expected if key not in payload)
    if missing:
        raise ValueError(
            "instance.segmentation_metadata is missing required fields: "
            + ", ".join(missing)
            + "."
        )
    if unexpected:
        raise ValueError(
            "instance.segmentation_metadata contains unsupported fields: "
            + ", ".join(unexpected)
            + "."
        )
    segment_number = payload["segment_number"]
    if type(segment_number) is not int or segment_number <= 0:
        raise ValueError("instance.segmentation_metadata.segment_number must be a positive integer.")
    return {
        "segment_number": segment_number,
        "segment_label": _require_non_empty_trimmed_string(
            payload["segment_label"],
            path="instance.segmentation_metadata.segment_label",
        ),
        "algorithm_type": _require_non_empty_trimmed_string(
            payload["algorithm_type"],
            path="instance.segmentation_metadata.algorithm_type",
        ),
        "algorithm_name": _require_non_empty_trimmed_string(
            payload["algorithm_name"],
            path="instance.segmentation_metadata.algorithm_name",
        ),
    }


def _canonicalise_seed(value: Any, *, path: str) -> Any:
    if type(value) is str:
        return _require_non_empty_trimmed_string(value, path=path)
    if type(value) is bool:
        return value
    if type(value) in {int, float}:
        return value
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str or not key:
                raise ValueError(f"{path} keys must be non-empty strings.")
            payload[key] = _canonicalise_seed(value[key], path=f"{path}.{key}")
        return payload
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonicalise_seed(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported seed value type {type(value).__name__!r}.")


def _object_id(object_kind: str, uid: str) -> str:
    return f"{object_kind}:{uid}"


def _validate_uid(value: Any, *, path: str) -> str:
    uid = _require_non_empty_trimmed_string(value, path=path)
    if not _UID_COMPONENT_RE.fullmatch(uid):
        raise ValueError(f"{path} must be a numeric dotted UID string.")
    if len(uid) > 64:
        raise ValueError(f"{path} must be at most 64 characters long.")
    if UID is not None and not UID(uid).is_valid:
        raise ValueError(f"{path} is not a valid DICOM UID.")
    return uid


def _require_object_kind(value: Any, *, path: str) -> str:
    kind = _require_non_empty_trimmed_string(value, path=path)
    if kind not in OBJECT_KIND_VOCABULARY:
        raise ValueError(
            f"{path} must be one of: {', '.join(OBJECT_KIND_VOCABULARY)}."
        )
    return kind


def _require_non_empty_trimmed_string(value: Any, *, path: str) -> str:
    if type(value) is not str or value == "":
        raise ValueError(f"{path} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{path} must not include leading or trailing whitespace.")
    return value


def _optional_trimmed_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_trimmed_string(value, path=path)


def _require_sha256_hex(value: Any, *, path: str) -> str:
    digest = _require_non_empty_trimmed_string(value, path=path)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{path} must be a lowercase SHA-256 hex digest.")
    return digest
