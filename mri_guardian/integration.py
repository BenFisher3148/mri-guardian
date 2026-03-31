"""Study-centred mock DICOM and PACS envelopes around the MRI Guardian Layer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from icsd.core.serialise import hash_canonical_json
from icsd.core.transition import TransitionResult

from .dicom_model import (
    DICOM_MODEL_SCHEMA_VERSION,
    FRAME_OF_REFERENCE_STATUS_VOCABULARY,
    LINEAGE_COMPLETENESS_VOCABULARY,
    MR_IMAGE_STORAGE_UID,
    SEGMENTATION_STORAGE_UID,
    SIDECAR_TYPE_ICSD_EVIDENCE_JSON,
    build_acquisition_provenance_id,
    build_acquisition_provenance_object,
    build_document_object,
    build_instance_object,
    build_object_registry,
    build_relationship,
    build_series_object,
    build_sidecar_id,
    build_sidecar_object,
    build_study_object,
    deterministic_uid,
    sop_class_name,
    validate_acquisition_provenance_object,
    validate_relationships,
    validate_sidecar_object,
    validate_standard_like_object,
)
from .harness import MULTI_SERIES_CASE_KIND, build_case_verification_report

try:
    from pydicom.uid import UID
except Exception:  # pragma: no cover - optional dependency fallback for local execution
    UID = None

_UID_COMPONENT_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
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
_OPTIONAL_STATE_FIELDS = (
    "ai_stack_identity",
    "study_acquired_at",
    "study_timestamp_provenance",
    "spatial_geometry",
    "prior_verified_contexts",
    "prior_verified_context",
    "prior_verified_contexts_by_series",
    "prior_verified_context_by_series",
)
_AI_RUNTIME_CONTEXT_FIELDS = (
    "inference_runtime_version",
    "inference_device_class",
)


def build_mock_dicom_input(
    state: dict[str, Any],
    *,
    source_ae_title: str = "MRI-AI-SCANNER",
    destination_ae_title: str = "MRI-GUARDIAN-LAYER",
    ai_runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mocked DICOM ingress envelope for the Guardian Layer."""
    report = build_case_verification_report(state)
    runtime_context = _canonical_ai_runtime_context(ai_runtime_context)
    study = _build_study_object_from_state(state)
    source_series, relationships = _build_source_series_graph(
        state=state,
        report=report,
        study=study,
    )
    payload: dict[str, Any] = {
        "schema_version": DICOM_MODEL_SCHEMA_VERSION,
        "kind": "mock_dicom_input",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": study,
        "source_series": source_series,
        "relationships": relationships,
        "sidecars": [],
        "integration_narrative": build_guardian_integration_narrative(
            current_guardian_decision="pending_verification",
        ),
        "debug": {
            "guardian_case_snapshot": deepcopy(state),
        },
    }
    if runtime_context is not None:
        payload["ai_runtime_context"] = runtime_context
    return _validate_mock_dicom_input_payload(payload)


def restore_case_state_from_dicom_input(dicom_input: dict[str, Any]) -> dict[str, Any]:
    """Return the Guardian Layer case state from the restore-only debug snapshot."""
    payload = _validate_mock_dicom_input_payload(dicom_input)
    debug = payload.get("debug")
    if not isinstance(debug, Mapping):
        raise ValueError("Mocked DICOM input is missing debug.guardian_case_snapshot.")
    case_state = debug.get("guardian_case_snapshot")
    if not isinstance(case_state, Mapping):
        raise ValueError("Mocked DICOM input is missing debug.guardian_case_snapshot.")
    raw_state = deepcopy(dict(case_state))
    if raw_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return raw_state
    if all(field in raw_state for field in _STATE_FIELDS):
        restored_state = {field: deepcopy(raw_state[field]) for field in _STATE_FIELDS}
        for field in _OPTIONAL_STATE_FIELDS:
            if field in raw_state:
                restored_state[field] = deepcopy(raw_state[field])
        return restored_state
    return raw_state


def build_mock_pacs_submission(
    result: TransitionResult,
    *,
    source_ae_title: str = "MRI-GUARDIAN-LAYER",
    destination_ae_title: str = "NHS-RAD-WS-PACS",
    bundle_reference: str | None = None,
    ai_runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mocked PACS handoff envelope for an accepted case."""
    state = result.after_state
    report = build_case_verification_report(state)
    if not report["release_eligible"]:
        raise ValueError("Mock PACS submission requires a release-eligible case.")
    runtime_context = _canonical_ai_runtime_context(ai_runtime_context)
    study = _build_study_object_from_state(state)
    source_series, relationships, source_index = _build_source_series_graph(
        state=result.before_state,
        report=build_case_verification_report(result.before_state),
        study=study,
        return_index=True,
    )
    released_series, release_relationships, manifest_document = _build_released_series_graph(
        result=result,
        report=report,
        study=study,
        source_index=source_index,
    )
    withheld_outputs, withheld_relationships = _build_withheld_outputs_for_release(
        state=state,
        report=report,
        study=study,
        source_index=source_index,
    )
    sidecars = _build_release_sidecars(
        result=result,
        manifest_document=manifest_document,
        bundle_reference=bundle_reference,
    )
    relationships.extend(release_relationships)
    relationships.extend(withheld_relationships)
    relationships.extend(
        build_relationship(
            relationship_type="sidecar_for",
            source_object_id=sidecar["object_id"],
            source_kind="sidecar",
            target_object_id=manifest_document["object_id"],
            target_kind="document",
        )
        for sidecar in sidecars
    )
    payload: dict[str, Any] = {
        "schema_version": DICOM_MODEL_SCHEMA_VERSION,
        "kind": "mock_pacs_submission",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": study,
        "source_series": source_series,
        "released_series": released_series,
        "withheld_outputs": withheld_outputs,
        "relationships": relationships,
        "sidecars": sidecars,
        "submission_status": "released",
        "verification_summary": _build_verification_summary(
            report=report,
            bundle_reference=bundle_reference,
            evidence_hash=hash_canonical_json(result.evidence.as_dict()),
            include_evidence_hash=True,
        ),
        "integration_narrative": build_guardian_integration_narrative(
            current_guardian_decision="released",
        ),
        "proof_contracts": _proof_contracts_for_report(report),
    }
    if runtime_context is not None:
        payload["ai_runtime_context"] = runtime_context
    return _validate_mock_pacs_payload(payload, expected_kind="mock_pacs_submission")


def build_mock_pacs_block_notice(
    state: dict[str, Any],
    *,
    source_ae_title: str = "MRI-GUARDIAN-LAYER",
    destination_ae_title: str = "NHS-RAD-WS-PACS",
    ai_runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mocked PACS block notice for a rejected case."""
    report = build_case_verification_report(state)
    if report["release_eligible"]:
        raise ValueError("Mock PACS block notice requires a rejected case.")
    runtime_context = _canonical_ai_runtime_context(ai_runtime_context)
    study = _build_study_object_from_state(state)
    source_series, relationships, source_index = _build_source_series_graph(
        state=state,
        report=report,
        study=study,
        return_index=True,
    )
    withheld_outputs, blocked_relationships = _build_withheld_outputs_for_block(
        state=state,
        report=report,
        study=study,
        source_index=source_index,
    )
    payload: dict[str, Any] = {
        "schema_version": DICOM_MODEL_SCHEMA_VERSION,
        "kind": "mock_pacs_block_notice",
        "transport": "dicom-c-store",
        "endpoint": {
            "source_ae_title": source_ae_title,
            "destination_ae_title": destination_ae_title,
        },
        "study": study,
        "source_series": source_series,
        "released_series": [],
        "withheld_outputs": withheld_outputs,
        "relationships": [*relationships, *blocked_relationships],
        "sidecars": [],
        "submission_status": "blocked",
        "blocked_reason": _primary_reason_summary(report),
        "failure_reasons": deepcopy(report["failure_reasons"]),
        "verification_summary": _build_verification_summary(
            report=report,
            bundle_reference=None,
            evidence_hash=None,
            include_evidence_hash=False,
        ),
        "integration_narrative": build_guardian_integration_narrative(
            current_guardian_decision="blocked",
        ),
        "proof_contracts": _proof_contracts_for_report(report),
    }
    if runtime_context is not None:
        payload["ai_runtime_context"] = runtime_context
    return _validate_mock_pacs_payload(payload, expected_kind="mock_pacs_block_notice")


def build_mock_integration_trace(
    *,
    dicom_input: dict[str, Any],
    pacs_output: dict[str, Any],
    ai_runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a mocked boundary trace from DICOM ingress to PACS egress."""
    validated_input = _validate_mock_dicom_input_payload(dicom_input)
    pacs_kind = pacs_output.get("kind")
    if pacs_kind not in {"mock_pacs_submission", "mock_pacs_block_notice"}:
        raise ValueError("Unsupported PACS output envelope.")
    validated_output = _validate_mock_pacs_payload(
        pacs_output,
        expected_kind=str(pacs_kind),
    )
    state = restore_case_state_from_dicom_input(validated_input)
    report = build_case_verification_report(state)
    runtime_context = _canonical_ai_runtime_context(
        ai_runtime_context
        if ai_runtime_context is not None
        else validated_output.get("ai_runtime_context", validated_input.get("ai_runtime_context"))
    )
    ingress_summary = _graph_summary(
        source_series=validated_input["source_series"],
        released_series=[],
        withheld_outputs=[],
        sidecars=[],
    )
    egress_summary = _graph_summary(
        source_series=validated_output["source_series"],
        released_series=validated_output["released_series"],
        withheld_outputs=validated_output["withheld_outputs"],
        sidecars=validated_output["sidecars"],
    )
    payload: dict[str, Any] = {
        "schema_version": DICOM_MODEL_SCHEMA_VERSION,
        "kind": "mock_integration_trace",
        "case_id": _string_or_fallback(state.get("case_id"), "unknown-case"),
        "integration_narrative": build_guardian_integration_narrative(
            current_guardian_decision=str(validated_output["submission_status"]),
        ),
        "pipeline": [
            _build_ingress_trace_stage(
                state=state,
                dicom_input=validated_input,
                summary=ingress_summary,
            ),
            _build_guardian_trace_stage(
                report=report,
                submission_status=str(validated_output["submission_status"]),
                egress_summary=egress_summary,
            ),
            _build_egress_trace_stage(
                pacs_output=validated_output,
                summary=egress_summary,
            ),
        ],
        "hashes": {
            "dicom_input_hash": hash_canonical_json(validated_input),
            "pacs_output_hash": hash_canonical_json(validated_output),
        },
    }
    if runtime_context is not None:
        payload["ai_runtime_context"] = runtime_context
    return payload


def build_guardian_integration_narrative(
    *,
    current_guardian_decision: str,
) -> dict[str, Any]:
    """Return a deterministic hospital-integration narrative for the Guardian Layer."""
    return {
        "interface_type": "GuardianIntegrationNarrative",
        "interface_version": DICOM_MODEL_SCHEMA_VERSION,
        "guardian_role": "verification_proxy",
        "dicom_compliance_story": {
            "ingress_transport": "dicom-c-store",
            "ingress_objects": [
                "MR Image Storage (mock-reviewed)",
                "Segmentation Storage (mock-reviewed)",
                "SR-like review artifact",
                "Acquisition provenance record",
            ],
            "egress_transport": "dicom-c-store",
            "egress_objects": [
                "MR Image Storage (mock-reviewed)",
                "Segmentation Storage (mock-reviewed)",
                "SR-like verification manifest",
                "JSON evidence sidecar",
            ],
            "story": (
                "The Guardian Layer uses a deterministic DICOM-shaped graph for reviewer "
                "familiarity and interoperability preparation, but it does not claim "
                "standards-certified DICOM dataset or SR byte compatibility."
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
                "No verified reconstruction, segmentation, reviewer-facing verification "
                "manifest, or evidence sidecar is forwarded from the Guardian path."
            ),
            "on_service_unavailable": (
                "The Guardian path does not auto-pass AI output through. Cases remain pending "
                "or require an explicit external fallback workflow outside this proxy."
            ),
        },
    }


def _build_study_object_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    source_case_id = _string_or_fallback(state.get("case_id"), "unknown-case")
    source_study_key = _string_or_fallback(state.get("study_uid"), "unknown-study")
    return build_study_object(
        study_instance_uid=deterministic_uid(
            object_kind="study",
            seed={
                "source_case_id": source_case_id,
                "source_study_key": source_study_key,
                "case_kind": _case_kind_from_state(state),
            },
        ),
        patient_id=_string_or_fallback(state.get("patient_pseudonym"), "unknown-patient"),
        extensions={
            "source_case_id": source_case_id,
            "source_study_key": source_study_key,
            "case_kind": _case_kind_from_state(state),
        },
    )


def _build_source_series_graph(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    return_index: bool = False,
) -> Any:
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    source_index: dict[str, dict[str, Any]] = {}
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        for series_report in report.get("series_reports", []):
            series_state = series_report["series_state"]
            series_key = _string_or_fallback(series_report.get("series_id"), "unknown-series-id")
            built = _build_source_payload_series(
                series_state=series_state,
                report=series_report,
                study=study,
                source_series_key=series_key,
                source_series_uid=_string_or_fallback(
                    series_report.get("series_uid"),
                    "unknown-series",
                ),
                resolved_role=_optional_string(series_report.get("resolved_role")),
                gating_status=_optional_string(series_report.get("gating_status")),
            )
            groups.extend(built["groups"])
            relationships.extend(built["relationships"])
            source_index[series_key] = built["index"]
    else:
        series_key = _string_or_fallback(state.get("series_uid"), "unknown-series")
        built = _build_source_payload_series(
            series_state=state,
            report=report,
            study=study,
            source_series_key=series_key,
            source_series_uid=series_key,
            resolved_role=None,
            gating_status=None,
        )
        groups.extend(built["groups"])
        relationships.extend(built["relationships"])
        source_index[series_key] = built["index"]
        source_index["single"] = built["index"]
    if return_index:
        return groups, relationships, source_index
    return groups, relationships


def _build_source_payload_series(
    *,
    series_state: Mapping[str, Any],
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    source_series_key: str,
    source_series_uid: str,
    resolved_role: str | None,
    gating_status: str | None,
) -> dict[str, Any]:
    study_uid = str(study["study_instance_uid"])
    frame_uid, frame_status = _frame_of_reference_lineage(series_state)
    source_identity = {
        "source_series_key": source_series_key,
        "source_series_uid": source_series_uid,
        "source_study_key": study["extensions"]["guardian_v1"]["source_study_key"],
    }
    source_hashes = _source_hashes(series_state, report)
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    mr_series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": "source_reconstruction_series",
                "source_identity": source_identity,
            },
        ),
        modality="MR",
        semantic_role="source_reconstruction_series",
        series_description="AI Reconstruction Candidate",
        frame_of_reference_uid=frame_uid,
        extensions=_series_extensions(
            source_series_key=source_series_key,
            source_series_uid=source_series_uid,
            resolved_role=resolved_role,
            gating_status=gating_status,
            output_status="source",
        ),
    )
    mr_instance = None
    mr_instances: list[dict[str, Any]] = []
    if source_hashes["ai_reconstruction_hash"] is not None:
        mr_instance = build_instance_object(
            sop_instance_uid=deterministic_uid(
                object_kind="instance",
                seed={
                    "parent_series_uid": mr_series["series_instance_uid"],
                    "semantic_role": "ai_reconstruction_image",
                    "source_hash": source_hashes["ai_reconstruction_hash"],
                },
            ),
            sop_class_uid=MR_IMAGE_STORAGE_UID,
            sop_class_name=sop_class_name(MR_IMAGE_STORAGE_UID),
            semantic_role="ai_reconstruction_image",
            content_hash=source_hashes["ai_reconstruction_hash"],
            extensions={"source_series_key": source_series_key},
        )
        mr_instances.append(mr_instance)
    acquisition_objects: list[dict[str, Any]] = []
    if source_hashes["raw_input_hash"] is not None:
        acquisition = build_acquisition_provenance_object(
            object_id=build_acquisition_provenance_id(
                seed={
                    "parent_series_uid": mr_series["series_instance_uid"],
                    "source_series_key": source_series_key,
                    "raw_input_hash": source_hashes["raw_input_hash"],
                }
            ),
            provenance_type="raw_input_lineage",
            raw_input_hash=source_hashes["raw_input_hash"],
            source_members=["raw_real", "raw_imag"],
            extensions={"source_series_key": source_series_key},
        )
        acquisition_objects.append(acquisition)
    groups.append(
        _series_group(
            status="source",
            series=mr_series,
            instances=mr_instances,
            acquisition_provenance=acquisition_objects,
            frame_of_reference_status=frame_status,
            lineage_completeness=(
                "complete"
                if mr_instance is not None and acquisition_objects
                else "partial"
            ),
            source_series_object_ids=[],
            source_instance_object_ids=[],
        )
    )
    relationships.append(
        build_relationship(
            relationship_type="contains_series",
            source_object_id=study["object_id"],
            source_kind="study",
            target_object_id=mr_series["object_id"],
            target_kind="series",
        )
    )
    for mr_object in mr_instances:
        relationships.append(
            build_relationship(
                relationship_type="contains_instance",
                source_object_id=mr_series["object_id"],
                source_kind="series",
                target_object_id=mr_object["object_id"],
                target_kind="instance",
            )
        )
    for acquisition in acquisition_objects:
        relationships.append(
            build_relationship(
                relationship_type="references",
                source_object_id=mr_series["object_id"],
                source_kind="series",
                target_object_id=acquisition["object_id"],
                target_kind="acquisition_provenance",
            )
        )

    seg_series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": "source_segmentation_series",
                "source_identity": source_identity,
            },
        ),
        modality="SEG",
        semantic_role="source_segmentation_series",
        series_description="AI Tumour Mask Candidate",
        frame_of_reference_uid=frame_uid,
        extensions=_series_extensions(
            source_series_key=source_series_key,
            source_series_uid=source_series_uid,
            resolved_role=resolved_role,
            gating_status=gating_status,
            output_status="source",
        ),
    )
    seg_instance = None
    seg_instances: list[dict[str, Any]] = []
    if source_hashes["ai_mask_hash"] is not None:
        seg_instance = build_instance_object(
            sop_instance_uid=deterministic_uid(
                object_kind="instance",
                seed={
                    "parent_series_uid": seg_series["series_instance_uid"],
                    "semantic_role": "ai_segmentation_mask",
                    "source_hash": source_hashes["ai_mask_hash"],
                },
            ),
            sop_class_uid=SEGMENTATION_STORAGE_UID,
            sop_class_name=sop_class_name(SEGMENTATION_STORAGE_UID),
            semantic_role="ai_segmentation_mask",
            content_hash=source_hashes["ai_mask_hash"],
            referenced_instance_uids=(
                []
                if mr_instance is None
                else [mr_instance["sop_instance_uid"]]
            ),
            segmentation_metadata={
                "segment_number": 1,
                "segment_label": "Tumour",
                "algorithm_type": "AUTOMATIC",
                "algorithm_name": _string_or_fallback(
                    series_state.get("ai_model_id"),
                    "unknown-ai-model",
                ),
            },
            extensions={"source_series_key": source_series_key},
        )
        seg_instances.append(seg_instance)
    groups.append(
        _series_group(
            status="source",
            series=seg_series,
            instances=seg_instances,
            acquisition_provenance=[],
            frame_of_reference_status=frame_status,
            lineage_completeness="complete" if seg_instance is not None else "partial",
            source_series_object_ids=[mr_series["object_id"]],
            source_instance_object_ids=(
                [] if mr_instance is None else [mr_instance["object_id"]]
            ),
        )
    )
    relationships.append(
        build_relationship(
            relationship_type="contains_series",
            source_object_id=study["object_id"],
            source_kind="study",
            target_object_id=seg_series["object_id"],
            target_kind="series",
        )
    )
    for seg_object in seg_instances:
        relationships.append(
            build_relationship(
                relationship_type="contains_instance",
                source_object_id=seg_series["object_id"],
                source_kind="series",
                target_object_id=seg_object["object_id"],
                target_kind="instance",
            )
        )
    if seg_instance is not None and mr_instance is not None:
        relationships.append(
            build_relationship(
                relationship_type="derived_from",
                source_object_id=seg_instance["object_id"],
                source_kind="instance",
                target_object_id=mr_instance["object_id"],
                target_kind="instance",
            )
        )
    if frame_uid is not None:
        relationships.append(
            build_relationship(
                relationship_type="same_frame_of_reference",
                source_object_id=mr_series["object_id"],
                source_kind="series",
                target_object_id=seg_series["object_id"],
                target_kind="series",
            )
        )

    claim_series = None
    claim_document = None
    claim_hash = source_hashes["claim_outputs_hash"]
    if claim_hash is not None:
        claim_series = build_series_object(
            series_instance_uid=deterministic_uid(
                object_kind="series",
                seed={
                    "parent_study_uid": study_uid,
                    "semantic_role": "source_claim_series",
                    "source_identity": source_identity,
                },
            ),
            modality="SR",
            semantic_role="source_claim_series",
            series_description="AI Volume Claim Review Artifact",
            extensions=_series_extensions(
                source_series_key=source_series_key,
                source_series_uid=source_series_uid,
                resolved_role=resolved_role,
                gating_status=gating_status,
                output_status="source",
            ),
        )
        referenced_uids: list[str] = []
        source_instance_object_ids: list[str] = []
        if mr_instance is not None:
            referenced_uids.append(mr_instance["sop_instance_uid"])
            source_instance_object_ids.append(mr_instance["object_id"])
        if seg_instance is not None:
            referenced_uids.append(seg_instance["sop_instance_uid"])
            source_instance_object_ids.append(seg_instance["object_id"])
        claim_document = build_document_object(
            sop_instance_uid=deterministic_uid(
                object_kind="document",
                seed={
                    "parent_series_uid": claim_series["series_instance_uid"],
                    "semantic_role": "ai_volume_claim_document",
                    "source_hash": claim_hash,
                },
            ),
            semantic_role="ai_volume_claim_document",
            document_kind="ai_volume_claim_sr_like",
            content_hash=claim_hash,
            referenced_instance_uids=referenced_uids,
            measurement_groups=_claim_measurements(series_state),
            observer_context={
                "observer_type": "software",
                "observer_name": _string_or_fallback(
                    series_state.get("ai_model_id"),
                    "unknown-ai-model",
                ),
            },
            extensions={"source_series_key": source_series_key},
        )
        groups.append(
            _series_group(
                status="source",
                series=claim_series,
                instances=[claim_document],
                acquisition_provenance=[],
                frame_of_reference_status="not_applicable",
                lineage_completeness="complete",
                source_series_object_ids=[mr_series["object_id"], seg_series["object_id"]],
                source_instance_object_ids=source_instance_object_ids,
            )
        )
        relationships.extend(
            [
                build_relationship(
                    relationship_type="contains_series",
                    source_object_id=study["object_id"],
                    source_kind="study",
                    target_object_id=claim_series["object_id"],
                    target_kind="series",
                ),
                build_relationship(
                    relationship_type="contains_instance",
                    source_object_id=claim_series["object_id"],
                    source_kind="series",
                    target_object_id=claim_document["object_id"],
                    target_kind="document",
                ),
            ]
        )
        for source_object in (mr_instance, seg_instance):
            if source_object is not None:
                relationships.append(
                    build_relationship(
                        relationship_type="references",
                        source_object_id=claim_document["object_id"],
                        source_kind="document",
                        target_object_id=source_object["object_id"],
                        target_kind="instance",
                    )
                )

    return {
        "groups": groups,
        "relationships": relationships,
        "index": {
            "source_series_key": source_series_key,
            "source_series_uid": source_series_uid,
            "resolved_role": resolved_role,
            "gating_status": gating_status,
            "mr_series": mr_series,
            "mr_instance": mr_instance,
            "seg_series": seg_series,
            "seg_instance": seg_instance,
            "claim_series": claim_series,
            "claim_document": claim_document,
            "frame_of_reference_uid": frame_uid,
            "frame_of_reference_status": frame_status,
        },
    }


def _build_released_series_graph(
    *,
    result: TransitionResult,
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    source_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    state = result.after_state
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    manifest_sources: list[Mapping[str, Any]] = []
    released_instances: list[dict[str, Any]] = []
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        series_reports_by_id = {
            _string_or_fallback(item.get("series_id"), "unknown-series-id"): item
            for item in report.get("series_reports", [])
        }
        for payload_entry in state["workstation_payload"]["series_payloads"]:
            source_series_key = _string_or_fallback(payload_entry.get("series_id"), "unknown-series-id")
            series_report = series_reports_by_id[source_series_key]
            source_entry = source_index[source_series_key]
            built = _build_released_output_pair(
                study=study,
                source_entry=source_entry,
                source_series_key=source_series_key,
                source_series_uid=_string_or_fallback(
                    payload_entry.get("series_uid"),
                    source_series_key,
                ),
                workstation_payload=payload_entry["workstation_payload"],
                resolved_role=_optional_string(payload_entry.get("resolved_role")),
                gating_status=_optional_string(payload_entry.get("gating_status")),
                ai_model_id=_string_or_fallback(
                    series_report["series_state"].get("ai_model_id"),
                    "unknown-ai-model",
                ),
            )
            groups.extend(built["groups"])
            relationships.extend(built["relationships"])
            released_instances.extend(built["released_instances"])
            manifest_sources.append(source_entry)
    else:
        source_entry = source_index.get(
            _string_or_fallback(state.get("series_uid"), "unknown-series"),
            source_index["single"],
        )
        built = _build_released_output_pair(
            study=study,
            source_entry=source_entry,
            source_series_key=_string_or_fallback(state.get("series_uid"), "unknown-series"),
            source_series_uid=_string_or_fallback(state.get("series_uid"), "unknown-series"),
            workstation_payload=state["workstation_payload"],
            resolved_role=None,
            gating_status=None,
            ai_model_id=_string_or_fallback(state.get("ai_model_id"), "unknown-ai-model"),
        )
        groups.extend(built["groups"])
        relationships.extend(built["relationships"])
        released_instances.extend(built["released_instances"])
        manifest_sources.append(source_entry)
    manifest_group, manifest_relationships, manifest_document = _build_manifest_release_group(
        state=state,
        report=report,
        study=study,
        source_entries=manifest_sources,
        released_instances=released_instances,
    )
    groups.append(manifest_group)
    relationships.extend(manifest_relationships)
    return groups, relationships, manifest_document


def _build_released_output_pair(
    *,
    study: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    source_series_key: str,
    source_series_uid: str,
    workstation_payload: Mapping[str, Any],
    resolved_role: str | None,
    gating_status: str | None,
    ai_model_id: str,
) -> dict[str, Any]:
    study_uid = str(study["study_instance_uid"])
    frame_uid = source_entry.get("frame_of_reference_uid")
    frame_status = str(source_entry.get("frame_of_reference_status", "no_upstream_lineage"))
    source_series_object_ids = [
        source_entry["mr_series"]["object_id"],
        source_entry["seg_series"]["object_id"],
    ]
    source_instance_object_ids = [
        item["object_id"]
        for item in (source_entry.get("mr_instance"), source_entry.get("seg_instance"))
        if item is not None
    ]
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    reconstruction_series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": "verified_reconstruction_series",
                "source_series_key": source_series_key,
                "source_series_uid": source_series_uid,
            },
        ),
        modality="MR",
        semantic_role="verified_reconstruction_series",
        series_description="Guardian Verified Reconstruction",
        frame_of_reference_uid=frame_uid,
        extensions=_series_extensions(
            source_series_key=source_series_key,
            source_series_uid=source_series_uid,
            resolved_role=resolved_role,
            gating_status=gating_status,
            output_status="released",
            downstream_release_allowed=True,
        ),
    )
    reconstruction_instance = build_instance_object(
        sop_instance_uid=deterministic_uid(
            object_kind="instance",
            seed={
                "parent_series_uid": reconstruction_series["series_instance_uid"],
                "semantic_role": "verified_reconstruction_image",
                "source_series_key": source_series_key,
                "content_hash": workstation_payload["reconstruction_hash"],
            },
        ),
        sop_class_uid=MR_IMAGE_STORAGE_UID,
        sop_class_name=sop_class_name(MR_IMAGE_STORAGE_UID),
        semantic_role="verified_reconstruction_image",
        content_hash=str(workstation_payload["reconstruction_hash"]),
        extensions={"source_series_key": source_series_key},
    )
    groups.append(
        _series_group(
            status="released",
            series=reconstruction_series,
            instances=[reconstruction_instance],
            acquisition_provenance=[],
            frame_of_reference_status=frame_status,
            lineage_completeness="complete" if source_entry.get("mr_instance") is not None else "partial",
            source_series_object_ids=[source_entry["mr_series"]["object_id"]],
            source_instance_object_ids=(
                [] if source_entry.get("mr_instance") is None else [source_entry["mr_instance"]["object_id"]]
            ),
        )
    )
    relationships.extend(
        [
            build_relationship(
                relationship_type="contains_series",
                source_object_id=study["object_id"],
                source_kind="study",
                target_object_id=reconstruction_series["object_id"],
                target_kind="series",
            ),
            build_relationship(
                relationship_type="contains_instance",
                source_object_id=reconstruction_series["object_id"],
                source_kind="series",
                target_object_id=reconstruction_instance["object_id"],
                target_kind="instance",
            ),
        ]
    )
    if source_entry.get("mr_instance") is not None:
        relationships.append(
            build_relationship(
                relationship_type="derived_from",
                source_object_id=reconstruction_instance["object_id"],
                source_kind="instance",
                target_object_id=source_entry["mr_instance"]["object_id"],
                target_kind="instance",
            )
        )

    segmentation_series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": "verified_segmentation_series",
                "source_series_key": source_series_key,
                "source_series_uid": source_series_uid,
            },
        ),
        modality="SEG",
        semantic_role="verified_segmentation_series",
        series_description="Guardian Verified Tumour Mask",
        frame_of_reference_uid=frame_uid,
        extensions=_series_extensions(
            source_series_key=source_series_key,
            source_series_uid=source_series_uid,
            resolved_role=resolved_role,
            gating_status=gating_status,
            output_status="released",
            downstream_release_allowed=True,
        ),
    )
    segmentation_instance = build_instance_object(
        sop_instance_uid=deterministic_uid(
            object_kind="instance",
            seed={
                "parent_series_uid": segmentation_series["series_instance_uid"],
                "semantic_role": "verified_segmentation_mask",
                "source_series_key": source_series_key,
                "content_hash": workstation_payload["verified_mask_hash"],
            },
        ),
        sop_class_uid=SEGMENTATION_STORAGE_UID,
        sop_class_name=sop_class_name(SEGMENTATION_STORAGE_UID),
        semantic_role="verified_segmentation_mask",
        content_hash=str(workstation_payload["verified_mask_hash"]),
        referenced_instance_uids=[reconstruction_instance["sop_instance_uid"]],
        segmentation_metadata={
            "segment_number": 1,
            "segment_label": "Tumour",
            "algorithm_type": "AUTOMATIC",
            "algorithm_name": ai_model_id,
        },
        extensions={"source_series_key": source_series_key},
    )
    groups.append(
        _series_group(
            status="released",
            series=segmentation_series,
            instances=[segmentation_instance],
            acquisition_provenance=[],
            frame_of_reference_status=frame_status,
            lineage_completeness="complete" if source_entry.get("seg_instance") is not None else "partial",
            source_series_object_ids=source_series_object_ids,
            source_instance_object_ids=source_instance_object_ids,
        )
    )
    relationships.extend(
        [
            build_relationship(
                relationship_type="contains_series",
                source_object_id=study["object_id"],
                source_kind="study",
                target_object_id=segmentation_series["object_id"],
                target_kind="series",
            ),
            build_relationship(
                relationship_type="contains_instance",
                source_object_id=segmentation_series["object_id"],
                source_kind="series",
                target_object_id=segmentation_instance["object_id"],
                target_kind="instance",
            ),
            build_relationship(
                relationship_type="references",
                source_object_id=segmentation_instance["object_id"],
                source_kind="instance",
                target_object_id=reconstruction_instance["object_id"],
                target_kind="instance",
            ),
        ]
    )
    if source_entry.get("seg_instance") is not None:
        relationships.append(
            build_relationship(
                relationship_type="derived_from",
                source_object_id=segmentation_instance["object_id"],
                source_kind="instance",
                target_object_id=source_entry["seg_instance"]["object_id"],
                target_kind="instance",
            )
        )
    if frame_uid is not None:
        relationships.append(
            build_relationship(
                relationship_type="same_frame_of_reference",
                source_object_id=reconstruction_series["object_id"],
                source_kind="series",
                target_object_id=segmentation_series["object_id"],
                target_kind="series",
            )
        )
    return {
        "groups": groups,
        "relationships": relationships,
        "released_instances": [reconstruction_instance, segmentation_instance],
    }


def _build_manifest_release_group(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    source_entries: Sequence[Mapping[str, Any]],
    released_instances: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    study_uid = str(study["study_instance_uid"])
    case_kind = _case_kind_from_state(state)
    is_multi = case_kind == MULTI_SERIES_CASE_KIND
    semantic_role = (
        "guardian_study_verification_manifest_series"
        if is_multi
        else "guardian_verification_manifest_series"
    )
    document_role = (
        "guardian_study_verification_manifest_document"
        if is_multi
        else "guardian_verification_manifest_document"
    )
    document_kind = (
        "guardian_study_verification_manifest_sr_like"
        if is_multi
        else "guardian_verification_manifest_sr_like"
    )
    manifest_series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": semantic_role,
                "case_kind": case_kind,
                "source_series_keys": sorted(
                    _string_or_fallback(item.get("source_series_key"), "unknown-series")
                    for item in source_entries
                ),
            },
        ),
        modality="SR",
        semantic_role=semantic_role,
        series_description=(
            "Guardian Study Verification Manifest"
            if is_multi
            else "Guardian Verification Manifest"
        ),
        extensions={
            "output_status": "released",
            "downstream_release_allowed": True,
        },
    )
    referenced_instance_uids = sorted(
        str(item["sop_instance_uid"]) for item in released_instances
    )
    manifest_hash = hash_canonical_json(state["verified_manifest"])
    manifest_document = build_document_object(
        sop_instance_uid=deterministic_uid(
            object_kind="document",
            seed={
                "parent_series_uid": manifest_series["series_instance_uid"],
                "semantic_role": document_role,
                "content_hash": manifest_hash,
            },
        ),
        semantic_role=document_role,
        document_kind=document_kind,
        content_hash=manifest_hash,
        referenced_instance_uids=referenced_instance_uids,
        measurement_groups=_manifest_measurements(state=state, report=report),
        observer_context={
            "observer_type": "software",
            "observer_name": "MRI Guardian Layer",
        },
        extensions={
            "profile_name": _string_or_fallback(report["profile"].get("profile_name"), "unknown-profile"),
            "decision_class": _string_or_fallback(
                report["decision_summary"].get("decision_class"),
                "unknown-decision",
            ),
        },
    )
    source_series_object_ids: list[str] = []
    source_instance_object_ids: list[str] = []
    for item in source_entries:
        for series_object in ("mr_series", "seg_series", "claim_series"):
            series_payload = item.get(series_object)
            if isinstance(series_payload, Mapping):
                source_series_object_ids.append(str(series_payload["object_id"]))
        for instance_object in ("mr_instance", "seg_instance", "claim_document"):
            instance_payload = item.get(instance_object)
            if isinstance(instance_payload, Mapping):
                source_instance_object_ids.append(str(instance_payload["object_id"]))
    group = _series_group(
        status="released",
        series=manifest_series,
        instances=[manifest_document],
        acquisition_provenance=[],
        frame_of_reference_status="not_applicable",
        lineage_completeness="complete",
        source_series_object_ids=source_series_object_ids,
        source_instance_object_ids=source_instance_object_ids,
    )
    relationships = [
        build_relationship(
            relationship_type="contains_series",
            source_object_id=study["object_id"],
            source_kind="study",
            target_object_id=manifest_series["object_id"],
            target_kind="series",
        ),
        build_relationship(
            relationship_type="contains_instance",
            source_object_id=manifest_series["object_id"],
            source_kind="series",
            target_object_id=manifest_document["object_id"],
            target_kind="document",
        ),
    ]
    released_instance_by_uid = {
        str(item["sop_instance_uid"]): item for item in released_instances
    }
    for referenced_uid in referenced_instance_uids:
        referenced_object = released_instance_by_uid[referenced_uid]
        relationships.append(
            build_relationship(
                relationship_type="references",
                source_object_id=manifest_document["object_id"],
                source_kind="document",
                target_object_id=referenced_object["object_id"],
                target_kind="instance",
            )
        )
    for source_entry in source_entries:
        claim_document = source_entry.get("claim_document")
        if isinstance(claim_document, Mapping):
            relationships.append(
                build_relationship(
                    relationship_type="derived_from",
                    source_object_id=manifest_document["object_id"],
                    source_kind="document",
                    target_object_id=str(claim_document["object_id"]),
                    target_kind="document",
                )
            )
    return group, relationships, manifest_document


def _build_release_sidecars(
    *,
    result: TransitionResult,
    manifest_document: Mapping[str, Any],
    bundle_reference: str | None,
) -> list[dict[str, Any]]:
    evidence_hash = hash_canonical_json(result.evidence.as_dict())
    sidecar = build_sidecar_object(
        sidecar_id=build_sidecar_id(
            seed={
                "manifest_instance_uid": manifest_document["sop_instance_uid"],
                "sidecar_type": SIDECAR_TYPE_ICSD_EVIDENCE_JSON,
                "evidence_hash": evidence_hash,
            }
        ),
        sidecar_type=SIDECAR_TYPE_ICSD_EVIDENCE_JSON,
        media_type="application/json",
        hash_value=evidence_hash,
        referenced_manifest_instance_uid=str(manifest_document["sop_instance_uid"]),
        bundle_reference=bundle_reference,
        bundle_member="evidence.json",
    )
    return [sidecar]


def _build_withheld_outputs_for_release(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    source_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if state.get("case_kind") != MULTI_SERIES_CASE_KIND:
        return [], []
    withheld_groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    withheld_entries = state["workstation_payload"].get("withheld_series", [])
    for withheld_entry in withheld_entries:
        source_series_key = _string_or_fallback(withheld_entry.get("series_id"), "unknown-series-id")
        source_entry = source_index[source_series_key]
        built = _build_withheld_output_pair(
            study=study,
            source_entry=source_entry,
            source_series_key=source_series_key,
            source_series_uid=_string_or_fallback(
                withheld_entry.get("series_uid"),
                source_series_key,
            ),
            resolved_role=_optional_string(withheld_entry.get("resolved_role")),
            gating_status=_optional_string(withheld_entry.get("gating_status")),
            withheld_reason=_string_or_fallback(
                withheld_entry.get("withheld_reason"),
                _primary_reason_summary(report),
            ),
        )
        withheld_groups.extend(built["groups"])
        relationships.extend(built["relationships"])
    return withheld_groups, relationships


def _build_withheld_outputs_for_block(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    study: Mapping[str, Any],
    source_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    withheld_groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        for series_report in report.get("series_reports", []):
            source_series_key = _string_or_fallback(series_report.get("series_id"), "unknown-series-id")
            source_entry = source_index[source_series_key]
            withheld_reason = _primary_reason_summary(report)
            if not bool(series_report.get("downstream_release_allowed")):
                withheld_reason = _string_or_fallback(
                    series_report.get("primary_reason_summary"),
                    withheld_reason,
                )
            built = _build_withheld_output_pair(
                study=study,
                source_entry=source_entry,
                source_series_key=source_series_key,
                source_series_uid=_string_or_fallback(
                    series_report.get("series_uid"),
                    source_series_key,
                ),
                resolved_role=_optional_string(series_report.get("resolved_role")),
                gating_status=_optional_string(series_report.get("gating_status")),
                withheld_reason=withheld_reason,
            )
            withheld_groups.extend(built["groups"])
            relationships.extend(built["relationships"])
        manifest_group = _build_withheld_manifest_group(
            study=study,
            state=state,
            report=report,
            source_entries=[
                source_index[_string_or_fallback(item.get("series_id"), "unknown-series-id")]
                for item in report.get("series_reports", [])
            ],
            withheld_reason=_primary_reason_summary(report),
        )
        withheld_groups.append(manifest_group["group"])
        relationships.extend(manifest_group["relationships"])
        return withheld_groups, relationships

    source_entry = source_index.get(
        _string_or_fallback(state.get("series_uid"), "unknown-series"),
        source_index["single"],
    )
    built = _build_withheld_output_pair(
        study=study,
        source_entry=source_entry,
        source_series_key=_string_or_fallback(state.get("series_uid"), "unknown-series"),
        source_series_uid=_string_or_fallback(state.get("series_uid"), "unknown-series"),
        resolved_role=None,
        gating_status=None,
        withheld_reason=_primary_reason_summary(report),
    )
    withheld_groups.extend(built["groups"])
    relationships.extend(built["relationships"])
    manifest_group = _build_withheld_manifest_group(
        study=study,
        state=state,
        report=report,
        source_entries=[source_entry],
        withheld_reason=_primary_reason_summary(report),
    )
    withheld_groups.append(manifest_group["group"])
    relationships.extend(manifest_group["relationships"])
    return withheld_groups, relationships


def _build_withheld_output_pair(
    *,
    study: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    source_series_key: str,
    source_series_uid: str,
    resolved_role: str | None,
    gating_status: str | None,
    withheld_reason: str,
) -> dict[str, Any]:
    study_uid = str(study["study_instance_uid"])
    frame_uid = source_entry.get("frame_of_reference_uid")
    frame_status = str(source_entry.get("frame_of_reference_status", "no_upstream_lineage"))
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    withheld_specs = (
        (
            "withheld_reconstruction_series",
            "MR",
            "Guardian Output Withheld - Reconstruction",
            source_entry["mr_series"],
            frame_uid,
            frame_status,
            [source_entry["mr_series"]["object_id"]],
            [] if source_entry.get("mr_instance") is None else [source_entry["mr_instance"]["object_id"]],
        ),
        (
            "withheld_segmentation_series",
            "SEG",
            "Guardian Output Withheld - Segmentation",
            source_entry["seg_series"],
            frame_uid,
            frame_status,
            [
                source_entry["mr_series"]["object_id"],
                source_entry["seg_series"]["object_id"],
            ],
            [
                item["object_id"]
                for item in (source_entry.get("mr_instance"), source_entry.get("seg_instance"))
                if item is not None
            ],
        ),
    )
    for semantic_role, modality, description, blocked_target, group_frame_uid, group_frame_status, source_series_ids, source_instance_ids in withheld_specs:
        series = build_series_object(
            series_instance_uid=deterministic_uid(
                object_kind="series",
                seed={
                    "parent_study_uid": study_uid,
                    "semantic_role": semantic_role,
                    "source_series_key": source_series_key,
                    "source_series_uid": source_series_uid,
                },
            ),
            modality=modality,
            semantic_role=semantic_role,
            series_description=description,
            frame_of_reference_uid=group_frame_uid,
            extensions=_series_extensions(
                source_series_key=source_series_key,
                source_series_uid=source_series_uid,
                resolved_role=resolved_role,
                gating_status=gating_status,
                output_status="withheld",
                downstream_release_allowed=False,
                withheld_reason=withheld_reason,
            ),
        )
        groups.append(
            _series_group(
                status="withheld",
                series=series,
                instances=[],
                acquisition_provenance=[],
                frame_of_reference_status=group_frame_status,
                lineage_completeness="partial",
                source_series_object_ids=source_series_ids,
                source_instance_object_ids=source_instance_ids,
                withheld_reason=withheld_reason,
            )
        )
        relationships.extend(
            [
                build_relationship(
                    relationship_type="contains_series",
                    source_object_id=study["object_id"],
                    source_kind="study",
                    target_object_id=series["object_id"],
                    target_kind="series",
                ),
                build_relationship(
                    relationship_type="blocked_output_for",
                    source_object_id=series["object_id"],
                    source_kind="series",
                    target_object_id=blocked_target["object_id"],
                    target_kind="series",
                ),
            ]
        )
    return {"groups": groups, "relationships": relationships}


def _build_withheld_manifest_group(
    *,
    study: Mapping[str, Any],
    state: Mapping[str, Any],
    report: Mapping[str, Any],
    source_entries: Sequence[Mapping[str, Any]],
    withheld_reason: str,
) -> dict[str, Any]:
    study_uid = str(study["study_instance_uid"])
    case_kind = _case_kind_from_state(state)
    semantic_role = (
        "withheld_guardian_study_manifest_series"
        if case_kind == MULTI_SERIES_CASE_KIND
        else "withheld_guardian_manifest_series"
    )
    series = build_series_object(
        series_instance_uid=deterministic_uid(
            object_kind="series",
            seed={
                "parent_study_uid": study_uid,
                "semantic_role": semantic_role,
                "case_kind": case_kind,
                "source_series_keys": sorted(
                    _string_or_fallback(item.get("source_series_key"), "unknown-series")
                    for item in source_entries
                ),
            },
        ),
        modality="SR",
        semantic_role=semantic_role,
        series_description=(
            "Guardian Study Manifest Withheld"
            if case_kind == MULTI_SERIES_CASE_KIND
            else "Guardian Verification Manifest Withheld"
        ),
        extensions={
            "output_status": "withheld",
            "downstream_release_allowed": False,
            "withheld_reason": withheld_reason,
        },
    )
    group = _series_group(
        status="withheld",
        series=series,
        instances=[],
        acquisition_provenance=[],
        frame_of_reference_status="not_applicable",
        lineage_completeness="partial",
        source_series_object_ids=[
            str(item[key]["object_id"])
            for item in source_entries
            for key in ("mr_series", "seg_series")
        ],
        source_instance_object_ids=[
            str(item[key]["object_id"])
            for item in source_entries
            for key in ("mr_instance", "seg_instance", "claim_document")
            if item.get(key) is not None
        ],
        withheld_reason=withheld_reason,
    )
    relationships = [
        build_relationship(
            relationship_type="contains_series",
            source_object_id=study["object_id"],
            source_kind="study",
            target_object_id=series["object_id"],
            target_kind="series",
        )
    ]
    return {"group": group, "relationships": relationships}


def _build_verification_summary(
    *,
    report: Mapping[str, Any],
    bundle_reference: str | None,
    evidence_hash: str | None,
    include_evidence_hash: bool,
) -> dict[str, Any]:
    contracts = report["contracts"]
    summary: dict[str, Any] = {
        "guardian_mode": "verification_proxy",
        "decision": "released" if report["release_eligible"] else "blocked",
        "decision_class": report["decision_summary"]["decision_class"],
        "canonical_decision": deepcopy(report["decision"]),
        "primary_reason": _primary_reason_summary(report),
        "pipeline_phase_statuses": _pipeline_phase_statuses(report),
        "proof_status": report["decision_summary"]["proof_status"],
        "profile_policy_status": report["decision_summary"]["profile_policy_status"],
        "verification_profile_descriptor_hash": hash_canonical_json(
            contracts["verification_profile_descriptor"]
        ),
        "bundle_reference": bundle_reference,
    }
    if include_evidence_hash:
        summary["evidence_hash"] = evidence_hash
    if report.get("case_kind") == MULTI_SERIES_CASE_KIND:
        study_temporal = report["analysis"]["study_temporal_aggregation"]
        summary.update(
            {
                "case_kind": MULTI_SERIES_CASE_KIND,
                "profile_descriptor_set_hash": hash_canonical_json(report["profile"]),
                "released_series_ids": list(report["analysis"]["released_series_ids"]),
                "withheld_series_ids": list(report["analysis"]["withheld_series_ids"]),
                "study_temporal_disposition": study_temporal["final_temporal_disposition"],
                "study_temporal_top_flags": list(study_temporal["top_flags"]),
                "study_temporal_top_reason_codes": list(study_temporal["top_reason_codes"]),
            }
        )
        return summary
    summary.update(
        {
            "profile_name": report["manifest"]["profile_name"],
            "profile_digest": report["manifest"]["profile_digest"],
            "verified_tumor_volume_mm3": report["analysis"].get("computed_tumor_volume_mm3"),
        }
    )
    summary.update(_single_series_temporal_summary(report))
    return summary


def _proof_contracts_for_report(report: Mapping[str, Any]) -> dict[str, Any]:
    contracts = report["contracts"]
    if report.get("case_kind") == MULTI_SERIES_CASE_KIND:
        payload: dict[str, Any] = {
            "verification_profile_descriptor": contracts["verification_profile_descriptor"],
            "cross_series_verification_policy": contracts["cross_series_verification_policy"],
            "cross_series_verification_trace": contracts["cross_series_verification_trace"],
            "study_temporal_aggregation_trace": contracts["study_temporal_aggregation_trace"],
            "series_contracts": [],
        }
        for series_report in report.get("series_reports", []):
            series_contracts = series_report["contracts"]
            payload["series_contracts"].append(
                {
                    "series_id": series_report["series_id"],
                    "verification_profile_descriptor": series_contracts["verification_profile_descriptor"],
                    "mask_derivation_policy": series_contracts["mask_derivation_policy"],
                    "verification_tolerance_policy": series_contracts["verification_tolerance_policy"],
                }
            )
        return payload
    payload = {
        "verification_profile_descriptor": contracts["verification_profile_descriptor"],
    }
    if "ai_claim_envelope" in contracts:
        payload["ai_claim_envelope"] = contracts["ai_claim_envelope"]
    for key in (
        "measurement_integrity_proof",
        "reconstruction_parity_proof",
        "segmentation_policy_proof",
        "volume_computation_proof",
        "proof_core_evidence_trace",
        "mask_derivation_policy",
        "verification_tolerance_policy",
        "longitudinal_selection_trace",
        "temporal_trajectory_trace",
        "longitudinal_registration_trace",
        "temporal_consistency_trace",
    ):
        if key in contracts:
            payload[key] = contracts[key]
    prior_verified_contexts = contracts.get("prior_verified_contexts")
    if prior_verified_contexts is not None:
        payload["prior_verified_contexts"] = prior_verified_contexts
    prior_verified_context = contracts.get("prior_verified_context")
    if prior_verified_context is not None:
        payload["prior_verified_context"] = prior_verified_context
    return payload


def _graph_summary(
    *,
    source_series: Sequence[Mapping[str, Any]],
    released_series: Sequence[Mapping[str, Any]],
    withheld_outputs: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lineage_groups = list(released_series) or list(withheld_outputs) or list(source_series)
    frame_counts = {key: 0 for key in FRAME_OF_REFERENCE_STATUS_VOCABULARY}
    completeness_values: list[str] = []
    for item in lineage_groups:
        lineage = item["lineage"]
        frame_counts[str(lineage["frame_of_reference_status"])] += 1
        completeness_values.append(str(lineage["lineage_completeness"]))
    if not completeness_values:
        lineage_status = "not_applicable"
    elif "partial" in completeness_values:
        lineage_status = "partial"
    elif any(value == "complete" for value in completeness_values):
        lineage_status = "complete"
    else:
        lineage_status = "not_applicable"
    withheld_series_keys = sorted(
        {
            str(item["series"]["extensions"]["guardian_v1"].get("source_series_key"))
            for item in withheld_outputs
            if item["series"]["extensions"]["guardian_v1"].get("source_series_key") is not None
        }
    )
    return {
        "source_series_count": len(source_series),
        "source_instance_count": sum(len(item["instances"]) for item in source_series),
        "acquisition_provenance_count": sum(
            len(item["acquisition_provenance"]) for item in source_series
        ),
        "released_series_count": len(released_series),
        "released_instance_count": sum(len(item["instances"]) for item in released_series),
        "withheld_series_count": len(withheld_series_keys),
        "withheld_output_count": len(withheld_outputs),
        "sidecar_count": len(sidecars),
        "frame_of_reference_summary": {
            key: value for key, value in frame_counts.items() if value > 0
        },
        "lineage_completeness_status": lineage_status,
    }


def _build_ingress_trace_stage(
    *,
    state: Mapping[str, Any],
    dicom_input: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    stage = {
        "stage": "dicom_ingress",
        "label": "DICOM Input",
        "transport": dicom_input["transport"],
        "ae_title": dicom_input["endpoint"]["source_ae_title"],
        "study_instance_uid": dicom_input["study"]["study_instance_uid"],
        "source_study_key": dicom_input["study"]["extensions"]["guardian_v1"]["source_study_key"],
        "source_series_count": summary["source_series_count"],
        "source_instance_count": summary["source_instance_count"],
        "acquisition_provenance_count": summary["acquisition_provenance_count"],
        "frame_of_reference_summary": summary["frame_of_reference_summary"],
        "lineage_completeness_status": summary["lineage_completeness_status"],
    }
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        stage["case_kind"] = MULTI_SERIES_CASE_KIND
        stage["series_count"] = len(state.get("series", []))
    else:
        stage["series_uid"] = _string_or_fallback(state.get("series_uid"), "unknown-series")
    return stage


def _build_guardian_trace_stage(
    *,
    report: Mapping[str, Any],
    submission_status: str,
    egress_summary: Mapping[str, Any],
) -> dict[str, Any]:
    stage = {
        "stage": "guardian_layer",
        "label": "MRI Guardian Layer",
        "mode": "verification_proxy",
        "decision": submission_status,
        "decision_class": report["decision_summary"]["decision_class"],
        "canonical_decision": deepcopy(report["decision"]),
        "primary_reason": _primary_reason_summary(report),
        "pipeline_phase_statuses": _pipeline_phase_statuses(report),
        "released_series_count": egress_summary["released_series_count"],
        "withheld_series_count": egress_summary["withheld_series_count"],
        "released_instance_count": egress_summary["released_instance_count"],
        "withheld_output_count": egress_summary["withheld_output_count"],
        "sidecar_count": egress_summary["sidecar_count"],
        "frame_of_reference_summary": egress_summary["frame_of_reference_summary"],
        "lineage_completeness_status": egress_summary["lineage_completeness_status"],
    }
    if report.get("case_kind") == MULTI_SERIES_CASE_KIND:
        stage["case_kind"] = MULTI_SERIES_CASE_KIND
        stage["study_temporal_disposition"] = report["analysis"]["study_temporal_aggregation"][
            "final_temporal_disposition"
        ]
        stage["study_temporal_top_flags"] = list(
            report["analysis"]["study_temporal_aggregation"]["top_flags"]
        )
        stage["study_temporal_top_reason_codes"] = list(
            report["analysis"]["study_temporal_aggregation"]["top_reason_codes"]
        )
        return stage
    stage.update(_single_series_temporal_summary(report))
    return stage


def _build_egress_trace_stage(
    *,
    pacs_output: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "pacs_egress",
        "label": (
            "PACS Output"
            if pacs_output["submission_status"] == "released"
            else "PACS Output Blocked"
        ),
        "transport": pacs_output["transport"],
        "ae_title": pacs_output["endpoint"]["destination_ae_title"],
        "artifact_kind": pacs_output["kind"],
        "released_series_count": summary["released_series_count"],
        "withheld_series_count": summary["withheld_series_count"],
        "released_instance_count": summary["released_instance_count"],
        "withheld_output_count": summary["withheld_output_count"],
        "sidecar_count": summary["sidecar_count"],
        "frame_of_reference_summary": summary["frame_of_reference_summary"],
        "lineage_completeness_status": summary["lineage_completeness_status"],
    }


def _validate_mock_dicom_input_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "kind",
        "transport",
        "endpoint",
        "study",
        "source_series",
        "relationships",
        "sidecars",
        "integration_narrative",
        "debug",
        "ai_runtime_context",
    }
    normalised = dict(payload)
    unexpected = sorted(key for key in normalised if key not in expected_fields)
    if unexpected:
        raise ValueError(
            "mock_dicom_input contains unsupported fields: " + ", ".join(unexpected) + "."
        )
    if normalised.get("schema_version") != DICOM_MODEL_SCHEMA_VERSION:
        raise ValueError("mock_dicom_input.schema_version is not supported.")
    if normalised.get("kind") != "mock_dicom_input":
        raise ValueError("Expected a mocked DICOM input envelope.")
    normalised["transport"] = _require_runtime_string(
        normalised.get("transport"),
        path="mock_dicom_input.transport",
    )
    normalised["endpoint"] = _validate_endpoint(normalised.get("endpoint"), path="mock_dicom_input.endpoint")
    normalised["study"] = _validate_standard_like_mapping(
        normalised.get("study"),
        object_kind="study",
        path="mock_dicom_input.study",
    )
    normalised["source_series"] = _validate_series_groups(
        normalised.get("source_series"),
        path="mock_dicom_input.source_series",
        allowed_statuses={"source"},
    )
    normalised["sidecars"] = _validate_sidecars(
        normalised.get("sidecars"),
        path="mock_dicom_input.sidecars",
    )
    normalised["relationships"] = _validate_payload_relationships(
        study=normalised["study"],
        source_series=normalised["source_series"],
        released_series=[],
        withheld_outputs=[],
        sidecars=normalised["sidecars"],
        relationships=normalised.get("relationships"),
        path="mock_dicom_input.relationships",
    )
    normalised["integration_narrative"] = _validate_mapping(
        normalised.get("integration_narrative"),
        path="mock_dicom_input.integration_narrative",
    )
    debug = _validate_mapping(normalised.get("debug"), path="mock_dicom_input.debug")
    unexpected_debug = sorted(key for key in debug if key != "guardian_case_snapshot")
    if unexpected_debug:
        raise ValueError(
            "mock_dicom_input.debug contains unsupported fields: "
            + ", ".join(unexpected_debug)
            + "."
        )
    if not isinstance(debug.get("guardian_case_snapshot"), Mapping):
        raise ValueError("Mocked DICOM input is missing debug.guardian_case_snapshot.")
    normalised["debug"] = {
        "guardian_case_snapshot": deepcopy(dict(debug["guardian_case_snapshot"]))
    }
    if "ai_runtime_context" in normalised:
        normalised["ai_runtime_context"] = _canonical_ai_runtime_context(
            normalised.get("ai_runtime_context")
        )
    return normalised


def _validate_mock_pacs_payload(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "kind",
        "transport",
        "endpoint",
        "study",
        "source_series",
        "released_series",
        "withheld_outputs",
        "relationships",
        "sidecars",
        "submission_status",
        "verification_summary",
        "integration_narrative",
        "proof_contracts",
        "ai_runtime_context",
        "blocked_reason",
        "failure_reasons",
    }
    normalised = dict(payload)
    unexpected = sorted(key for key in normalised if key not in expected_fields)
    if unexpected:
        raise ValueError(
            f"{expected_kind} contains unsupported fields: " + ", ".join(unexpected) + "."
        )
    if normalised.get("schema_version") != DICOM_MODEL_SCHEMA_VERSION:
        raise ValueError(f"{expected_kind}.schema_version is not supported.")
    if normalised.get("kind") != expected_kind:
        raise ValueError(f"Expected {expected_kind!r}.")
    submission_status = "released" if expected_kind == "mock_pacs_submission" else "blocked"
    if normalised.get("submission_status") != submission_status:
        raise ValueError(f"{expected_kind}.submission_status must be {submission_status!r}.")
    normalised["transport"] = _require_runtime_string(
        normalised.get("transport"),
        path=f"{expected_kind}.transport",
    )
    normalised["endpoint"] = _validate_endpoint(normalised.get("endpoint"), path=f"{expected_kind}.endpoint")
    normalised["study"] = _validate_standard_like_mapping(
        normalised.get("study"),
        object_kind="study",
        path=f"{expected_kind}.study",
    )
    normalised["source_series"] = _validate_series_groups(
        normalised.get("source_series"),
        path=f"{expected_kind}.source_series",
        allowed_statuses={"source"},
    )
    normalised["released_series"] = _validate_series_groups(
        normalised.get("released_series"),
        path=f"{expected_kind}.released_series",
        allowed_statuses={"released"},
    )
    normalised["withheld_outputs"] = _validate_series_groups(
        normalised.get("withheld_outputs"),
        path=f"{expected_kind}.withheld_outputs",
        allowed_statuses={"withheld"},
    )
    normalised["sidecars"] = _validate_sidecars(
        normalised.get("sidecars"),
        path=f"{expected_kind}.sidecars",
    )
    normalised["relationships"] = _validate_payload_relationships(
        study=normalised["study"],
        source_series=normalised["source_series"],
        released_series=normalised["released_series"],
        withheld_outputs=normalised["withheld_outputs"],
        sidecars=normalised["sidecars"],
        relationships=normalised.get("relationships"),
        path=f"{expected_kind}.relationships",
    )
    normalised["verification_summary"] = _validate_mapping(
        normalised.get("verification_summary"),
        path=f"{expected_kind}.verification_summary",
    )
    normalised["integration_narrative"] = _validate_mapping(
        normalised.get("integration_narrative"),
        path=f"{expected_kind}.integration_narrative",
    )
    normalised["proof_contracts"] = _validate_mapping(
        normalised.get("proof_contracts"),
        path=f"{expected_kind}.proof_contracts",
    )
    if expected_kind == "mock_pacs_block_notice":
        normalised["blocked_reason"] = _require_runtime_string(
            normalised.get("blocked_reason"),
            path="mock_pacs_block_notice.blocked_reason",
        )
        failure_reasons = normalised.get("failure_reasons")
        if not isinstance(failure_reasons, list):
            raise ValueError("mock_pacs_block_notice.failure_reasons must be a list.")
        normalised["failure_reasons"] = deepcopy(failure_reasons)
    if "ai_runtime_context" in normalised:
        normalised["ai_runtime_context"] = _canonical_ai_runtime_context(
            normalised.get("ai_runtime_context")
        )
    return normalised


def _validate_endpoint(value: Any, *, path: str) -> dict[str, str]:
    endpoint = _validate_mapping(value, path=path)
    expected = {"source_ae_title", "destination_ae_title"}
    unexpected = sorted(key for key in endpoint if key not in expected)
    missing = sorted(key for key in expected if key not in endpoint)
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}.")
    if unexpected:
        raise ValueError(f"{path} contains unsupported fields: {', '.join(unexpected)}.")
    return {
        "source_ae_title": _require_runtime_string(
            endpoint["source_ae_title"],
            path=f"{path}.source_ae_title",
        ),
        "destination_ae_title": _require_runtime_string(
            endpoint["destination_ae_title"],
            path=f"{path}.destination_ae_title",
        ),
    }


def _validate_series_groups(
    value: Any,
    *,
    path: str,
    allowed_statuses: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    normalised: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        group = _validate_mapping(item, path=item_path)
        expected_fields = {
            "status",
            "series",
            "instances",
            "acquisition_provenance",
            "lineage",
            "withheld_reason",
        }
        unexpected = sorted(key for key in group if key not in expected_fields)
        if unexpected:
            raise ValueError(
                f"{item_path} contains unsupported fields: {', '.join(unexpected)}."
            )
        status = _require_runtime_string(group.get("status"), path=f"{item_path}.status")
        if status not in allowed_statuses:
            raise ValueError(
                f"{item_path}.status must be one of: {', '.join(sorted(allowed_statuses))}."
            )
        if status == "withheld":
            withheld_reason = _require_runtime_string(
                group.get("withheld_reason"),
                path=f"{item_path}.withheld_reason",
            )
        elif "withheld_reason" in group:
            raise ValueError(f"{item_path}.withheld_reason is only valid for withheld outputs.")
        else:
            withheld_reason = None
        series = _validate_standard_like_mapping(
            group.get("series"),
            object_kind="series",
            path=f"{item_path}.series",
        )
        instances_value = group.get("instances")
        if not isinstance(instances_value, list):
            raise ValueError(f"{item_path}.instances must be a list.")
        instances: list[dict[str, Any]] = []
        for instance_index, instance_item in enumerate(instances_value):
            instance_path = f"{item_path}.instances[{instance_index}]"
            instance_payload = _validate_mapping(instance_item, path=instance_path)
            object_kind = instance_payload.get("object_kind")
            if object_kind == "instance":
                instances.append(
                    _validate_standard_like_mapping(
                        instance_payload,
                        object_kind="instance",
                        path=instance_path,
                    )
                )
            elif object_kind == "document":
                instances.append(
                    _validate_standard_like_mapping(
                        instance_payload,
                        object_kind="document",
                        path=instance_path,
                    )
                )
            else:
                raise ValueError(f"{instance_path}.object_kind must be 'instance' or 'document'.")
        acquisition_value = group.get("acquisition_provenance")
        if not isinstance(acquisition_value, list):
            raise ValueError(f"{item_path}.acquisition_provenance must be a list.")
        acquisition_objects = [
            _validate_acquisition_provenance_mapping(
                acquisition_item,
                path=f"{item_path}.acquisition_provenance[{acq_index}]",
            )
            for acq_index, acquisition_item in enumerate(acquisition_value)
        ]
        lineage = _validate_lineage(
            group.get("lineage"),
            path=f"{item_path}.lineage",
        )
        normalised_group = {
            "status": status,
            "series": series,
            "instances": instances,
            "acquisition_provenance": acquisition_objects,
            "lineage": lineage,
        }
        if withheld_reason is not None:
            normalised_group["withheld_reason"] = withheld_reason
        normalised.append(normalised_group)
    return normalised


def _validate_lineage(value: Any, *, path: str) -> dict[str, Any]:
    lineage = _validate_mapping(value, path=path)
    expected = {
        "source_series_object_ids",
        "source_instance_object_ids",
        "lineage_completeness",
        "frame_of_reference_status",
    }
    unexpected = sorted(key for key in lineage if key not in expected)
    missing = sorted(key for key in expected if key not in lineage)
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}.")
    if unexpected:
        raise ValueError(f"{path} contains unsupported fields: {', '.join(unexpected)}.")
    source_series_object_ids = _validate_string_list(
        lineage["source_series_object_ids"],
        path=f"{path}.source_series_object_ids",
    )
    source_instance_object_ids = _validate_string_list(
        lineage["source_instance_object_ids"],
        path=f"{path}.source_instance_object_ids",
    )
    lineage_completeness = _require_runtime_string(
        lineage["lineage_completeness"],
        path=f"{path}.lineage_completeness",
    )
    if lineage_completeness not in LINEAGE_COMPLETENESS_VOCABULARY:
        raise ValueError(
            f"{path}.lineage_completeness must be one of: {', '.join(LINEAGE_COMPLETENESS_VOCABULARY)}."
        )
    frame_of_reference_status = _require_runtime_string(
        lineage["frame_of_reference_status"],
        path=f"{path}.frame_of_reference_status",
    )
    if frame_of_reference_status not in FRAME_OF_REFERENCE_STATUS_VOCABULARY:
        raise ValueError(
            f"{path}.frame_of_reference_status must be one of: {', '.join(FRAME_OF_REFERENCE_STATUS_VOCABULARY)}."
        )
    return {
        "source_series_object_ids": source_series_object_ids,
        "source_instance_object_ids": source_instance_object_ids,
        "lineage_completeness": lineage_completeness,
        "frame_of_reference_status": frame_of_reference_status,
    }


def _validate_sidecars(value: Any, *, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    return [
        _validate_sidecar_mapping(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_payload_relationships(
    *,
    study: Mapping[str, Any],
    source_series: Sequence[Mapping[str, Any]],
    released_series: Sequence[Mapping[str, Any]],
    withheld_outputs: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
    relationships: Any,
    path: str,
) -> list[dict[str, Any]]:
    if not isinstance(relationships, list):
        raise ValueError(f"{path} must be a list.")
    objects = [
        study,
        *_flatten_group_objects(source_series),
        *_flatten_group_objects(released_series),
        *_flatten_group_objects(withheld_outputs),
        *sidecars,
    ]
    registry = build_object_registry(objects)
    normalised = validate_relationships(relationships, known_objects=registry)
    _validate_lineage_references(source_series, registry, path=f"{path}.source_series")
    _validate_lineage_references(released_series, registry, path=f"{path}.released_series")
    _validate_lineage_references(withheld_outputs, registry, path=f"{path}.withheld_outputs")
    _validate_sidecar_manifest_links(
        sidecars=sidecars,
        groups=[*source_series, *released_series, *withheld_outputs],
        path=path,
    )
    return normalised


def _validate_lineage_references(
    groups: Sequence[Mapping[str, Any]],
    registry: Mapping[str, str],
    *,
    path: str,
) -> None:
    for index, item in enumerate(groups):
        lineage = item["lineage"]
        for series_index, object_id in enumerate(lineage["source_series_object_ids"]):
            if registry.get(object_id) != "series":
                raise ValueError(
                    f"{path}[{index}].lineage.source_series_object_ids[{series_index}] "
                    "must reference a known series object."
                )
        for instance_index, object_id in enumerate(lineage["source_instance_object_ids"]):
            object_kind = registry.get(object_id)
            if object_kind not in {"instance", "document"}:
                raise ValueError(
                    f"{path}[{index}].lineage.source_instance_object_ids[{instance_index}] "
                    "must reference a known instance or document object."
                )


def _validate_sidecar_manifest_links(
    *,
    sidecars: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    path: str,
) -> None:
    document_uids = {
        str(instance["sop_instance_uid"])
        for group in groups
        for instance in group["instances"]
        if instance["object_kind"] == "document"
    }
    for index, sidecar in enumerate(sidecars):
        if sidecar["referenced_manifest_instance_uid"] not in document_uids:
            raise ValueError(
                f"{path}.sidecars[{index}] references an unknown manifest document UID."
            )


def _flatten_group_objects(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in groups:
        flattened.append(dict(item["series"]))
        flattened.extend(dict(instance) for instance in item["instances"])
        flattened.extend(dict(entry) for entry in item["acquisition_provenance"])
    return flattened


def _validate_standard_like_mapping(
    value: Any,
    *,
    object_kind: str,
    path: str,
) -> dict[str, Any]:
    payload = _validate_mapping(value, path=path)
    return validate_standard_like_object(payload, object_kind=object_kind)


def _validate_sidecar_mapping(value: Any, *, path: str) -> dict[str, Any]:
    payload = _validate_mapping(value, path=path)
    return validate_sidecar_object(payload)


def _validate_acquisition_provenance_mapping(value: Any, *, path: str) -> dict[str, Any]:
    payload = _validate_mapping(value, path=path)
    return validate_acquisition_provenance_object(payload)


def _validate_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    return dict(value)


def _validate_string_list(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    return [
        _require_runtime_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_runtime_string(value: Any, *, path: str) -> str:
    if type(value) is not str or value == "":
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _canonical_ai_runtime_context(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("ai_runtime_context must be an object.")
    payload = dict(value)
    missing_fields = [field for field in _AI_RUNTIME_CONTEXT_FIELDS if field not in payload]
    if missing_fields:
        raise ValueError(
            "ai_runtime_context is missing required fields: "
            + ", ".join(missing_fields)
            + "."
        )
    unexpected_fields = sorted(
        field for field in payload if field not in set(_AI_RUNTIME_CONTEXT_FIELDS)
    )
    if unexpected_fields:
        raise ValueError(
            "ai_runtime_context contains unsupported fields: "
            + ", ".join(unexpected_fields)
            + "."
        )
    return {
        "inference_runtime_version": _require_runtime_string(
            payload["inference_runtime_version"],
            path="ai_runtime_context.inference_runtime_version",
        ),
        "inference_device_class": _require_runtime_string(
            payload["inference_device_class"],
            path="ai_runtime_context.inference_device_class",
        ),
    }


def _input_field_hash(state: Mapping[str, Any], field: str) -> str | None:
    value = state.get(field)
    if value is None:
        return None
    return hash_canonical_json(value)


def _raw_kspace_hash_from_state(state: Mapping[str, Any]) -> str | None:
    if "raw_real" not in state or "raw_imag" not in state:
        return None
    return hash_canonical_json({"raw_real": state["raw_real"], "raw_imag": state["raw_imag"]})


def _fallback_claim_outputs_hash(
    state: Mapping[str, Any],
    *,
    raw_hash: str | None,
    ai_reconstruction_hash: str | None,
    ai_mask_hash: str | None,
) -> str | None:
    payload: dict[str, Any] = {}
    if raw_hash is not None:
        payload["raw_kspace_hash"] = raw_hash
    if ai_reconstruction_hash is not None:
        payload["ai_reconstruction_hash"] = ai_reconstruction_hash
    if ai_mask_hash is not None:
        payload["ai_mask_hash"] = ai_mask_hash
    claimed_volume = state.get("tumor_volume_mm3")
    if isinstance(claimed_volume, (int, float)):
        payload["claimed_tumor_volume_mm3"] = float(claimed_volume)
    if not payload:
        return None
    return hash_canonical_json(payload)


def _source_hashes(state: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, str | None]:
    analysis = report.get("analysis", {})
    contracts = report.get("contracts", {})
    claim_envelope = contracts.get("ai_claim_envelope")
    raw_hash = analysis.get("raw_hash") or _raw_kspace_hash_from_state(state)
    ai_reconstruction_hash = (
        claim_envelope["claimed_outputs"]["ai_reconstruction_hash"]
        if isinstance(claim_envelope, Mapping)
        else analysis.get("ai_reconstruction_hash") or _input_field_hash(state, "ai_reconstruction")
    )
    ai_mask_hash = (
        claim_envelope["claimed_outputs"]["ai_mask_hash"]
        if isinstance(claim_envelope, Mapping)
        else analysis.get("ai_mask_hash") or _input_field_hash(state, "tumor_mask")
    )
    claim_outputs_hash = (
        hash_canonical_json(claim_envelope["claimed_outputs"])
        if isinstance(claim_envelope, Mapping)
        else _fallback_claim_outputs_hash(
            state,
            raw_hash=raw_hash,
            ai_reconstruction_hash=ai_reconstruction_hash,
            ai_mask_hash=ai_mask_hash,
        )
    )
    return {
        "raw_input_hash": raw_hash,
        "ai_reconstruction_hash": ai_reconstruction_hash,
        "ai_mask_hash": ai_mask_hash,
        "claim_outputs_hash": claim_outputs_hash,
    }


def _series_group(
    *,
    status: str,
    series: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    acquisition_provenance: Sequence[Mapping[str, Any]],
    frame_of_reference_status: str,
    lineage_completeness: str,
    source_series_object_ids: Sequence[str],
    source_instance_object_ids: Sequence[str],
    withheld_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "series": dict(series),
        "instances": [dict(item) for item in instances],
        "acquisition_provenance": [dict(item) for item in acquisition_provenance],
        "lineage": {
            "source_series_object_ids": list(source_series_object_ids),
            "source_instance_object_ids": list(source_instance_object_ids),
            "lineage_completeness": lineage_completeness,
            "frame_of_reference_status": frame_of_reference_status,
        },
    }
    if withheld_reason is not None:
        payload["withheld_reason"] = withheld_reason
    return payload


def _series_extensions(
    *,
    source_series_key: str,
    source_series_uid: str,
    resolved_role: str | None,
    gating_status: str | None,
    output_status: str,
    downstream_release_allowed: bool | None = None,
    withheld_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_series_key": source_series_key,
        "source_series_uid": source_series_uid,
        "output_status": output_status,
    }
    if resolved_role is not None:
        payload["resolved_role"] = resolved_role
    if gating_status is not None:
        payload["gating_status"] = gating_status
    if downstream_release_allowed is not None:
        payload["downstream_release_allowed"] = downstream_release_allowed
    if withheld_reason is not None:
        payload["withheld_reason"] = withheld_reason
    return payload


def _claim_measurements(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    claimed_volume = state.get("tumor_volume_mm3")
    if not isinstance(claimed_volume, (int, float)):
        return []
    return [
        {
            "measurement_name": "claimed_tumor_volume_mm3",
            "value": float(claimed_volume),
            "units": "mm3",
        }
    ]


def _manifest_measurements(
    *,
    state: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return [
            {
                "measurement_name": "released_series_count",
                "value": float(len(state["workstation_payload"]["series_payloads"])),
                "units": "count",
            },
            {
                "measurement_name": "withheld_series_count",
                "value": float(len(state["workstation_payload"].get("withheld_series", []))),
                "units": "count",
            },
        ]
    measurements: list[dict[str, Any]] = []
    claimed_volume = report["contracts"].get("ai_claim_envelope", {}).get("claimed_outputs", {}).get(
        "claimed_tumor_volume_mm3"
    )
    if isinstance(claimed_volume, (int, float)):
        measurements.append(
            {
                "measurement_name": "claimed_tumor_volume_mm3",
                "value": float(claimed_volume),
                "units": "mm3",
            }
        )
    verified_volume = state["workstation_payload"].get("tumor_volume_mm3")
    if isinstance(verified_volume, (int, float)):
        measurements.append(
            {
                "measurement_name": "verified_tumor_volume_mm3",
                "value": float(verified_volume),
                "units": "mm3",
            }
        )
    return measurements


def _frame_of_reference_lineage(state: Mapping[str, Any]) -> tuple[str | None, str]:
    spatial_geometry = state.get("spatial_geometry")
    if not isinstance(spatial_geometry, Mapping):
        return None, "no_upstream_lineage"
    frame_uid = spatial_geometry.get("frame_of_reference_uid")
    if type(frame_uid) is not str or frame_uid == "":
        return None, "no_upstream_lineage"
    if _is_valid_dicom_uid(frame_uid):
        return frame_uid, "explicit_uid"
    return None, "upstream_non_dicom_uid"


def _is_valid_dicom_uid(value: str) -> bool:
    if not _UID_COMPONENT_RE.fullmatch(value):
        return False
    if len(value) > 64:
        return False
    if UID is not None and not UID(value).is_valid:
        return False
    return True


def _case_kind_from_state(state: Mapping[str, Any]) -> str:
    return MULTI_SERIES_CASE_KIND if state.get("case_kind") == MULTI_SERIES_CASE_KIND else "single_series_v1"


def _pipeline_phase_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: phase["status"]
        for key, phase in report["verification_pipeline"]["phases"].items()
    }


def _primary_reason_summary(report: Mapping[str, Any]) -> str:
    reasons = report["failure_reasons"]
    if reasons:
        return str(reasons[0]["summary"])
    decision = report["decision"]
    if decision["reason_codes"]:
        return str(decision["reason_codes"][0])
    return "final_release_authorized" if report["release_eligible"] else "decision_blocked"


def _single_series_temporal_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    temporal_trace = report["contracts"].get("temporal_consistency_trace")
    if not isinstance(temporal_trace, Mapping):
        return {}
    selection_trace = report["contracts"].get("longitudinal_selection_trace")
    trajectory_trace = report["contracts"].get("temporal_trajectory_trace")
    registration_trace = report["contracts"].get("longitudinal_registration_trace")
    temporal_reason_codes = sorted(
        {
            str(item.get("code"))
            for item in report["manifest"]["clinical_plausibility"]["heuristics"]
            if isinstance(item, Mapping) and str(item.get("code", "")).startswith("temporal_")
        }
    )
    return {
        "temporal_comparison_status": temporal_trace["comparison_status"],
        "temporal_comparison_scope": temporal_trace["comparison_scope"],
        "temporal_spatial_comparison_mode": temporal_trace["spatial_comparison_mode"],
        "temporal_primary_selection_mode": (
            None
            if not isinstance(selection_trace, Mapping)
            else selection_trace["primary_selection_mode"]
        ),
        "temporal_selected_prior_count": (
            None
            if not isinstance(selection_trace, Mapping)
            else selection_trace["selected_prior_count"]
        ),
        "temporal_trajectory_class": (
            None
            if not isinstance(trajectory_trace, Mapping)
            else trajectory_trace["trajectory_class"]
        ),
        "temporal_registration_status": (
            None
            if not isinstance(registration_trace, Mapping)
            else registration_trace["status"]
        ),
        "temporal_top_flags": list(temporal_trace["flags"]),
        "temporal_trajectory_flags": (
            []
            if not isinstance(trajectory_trace, Mapping)
            else list(trajectory_trace["flags"])
        ),
        "temporal_top_reason_codes": temporal_reason_codes,
    }


def _string_or_fallback(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
