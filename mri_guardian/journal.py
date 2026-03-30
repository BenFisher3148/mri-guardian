"""Deterministic operational decision journal derived from verification reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict

from icsd.core.serialise import hash_canonical_json

SINGLE_SERIES_CASE_KIND = "single_series_v1"
DECISION_JOURNAL_SCHEMA_VERSION = "0.1"
DECISION_JOURNAL_TYPE = "DecisionJournal"

EVENT_ID_INPUT_RECEIVED = "input_received"
EVENT_ID_STRUCTURAL_VALIDATION = "structural_validation"
EVENT_ID_RECONSTRUCTION = "reconstruction"
EVENT_ID_PARITY_CHECKS = "parity_checks"
EVENT_ID_SEGMENTATION_VERIFICATION = "segmentation_verification"
EVENT_ID_VOLUME_PROOF = "volume_proof"
EVENT_ID_ADVISORY_HEURISTICS = "advisory_heuristics"
EVENT_ID_MULTI_SERIES_AGGREGATION = "multi_series_aggregation"
EVENT_ID_FINAL_DECISION = "final_decision"
EVENT_ID_TRANSPARENCY_ANCHOR = "transparency_anchor"
_CANONICAL_EVENT_ORDER = (
    EVENT_ID_INPUT_RECEIVED,
    EVENT_ID_STRUCTURAL_VALIDATION,
    EVENT_ID_RECONSTRUCTION,
    EVENT_ID_PARITY_CHECKS,
    EVENT_ID_SEGMENTATION_VERIFICATION,
    EVENT_ID_VOLUME_PROOF,
    EVENT_ID_ADVISORY_HEURISTICS,
    EVENT_ID_MULTI_SERIES_AGGREGATION,
    EVENT_ID_FINAL_DECISION,
    EVENT_ID_TRANSPARENCY_ANCHOR,
)
_ALLOWED_EVENT_IDS = frozenset(_CANONICAL_EVENT_ORDER)

STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ADVISORY = "advisory"
STATUS_RECORDED = "recorded"
STATUS_ABSENT = "absent"
STATUS_NOT_APPLICABLE = "not_applicable"
_ALLOWED_ENTRY_STATUSES = frozenset(
    {
        STATUS_NOT_EVALUATED,
        STATUS_PASSED,
        STATUS_FAILED,
        STATUS_ADVISORY,
        STATUS_RECORDED,
        STATUS_ABSENT,
        STATUS_NOT_APPLICABLE,
    }
)

RESIDUAL_RISK_BOUNDED = "bounded"
RESIDUAL_RISK_WORKFLOW_EXTERNAL = "workflow_external"
RESIDUAL_RISK_OPERATOR_RELEVANT = "operator_relevant"
_ALLOWED_RESIDUAL_RISK_CLASSIFICATIONS = frozenset(
    {
        RESIDUAL_RISK_BOUNDED,
        RESIDUAL_RISK_WORKFLOW_EXTERNAL,
        RESIDUAL_RISK_OPERATOR_RELEVANT,
    }
)

PIPELINE_PHASE_INTAKE_VALIDATION = "intake_validation"
PIPELINE_PHASE_STRUCTURAL_VALIDITY = "structural_validity"
PIPELINE_PHASE_PROOF_CORE = "proof_core"
PIPELINE_PHASE_CLINICAL_PLAUSIBILITY = "clinical_plausibility"
PIPELINE_PHASE_POLICY_EVALUATION = "policy_evaluation"
PIPELINE_PHASE_MULTI_SERIES_AGGREGATION = "multi_series_aggregation"
PIPELINE_PHASE_RELEASE_CONTROL = "release_control"

PROOF_STAGE_MEASUREMENT_INTEGRITY = "MeasurementIntegrityProof"
PROOF_STAGE_RECONSTRUCTION_PARITY = "ReconstructionParityProof"
PROOF_STAGE_SEGMENTATION_POLICY = "SegmentationPolicyProof"
PROOF_STAGE_VOLUME_COMPUTATION = "VolumeComputationProof"

_CHILD_OUTCOME_FIELDS = (
    "series_id",
    "decision_class",
    "final_disposition",
    "gating_status",
)


class TransparencyReceiptContext(TypedDict):
    """Minimal transparency context admissible for decision-journal derivation."""

    reference: str
    entry_hash: str
    previous_entry_hash: str | None
    checkpoint: str | None
    entry_index: int
    log_length: int


class _JournalProjection(TypedDict):
    case_id: str
    case_kind: str
    profile_identity: dict[str, Any]
    transparency_artifact_allowed: bool
    verification_report_hash: str
    decision: dict[str, Any]
    decision_summary: dict[str, Any]
    decision_reason_codes: list[str]
    decision_reasons: list[dict[str, Any]]
    failure_reasons: list[dict[str, Any]]
    verification_pipeline: dict[str, Any]
    analysis: dict[str, Any]
    contracts: dict[str, Any]
    manifest: dict[str, Any]
    is_multi_series: bool
    child_outcomes: NotRequired[list[dict[str, str]]]


def build_transparency_receipt_context(
    *,
    reference: str,
    entry_hash: str,
    previous_entry_hash: str | None,
    checkpoint: str | None,
    entry_index: int,
    log_length: int,
) -> TransparencyReceiptContext:
    """Return the minimal admissible transparency context for journal derivation."""
    payload: TransparencyReceiptContext = {
        "reference": str(reference),
        "entry_hash": str(entry_hash),
        "previous_entry_hash": None if previous_entry_hash is None else str(previous_entry_hash),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "entry_index": int(entry_index),
        "log_length": int(log_length),
    }
    _validate_transparency_receipt_context(payload)
    return payload


def build_decision_journal(
    verification_report: Mapping[str, Any],
    *,
    transparency_receipt_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic decision journal derived from a verification report."""
    projection = _project_report_for_decision_journal(verification_report)
    receipt_context = (
        None
        if transparency_receipt_context is None
        else _normalise_transparency_receipt_context(transparency_receipt_context)
    )
    journal = _build_decision_journal_from_projection(
        projection,
        transparency_receipt_context=receipt_context,
    )
    validate_decision_journal(journal, expected_reason_codes=projection["decision_reason_codes"])
    return journal


def validate_decision_journal(
    journal: Mapping[str, Any],
    *,
    expected_reason_codes: list[str] | None = None,
) -> None:
    """Validate a derived decision-journal payload."""
    if journal.get("journal_type") != DECISION_JOURNAL_TYPE:
        raise ValueError(
            f"decision_journal.journal_type must be {DECISION_JOURNAL_TYPE!r}."
        )
    if not isinstance(journal.get("entries"), list) or not journal["entries"]:
        raise ValueError("decision_journal.entries must be a non-empty list.")
    reason_codes = journal.get("reason_codes")
    if not isinstance(reason_codes, list) or any(type(item) is not str for item in reason_codes):
        raise ValueError("decision_journal.reason_codes must be a list of strings.")
    if expected_reason_codes is not None and list(expected_reason_codes) != list(reason_codes):
        raise ValueError(
            "decision_journal.reason_codes must match the canonical case-level decision reasons."
        )
    summary = journal.get("journal_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("decision_journal.journal_summary must be a mapping.")
    _validate_transparency_status(str(summary.get("transparency_status")))
    _validate_residual_risk(summary.get("residual_risk"), path="journal_summary.residual_risk")
    previous_event_index = -1
    for index, entry in enumerate(journal["entries"]):
        if not isinstance(entry, Mapping):
            raise ValueError(f"decision_journal.entries[{index}] must be a mapping.")
        event_id = str(entry.get("event_id"))
        if event_id not in _ALLOWED_EVENT_IDS:
            raise ValueError(f"Unsupported decision-journal event_id: {event_id!r}.")
        event_index = _CANONICAL_EVENT_ORDER.index(event_id)
        if event_index <= previous_event_index:
            raise ValueError("decision_journal entry order must follow the canonical event vocabulary.")
        previous_event_index = event_index
        if entry.get("sequence") != index:
            raise ValueError("decision_journal entry sequence values must be contiguous and canonical.")
        status = str(entry.get("status"))
        if status not in _ALLOWED_ENTRY_STATUSES:
            raise ValueError(f"Unsupported decision-journal status: {status!r}.")
        if type(entry.get("summary")) is not str or not str(entry["summary"]).strip():
            raise ValueError("decision_journal entry summaries must be non-empty canonical text.")
        entry_reason_codes = entry.get("reason_codes")
        if not isinstance(entry_reason_codes, list) or any(type(item) is not str for item in entry_reason_codes):
            raise ValueError("decision_journal entry reason_codes must be a list of strings.")
        if any(code not in reason_codes for code in entry_reason_codes):
            raise ValueError(
                "decision_journal entry reason_codes must be a subset of the case-level reason codes."
            )
        evidence_paths = entry.get("evidence_paths")
        if not isinstance(evidence_paths, list) or any(type(item) is not str for item in evidence_paths):
            raise ValueError("decision_journal entry evidence_paths must be a list of strings.")
        for evidence_path in evidence_paths:
            _validate_evidence_path(evidence_path)
        hazards_triggered = entry.get("hazards_triggered")
        if not isinstance(hazards_triggered, list):
            raise ValueError("decision_journal entry hazards_triggered must be a list.")
        for hazard_index, hazard in enumerate(hazards_triggered):
            _validate_code_label_mapping(
                hazard,
                path=f"entries[{index}].hazards_triggered[{hazard_index}]",
                code_field="code",
            )
        mitigations_applied = entry.get("mitigations_applied")
        if not isinstance(mitigations_applied, list):
            raise ValueError("decision_journal entry mitigations_applied must be a list.")
        for mitigation_index, mitigation in enumerate(mitigations_applied):
            _validate_code_label_mapping(
                mitigation,
                path=f"entries[{index}].mitigations_applied[{mitigation_index}]",
                code_field="control_id",
            )
        _validate_residual_risk(
            entry.get("residual_risk"),
            path=f"entries[{index}].residual_risk",
        )
        child_outcomes = entry.get("child_outcomes")
        if child_outcomes is None:
            continue
        if event_id != EVENT_ID_MULTI_SERIES_AGGREGATION:
            raise ValueError("decision_journal child_outcomes are only allowed on multi_series_aggregation.")
        if not isinstance(child_outcomes, list):
            raise ValueError("decision_journal child_outcomes must be a list when present.")
        for child_index, child_outcome in enumerate(child_outcomes):
            if not isinstance(child_outcome, Mapping):
                raise ValueError(
                    f"decision_journal child_outcomes[{child_index}] must be a mapping."
                )
            unexpected = sorted(set(child_outcome) - set(_CHILD_OUTCOME_FIELDS))
            if unexpected:
                raise ValueError(
                    "decision_journal child_outcomes may only contain compact child fields: "
                    f"{', '.join(unexpected)}"
                )
            for field_name in _CHILD_OUTCOME_FIELDS:
                value = child_outcome.get(field_name)
                if type(value) is not str or not value:
                    raise ValueError(
                        f"decision_journal child_outcomes[{child_index}].{field_name} must be a non-empty string."
                    )


def _build_decision_journal_from_projection(
    projection: _JournalProjection,
    *,
    transparency_receipt_context: TransparencyReceiptContext | None,
) -> dict[str, Any]:
    transparency_status = _transparency_status(
        projection,
        transparency_receipt_context=transparency_receipt_context,
    )
    entries: list[dict[str, Any]] = []
    entries.append(_input_received_entry(projection))
    entries.append(_structural_validation_entry(projection))
    if not projection["is_multi_series"] and _proof_stage_events_reached(projection):
        entries.append(_reconstruction_entry(projection))
        entries.append(_parity_checks_entry(projection))
        entries.append(_segmentation_verification_entry(projection))
        entries.append(_volume_proof_entry(projection))
    if _clinical_plausibility_evaluated(projection):
        entries.append(_advisory_heuristics_entry(projection))
    if projection["is_multi_series"]:
        entries.append(_multi_series_aggregation_entry(projection))
    entries.append(_final_decision_entry(projection))
    entries.append(
        _transparency_anchor_entry(
            projection,
            transparency_status=transparency_status,
            transparency_receipt_context=transparency_receipt_context,
        )
    )
    for sequence, entry in enumerate(entries):
        entry["sequence"] = sequence
    final_residual_risk = _final_residual_risk(projection)
    return {
        "schema_version": DECISION_JOURNAL_SCHEMA_VERSION,
        "journal_type": DECISION_JOURNAL_TYPE,
        "case_id": projection["case_id"],
        "case_kind": projection["case_kind"],
        "profile_identity": dict(projection["profile_identity"]),
        "verification_report_hash": projection["verification_report_hash"],
        "final_disposition": projection["decision"]["final_disposition"],
        "decision_class": projection["decision"]["decision_class"],
        "severity": projection["decision"]["severity"],
        "reason_codes": list(projection["decision_reason_codes"]),
        "journal_summary": {
            "transparency_status": transparency_status,
            "residual_risk": final_residual_risk,
        },
        "entries": entries,
    }


def _project_report_for_decision_journal(
    verification_report: Mapping[str, Any],
) -> _JournalProjection:
    if not isinstance(verification_report, Mapping):
        raise ValueError("verification_report must be a mapping.")
    decision = verification_report.get("decision")
    verification_pipeline = verification_report.get("verification_pipeline")
    if not isinstance(decision, Mapping):
        raise ValueError("verification_report.decision must be present for decision-journal derivation.")
    if not isinstance(verification_pipeline, Mapping):
        raise ValueError(
            "verification_report.verification_pipeline must be present for decision-journal derivation."
        )
    profile = verification_report.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("verification_report.profile must be present for decision-journal derivation.")
    is_multi_series = bool(
        verification_report.get("case_kind") == "multi_series_v2"
        or profile.get("interface_type") == "MultiSeriesVerificationProfileDescriptorSet"
    )
    payload: _JournalProjection = {
        "case_id": _report_case_id(verification_report),
        "case_kind": str(
            verification_report.get(
                "case_kind",
                verification_report.get("manifest", {}).get("case_kind", SINGLE_SERIES_CASE_KIND),
            )
        ),
        "profile_identity": _profile_identity(profile),
        "transparency_artifact_allowed": _profile_allows_transparency(profile),
        "verification_report_hash": hash_canonical_json(verification_report),
        "decision": dict(decision),
        "decision_summary": dict(verification_report.get("decision_summary", {})),
        "decision_reason_codes": list(decision.get("reason_codes", [])),
        "decision_reasons": [dict(item) for item in decision.get("decision_reasons", [])],
        "failure_reasons": [dict(item) for item in verification_report.get("failure_reasons", [])],
        "verification_pipeline": dict(verification_pipeline),
        "analysis": dict(verification_report.get("analysis", {})),
        "contracts": dict(verification_report.get("contracts", {})),
        "manifest": dict(verification_report.get("manifest", {})),
        "is_multi_series": is_multi_series,
    }
    if is_multi_series:
        payload["child_outcomes"] = [
            {
                "series_id": str(item["series_id"]),
                "decision_class": str(item["decision"]["decision_class"]),
                "final_disposition": str(item["decision"]["final_disposition"]),
                "gating_status": str(item["gating_status"]),
            }
            for item in verification_report.get("series_reports", [])
            if isinstance(item, Mapping) and isinstance(item.get("decision"), Mapping)
        ]
    return payload


def _report_case_id(verification_report: Mapping[str, Any]) -> str:
    for value in (
        verification_report.get("case_id"),
        verification_report.get("analysis", {}).get("case_id"),
        verification_report.get("manifest", {}).get("case_id"),
        verification_report.get("contracts", {}).get("ai_claim_envelope", {}).get("case_id"),
        verification_report.get("contracts", {}).get("cross_series_verification_trace", {}).get("case_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value
    return "unknown-case"


def _profile_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    if profile.get("interface_type") == "MultiSeriesVerificationProfileDescriptorSet":
        series_descriptors = list(profile.get("series_descriptors", []))
        return {
            "kind": "multi_series_profile_set",
            "descriptor_set_hash": hash_canonical_json(profile),
            "series_count": len(series_descriptors),
        }
    return {
        "kind": "single_profile",
        "profile_name": str(profile.get("profile_name")),
        "profile_revision": str(profile.get("profile_revision")),
        "profile_digest": str(profile.get("profile_digest")),
    }


def _profile_allows_transparency(profile: Mapping[str, Any]) -> bool:
    release_semantics = profile.get("release_semantics")
    if not isinstance(release_semantics, Mapping):
        return True
    return bool(release_semantics.get("transparency_artifact_allowed", False))


def _normalise_transparency_receipt_context(
    transparency_receipt_context: Mapping[str, Any],
) -> TransparencyReceiptContext:
    return build_transparency_receipt_context(
        reference=str(transparency_receipt_context["reference"]),
        entry_hash=str(transparency_receipt_context["entry_hash"]),
        previous_entry_hash=transparency_receipt_context.get("previous_entry_hash"),
        checkpoint=transparency_receipt_context.get("checkpoint"),
        entry_index=int(transparency_receipt_context["entry_index"]),
        log_length=int(transparency_receipt_context["log_length"]),
    )


def _validate_transparency_receipt_context(
    transparency_receipt_context: Mapping[str, Any],
) -> None:
    if type(transparency_receipt_context.get("reference")) is not str:
        raise ValueError("transparency receipt context reference must be a string.")
    if type(transparency_receipt_context.get("entry_hash")) is not str:
        raise ValueError("transparency receipt context entry_hash must be a string.")
    previous_entry_hash = transparency_receipt_context.get("previous_entry_hash")
    if previous_entry_hash is not None and type(previous_entry_hash) is not str:
        raise ValueError("transparency receipt context previous_entry_hash must be a string or None.")
    checkpoint = transparency_receipt_context.get("checkpoint")
    if checkpoint is not None and type(checkpoint) is not str:
        raise ValueError("transparency receipt context checkpoint must be a string or None.")
    if type(transparency_receipt_context.get("entry_index")) is not int:
        raise ValueError("transparency receipt context entry_index must be an int.")
    if type(transparency_receipt_context.get("log_length")) is not int:
        raise ValueError("transparency receipt context log_length must be an int.")


def _input_received_entry(projection: _JournalProjection) -> dict[str, Any]:
    input_hash_path = _first_present_evidence_path(
        projection,
        (
            "verification_report.json#/analysis/input_state_hash",
            "verification_report.json#/analysis/raw_hash",
        ),
    )
    return _entry(
        event_id=EVENT_ID_INPUT_RECEIVED,
        pipeline_phase=PIPELINE_PHASE_INTAKE_VALIDATION,
        status=STATUS_RECORDED,
        summary=(
            "Recorded the case input and fixed the canonical report identity."
            if not projection["is_multi_series"]
            else "Recorded the study input and fixed the canonical report identity."
        ),
        reason_codes=[],
        evidence_paths=[] if input_hash_path is None else [input_hash_path],
        hazards_triggered=[],
        mitigations_applied=[],
        residual_risk=_residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary=(
                "Input receipt alone does not authorize release; later verification stages remain authoritative."
            ),
        ),
    )


def _structural_validation_entry(projection: _JournalProjection) -> dict[str, Any]:
    status = (
        STATUS_FAILED
        if projection["decision"]["decision_class"] == "blocked_structural"
        else STATUS_PASSED
    )
    reason_codes = _event_reason_codes(
        projection,
        pipeline_phases=(PIPELINE_PHASE_INTAKE_VALIDATION, PIPELINE_PHASE_STRUCTURAL_VALIDITY),
    )
    return _entry(
        event_id=EVENT_ID_STRUCTURAL_VALIDATION,
        pipeline_phase=PIPELINE_PHASE_STRUCTURAL_VALIDITY,
        status=status,
        summary=(
            "Input and structural validation completed without blocking findings."
            if status == STATUS_PASSED
            else "Input or structural validation blocked deterministic proof construction."
        ),
        reason_codes=reason_codes,
        evidence_paths=[
            "verification_report.json#/verification_pipeline/phases/intake_validation",
            "verification_report.json#/verification_pipeline/phases/structural_validity",
        ],
        hazards_triggered=_structural_hazards(status),
        mitigations_applied=_structural_mitigations(status),
        residual_risk=(
            _residual_risk(
                classification=RESIDUAL_RISK_BOUNDED,
                summary=(
                    "Structural checks passed; remaining risk is governed by proof, policy, and release control."
                ),
            )
            if status == STATUS_PASSED
            else _residual_risk(
                classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
                summary=(
                    "Guardian blocked before trustworthy proof construction; any further handling sits outside the Guardian release path."
                ),
            )
        ),
    )


def _proof_stage_events_reached(projection: _JournalProjection) -> bool:
    proof_core = dict(projection["verification_pipeline"].get("phases", {})).get(
        PIPELINE_PHASE_PROOF_CORE
    )
    if not isinstance(proof_core, Mapping):
        return False
    return all(
        key in projection["contracts"]
        for key in (
            "measurement_integrity_proof",
            "reconstruction_parity_proof",
            "segmentation_policy_proof",
            "volume_computation_proof",
        )
    )


def _reconstruction_entry(projection: _JournalProjection) -> dict[str, Any]:
    evidence_path = _first_present_evidence_path(
        projection,
        ("verification_report.json#/analysis/derived_reconstruction_hash",),
    )
    return _entry(
        event_id=EVENT_ID_RECONSTRUCTION,
        pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
        status=STATUS_RECORDED,
        summary=(
            "Materialized the raw-derived reconstruction surface used by the downstream proof stages."
        ),
        reason_codes=[],
        evidence_paths=[] if evidence_path is None else [evidence_path],
        hazards_triggered=[],
        mitigations_applied=[],
        residual_risk=_residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary=(
                "Reconstruction materialization is not a release decision; downstream proof outcomes remain authoritative."
            ),
        ),
    )


def _parity_checks_entry(projection: _JournalProjection) -> dict[str, Any]:
    measurement_status = _proof_stage_status(projection, PROOF_STAGE_MEASUREMENT_INTEGRITY)
    reconstruction_status = _proof_stage_status(projection, PROOF_STAGE_RECONSTRUCTION_PARITY)
    decision_reason_codes = _event_reason_codes(
        projection,
        pipeline_phases=(PIPELINE_PHASE_PROOF_CORE,),
        proof_stages=(
            PROOF_STAGE_MEASUREMENT_INTEGRITY,
            PROOF_STAGE_RECONSTRUCTION_PARITY,
        ),
    )
    if STATUS_FAILED in {measurement_status, reconstruction_status}:
        status = STATUS_FAILED
    elif decision_reason_codes:
        status = STATUS_ADVISORY
    else:
        status = STATUS_PASSED
    return _entry(
        event_id=EVENT_ID_PARITY_CHECKS,
        pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
        status=status,
        summary={
            STATUS_PASSED: "Measurement anchoring and reconstruction parity both passed within policy.",
            STATUS_FAILED: "Measurement anchoring or reconstruction parity failed within proof_core.",
            STATUS_ADVISORY: "Measurement anchoring and reconstruction parity passed with near-boundary tolerance proximity.",
        }[status],
        reason_codes=decision_reason_codes,
        evidence_paths=_existing_evidence_paths(
            projection,
            (
                "verification_report.json#/contracts/measurement_integrity_proof",
                "verification_report.json#/contracts/reconstruction_parity_proof",
            ),
        ),
        hazards_triggered=_parity_hazards(
            projection,
            status=status,
            decision_reason_codes=decision_reason_codes,
        ),
        mitigations_applied=_parity_mitigations(
            projection,
            status=status,
        ),
        residual_risk=_parity_residual_risk(status),
    )


def _segmentation_verification_entry(projection: _JournalProjection) -> dict[str, Any]:
    status = _proof_stage_status(projection, PROOF_STAGE_SEGMENTATION_POLICY)
    if status not in {STATUS_PASSED, STATUS_FAILED}:
        status = STATUS_NOT_EVALUATED
    return _entry(
        event_id=EVENT_ID_SEGMENTATION_VERIFICATION,
        pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
        proof_stage=PROOF_STAGE_SEGMENTATION_POLICY,
        status=status,
        summary=(
            "Segmentation verification passed against the policy-derived mask."
            if status == STATUS_PASSED
            else "Segmentation verification failed against the policy-derived mask."
        ),
        reason_codes=_event_reason_codes(
            projection,
            pipeline_phases=(PIPELINE_PHASE_PROOF_CORE,),
            proof_stages=(PROOF_STAGE_SEGMENTATION_POLICY,),
        ),
        evidence_paths=_existing_evidence_paths(
            projection,
            ("verification_report.json#/contracts/segmentation_policy_proof",),
        ),
        hazards_triggered=[],
        mitigations_applied=_segmentation_mitigations(status),
        residual_risk=_segmentation_residual_risk(status),
    )


def _volume_proof_entry(projection: _JournalProjection) -> dict[str, Any]:
    stage_status = _proof_stage_status(projection, PROOF_STAGE_VOLUME_COMPUTATION)
    reason_codes = _event_reason_codes(
        projection,
        pipeline_phases=(PIPELINE_PHASE_PROOF_CORE,),
        proof_stages=(PROOF_STAGE_VOLUME_COMPUTATION,),
    )
    if stage_status == STATUS_FAILED:
        status = STATUS_FAILED
    elif reason_codes:
        status = STATUS_ADVISORY
    else:
        status = STATUS_PASSED
    return _entry(
        event_id=EVENT_ID_VOLUME_PROOF,
        pipeline_phase=PIPELINE_PHASE_PROOF_CORE,
        proof_stage=PROOF_STAGE_VOLUME_COMPUTATION,
        status=status,
        summary={
            STATUS_PASSED: "Volume proof passed against verified-mask geometry within policy.",
            STATUS_FAILED: "Volume proof failed against verified-mask geometry within policy.",
            STATUS_ADVISORY: "Volume proof passed with near-boundary tolerance proximity.",
        }[status],
        reason_codes=reason_codes,
        evidence_paths=_existing_evidence_paths(
            projection,
            ("verification_report.json#/contracts/volume_computation_proof",),
        ),
        hazards_triggered=_volume_hazards(status),
        mitigations_applied=_volume_mitigations(status),
        residual_risk=_volume_residual_risk(status),
    )


def _clinical_plausibility_evaluated(projection: _JournalProjection) -> bool:
    phase_status = _phase_status(projection, PIPELINE_PHASE_CLINICAL_PLAUSIBILITY)
    return phase_status not in {"", "not_evaluated"}


def _advisory_heuristics_entry(projection: _JournalProjection) -> dict[str, Any]:
    phase_status = _phase_status(projection, PIPELINE_PHASE_CLINICAL_PLAUSIBILITY)
    if phase_status == "failed":
        status = STATUS_FAILED
    elif phase_status in {"warning", "advisory"}:
        status = STATUS_ADVISORY
    else:
        status = STATUS_PASSED
    return _entry(
        event_id=EVENT_ID_ADVISORY_HEURISTICS,
        pipeline_phase=PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,
        status=status,
        summary={
            STATUS_PASSED: "Clinical plausibility and temporal heuristics were evaluated without release-changing findings.",
            STATUS_ADVISORY: "Clinical plausibility or temporal heuristics recorded non-blocking review findings.",
            STATUS_FAILED: "Clinical plausibility heuristics produced blocking findings under the active policy.",
        }[status],
        reason_codes=_event_reason_codes(
            projection,
            pipeline_phases=(PIPELINE_PHASE_CLINICAL_PLAUSIBILITY,),
        ),
        evidence_paths=_existing_evidence_paths(
            projection,
            ("verification_report.json#/manifest/clinical_plausibility",),
        ),
        hazards_triggered=[],
        mitigations_applied=_heuristic_mitigations(status),
        residual_risk=_heuristic_residual_risk(status),
    )


def _multi_series_aggregation_entry(projection: _JournalProjection) -> dict[str, Any]:
    status = (
        STATUS_PASSED
        if projection["decision"]["final_disposition"] == "released"
        else STATUS_FAILED
    )
    event = _entry(
        event_id=EVENT_ID_MULTI_SERIES_AGGREGATION,
        pipeline_phase=PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,
        status=status,
        summary=(
            "Study aggregation accepted the required series set for downstream release."
            if status == STATUS_PASSED
            else "Study aggregation blocked downstream release for the required series set."
        ),
        reason_codes=_event_reason_codes(
            projection,
            pipeline_phases=(PIPELINE_PHASE_MULTI_SERIES_AGGREGATION,),
        ),
        evidence_paths=_existing_evidence_paths(
            projection,
            (
                "verification_report.json#/contracts/cross_series_verification_trace",
                "verification_report.json#/contracts/study_temporal_aggregation_trace",
            ),
        ),
        hazards_triggered=_multi_series_hazards(projection, status),
        mitigations_applied=_multi_series_mitigations(status),
        residual_risk=_multi_series_residual_risk(projection, status),
    )
    event["child_outcomes"] = list(projection.get("child_outcomes", []))
    return event


def _final_decision_entry(projection: _JournalProjection) -> dict[str, Any]:
    released = projection["decision"]["final_disposition"] == "released"
    return _entry(
        event_id=EVENT_ID_FINAL_DECISION,
        pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
        status=STATUS_PASSED if released else STATUS_FAILED,
        summary=(
            "The canonical case decision released the Guardian output."
            if not projection["is_multi_series"] and released
            else "The canonical case decision blocked the Guardian output."
            if not projection["is_multi_series"]
            else "The canonical study decision released the Guardian output."
            if released
            else "The canonical study decision blocked the Guardian output."
        ),
        reason_codes=list(projection["decision_reason_codes"]),
        evidence_paths=["verification_report.json#/decision"],
        hazards_triggered=[],
        mitigations_applied=_final_decision_mitigations(released),
        residual_risk=_final_residual_risk(projection),
    )


def _transparency_anchor_entry(
    projection: _JournalProjection,
    *,
    transparency_status: str,
    transparency_receipt_context: TransparencyReceiptContext | None,
) -> dict[str, Any]:
    del projection
    del transparency_receipt_context
    return _entry(
        event_id=EVENT_ID_TRANSPARENCY_ANCHOR,
        pipeline_phase=PIPELINE_PHASE_RELEASE_CONTROL,
        status=transparency_status,
        summary={
            STATUS_RECORDED: "Transparency anchoring was recorded for the evidence envelope.",
            STATUS_ABSENT: "No transparency anchoring record is available for this journaled case outcome.",
            STATUS_NOT_APPLICABLE: "Transparency anchoring was not applicable for this non-final report outcome.",
        }[transparency_status],
        reason_codes=[],
        evidence_paths=["transparency-log.jsonl"] if transparency_status == STATUS_RECORDED else [],
        hazards_triggered=[],
        mitigations_applied=_transparency_mitigations(transparency_status),
        residual_risk=_transparency_residual_risk(transparency_status),
    )


def _transparency_status(
    projection: _JournalProjection,
    *,
    transparency_receipt_context: TransparencyReceiptContext | None,
) -> str:
    if transparency_receipt_context is not None:
        return STATUS_RECORDED
    if not projection["transparency_artifact_allowed"]:
        return STATUS_NOT_APPLICABLE
    return STATUS_ABSENT


def _phase_status(projection: _JournalProjection, pipeline_phase: str) -> str:
    phases = dict(projection["verification_pipeline"].get("phases", {}))
    payload = phases.get(pipeline_phase, {})
    if not isinstance(payload, Mapping):
        return ""
    return str(payload.get("status", ""))


def _proof_stage_status(projection: _JournalProjection, proof_stage: str) -> str:
    phases = dict(projection["verification_pipeline"].get("phases", {}))
    proof_core = phases.get(PIPELINE_PHASE_PROOF_CORE, {})
    if not isinstance(proof_core, Mapping):
        return STATUS_NOT_EVALUATED
    for stage in proof_core.get("proof_stages", []):
        if str(stage.get("proof_stage")) != proof_stage:
            continue
        raw_status = str(stage.get("status"))
        return {
            "passed": STATUS_PASSED,
            "failed": STATUS_FAILED,
            "not_evaluated": STATUS_NOT_EVALUATED,
        }.get(raw_status, STATUS_NOT_EVALUATED)
    return STATUS_NOT_EVALUATED


def _event_reason_codes(
    projection: _JournalProjection,
    *,
    pipeline_phases: tuple[str, ...],
    proof_stages: tuple[str, ...] | None = None,
) -> list[str]:
    allowed = set(projection["decision_reason_codes"])
    filtered: list[str] = []
    for item in projection["decision_reasons"]:
        if str(item.get("pipeline_phase")) not in pipeline_phases:
            continue
        if proof_stages is not None and str(item.get("proof_stage", "")) not in proof_stages:
            continue
        code = str(item.get("code"))
        if code in allowed and code not in filtered:
            filtered.append(code)
    return filtered


def _first_present_evidence_path(
    projection: _JournalProjection,
    candidates: tuple[str, ...],
) -> str | None:
    available = _existing_evidence_paths(projection, candidates)
    return None if not available else available[0]


def _existing_evidence_paths(
    projection: _JournalProjection,
    candidates: tuple[str, ...],
) -> list[str]:
    available: list[str] = []
    for candidate in candidates:
        if _evidence_path_exists(projection, candidate):
            available.append(candidate)
    return available


def _evidence_path_exists(projection: _JournalProjection, evidence_path: str) -> bool:
    if evidence_path == "transparency-log.jsonl":
        return True
    if not evidence_path.startswith("verification_report.json#/"):
        return False
    pointer = evidence_path.removeprefix("verification_report.json#/")
    current: Any = {
        "analysis": projection["analysis"],
        "contracts": projection["contracts"],
        "manifest": projection["manifest"],
        "decision": projection["decision"],
        "decision_summary": projection["decision_summary"],
        "verification_pipeline": projection["verification_pipeline"],
    }
    for part in pointer.split("/"):
        if isinstance(current, Mapping):
            if part not in current:
                return False
            current = current[part]
            continue
        if isinstance(current, list):
            if not part.isdigit():
                return False
            index = int(part)
            if index >= len(current):
                return False
            current = current[index]
            continue
        return False
    return True


def _entry(
    *,
    event_id: str,
    pipeline_phase: str,
    status: str,
    summary: str,
    reason_codes: list[str],
    evidence_paths: list[str],
    hazards_triggered: list[dict[str, str]],
    mitigations_applied: list[dict[str, str]],
    residual_risk: dict[str, str],
    proof_stage: str | None = None,
) -> dict[str, Any]:
    payload = {
        "event_id": event_id,
        "pipeline_phase": pipeline_phase,
        "status": status,
        "summary": summary,
        "reason_codes": list(reason_codes),
        "evidence_paths": list(evidence_paths),
        "hazards_triggered": list(hazards_triggered),
        "mitigations_applied": list(mitigations_applied),
        "residual_risk": dict(residual_risk),
    }
    if proof_stage is not None:
        payload["proof_stage"] = proof_stage
    return payload


def _code_label(code: str, label: str) -> dict[str, str]:
    return {"code": code, "label": label}


def _control(control_id: str, label: str) -> dict[str, str]:
    return {"control_id": control_id, "label": label}


def _residual_risk(*, classification: str, summary: str) -> dict[str, str]:
    return {
        "classification": classification,
        "summary": summary,
    }


def _structural_hazards(status: str) -> list[dict[str, str]]:
    if status != STATUS_FAILED:
        return []
    return [_code_label("misconfigured_verification_process", "Misconfigured verification process")]


def _structural_mitigations(status: str) -> list[dict[str, str]]:
    if status == STATUS_FAILED:
        return [
            _control("intake_validation", "Input acceptance gate"),
            _control("structural_validity", "Structural validity checks"),
            _control("verification_profile_descriptor", "Verification profile descriptor"),
            _control("release_control", "Release control"),
        ]
    return [
        _control("intake_validation", "Input acceptance gate"),
        _control("structural_validity", "Structural validity checks"),
    ]


def _parity_hazards(
    projection: _JournalProjection,
    *,
    status: str,
    decision_reason_codes: list[str],
) -> list[dict[str, str]]:
    if status != STATUS_FAILED:
        return []
    if _proof_stage_status(projection, PROOF_STAGE_MEASUREMENT_INTEGRITY) == STATUS_FAILED:
        return [_code_label("measurement_domain_integrity_distortion", "Measurement-domain integrity distortion")]
    if _proof_stage_status(projection, PROOF_STAGE_RECONSTRUCTION_PARITY) == STATUS_FAILED:
        return [_code_label("image_integrity_distortion", "Image integrity distortion")]
    if decision_reason_codes:
        return []
    return []


def _parity_mitigations(
    projection: _JournalProjection,
    *,
    status: str,
) -> list[dict[str, str]]:
    controls = [
        _control("measurement_integrity_proof", "Measurement integrity proof"),
        _control("reconstruction_parity_proof", "Reconstruction parity proof"),
        _control("verification_tolerance_policy", "Verification tolerance policy"),
    ]
    if status == STATUS_FAILED:
        controls.extend(
            [
                _control("proof_core_evidence_trace", "Proof-core evidence trace"),
                _control("release_control", "Release control"),
            ]
        )
    return controls


def _parity_residual_risk(status: str) -> dict[str, str]:
    if status == STATUS_PASSED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Parity checks passed within policy; remaining risk is bounded by later mask, volume, and policy review.",
        )
    if status == STATUS_ADVISORY:
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Parity passed close to a configured boundary and should be reviewed in context.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_BOUNDED,
        summary="Guardian blocked release on measurement or reconstruction mismatch; residual risk remains only outside the blocked Guardian path.",
    )


def _segmentation_mitigations(status: str) -> list[dict[str, str]]:
    controls = [
        _control("mask_derivation_policy", "Mask derivation policy"),
        _control("segmentation_policy_proof", "Segmentation policy proof"),
    ]
    if status == STATUS_FAILED:
        controls.extend(
            [
                _control("proof_core_evidence_trace", "Proof-core evidence trace"),
                _control("release_control", "Release control"),
            ]
        )
    return controls


def _segmentation_residual_risk(status: str) -> dict[str, str]:
    if status == STATUS_PASSED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Mask verification passed; remaining risk is bounded by volume, policy, and downstream clinical review.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_BOUNDED,
        summary="Guardian blocked release on verified-mask disagreement; residual risk remains only outside the blocked Guardian path.",
    )


def _volume_hazards(status: str) -> list[dict[str, str]]:
    if status != STATUS_FAILED:
        return []
    return [_code_label("quantitative_misstatement", "Quantitative misstatement")]


def _volume_mitigations(status: str) -> list[dict[str, str]]:
    controls = [_control("volume_computation_proof", "Volume computation proof")]
    if status == STATUS_FAILED:
        controls.extend(
            [
                _control("segmentation_policy_proof", "Segmentation policy proof"),
                _control("proof_core_evidence_trace", "Proof-core evidence trace"),
                _control("release_control", "Release control"),
            ]
        )
    return controls


def _volume_residual_risk(status: str) -> dict[str, str]:
    if status == STATUS_PASSED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Volume proof passed within policy; remaining risk is bounded by downstream review of the released case.",
        )
    if status == STATUS_ADVISORY:
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Volume proof passed close to a configured boundary and remains review-relevant.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_BOUNDED,
        summary="Guardian blocked release on verified-volume disagreement; residual risk remains only outside the blocked Guardian path.",
    )


def _heuristic_mitigations(status: str) -> list[dict[str, str]]:
    controls = [_control("clinical_plausibility", "Clinical plausibility review")]
    if status != STATUS_PASSED:
        controls.append(_control("policy_evaluation", "Policy evaluation"))
    return controls


def _heuristic_residual_risk(status: str) -> dict[str, str]:
    if status == STATUS_PASSED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Heuristics added no release-changing findings; remaining risk is bounded by the canonical decision.",
        )
    if status == STATUS_ADVISORY:
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Non-blocking heuristic findings remain review-relevant to an operator.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
        summary="Blocking heuristic findings require explicit operator or workflow handling.",
    )


def _multi_series_hazards(
    projection: _JournalProjection,
    status: str,
) -> list[dict[str, str]]:
    if status != STATUS_FAILED:
        return []
    if projection["decision_summary"].get("block_category") == "positive_lesion_corroboration_failed":
        return [_code_label("false_corroborative_confidence", "False corroborative confidence")]
    return []


def _multi_series_mitigations(status: str) -> list[dict[str, str]]:
    controls = [
        _control("cross_series_verification_trace", "Cross-series verification trace"),
        _control("multi_series_aggregation", "Multi-series aggregation gate"),
    ]
    if status == STATUS_FAILED:
        controls.append(_control("release_control", "Release control"))
    return controls


def _multi_series_residual_risk(
    projection: _JournalProjection,
    status: str,
) -> dict[str, str]:
    if status == STATUS_PASSED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Study aggregation accepted the required series set; remaining risk is bounded by review of the released study.",
        )
    block_category = str(projection["decision_summary"].get("block_category", ""))
    if block_category in {"missing_required_series", "required_series_geometry_mismatch"}:
        return _residual_risk(
            classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
            summary="Study aggregation blocked before a trustworthy merged release could be formed; any follow-on handling sits outside the Guardian release path.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_BOUNDED,
        summary="Study aggregation blocked release on cross-series verification grounds; residual risk remains only outside the blocked Guardian path.",
    )


def _final_decision_mitigations(released: bool) -> list[dict[str, str]]:
    del released
    return [_control("release_control", "Release control")]


def _final_residual_risk(projection: _JournalProjection) -> dict[str, str]:
    decision_class = str(projection["decision"]["decision_class"])
    if decision_class == "verified_clean":
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Guardian released the case without advisory modifiers; residual risk remains bounded by external clinical judgement and upstream acquisition correctness.",
        )
    if decision_class == "verified_boundary_hugging":
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Guardian released the case close to a configured proof boundary; the outcome remains review-relevant to an operator.",
        )
    if decision_class == "verified_with_advisory":
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Guardian released the case with non-blocking advisory findings; the outcome remains review-relevant to an operator.",
        )
    if decision_class == "blocked_structural":
        return _residual_risk(
            classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
            summary="Guardian blocked the case before trustworthy proof construction; any further handling sits outside the Guardian release path.",
        )
    if decision_class == "blocked_physical":
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Guardian blocked release on measured-data verification grounds; residual risk remains only if upstream data are wrong yet internally consistent.",
        )
    if decision_class == "blocked_policy":
        if "non_final_profile_selected" in projection["decision_reason_codes"]:
            return _residual_risk(
                classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
                summary="Guardian withheld final release because the selected profile is non-final; any downstream handling sits outside the Guardian release path.",
            )
        return _residual_risk(
            classification=RESIDUAL_RISK_OPERATOR_RELEVANT,
            summary="Guardian blocked final release under configured policy despite available verification evidence; the outcome remains workflow- and operator-relevant.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_BOUNDED,
        summary="Residual risk is bounded by the canonical Guardian decision.",
    )


def _transparency_mitigations(transparency_status: str) -> list[dict[str, str]]:
    if transparency_status != STATUS_RECORDED:
        return []
    return [_control("transparency_log_hash_chain", "Transparency log hash chain")]


def _transparency_residual_risk(transparency_status: str) -> dict[str, str]:
    if transparency_status == STATUS_RECORDED:
        return _residual_risk(
            classification=RESIDUAL_RISK_BOUNDED,
            summary="Audit anchoring is recorded in the transparency log snapshot.",
        )
    if transparency_status == STATUS_NOT_APPLICABLE:
        return _residual_risk(
            classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
            summary="Transparency anchoring is not applicable for this non-final report outcome.",
        )
    return _residual_risk(
        classification=RESIDUAL_RISK_WORKFLOW_EXTERNAL,
        summary="No transparency anchor is available in the current artifact set for this case outcome.",
    )


def _validate_transparency_status(status: str) -> None:
    if status not in {STATUS_RECORDED, STATUS_ABSENT, STATUS_NOT_APPLICABLE}:
        raise ValueError(
            "decision_journal transparency status must be recorded, absent, or not_applicable."
        )


def _validate_residual_risk(payload: Any, *, path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    classification = payload.get("classification")
    if classification not in _ALLOWED_RESIDUAL_RISK_CLASSIFICATIONS:
        raise ValueError(f"{path}.classification is not a supported residual-risk value.")
    if type(payload.get("summary")) is not str or not str(payload["summary"]).strip():
        raise ValueError(f"{path}.summary must be a non-empty string.")


def _validate_code_label_mapping(
    payload: Any,
    *,
    path: str,
    code_field: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    if type(payload.get(code_field)) is not str or not str(payload[code_field]).strip():
        raise ValueError(f"{path}.{code_field} must be a non-empty string.")
    if type(payload.get("label")) is not str or not str(payload["label"]).strip():
        raise ValueError(f"{path}.label must be a non-empty string.")


def _validate_evidence_path(evidence_path: str) -> None:
    if evidence_path == "transparency-log.jsonl":
        return
    if not evidence_path.startswith("verification_report.json#/"):
        raise ValueError(
            "decision_journal evidence_paths must reference verification_report.json#/... "
            "or transparency-log.jsonl."
        )
