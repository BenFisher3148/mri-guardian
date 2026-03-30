"""Deterministic demo runner for the MRI Guardian Layer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from icsd.core.serialise import canonical_json_dumps

from .bundle import (
    MRI_GUARD_BUNDLE_EXTENSION,
    verify_bundle,
    write_portable_evidence_bundle,
)
from .harness import (
    MULTI_SERIES_CASE_KIND,
    MriGuardianHarness,
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
    build_multi_series_demo_authority_sources,
    build_multi_series_demo_case,
    render_release_bundle_json,
)
from .integration import (
    build_guardian_integration_narrative,
    build_mock_dicom_input,
    build_mock_integration_trace,
    build_mock_pacs_block_notice,
    build_mock_pacs_submission,
)
from .profiles import DEFAULT_PROFILE_NAME, get_verification_profile
from .tamper import (
    build_tamper_mode_base_case,
    build_tamper_scenarios,
    build_verification_threat_corpus,
)
from .timing import build_guardian_performance_profile
from .viewer import render_side_by_side_report_html

DEMO_TIMESTAMP = "2026-03-26T00:00:00Z"
_PROFILE_ARTIFACT_NAMES = (
    "verification_profile.json",
    "verification_profile_descriptor.json",
)


def run_demo(
    *,
    output_dir: Path | str,
    profile_name: str = DEFAULT_PROFILE_NAME,
    tamper_mode: bool = False,
    case_kind: str = "single_series_v1",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison_rejected_candidate.html").unlink(missing_ok=True)
    (output_path / "prior_verified_contexts.json").unlink(missing_ok=True)
    (output_path / "prior_verified_context.json").unlink(missing_ok=True)
    (output_path / "longitudinal_selection_trace.json").unlink(missing_ok=True)
    (output_path / "temporal_trajectory_trace.json").unlink(missing_ok=True)
    (output_path / "longitudinal_registration_trace.json").unlink(missing_ok=True)
    (output_path / "temporal_consistency_trace.json").unlink(missing_ok=True)
    (output_path / "study_temporal_aggregation_trace.json").unlink(missing_ok=True)
    (output_path / "verification_threat_corpus.json").unlink(missing_ok=True)
    descriptor = get_verification_profile(profile_name)
    if case_kind == MULTI_SERIES_CASE_KIND:
        case_state = build_multi_series_demo_case()
    else:
        case_state = build_demo_case(profile_name=descriptor.profile_name)
    pending_report = build_case_verification_report(
        case_state,
    )
    _write_profile_artifacts(output_path, pending_report["profile"])
    (output_path / "verification_report.json").write_text(
        f"{canonical_json_dumps(pending_report)}\n",
        encoding="utf-8",
    )
    _write_single_series_temporal_artifacts(output_path, pending_report)
    (output_path / "performance_profile.json").write_text(
        f"{canonical_json_dumps(_build_demo_performance_profile(case_state, pending_report, descriptor.profile_name))}\n",
        encoding="utf-8",
    )
    (output_path / "comparison.html").write_text(
        render_side_by_side_report_html(
            case_state,
            report_title="Accepted Case: Guardian Layer Verification",
            profile_name=descriptor.profile_name,
        ),
        encoding="utf-8",
    )
    if not descriptor.release_semantics["final_release_allowed"]:
        if tamper_mode:
            _write_tamper_mode_reports(output_path, profile_name=descriptor.profile_name)
        return

    dicom_input = build_mock_dicom_input(case_state)
    (output_path / "dicom_input.json").write_text(
        f"{canonical_json_dumps(dicom_input)}\n",
        encoding="utf-8",
    )
    (output_path / "integration_narrative.json").write_text(
        f"{canonical_json_dumps(build_guardian_integration_narrative(current_guardian_decision='released'))}\n",
        encoding="utf-8",
    )
    harness = MriGuardianHarness(
        transparency_log_path=output_path / "transparency-log.jsonl",
        default_profile_name=descriptor.profile_name,
    )
    if case_kind == MULTI_SERIES_CASE_KIND:
        result = harness.release_multi_series_case(
            state=case_state,
            actor="guardian:demo",
            authority_sources_by_series_id=build_multi_series_demo_authority_sources(),
            timestamp=DEMO_TIMESTAMP,
        )
    else:
        result = harness.release_case(
            state=case_state,
            actor="guardian:demo",
            authority_sources=build_demo_authority_sources(
                case_id=case_state["case_id"],
                profile_name=descriptor.profile_name,
            ),
            profile_name=descriptor.profile_name,
            timestamp=DEMO_TIMESTAMP,
        )
    (output_path / "evidence.json").write_text(
        f"{result.evidence_json}\n",
        encoding="utf-8",
    )
    (output_path / "workstation_payload.json").write_text(
        f"{render_release_bundle_json(result)}\n",
        encoding="utf-8",
    )
    released_state = canonical_json_dumps(result.after_state)
    (output_path / "released_state.json").write_text(
        f"{released_state}\n",
        encoding="utf-8",
    )
    report = build_case_verification_report(result.after_state)
    _write_profile_artifacts(output_path, report["contracts"]["verification_profile_descriptor"])
    (output_path / "verification_report.json").write_text(
        f"{canonical_json_dumps(report)}\n",
        encoding="utf-8",
    )
    _write_single_series_temporal_artifacts(output_path, report)
    if result.after_state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        _write_multi_series_contract_artifacts(output_path, report)
    else:
        (output_path / "ai_claim_envelope.json").write_text(
            f"{canonical_json_dumps(report['contracts']['ai_claim_envelope'])}\n",
            encoding="utf-8",
        )
        (output_path / "measurement_integrity_proof.json").write_text(
            f"{canonical_json_dumps(report['contracts']['measurement_integrity_proof'])}\n",
            encoding="utf-8",
        )
        (output_path / "reconstruction_parity_proof.json").write_text(
            f"{canonical_json_dumps(report['contracts']['reconstruction_parity_proof'])}\n",
            encoding="utf-8",
        )
        (output_path / "segmentation_policy_proof.json").write_text(
            f"{canonical_json_dumps(report['contracts']['segmentation_policy_proof'])}\n",
            encoding="utf-8",
        )
        (output_path / "volume_computation_proof.json").write_text(
            f"{canonical_json_dumps(report['contracts']['volume_computation_proof'])}\n",
            encoding="utf-8",
        )
        (output_path / "proof_core_evidence_trace.json").write_text(
            f"{canonical_json_dumps(report['contracts']['proof_core_evidence_trace'])}\n",
            encoding="utf-8",
        )
        (output_path / "mask_derivation_policy.json").write_text(
            f"{canonical_json_dumps(report['contracts']['mask_derivation_policy'])}\n",
            encoding="utf-8",
        )
        (output_path / "verification_tolerance_policy.json").write_text(
            f"{canonical_json_dumps(report['contracts']['verification_tolerance_policy'])}\n",
            encoding="utf-8",
        )
    bundle_path = output_path / f"{case_state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
    write_portable_evidence_bundle(
        result=result,
        transparency_log_path=output_path / "transparency-log.jsonl",
        output_path=bundle_path,
    )
    bundle_verification = verify_bundle(bundle_path)
    if not bundle_verification["verified"]:
        msg = f"Portable bundle verification failed: {bundle_verification['failures']!r}"
        raise ValueError(msg)
    pacs_submission = build_mock_pacs_submission(
        result,
        bundle_reference=bundle_path.name,
    )
    integration_trace = build_mock_integration_trace(
        dicom_input=dicom_input,
        pacs_output=pacs_submission,
    )
    (output_path / "pacs_submission.json").write_text(
        f"{canonical_json_dumps(pacs_submission)}\n",
        encoding="utf-8",
    )
    (output_path / "integration_trace.json").write_text(
        f"{canonical_json_dumps(integration_trace)}\n",
        encoding="utf-8",
    )
    (output_path / "bundle_verification.json").write_text(
        f"{canonical_json_dumps(bundle_verification)}\n",
        encoding="utf-8",
    )
    if tamper_mode:
        _write_tamper_mode_reports(output_path, profile_name=descriptor.profile_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_artifacts",
        help="Directory where evidence and workstation payload files will be written.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        help="Guardian runtime profile to use for the demo artifacts.",
    )
    parser.add_argument(
        "--tamper-mode",
        action="store_true",
        help="Write deterministic tamper-mode reports that demonstrate live failure detection.",
    )
    parser.add_argument(
        "--case-kind",
        choices=("single_series_v1", MULTI_SERIES_CASE_KIND),
        default="single_series_v1",
        help="Generate either the legacy single-series demo or the additive multi-series v2 demo.",
    )
    args = parser.parse_args()
    run_demo(
        output_dir=args.output_dir,
        profile_name=args.profile,
        tamper_mode=args.tamper_mode,
        case_kind=args.case_kind,
    )
    return 0


def _write_tamper_mode_reports(output_path: Path, *, profile_name: str) -> None:
    baseline_state = build_tamper_mode_base_case(profile_name=profile_name)
    (output_path / "tamper_baseline.html").write_text(
        render_side_by_side_report_html(
            baseline_state,
            report_title="Tamper Mode Baseline: Verified Reality",
            profile_name=profile_name,
        ),
        encoding="utf-8",
    )
    corpus = build_verification_threat_corpus(profile_name=profile_name)
    corpus_entries: list[dict[str, Any]] = []
    compatibility_slugs = {
        item["slug"] for item in build_tamper_scenarios(profile_name=profile_name)
    }
    for scenario in corpus["entries"]:
        report_path = output_path / f"tamper_{scenario['slug']}.html"
        dicom_input = build_mock_dicom_input(scenario["state"])
        pacs_block = build_mock_pacs_block_notice(scenario["state"])
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_block,
        )
        render_profile_name = scenario.get("profile_name") or profile_name
        report_path.write_text(
            render_side_by_side_report_html(
                scenario["state"],
                report_title=scenario["title"],
                profile_name=render_profile_name,
            ),
            encoding="utf-8",
        )
        dicom_path = output_path / f"tamper_{scenario['slug']}_dicom_input.json"
        block_path = output_path / f"tamper_{scenario['slug']}_pacs_block.json"
        trace_path = output_path / f"tamper_{scenario['slug']}_integration_trace.json"
        dicom_path.write_text(
            f"{canonical_json_dumps(dicom_input)}\n",
            encoding="utf-8",
        )
        block_path.write_text(
            f"{canonical_json_dumps(pacs_block)}\n",
            encoding="utf-8",
        )
        trace_path.write_text(
            f"{canonical_json_dumps(integration_trace)}\n",
            encoding="utf-8",
        )
        corpus_entry = {key: value for key, value in scenario.items() if key != "state"}
        corpus_entry["legacy_curated_subset"] = scenario["slug"] in compatibility_slugs
        corpus_entry["artifact_paths"] = {
            "html": report_path.name,
            "dicom_input": dicom_path.name,
            "pacs_block": block_path.name,
            "integration_trace": trace_path.name,
        }
        corpus_entries.append(corpus_entry)
    corpus_payload = {
        key: value for key, value in corpus.items() if key != "entries"
    }
    corpus_payload["entries"] = corpus_entries
    (output_path / "verification_threat_corpus.json").write_text(
        f"{canonical_json_dumps(corpus_payload)}\n",
        encoding="utf-8",
    )


def _write_profile_artifacts(output_path: Path, payload: dict[str, object]) -> None:
    content = f"{canonical_json_dumps(payload)}\n"
    for filename in _PROFILE_ARTIFACT_NAMES:
        (output_path / filename).write_text(content, encoding="utf-8")


def _write_multi_series_contract_artifacts(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    (output_path / "study_temporal_aggregation_trace.json").write_text(
        f"{canonical_json_dumps(report['contracts']['study_temporal_aggregation_trace'])}\n",
        encoding="utf-8",
    )
    (output_path / "cross_series_verification_policy.json").write_text(
        f"{canonical_json_dumps(report['contracts']['cross_series_verification_policy'])}\n",
        encoding="utf-8",
    )
    (output_path / "cross_series_verification_trace.json").write_text(
        f"{canonical_json_dumps(report['contracts']['cross_series_verification_trace'])}\n",
        encoding="utf-8",
    )
    series_contracts = {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "series": [
            {
                "series_id": item["series_id"],
                "resolved_role": item["resolved_role"],
                "gating_status": item["gating_status"],
                "series_proof_status": item["series_proof_status"],
                "downstream_release_allowed": item["downstream_release_allowed"],
                "contracts": item["contracts"],
            }
            for item in report["series_reports"]
        ],
    }
    (output_path / "series_contracts.json").write_text(
        f"{canonical_json_dumps(series_contracts)}\n",
        encoding="utf-8",
    )


def _write_single_series_temporal_artifacts(
    output_path: Path,
    report: dict[str, Any],
) -> None:
    contracts = report["contracts"]
    prior_verified_contexts = contracts.get("prior_verified_contexts")
    prior_verified_context = contracts.get("prior_verified_context")
    if prior_verified_context is None and prior_verified_contexts is None:
        return
    if prior_verified_contexts is not None:
        (output_path / "prior_verified_contexts.json").write_text(
            f"{canonical_json_dumps(prior_verified_contexts)}\n",
            encoding="utf-8",
        )
    if prior_verified_context is not None:
        (output_path / "prior_verified_context.json").write_text(
            f"{canonical_json_dumps(prior_verified_context)}\n",
            encoding="utf-8",
        )
    (output_path / "longitudinal_selection_trace.json").write_text(
        f"{canonical_json_dumps(contracts['longitudinal_selection_trace'])}\n",
        encoding="utf-8",
    )
    (output_path / "temporal_trajectory_trace.json").write_text(
        f"{canonical_json_dumps(contracts['temporal_trajectory_trace'])}\n",
        encoding="utf-8",
    )
    (output_path / "longitudinal_registration_trace.json").write_text(
        f"{canonical_json_dumps(contracts['longitudinal_registration_trace'])}\n",
        encoding="utf-8",
    )
    (output_path / "temporal_consistency_trace.json").write_text(
        f"{canonical_json_dumps(contracts['temporal_consistency_trace'])}\n",
        encoding="utf-8",
    )


def _build_demo_performance_profile(
    state: dict[str, Any],
    pending_report: dict[str, Any],
    fallback_profile_name: str,
) -> dict[str, Any]:
    if state.get("case_kind") != MULTI_SERIES_CASE_KIND:
        return build_guardian_performance_profile(fallback_profile_name)
    return {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "series": [
            {
                "series_id": item["series_id"],
                "profile_name": item["profile"]["profile_name"],
                "performance_profile": build_guardian_performance_profile(
                    item["profile"]["profile_name"]
                ),
            }
            for item in pending_report["series_reports"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
