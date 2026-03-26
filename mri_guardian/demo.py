"""Deterministic demo runner for the Verifiable MRI Guardian."""

from __future__ import annotations

import argparse
from pathlib import Path

from icsd.core.serialise import canonical_json_dumps

from .harness import (
    MriGuardianHarness,
    build_demo_authority_sources,
    build_demo_case,
    render_release_bundle_json,
)
from .tamper import build_tamper_mode_base_case, build_tamper_scenarios
from .viewer import render_side_by_side_report_html

DEMO_TIMESTAMP = "2026-03-26T00:00:00Z"


def run_demo(*, output_dir: Path | str, tamper_mode: bool = False) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison_rejected_candidate.html").unlink(missing_ok=True)
    case_state = build_demo_case()
    harness = MriGuardianHarness(
        transparency_log_path=output_path / "transparency-log.jsonl",
    )
    result = harness.release_case(
        state=case_state,
        actor="guardian:demo",
        authority_sources=build_demo_authority_sources(case_id=case_state["case_id"]),
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
    (output_path / "comparison.html").write_text(
        render_side_by_side_report_html(
            case_state,
            report_title="Accepted Case: AI vs Verified Reality",
        ),
        encoding="utf-8",
    )
    if tamper_mode:
        _write_tamper_mode_reports(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_artifacts",
        help="Directory where evidence and workstation payload files will be written.",
    )
    parser.add_argument(
        "--tamper-mode",
        action="store_true",
        help="Write deterministic tamper-mode reports that demonstrate live failure detection.",
    )
    args = parser.parse_args()
    run_demo(output_dir=args.output_dir, tamper_mode=args.tamper_mode)
    return 0


def _write_tamper_mode_reports(output_path: Path) -> None:
    baseline_state = build_tamper_mode_base_case()
    (output_path / "tamper_baseline.html").write_text(
        render_side_by_side_report_html(
            baseline_state,
            report_title="Tamper Mode Baseline: Verified Reality",
        ),
        encoding="utf-8",
    )
    for scenario in build_tamper_scenarios():
        report_path = output_path / f"tamper_{scenario['slug']}.html"
        report_path.write_text(
            render_side_by_side_report_html(
                scenario["state"],
                report_title=scenario["title"],
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
