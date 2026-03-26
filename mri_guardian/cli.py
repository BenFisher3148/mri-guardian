"""Command-line entrypoints for the MRI Guardian Layer."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from typing import Any

from icsd.core.evidence import validate_evidence_payload
from icsd.core.serialise import canonical_json_dumps

from .bundle import MRI_GUARD_BUNDLE_EXTENSION, verify_bundle
from .demo import run_demo
from .harness import MULTI_SERIES_CASE_KIND
from .profiles import DEFAULT_PROFILE_NAME


def build_cli_parser() -> argparse.ArgumentParser:
    """Return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="mri-guardian",
        description="One-command demo, verify, and test entrypoints for the MRI Guardian Layer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Generate deterministic demo artifacts.",
    )
    demo_parser.add_argument(
        "--output-dir",
        default="demo_artifacts",
        help="Directory where demo artifacts will be written.",
    )
    demo_parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        help="Guardian runtime profile to use for the demo artifacts.",
    )
    demo_parser.add_argument(
        "--tamper-mode",
        action="store_true",
        help="Include deterministic tamper-mode artifacts.",
    )
    demo_parser.add_argument(
        "--case-kind",
        choices=("single_series_v1", MULTI_SERIES_CASE_KIND),
        default="single_series_v1",
        help="Generate either the legacy single-series demo or the additive multi-series v2 demo.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify evidence and bundle artifacts from a file or output directory.",
    )
    verify_parser.add_argument(
        "target",
        nargs="?",
        default="demo_artifacts",
        help="Directory, .mri-guard-bundle archive, bundle.json, or evidence.json to verify.",
    )
    verify_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Render verification output as text or JSON.",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Run the guardian unit test suite.",
    )
    test_parser.add_argument(
        "--start-dir",
        default="tests",
        help="Test discovery start directory.",
    )
    test_parser.add_argument(
        "--pattern",
        default="test*.py",
        help="unittest discovery pattern.",
    )
    test_parser.add_argument(
        "--top-level-dir",
        default=None,
        help="Optional top-level directory for unittest discovery.",
    )
    test_parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="unittest runner verbosity.",
    )
    return parser


def run_verify_command(target: Path | str) -> dict[str, Any]:
    """Verify evidence and/or bundle artifacts from a target path."""
    path = Path(target)
    if not path.exists():
        raise ValueError(f"Verification target not found: {path}")

    parts: dict[str, Any] = {}
    if path.is_dir():
        evidence_path = path / "evidence.json"
        if evidence_path.exists():
            parts["evidence"] = _verify_evidence_file(evidence_path)
        bundle_candidates = sorted(path.glob(f"*{MRI_GUARD_BUNDLE_EXTENSION}"))
        if len(bundle_candidates) > 1:
            raise ValueError(
                "Verification target directory contains multiple .mri-guard-bundle files; "
                "pass one bundle path explicitly."
            )
        if bundle_candidates:
            parts["bundle"] = verify_bundle(bundle_candidates[0])
    elif path.suffix.lower() == MRI_GUARD_BUNDLE_EXTENSION:
        parts["bundle"] = verify_bundle(path)
    elif path.name == "bundle.json":
        parts["bundle"] = verify_bundle(path)
    elif path.suffix.lower() == ".json":
        parts["evidence"] = _verify_evidence_file(path)
    else:
        raise ValueError(
            "Unsupported verification target. Use a directory, evidence.json, bundle.json, "
            "or a .mri-guard-bundle archive."
        )

    if not parts:
        raise ValueError(f"No verifiable artifacts were found under {path}.")
    verified = all(_part_verified(payload) for payload in parts.values())
    failures: list[str] = []
    for payload in parts.values():
        failures.extend(_part_failures(payload))
    return {
        "target": str(path),
        "verified": verified,
        "parts": parts,
        "failures": failures,
    }


def run_test_command(
    *,
    start_dir: str = "tests",
    pattern: str = "test*.py",
    top_level_dir: str | None = None,
    verbosity: int = 2,
) -> dict[str, Any]:
    """Run unittest discovery and return a structured result."""
    suite = unittest.defaultTestLoader.discover(
        start_dir=start_dir,
        pattern=pattern,
        top_level_dir=top_level_dir,
    )
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return {
        "start_dir": start_dir,
        "pattern": pattern,
        "top_level_dir": top_level_dir,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo(
            output_dir=args.output_dir,
            profile_name=args.profile,
            tamper_mode=args.tamper_mode,
            case_kind=args.case_kind,
        )
        print(f"Demo artifacts written to {Path(args.output_dir)}")
        return 0
    if args.command == "verify":
        result = run_verify_command(args.target)
        if args.format == "json":
            print(canonical_json_dumps(result))
        else:
            print(_format_verify_text(result))
        return 0 if result["verified"] else 1
    if args.command == "test":
        result = run_test_command(
            start_dir=args.start_dir,
            pattern=args.pattern,
            top_level_dir=args.top_level_dir,
            verbosity=args.verbosity,
        )
        return 0 if result["successful"] else 1
    parser.error(f"Unknown command: {args.command}")
    return 2


def _verify_evidence_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = list(validate_evidence_payload(payload))
    return {
        "path": str(path),
        "verified": not failures,
        "failures": failures,
    }


def _part_verified(payload: dict[str, Any]) -> bool:
    return bool(payload.get("verified"))


def _part_failures(payload: dict[str, Any]) -> list[str]:
    failures = payload.get("failures", [])
    return [str(item) for item in failures]


def _format_verify_text(result: dict[str, Any]) -> str:
    lines = [f"Target: {result['target']}"]
    for name in ("evidence", "bundle"):
        payload = result["parts"].get(name)
        if payload is None:
            continue
        status = "OK" if payload.get("verified") else "FAILED"
        path = payload.get("path") or payload.get("bundle_path") or result["target"]
        lines.append(f"{name.title()}: {status} ({path})")
        for failure in _part_failures(payload):
            lines.append(f"  - {failure}")
    overall = "VERIFIED" if result["verified"] else "FAILED"
    lines.append(f"Overall: {overall}")
    return "\n".join(lines)
