from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from icsd.core.errors import InvariantViolationError
from icsd.core.evidence import validate_evidence_payload

from mri_guardian.harness import (
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    build_demo_authority_sources,
    build_demo_case,
)


class MriGuardianHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_root = Path("test_runs") / uuid4().hex
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(self._cleanup_runtime_root)

    def test_demo_case_releases_and_generates_verifiable_evidence(self) -> None:
        state = build_demo_case()
        harness = self._build_harness()
        result = harness.release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )

        self.assertEqual(result.after_state["release_status"], RELEASE_STATUS_RELEASED)
        self.assertEqual(validate_evidence_payload(result.evidence.as_dict()), ())
        self.assertTrue(harness.verify_evidence(result.evidence))
        self.assertTrue(
            result.evidence.verify_integrity(
                before_state=result.before_state,
                after_state=result.after_state,
            )
        )
        self.assertEqual(
            result.after_state["workstation_payload"]["pixel_source"],
            "harness_reconstructed_from_raw_kspace",
        )

    def test_rejects_ai_reconstruction_with_invented_pixels(self) -> None:
        state = build_demo_case()
        state["ai_reconstruction"][0][0][0] = 0.5
        harness = self._build_harness()

        with self.assertRaises(InvariantViolationError):
            harness.release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_rejects_incorrect_tumor_volume_claim(self) -> None:
        state = build_demo_case()
        state["tumor_volume_mm3"] = 9.9
        harness = self._build_harness()

        with self.assertRaises(InvariantViolationError):
            harness.release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def _build_harness(self) -> MriGuardianHarness:
        return MriGuardianHarness(
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
        )

    def _cleanup_runtime_root(self) -> None:
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        parent = self.runtime_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    unittest.main()
