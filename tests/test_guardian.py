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
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
)
from mri_guardian.tamper import build_tamper_scenarios
from mri_guardian.viewer import build_side_by_side_view_model, render_side_by_side_report_html


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
        self.assertTrue(result.after_state["verified_manifest"]["mask_verification"]["verified"])
        self.assertEqual(
            result.after_state["workstation_payload"]["pixel_source"],
            "harness_reconstructed_from_raw_kspace",
        )
        self.assertIn("verified_mask", result.after_state["workstation_payload"])

    def test_rejects_ai_mask_with_fake_tumour_voxel(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1
        harness = self._build_harness()

        with self.assertRaises(InvariantViolationError):
            harness.release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
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

    def test_view_model_reports_no_visual_findings_for_clean_case(self) -> None:
        state = build_demo_case()

        view_model = build_side_by_side_view_model(state)

        self.assertTrue(view_model["release_eligible"])
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED")
        self.assertEqual(view_model["summary"]["mask_voxel_delta"], 0)
        self.assertEqual(view_model["summary"]["rejected_region_count"], 0)
        self.assertEqual(view_model["summary"]["intensity_mismatch_voxel_count"], 0)
        self.assertEqual(view_model["differences"]["rows"], [])

    def test_view_model_flags_rejected_mask_region_for_fake_tumour_voxel(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1

        view_model = build_side_by_side_view_model(state)

        self.assertFalse(view_model["release_eligible"])
        self.assertEqual(view_model["verification"]["headline"], "Verification FAILED")
        self.assertEqual(
            view_model["verification"]["primary_reason_summary"],
            "mask_voxel_mismatch (delta = 1 voxels)",
        )
        self.assertEqual(view_model["summary"]["mask_voxel_delta"], 1)
        self.assertEqual(view_model["summary"]["rejected_region_count"], 1)
        self.assertEqual(view_model["summary"]["intensity_mismatch_voxel_count"], 0)
        self.assertEqual(view_model["differences"]["rows"][0]["slice_index"], 0)
        self.assertEqual(view_model["differences"]["rows"][0]["row_index"], 0)
        self.assertEqual(view_model["differences"]["rows"][0]["column_index"], 0)
        self.assertEqual(view_model["differences"]["rows"][0]["code"], "mask_false_positive")
        self.assertEqual(view_model["differences"]["rows"][0]["status_label"], "Rejected Region")
        self.assertEqual(view_model["slices"][0]["verified_mask"][0][0], 0)

    def test_html_report_contains_side_by_side_labels_and_difference_register(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1

        html = render_side_by_side_report_html(
            state,
            report_title="Tamper Mode: Fake Tumour Injection",
        )

        self.assertIn("AI Reconstruction + Tumour Mask", html)
        self.assertIn("Deterministic Reconstruction + Verified Mask", html)
        self.assertIn("Difference Register", html)
        self.assertIn("Rejected Region", html)
        self.assertIn("Verification FAILED", html)
        self.assertIn("mask_voxel_mismatch", html)

    def test_tamper_mode_scenarios_surface_expected_primary_failures(self) -> None:
        scenarios = {scenario["slug"]: scenario for scenario in build_tamper_scenarios()}

        fake_view = build_side_by_side_view_model(scenarios["fake_tumour"]["state"])
        shift_view = build_side_by_side_view_model(scenarios["shifted_mask"]["state"])
        intensity_view = build_side_by_side_view_model(
            scenarios["intensity_manipulation"]["state"]
        )

        self.assertEqual(
            fake_view["verification"]["primary_reason_summary"],
            "mask_voxel_mismatch (delta = 342 voxels)",
        )
        self.assertEqual(fake_view["summary"]["mask_voxel_delta"], 342)
        self.assertEqual(shift_view["summary"]["mask_voxel_delta"], 46)
        self.assertTrue(
            shift_view["verification"]["primary_reason_summary"].startswith(
                "mask_voxel_mismatch"
            )
        )
        self.assertEqual(
            intensity_view["verification"]["primary_reason_summary"],
            "intensity_manipulation_detected (delta = 147 voxels; max_abs_error = 0.55)",
        )
        self.assertEqual(intensity_view["summary"]["intensity_mismatch_voxel_count"], 147)

    def test_verification_report_exposes_mask_and_intensity_failure_codes(self) -> None:
        scenarios = {scenario["slug"]: scenario for scenario in build_tamper_scenarios()}

        fake_report = build_case_verification_report(scenarios["fake_tumour"]["state"])
        intensity_report = build_case_verification_report(
            scenarios["intensity_manipulation"]["state"]
        )

        self.assertFalse(fake_report["release_eligible"])
        self.assertEqual(fake_report["failure_reasons"][0]["code"], "mask_voxel_mismatch")
        self.assertEqual(fake_report["failure_reasons"][0]["delta"], 342)
        self.assertFalse(intensity_report["release_eligible"])
        self.assertEqual(
            intensity_report["failure_reasons"][0]["code"],
            "intensity_manipulation_detected",
        )
        self.assertEqual(intensity_report["failure_reasons"][0]["delta"], 147)

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
