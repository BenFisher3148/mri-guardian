from __future__ import annotations

import json
import shutil
import unittest
from zipfile import ZipFile, ZipInfo
from pathlib import Path
from uuid import uuid4

import numpy as np

from mri_guardian.bundle import (
    MRI_GUARD_BUNDLE_EXTENSION,
    verify_bundle,
    write_portable_evidence_bundle,
)
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json
from icsd.core.errors import InvariantViolationError
from icsd.core.evidence import validate_evidence_payload

from mri_guardian.harness import (
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
)
from mri_guardian.integration import (
    build_mock_dicom_input,
    build_mock_integration_trace,
    build_mock_pacs_block_notice,
    build_mock_pacs_submission,
    restore_case_state_from_dicom_input,
)
from mri_guardian.tamper import build_tamper_scenarios
from mri_guardian.timing import build_timing_overlay_snapshot
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
        self.assertIn("ai_claim_envelope", result.after_state["verified_manifest"])
        self.assertIn("verified_reconstruction_trace", result.after_state["verified_manifest"])
        self.assertIn("mask_derivation_policy", result.after_state["verified_manifest"])
        self.assertIn("verification_tolerance_policy", result.after_state["verified_manifest"])
        self.assertEqual(
            result.after_state["verified_manifest"]["verified_reconstruction_trace"][
                "claim_envelope_hash"
            ],
            hash_canonical_json(result.after_state["verified_manifest"]["ai_claim_envelope"]),
        )
        self.assertEqual(
            result.after_state["verified_manifest"]["verified_reconstruction_trace"][
                "tolerance_policy_hash"
            ],
            hash_canonical_json(
                result.after_state["verified_manifest"]["verification_tolerance_policy"]
            ),
        )
        self.assertEqual(
            result.after_state["verified_manifest"]["ai_reconstruction_match"]["metrics"][
                "per_voxel_abs_error"
            ]["scope"],
            "per_voxel",
        )
        self.assertEqual(
            result.after_state["verified_manifest"]["measurement_consistency"]["metrics"][
                "global_relative_l2_error"
            ]["scope"],
            "global_norm",
        )
        self.assertEqual(
            result.after_state["workstation_payload"]["pixel_source"],
            "harness_reconstructed_from_raw_kspace",
        )
        self.assertIn("verified_mask", result.after_state["workstation_payload"])
        self.assertIn("verification_tiers", result.after_state["verified_manifest"])
        self.assertEqual(
            result.after_state["verified_manifest"]["verification_tiers"]["level_2"]["status"],
            "passed",
        )

    def test_portable_bundle_round_trips_and_replays_identically(self) -> None:
        state = build_demo_case()
        harness = self._build_harness()
        result = harness.release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        bundle_path = self.runtime_root / f"{state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        with ZipFile(bundle_path, mode="r") as archive:
            manifest = self._read_bundle_json(archive, "bundle.json")
            transcript = self._read_bundle_json(archive, "transcript.json")
            signatures = self._read_bundle_json(archive, "signatures.json")
            replay = self._read_bundle_json(archive, "replay.json")

        self.assertEqual(manifest["verdict"], "This scan is verifiably correct.")
        self.assertEqual(
            manifest["hashes"]["raw_kspace_hash"],
            result.after_state["verified_manifest"]["raw_hash"],
        )
        self.assertEqual(
            manifest["hashes"]["ai_claim_envelope_hash"],
            hash_canonical_json(result.after_state["verified_manifest"]["ai_claim_envelope"]),
        )
        self.assertEqual(
            manifest["hashes"]["verified_reconstruction_trace_hash"],
            hash_canonical_json(
                result.after_state["verified_manifest"]["verified_reconstruction_trace"]
            ),
        )
        self.assertEqual(
            manifest["hashes"]["verification_tolerance_policy_hash"],
            hash_canonical_json(
                result.after_state["verified_manifest"]["verification_tolerance_policy"]
            ),
        )
        self.assertEqual(
            manifest["hashes"]["reconstruction_hash"],
            result.after_state["workstation_payload"]["reconstruction_hash"],
        )
        self.assertEqual(
            manifest["hashes"]["verified_mask_hash"],
            result.after_state["workstation_payload"]["verified_mask_hash"],
        )
        self.assertEqual(transcript["statement"], "This scan is verifiably correct.")
        self.assertEqual(
            replay["expected_hashes"]["released_state_hash"],
            manifest["hashes"]["released_state_hash"],
        )
        self.assertEqual(
            signatures["members"][0]["algorithm"],
            "sha256",
        )

    def test_verify_bundle_accepts_extracted_bundle_json(self) -> None:
        state = build_demo_case()
        harness = self._build_harness()
        result = harness.release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        bundle_path = self.runtime_root / f"{state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
        extracted_root = self.runtime_root / "extracted_bundle"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(extracted_root)

        verification = verify_bundle(extracted_root / "bundle.json")

        self.assertTrue(verification["verified"])
        self.assertEqual(verification["details"]["source_kind"], "directory")

    def test_portable_bundle_detects_transcript_tampering(self) -> None:
        state = build_demo_case()
        harness = self._build_harness()
        result = harness.release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        bundle_path = self.runtime_root / f"{state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"{state['case_id']}-tampered{MRI_GUARD_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            transcript = self._read_bundle_json(archive, "transcript.json")
        transcript["statement"] = "This scan is verifiably correct, trust me."
        self._rewrite_bundle_member(
            source=bundle_path,
            destination=tampered_path,
            member_name="transcript.json",
            payload=self._json_bytes(transcript),
        )

        verification = verify_bundle(tampered_path)

        self.assertFalse(verification["verified"])
        self.assertTrue(
            any("transcript.json" in failure or "signatures.json" in failure for failure in verification["failures"])
        )

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

    def test_rejects_distributed_low_amplitude_attack_by_global_l2_bound(self) -> None:
        state = build_demo_case()
        reconstruction = np.asarray(state["ai_reconstruction"], dtype=np.float64)
        reconstruction += 9e-10
        state["ai_reconstruction"] = reconstruction.tolist()

        report = build_case_verification_report(state)
        harness = self._build_harness()

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["failure_reasons"][0]["code"], "intensity_manipulation_detected")
        self.assertEqual(report["analysis"]["reconstruction_mismatch_voxel_count"], 0)
        self.assertLess(report["analysis"]["max_ai_abs_error"], 1e-9)
        self.assertGreater(report["analysis"]["relative_ai_l2_error"], 1e-12)
        self.assertEqual(
            report["manifest"]["ai_reconstruction_match"]["metrics"]["per_voxel_abs_error"][
                "boundary_status"
            ],
            "near_boundary",
        )
        self.assertEqual(
            report["manifest"]["ai_reconstruction_match"]["metrics"]["global_relative_l2_error"][
                "boundary_status"
            ],
            "exceeded",
        )
        with self.assertRaises(InvariantViolationError):
            harness.release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_mock_dicom_input_round_trips_guardian_case_state(self) -> None:
        state = build_demo_case()

        dicom_input = build_mock_dicom_input(state)
        restored_state = restore_case_state_from_dicom_input(dicom_input)

        self.assertEqual(restored_state, state)
        self.assertEqual(dicom_input["endpoint"]["destination_ae_title"], "MRI-GUARDIAN-LAYER")
        self.assertEqual(dicom_input["objects"][0]["role"], "raw_kspace")
        self.assertEqual(dicom_input["ai_claim_envelope"]["interface_type"], "AIClaimEnvelope")

    def test_mock_pacs_submission_exposes_verified_handoff_contract(self) -> None:
        state = build_demo_case()
        harness = self._build_harness()
        result = harness.release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )

        dicom_input = build_mock_dicom_input(state)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference="mri-case-0001.mri-guard-bundle",
        )
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_submission,
        )

        self.assertEqual(pacs_submission["submission_status"], "released")
        self.assertEqual(
            pacs_submission["verification_summary"]["guardian_mode"],
            "verification_proxy",
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["bundle_reference"],
            "mri-case-0001.mri-guard-bundle",
        )
        self.assertEqual(pacs_submission["series"][0]["role"], "verified_reconstruction")
        self.assertEqual(
            pacs_submission["proof_contracts"]["verified_reconstruction_trace"][
                "interface_type"
            ],
            "VerifiedReconstructionTrace",
        )
        self.assertEqual(integration_trace["pipeline"][0]["label"], "DICOM Input")
        self.assertEqual(integration_trace["pipeline"][2]["label"], "PACS Output")

    def test_mock_pacs_block_notice_withholds_tampered_case(self) -> None:
        state = build_tamper_scenarios()[0]["state"]

        dicom_input = build_mock_dicom_input(state)
        pacs_block = build_mock_pacs_block_notice(state)
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_block,
        )

        self.assertEqual(pacs_block["submission_status"], "blocked")
        self.assertEqual(
            pacs_block["blocked_reason"],
            "mask_voxel_mismatch (delta = 342 voxels)",
        )
        self.assertEqual(pacs_block["withheld_series"][0]["role"], "verified_reconstruction")
        self.assertEqual(
            pacs_block["proof_contracts"]["mask_derivation_policy"]["interface_type"],
            "MaskDerivationPolicy",
        )
        self.assertEqual(
            pacs_block["proof_contracts"]["verification_tolerance_policy"][
                "interface_type"
            ],
            "VerificationTolerancePolicy",
        )
        self.assertEqual(integration_trace["pipeline"][1]["decision"], "blocked")
        self.assertEqual(integration_trace["pipeline"][2]["label"], "PACS Output Blocked")

    def test_view_model_reports_no_visual_findings_for_clean_case(self) -> None:
        state = build_demo_case()

        view_model = build_side_by_side_view_model(state)

        self.assertTrue(view_model["release_eligible"])
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED")
        self.assertEqual(
            [tier["status"] for tier in view_model["verification"]["tiers"]],
            ["passed", "passed", "passed"],
        )
        self.assertGreater(view_model["timing"]["reconstruction_ms"], 0)
        self.assertGreaterEqual(view_model["timing"]["verification_ms"], 0)
        self.assertGreaterEqual(view_model["timing"]["overhead_percent"], 0)
        self.assertEqual(view_model["summary"]["mask_voxel_delta"], 0)
        self.assertEqual(view_model["summary"]["rejected_region_count"], 0)
        self.assertEqual(view_model["summary"]["intensity_mismatch_voxel_count"], 0)
        self.assertEqual(view_model["clinical_summary"]["status_label"], "VERIFIED")
        self.assertEqual(
            view_model["clinical_summary"]["confidence_label"],
            "Derived from raw data (not AI)",
        )
        self.assertIn("...", view_model["clinical_summary"]["audit_id"])
        self.assertEqual(view_model["integration"]["ingress_label"], "Mock DICOM Input")
        self.assertEqual(view_model["integration"]["egress_label"], "Mock PACS Output")
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
        self.assertEqual(view_model["clinical_summary"]["status_label"], "NOT VERIFIED")
        self.assertEqual(
            view_model["clinical_summary"]["confidence_label"],
            "Not released from AI claim",
        )
        self.assertEqual(view_model["integration"]["egress_label"], "PACS Output Blocked")
        self.assertEqual(view_model["slices"][0]["verified_mask"][0][0], 0)

    def test_html_report_contains_side_by_side_labels_and_difference_register(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1

        html = render_side_by_side_report_html(
            state,
            report_title="Tamper Mode: Fake Tumour Injection",
        )

        self.assertIn("MRI Guardian Layer", html)
        self.assertIn("Status", html)
        self.assertIn("Tumour Volume", html)
        self.assertIn("Confidence", html)
        self.assertIn("Audit ID", html)
        self.assertIn("Mock DICOM Input", html)
        self.assertIn("AI Reconstruction + Proposed Tumour Mask", html)
        self.assertIn("Guardian Layer Reconstruction + Verified Mask", html)
        self.assertIn("PACS Output Blocked", html)
        self.assertIn("Difference Register", html)
        self.assertIn("Rejected Region", html)
        self.assertIn("Verification FAILED", html)
        self.assertIn("mask_voxel_mismatch", html)
        self.assertIn("verification layer", html)
        self.assertIn("Level 1: Structural Validity", html)
        self.assertIn("Level 2: Physical Consistency", html)
        self.assertIn("Level 3: Clinical Plausibility", html)
        self.assertIn("Reconstruction", html)
        self.assertIn("Verification", html)
        self.assertIn("Total Overhead", html)

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
        self.assertEqual(fake_view["verification"]["tiers"][0]["status"], "passed")
        self.assertEqual(fake_view["verification"]["tiers"][1]["status"], "failed")
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
        self.assertEqual(intensity_view["verification"]["tiers"][0]["status"], "passed")
        self.assertEqual(intensity_view["verification"]["tiers"][1]["status"], "failed")
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
        self.assertEqual(fake_report["contracts"]["ai_claim_envelope"]["interface_type"], "AIClaimEnvelope")
        self.assertEqual(
            fake_report["contracts"]["verified_reconstruction_trace"]["interface_type"],
            "VerifiedReconstructionTrace",
        )
        self.assertEqual(fake_report["verification_tiers"]["level_1"]["status"], "passed")
        self.assertEqual(fake_report["verification_tiers"]["level_2"]["status"], "failed")
        self.assertEqual(fake_report["verification_tiers"]["level_3"]["status"], "passed")
        self.assertFalse(intensity_report["release_eligible"])
        self.assertEqual(
            intensity_report["failure_reasons"][0]["code"],
            "intensity_manipulation_detected",
        )
        self.assertEqual(intensity_report["failure_reasons"][0]["delta"], 147)
        self.assertEqual(intensity_report["verification_tiers"]["level_1"]["status"], "passed")
        self.assertEqual(intensity_report["verification_tiers"]["level_2"]["status"], "failed")

    def test_clinical_plausibility_warning_is_advisory_only(self) -> None:
        state = self._build_advisory_warning_case()

        report = build_case_verification_report(state)
        view_model = build_side_by_side_view_model(state)

        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["verification_tiers"]["level_1"]["status"], "passed")
        self.assertEqual(report["verification_tiers"]["level_2"]["status"], "passed")
        self.assertEqual(report["verification_tiers"]["level_3"]["status"], "warning")
        self.assertEqual(
            report["manifest"]["clinical_plausibility"]["warning_count"],
            1,
        )
        self.assertTrue(view_model["release_eligible"])
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED")
        self.assertEqual(
            view_model["verification"]["primary_reason_summary"],
            "all_blocking_checks_passed; clinical_plausibility_review_recommended",
        )
        self.assertEqual(view_model["verification"]["tiers"][2]["status_label"], "ADVISORY WARN")

    def test_timing_overlay_snapshot_is_positive_and_has_probe_metadata(self) -> None:
        snapshot = build_timing_overlay_snapshot()

        self.assertGreater(snapshot["reconstruction_ms"], 0)
        self.assertGreater(snapshot["verification_ms"], 0)
        self.assertGreater(snapshot["total_pipeline_ms"], 0)
        self.assertGreaterEqual(snapshot["overhead_percent"], 0)
        self.assertEqual(snapshot["sample_runs"], 7)
        self.assertEqual(snapshot["warmup_runs"], 2)
        self.assertEqual(len(snapshot["probe_shape"]), 3)
        self.assertIn("NumPy FFT core", snapshot["backend"])

    def _build_harness(self) -> MriGuardianHarness:
        return MriGuardianHarness(
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
        )

    def _read_bundle_json(self, archive: ZipFile, member_name: str) -> dict[str, object]:
        return json.loads(archive.read(member_name).decode("utf-8"))

    def _rewrite_bundle_member(
        self,
        *,
        source: Path,
        destination: Path,
        member_name: str,
        payload: bytes,
    ) -> None:
        with ZipFile(source, mode="r") as src, ZipFile(destination, mode="w") as dst:
            for info in src.infolist():
                member_payload = payload if info.filename == member_name else src.read(info.filename)
                clone = ZipInfo(filename=info.filename, date_time=info.date_time)
                clone.compress_type = info.compress_type
                clone.create_system = info.create_system
                clone.external_attr = info.external_attr
                dst.writestr(clone, member_payload)

    def _json_bytes(self, payload: dict[str, object]) -> bytes:
        return f"{canonical_json_dumps(payload)}\n".encode("utf-8")

    def _build_advisory_warning_case(self) -> dict[str, object]:
        state = build_demo_case()
        reconstruction = np.asarray(
            [
                [[3.0, 3.0], [3.0, 3.0]],
                [[3.0, 3.0], [3.0, 0.0]],
            ],
            dtype=np.float64,
        )
        mask = (reconstruction >= 2.5).astype(np.int64)
        raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
        voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
        state["raw_real"] = raw_kspace.real.tolist()
        state["raw_imag"] = raw_kspace.imag.tolist()
        state["ai_reconstruction"] = reconstruction.tolist()
        state["tumor_mask"] = mask.tolist()
        state["tumor_volume_mm3"] = round(float(mask.sum() * np.prod(voxel_spacing)), 9)
        return state

    def _cleanup_runtime_root(self) -> None:
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        parent = self.runtime_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    unittest.main()
