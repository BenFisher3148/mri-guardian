from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZipFile, ZipInfo

import numpy as np

from icsd.core.errors import InvariantViolationError
from icsd.core.evidence import validate_evidence_payload
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json

import mri_guardian.profiles as profile_registry_module
from mri_guardian.bundle import (
    MRI_GUARD_BUNDLE_EXTENSION,
    verify_bundle,
    write_portable_evidence_bundle,
)
from mri_guardian.cli import run_test_command, run_verify_command
from mri_guardian.demo import run_demo
from mri_guardian.harness import (
    MRI_GUARDIAN_POLICY_NAME,
    MRI_GUARDIAN_POLICY_VERSION,
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
)
from mri_guardian.integration import (
    build_guardian_integration_narrative,
    build_mock_dicom_input,
    build_mock_integration_trace,
    build_mock_pacs_block_notice,
    build_mock_pacs_submission,
    restore_case_state_from_dicom_input,
)
from mri_guardian.mask_engine import (
    CONNECTED_COMPONENT_ALGORITHM,
    MORPHOLOGY_ALGORITHM,
    SLICE_CONSISTENCY_ALGORITHM,
    THRESHOLD_ALGORITHM,
    derive_verified_mask,
    list_approved_mask_algorithms,
    resolve_mask_policy,
)
from mri_guardian.profiles import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_VERIFICATION_PROFILE_NAME,
    VerificationProfileSpec,
    build_verification_profile_descriptor,
    canonicalise_verification_profile_descriptor,
    get_verification_profile,
    list_verification_profiles,
    verification_profile_descriptor_from_dict,
)
from mri_guardian.tamper import build_tamper_scenarios
from mri_guardian.timing import (
    build_guardian_performance_profile,
    build_timing_overlay_snapshot,
)
from mri_guardian.viewer import (
    build_side_by_side_view_model,
    render_side_by_side_report_html,
)


def _reconstruction_with_hot_voxels(
    shape: tuple[int, int, int],
    hot_voxels: tuple[tuple[int, int, int], ...],
    *,
    hot_value: float = 4.0,
) -> np.ndarray:
    reconstruction = np.zeros(shape, dtype=np.float64)
    for z_index, y_index, x_index in hot_voxels:
        reconstruction[z_index, y_index, x_index] = hot_value
    return reconstruction


class MriGuardianHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_root = Path("test_runs") / uuid4().hex
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(self._cleanup_runtime_root)

    def test_profile_registry_lists_expected_profiles_and_compat_aliases(self) -> None:
        self.assertEqual(
            list_verification_profiles(),
            (
                "brain_t1_baseline",
                "brain_t1_morphology_review",
                "brain_t1_component_review",
                "brain_t1_slice_review",
                "brain_t2_baseline",
                "contrast_enhanced_review",
                "strict_research_mode",
                "fast_triage_mode",
            ),
        )
        self.assertEqual(DEFAULT_PROFILE_NAME, "brain_t1_baseline")
        self.assertEqual(DEFAULT_VERIFICATION_PROFILE_NAME, DEFAULT_PROFILE_NAME)
        self.assertEqual(MRI_GUARDIAN_POLICY_NAME, DEFAULT_PROFILE_NAME)
        self.assertEqual(
            MRI_GUARDIAN_POLICY_VERSION,
            get_verification_profile(DEFAULT_PROFILE_NAME).icsd_policy_version,
        )
        descriptor = get_verification_profile("fast_triage_mode")
        self.assertIsInstance(descriptor, VerificationProfileSpec)
        self.assertFalse(descriptor.release_semantics["final_release_allowed"])
        self.assertFalse(descriptor.release_semantics["portable_bundle_allowed"])

    def test_seeded_profiles_expose_expected_thresholds_authorities_and_budgets(self) -> None:
        expected = {
            "brain_t1_baseline": {
                "algorithm": THRESHOLD_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 80.0,
                "worst_case_budget": 180.0,
            },
            "brain_t1_morphology_review": {
                "algorithm": MORPHOLOGY_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 90.0,
                "worst_case_budget": 190.0,
            },
            "brain_t1_component_review": {
                "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 95.0,
                "worst_case_budget": 200.0,
            },
            "brain_t1_slice_review": {
                "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 100.0,
                "worst_case_budget": 210.0,
            },
            "brain_t2_baseline": {
                "algorithm": THRESHOLD_ALGORITHM,
                "threshold": 1.75,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 80.0,
                "worst_case_budget": 180.0,
            },
            "contrast_enhanced_review": {
                "algorithm": THRESHOLD_ALGORITHM,
                "threshold": 3.0,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                    "contrast_protocol",
                ),
                "mask_severity": "warning",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 90.0,
                "worst_case_budget": 190.0,
            },
            "strict_research_mode": {
                "algorithm": THRESHOLD_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                    "sequence_manifest",
                    "research_protocol",
                ),
                "mask_severity": "blocking",
                "contrast_enabled": True,
                "slice_enabled": True,
                "representative_budget": 95.0,
                "worst_case_budget": 200.0,
            },
            "fast_triage_mode": {
                "algorithm": THRESHOLD_ALGORITHM,
                "threshold": 2.5,
                "authorities": (
                    "scanner_manifest",
                    "voxel_geometry",
                    "model_manifest",
                ),
                "mask_severity": "advisory",
                "contrast_enabled": False,
                "slice_enabled": False,
                "representative_budget": 60.0,
                "worst_case_budget": 140.0,
            },
        }

        for profile_name, spec in expected.items():
            with self.subTest(profile_name=profile_name):
                descriptor = get_verification_profile(profile_name)
                self.assertEqual(descriptor.mask_algorithm, spec["algorithm"])
                self.assertEqual(descriptor.mask_threshold, spec["threshold"])
                self.assertEqual(
                    descriptor.required_authority_source_types,
                    spec["authorities"],
                )
                self.assertEqual(
                    descriptor.heuristic("mask_fraction_guardrail")["severity"],
                    spec["mask_severity"],
                )
                self.assertEqual(
                    bool(descriptor.heuristic("inside_outside_contrast")["enabled"]),
                    spec["contrast_enabled"],
                )
                self.assertEqual(
                    bool(descriptor.heuristic("slice_span")["enabled"]),
                    spec["slice_enabled"],
                )
                self.assertEqual(
                    descriptor.performance_expectations["representative_total_ms_budget"],
                    spec["representative_budget"],
                )
                self.assertEqual(
                    descriptor.performance_expectations["worst_case_total_ms_budget"],
                    spec["worst_case_budget"],
                )

    def test_approved_mask_algorithm_catalogue_lists_all_approved_engines(self) -> None:
        self.assertEqual(
            list_approved_mask_algorithms(),
            (
                THRESHOLD_ALGORITHM,
                MORPHOLOGY_ALGORITHM,
                CONNECTED_COMPONENT_ALGORITHM,
                SLICE_CONSISTENCY_ALGORITHM,
            ),
        )

    def test_unknown_profile_lookup_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Guardian runtime profile"):
            get_verification_profile("unknown_profile")

    def test_profile_descriptor_round_trip_has_stable_digest(self) -> None:
        descriptor_payload = build_verification_profile_descriptor("brain_t1_baseline")
        canonical = canonicalise_verification_profile_descriptor(descriptor_payload)
        reloaded_payload = json.loads(canonical)
        hydrated = verification_profile_descriptor_from_dict(reloaded_payload)

        self.assertEqual(reloaded_payload, descriptor_payload)
        self.assertEqual(reloaded_payload["profile_revision_scope"], "profile_name")
        self.assertEqual(reloaded_payload["profile_digest"], hydrated.profile_digest())
        self.assertEqual(
            canonicalise_verification_profile_descriptor(reloaded_payload),
            canonical,
        )

    def test_profile_descriptor_equivalence_and_non_equivalence(self) -> None:
        baseline = verification_profile_descriptor_from_dict(
            build_verification_profile_descriptor("brain_t1_baseline")
        )
        same_baseline = verification_profile_descriptor_from_dict(
            build_verification_profile_descriptor("brain_t1_baseline")
        )
        t2_profile = verification_profile_descriptor_from_dict(
            build_verification_profile_descriptor("brain_t2_baseline")
        )
        changed_threshold = replace(
            baseline,
            mask_rules={
                **baseline.mask_rules,
                "parameters": {
                    **baseline.mask_rules["parameters"],
                    "threshold": 9.9,
                },
            },
        )

        self.assertEqual(baseline.profile_digest(), same_baseline.profile_digest())
        self.assertNotEqual(baseline.profile_digest(), t2_profile.profile_digest())
        self.assertNotEqual(baseline.profile_digest(), changed_threshold.profile_digest())

    def test_profile_descriptor_digest_mismatch_is_rejected(self) -> None:
        descriptor_payload = build_verification_profile_descriptor("brain_t1_baseline")
        descriptor_payload["mask_rules"]["parameters"]["threshold"] = 9.9

        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verification_profile_descriptor_from_dict(descriptor_payload)

    def test_mask_policy_resolution_rejects_unknown_or_inapplicable_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "not applicable"):
            resolve_mask_policy(
                {
                    "algorithm": THRESHOLD_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "opening_radius_voxels": 1,
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "unsupported top-level keys"):
            resolve_mask_policy(
                {
                    "algorithm": THRESHOLD_ALGORITHM,
                    "parameters": {"threshold": 2.5},
                    "unexpected": "value",
                }
            )

    def test_mask_policy_resolution_normalises_defaults_stably(self) -> None:
        resolved_default = resolve_mask_policy(
            {
                "algorithm": MORPHOLOGY_ALGORITHM,
                "parameters": {"threshold": 2.5},
            }
        )
        resolved_explicit = resolve_mask_policy(
            {
                "algorithm": MORPHOLOGY_ALGORITHM,
                "parameters": {
                    "comparison": "GREATER_THAN_OR_EQUAL",
                    "threshold": 2.5,
                    "opening_radius_voxels": 1,
                    "closing_radius_voxels": 1,
                },
            }
        )
        self.assertEqual(resolved_default.to_dict(), resolved_explicit.to_dict())
        self.assertEqual(
            list(resolved_default.parameters),
            [
                "comparison",
                "threshold",
                "opening_radius_voxels",
                "closing_radius_voxels",
            ],
        )
        morphology_one = build_verification_profile_descriptor("brain_t1_morphology_review")
        morphology_two = build_verification_profile_descriptor("brain_t1_morphology_review")
        self.assertEqual(morphology_one["profile_digest"], morphology_two["profile_digest"])

        component_default = resolve_mask_policy(
            {
                "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                "parameters": {
                    "threshold": 2.5,
                    "min_component_voxels": 8,
                },
            }
        )
        component_explicit = resolve_mask_policy(
            {
                "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                "parameters": {
                    "comparison": "GREATER_THAN_OR_EQUAL",
                    "threshold": 2.5,
                    "connectivity": "FACES",
                    "min_component_voxels": 8,
                    "selection_rule": "KEEP_ALL_COMPONENTS_MEETING_MIN_SIZE",
                },
            }
        )
        self.assertEqual(component_default.to_dict(), component_explicit.to_dict())
        self.assertEqual(
            list(component_default.parameters),
            [
                "comparison",
                "threshold",
                "connectivity",
                "min_component_voxels",
                "selection_rule",
            ],
        )

        slice_default = resolve_mask_policy(
            {
                "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                "parameters": {
                    "threshold": 2.5,
                    "min_consecutive_slices": 2,
                },
            }
        )
        slice_explicit = resolve_mask_policy(
            {
                "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                "parameters": {
                    "comparison": "GREATER_THAN_OR_EQUAL",
                    "threshold": 2.5,
                    "consistency_unit": "IN_SLICE_COMPONENT",
                    "in_slice_connectivity": "FACES_2D",
                    "min_consecutive_slices": 2,
                    "overlap_rule": "ANY_OVERLAP",
                },
            }
        )
        self.assertEqual(slice_default.to_dict(), slice_explicit.to_dict())
        self.assertEqual(
            list(slice_default.parameters),
            [
                "comparison",
                "threshold",
                "consistency_unit",
                "in_slice_connectivity",
                "min_consecutive_slices",
                "overlap_rule",
            ],
        )
        slice_one = build_verification_profile_descriptor("brain_t1_slice_review")
        slice_two = build_verification_profile_descriptor("brain_t1_slice_review")
        self.assertEqual(slice_one["profile_digest"], slice_two["profile_digest"])

    def test_component_mask_policy_rejects_invalid_connectivity_selection_and_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_mask_policy(
                {
                    "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "connectivity": "edges",
                        "min_component_voxels": 8,
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_mask_policy(
                {
                    "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "min_component_voxels": 8,
                        "selection_rule": "keep_largest_component",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "must be >="):
            resolve_mask_policy(
                {
                    "algorithm": CONNECTED_COMPONENT_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "min_component_voxels": 0,
                    },
                }
            )

    def test_slice_mask_policy_rejects_invalid_unit_connectivity_overlap_and_run_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_mask_policy(
                {
                    "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "consistency_unit": "voxel",
                        "min_consecutive_slices": 2,
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_mask_policy(
                {
                    "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "min_consecutive_slices": 2,
                        "in_slice_connectivity": "edges_2d",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            resolve_mask_policy(
                {
                    "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "min_consecutive_slices": 2,
                        "overlap_rule": "min_overlap_fraction",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "must be >="):
            resolve_mask_policy(
                {
                    "algorithm": SLICE_CONSISTENCY_ALGORITHM,
                    "parameters": {
                        "threshold": 2.5,
                        "min_consecutive_slices": 1,
                    },
                }
            )

    def test_profile_descriptor_canonicalises_equivalent_mask_rules(self) -> None:
        canonical_payload = build_verification_profile_descriptor("brain_t1_morphology_review")
        noncanonical_payload = json.loads(canonical_json_dumps(canonical_payload))
        del noncanonical_payload["mask_rules"]["parameters"]["opening_radius_voxels"]
        del noncanonical_payload["mask_rules"]["parameters"]["closing_radius_voxels"]
        reloaded = verification_profile_descriptor_from_dict(noncanonical_payload)

        self.assertEqual(reloaded.to_dict(), canonical_payload)
        self.assertEqual(reloaded.profile_digest(), canonical_payload["profile_digest"])

    def test_slice_profile_descriptor_canonicalises_equivalent_mask_rules(self) -> None:
        canonical_payload = build_verification_profile_descriptor("brain_t1_slice_review")
        noncanonical_payload = json.loads(canonical_json_dumps(canonical_payload))
        del noncanonical_payload["mask_rules"]["parameters"]["consistency_unit"]
        del noncanonical_payload["mask_rules"]["parameters"]["in_slice_connectivity"]
        del noncanonical_payload["mask_rules"]["parameters"]["overlap_rule"]
        reloaded = verification_profile_descriptor_from_dict(noncanonical_payload)

        self.assertEqual(reloaded.to_dict(), canonical_payload)
        self.assertEqual(reloaded.profile_digest(), canonical_payload["profile_digest"])

    def test_component_profile_descriptor_canonicalises_equivalent_mask_rules(self) -> None:
        canonical_payload = build_verification_profile_descriptor("brain_t1_component_review")
        noncanonical_payload = json.loads(canonical_json_dumps(canonical_payload))
        del noncanonical_payload["mask_rules"]["parameters"]["connectivity"]
        del noncanonical_payload["mask_rules"]["parameters"]["selection_rule"]
        reloaded = verification_profile_descriptor_from_dict(noncanonical_payload)

        self.assertEqual(reloaded.to_dict(), canonical_payload)
        self.assertEqual(reloaded.profile_digest(), canonical_payload["profile_digest"])

    def test_demo_case_releases_and_generates_verifiable_evidence(self) -> None:
        state, result = self._release_demo_case()

        self.assertEqual(result.after_state["release_status"], RELEASE_STATUS_RELEASED)
        self.assertEqual(validate_evidence_payload(result.evidence.as_dict()), ())
        self.assertTrue(
            result.evidence.verify_integrity(
                before_state=result.before_state,
                after_state=result.after_state,
            )
        )
        self.assertTrue(self._build_harness().verify_evidence(result.evidence))
        manifest = result.after_state["verified_manifest"]
        workstation_payload = result.after_state["workstation_payload"]
        self.assertEqual(manifest["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(manifest["profile_revision"], "1")
        self.assertEqual(
            manifest["profile_digest"],
            get_verification_profile(DEFAULT_PROFILE_NAME).profile_digest(),
        )
        self.assertEqual(manifest["decision_summary"]["proof_status"], "passed")
        self.assertEqual(manifest["decision_summary"]["profile_policy_status"], "passed")
        self.assertEqual(manifest["decision_summary"]["block_category"], "none")
        self.assertEqual(
            manifest["verification_profile_descriptor"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(
            manifest["mask_derivation_policy"]["profile_digest"],
            manifest["profile_digest"],
        )
        self.assertEqual(
            manifest["mask_verification"]["policy"]["algorithm"],
            THRESHOLD_ALGORITHM,
        )
        self.assertTrue(manifest["mask_verification"]["comparison"]["verified"])
        self.assertEqual(manifest["mask_verification"]["comparison"]["voxel_delta"], 0)
        self.assertEqual(manifest["mask_verification"]["threshold"], 2.5)
        self.assertEqual(
            manifest["verification_tolerance_policy"]["icsd_policy_id"],
            manifest["icsd_policy_id"],
        )
        self.assertEqual(
            manifest["verified_reconstruction_trace"]["profile_revision"],
            manifest["profile_revision"],
        )
        self.assertEqual(workstation_payload["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(workstation_payload["pixel_source"], "harness_reconstructed_from_raw_kspace")
        self.assertEqual(workstation_payload["overlay_mode"], "profile_verified_mask_on_measured_grid")
        report = build_case_verification_report(result.after_state)
        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["contracts"]["verification_profile_descriptor"], report["profile"])
        self.assertEqual(
            report["contracts"]["verified_reconstruction_trace"]["claim_envelope_hash"],
            hash_canonical_json(report["contracts"]["ai_claim_envelope"]),
        )
        self.assertEqual(
            report["contracts"]["verified_reconstruction_trace"]["tolerance_policy_hash"],
            hash_canonical_json(report["contracts"]["verification_tolerance_policy"]),
        )
        self.assertEqual(state["verification_profile_descriptor"], report["profile"])

    def test_morphology_profile_releases_and_keeps_mask_policy_nested(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_morphology_review")
        manifest = result.after_state["verified_manifest"]
        mask_verification = manifest["mask_verification"]

        self.assertEqual(manifest["profile_name"], "brain_t1_morphology_review")
        self.assertEqual(
            manifest["mask_derivation_policy"]["algorithm"],
            MORPHOLOGY_ALGORITHM,
        )
        self.assertEqual(
            mask_verification["policy"]["parameters"],
            {
                "comparison": "greater_than_or_equal",
                "threshold": 2.5,
                "opening_radius_voxels": 1,
                "closing_radius_voxels": 1,
            },
        )
        self.assertTrue(mask_verification["comparison"]["verified"])
        self.assertEqual(
            result.after_state["workstation_payload"]["verified_mask_hash"],
            manifest["verified_mask_hash"],
        )

    def test_component_profile_releases_and_keeps_mask_policy_nested(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_component_review")
        manifest = result.after_state["verified_manifest"]
        mask_verification = manifest["mask_verification"]

        self.assertEqual(manifest["profile_name"], "brain_t1_component_review")
        self.assertEqual(
            manifest["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )
        self.assertEqual(
            mask_verification["policy"]["parameters"],
            {
                "comparison": "greater_than_or_equal",
                "threshold": 2.5,
                "connectivity": "faces",
                "min_component_voxels": 8,
                "selection_rule": "keep_all_components_meeting_min_size",
            },
        )
        self.assertTrue(mask_verification["comparison"]["verified"])

    def test_slice_profile_releases_and_keeps_mask_policy_nested(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_slice_review")
        manifest = result.after_state["verified_manifest"]
        mask_verification = manifest["mask_verification"]

        self.assertEqual(manifest["profile_name"], "brain_t1_slice_review")
        self.assertEqual(
            manifest["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )
        self.assertEqual(
            mask_verification["policy"]["parameters"],
            {
                "comparison": "greater_than_or_equal",
                "threshold": 2.5,
                "consistency_unit": "in_slice_component",
                "in_slice_connectivity": "faces_2d",
                "min_consecutive_slices": 2,
                "overlap_rule": "any_overlap",
            },
        )
        self.assertTrue(mask_verification["comparison"]["verified"])

    def test_profile_specific_authority_requirements_are_enforced(self) -> None:
        contrast_state = build_demo_case(profile_name="contrast_enhanced_review")
        contrast_harness = self._build_harness(default_profile_name="contrast_enhanced_review")
        with self.assertRaisesRegex(ValueError, "contrast_protocol"):
            contrast_harness.release_case(
                state=contrast_state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=contrast_state["case_id"]),
                profile_name="contrast_enhanced_review",
                timestamp="2026-03-26T00:00:00Z",
            )
        contrast_result = contrast_harness.release_case(
            state=contrast_state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(
                case_id=contrast_state["case_id"],
                profile_name="contrast_enhanced_review",
            ),
            profile_name="contrast_enhanced_review",
            timestamp="2026-03-26T00:00:00Z",
        )
        self.assertEqual(
            contrast_result.after_state["verified_manifest"]["profile_name"],
            "contrast_enhanced_review",
        )

        strict_state = build_demo_case(profile_name="strict_research_mode")
        strict_harness = self._build_harness(default_profile_name="strict_research_mode")
        with self.assertRaisesRegex(ValueError, "research_protocol"):
            strict_harness.release_case(
                state=strict_state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=strict_state["case_id"]),
                profile_name="strict_research_mode",
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_warning_findings_accumulate_without_blocking_final_release(self) -> None:
        state = self._build_warning_stack_case()

        report = build_case_verification_report(state)
        view_model = build_side_by_side_view_model(state)

        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "passed")
        self.assertEqual(report["decision_summary"]["profile_policy_status"], "passed")
        self.assertEqual(report["decision_summary"]["decision_class"], "final_release_candidate")
        self.assertEqual(report["decision_summary"]["block_category"], "none")
        self.assertTrue(report["decision_summary"]["operator_review_required"])
        self.assertEqual(report["manifest"]["clinical_plausibility"]["warning_count"], 2)
        self.assertEqual(report["manifest"]["clinical_plausibility"]["status"], "warning")
        self.assertEqual(
            view_model["verification"]["primary_reason_summary"],
            "all_blocking_checks_passed; clinical_plausibility_review_recommended",
        )
        self.assertEqual(view_model["integration"]["egress_label"], "Mock PACS Output")

    def test_brain_t2_warning_allows_final_release_with_operator_review(self) -> None:
        state = build_demo_case(profile_name="brain_t2_baseline")
        report = build_case_verification_report(state, profile_name="brain_t2_baseline")
        view_model = build_side_by_side_view_model(state, profile_name="brain_t2_baseline")

        self.assertTrue(report["release_eligible"])
        self.assertTrue(report["decision_summary"]["operator_review_required"])
        self.assertEqual(report["verification_tiers"]["level_3"]["status"], "warning")
        self.assertEqual(report["manifest"]["clinical_plausibility"]["warning_count"], 1)
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED")
        self.assertEqual(view_model["verification"]["tiers"][2]["status_label"], "ADVISORY WARN")

    def test_strict_research_mode_blocks_on_heuristic_failure(self) -> None:
        state = build_demo_case(profile_name="strict_research_mode")
        report = build_case_verification_report(state, profile_name="strict_research_mode")

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "passed")
        self.assertEqual(report["decision_summary"]["profile_policy_status"], "blocked")
        self.assertEqual(report["decision_summary"]["block_category"], "heuristic_block")
        self.assertEqual(report["failure_reasons"][0]["category"], "heuristic_block")
        self.assertEqual(report["failure_reasons"][0]["code"], "slice_span")
        self.assertEqual(report["manifest"]["clinical_plausibility"]["blocking_failure_count"], 1)
        self.assertEqual(report["verification_tiers"]["level_3"]["status"], "failed")

    def test_fast_triage_profile_is_non_final_and_rejected_for_release(self) -> None:
        state = build_demo_case(profile_name="fast_triage_mode")
        report = build_case_verification_report(state, profile_name="fast_triage_mode")
        view_model = build_side_by_side_view_model(state, profile_name="fast_triage_mode")

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "passed")
        self.assertEqual(report["decision_summary"]["profile_policy_status"], "blocked")
        self.assertEqual(report["decision_summary"]["decision_class"], "triage_only")
        self.assertEqual(report["decision_summary"]["block_category"], "profile_policy_block")
        self.assertEqual(report["failure_reasons"][0]["code"], "non_final_profile_selected")
        self.assertEqual(view_model["verification"]["headline"], "TRIAGE ONLY")
        self.assertEqual(view_model["clinical_summary"]["status_label"], "TRIAGE ONLY")
        self.assertEqual(view_model["integration"]["egress_label"], "No Final PACS Output")
        with self.assertRaisesRegex(ValueError, "non-final"):
            self._build_harness(default_profile_name="fast_triage_mode").release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(
                    case_id=state["case_id"],
                    profile_name="fast_triage_mode",
                ),
                profile_name="fast_triage_mode",
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_rejects_ai_mask_with_fake_tumour_voxel(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1

        with self.assertRaises(InvariantViolationError):
            self._build_harness().release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_rejects_ai_reconstruction_with_invented_pixels(self) -> None:
        state = build_demo_case()
        state["ai_reconstruction"][0][0][0] = 0.5

        with self.assertRaises(InvariantViolationError):
            self._build_harness().release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_rejects_incorrect_tumor_volume_claim(self) -> None:
        state = build_demo_case()
        state["tumor_volume_mm3"] = 9.9

        with self.assertRaises(InvariantViolationError):
            self._build_harness().release_case(
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

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "failed")
        self.assertEqual(report["decision_summary"]["block_category"], "proof_failure")
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
            self._build_harness().release_case(
                state=state,
                actor="guardian:test",
                authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
                timestamp="2026-03-26T00:00:00Z",
            )

    def test_morphology_engine_edge_cases_are_deterministic(self) -> None:
        morphology_rules = {
            "algorithm": MORPHOLOGY_ALGORITHM,
            "parameters": {"threshold": 2.5},
        }
        centered_all_one_expected = np.zeros((3, 3, 3), dtype=np.int64)
        centered_all_one_expected[1, 1, 1] = 1
        centered_block_expected = np.zeros((5, 5, 5), dtype=np.int64)
        centered_block_expected[2, 2, 2] = 1
        centered_block_expected[1, 2, 2] = 1
        centered_block_expected[3, 2, 2] = 1
        centered_block_expected[2, 1, 2] = 1
        centered_block_expected[2, 3, 2] = 1
        centered_block_expected[2, 2, 1] = 1
        centered_block_expected[2, 2, 3] = 1
        cases = (
            ("all_zero", np.zeros((3, 3, 3), dtype=np.float64), np.zeros((3, 3, 3), dtype=np.int64)),
            (
                "all_one_bounded_volume",
                np.full((3, 3, 3), 4.0, dtype=np.float64),
                centered_all_one_expected,
            ),
            (
                "single_voxel_removed",
                _reconstruction_with_hot_voxels((3, 3, 3), ((1, 1, 1),)),
                np.zeros((3, 3, 3), dtype=np.int64),
            ),
            (
                "thin_line_removed",
                _reconstruction_with_hot_voxels((5, 5, 5), ((2, 2, 1), (2, 2, 2), (2, 2, 3))),
                np.zeros((5, 5, 5), dtype=np.int64),
            ),
            (
                "corner_touching_pair_removed",
                _reconstruction_with_hot_voxels((5, 5, 5), ((2, 2, 2), (3, 3, 3))),
                np.zeros((5, 5, 5), dtype=np.int64),
            ),
            (
                "border_touching_cube_removed",
                _reconstruction_with_hot_voxels(
                    (5, 5, 5),
                    tuple((z, y, x) for z in range(2) for y in range(2) for x in range(2)),
                ),
                np.zeros((5, 5, 5), dtype=np.int64),
            ),
            (
                "centered_block_survives_as_cross",
                _reconstruction_with_hot_voxels(
                    (5, 5, 5),
                    tuple((z, y, x) for z in range(1, 4) for y in range(1, 4) for x in range(1, 4)),
                ),
                centered_block_expected,
            ),
        )

        for label, reconstruction, expected in cases:
            with self.subTest(case=label):
                first = derive_verified_mask(reconstruction, morphology_rules).verified_mask
                second = derive_verified_mask(reconstruction, morphology_rules).verified_mask
                self.assertTrue(np.array_equal(first, second))
                self.assertTrue(np.array_equal(first, expected))

    def test_component_engine_edge_cases_are_deterministic(self) -> None:
        component_rules = {
            "algorithm": CONNECTED_COMPONENT_ALGORITHM,
            "parameters": {
                "threshold": 2.5,
                "min_component_voxels": 8,
            },
        }
        all_one_expected = np.ones((3, 3, 3), dtype=np.int64)
        exactly_min_expected = np.zeros((5, 5, 5), dtype=np.int64)
        exactly_min_expected[1:3, 1:3, 1:3] = 1
        two_eligible_expected = np.zeros((6, 6, 6), dtype=np.int64)
        two_eligible_expected[0:2, 0:2, 0:2] = 1
        two_eligible_expected[4:6, 4:6, 4:6] = 1
        border_component_expected = np.zeros((5, 5, 5), dtype=np.int64)
        border_component_expected[0:2, 0:2, 0:2] = 1
        mixed_size_expected = np.zeros((6, 6, 6), dtype=np.int64)
        mixed_size_expected[0:2, 0:2, 0:2] = 1
        edge_touching_blocks = tuple(
            list(
                (2, y_index, x_index)
                for y_index in range(1, 3)
                for x_index in range(1, 3)
            )
            + list(
                (3, y_index, x_index)
                for y_index in range(1, 3)
                for x_index in range(3, 5)
            )
        )
        corner_touching_blocks = tuple(
            list(
                (2, y_index, x_index)
                for y_index in range(1, 3)
                for x_index in range(1, 3)
            )
            + list(
                (3, y_index, x_index)
                for y_index in range(3, 5)
                for x_index in range(3, 5)
            )
        )
        cases = (
            ("all_zero", np.zeros((3, 3, 3), dtype=np.float64), np.zeros((3, 3, 3), dtype=np.int64)),
            (
                "all_one_bounded_volume",
                np.full((3, 3, 3), 4.0, dtype=np.float64),
                all_one_expected,
            ),
            (
                "single_component_below_min_removed",
                _reconstruction_with_hot_voxels(
                    (5, 5, 5),
                    (
                        (1, 1, 1),
                        (1, 1, 2),
                        (1, 1, 3),
                        (1, 2, 1),
                        (1, 2, 2),
                        (2, 1, 1),
                        (2, 1, 2),
                    ),
                ),
                np.zeros((5, 5, 5), dtype=np.int64),
            ),
            (
                "single_component_exactly_min_survives",
                _reconstruction_with_hot_voxels(
                    (5, 5, 5),
                    tuple((z, y, x) for z in range(1, 3) for y in range(1, 3) for x in range(1, 3)),
                ),
                exactly_min_expected,
            ),
            (
                "mixed_size_components_keep_only_eligible",
                _reconstruction_with_hot_voxels(
                    (6, 6, 6),
                    tuple((z, y, x) for z in range(0, 2) for y in range(0, 2) for x in range(0, 2))
                    + ((4, 4, 4), (4, 4, 5), (4, 5, 4), (4, 5, 5)),
                ),
                mixed_size_expected,
            ),
            (
                "two_eligible_components_both_survive",
                _reconstruction_with_hot_voxels(
                    (6, 6, 6),
                    tuple((z, y, x) for z in range(0, 2) for y in range(0, 2) for x in range(0, 2))
                    + tuple((z, y, x) for z in range(4, 6) for y in range(4, 6) for x in range(4, 6)),
                ),
                two_eligible_expected,
            ),
            (
                "edge_touching_components_stay_separate_under_faces",
                _reconstruction_with_hot_voxels((6, 6, 6), edge_touching_blocks),
                np.zeros((6, 6, 6), dtype=np.int64),
            ),
            (
                "corner_touching_components_stay_separate_under_faces",
                _reconstruction_with_hot_voxels((6, 6, 6), corner_touching_blocks),
                np.zeros((6, 6, 6), dtype=np.int64),
            ),
            (
                "border_touching_component_survives",
                _reconstruction_with_hot_voxels(
                    (5, 5, 5),
                    tuple((z, y, x) for z in range(0, 2) for y in range(0, 2) for x in range(0, 2)),
                ),
                border_component_expected,
            ),
        )

        for label, reconstruction, expected in cases:
            with self.subTest(case=label):
                first = derive_verified_mask(reconstruction, component_rules).verified_mask
                second = derive_verified_mask(reconstruction, component_rules).verified_mask
                self.assertTrue(np.array_equal(first, second))
                self.assertTrue(np.array_equal(first, expected))

    def test_slice_engine_edge_cases_are_deterministic(self) -> None:
        slice_rules = {
            "algorithm": SLICE_CONSISTENCY_ALGORITHM,
            "parameters": {
                "threshold": 2.5,
                "min_consecutive_slices": 2,
            },
        }
        all_one_expected = np.ones((3, 3, 3), dtype=np.int64)
        overlapping_pair_expected = np.zeros((2, 4, 4), dtype=np.int64)
        overlapping_pair_expected[:, 1:3, 1:3] = 1
        border_pair_expected = np.zeros((2, 4, 4), dtype=np.int64)
        border_pair_expected[:, 0:2, 0:2] = 1
        corner_touching_same_slice = (
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 3, 4),
            (1, 4, 3),
            (1, 4, 4),
        )
        cases = (
            ("all_zero", np.zeros((3, 3, 3), dtype=np.float64), np.zeros((3, 3, 3), dtype=np.int64)),
            (
                "all_one_multi_slice_volume_survives",
                np.full((3, 3, 3), 4.0, dtype=np.float64),
                all_one_expected,
            ),
            (
                "single_slice_component_removed",
                _reconstruction_with_hot_voxels(
                    (3, 4, 4),
                    tuple((1, y_index, x_index) for y_index in range(1, 3) for x_index in range(1, 3)),
                ),
                np.zeros((3, 4, 4), dtype=np.int64),
            ),
            (
                "exactly_two_overlapping_consecutive_slices_survive",
                _reconstruction_with_hot_voxels(
                    (2, 4, 4),
                    tuple((z_index, y_index, x_index) for z_index in range(2) for y_index in range(1, 3) for x_index in range(1, 3)),
                ),
                overlapping_pair_expected,
            ),
            (
                "two_adjacent_slices_with_no_overlap_are_removed",
                _reconstruction_with_hot_voxels(
                    (2, 5, 5),
                    tuple((0, y_index, x_index) for y_index in range(0, 2) for x_index in range(0, 2))
                    + tuple((1, y_index, x_index) for y_index in range(3, 5) for x_index in range(3, 5)),
                ),
                np.zeros((2, 5, 5), dtype=np.int64),
            ),
            (
                "run_with_one_slice_gap_does_not_survive",
                _reconstruction_with_hot_voxels(
                    (3, 4, 4),
                    tuple((z_index, y_index, x_index) for z_index in (0, 2) for y_index in range(1, 3) for x_index in range(1, 3)),
                ),
                np.zeros((3, 4, 4), dtype=np.int64),
            ),
            (
                "corner_touching_regions_stay_separate_under_faces_2d",
                _reconstruction_with_hot_voxels((3, 6, 6), corner_touching_same_slice),
                np.zeros((3, 6, 6), dtype=np.int64),
            ),
            (
                "border_touching_component_survives_with_adjacent_support",
                _reconstruction_with_hot_voxels(
                    (2, 4, 4),
                    tuple((z_index, y_index, x_index) for z_index in range(2) for y_index in range(0, 2) for x_index in range(0, 2)),
                ),
                border_pair_expected,
            ),
        )

        for label, reconstruction, expected in cases:
            with self.subTest(case=label):
                first_execution = derive_verified_mask(reconstruction, slice_rules)
                second_execution = derive_verified_mask(reconstruction, slice_rules)
                self.assertTrue(np.array_equal(first_execution.verified_mask, second_execution.verified_mask))
                self.assertTrue(np.array_equal(first_execution.verified_mask, expected))
                self.assertIn(
                    "slice_component_survival_is_defined_by_qualifying_run_membership_not_label_rank",
                    first_execution.runtime_metadata.debug_notes,
                )

    def test_slice_engine_three_slice_chain_with_middle_branch_retains_all_components_on_qualifying_paths(self) -> None:
        slice_rules = {
            "algorithm": SLICE_CONSISTENCY_ALGORITHM,
            "parameters": {
                "threshold": 2.5,
                "min_consecutive_slices": 3,
            },
        }
        hot_voxels = (
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (1, 2, 1),
            (1, 2, 3),
            (2, 2, 1),
            (2, 2, 2),
            (2, 2, 3),
        )
        expected = np.zeros((3, 5, 5), dtype=np.int64)
        expected[0, 2, 1:4] = 1
        expected[1, 2, 1] = 1
        expected[1, 2, 3] = 1
        expected[2, 2, 1:4] = 1

        verified = derive_verified_mask(
            _reconstruction_with_hot_voxels((3, 5, 5), hot_voxels),
            slice_rules,
        ).verified_mask

        self.assertTrue(np.array_equal(verified, expected))

    def test_slice_engine_only_retains_components_on_qualifying_paths(self) -> None:
        slice_rules = {
            "algorithm": SLICE_CONSISTENCY_ALGORITHM,
            "parameters": {
                "threshold": 2.5,
                "min_consecutive_slices": 3,
            },
        }
        hot_voxels = (
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (1, 2, 1),
            (1, 2, 3),
            (2, 2, 1),
        )
        expected = np.zeros((3, 5, 5), dtype=np.int64)
        expected[0, 2, 1:4] = 1
        expected[1, 2, 1] = 1
        expected[2, 2, 1] = 1

        verified = derive_verified_mask(
            _reconstruction_with_hot_voxels((3, 5, 5), hot_voxels),
            slice_rules,
        ).verified_mask

        self.assertTrue(np.array_equal(verified, expected))

    def test_cross_engine_policy_non_equivalence_blocks_threshold_mask_under_morphology(self) -> None:
        baseline_state = build_demo_case(profile_name="brain_t1_baseline")
        baseline_report = build_case_verification_report(
            baseline_state,
            profile_name="brain_t1_baseline",
        )
        morphology_report = build_case_verification_report(
            baseline_state,
            profile_name="brain_t1_morphology_review",
        )

        self.assertTrue(baseline_report["release_eligible"])
        self.assertFalse(morphology_report["release_eligible"])
        self.assertEqual(
            morphology_report["manifest"]["mask_derivation_policy"]["algorithm"],
            MORPHOLOGY_ALGORITHM,
        )
        self.assertGreater(
            morphology_report["manifest"]["mask_verification"]["comparison"]["voxel_delta"],
            0,
        )

    def test_cross_engine_policy_non_equivalence_blocks_threshold_mask_under_component(self) -> None:
        component_state = build_demo_case(profile_name="brain_t1_component_review")
        reconstruction = np.asarray(component_state["ai_reconstruction"], dtype=np.float64)
        threshold_mask = (reconstruction >= 2.5).astype(np.int64)
        component_state["tumor_mask"] = threshold_mask.tolist()
        component_state["tumor_volume_mm3"] = round(
            float(
                threshold_mask.sum()
                * np.prod(np.asarray(component_state["voxel_spacing_mm"], dtype=np.float64))
            ),
            9,
        )

        component_report = build_case_verification_report(
            component_state,
            profile_name="brain_t1_component_review",
        )

        self.assertFalse(component_report["release_eligible"])
        self.assertEqual(
            component_report["manifest"]["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )
        self.assertGreater(
            component_report["manifest"]["mask_verification"]["comparison"]["voxel_delta"],
            0,
        )

    def test_cross_engine_policy_non_equivalence_blocks_threshold_mask_under_slice_consistency(self) -> None:
        slice_state = build_demo_case(profile_name="brain_t1_slice_review")
        reconstruction = np.asarray(slice_state["ai_reconstruction"], dtype=np.float64)
        threshold_mask = (reconstruction >= 2.5).astype(np.int64)
        slice_state["tumor_mask"] = threshold_mask.tolist()
        slice_state["tumor_volume_mm3"] = round(
            float(
                threshold_mask.sum()
                * np.prod(np.asarray(slice_state["voxel_spacing_mm"], dtype=np.float64))
            ),
            9,
        )

        slice_report = build_case_verification_report(
            slice_state,
            profile_name="brain_t1_slice_review",
        )

        self.assertFalse(slice_report["release_eligible"])
        self.assertEqual(
            slice_report["manifest"]["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )
        self.assertGreater(
            slice_report["manifest"]["mask_verification"]["comparison"]["voxel_delta"],
            0,
        )

    def test_mock_dicom_input_round_trips_guardian_case_state(self) -> None:
        state = build_demo_case()

        dicom_input = build_mock_dicom_input(state)
        restored_state = restore_case_state_from_dicom_input(dicom_input)

        self.assertEqual(restored_state, state)
        self.assertEqual(dicom_input["endpoint"]["destination_ae_title"], "MRI-GUARDIAN-LAYER")
        self.assertEqual(dicom_input["objects"][0]["role"], "raw_kspace")
        self.assertEqual(dicom_input["ai_claim_envelope"]["interface_type"], "AIClaimEnvelope")
        self.assertEqual(
            dicom_input["verification_profile_descriptor"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(
            dicom_input["integration_narrative"]["hospital_placement"]["position"],
            "between_reconstruction_ai_and_pacs",
        )

    def test_mock_pacs_submission_exposes_verified_handoff_contract(self) -> None:
        _, result = self._release_demo_case()

        dicom_input = build_mock_dicom_input(result.before_state)
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
            pacs_submission["verification_summary"]["decision_class"],
            "final_release_candidate",
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(
            pacs_submission["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(pacs_submission["series"][0]["role"], "verified_reconstruction")
        self.assertEqual(
            pacs_submission["integration_narrative"]["failure_mode"]["current_guardian_decision"],
            "released",
        )
        self.assertEqual(integration_trace["pipeline"][0]["label"], "DICOM Input")
        self.assertEqual(integration_trace["pipeline"][1]["decision_class"], "final_release_candidate")
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
        self.assertEqual(pacs_block["blocked_reason"], "mask_voxel_mismatch (delta = 342 voxels)")
        self.assertEqual(pacs_block["verification_summary"]["proof_status"], "failed")
        self.assertEqual(pacs_block["verification_summary"]["profile_policy_status"], "passed")
        self.assertEqual(
            pacs_block["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(integration_trace["pipeline"][1]["decision"], "blocked")
        self.assertEqual(integration_trace["pipeline"][2]["label"], "PACS Output Blocked")

    def test_view_model_reports_clean_baseline_case(self) -> None:
        view_model = build_side_by_side_view_model(build_demo_case())

        self.assertTrue(view_model["release_eligible"])
        self.assertEqual(view_model["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED")
        self.assertEqual(
            [tier["status"] for tier in view_model["verification"]["tiers"]],
            ["passed", "passed", "passed"],
        )
        self.assertEqual(view_model["decision_summary"]["decision_class"], "final_release_candidate")
        self.assertGreater(view_model["timing"]["reconstruction_ms"], 0)
        self.assertGreaterEqual(view_model["timing"]["verification_ms"], 0)
        self.assertGreaterEqual(view_model["timing"]["overhead_percent"], 0)
        self.assertIn("budget_assessment", view_model["timing"])
        self.assertEqual(view_model["summary"]["mask_voxel_delta"], 0)
        self.assertEqual(view_model["clinical_summary"]["status_label"], "VERIFIED")
        self.assertEqual(view_model["integration"]["egress_label"], "Mock PACS Output")
        self.assertEqual(view_model["differences"]["rows"], [])

    def test_view_model_marks_triage_only_profile(self) -> None:
        state = build_demo_case(profile_name="fast_triage_mode")
        view_model = build_side_by_side_view_model(state, profile_name="fast_triage_mode")

        self.assertFalse(view_model["release_eligible"])
        self.assertEqual(view_model["verification"]["headline"], "TRIAGE ONLY")
        self.assertEqual(view_model["clinical_summary"]["status_state"], "triage")
        self.assertEqual(view_model["integration"]["egress_label"], "No Final PACS Output")

    def test_html_report_contains_profile_and_difference_register(self) -> None:
        state = build_demo_case()
        state["tumor_mask"][0][0][0] = 1

        html = render_side_by_side_report_html(
            state,
            report_title="Tamper Mode: Fake Tumour Injection",
        )

        self.assertIn("MRI Guardian Layer", html)
        self.assertIn("Profile brain_t1_baseline r1", html)
        self.assertIn("Difference Register", html)
        self.assertIn("Verification FAILED", html)
        self.assertIn("mask_voxel_mismatch", html)
        self.assertIn("Level 1: Structural Validity", html)
        self.assertIn("Level 2: Physical Consistency", html)
        self.assertIn("Level 3: Clinical Plausibility", html)
        self.assertIn("Streaming / Partial Validation", html)

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
        self.assertEqual(fake_view["verification"]["tiers"][1]["status"], "failed")
        self.assertEqual(fake_view["summary"]["mask_voxel_delta"], 342)
        self.assertEqual(
            shift_view["verification"]["primary_reason_summary"],
            "mask_voxel_mismatch (delta = 46 voxels)",
        )
        self.assertEqual(shift_view["summary"]["mask_voxel_delta"], 46)
        self.assertEqual(
            intensity_view["verification"]["primary_reason_summary"],
            "intensity_manipulation_detected (delta = 147 voxels; max_abs_error = 0.55)",
        )
        self.assertEqual(intensity_view["summary"]["intensity_mismatch_voxel_count"], 147)

    def test_component_profile_tamper_mode_surfaces_micro_island_mismatch(self) -> None:
        scenarios = {
            scenario["slug"]: scenario
            for scenario in build_tamper_scenarios(profile_name="brain_t1_component_review")
        }

        micro_island_view = build_side_by_side_view_model(
            scenarios["component_micro_island"]["state"],
            profile_name="brain_t1_component_review",
        )
        micro_island_report = build_case_verification_report(
            scenarios["component_micro_island"]["state"],
            profile_name="brain_t1_component_review",
        )

        self.assertEqual(
            micro_island_view["verification"]["primary_reason_summary"],
            "mask_voxel_mismatch (delta = 4 voxels)",
        )
        self.assertEqual(micro_island_view["summary"]["mask_voxel_delta"], 4)
        self.assertEqual(
            micro_island_report["contracts"]["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )

    def test_slice_profile_tamper_mode_surfaces_single_slice_island_mismatch(self) -> None:
        scenarios = {
            scenario["slug"]: scenario
            for scenario in build_tamper_scenarios(profile_name="brain_t1_slice_review")
        }

        single_slice_view = build_side_by_side_view_model(
            scenarios["slice_single_slice_island"]["state"],
            profile_name="brain_t1_slice_review",
        )
        single_slice_report = build_case_verification_report(
            scenarios["slice_single_slice_island"]["state"],
            profile_name="brain_t1_slice_review",
        )

        self.assertEqual(
            single_slice_view["verification"]["primary_reason_summary"],
            "mask_voxel_mismatch (delta = 4 voxels)",
        )
        self.assertEqual(single_slice_view["summary"]["mask_voxel_delta"], 4)
        self.assertEqual(
            single_slice_report["contracts"]["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )

    def test_timing_overlay_snapshot_is_positive_and_profile_scoped(self) -> None:
        profile = build_guardian_performance_profile("brain_t2_baseline")
        snapshot = build_timing_overlay_snapshot("brain_t2_baseline")

        self.assertEqual(snapshot, profile)
        self.assertEqual(snapshot["profile_name"], "brain_t2_baseline")
        self.assertEqual(snapshot["profile_revision"], "1")
        self.assertGreater(snapshot["reconstruction_ms"], 0)
        self.assertGreater(snapshot["verification_ms"], 0)
        self.assertGreater(snapshot["total_pipeline_ms"], 0)
        self.assertGreaterEqual(snapshot["overhead_percent"], 0)
        self.assertFalse(snapshot["performance_expectations"]["normative"])
        self.assertIn("budget_assessment", snapshot)
        self.assertEqual(len(snapshot["benchmarks"]), 3)
        self.assertEqual(snapshot["streaming_strategy"]["recommended_slab_depth_slices"], 4)

    def test_portable_bundle_round_trips_and_replays_identically(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"baseline{MRI_GUARD_BUNDLE_EXTENSION}"

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
            profile_descriptor = self._read_bundle_json(
                archive,
                "verification_profile_descriptor.json",
            )

        self.assertEqual(manifest["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(manifest["profile_digest"], profile_descriptor["profile_digest"])
        self.assertEqual(
            manifest["hashes"]["verification_profile_descriptor_hash"],
            hash_canonical_json(profile_descriptor),
        )
        self.assertEqual(
            manifest["hashes"]["ai_claim_envelope_hash"],
            hash_canonical_json(result.after_state["verified_manifest"]["ai_claim_envelope"]),
        )
        self.assertEqual(transcript["events"][0]["stage"], "profile")
        self.assertEqual(replay["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(
            replay["expected_hashes"]["verification_profile_descriptor_hash"],
            hash_canonical_json(profile_descriptor),
        )
        self.assertEqual(signatures["members"][0]["algorithm"], "sha256")

    def test_non_default_profile_bundle_replays_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="contrast_enhanced_review")
        bundle_path = self.runtime_root / f"contrast{MRI_GUARD_BUNDLE_EXTENSION}"

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
            profile_descriptor = self._read_bundle_json(
                archive,
                "verification_profile_descriptor.json",
            )
        self.assertEqual(manifest["profile_name"], "contrast_enhanced_review")
        self.assertEqual(profile_descriptor["profile_name"], "contrast_enhanced_review")

    def test_morphology_profile_bundle_and_mock_pacs_flow_replay_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_morphology_review")
        bundle_path = self.runtime_root / f"morphology{MRI_GUARD_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference=bundle_path.name,
        )

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        self.assertEqual(
            pacs_submission["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            MORPHOLOGY_ALGORITHM,
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["profile_name"],
            "brain_t1_morphology_review",
        )

    def test_component_profile_bundle_and_mock_pacs_flow_replay_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_component_review")
        bundle_path = self.runtime_root / f"component{MRI_GUARD_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference=bundle_path.name,
        )

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        self.assertEqual(
            pacs_submission["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["profile_name"],
            "brain_t1_component_review",
        )

    def test_slice_profile_bundle_and_mock_pacs_flow_replay_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_slice_review")
        bundle_path = self.runtime_root / f"slice{MRI_GUARD_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference=bundle_path.name,
        )

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        self.assertEqual(
            pacs_submission["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["profile_name"],
            "brain_t1_slice_review",
        )

    def test_replay_uses_embedded_profile_when_runtime_default_changes(self) -> None:
        _, result = self._release_demo_case(profile_name="contrast_enhanced_review")
        bundle_path = self.runtime_root / f"default-shift{MRI_GUARD_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )

        with patch.object(profile_registry_module, "DEFAULT_PROFILE_NAME", "brain_t2_baseline"):
            verification = verify_bundle(bundle_path)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])

    def test_replay_fails_with_embedded_profile_digest_mismatch_when_runtime_registry_changes(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"digest-mismatch{MRI_GUARD_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"digest-mismatch-tampered{MRI_GUARD_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            replay = self._read_bundle_json(archive, "replay.json")
            member_bytes = {
                info.filename: archive.read(info.filename)
                for info in archive.infolist()
            }
        replay["profile_digest"] = "0" * 64
        member_bytes["replay.json"] = self._json_bytes(replay)
        member_bytes["signatures.json"] = self._json_bytes(
            self._build_signature_catalog(
                {
                    name: payload
                    for name, payload in member_bytes.items()
                    if name != "signatures.json"
                }
            )
        )
        self._rewrite_bundle_members(
            source=bundle_path,
            destination=tampered_path,
            member_bytes=member_bytes,
        )

        verification = verify_bundle(tampered_path)

        self.assertFalse(verification["verified"])
        self.assertTrue(
            any(
                "embedded_profile_digest_mismatch" in failure
                for failure in verification["failures"]
            )
        )

    def test_verify_bundle_accepts_extracted_bundle_json(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"extracted{MRI_GUARD_BUNDLE_EXTENSION}"
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
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"clean{MRI_GUARD_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"tampered{MRI_GUARD_BUNDLE_EXTENSION}"

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
            any(
                "transcript.json" in failure or "signatures.json" in failure
                for failure in verification["failures"]
            )
        )

    def test_cli_verify_command_reports_directory_success(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"{result.after_state['case_id']}{MRI_GUARD_BUNDLE_EXTENSION}"
        (self.runtime_root / "evidence.json").write_text(
            f"{result.evidence_json}\n",
            encoding="utf-8",
        )
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )

        verification = run_verify_command(self.runtime_root)

        self.assertTrue(verification["verified"])
        self.assertIn("evidence", verification["parts"])
        self.assertIn("bundle", verification["parts"])

    def test_cli_test_command_runs_sample_suite(self) -> None:
        sample_root = self.runtime_root / "sample_tests"
        sample_root.mkdir()
        (sample_root / "test_smoke.py").write_text(
            "import unittest\n\n"
            "class SmokeTest(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )

        result = run_test_command(
            start_dir=str(sample_root),
            pattern="test_smoke.py",
            top_level_dir=str(sample_root),
            verbosity=0,
        )

        self.assertTrue(result["successful"])
        self.assertEqual(result["tests_run"], 1)

    def test_python_module_cli_demo_and_verify_commands_work(self) -> None:
        demo_root = self.runtime_root / "cli_demo"
        demo_command = [
            sys.executable,
            "-m",
            "mri_guardian",
            "demo",
            "--output-dir",
            str(demo_root),
            "--tamper-mode",
        ]
        verify_command = [
            sys.executable,
            "-m",
            "mri_guardian",
            "verify",
            str(demo_root),
            "--format",
            "json",
        ]

        demo_run = subprocess.run(
            demo_command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        verify_run = subprocess.run(
            verify_command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(demo_run.returncode, 0, demo_run.stderr)
        self.assertTrue((demo_root / "comparison.html").exists())
        self.assertEqual(verify_run.returncode, 0, verify_run.stderr)
        verification = json.loads(verify_run.stdout)
        self.assertTrue(verification["verified"])

    def test_python_module_cli_demo_rejects_unknown_profile(self) -> None:
        demo_root = self.runtime_root / "cli_demo_invalid"
        demo_command = [
            sys.executable,
            "-m",
            "mri_guardian",
            "demo",
            "--output-dir",
            str(demo_root),
            "--profile",
            "not-a-profile",
        ]

        demo_run = subprocess.run(
            demo_command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(demo_run.returncode, 0)
        self.assertIn("Unknown Guardian runtime profile", demo_run.stderr)

    def test_run_demo_triage_emits_only_non_final_artifacts(self) -> None:
        demo_root = self.runtime_root / "triage_demo"

        run_demo(
            output_dir=demo_root,
            profile_name="fast_triage_mode",
            tamper_mode=False,
        )

        present = {path.name for path in demo_root.iterdir()}
        self.assertEqual(
            present,
            {
                "comparison.html",
                "performance_profile.json",
                "verification_profile.json",
                "verification_profile_descriptor.json",
                "verification_report.json",
            },
        )

    def test_run_demo_morphology_profile_emits_verified_artifacts(self) -> None:
        demo_root = self.runtime_root / "morphology_demo"

        run_demo(
            output_dir=demo_root,
            profile_name="brain_t1_morphology_review",
            tamper_mode=True,
        )
        verification = run_verify_command(demo_root)
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))
        tamper_block = json.loads(
            (demo_root / "tamper_fake_tumour_pacs_block.json").read_text(encoding="utf-8")
        )

        self.assertTrue(verification["verified"])
        self.assertEqual(report["profile"]["profile_name"], "brain_t1_morphology_review")
        self.assertEqual(
            report["contracts"]["mask_derivation_policy"]["algorithm"],
            MORPHOLOGY_ALGORITHM,
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            "brain_t1_morphology_review",
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            MORPHOLOGY_ALGORITHM,
        )
        self.assertTrue((demo_root / "tamper_fake_tumour.html").exists())

    def test_run_demo_component_profile_emits_verified_artifacts(self) -> None:
        demo_root = self.runtime_root / "component_demo"

        run_demo(
            output_dir=demo_root,
            profile_name="brain_t1_component_review",
            tamper_mode=True,
        )
        verification = run_verify_command(demo_root)
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))
        tamper_block = json.loads(
            (demo_root / "tamper_component_micro_island_pacs_block.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(verification["verified"])
        self.assertEqual(report["profile"]["profile_name"], "brain_t1_component_review")
        self.assertEqual(
            report["contracts"]["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            "brain_t1_component_review",
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            CONNECTED_COMPONENT_ALGORITHM,
        )
        self.assertTrue((demo_root / "tamper_component_micro_island.html").exists())

    def test_run_demo_slice_profile_emits_verified_artifacts(self) -> None:
        demo_root = self.runtime_root / "slice_demo"

        run_demo(
            output_dir=demo_root,
            profile_name="brain_t1_slice_review",
            tamper_mode=True,
        )
        verification = run_verify_command(demo_root)
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))
        tamper_block = json.loads(
            (demo_root / "tamper_slice_single_slice_island_pacs_block.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(verification["verified"])
        self.assertEqual(report["profile"]["profile_name"], "brain_t1_slice_review")
        self.assertEqual(
            report["contracts"]["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            "brain_t1_slice_review",
        )
        self.assertEqual(
            tamper_block["proof_contracts"]["mask_derivation_policy"]["algorithm"],
            SLICE_CONSISTENCY_ALGORITHM,
        )
        self.assertTrue((demo_root / "tamper_slice_single_slice_island.html").exists())

    def test_guardian_integration_narrative_is_fail_closed_and_gateway_positioned(self) -> None:
        narrative = build_guardian_integration_narrative(current_guardian_decision="blocked")

        self.assertEqual(narrative["guardian_role"], "verification_proxy")
        self.assertEqual(narrative["failure_mode"]["mode"], "fail_closed")
        self.assertEqual(
            narrative["hospital_placement"]["position"],
            "between_reconstruction_ai_and_pacs",
        )
        self.assertEqual(narrative["dicom_compliance_story"]["ingress_transport"], "dicom-c-store")

    def _build_harness(
        self,
        *,
        default_profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> MriGuardianHarness:
        return MriGuardianHarness(
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            default_profile_name=default_profile_name,
        )

    def _release_demo_case(
        self,
        *,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> tuple[dict[str, object], object]:
        state = build_demo_case(profile_name=profile_name)
        result = self._build_harness(default_profile_name=profile_name).release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(
                case_id=state["case_id"],
                profile_name=profile_name,
            ),
            profile_name=profile_name,
            timestamp="2026-03-26T00:00:00Z",
        )
        return state, result

    def _build_warning_stack_case(self) -> dict[str, object]:
        state = build_demo_case()
        reconstruction = np.asarray(
            [
                [[3.0, 3.0], [3.0, 3.0]],
                [[0.0, 0.0], [0.0, 0.0]],
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

    def _rewrite_bundle_members(
        self,
        *,
        source: Path,
        destination: Path,
        member_bytes: dict[str, bytes],
    ) -> None:
        with ZipFile(source, mode="r") as src, ZipFile(destination, mode="w") as dst:
            for info in src.infolist():
                clone = ZipInfo(filename=info.filename, date_time=info.date_time)
                clone.compress_type = info.compress_type
                clone.create_system = info.create_system
                clone.external_attr = info.external_attr
                dst.writestr(clone, member_bytes[info.filename])

    def _build_signature_catalog(self, member_bytes: dict[str, bytes]) -> dict[str, object]:
        return {
            "schema_version": "0.3",
            "signature_type": "sha256_integrity_catalog",
            "members": [
                {
                    "path": member_name,
                    "algorithm": "sha256",
                    "value": hashlib.sha256(payload).hexdigest(),
                }
                for member_name, payload in sorted(member_bytes.items())
            ],
        }

    def _json_bytes(self, payload: dict[str, object]) -> bytes:
        return f"{canonical_json_dumps(payload)}\n".encode("utf-8")

    def _cleanup_runtime_root(self) -> None:
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        parent = self.runtime_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    unittest.main()
