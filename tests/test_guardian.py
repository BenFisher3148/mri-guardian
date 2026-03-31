from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import unittest
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from icsd.core.errors import InvariantViolationError
from icsd.core.evidence import validate_evidence_payload
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json

import mri_guardian.profiles as profile_registry_module
from mri_guardian.bundle import (
    GUARDIAN_BUNDLE_EXTENSION,
    verify_bundle,
    write_portable_evidence_bundle,
)
from mri_guardian.cli import run_test_command, run_verify_command
from mri_guardian.dicom_model import (
    DICOM_MODEL_SCHEMA_VERSION,
    GUARDIAN_EXTENSION_NAMESPACE,
    SIDECAR_TYPE_ICSD_EVIDENCE_JSON,
    deterministic_uid,
)
from mri_guardian.demo import run_demo
from mri_guardian.harness import (
    MRI_GUARDIAN_POLICY_NAME,
    MRI_GUARDIAN_POLICY_VERSION,
    MULTI_SERIES_CASE_KIND,
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
    build_multi_series_case_verification_report,
    build_multi_series_demo_authority_sources,
    build_multi_series_demo_case,
    _validate_operational_subphases,
    _temporal_pair_provenance,
)
from mri_guardian.integration import (
    build_guardian_integration_narrative,
    build_mock_dicom_input,
    build_mock_integration_trace,
    build_mock_pacs_block_notice,
    build_mock_pacs_submission,
    restore_case_state_from_dicom_input,
)
from mri_guardian.journal import build_decision_journal, validate_decision_journal
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
from mri_guardian.tamper import (
    build_tamper_scenarios,
    build_verification_threat_corpus,
)
from mri_guardian.timing import (
    build_guardian_performance_profile,
    build_timing_overlay_snapshot,
)
from mri_guardian.transparency import JsonlHashChainTransparencyLog
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


class _BlockingReadTransparencyLog(JsonlHashChainTransparencyLog):
    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self.read_entered = threading.Event()
        self.release_read = threading.Event()
        self._block_next_read = False

    def arm_next_read(self) -> None:
        self.read_entered.clear()
        self.release_read.clear()
        self._block_next_read = True

    def _read_entries(self):
        if self._block_next_read:
            self._block_next_read = False
            self.read_entered.set()
            if not self.release_read.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the blocking transparency read.")
        return super()._read_entries()


class MriGuardianHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_root = Path("test_runs") / uuid4().hex
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(self._cleanup_runtime_root)

    def _attach_ai_stack_identity(
        self,
        state: dict[str, object],
        *,
        compatibility_type: str = "exact_sequence_id",
        declared_value: str | None = None,
        parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if declared_value is None:
            if compatibility_type == "sequence_family":
                descriptor = verification_profile_descriptor_from_dict(
                    dict(state["verification_profile_descriptor"])
                )
                declared_value = descriptor.sequence_family
            else:
                declared_value = str(state["sequence_id"])
        state["ai_stack_identity"] = {
            "model_version_id": "raw-mri-ai-v4.1.0",
            "weights_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "preprocessing_pipeline_sha256": (
                "1111111111111111111111111111111111111111111111111111111111111111"
            ),
            "postprocessing_pipeline_sha256": (
                "2222222222222222222222222222222222222222222222222222222222222222"
            ),
            "input_normalization": {
                "scheme": "zscore_per_channel",
                "parameters": (
                    parameters
                    if parameters is not None
                    else {
                        "channel_order": {"secondary": 1, "primary": 0},
                        "clip_range": [0.0, 1.0],
                        "epsilon": 1e-6,
                        "per_channel": True,
                    }
                ),
            },
            "sequence_model_compatibility": {
                "compatibility_type": compatibility_type,
                "declared_value": declared_value,
            },
        }
        return state

    def _ai_runtime_context(self) -> dict[str, str]:
        return {
            "inference_runtime_version": "guardian-runtime/2.3.4",
            "inference_device_class": "cuda_sm80",
        }

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
            manifest["segmentation_policy_proof"]["policy"]["algorithm"],
            THRESHOLD_ALGORITHM,
        )
        self.assertTrue(manifest["segmentation_policy_proof"]["comparison"]["verified"])
        self.assertEqual(manifest["segmentation_policy_proof"]["comparison"]["voxel_delta"], 0)
        self.assertEqual(manifest["segmentation_policy_proof"]["threshold"], 2.5)
        self.assertEqual(
            manifest["verification_tolerance_policy"]["icsd_policy_id"],
            manifest["icsd_policy_id"],
        )
        self.assertEqual(
            manifest["proof_core_evidence_trace"]["profile_revision"],
            manifest["profile_revision"],
        )
        self.assertEqual(workstation_payload["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(workstation_payload["pixel_source"], "harness_reconstructed_from_raw_kspace")
        self.assertEqual(workstation_payload["overlay_mode"], "profile_verified_mask_on_measured_grid")
        report = build_case_verification_report(result.after_state)
        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["contracts"]["verification_profile_descriptor"], report["profile"])
        self.assertEqual(
            report["contracts"]["proof_core_evidence_trace"]["ai_claim_envelope_hash"],
            hash_canonical_json(report["contracts"]["ai_claim_envelope"]),
        )
        self.assertEqual(
            report["contracts"]["measurement_integrity_proof"]["tolerance_policy_hash"],
            hash_canonical_json(report["contracts"]["verification_tolerance_policy"]),
        )
        self.assertEqual(
            report["contracts"]["proof_core_evidence_trace"]["proof_stage_hashes"][
                "MeasurementIntegrityProof"
            ],
            hash_canonical_json(report["contracts"]["measurement_integrity_proof"]),
        )
        self.assertEqual(state["verification_profile_descriptor"], report["profile"])

    def test_temporal_trace_is_not_evaluated_without_prior_context(self) -> None:
        state = build_demo_case()

        report = build_case_verification_report(state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "not_evaluated")
        self.assertEqual(temporal_trace["comparison_scope"], "not_evaluated")
        self.assertEqual(
            temporal_trace["prior_admissibility"]["reason_code"],
            "no_prior_context",
        )
        self.assertEqual(
            report["manifest"]["clinical_plausibility"]["temporal_consistency"],
            temporal_trace,
        )

    def test_temporal_pair_provenance_derivation_uses_deterministic_precedence(self) -> None:
        cases = (
            (
                "metadata_pair",
                "trusted_study_metadata",
                "trusted_study_metadata",
                "trusted_metadata_pair",
            ),
            (
                "verified_bundle_pair",
                "verified_bundle",
                "verified_bundle",
                "verified_bundle_pair",
            ),
            (
                "trusted_plus_verified_bundle_is_mixed",
                "trusted_study_metadata",
                "verified_bundle",
                "mixed_provenance",
            ),
            (
                "verified_bundle_plus_trusted_is_mixed",
                "verified_bundle",
                "trusted_study_metadata",
                "mixed_provenance",
            ),
            (
                "missing_plus_trusted",
                "missing",
                "trusted_study_metadata",
                "timestamp_missing",
            ),
            (
                "untrusted_dominates_verified_bundle",
                "untrusted",
                "verified_bundle",
                "timestamp_untrusted",
            ),
            (
                "untrusted_dominates_missing",
                "missing",
                "untrusted",
                "timestamp_untrusted",
            ),
        )

        for label, current_provenance, prior_provenance, expected_pair_provenance in cases:
            with self.subTest(case=label):
                self.assertEqual(
                    _temporal_pair_provenance(current_provenance, prior_provenance),
                    expected_pair_provenance,
                )

    def test_temporal_aligned_full_comparison_supports_trusted_metadata_pair(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-07-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="trusted_study_metadata",
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]
        view_model = build_side_by_side_view_model(current_state)
        html = render_side_by_side_report_html(current_state)

        self.assertEqual(temporal_trace["comparison_status"], "aligned")
        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertEqual(
            temporal_trace["chronology_assessment"]["pair_provenance"],
            "trusted_metadata_pair",
        )
        self.assertEqual(
            temporal_trace["computed_metrics"]["volume_change_direction"],
            "stable_band",
        )
        self.assertEqual(
            temporal_trace["computed_metrics"]["normalized_rate_direction"],
            "stable_band",
        )
        self.assertTrue(view_model["longitudinal"]["overlay_enabled"])
        self.assertIn("Longitudinal Summary", html)
        self.assertIn("Prior verified overlay", html)

    def test_temporal_mixed_provenance_degrades_but_keeps_full_comparison(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-07-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="verified_bundle",
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "degraded")
        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertEqual(
            temporal_trace["chronology_assessment"]["pair_provenance"],
            "mixed_provenance",
        )
        self.assertEqual(
            temporal_trace["chronology_assessment"]["reason_code"],
            "mixed_provenance",
        )
        self.assertTrue(temporal_trace["chronology_assessment"]["time_metrics_available"])
        self.assertEqual(
            temporal_trace["computed_metrics"]["normalized_rate_direction"],
            "stable_band",
        )

    def test_temporal_missing_current_timestamp_degrades_but_keeps_full_spatial_comparison(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-03-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at=None,
            study_timestamp_provenance="missing",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="trusted_study_metadata",
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]
        skipped_by_metric = {
            item["metric"]: item["reason_code"] for item in temporal_trace["skipped_metrics"]
        }

        self.assertEqual(temporal_trace["comparison_status"], "degraded")
        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertEqual(
            temporal_trace["chronology_assessment"]["reason_code"],
            "timestamp_missing",
        )
        self.assertIn("temporal_rate_not_evaluated", temporal_trace["flags"])
        self.assertEqual(
            skipped_by_metric["signed_normalized_percent_change_per_30d"],
            "timestamp_missing",
        )
        self.assertEqual(
            temporal_trace["computed_metrics"]["intersection_over_union"],
            1.0,
        )
        self.assertEqual(
            temporal_trace["computed_metrics"]["global_centroid_shift_mm"],
            0.0,
        )

    def test_temporal_missing_prior_bundle_identity_is_inadmissible_but_non_blocking(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-03-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-09-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"]["prior_bundle_sha256"] = ""

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "unsuitable")
        self.assertEqual(temporal_trace["comparison_scope"], "none")
        self.assertEqual(
            temporal_trace["prior_admissibility"]["reason_code"],
            "prior_bundle_identity_missing",
        )
        self.assertEqual(report["decision"]["final_disposition"], "released")
        self.assertNotEqual(report["decision"]["decision_class"], "blocked_structural")

    def test_temporal_equal_timestamps_degrade_and_skip_normalized_rate(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-04-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-04-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="trusted_study_metadata",
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "degraded")
        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertEqual(temporal_trace["chronology_assessment"]["interval_days"], 0.0)
        self.assertEqual(
            temporal_trace["chronology_assessment"]["reason_code"],
            "zero_interval_days",
        )
        self.assertIn(
            {
                "metric": "signed_normalized_percent_change_per_30d",
                "reason_code": "zero_interval_days",
            },
            temporal_trace["skipped_metrics"],
        )

    def test_temporal_affine_geometry_mismatch_uses_registered_full_comparison(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-02-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        geometry_shifted = self._base_spatial_geometry()
        geometry_shifted["affine_ijk_to_mm"][0][3] = 1.0
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-08-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
            spatial_geometry=geometry_shifted,
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]
        registration_trace = report["contracts"]["longitudinal_registration_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "degraded")
        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertEqual(temporal_trace["spatial_comparison_mode"], "registered")
        self.assertEqual(
            temporal_trace["geometry_assessment"]["reason_code"],
            "geometry_registered_match",
        )
        self.assertEqual(registration_trace["status"], "used")
        self.assertTrue(registration_trace["quality"]["spatial_use_allowed"])
        self.assertNotIn(
            {
                "metric": "intersection_over_union",
                "reason_code": "geometry_not_exactly_comparable",
            },
            temporal_trace["skipped_metrics"],
        )

    def test_temporal_interval_beyond_v1_max_suppresses_all_comparison(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2010-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2026-03-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_status"], "unsuitable")
        self.assertEqual(temporal_trace["comparison_scope"], "none")
        self.assertEqual(
            temporal_trace["chronology_assessment"]["reason_code"],
            "interval_exceeds_max_days",
        )

    def test_temporal_empty_masks_skip_iou_and_centroid_deterministically(self) -> None:
        prior_state = self._attach_temporal_metadata(
            self._build_zero_mask_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        current_state = self._attach_temporal_metadata(
            self._build_zero_mask_case(),
            study_acquired_at="2025-06-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
        )

        report = build_case_verification_report(current_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]

        self.assertEqual(temporal_trace["comparison_scope"], "full")
        self.assertIn(
            {
                "metric": "intersection_over_union",
                "reason_code": "both_masks_empty",
            },
            temporal_trace["skipped_metrics"],
        )
        self.assertIn(
            {
                "metric": "global_centroid_shift_mm",
                "reason_code": "empty_prior_mask",
            },
            temporal_trace["skipped_metrics"],
        )
        self.assertIn(
            {
                "metric": "signed_volume_change_percent",
                "reason_code": "zero_prior_volume",
            },
            temporal_trace["skipped_metrics"],
        )
        self.assertEqual(
            temporal_trace["computed_metrics"]["volume_change_direction"],
            "not_evaluated",
        )

    def test_temporal_prior_with_advisory_release_is_admissible_and_bundle_surfaces_contracts(self) -> None:
        prior_state = self._attach_temporal_metadata(
            self._build_warning_stack_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        self.assertEqual(prior_report["decision"]["decision_class"], "verified_with_advisory")

        current_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-06-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
        )
        harness = self._build_harness()
        result = harness.release_case(
            state=current_state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=current_state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        report = build_case_verification_report(result.after_state)
        temporal_trace = report["contracts"]["temporal_consistency_trace"]
        bundle_path = self.runtime_root / f"temporal{GUARDIAN_BUNDLE_EXTENSION}"

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
        integration_trace = build_mock_integration_trace(
            dicom_input=build_mock_dicom_input(result.before_state),
            pacs_output=pacs_submission,
        )

        self.assertTrue(result.after_state["release_status"] == RELEASE_STATUS_RELEASED)
        self.assertTrue(verification["verified"])
        self.assertEqual(temporal_trace["prior_admissibility"]["status"], "admissible")
        self.assertIn("prior_verified_context", pacs_submission["proof_contracts"])
        self.assertIn("temporal_consistency_trace", pacs_submission["proof_contracts"])
        self.assertEqual(
            pacs_submission["verification_summary"]["temporal_comparison_status"],
            temporal_trace["comparison_status"],
        )
        self.assertEqual(
            integration_trace["pipeline"][1]["temporal_comparison_scope"],
            temporal_trace["comparison_scope"],
        )
        with ZipFile(bundle_path, mode="r") as archive:
            member_names = set(archive.namelist())
            manifest = self._read_bundle_json(archive, "bundle.json")
        self.assertEqual(
            member_names,
            {
                "bundle.json",
                "input_state.json",
                "released_state.json",
                "workstation_payload.json",
                "verification_report.json",
                "decision_journal.json",
                "evidence.json",
                "transparency-prefix.jsonl",
            },
        )
        self.assertEqual(
            manifest["replay_inputs"]["contracts"]["prior_verified_context"],
            report["contracts"]["prior_verified_context"],
        )
        self.assertEqual(
            manifest["proof_context"]["derived_traces"]["longitudinal_selection_trace"],
            report["contracts"]["longitudinal_selection_trace"],
        )
        self.assertEqual(
            manifest["proof_context"]["derived_traces"]["temporal_trajectory_trace"],
            report["contracts"]["temporal_trajectory_trace"],
        )
        self.assertEqual(
            manifest["proof_context"]["derived_traces"]["longitudinal_registration_trace"],
            report["contracts"]["longitudinal_registration_trace"],
        )
        self.assertEqual(
            manifest["proof_context"]["derived_traces"]["temporal_consistency_trace"],
            report["contracts"]["temporal_consistency_trace"],
        )

    def test_longitudinal_selection_trace_is_deterministic_for_multi_prior_history(self) -> None:
        prior_zero_state = self._attach_temporal_metadata(
            self._build_zero_mask_case(),
            study_acquired_at="2024-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_zero_result = self._release_demo_case_with_state(prior_zero_state)
        prior_zero_report = build_case_verification_report(prior_zero_result.after_state)
        prior_mid_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_mid_result = self._release_demo_case_with_state(prior_mid_state)
        prior_mid_report = build_case_verification_report(prior_mid_result.after_state)
        prior_recent_state = self._attach_temporal_metadata(
            self._build_warning_stack_case(),
            study_acquired_at="2025-06-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_recent_result = self._release_demo_case_with_state(prior_recent_state)
        prior_recent_report = build_case_verification_report(prior_recent_result.after_state)
        current_state = self._attach_temporal_metadata(
            self._build_warning_stack_case(),
            study_acquired_at="2026-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        prior_zero_context = self._build_prior_verified_context_from_release(
            released_state=prior_zero_result.after_state,
            released_report=prior_zero_report,
        )
        prior_mid_context = self._build_prior_verified_context_from_release(
            released_state=prior_mid_result.after_state,
            released_report=prior_mid_report,
        )
        prior_recent_context = self._build_prior_verified_context_from_release(
            released_state=prior_recent_result.after_state,
            released_report=prior_recent_report,
        )
        current_state["prior_verified_contexts"] = [
            prior_recent_context,
            prior_zero_context,
            prior_mid_context,
        ]

        report = build_case_verification_report(current_state)
        selection_trace = report["contracts"]["longitudinal_selection_trace"]
        trajectory_trace = report["contracts"]["temporal_trajectory_trace"]

        self.assertEqual(selection_trace["selected_prior_count"], 3)
        self.assertEqual(selection_trace["primary_selection_mode"], "trajectory_most_recent")
        self.assertEqual(
            report["contracts"]["prior_verified_context"]["prior_bundle_sha256"],
            prior_recent_context["prior_bundle_sha256"],
        )
        self.assertEqual(
            [entry["study_acquired_at"] for entry in selection_trace["entries"]],
            [
                "2024-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-06-01T00:00:00Z",
            ],
        )
        self.assertEqual(trajectory_trace["trajectory_class"], "monotonic_growth")
        self.assertIn("trajectory_recurrence_after_zero", trajectory_trace["flags"])

    def test_strict_research_profile_can_block_on_temporal_growth(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(profile_name="strict_research_mode"),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        self._rewrite_case_for_profile(
            prior_state,
            _reconstruction_with_hot_voxels(
                (4, 4, 4),
                ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
                hot_value=4.0,
            ),
        )
        _, prior_result = self._release_demo_case_with_state(
            prior_state,
            profile_name="strict_research_mode",
        )
        prior_report = build_case_verification_report(
            prior_result.after_state,
            profile_name="strict_research_mode",
        )
        current_state = self._attach_temporal_metadata(
            build_demo_case(profile_name="strict_research_mode"),
            study_acquired_at="2025-07-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        self._rewrite_case_for_profile(
            current_state,
            _reconstruction_with_hot_voxels(
                (4, 4, 4),
                (
                    (0, 0, 0),
                    (0, 0, 1),
                    (1, 1, 1),
                    (1, 1, 2),
                    (2, 2, 2),
                    (2, 2, 3),
                ),
                hot_value=4.0,
            ),
        )
        current_state["prior_verified_context"] = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
            timestamp_provenance="trusted_study_metadata",
        )

        report = build_case_verification_report(
            current_state,
            profile_name="strict_research_mode",
        )

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["block_category"], "heuristic_block")
        self.assertIn(
            "temporal_volume_change_warning",
            {
                item["code"]
                for item in report["manifest"]["clinical_plausibility"]["blocking_failures"]
            },
        )
        self.assertEqual(
            report["contracts"]["temporal_consistency_trace"]["comparison_status"],
            "aligned",
        )

    def test_multi_series_temporal_aggregation_tracks_required_series_coverage(self) -> None:
        prior_state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )
        _, prior_result = self._release_demo_case_with_state(prior_state)
        prior_report = build_case_verification_report(prior_result.after_state)
        primary_prior_context = self._build_prior_verified_context_from_release(
            released_state=prior_result.after_state,
            released_report=prior_report,
        )
        state = build_multi_series_demo_case()
        primary_entry = self._series_entry(state, "t1-primary")
        primary_entry["study_acquired_at"] = "2025-08-01T00:00:00Z"
        primary_entry["study_timestamp_provenance"] = "trusted_study_metadata"
        primary_entry["spatial_geometry"] = deepcopy(self._base_spatial_geometry())
        state["prior_verified_context_by_series"] = {"t1-primary": primary_prior_context}

        report = build_multi_series_case_verification_report(state)
        study_temporal_trace = report["contracts"]["study_temporal_aggregation_trace"]

        self.assertEqual(study_temporal_trace["final_temporal_disposition"], "advisory")
        self.assertIn(
            "t2-supporting",
            study_temporal_trace["gating_summary"]["required_series_without_usable_temporal"],
        )
        self.assertIn(
            "flair-supporting",
            study_temporal_trace["gating_summary"]["required_series_without_usable_temporal"],
        )
        primary_series = next(
            item for item in study_temporal_trace["per_series"] if item["series_id"] == "t1-primary"
        )
        self.assertTrue(primary_series["usable_temporal_result"])


    def test_ai_claim_envelope_preserves_legacy_shape_when_ai_stack_identity_is_absent(self) -> None:
        state = build_demo_case()

        report = build_case_verification_report(state)
        expected_claim_envelope = {
            "interface_type": "AIClaimEnvelope",
            "interface_version": "2026.03",
            "claim_scope": "reconstruction_segmentation_volume",
            "case_id": state["case_id"],
            "study_uid": state["study_uid"],
            "series_uid": state["series_uid"],
            "patient_pseudonym": state["patient_pseudonym"],
            "ai_model_id": state["ai_model_id"],
            "raw_kspace_hash": report["analysis"]["raw_hash"],
            "voxel_spacing_mm": state["voxel_spacing_mm"],
            "claimed_outputs": {
                "ai_reconstruction_hash": report["analysis"]["ai_reconstruction_hash"],
                "ai_mask_hash": report["analysis"]["ai_mask_hash"],
                "claimed_tumor_volume_mm3": state["tumor_volume_mm3"],
            },
        }

        self.assertEqual(report["contracts"]["ai_claim_envelope"], expected_claim_envelope)
        self.assertNotIn("ai_stack_identity", report["contracts"]["ai_claim_envelope"])
        self.assertEqual(
            canonical_json_dumps(report["contracts"]["ai_claim_envelope"]),
            canonical_json_dumps(expected_claim_envelope),
        )

    def test_ai_claim_envelope_binds_declared_ai_stack_identity_into_claim_hashes(self) -> None:
        state_without_provenance = build_demo_case()
        state_with_provenance = self._attach_ai_stack_identity(build_demo_case())

        baseline_report = build_case_verification_report(state_without_provenance)
        provenance_report = build_case_verification_report(state_with_provenance)

        self.assertNotIn("ai_stack_identity", baseline_report["contracts"]["ai_claim_envelope"])
        self.assertEqual(
            provenance_report["contracts"]["ai_claim_envelope"]["ai_stack_identity"],
            state_with_provenance["ai_stack_identity"],
        )
        self.assertNotEqual(
            hash_canonical_json(baseline_report["contracts"]["ai_claim_envelope"]),
            hash_canonical_json(provenance_report["contracts"]["ai_claim_envelope"]),
        )
        self.assertEqual(
            provenance_report["contracts"]["proof_core_evidence_trace"]["ai_claim_envelope_hash"],
            hash_canonical_json(provenance_report["contracts"]["ai_claim_envelope"]),
        )

    def test_ai_stack_identity_rejects_invalid_json_like_parameters(self) -> None:
        invalid_parameters = (
            {"bad": None},
            {"bad": (1, 2)},
            {"bad": np.float64(1.0)},
            {"bad": float("inf")},
            {"bad": object()},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=repr(parameters)):
                state = self._attach_ai_stack_identity(
                    build_demo_case(),
                    parameters=parameters,
                )
                report = build_case_verification_report(state)

                self.assertFalse(report["release_eligible"])
                self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
                self.assertEqual(
                    report["failure_reasons"][0]["code"],
                    "invalid_ai_stack_identity",
                )

    def test_ai_stack_identity_rejects_invalid_sha256_formats_and_empty_identifier_fields(self) -> None:
        invalid_overrides = {
            "weights_sha256": " 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "preprocessing_pipeline_sha256": (
                "0x1111111111111111111111111111111111111111111111111111111111111111"
            ),
            "postprocessing_pipeline_sha256": (
                "ABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCDEFABCD"
            ),
            "model_version_id": "",
        }

        for field_name, invalid_value in invalid_overrides.items():
            with self.subTest(field_name=field_name):
                state = self._attach_ai_stack_identity(build_demo_case())
                state["ai_stack_identity"][field_name] = invalid_value
                report = build_case_verification_report(state)

                self.assertFalse(report["release_eligible"])
                self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
                self.assertEqual(
                    report["failure_reasons"][0]["code"],
                    "invalid_ai_stack_identity",
                )

    def test_ai_stack_identity_preserves_type_differences_and_mapping_order_independence(self) -> None:
        first_parameters = OrderedDict(
            [
                ("bool_flag", True),
                ("int_value", 1),
                ("float_value", 1.0),
                ("string_value", "1"),
                (
                    "nested",
                    OrderedDict(
                        [
                            (
                                "secondary",
                                OrderedDict([("beta", 2), ("alpha", 1)]),
                            ),
                            ("primary", [True, 1, 1.0, "1"]),
                        ]
                    ),
                ),
            ]
        )
        second_parameters = OrderedDict(
            [
                (
                    "nested",
                    OrderedDict(
                        [
                            ("primary", [True, 1, 1.0, "1"]),
                            (
                                "secondary",
                                OrderedDict([("alpha", 1), ("beta", 2)]),
                            ),
                        ]
                    ),
                ),
                ("string_value", "1"),
                ("float_value", 1.0),
                ("int_value", 1),
                ("bool_flag", True),
            ]
        )

        first_report = build_case_verification_report(
            self._attach_ai_stack_identity(
                build_demo_case(),
                parameters=first_parameters,
            )
        )
        second_report = build_case_verification_report(
            self._attach_ai_stack_identity(
                build_demo_case(),
                parameters=second_parameters,
            )
        )

        first_claim = first_report["contracts"]["ai_claim_envelope"]
        second_claim = second_report["contracts"]["ai_claim_envelope"]
        parameters = first_claim["ai_stack_identity"]["input_normalization"]["parameters"]

        self.assertIs(type(parameters["bool_flag"]), bool)
        self.assertIs(type(parameters["int_value"]), int)
        self.assertIs(type(parameters["float_value"]), float)
        self.assertIs(type(parameters["string_value"]), str)
        self.assertEqual(canonical_json_dumps(first_claim), canonical_json_dumps(second_claim))
        self.assertEqual(hash_canonical_json(first_claim), hash_canonical_json(second_claim))

    def test_ai_stack_identity_rejects_padded_declared_string_fields(self) -> None:
        def _set_model_version(state: dict[str, object]) -> None:
            state["ai_stack_identity"]["model_version_id"] = " raw-mri-ai-v4.1.0 "

        def _set_scheme(state: dict[str, object]) -> None:
            state["ai_stack_identity"]["input_normalization"]["scheme"] = " zscore_per_channel "

        invalid_mutations = (_set_model_version, _set_scheme)

        for mutate in invalid_mutations:
            with self.subTest(mutation=mutate.__name__):
                state = self._attach_ai_stack_identity(build_demo_case())
                mutate(state)
                report = build_case_verification_report(state)

                self.assertFalse(report["release_eligible"])
                self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
                self.assertEqual(
                    report["failure_reasons"][0]["code"],
                    "invalid_ai_stack_identity",
                )

    def test_ai_stack_identity_rejects_sequence_compatibility_mismatches(self) -> None:
        cases = (
            ("exact_sequence_id", "wrong-sequence"),
            ("sequence_family", "not-a-known-family"),
            ("sequence_family", "brain_t2"),
        )

        for compatibility_type, declared_value in cases:
            with self.subTest(
                compatibility_type=compatibility_type,
                declared_value=declared_value,
            ):
                state = self._attach_ai_stack_identity(
                    build_demo_case(),
                    compatibility_type=compatibility_type,
                    declared_value=declared_value,
                )
                report = build_case_verification_report(state)

                self.assertFalse(report["release_eligible"])
                self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
                self.assertEqual(
                    report["failure_reasons"][0]["code"],
                    "invalid_ai_stack_identity",
                )

    def test_morphology_profile_releases_and_keeps_mask_policy_nested(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_morphology_review")
        manifest = result.after_state["verified_manifest"]
        mask_verification = manifest["segmentation_policy_proof"]

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
        mask_verification = manifest["segmentation_policy_proof"]

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
        mask_verification = manifest["segmentation_policy_proof"]

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
        self.assertEqual(report["decision"]["decision_class"], "verified_with_advisory")
        self.assertEqual(report["decision"]["pipeline_phase"], "clinical_plausibility")
        self.assertTrue(report["decision"]["success_modifiers"]["has_non_blocking_advisory"])
        self.assertEqual(report["manifest"]["clinical_plausibility"]["warning_count"], 2)
        self.assertEqual(report["manifest"]["clinical_plausibility"]["status"], "warning")
        self.assertEqual(view_model["decision"]["decision_class"], "verified_with_advisory")
        self.assertEqual(
            view_model["verification"]["primary_reason_summary"],
            "clinical_plausibility_non_blocking_finding",
        )
        self.assertEqual(view_model["integration"]["egress_label"], "Mock PACS Output")

    def test_brain_t2_warning_allows_final_release_with_operator_review(self) -> None:
        state = build_demo_case(profile_name="brain_t2_baseline")
        report = build_case_verification_report(state, profile_name="brain_t2_baseline")
        view_model = build_side_by_side_view_model(state, profile_name="brain_t2_baseline")

        self.assertTrue(report["release_eligible"])
        self.assertTrue(report["decision_summary"]["operator_review_required"])
        self.assertEqual(report["decision"]["decision_class"], "verified_with_advisory")
        self.assertEqual(
            report["verification_pipeline"]["phases"]["clinical_plausibility"]["status"],
            "warning",
        )
        self.assertEqual(report["manifest"]["clinical_plausibility"]["warning_count"], 1)
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED WITH ADVISORY")
        self.assertEqual(view_model["verification"]["tiers"][2]["status_label"], "ADVISORY WARN")

    def test_strict_research_mode_blocks_on_heuristic_failure(self) -> None:
        state = build_demo_case(profile_name="strict_research_mode")
        report = build_case_verification_report(state, profile_name="strict_research_mode")

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "passed")
        self.assertEqual(report["decision_summary"]["profile_policy_status"], "blocked")
        self.assertEqual(report["decision_summary"]["block_category"], "heuristic_block")
        self.assertEqual(report["decision"]["decision_class"], "blocked_policy")
        self.assertEqual(report["decision"]["pipeline_phase"], "policy_evaluation")
        self.assertEqual(report["failure_reasons"][0]["category"], "heuristic_block")
        self.assertEqual(report["failure_reasons"][0]["code"], "slice_span")
        self.assertEqual(report["manifest"]["clinical_plausibility"]["blocking_failure_count"], 1)
        self.assertEqual(
            report["verification_pipeline"]["phases"]["policy_evaluation"]["status"],
            "blocked",
        )

    def test_fast_triage_profile_is_non_final_and_rejected_for_release(self) -> None:
        state = build_demo_case(profile_name="fast_triage_mode")
        report = build_case_verification_report(state, profile_name="fast_triage_mode")
        view_model = build_side_by_side_view_model(state, profile_name="fast_triage_mode")

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["proof_status"], "passed")
        self.assertEqual(report["decision_summary"]["profile_policy_status"], "blocked")
        self.assertEqual(report["decision_summary"]["decision_class"], "triage_only")
        self.assertEqual(report["decision_summary"]["block_category"], "profile_policy_block")
        self.assertEqual(report["decision"]["decision_class"], "blocked_policy")
        self.assertEqual(report["decision"]["reason_codes"][0], "non_final_profile_selected")
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
        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")
        self.assertEqual(report["decision"]["pipeline_phase"], "proof_core")
        self.assertEqual(report["decision"]["proof_stage"], "ReconstructionParityProof")
        self.assertTrue(report["decision"]["success_modifiers"]["near_proof_threshold"])
        self.assertEqual(report["failure_reasons"][0]["code"], "intensity_manipulation_detected")
        self.assertEqual(report["analysis"]["reconstruction_mismatch_voxel_count"], 0)
        self.assertLess(report["analysis"]["max_ai_abs_error"], 1e-9)
        self.assertGreater(report["analysis"]["relative_ai_l2_error"], 1e-12)
        self.assertEqual(
            report["manifest"]["reconstruction_parity_proof"]["metrics"]["per_voxel_abs_error"][
                "boundary_status"
            ],
            "near_boundary",
        )
        self.assertEqual(
            report["manifest"]["reconstruction_parity_proof"]["metrics"][
                "global_relative_l2_error"
            ][
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

    def test_near_boundary_release_uses_boundary_hugging_class(self) -> None:
        state = build_demo_case()
        state["tumor_volume_mm3"] = round(float(state["tumor_volume_mm3"]) + 8.5e-10, 9)

        report = build_case_verification_report(state)

        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "verified_boundary_hugging")
        self.assertEqual(report["decision"]["pipeline_phase"], "proof_core")
        self.assertEqual(report["decision"]["proof_stage"], "VolumeComputationProof")
        self.assertTrue(report["decision"]["success_modifiers"]["near_proof_threshold"])
        self.assertFalse(report["decision"]["success_modifiers"]["has_non_blocking_advisory"])
        self.assertIn(
            "tumor_volume_absolute_error_near_boundary",
            report["decision"]["reason_codes"],
        )

    def test_advisory_precedence_wins_over_near_boundary_release(self) -> None:
        state = self._build_warning_stack_case()
        state["tumor_volume_mm3"] = round(float(state["tumor_volume_mm3"]) + 8.5e-10, 9)

        report = build_case_verification_report(state)

        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "verified_with_advisory")
        self.assertEqual(report["decision"]["reason_codes"][0], "clinical_plausibility_non_blocking_finding")
        self.assertTrue(report["decision"]["success_modifiers"]["near_proof_threshold"])
        self.assertTrue(report["decision"]["success_modifiers"]["has_non_blocking_advisory"])
        self.assertIn(
            "tumor_volume_absolute_error_near_boundary",
            report["decision"]["reason_codes"],
        )

    def test_single_series_operational_subphases_materialize_frozen_precheck_and_fullproof(self) -> None:
        report = build_case_verification_report(build_demo_case())

        operational_subphases = report["verification_pipeline"]["operational_subphases"]
        _validate_operational_subphases(operational_subphases)

        self.assertEqual(
            operational_subphases["subphase_order"],
            ["pre_check", "full_proof"],
        )
        pre_check = operational_subphases["subphases"]["pre_check"]
        full_proof = operational_subphases["subphases"]["full_proof"]
        self.assertEqual(
            [item["code"] for item in pre_check["checks"]],
            [
                "structure_validation",
                "hash_checks",
                "shape_consistency",
                "quick_anomaly_detection",
            ],
        )
        self.assertEqual(pre_check["status"], "passed")
        self.assertEqual(pre_check["checks"][1]["status"], "passed")
        self.assertEqual(pre_check["checks"][-1]["status"], "not_evaluated")
        self.assertEqual(full_proof["status"], "passed")
        self.assertEqual(
            [item["proof_stage"] for item in full_proof["proof_stages"]],
            [
                "MeasurementIntegrityProof",
                "ReconstructionParityProof",
                "SegmentationPolicyProof",
                "VolumeComputationProof",
            ],
        )
        self.assertTrue(all(item["status"] == "passed" for item in full_proof["proof_stages"]))

    def test_structural_single_series_operational_subphases_keep_not_evaluated_full_proof(self) -> None:
        state = build_demo_case()
        state["case_id"] = ""

        report = build_case_verification_report(state)

        operational_subphases = report["verification_pipeline"]["operational_subphases"]
        _validate_operational_subphases(operational_subphases)

        pre_check = operational_subphases["subphases"]["pre_check"]
        full_proof = operational_subphases["subphases"]["full_proof"]
        self.assertEqual(pre_check["status"], "failed")
        self.assertEqual(full_proof["status"], "not_evaluated")
        self.assertEqual(
            [item["status"] for item in full_proof["proof_stages"]],
            ["not_evaluated", "not_evaluated", "not_evaluated", "not_evaluated"],
        )
        self.assertEqual(report["decision"]["pipeline_phase"], "structural_validity")

    def test_operational_subphases_allow_advisory_precheck_without_changing_decision_ownership(self) -> None:
        state = self._attach_ai_stack_identity(
            build_demo_case(),
            compatibility_type="sequence_family",
        )

        report = build_case_verification_report(state)

        operational_subphases = report["verification_pipeline"]["operational_subphases"]
        _validate_operational_subphases(operational_subphases)

        pre_check = operational_subphases["subphases"]["pre_check"]
        self.assertEqual(pre_check["status"], "advisory")
        self.assertEqual(pre_check["checks"][-1]["status"], "advisory")
        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "verified_clean")
        self.assertEqual(report["decision"]["pipeline_phase"], "release_control")

    def test_multi_series_parent_omits_operational_subphases_while_child_reports_keep_them(self) -> None:
        report = build_multi_series_case_verification_report(build_multi_series_demo_case())

        self.assertNotIn("operational_subphases", report["verification_pipeline"])
        self.assertIn(
            "operational_subphases",
            report["series_reports"][0]["verification_pipeline"],
        )

    def test_operational_subphases_validation_rejects_non_conforming_generated_output(self) -> None:
        operational_subphases = deepcopy(
            build_case_verification_report(build_demo_case())["verification_pipeline"][
                "operational_subphases"
            ]
        )

        invalid_subphase_name = deepcopy(operational_subphases)
        invalid_subphase_name["subphase_order"][0] = "unexpected_subphase"
        with self.assertRaises(ValueError):
            _validate_operational_subphases(invalid_subphase_name)

        invalid_check_name = deepcopy(operational_subphases)
        invalid_check_name["subphases"]["pre_check"]["checks"][1]["code"] = "unexpected_check"
        with self.assertRaises(ValueError):
            _validate_operational_subphases(invalid_check_name)

        invalid_check_status = deepcopy(operational_subphases)
        invalid_check_status["subphases"]["pre_check"]["checks"][-1]["status"] = "failed"
        with self.assertRaises(ValueError):
            _validate_operational_subphases(invalid_check_status)

        invalid_operational_label = deepcopy(operational_subphases)
        invalid_operational_label["subphases"]["full_proof"]["proof_stages"][0][
            "proof_stage"
        ] = "FFT / reprojection"
        with self.assertRaises(ValueError):
            _validate_operational_subphases(invalid_operational_label)

    def test_decision_journal_clean_release_uses_canonical_order_and_scoped_evidence(self) -> None:
        state = build_demo_case()

        report = build_case_verification_report(state)
        journal = build_decision_journal(report)

        validate_decision_journal(journal, expected_reason_codes=report["decision"]["reason_codes"])
        self.assertEqual(journal["case_id"], state["case_id"])
        self.assertEqual(journal["case_kind"], "single_series_v1")
        self.assertEqual(journal["decision_class"], "verified_clean")
        self.assertEqual(
            [entry["event_id"] for entry in journal["entries"]],
            [
                "input_received",
                "structural_validation",
                "reconstruction",
                "parity_checks",
                "segmentation_verification",
                "volume_proof",
                "advisory_heuristics",
                "final_decision",
                "transparency_anchor",
            ],
        )
        self.assertEqual(journal["entries"][-1]["status"], "absent")
        self.assertEqual(journal["entries"][-1]["evidence_paths"], [])
        self.assertEqual(
            journal["entries"][-2]["reason_codes"],
            report["decision"]["reason_codes"],
        )

    def test_decision_journal_boundary_and_advisory_cases_use_frozen_statuses(self) -> None:
        boundary_state = build_demo_case()
        boundary_state["tumor_volume_mm3"] = round(
            float(boundary_state["tumor_volume_mm3"]) + 8.5e-10,
            9,
        )
        advisory_state = self._build_warning_stack_case()
        advisory_state["tumor_volume_mm3"] = round(
            float(advisory_state["tumor_volume_mm3"]) + 8.5e-10,
            9,
        )

        boundary_journal = build_decision_journal(build_case_verification_report(boundary_state))
        advisory_journal = build_decision_journal(build_case_verification_report(advisory_state))

        boundary_volume = next(
            entry for entry in boundary_journal["entries"] if entry["event_id"] == "volume_proof"
        )
        advisory_heuristics = next(
            entry
            for entry in advisory_journal["entries"]
            if entry["event_id"] == "advisory_heuristics"
        )
        boundary_final_decision = next(
            entry for entry in boundary_journal["entries"] if entry["event_id"] == "final_decision"
        )
        advisory_final_decision = next(
            entry for entry in advisory_journal["entries"] if entry["event_id"] == "final_decision"
        )
        self.assertEqual(boundary_volume["status"], "advisory")
        self.assertIn(
            "tumor_volume_absolute_error_near_boundary",
            boundary_volume["reason_codes"],
        )
        self.assertEqual(boundary_final_decision["status"], "advisory")
        self.assertEqual(
            advisory_journal["journal_summary"]["residual_risk"]["classification"],
            "operator_relevant",
        )
        self.assertEqual(advisory_heuristics["status"], "advisory")
        self.assertEqual(advisory_final_decision["status"], "advisory")

    def test_decision_journal_handles_structural_physical_and_non_final_policy_outcomes(self) -> None:
        structural_state = build_demo_case()
        structural_state["case_id"] = ""
        physical_state = build_demo_case()
        physical_reconstruction = np.asarray(physical_state["ai_reconstruction"], dtype=np.float64)
        physical_reconstruction += 9e-10
        physical_state["ai_reconstruction"] = physical_reconstruction.tolist()
        triage_state = build_demo_case(profile_name="fast_triage_mode")

        structural_journal = build_decision_journal(build_case_verification_report(structural_state))
        physical_journal = build_decision_journal(build_case_verification_report(physical_state))
        triage_journal = build_decision_journal(build_case_verification_report(triage_state))

        self.assertEqual(
            [entry["event_id"] for entry in structural_journal["entries"]],
            [
                "input_received",
                "structural_validation",
                "final_decision",
                "transparency_anchor",
            ],
        )
        parity_event = next(
            entry for entry in physical_journal["entries"] if entry["event_id"] == "parity_checks"
        )
        self.assertEqual(parity_event["status"], "failed")
        self.assertEqual(
            parity_event["hazards_triggered"][0]["code"],
            "image_integrity_distortion",
        )
        self.assertEqual(
            triage_journal["journal_summary"]["residual_risk"]["classification"],
            "workflow_external",
        )
        self.assertEqual(triage_journal["entries"][-1]["status"], "not_applicable")

    def test_decision_journal_multi_series_keeps_compact_child_outcomes_only(self) -> None:
        report = build_multi_series_case_verification_report(build_multi_series_demo_case())

        journal = build_decision_journal(report)
        validate_decision_journal(journal, expected_reason_codes=report["decision"]["reason_codes"])

        self.assertNotIn("reconstruction", [entry["event_id"] for entry in journal["entries"]])
        aggregation_event = next(
            entry
            for entry in journal["entries"]
            if entry["event_id"] == "multi_series_aggregation"
        )
        self.assertTrue(aggregation_event["child_outcomes"])
        self.assertEqual(aggregation_event["status"], "advisory")
        self.assertEqual(
            aggregation_event["residual_risk"]["classification"],
            "operator_relevant",
        )
        self.assertEqual(
            sorted(aggregation_event["child_outcomes"][0].keys()),
            sorted(
                [
                    "series_id",
                    "decision_class",
                    "final_disposition",
                    "gating_status",
                ]
            ),
        )

    def test_decision_journal_validation_rejects_unknown_event_ids(self) -> None:
        journal = build_decision_journal(build_case_verification_report(build_demo_case()))
        journal["entries"][0]["event_id"] = "unexpected_event"

        with self.assertRaises(ValueError):
            validate_decision_journal(journal)

    def test_decision_journal_validation_rejects_missing_required_events_and_invalid_alignment(self) -> None:
        journal = build_decision_journal(build_case_verification_report(build_demo_case()))
        journal["entries"] = [
            entry for entry in journal["entries"] if entry["event_id"] != "final_decision"
        ]

        with self.assertRaises(ValueError):
            validate_decision_journal(journal)

        journal = build_decision_journal(build_case_verification_report(build_demo_case()))
        journal["entries"][-1]["status"] = "recorded"

        with self.assertRaises(ValueError):
            validate_decision_journal(journal)

        journal = build_decision_journal(build_case_verification_report(build_demo_case()))
        journal["entries"][-2]["proof_stage"] = "SegmentationPolicyProof"

        with self.assertRaises(ValueError):
            validate_decision_journal(journal)

    def test_invalid_embedded_profile_descriptor_returns_structural_block_report(self) -> None:
        state = build_demo_case()
        state["verification_profile_descriptor"]["profile_digest"] = "bad-digest"

        report = build_case_verification_report(state)
        view_model = build_side_by_side_view_model(state)
        pacs_block = build_mock_pacs_block_notice(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
        self.assertEqual(report["decision"]["pipeline_phase"], "intake_validation")
        self.assertEqual(report["decision_summary"]["block_category"], "structural_failure")
        self.assertIsNotNone(report["analysis"]["input_state_hash"])
        self.assertEqual(view_model["verification"]["headline"], "Structural BLOCKED")
        self.assertEqual(view_model["slices"], [])
        self.assertEqual(
            pacs_block["verification_summary"]["canonical_decision"]["decision_class"],
            "blocked_structural",
        )

    def test_structural_block_dicom_input_and_trace_remain_available(self) -> None:
        state = build_demo_case()
        state["verification_profile_descriptor"] = {"profile_name": "does_not_exist"}

        dicom_input = build_mock_dicom_input(state)
        pacs_block = build_mock_pacs_block_notice(state)
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_block,
        )

        self.assertEqual(dicom_input["kind"], "mock_dicom_input")
        self.assertEqual(dicom_input["schema_version"], DICOM_MODEL_SCHEMA_VERSION)
        self.assertEqual(
            dicom_input["study"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE]["source_case_id"],
            state["case_id"],
        )
        self.assertNotIn("ai_claim_envelope", dicom_input)
        self.assertEqual(dicom_input["debug"]["guardian_case_snapshot"], state)
        self.assertEqual(
            integration_trace["pipeline"][1]["canonical_decision"]["decision_class"],
            "blocked_structural",
        )

    def test_structural_validity_failure_uses_minimal_report_floor(self) -> None:
        state = build_demo_case()
        state["case_id"] = ""

        report = build_case_verification_report(state)
        pacs_block = build_mock_pacs_block_notice(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
        self.assertEqual(report["decision"]["pipeline_phase"], "structural_validity")
        self.assertEqual(
            report["verification_pipeline"]["phases"]["proof_core"]["status"],
            "not_evaluated",
        )
        self.assertEqual(
            pacs_block["study"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE]["source_case_id"],
            "unknown-case",
        )
        self.assertEqual(
            pacs_block["verification_summary"]["canonical_decision"]["decision_class"],
            "blocked_structural",
        )

    def test_structural_block_dicom_input_accepts_partial_case_state(self) -> None:
        state = build_demo_case()
        state.pop("series_uid")

        dicom_input = build_mock_dicom_input(state)
        pacs_block = build_mock_pacs_block_notice(state)
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_block,
        )

        self.assertEqual(
            dicom_input["source_series"][0]["series"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE][
                "source_series_uid"
            ],
            "unknown-series",
        )
        self.assertEqual(
            integration_trace["pipeline"][0]["series_uid"],
            "unknown-series",
        )
        self.assertEqual(
            integration_trace["pipeline"][1]["canonical_decision"]["decision_class"],
            "blocked_structural",
        )

    def test_internal_exception_during_structural_block_construction_still_raises(self) -> None:
        state = build_demo_case()
        state["case_id"] = ""

        with patch("mri_guardian.harness._build_structural_block_report", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                build_case_verification_report(state)

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

    def test_derive_verified_mask_rejects_non_finite_reconstruction_values(self) -> None:
        threshold_rules = {
            "algorithm": THRESHOLD_ALGORITHM,
            "parameters": {"threshold": 2.5},
        }

        for value in (np.nan, np.inf, -np.inf):
            reconstruction = np.zeros((1, 1, 1), dtype=np.float64)
            reconstruction[0, 0, 0] = value

            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    derive_verified_mask(reconstruction, threshold_rules)

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
            morphology_report["manifest"]["segmentation_policy_proof"]["comparison"][
                "voxel_delta"
            ],
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
            component_report["manifest"]["segmentation_policy_proof"]["comparison"][
                "voxel_delta"
            ],
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
            slice_report["manifest"]["segmentation_policy_proof"]["comparison"][
                "voxel_delta"
            ],
            0,
        )

    def test_mock_dicom_input_round_trips_guardian_case_state(self) -> None:
        state = build_demo_case()

        dicom_input = build_mock_dicom_input(state)
        restored_state = restore_case_state_from_dicom_input(dicom_input)
        reconstruction_group = next(
            group
            for group in dicom_input["source_series"]
            if group["series"]["semantic_role"] == "source_reconstruction_series"
        )
        claim_group = next(
            group
            for group in dicom_input["source_series"]
            if group["series"]["semantic_role"] == "source_claim_series"
        )

        self.assertEqual(restored_state, state)
        self.assertEqual(dicom_input["schema_version"], DICOM_MODEL_SCHEMA_VERSION)
        self.assertEqual(dicom_input["endpoint"]["destination_ae_title"], "MRI-GUARDIAN-LAYER")
        self.assertEqual(
            dicom_input["study"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE]["source_case_id"],
            state["case_id"],
        )
        self.assertEqual(
            reconstruction_group["acquisition_provenance"][0]["object_kind"],
            "acquisition_provenance",
        )
        self.assertEqual(
            claim_group["instances"][0]["document_kind"],
            "ai_volume_claim_sr_like",
        )
        self.assertEqual(
            dicom_input["debug"]["guardian_case_snapshot"]["verification_profile_descriptor"][
                "profile_name"
            ],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(
            dicom_input["integration_narrative"]["hospital_placement"]["position"],
            "between_reconstruction_ai_and_pacs",
        )

    def test_mock_dicom_input_round_trips_optional_ai_stack_identity(self) -> None:
        state = self._attach_ai_stack_identity(build_demo_case())

        dicom_input = build_mock_dicom_input(state)
        restored_state = restore_case_state_from_dicom_input(dicom_input)
        claim_group = next(
            group
            for group in dicom_input["source_series"]
            if group["series"]["semantic_role"] == "source_claim_series"
        )

        self.assertEqual(restored_state, state)
        self.assertEqual(
            dicom_input["debug"]["guardian_case_snapshot"]["ai_stack_identity"],
            state["ai_stack_identity"],
        )
        self.assertNotIn("ai_stack_identity", canonical_json_dumps(claim_group))

    def test_mock_pacs_submission_exposes_verified_handoff_contract(self) -> None:
        _, result = self._release_demo_case()

        dicom_input = build_mock_dicom_input(result.before_state)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference="mri-case-0001.guardian-bundle",
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
        self.assertEqual(
            [group["series"]["semantic_role"] for group in pacs_submission["released_series"]],
            [
                "verified_reconstruction_series",
                "verified_segmentation_series",
                "guardian_verification_manifest_series",
            ],
        )
        self.assertEqual(
            pacs_submission["sidecars"][0]["sidecar_type"],
            SIDECAR_TYPE_ICSD_EVIDENCE_JSON,
        )
        self.assertEqual(
            pacs_submission["integration_narrative"]["failure_mode"]["current_guardian_decision"],
            "released",
        )
        self.assertEqual(integration_trace["pipeline"][0]["label"], "DICOM Input")
        self.assertEqual(integration_trace["pipeline"][1]["decision_class"], "final_release_candidate")
        self.assertEqual(integration_trace["pipeline"][2]["label"], "PACS Output")

    def test_dicom_uid_generation_is_stable_and_separates_object_identity_classes(self) -> None:
        shared_seed = {"source_case_id": "case-1", "source_study_key": "study-1"}
        study_uid = deterministic_uid(object_kind="study", seed=shared_seed)
        same_study_uid = deterministic_uid(
            object_kind="study",
            seed={"source_study_key": "study-1", "source_case_id": "case-1"},
        )
        series_uid = deterministic_uid(object_kind="series", seed=shared_seed)
        child_uid = deterministic_uid(
            object_kind="instance",
            seed={
                "parent_series_uid": series_uid,
                "semantic_role": "verified_reconstruction_image",
                "content_hash": "a" * 64,
            },
        )
        different_parent_child_uid = deterministic_uid(
            object_kind="instance",
            seed={
                "parent_series_uid": deterministic_uid(
                    object_kind="series",
                    seed={"source_case_id": "case-1", "source_study_key": "study-2"},
                ),
                "semantic_role": "verified_reconstruction_image",
                "content_hash": "a" * 64,
            },
        )

        self.assertEqual(study_uid, same_study_uid)
        self.assertNotEqual(study_uid, series_uid)
        self.assertNotEqual(child_uid, different_parent_child_uid)

    def test_mock_dicom_input_rejects_guardian_fields_outside_extension_namespace(self) -> None:
        dicom_input = build_mock_dicom_input(build_demo_case())
        dicom_input["study"]["case_kind"] = "single_series_v1"

        with self.assertRaisesRegex(ValueError, "outside the Guardian extension container"):
            restore_case_state_from_dicom_input(dicom_input)

    def test_mock_dicom_input_rejects_unknown_guardian_extension_keys(self) -> None:
        dicom_input = build_mock_dicom_input(build_demo_case())
        dicom_input["study"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE]["unexpected"] = "value"

        with self.assertRaisesRegex(ValueError, "contains unsupported fields"):
            restore_case_state_from_dicom_input(dicom_input)

    def test_restore_case_state_requires_quarantined_debug_snapshot_only(self) -> None:
        dicom_input = build_mock_dicom_input(build_demo_case())
        reviewer_surface = {key: value for key, value in dicom_input.items() if key != "debug"}

        self.assertNotIn("guardian_case_snapshot", canonical_json_dumps(reviewer_surface))
        dicom_input["debug"] = {}

        with self.assertRaisesRegex(ValueError, "missing debug.guardian_case_snapshot"):
            restore_case_state_from_dicom_input(dicom_input)

    def test_mock_dicom_input_preserves_frame_of_reference_honesty(self) -> None:
        state = self._attach_temporal_metadata(
            build_demo_case(),
            study_acquired_at="2025-01-01T00:00:00Z",
            study_timestamp_provenance="trusted_study_metadata",
        )

        dicom_input = build_mock_dicom_input(state)
        reconstruction_group = next(
            group
            for group in dicom_input["source_series"]
            if group["series"]["semantic_role"] == "source_reconstruction_series"
        )
        segmentation_group = next(
            group
            for group in dicom_input["source_series"]
            if group["series"]["semantic_role"] == "source_segmentation_series"
        )

        self.assertNotIn("frame_of_reference_uid", reconstruction_group["series"])
        self.assertNotIn("frame_of_reference_uid", segmentation_group["series"])
        self.assertEqual(
            reconstruction_group["lineage"]["frame_of_reference_status"],
            "upstream_non_dicom_uid",
        )
        self.assertEqual(
            segmentation_group["lineage"]["frame_of_reference_status"],
            "upstream_non_dicom_uid",
        )

    def test_single_series_proof_contracts_include_ai_claim_envelope(self) -> None:
        state = self._attach_ai_stack_identity(build_demo_case())
        result = self._build_harness().release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )

        pacs_submission = build_mock_pacs_submission(result)
        structural_block_state = build_demo_case()
        structural_block_state["verification_profile_descriptor"] = {
            "profile_name": "does_not_exist"
        }
        pacs_block = build_mock_pacs_block_notice(structural_block_state)

        self.assertIn("ai_claim_envelope", pacs_submission["proof_contracts"])
        self.assertEqual(
            pacs_submission["proof_contracts"]["ai_claim_envelope"]["ai_stack_identity"],
            state["ai_stack_identity"],
        )
        self.assertNotIn("ai_claim_envelope", pacs_block["proof_contracts"])

    def test_ai_runtime_context_remains_integration_only(self) -> None:
        runtime_context = self._ai_runtime_context()
        state = self._attach_ai_stack_identity(build_demo_case())
        dicom_input = build_mock_dicom_input(
            state,
            ai_runtime_context=runtime_context,
        )
        result = self._build_harness().release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        pacs_submission = build_mock_pacs_submission(
            result,
            ai_runtime_context=runtime_context,
        )
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_submission,
        )
        restored_state = restore_case_state_from_dicom_input(dicom_input)
        report = build_case_verification_report(result.after_state)

        self.assertEqual(dicom_input["ai_runtime_context"], runtime_context)
        self.assertEqual(pacs_submission["ai_runtime_context"], runtime_context)
        self.assertEqual(integration_trace["ai_runtime_context"], runtime_context)
        self.assertNotIn("ai_runtime_context", restored_state)
        self.assertNotIn("ai_runtime_context", result.after_state)
        self.assertNotIn("ai_runtime_context", result.after_state["verified_manifest"])
        self.assertNotIn("ai_runtime_context", report["contracts"]["ai_claim_envelope"])

    def test_ai_runtime_context_validation_rejects_malformed_payloads(self) -> None:
        invalid_contexts = (
            {"inference_runtime_version": "guardian-runtime/2.3.4"},
            {
                "inference_runtime_version": "guardian-runtime/2.3.4",
                "inference_device_class": "",
            },
        )

        for invalid_context in invalid_contexts:
            with self.subTest(invalid_context=invalid_context):
                with self.assertRaisesRegex(ValueError, "ai_runtime_context"):
                    build_mock_dicom_input(
                        build_demo_case(),
                        ai_runtime_context=invalid_context,
                    )

    def test_ai_runtime_context_accepts_mapping_inputs_on_blocked_pacs_flow(self) -> None:
        runtime_context = OrderedDict(
            [
                ("inference_device_class", "cuda_sm80"),
                ("inference_runtime_version", "guardian-runtime/2.3.4"),
            ]
        )
        structural_block_state = build_demo_case()
        structural_block_state["verification_profile_descriptor"] = {
            "profile_name": "does_not_exist"
        }

        dicom_input = build_mock_dicom_input(
            structural_block_state,
            ai_runtime_context=runtime_context,
        )
        pacs_block = build_mock_pacs_block_notice(
            structural_block_state,
            ai_runtime_context=runtime_context,
        )
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_block,
        )

        self.assertEqual(pacs_block["ai_runtime_context"], self._ai_runtime_context())
        self.assertEqual(integration_trace["ai_runtime_context"], self._ai_runtime_context())
        self.assertNotIn("ai_runtime_context", pacs_block["proof_contracts"])

    def test_boundary_release_surfaces_canonical_reason_in_viewer_and_pacs(self) -> None:
        state = build_demo_case()
        state["tumor_volume_mm3"] = round(float(state["tumor_volume_mm3"]) + 8.5e-10, 9)

        view_model = build_side_by_side_view_model(state)
        result = self._build_harness().release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        pacs_submission = build_mock_pacs_submission(result)

        self.assertEqual(view_model["decision"]["decision_class"], "verified_boundary_hugging")
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED NEAR BOUNDARY")
        self.assertEqual(view_model["verification"]["primary_reason_summary"], "proof_metric_near_boundary")
        self.assertEqual(
            pacs_submission["verification_summary"]["primary_reason"],
            "proof_metric_near_boundary",
        )

    def test_advisory_release_surfaces_canonical_reason_in_viewer_and_pacs(self) -> None:
        state = self._build_warning_stack_case()

        view_model = build_side_by_side_view_model(state)
        result = self._build_harness().release_case(
            state=state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        pacs_submission = build_mock_pacs_submission(result)

        self.assertEqual(view_model["decision"]["decision_class"], "verified_with_advisory")
        self.assertEqual(view_model["verification"]["headline"], "Verification PASSED WITH ADVISORY")
        self.assertEqual(
            view_model["verification"]["primary_reason_summary"],
            "clinical_plausibility_non_blocking_finding",
        )
        self.assertEqual(
            pacs_submission["verification_summary"]["primary_reason"],
            "clinical_plausibility_non_blocking_finding",
        )

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
            pacs_block["verification_summary"]["canonical_decision"]["decision_class"],
            "blocked_physical",
        )
        self.assertEqual(
            pacs_block["proof_contracts"]["verification_profile_descriptor"]["profile_name"],
            DEFAULT_PROFILE_NAME,
        )
        self.assertEqual(integration_trace["pipeline"][1]["decision"], "blocked")
        self.assertEqual(integration_trace["pipeline"][2]["label"], "PACS Output Blocked")

    def test_multi_series_duplicate_series_id_is_structural_shape_failure(self) -> None:
        state = build_multi_series_demo_case()
        state["series"][1]["series_id"] = state["series"][0]["series_id"]

        with self.assertRaisesRegex(ValueError, "duplicate payload series_id"):
            build_multi_series_case_verification_report(state)

    def test_multi_series_requires_non_empty_supporting_list(self) -> None:
        state = build_multi_series_demo_case()
        state["cross_series_policy"]["supporting_series_ids"] = []
        state["cross_series_policy"]["required_series_ids"] = ["t1-primary"]

        with self.assertRaisesRegex(ValueError, "supporting_series_ids"):
            build_multi_series_case_verification_report(state)

    def test_multi_series_primary_cannot_appear_in_supporting_series_ids(self) -> None:
        state = build_multi_series_demo_case()
        state["cross_series_policy"]["supporting_series_ids"] = [
            "t1-primary",
            "t2-supporting",
        ]
        state["cross_series_policy"]["required_series_ids"] = [
            "t1-primary",
            "t2-supporting",
        ]

        with self.assertRaisesRegex(ValueError, "must not include primary_series_id"):
            build_multi_series_case_verification_report(state)

    def test_multi_series_demo_case_releases_and_records_study_manifest(self) -> None:
        state, result = self._release_multi_series_demo_case()
        manifest = result.after_state["verified_manifest"]
        workstation_payload = result.after_state["workstation_payload"]

        self.assertEqual(result.after_state["release_status"], RELEASE_STATUS_RELEASED)
        self.assertEqual(manifest["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertEqual(len(manifest["series_manifests"]), 4)
        self.assertEqual(
            [item["resolved_role"] for item in manifest["series_manifests"]],
            ["primary", "supporting", "supporting", "auxiliary"],
        )
        self.assertTrue(manifest["cross_series_verification"]["geometry_verification"]["verified"])
        self.assertTrue(manifest["cross_series_verification"]["lesion_corroboration"]["verified"])
        self.assertEqual(len(workstation_payload["series_payloads"]), 4)
        self.assertEqual(workstation_payload["withheld_series"], [])
        report = build_multi_series_case_verification_report(result.after_state)
        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["contracts"]["verification_profile_descriptor"], report["profile"])

    def test_multi_series_claim_provenance_remains_per_series_only(self) -> None:
        state = build_multi_series_demo_case()
        primary_series = self._series_entry(state, "t1-primary")
        auxiliary_series = self._series_entry(state, "derived-auxiliary")
        self._attach_ai_stack_identity(primary_series)
        self._attach_ai_stack_identity(
            auxiliary_series,
            compatibility_type="sequence_family",
        )

        report = build_multi_series_case_verification_report(state)
        series_reports = {
            item["series_id"]: item
            for item in report["series_reports"]
        }

        self.assertIn(
            "ai_stack_identity",
            series_reports["t1-primary"]["contracts"]["ai_claim_envelope"],
        )
        self.assertNotIn(
            "ai_stack_identity",
            series_reports["t2-supporting"]["contracts"]["ai_claim_envelope"],
        )
        self.assertIn(
            "ai_stack_identity",
            series_reports["derived-auxiliary"]["contracts"]["ai_claim_envelope"],
        )
        self.assertNotIn("ai_claim_envelope", report["contracts"])
        self.assertEqual(report["contracts"]["verification_profile_descriptor"], report["profile"])

    def test_multi_series_geometry_mismatch_blocks_merged_release(self) -> None:
        state = build_multi_series_demo_case()
        supporting = self._series_entry(state, "t2-supporting")
        supporting["voxel_spacing_mm"] = [1.0, 0.8, 2.5]
        supporting["tumor_volume_mm3"] = round(
            float(np.sum(supporting["tumor_mask"]) * np.prod(supporting["voxel_spacing_mm"])),
            9,
        )

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "required_series_geometry_mismatch",
        )
        self.assertIn("t2-supporting", report["failure_reasons"][0]["details"]["mismatched_series_ids"])

    def test_multi_series_missing_required_series_is_reported_explicitly(self) -> None:
        state = build_multi_series_demo_case()
        state["cross_series_policy"]["supporting_series_ids"] = ["t2-supporting", "missing-support"]
        state["cross_series_policy"]["required_series_ids"] = ["t1-primary", "t2-supporting", "missing-support"]

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision_summary"]["block_category"], "missing_required_series")
        self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
        self.assertEqual(
            report["failure_reasons"][0]["details"]["missing_required_series_ids"],
            ["missing-support"],
        )

    def test_multi_series_required_supporting_series_proof_failure_blocks_release(self) -> None:
        state = build_multi_series_demo_case()
        supporting = self._series_entry(state, "t2-supporting")
        supporting["tumor_mask"][1][1][1] = 1
        voxel_spacing = np.asarray(supporting["voxel_spacing_mm"], dtype=np.float64)
        supporting["tumor_volume_mm3"] = round(
            float(np.sum(supporting["tumor_mask"]) * np.prod(voxel_spacing)),
            9,
        )

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "required_series_not_release_eligible",
        )
        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")
        self.assertIn("t2-supporting", report["failure_reasons"][0]["details"]["series_ids"])

    def test_multi_series_primary_positive_requires_at_least_one_supporting_overlap(self) -> None:
        state = build_multi_series_demo_case()

        report = build_multi_series_case_verification_report(state)

        self.assertTrue(report["release_eligible"])
        supporting = report["analysis"]["lesion_corroboration"]["supporting_series"]
        overlap_map = {item["series_id"]: item["intersection_voxel_count"] for item in supporting}
        self.assertGreater(overlap_map["t2-supporting"], 0)
        self.assertEqual(overlap_map["flair-supporting"], 0)

    def test_multi_series_primary_positive_without_any_supporting_overlap_blocks_release(self) -> None:
        state = build_multi_series_demo_case()
        support_shape = np.asarray(self._series_entry(state, "t2-supporting")["ai_reconstruction"]).shape
        t2_reconstruction = np.zeros(support_shape, dtype=np.float64)
        flair_reconstruction = np.zeros(support_shape, dtype=np.float64)
        t2_reconstruction[1][1][1] = 4.0
        flair_reconstruction[1][0][0] = 4.0
        self._rewrite_series_for_profile(state, "t2-supporting", t2_reconstruction)
        self._rewrite_series_for_profile(state, "flair-supporting", flair_reconstruction)

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "positive_lesion_corroboration_failed",
        )
        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")
        supporting = report["analysis"]["lesion_corroboration"]["supporting_series"]
        self.assertTrue(report["analysis"]["lesion_corroboration"]["evaluated"])
        self.assertTrue(all(item["intersection_voxel_count"] == 0 for item in supporting))

    def test_multi_series_supporting_only_lesion_blocks_release(self) -> None:
        state = build_multi_series_demo_case()
        primary_shape = np.asarray(self._series_entry(state, "t1-primary")["ai_reconstruction"]).shape
        self._rewrite_series_for_profile(
            state,
            "t1-primary",
            np.zeros(primary_shape, dtype=np.float64),
        )

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "positive_lesion_corroboration_failed",
        )
        self.assertIn("primary verified mask is empty", report["failure_reasons"][-1]["message"])

    def test_multi_series_all_required_empty_masks_pass_corroboration(self) -> None:
        state = build_multi_series_demo_case()
        for series_id in ("t1-primary", "t2-supporting", "flair-supporting"):
            shape = np.asarray(self._series_entry(state, series_id)["ai_reconstruction"]).shape
            self._rewrite_series_for_profile(
                state,
                series_id,
                np.zeros(shape, dtype=np.float64),
            )

        report = build_multi_series_case_verification_report(state)

        self.assertTrue(report["release_eligible"])
        self.assertTrue(report["analysis"]["lesion_corroboration"]["verified"])

    def test_multi_series_auxiliary_failure_is_reported_but_non_gating(self) -> None:
        state = build_multi_series_demo_case()
        auxiliary = self._series_entry(state, "derived-auxiliary")
        auxiliary["tumor_mask"][0][0][0] = 1
        voxel_spacing = np.asarray(auxiliary["voxel_spacing_mm"], dtype=np.float64)
        auxiliary["tumor_volume_mm3"] = round(float(np.sum(auxiliary["tumor_mask"]) * np.prod(voxel_spacing)), 9)

        report = build_multi_series_case_verification_report(state)
        _, result = self._release_multi_series_demo_case(state=state)

        self.assertTrue(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "verified_with_advisory")
        auxiliary_report = next(
            item for item in report["series_reports"] if item["series_id"] == "derived-auxiliary"
        )
        self.assertFalse(auxiliary_report["series_release_eligible"])
        self.assertFalse(auxiliary_report["downstream_release_allowed"])
        self.assertEqual(len(result.after_state["workstation_payload"]["series_payloads"]), 3)
        self.assertEqual(
            result.after_state["workstation_payload"]["withheld_series"][0]["series_id"],
            "derived-auxiliary",
        )

    def test_multi_series_auxiliary_geometry_and_lesion_do_not_gate_release(self) -> None:
        state = build_multi_series_demo_case()
        auxiliary = self._series_entry(state, "derived-auxiliary")
        auxiliary["voxel_spacing_mm"] = [1.0, 0.8, 2.5]
        lesion = np.full((2, 2, 2), 40.0, dtype=np.float64)
        self._rewrite_series_for_profile(state, "derived-auxiliary", lesion)

        report = build_multi_series_case_verification_report(state)

        self.assertTrue(report["release_eligible"])
        auxiliary_report = next(
            item for item in report["series_reports"] if item["series_id"] == "derived-auxiliary"
        )
        self.assertEqual(auxiliary_report["resolved_role"], "auxiliary")
        self.assertTrue(auxiliary_report["series_release_eligible"])

    def test_multi_series_required_child_class_precedence_prefers_structural_over_policy(self) -> None:
        state = build_multi_series_demo_case()
        primary = self._series_entry(state, "t1-primary")
        primary["raw_real"] = [0.0]
        supporting = self._series_entry(state, "t2-supporting")
        supporting["verification_profile_descriptor"] = build_verification_profile_descriptor(
            "fast_triage_mode"
        )

        report = build_multi_series_case_verification_report(state)

        self.assertFalse(report["release_eligible"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "required_series_not_release_eligible",
        )
        self.assertEqual(report["decision"]["decision_class"], "blocked_structural")
        self.assertIn(
            "required_series_t1-primary_blocked_structural",
            report["decision"]["reason_codes"],
        )

    def test_multi_series_viewer_preserves_ordered_role_annotations(self) -> None:
        state = build_multi_series_demo_case()

        view_model = build_side_by_side_view_model(state)
        html = render_side_by_side_report_html(
            state,
            report_title="Multi-Series Study Verification",
        )

        self.assertEqual(view_model["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertEqual(
            [item["series_id"] for item in view_model["series"]],
            ["t1-primary", "t2-supporting", "flair-supporting", "derived-auxiliary"],
        )
        self.assertEqual(
            [item["resolved_role"] for item in view_model["series"]],
            ["primary", "supporting", "supporting", "auxiliary"],
        )
        self.assertIn("Study Gate", html)
        self.assertIn("derived-auxiliary", html)
        self.assertIn("supporting", html)
        self.assertIn("2x2x2 @ 0.8/0.8/2.5 mm", html)
        self.assertNotIn("{&#x27;shape&#x27;:", html)

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
        self.assertEqual(view_model["decision"]["decision_class"], "verified_clean")
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
        self.assertIn("Level 2: Proof Core", html)
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

    def test_measurement_domain_tamper_fixes_primary_proof_stage(self) -> None:
        entry = next(
            item
            for item in build_verification_threat_corpus()["entries"]
            if item["slug"] == "measurement_domain_tamper"
        )
        report = build_case_verification_report(
            entry["state"],
            profile_name=entry["profile_name"],
        )

        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")
        self.assertEqual(report["decision"]["pipeline_phase"], "proof_core")
        self.assertEqual(report["decision"]["proof_stage"], "MeasurementIntegrityProof")
        self.assertEqual(report["failure_reasons"][0]["code"], "raw_measurement_inconsistency")
        self.assertEqual(
            report["verification_pipeline"]["phases"]["proof_core"]["proof_stages"][0][
                "proof_stage"
            ],
            "MeasurementIntegrityProof",
        )

    def test_verification_threat_corpus_lists_required_entries_with_valid_metadata(
        self,
    ) -> None:
        corpus = build_verification_threat_corpus()
        entries = {entry["slug"]: entry for entry in corpus["entries"]}

        self.assertEqual(corpus["interface_type"], "VerificationThreatCorpus")
        self.assertEqual(corpus["revision"], "v1")
        self.assertIn("qualitative", corpus["hazard_severity_note"])
        self.assertIn("Normative invariants", corpus["expectation_model"]["normative"])
        self.assertIn(
            "Observational expectations",
            corpus["expectation_model"]["observational"],
        )
        self.assertEqual(
            set(entries),
            {
                "fake_tumour",
                "shifted_mask",
                "measurement_domain_tamper",
                "intensity_manipulation",
                "subtle_distributed_perturbation",
                "structured_ringing_artefact",
                "slice_local_anomaly_injection",
                "mask_erosion",
                "mask_dilation",
                "volume_rounding_attack",
                "mixed_frequency_perturbation",
                "stale_mask_replay",
                "cross_series_misregistration",
                "component_micro_island",
                "slice_single_slice_island",
            },
        )
        self.assertEqual(
            len({entry["corpus_id"] for entry in corpus["entries"]}),
            len(corpus["entries"]),
        )
        self.assertFalse(
            any(entry["corpus_status"] == "deferred" for entry in corpus["entries"])
        )

        for entry in corpus["entries"]:
            with self.subTest(slug=entry["slug"]):
                self.assertTrue(entry["corpus_id"].startswith("VFTC-"))
                self.assertIn(
                    entry["scenario_surface"],
                    {
                        "single_series_reconstruction",
                        "single_series_mask",
                        "single_series_quantification",
                        "multi_series_corroboration",
                        "state_reuse",
                        "profile_scoped_mask_policy",
                    },
                )
                self.assertIn(
                    entry["corpus_status"],
                    {"active", "profile_scoped", "deferred"},
                )
                self.assertIn(
                    entry["hazard_severity"],
                    {"minor", "moderate", "major", "critical"},
                )
                contract = entry["determinism_contract"]
                self.assertIn("builder_id", contract)
                self.assertIn("fixed_inputs", contract)
                self.assertIn("seed_strategy", contract)
                self.assertIn(
                    entry["pipeline_phase"],
                    {"proof_core", "multi_series_aggregation"},
                )
                if entry["pipeline_phase"] == "proof_core":
                    self.assertIn(
                        entry["proof_stage"],
                        {
                            "MeasurementIntegrityProof",
                            "ReconstructionParityProof",
                            "SegmentationPolicyProof",
                            "VolumeComputationProof",
                        },
                    )
                else:
                    self.assertNotIn("proof_stage", entry)
                roles = {item["control_role"] for item in entry["mitigation_mapping"]}
                if entry["corpus_status"] == "active":
                    self.assertTrue(
                        roles & {"detective", "preventative"},
                        msg=f"{entry['slug']} missing detective/preventative control",
                    )
                    self.assertTrue(
                        roles & {"gating", "evidentiary"},
                        msg=f"{entry['slug']} missing gating/evidentiary control",
                    )
        self.assertEqual(entries["component_micro_island"]["corpus_status"], "profile_scoped")
        self.assertEqual(
            entries["slice_single_slice_island"]["corpus_status"], "profile_scoped"
        )
        for slug in entries:
            if slug not in {"component_micro_island", "slice_single_slice_island"}:
                self.assertEqual(entries[slug]["corpus_status"], "active")

    def test_verification_threat_corpus_compatibility_subset_preserves_current_slugs(
        self,
    ) -> None:
        self.assertEqual(
            tuple(item["slug"] for item in build_tamper_scenarios()),
            ("fake_tumour", "shifted_mask", "intensity_manipulation"),
        )
        self.assertEqual(
            tuple(
                item["slug"]
                for item in build_tamper_scenarios(
                    profile_name="brain_t1_component_review"
                )
            ),
            (
                "fake_tumour",
                "shifted_mask",
                "intensity_manipulation",
                "component_micro_island",
            ),
        )
        self.assertEqual(
            tuple(
                item["slug"]
                for item in build_tamper_scenarios(profile_name="brain_t1_slice_review")
            ),
            (
                "fake_tumour",
                "shifted_mask",
                "intensity_manipulation",
                "slice_single_slice_island",
            ),
        )

    def test_verification_threat_corpus_entries_follow_expected_fail_closed_routes(
        self,
    ) -> None:
        route_by_decision_class = {
            "blocked_structural": "structural",
            "blocked_physical": "physical",
            "blocked_policy": "policy",
        }
        for entry in build_verification_threat_corpus()["entries"]:
            with self.subTest(slug=entry["slug"]):
                if entry["state"].get("case_kind") == MULTI_SERIES_CASE_KIND:
                    report = build_multi_series_case_verification_report(entry["state"])
                else:
                    report = build_case_verification_report(
                        entry["state"],
                        profile_name=entry["profile_name"],
                    )
                self.assertEqual(
                    report["decision"]["decision_class"],
                    entry["normative_invariants"]["expected_decision_class"],
                )
                self.assertEqual(
                    route_by_decision_class[report["decision"]["decision_class"]],
                    entry["normative_invariants"]["expected_failure_route"],
                )
                self.assertEqual(
                    report["decision"]["pipeline_phase"],
                    entry["pipeline_phase"],
                )
                if "proof_stage" in entry:
                    self.assertEqual(report["decision"]["proof_stage"], entry["proof_stage"])
                else:
                    self.assertNotIn("proof_stage", report["decision"])
                expected_block_category = entry["observational_expectations"].get(
                    "expected_block_category"
                )
                if expected_block_category is not None:
                    self.assertEqual(
                        report["decision_summary"]["block_category"],
                        expected_block_category,
                    )
                expected_reason_family = entry["observational_expectations"].get(
                    "expected_primary_reason_family"
                )
                if expected_reason_family is not None:
                    self.assertEqual(
                        report["failure_reasons"][0]["code"],
                        expected_reason_family,
                    )

    def test_verification_threat_corpus_reason_family_is_only_used_for_durable_subset(
        self,
    ) -> None:
        entries = build_verification_threat_corpus()["entries"]
        actual = {
            entry["slug"]
            for entry in entries
            if "expected_primary_reason_family"
            in entry["observational_expectations"]
        }
        self.assertEqual(
            actual,
            {
                "fake_tumour",
                "shifted_mask",
                "measurement_domain_tamper",
                "intensity_manipulation",
                "subtle_distributed_perturbation",
                "mask_erosion",
                "mask_dilation",
                "volume_rounding_attack",
                "stale_mask_replay",
                "cross_series_misregistration",
                "component_micro_island",
                "slice_single_slice_island",
            },
        )

    def test_verification_threat_corpus_stale_mask_replay_stays_in_semantic_verification(
        self,
    ) -> None:
        entry = next(
            item
            for item in build_verification_threat_corpus()["entries"]
            if item["slug"] == "stale_mask_replay"
        )
        report = build_case_verification_report(
            entry["state"],
            profile_name=entry["profile_name"],
        )

        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")
        self.assertEqual(report["decision"]["pipeline_phase"], "proof_core")
        self.assertEqual(report["decision"]["proof_stage"], "SegmentationPolicyProof")
        self.assertIn(
            {"pipeline_phase": "structural_validity", "outcome": "passed"},
            report["decision"]["decision_lineage"],
        )
        self.assertEqual(report["failure_reasons"][0]["code"], "mask_voxel_mismatch")
        self.assertIn("Not manifest tampering", entry["provenance_scope"])

    def test_verification_threat_corpus_cross_series_misregistration_preserves_geometry(
        self,
    ) -> None:
        entry = next(
            item
            for item in build_verification_threat_corpus()["entries"]
            if item["slug"] == "cross_series_misregistration"
        )
        report = build_multi_series_case_verification_report(entry["state"])

        self.assertTrue(report["analysis"]["geometry_verification"]["verified"])
        self.assertFalse(report["analysis"]["lesion_corroboration"]["verified"])
        self.assertEqual(
            report["decision_summary"]["block_category"],
            "positive_lesion_corroboration_failed",
        )
        self.assertEqual(report["decision"]["decision_class"], "blocked_physical")

    def test_run_demo_tamper_mode_emits_verification_threat_corpus_manifest(self) -> None:
        demo_root = self.runtime_root / "threat_corpus_demo"

        run_demo(
            output_dir=demo_root,
            profile_name="brain_t1_baseline",
            tamper_mode=True,
        )

        corpus_manifest = json.loads(
            (demo_root / "verification_threat_corpus.json").read_text(
                encoding="utf-8"
            )
        )
        entries = {item["slug"]: item for item in corpus_manifest["entries"]}

        self.assertIn("structured_ringing_artefact", entries)
        self.assertTrue(entries["fake_tumour"]["legacy_curated_subset"])
        self.assertFalse(entries["structured_ringing_artefact"]["legacy_curated_subset"])
        self.assertTrue(
            (demo_root / entries["structured_ringing_artefact"]["artifact_paths"]["html"]).exists()
        )
        self.assertTrue((demo_root / "tamper_cross_series_misregistration.html").exists())

    def test_docs_reference_threat_corpus_and_canonical_ids(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        security = Path("docs/SECURITY.md").read_text(encoding="utf-8")
        clinical = Path("docs/CLINICAL_RISK_ASSURANCE_MAPPING.md").read_text(
            encoding="utf-8"
        )
        corpus_doc = Path("docs/VERIFICATION_THREAT_CORPUS.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("VERIFICATION_THREAT_CORPUS.md", readme)
        self.assertIn("VERIFICATION_THREAT_CORPUS.md", security)
        self.assertIn("VFTC-001", clinical)
        self.assertIn("VFTC-012", corpus_doc)
        self.assertIn("verification_threat_corpus.json", corpus_doc)
        self.assertIn("0013-formal-verification-threat-corpus-v1.md", corpus_doc)
        self.assertIn(
            "0013-formal-verification-threat-corpus-v1.md",
            Path("docs/adr/README.md").read_text(encoding="utf-8"),
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

    def test_timing_profile_accessors_return_isolated_snapshots(self) -> None:
        profile = build_guardian_performance_profile("brain_t2_baseline")
        snapshot = build_timing_overlay_snapshot("brain_t2_baseline")

        self.assertEqual(profile, snapshot)
        self.assertIsNot(profile, snapshot)
        self.assertIsNot(profile["benchmarks"], snapshot["benchmarks"])
        self.assertIsNot(profile["benchmarks"][0], snapshot["benchmarks"][0])

        profile["benchmarks"][0]["label"] = "tampered profile label"
        profile["streaming_strategy"]["recommended_slab_depth_slices"] = 99
        snapshot["budget_assessment"]["representative_total_ms"]["within_budget"] = False

        fresh_profile = build_guardian_performance_profile("brain_t2_baseline")
        fresh_snapshot = build_timing_overlay_snapshot("brain_t2_baseline")

        self.assertNotEqual(
            fresh_profile["benchmarks"][0]["label"],
            "tampered profile label",
        )
        self.assertEqual(
            fresh_profile["streaming_strategy"]["recommended_slab_depth_slices"],
            4,
        )
        self.assertEqual(
            fresh_snapshot["budget_assessment"]["representative_total_ms"]["within_budget"],
            fresh_profile["budget_assessment"]["representative_total_ms"]["within_budget"],
        )
        self.assertEqual(fresh_profile, fresh_snapshot)

    def test_transparency_log_serialises_threaded_appends(self) -> None:
        log = _BlockingReadTransparencyLog(self.runtime_root / "threaded-transparency-log.jsonl")
        errors: list[BaseException] = []
        second_append_done = threading.Event()

        def append_payload(case_id: str, *, done_event: threading.Event | None = None) -> None:
            try:
                log.append(evidence_payload={"case_id": case_id})
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)
            finally:
                if done_event is not None:
                    done_event.set()

        log.arm_next_read()
        first_writer = threading.Thread(target=append_payload, args=("thread-one",))
        second_writer = threading.Thread(
            target=append_payload,
            args=("thread-two",),
            kwargs={"done_event": second_append_done},
        )

        first_writer.start()
        self.assertTrue(log.read_entered.wait(timeout=5))
        second_writer.start()

        time.sleep(0.1)
        self.assertFalse(second_append_done.is_set())

        log.release_read.set()
        first_writer.join(timeout=5)
        second_writer.join(timeout=5)

        self.assertFalse(first_writer.is_alive())
        self.assertFalse(second_writer.is_alive())
        self.assertFalse(errors, repr(errors))

        entries = log._read_entries()

        self.assertEqual([entry["index"] for entry in entries], [0, 1])
        self.assertIsNone(log._verify_chain(entries))

    def test_transparency_log_serialises_process_appends(self) -> None:
        log_path = self.runtime_root / "process-transparency-log.jsonl"
        start_path = self.runtime_root / "start-process-appends.signal"
        child_script = (
            "import json\n"
            "import sys\n"
            "import time\n"
            "from pathlib import Path\n"
            "from mri_guardian.transparency import JsonlHashChainTransparencyLog\n"
            "original = JsonlHashChainTransparencyLog._read_entries\n"
            "def delayed_read(self):\n"
            "    time.sleep(0.25)\n"
            "    return original(self)\n"
            "JsonlHashChainTransparencyLog._read_entries = delayed_read\n"
            "log_path = Path(sys.argv[1])\n"
            "start_path = Path(sys.argv[2])\n"
            "payload = json.loads(sys.argv[3])\n"
            "while not start_path.exists():\n"
            "    time.sleep(0.01)\n"
            "receipt = JsonlHashChainTransparencyLog(log_path).append(evidence_payload=payload)\n"
            "print(json.dumps(receipt))\n"
        )
        commands = [
            [
                sys.executable,
                "-c",
                child_script,
                str(log_path),
                str(start_path),
                json.dumps({"case_id": "process-one"}),
            ],
            [
                sys.executable,
                "-c",
                child_script,
                str(log_path),
                str(start_path),
                json.dumps({"case_id": "process-two"}),
            ],
        ]
        processes = [
            subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for command in commands
        ]

        time.sleep(0.1)
        start_path.write_text("go\n", encoding="utf-8")
        outputs = [process.communicate(timeout=10) for process in processes]

        for process, (_, stderr) in zip(processes, outputs):
            self.assertEqual(process.returncode, 0, stderr)

        entries = JsonlHashChainTransparencyLog(log_path)._read_entries()

        self.assertEqual([entry["index"] for entry in entries], [0, 1])
        self.assertIsNone(JsonlHashChainTransparencyLog(log_path)._verify_chain(entries))

    def test_transparency_log_verify_waits_for_locked_snapshot(self) -> None:
        log = _BlockingReadTransparencyLog(self.runtime_root / "snapshot-transparency-log.jsonl")
        baseline_payload = {"case_id": "baseline"}
        log.append(evidence_payload=baseline_payload)

        errors: list[BaseException] = []
        verify_done = threading.Event()
        verify_result: dict[str, object] = {}

        def append_payload() -> None:
            try:
                log.append(evidence_payload={"case_id": "next"})
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)

        def verify_payload() -> None:
            try:
                verify_result.update(log.verify(evidence_payload=baseline_payload))
            except BaseException as exc:  # pragma: no cover - test plumbing
                errors.append(exc)
            finally:
                verify_done.set()

        log.arm_next_read()
        writer = threading.Thread(target=append_payload)
        writer.start()
        self.assertTrue(log.read_entered.wait(timeout=5))

        verifier = threading.Thread(target=verify_payload)
        verifier.start()

        time.sleep(0.1)
        self.assertFalse(verify_done.is_set())

        log.release_read.set()
        writer.join(timeout=5)
        verifier.join(timeout=5)

        self.assertFalse(writer.is_alive())
        self.assertFalse(verifier.is_alive())
        self.assertFalse(errors, repr(errors))
        self.assertTrue(verify_result["verified"])

    def test_portable_bundle_round_trips_and_replays_identically(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"baseline{GUARDIAN_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)
        expected_report = build_case_verification_report(result.after_state)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        with ZipFile(bundle_path, mode="r") as archive:
            manifest = self._read_bundle_json(archive, "bundle.json")
            decision_journal = self._read_bundle_json(archive, "decision_journal.json")
        self.assertEqual(manifest["bundle_type"], "dpa-guardian-bundle")
        self.assertEqual(manifest["bundle_schema_version"], "1")
        self.assertEqual(manifest["replay_inputs"]["profile_name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(
            manifest["proof_context"]["policy_identity"]["verification_profile_descriptor_hash"],
            hash_canonical_json(expected_report["contracts"]["verification_profile_descriptor"]),
        )
        self.assertEqual(
            manifest["proof_context"]["derived_traces"]["proof_core_evidence_trace"],
            expected_report["contracts"]["proof_core_evidence_trace"],
        )
        self.assertEqual(decision_journal["entries"][-1]["status"], "recorded")
        self.assertEqual(
            manifest["integrity_manifest"]["member_order"],
            [
                "bundle.json",
                "input_state.json",
                "released_state.json",
                "workstation_payload.json",
                "verification_report.json",
                "decision_journal.json",
                "evidence.json",
                "transparency-prefix.jsonl",
            ],
        )
        self.assertEqual(manifest["integrity_manifest"]["member_hashes"][0]["algorithm"], "sha256")

    def test_ai_stack_identity_changes_bundle_and_replay_hash_surfaces(self) -> None:
        _, baseline_result = self._release_demo_case()
        provenance_state = self._attach_ai_stack_identity(build_demo_case())
        provenance_result = self._build_harness().release_case(
            state=provenance_state,
            actor="guardian:test",
            authority_sources=build_demo_authority_sources(case_id=provenance_state["case_id"]),
            timestamp="2026-03-26T00:00:00Z",
        )
        baseline_bundle_path = self.runtime_root / f"baseline-provenance{GUARDIAN_BUNDLE_EXTENSION}"
        provenance_bundle_path = self.runtime_root / f"declared-provenance{GUARDIAN_BUNDLE_EXTENSION}"
        transparency_log_path = self.runtime_root / "transparency-log.jsonl"

        write_portable_evidence_bundle(
            result=baseline_result,
            transparency_log_path=transparency_log_path,
            output_path=baseline_bundle_path,
        )
        write_portable_evidence_bundle(
            result=provenance_result,
            transparency_log_path=transparency_log_path,
            output_path=provenance_bundle_path,
        )
        verification = verify_bundle(provenance_bundle_path)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        with ZipFile(baseline_bundle_path, mode="r") as baseline_archive:
            baseline_bundle = self._read_bundle_json(baseline_archive, "bundle.json")
        with ZipFile(provenance_bundle_path, mode="r") as provenance_archive:
            provenance_bundle = self._read_bundle_json(provenance_archive, "bundle.json")

        self.assertNotEqual(
            baseline_bundle["claim"],
            provenance_bundle["claim"],
        )
        self.assertNotEqual(
            self._member_hash_from_manifest(baseline_bundle, "released_state.json"),
            self._member_hash_from_manifest(provenance_bundle, "released_state.json"),
        )
        self.assertNotEqual(
            self._member_hash_from_manifest(baseline_bundle, "verification_report.json"),
            self._member_hash_from_manifest(provenance_bundle, "verification_report.json"),
        )
        self.assertNotEqual(
            baseline_bundle["integrity_manifest"]["bundle_hash"],
            provenance_bundle["integrity_manifest"]["bundle_hash"],
        )

    def test_non_default_profile_bundle_replays_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="contrast_enhanced_review")
        bundle_path = self.runtime_root / f"contrast{GUARDIAN_BUNDLE_EXTENSION}"

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
        self.assertEqual(
            manifest["proof_context"]["profile_identity"]["profile_name"],
            "contrast_enhanced_review",
        )
        self.assertEqual(manifest["replay_inputs"]["profile_name"], "contrast_enhanced_review")

    def test_morphology_profile_bundle_and_mock_pacs_flow_replay_identically(self) -> None:
        _, result = self._release_demo_case(profile_name="brain_t1_morphology_review")
        bundle_path = self.runtime_root / f"morphology{GUARDIAN_BUNDLE_EXTENSION}"

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
        bundle_path = self.runtime_root / f"component{GUARDIAN_BUNDLE_EXTENSION}"

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
        bundle_path = self.runtime_root / f"slice{GUARDIAN_BUNDLE_EXTENSION}"

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
        bundle_path = self.runtime_root / f"default-shift{GUARDIAN_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )

        with patch.object(profile_registry_module, "DEFAULT_PROFILE_NAME", "brain_t2_baseline"):
            verification = verify_bundle(bundle_path)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])

    def test_replay_fails_with_embedded_profile_descriptor_mismatch_when_bundle_descriptor_is_tampered(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"digest-mismatch{GUARDIAN_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"digest-mismatch-tampered{GUARDIAN_BUNDLE_EXTENSION}"
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            manifest = self._read_bundle_json(archive, "bundle.json")
        manifest["replay_inputs"]["contracts"]["verification_profile_descriptor"]["profile_digest"] = (
            "0" * 64
        )
        self._rewrite_bundle_members(
            source=bundle_path,
            destination=tampered_path,
            member_bytes={"bundle.json": self._json_bytes(manifest)},
        )

        verification = verify_bundle(tampered_path)

        self.assertFalse(verification["verified"])
        self.assertTrue(
            any(
                "embedded_profile_descriptor_mismatch" in failure
                for failure in verification["failures"]
            )
        )

    def test_verify_bundle_accepts_extracted_bundle_json(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"extracted{GUARDIAN_BUNDLE_EXTENSION}"
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

    def test_verify_bundle_detects_tampered_extracted_proof_critical_member(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"extracted-proof-critical{GUARDIAN_BUNDLE_EXTENSION}"
        extracted_root = self.runtime_root / "extracted_proof_critical_bundle"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(extracted_root)
        journal_path = extracted_root / "decision_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["entries"][0]["summary"] = "Free-form tampered summary."
        journal_path.write_text(f"{canonical_json_dumps(journal)}\n", encoding="utf-8")

        verification = verify_bundle(extracted_root / "bundle.json")

        self.assertFalse(verification["verified"])
        self.assertTrue(
            any(
                "decision_journal.json" in failure or "integrity_manifest" in failure
                for failure in verification["failures"]
            )
        )

    def test_portable_bundle_attachment_tampering_breaks_integrity_but_not_replay(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"clean{GUARDIAN_BUNDLE_EXTENSION}"
        attached_path = self.runtime_root / f"attached{GUARDIAN_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"tampered{GUARDIAN_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        self._add_bundle_attachment(
            source=bundle_path,
            destination=attached_path,
            member_name="attachment.txt",
            payload=b"auxiliary attachment\n",
            media_type="text/plain",
        )
        verification = verify_bundle(attached_path)

        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])

        self._rewrite_bundle_member(
            source=attached_path,
            destination=tampered_path,
            member_name="attachment.txt",
            payload=b"tampered auxiliary attachment\n",
        )

        verification = verify_bundle(tampered_path)

        self.assertFalse(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        self.assertTrue(any("integrity_manifest" in failure for failure in verification["failures"]))

    def test_portable_bundle_detects_decision_journal_tampering(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"clean-journal{GUARDIAN_BUNDLE_EXTENSION}"
        tampered_path = self.runtime_root / f"tampered-journal{GUARDIAN_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            journal = self._read_bundle_json(archive, "decision_journal.json")
        journal["entries"][0]["summary"] = "Free-form tampered summary."
        self._rewrite_bundle_member(
            source=bundle_path,
            destination=tampered_path,
            member_name="decision_journal.json",
            payload=self._json_bytes(journal),
        )

        verification = verify_bundle(tampered_path)

        self.assertFalse(verification["verified"])
        self.assertTrue(
            any(
                "decision_journal.json" in failure or "integrity_manifest" in failure
                for failure in verification["failures"]
            )
        )

    def test_multi_series_mock_dicom_pacs_and_bundle_replay_identically(self) -> None:
        state, result = self._release_multi_series_demo_case()
        bundle_path = self.runtime_root / f"{state['case_id']}{GUARDIAN_BUNDLE_EXTENSION}"
        dicom_input = build_mock_dicom_input(result.before_state)
        pacs_submission = build_mock_pacs_submission(
            result,
            bundle_reference=bundle_path.name,
        )
        integration_trace = build_mock_integration_trace(
            dicom_input=dicom_input,
            pacs_output=pacs_submission,
        )

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        verification = verify_bundle(bundle_path)
        with ZipFile(bundle_path, mode="r") as archive:
            manifest = self._read_bundle_json(archive, "bundle.json")
        first_series_claim = manifest["claim"]["series_claims"][0]
        first_series_contract = manifest["replay_inputs"]["contracts"]["series_contracts"][0]

        self.assertEqual(
            dicom_input["study"]["extensions"][GUARDIAN_EXTENSION_NAMESPACE]["case_kind"],
            MULTI_SERIES_CASE_KIND,
        )
        self.assertEqual(pacs_submission["verification_summary"]["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertEqual(integration_trace["pipeline"][1]["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertNotIn("profile_name", pacs_submission["verification_summary"])
        self.assertTrue(verification["verified"])
        self.assertTrue(verification["details"]["replay_verified"])
        self.assertNotIn("resolved_role", first_series_claim)
        self.assertNotIn("gating_status", first_series_claim)
        self.assertEqual(
            set(first_series_contract),
            {
                "series_id",
                "verification_profile_descriptor",
                "mask_derivation_policy",
                "verification_tolerance_policy",
            },
        )

    def test_run_demo_multi_series_emits_verified_artifacts(self) -> None:
        demo_root = self.runtime_root / "multi_series_demo"

        run_demo(
            output_dir=demo_root,
            case_kind=MULTI_SERIES_CASE_KIND,
        )
        verification = run_verify_command(demo_root)
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))
        pacs_submission = json.loads((demo_root / "pacs_submission.json").read_text(encoding="utf-8"))

        self.assertTrue(verification["verified"])
        self.assertEqual(report["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertEqual(pacs_submission["verification_summary"]["case_kind"], MULTI_SERIES_CASE_KIND)
        self.assertTrue((demo_root / "comparison.html").exists())
        self.assertTrue((demo_root / "decision_journal.json").exists())
        self.assertTrue((demo_root / "bundle_verification.json").exists())

    def test_run_demo_reuses_output_directory_without_stale_bundle_artifacts(self) -> None:
        demo_root = self.runtime_root / "reused_demo"

        run_demo(
            output_dir=demo_root,
            tamper_mode=True,
        )
        (demo_root / "legacy.mri-guard-bundle").write_text("legacy", encoding="utf-8")
        run_demo(
            output_dir=demo_root,
            case_kind=MULTI_SERIES_CASE_KIND,
        )

        present_after_multi = {path.name for path in demo_root.iterdir()}
        guardian_bundles = sorted(path.name for path in demo_root.glob(f"*{GUARDIAN_BUNDLE_EXTENSION}"))
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))

        self.assertEqual(guardian_bundles, ["mri-study-0001.guardian-bundle"])
        self.assertNotIn("legacy.mri-guard-bundle", present_after_multi)
        self.assertNotIn("ai_claim_envelope.json", present_after_multi)
        self.assertNotIn("measurement_integrity_proof.json", present_after_multi)
        self.assertIn("series_contracts.json", present_after_multi)
        self.assertEqual(report["case_kind"], MULTI_SERIES_CASE_KIND)

        run_demo(
            output_dir=demo_root,
            profile_name="fast_triage_mode",
            tamper_mode=False,
        )
        present_after_triage = {path.name for path in demo_root.iterdir()}

        self.assertEqual(
            present_after_triage,
            {
                "comparison.html",
                "decision_journal.json",
                "performance_profile.json",
                "verification_profile.json",
                "verification_profile_descriptor.json",
                "verification_report.json",
            },
        )

    def test_cli_verify_command_reports_directory_success(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"{result.after_state['case_id']}{GUARDIAN_BUNDLE_EXTENSION}"
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

    def test_cli_verify_command_accepts_extracted_bundle_directory(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"extracted-cli{GUARDIAN_BUNDLE_EXTENSION}"
        extracted_root = self.runtime_root / "extracted_cli_bundle"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(extracted_root)

        verification = run_verify_command(extracted_root)

        self.assertTrue(verification["verified"])
        self.assertIn("bundle", verification["parts"])
        self.assertIn("evidence", verification["parts"])
        self.assertEqual(verification["parts"]["bundle"]["details"]["source_kind"], "directory")

    def test_cli_verify_command_fails_on_tampered_extracted_bundle_manifest(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"tampered-manifest{GUARDIAN_BUNDLE_EXTENSION}"
        extracted_root = self.runtime_root / "tampered_manifest_bundle"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(extracted_root)
        manifest_path = extracted_root / "bundle.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["integrity_manifest"]["bundle_hash"] = "0" * 64
        manifest_path.write_text(f"{canonical_json_dumps(manifest)}\n", encoding="utf-8")

        verification = run_verify_command(extracted_root)

        self.assertFalse(verification["verified"])
        self.assertFalse(verification["parts"]["bundle"]["verified"])

    def test_cli_verify_command_fails_on_tampered_extracted_bundle_member(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"tampered-member{GUARDIAN_BUNDLE_EXTENSION}"
        extracted_root = self.runtime_root / "tampered_member_bundle"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(extracted_root)
        journal_path = extracted_root / "decision_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["entries"][0]["summary"] = "Free-form tampered summary."
        journal_path.write_text(f"{canonical_json_dumps(journal)}\n", encoding="utf-8")

        verification = run_verify_command(extracted_root)

        self.assertFalse(verification["verified"])
        self.assertFalse(verification["parts"]["bundle"]["verified"])

    def test_cli_verify_command_allows_evidence_only_directory(self) -> None:
        _, result = self._release_demo_case()
        (self.runtime_root / "evidence.json").write_text(
            f"{result.evidence_json}\n",
            encoding="utf-8",
        )
        (self.runtime_root / "bundle.json").write_text("{}\n", encoding="utf-8")

        verification = run_verify_command(self.runtime_root)

        self.assertTrue(verification["verified"])
        self.assertIn("evidence", verification["parts"])
        self.assertNotIn("bundle", verification["parts"])

    def test_cli_verify_command_rejects_directory_without_verifiable_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "No verifiable artifacts"):
            run_verify_command(self.runtime_root)

    def test_cli_verify_command_rejects_ambiguous_top_level_bundle_layouts(self) -> None:
        _, result = self._release_demo_case()
        bundle_path = self.runtime_root / f"ambiguous{GUARDIAN_BUNDLE_EXTENSION}"

        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(self.runtime_root)

        with self.assertRaisesRegex(ValueError, "contains both a \\.guardian-bundle archive"):
            run_verify_command(self.runtime_root)

    def test_cli_verify_command_ignores_nested_bundle_artifacts(self) -> None:
        _, result = self._release_demo_case()
        nested_root = self.runtime_root / "nested"
        bundle_path = nested_root / f"nested{GUARDIAN_BUNDLE_EXTENSION}"

        nested_root.mkdir()
        write_portable_evidence_bundle(
            result=result,
            transparency_log_path=self.runtime_root / "transparency-log.jsonl",
            output_path=bundle_path,
        )
        with ZipFile(bundle_path, mode="r") as archive:
            archive.extractall(nested_root)

        with self.assertRaisesRegex(ValueError, "No verifiable artifacts"):
            run_verify_command(self.runtime_root)

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

    def test_python_module_cli_demo_and_verify_commands_work_for_multi_series(self) -> None:
        demo_root = self.runtime_root / "cli_demo_multi_series"
        demo_command = [
            sys.executable,
            "-m",
            "mri_guardian",
            "demo",
            "--output-dir",
            str(demo_root),
            "--case-kind",
            MULTI_SERIES_CASE_KIND,
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
        self.assertEqual(verify_run.returncode, 0, verify_run.stderr)
        verification = json.loads(verify_run.stdout)
        report = json.loads((demo_root / "verification_report.json").read_text(encoding="utf-8"))
        self.assertTrue(verification["verified"])
        self.assertEqual(report["case_kind"], MULTI_SERIES_CASE_KIND)

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
                "decision_journal.json",
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

    def _base_spatial_geometry(self) -> dict[str, object]:
        return {
            "frame_of_reference_uid": "for-0001",
            "affine_ijk_to_mm": [
                [0.8, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }

    def _attach_temporal_metadata(
        self,
        state: dict[str, object],
        *,
        study_acquired_at: str | None,
        study_timestamp_provenance: str,
        spatial_geometry: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if study_acquired_at is None:
            state.pop("study_acquired_at", None)
        else:
            state["study_acquired_at"] = study_acquired_at
        state["study_timestamp_provenance"] = study_timestamp_provenance
        state["spatial_geometry"] = deepcopy(
            self._base_spatial_geometry() if spatial_geometry is None else spatial_geometry
        )
        return state

    def _build_prior_verified_context_from_release(
        self,
        *,
        released_state: dict[str, object],
        released_report: dict[str, object],
        timestamp_provenance: str = "verified_bundle",
        spatial_geometry: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "prior_bundle_sha256": hashlib.sha256(
                canonical_json_dumps(
                    {
                        "case_id": released_state["case_id"],
                        "released_state_hash": hash_canonical_json(released_state),
                        "verification_report_hash": hash_canonical_json(released_report),
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "prior_case_id": released_state["case_id"],
            "prior_study_uid": released_state["study_uid"],
            "prior_series_uid": released_state["series_uid"],
            "prior_released_state_hash": hash_canonical_json(released_state),
            "prior_verification_report_hash": hash_canonical_json(released_report),
            "prior_profile": {
                "profile_name": released_report["profile"]["profile_name"],
                "profile_revision": released_report["profile"]["profile_revision"],
                "profile_digest": released_report["profile"]["profile_digest"],
                "sequence_family": released_report["profile"]["sequence_family"],
            },
            "prior_release": {
                "release_status": released_state["release_status"],
                "final_disposition": released_report["decision"]["final_disposition"],
                "decision_class": released_report["decision"]["decision_class"],
            },
            "study_acquired_at": released_state.get("study_acquired_at"),
            "timestamp_provenance": timestamp_provenance,
            "verified_mask": deepcopy(released_report["analysis"]["verified_mask"]),
            "verified_mask_hash": released_report["analysis"]["verified_mask_hash"],
            "verified_volume_mm3": released_report["analysis"]["computed_tumor_volume_mm3"],
            "verified_mask_voxel_count": int(
                released_report["analysis"]["verified_mask_voxel_count"]
            ),
            "voxel_spacing_mm": deepcopy(released_report["analysis"]["voxel_spacing_mm"]),
            "spatial_geometry": deepcopy(
                released_report["analysis"]["spatial_geometry"]
                if spatial_geometry is None
                else spatial_geometry
            ),
        }

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

    def _release_demo_case_with_state(
        self,
        state: dict[str, object],
        *,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ) -> tuple[dict[str, object], object]:
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

    def _release_multi_series_demo_case(
        self,
        *,
        state: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], object]:
        working_state = build_multi_series_demo_case() if state is None else state
        result = self._build_harness().release_multi_series_case(
            state=working_state,
            actor="guardian:test",
            authority_sources_by_series_id=build_multi_series_demo_authority_sources(),
            timestamp="2026-03-26T00:00:00Z",
        )
        return working_state, result

    def _series_entry(self, state: dict[str, object], series_id: str) -> dict[str, object]:
        for entry in state["series"]:
            if entry["series_id"] == series_id:
                return entry
        raise AssertionError(f"Series {series_id!r} not found.")

    def _rewrite_case_for_profile(
        self,
        state: dict[str, object],
        reconstruction: np.ndarray,
    ) -> None:
        descriptor = verification_profile_descriptor_from_dict(
            dict(state["verification_profile_descriptor"])
        )
        raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
        derived_reconstruction = np.fft.ifft2(raw_kspace, axes=(-2, -1)).real
        verified_mask = derive_verified_mask(
            derived_reconstruction,
            descriptor.mask_rules,
        ).verified_mask
        voxel_spacing = np.asarray(state["voxel_spacing_mm"], dtype=np.float64)
        state["raw_real"] = raw_kspace.real.tolist()
        state["raw_imag"] = raw_kspace.imag.tolist()
        state["ai_reconstruction"] = derived_reconstruction.tolist()
        state["tumor_mask"] = verified_mask.astype(np.int64).tolist()
        state["tumor_volume_mm3"] = round(
            float(np.count_nonzero(verified_mask) * np.prod(voxel_spacing)),
            9,
        )

    def _rewrite_series_for_profile(
        self,
        state: dict[str, object],
        series_id: str,
        reconstruction: np.ndarray,
    ) -> None:
        series_entry = self._series_entry(state, series_id)
        descriptor = verification_profile_descriptor_from_dict(
            dict(series_entry["verification_profile_descriptor"])
        )
        raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
        derived_reconstruction = np.fft.ifft2(raw_kspace, axes=(-2, -1)).real
        verified_mask = derive_verified_mask(
            derived_reconstruction,
            descriptor.mask_rules,
        ).verified_mask
        voxel_spacing = np.asarray(series_entry["voxel_spacing_mm"], dtype=np.float64)
        series_entry["raw_real"] = raw_kspace.real.tolist()
        series_entry["raw_imag"] = raw_kspace.imag.tolist()
        series_entry["ai_reconstruction"] = derived_reconstruction.tolist()
        series_entry["tumor_mask"] = verified_mask.astype(np.int64).tolist()
        series_entry["tumor_volume_mm3"] = round(
            float(np.count_nonzero(verified_mask) * np.prod(voxel_spacing)),
            9,
        )

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

    def _build_zero_mask_case(self) -> dict[str, object]:
        state = build_demo_case()
        reconstruction = np.zeros_like(np.asarray(state["ai_reconstruction"], dtype=np.float64))
        raw_kspace = np.fft.fft2(reconstruction, axes=(-2, -1))
        zero_mask = np.zeros_like(reconstruction, dtype=np.int64)
        state["raw_real"] = raw_kspace.real.tolist()
        state["raw_imag"] = raw_kspace.imag.tolist()
        state["ai_reconstruction"] = reconstruction.tolist()
        state["tumor_mask"] = zero_mask.tolist()
        state["tumor_volume_mm3"] = 0.0
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
                dst.writestr(clone, member_bytes.get(info.filename, src.read(info.filename)))

    def _member_hash_from_manifest(
        self,
        manifest: dict[str, object],
        member_name: str,
    ) -> str:
        for item in manifest["integrity_manifest"]["member_hashes"]:
            if item["member"] == member_name:
                return str(item["value"])
        raise AssertionError(f"Missing member hash for {member_name!r}")

    def _build_integrity_manifest(
        self,
        bundle_payload: dict[str, object],
        member_bytes: dict[str, bytes],
        attachments: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        member_order = [
            "bundle.json",
            "input_state.json",
            "released_state.json",
            "workstation_payload.json",
            "verification_report.json",
            "decision_journal.json",
            "evidence.json",
            "transparency-prefix.jsonl",
        ]
        if attachments:
            member_order.extend(item["member"] for item in attachments)
        redacted_bundle = deepcopy(bundle_payload)
        redacted_bundle["integrity_manifest"] = "__INTEGRITY_MANIFEST_REDACTED__"
        bundle_descriptor_hash = hash_canonical_json(redacted_bundle)
        member_hashes = []
        for member in member_order:
            value = (
                bundle_descriptor_hash
                if member == "bundle.json"
                else hashlib.sha256(member_bytes[member]).hexdigest()
            )
            member_hashes.append(
                {"member": member, "algorithm": "sha256", "value": value}
            )
        return {
            "member_order": member_order,
            "member_hashes": member_hashes,
            "proof_critical_members": member_order[1:8],
            "bundle_descriptor_hash": bundle_descriptor_hash,
            "bundle_hash": hash_canonical_json(member_hashes),
        }

    def _add_bundle_attachment(
        self,
        *,
        source: Path,
        destination: Path,
        member_name: str,
        payload: bytes,
        media_type: str,
    ) -> None:
        with ZipFile(source, mode="r") as src:
            info_by_name = {info.filename: info for info in src.infolist()}
            member_bytes = {info.filename: src.read(info.filename) for info in src.infolist()}
        manifest = json.loads(member_bytes["bundle.json"].decode("utf-8"))
        attachments = [{"member": member_name, "media_type": media_type}]
        manifest["attachments"] = attachments
        member_bytes[member_name] = payload
        manifest["integrity_manifest"] = self._build_integrity_manifest(
            manifest,
            member_bytes,
            attachments=attachments,
        )
        member_bytes["bundle.json"] = self._json_bytes(manifest)
        with ZipFile(destination, mode="w") as dst:
            for bundled_name in manifest["integrity_manifest"]["member_order"]:
                info = info_by_name.get(bundled_name)
                clone = ZipInfo(
                    filename=bundled_name,
                    date_time=info.date_time if info is not None else (2026, 3, 26, 0, 0, 0),
                )
                clone.compress_type = ZIP_STORED
                clone.create_system = 0 if info is None else info.create_system
                clone.external_attr = 0o600 << 16 if info is None else info.external_attr
                dst.writestr(clone, member_bytes[bundled_name])

    def _json_bytes(self, payload: dict[str, object]) -> bytes:
        return f"{canonical_json_dumps(payload)}\n".encode("utf-8")

    def _cleanup_runtime_root(self) -> None:
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        parent = self.runtime_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


if __name__ == "__main__":
    unittest.main()
