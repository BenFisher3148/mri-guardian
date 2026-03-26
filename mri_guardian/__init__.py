"""MRI Guardian Layer built on top of dpa-icsd."""

from .bundle import (
    MRI_GUARD_BUNDLE_EXTENSION,
    verify_bundle,
    verify_portable_evidence_bundle,
    write_portable_evidence_bundle,
)
from .contracts import (
    AIClaimEnvelope,
    MaskDerivationPolicy,
    VerificationTolerancePolicy,
    VerifiedReconstructionTrace,
    build_ai_claim_envelope,
    build_mask_derivation_policy,
    build_verification_tolerance_policy,
    build_verified_reconstruction_trace,
)
from .harness import (
    MRI_GUARDIAN_POLICY_NAME,
    MRI_GUARDIAN_POLICY_VERSION,
    RELEASE_STATUS_PENDING,
    RELEASE_STATUS_RELEASED,
    MriGuardianHarness,
    build_case_verification_report,
    build_demo_authority_sources,
    build_demo_case,
)
from .integration import (
    build_guardian_integration_narrative,
    build_mock_dicom_input,
    build_mock_integration_trace,
    build_mock_pacs_block_notice,
    build_mock_pacs_submission,
    restore_case_state_from_dicom_input,
)
from .tamper import build_tamper_mode_base_case, build_tamper_scenarios
from .timing import build_guardian_performance_profile, build_timing_overlay_snapshot
from .viewer import build_side_by_side_view_model, render_side_by_side_report_html

__all__ = [
    "MRI_GUARD_BUNDLE_EXTENSION",
    "MRI_GUARDIAN_POLICY_NAME",
    "MRI_GUARDIAN_POLICY_VERSION",
    "RELEASE_STATUS_PENDING",
    "RELEASE_STATUS_RELEASED",
    "MriGuardianHarness",
    "verify_bundle",
    "verify_portable_evidence_bundle",
    "write_portable_evidence_bundle",
    "AIClaimEnvelope",
    "MaskDerivationPolicy",
    "VerificationTolerancePolicy",
    "VerifiedReconstructionTrace",
    "build_ai_claim_envelope",
    "build_mask_derivation_policy",
    "build_verification_tolerance_policy",
    "build_verified_reconstruction_trace",
    "build_case_verification_report",
    "build_demo_authority_sources",
    "build_demo_case",
    "build_guardian_integration_narrative",
    "build_mock_dicom_input",
    "build_mock_integration_trace",
    "build_mock_pacs_block_notice",
    "build_mock_pacs_submission",
    "restore_case_state_from_dicom_input",
    "build_tamper_mode_base_case",
    "build_tamper_scenarios",
    "build_guardian_performance_profile",
    "build_timing_overlay_snapshot",
    "build_side_by_side_view_model",
    "render_side_by_side_report_html",
]
