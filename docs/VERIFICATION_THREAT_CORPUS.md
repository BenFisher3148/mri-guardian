# Verification Threat Corpus

This document describes the executable verification threat corpus used by the MRI Guardian Layer prototype.

The corpus is a deterministic adversarial and failure-scenario registry. It is not a compliance artefact, not a formal clinical safety case, and not a patient-specific risk score.

Tests are treated as executable evidence of control behaviour, not illustrative examples.

The architectural decision for this surface is recorded in [`adr/0013-formal-verification-threat-corpus-v1.md`](adr/0013-formal-verification-threat-corpus-v1.md).

## Model

- Canonical builder: `build_verification_threat_corpus(...)`
- Compatibility adapter: `build_tamper_scenarios()`
- Corpus severity is qualitative documentation metadata only. It is not runtime `decision.severity`.
- Normative invariants are the expected `decision_class` and fail-closed route.
- Observational expectations are asserted only where intentionally durable, such as stable block categories or stable primary reason families.

## Controlled Vocabulary

### Scenario families

- `reconstruction_integrity`
- `mask_semantics`
- `quantitative_claims`
- `temporal_context`
- `multi_series_corroboration`
- `profile_scoped_policy`

### Scenario surfaces

- `single_series_reconstruction`
- `single_series_mask`
- `single_series_quantification`
- `multi_series_corroboration`
- `state_reuse`
- `profile_scoped_mask_policy`

### Control roles

- `preventative`
- `detective`
- `gating`
- `evidentiary`

### Corpus status

- `active`: executable baseline corpus entry
- `profile_scoped`: executable entry tied to a specific profile family
- `deferred`: reserved for future corpus growth, not used in V1

## Reconstruction Integrity

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-003` | `intensity_manipulation` | `local_intensity_patch_mutation` | `image_integrity_distortion` | False positive or false negative interpretation burden | `major` | `VerificationTolerancePolicy` (detective), `manifest.ai_reconstruction_match` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed patch at `z=0:3`, `y=0:7`, `x=16:23`, delta `0.55` | `blocked_physical` via `physical` route | `test_rejects_ai_reconstruction_with_invented_pixels`; `manifest.ai_reconstruction_match` |
| `VFTC-004` | `subtle_distributed_perturbation` | `distributed_low_amplitude_voxel_perturbation` | `image_integrity_distortion` | Boundary-hugging corruption can look visually benign | `major` | `VerificationTolerancePolicy` (detective), `manifest.ai_reconstruction_match` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed global delta `9e-10` on all voxels | `blocked_physical` via `physical` route | `test_rejects_distributed_low_amplitude_attack_by_global_l2_bound`; `decision.success_modifiers.near_proof_threshold` |
| `VFTC-005` | `structured_ringing_artefact` | `ringing_pattern_injection` | `image_integrity_distortion` | Halo-like artefacts can bias human review | `major` | `VerificationTolerancePolicy` (detective), `manifest.ai_reconstruction_match` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed cosine ringing envelope, no RNG | `blocked_physical` via `physical` route | `manifest.ai_reconstruction_match` |
| `VFTC-006` | `slice_local_anomaly_injection` | `slice_scoped_signal_injection` | `false_positive_finding` | Single-slice false lesion cue can trigger avoidable escalation | `major` | `VerificationTolerancePolicy` (detective), `manifest.ai_reconstruction_match` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed slice `1`, patch `y=9:15`, `x=9:15`, delta `0.85` | `blocked_physical` via `physical` route | `test_rejects_ai_reconstruction_with_invented_pixels`; `manifest.ai_reconstruction_match` |
| `VFTC-010` | `mixed_frequency_perturbation` | `mixed_domain_frequency_perturbation` | `image_integrity_distortion` | Hybrid artefacts can escape purely local visual heuristics | `major` | `VerificationTolerancePolicy` (detective), `manifest.ai_reconstruction_match` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed low-frequency drift plus fixed high-frequency sign pattern | `blocked_physical` via `physical` route | `test_rejects_distributed_low_amplitude_attack_by_global_l2_bound`; `manifest.ai_reconstruction_match` |

## Mask Semantics

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-001` | `fake_tumour` | `unsupported_mask_voxel_injection` | `false_positive_finding` | Unnecessary escalation; possible invasive follow-up | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed mask patch of `342` voxels | `blocked_physical` via `physical` route | `test_rejects_ai_mask_with_fake_tumour_voxel`; `manifest.mask_verification.comparison` |
| `VFTC-002` | `shifted_mask` | `spatial_mask_shift` | `spatial_mislocalised_finding` | Mislocalised review attention and avoidable follow-up | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed translation `[0,0,1]` voxels | `blocked_physical` via `physical` route | `test_tamper_mode_scenarios_surface_expected_primary_failures`; `manifest.mask_verification.comparison` |
| `VFTC-007` | `mask_erosion` | `mask_undersegmentation_mutation` | `false_negative_finding` | Under-call risk and delayed follow-up | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed erosion of `x=16` face | `blocked_physical` via `physical` route | `manifest.mask_verification.comparison` |
| `VFTC-008` | `mask_dilation` | `mask_oversegmentation_mutation` | `false_positive_finding` | Over-call risk and avoidable diagnostic escalation | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed dilation on `z=1:5`, `y=8:16`, `x=17` | `blocked_physical` via `physical` route | `manifest.mask_verification.comparison` |

## Quantitative Claims

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-009` | `volume_rounding_attack` | `scalar_volume_rounding_mismatch` | `quantitative_misstatement` | Misleading trend framing and inappropriate follow-up planning | `moderate` | `manifest.tumor_volume_proof` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed scalar delta `0.5 mm3` | `blocked_physical` via `physical` route | `test_rejects_incorrect_tumor_volume_claim`; `manifest.tumor_volume_proof` |

## Temporal Context

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-011` | `stale_mask_replay` | `stale_mask_semantic_reuse` | `stale_context_driven_misstatement` | Out-of-date lesion state can misframe present review and follow-up | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed replay source with preserved syntax and structure; forced into semantic verification, not intake rejection | `blocked_physical` via `physical` route | `manifest.mask_verification.comparison`; `VerifiedReconstructionTrace` |

`stale_mask_replay` is intentionally scoped to state-level semantic mismatch. It is not manifest tampering, bundle replay corruption, transport fraud, or provenance fraud.

## Multi-Series Corroboration

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-012` | `cross_series_misregistration` | `cross_series_spatial_offset_without_geometry_mismatch` | `false_corroborative_confidence` | Study-level confidence can be distorted by misaligned support | `major` | `manifest.multi_series_aggregation.geometry_verification` (preventative), `cross_series_verification_trace.lesion_corroboration` (detective), `StudyTemporalAggregationTrace` (evidentiary), `release_control` (gating) | Fixed offset on `t2-supporting`; metadata preserved so geometry remains admissible and failure lands in corroboration | `blocked_physical` via `physical` route | `test_multi_series_primary_positive_without_any_supporting_overlap_blocks_release`; `analysis.lesion_corroboration` |

`cross_series_misregistration` is locked to one invariant: geometry verification still passes, required series remain admissible, and the case fails because lesion corroboration fails.

## Profile-Scoped Policy

| Corpus ID | Slug | Technical mechanism | Clinical hazard | Downstream effect | Severity | Control mapping | Determinism | Normative invariant | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VFTC-013` | `component_micro_island` | `disconnected_component_restoration` | `false_positive_finding` | Detached positives can inflate lesion footprint | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed component-profile reconstruction and connected-component policy | `blocked_physical` via `physical` route | `test_component_profile_tamper_mode_surfaces_micro_island_mismatch`; `contracts.mask_derivation_policy` |
| `VFTC-014` | `slice_single_slice_island` | `single_slice_component_restoration` | `false_positive_finding` | Transient single-slice positives can trigger avoidable review | `major` | `MaskDerivationPolicy` (detective), `manifest.mask_verification.comparison` (detective), `VerifiedReconstructionTrace` (evidentiary), `release_control` (gating) | Fixed slice-profile reconstruction and slice-consistency policy | `blocked_physical` via `physical` route | `test_slice_profile_tamper_mode_surfaces_single_slice_island_mismatch`; `contracts.mask_derivation_policy` |

## Notes

- `verification_threat_corpus.json` is the demo artifact that serialises corpus metadata with generated file paths.
- `build_tamper_scenarios()` remains the curated compatibility subset used by older demo and viewer expectations.
- Active scenarios must include at least one detective or preventative control and at least one gating or evidentiary control.
