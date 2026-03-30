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

## Corpus Map

The corpus now fixes two architectural metadata fields:

- `pipeline_phase`: the intended primary assurance-flow phase exercised by the scenario
- `proof_stage`: the intended primary proof stage when `pipeline_phase = proof_core`

`proof_stage` is primary-stage metadata only. Downstream propagated failures do not reclassify the corpus entry.

| Corpus ID | Slug | Pipeline phase | Proof stage | Technical mechanism | Key evidence surfaces |
| --- | --- | --- | --- | --- | --- |
| `VFTC-001` | `fake_tumour` | `proof_core` | `SegmentationPolicyProof` | `unsupported_mask_voxel_injection` | `manifest.segmentation_policy_proof.comparison`; `ProofCoreEvidenceTrace` |
| `VFTC-002` | `shifted_mask` | `proof_core` | `SegmentationPolicyProof` | `spatial_mask_shift` | `manifest.segmentation_policy_proof.comparison`; `ProofCoreEvidenceTrace` |
| `VFTC-003` | `intensity_manipulation` | `proof_core` | `ReconstructionParityProof` | `local_intensity_patch_mutation` | `manifest.reconstruction_parity_proof`; `ProofCoreEvidenceTrace` |
| `VFTC-004` | `subtle_distributed_perturbation` | `proof_core` | `ReconstructionParityProof` | `distributed_low_amplitude_voxel_perturbation` | `manifest.reconstruction_parity_proof.metrics.global_relative_l2_error`; `decision.success_modifiers.near_proof_threshold` |
| `VFTC-005` | `structured_ringing_artefact` | `proof_core` | `ReconstructionParityProof` | `ringing_pattern_injection` | `manifest.reconstruction_parity_proof`; `ProofCoreEvidenceTrace` |
| `VFTC-006` | `slice_local_anomaly_injection` | `proof_core` | `ReconstructionParityProof` | `slice_scoped_signal_injection` | `manifest.reconstruction_parity_proof`; `ProofCoreEvidenceTrace` |
| `VFTC-007` | `mask_erosion` | `proof_core` | `SegmentationPolicyProof` | `mask_undersegmentation_mutation` | `manifest.segmentation_policy_proof.comparison`; `ProofCoreEvidenceTrace` |
| `VFTC-008` | `mask_dilation` | `proof_core` | `SegmentationPolicyProof` | `mask_oversegmentation_mutation` | `manifest.segmentation_policy_proof.comparison`; `ProofCoreEvidenceTrace` |
| `VFTC-009` | `volume_rounding_attack` | `proof_core` | `VolumeComputationProof` | `scalar_volume_rounding_mismatch` | `manifest.volume_computation_proof`; `ProofCoreEvidenceTrace` |
| `VFTC-010` | `mixed_frequency_perturbation` | `proof_core` | `ReconstructionParityProof` | `mixed_domain_frequency_perturbation` | `manifest.reconstruction_parity_proof`; `ProofCoreEvidenceTrace` |
| `VFTC-011` | `stale_mask_replay` | `proof_core` | `SegmentationPolicyProof` | `stale_mask_semantic_reuse` | `manifest.segmentation_policy_proof.comparison`; `ProofCoreEvidenceTrace` |
| `VFTC-012` | `cross_series_misregistration` | `multi_series_aggregation` |  | `cross_series_spatial_offset_without_geometry_mismatch` | `analysis.lesion_corroboration`; `contracts.cross_series_verification_trace` |
| `VFTC-013` | `component_micro_island` | `proof_core` | `SegmentationPolicyProof` | `disconnected_component_restoration` | `contracts.mask_derivation_policy`; `manifest.segmentation_policy_proof.comparison` |
| `VFTC-014` | `slice_single_slice_island` | `proof_core` | `SegmentationPolicyProof` | `single_slice_component_restoration` | `contracts.mask_derivation_policy`; `manifest.segmentation_policy_proof.comparison` |
| `VFTC-015` | `measurement_domain_tamper` | `proof_core` | `MeasurementIntegrityProof` | `raw_measurement_patch_mutation` | `manifest.measurement_integrity_proof`; `contracts.proof_core_evidence_trace` |

## Notes

- `verification_threat_corpus.json` is the demo artifact that serialises corpus metadata with generated file paths.
- `build_tamper_scenarios()` remains the curated compatibility subset used by older demo and viewer expectations.
- Active scenarios must include at least one detective or preventative control and at least one gating or evidentiary control.
- `stale_mask_replay` is intentionally scoped to state-level semantic mismatch. It is not manifest tampering, bundle replay corruption, transport fraud, or provenance fraud.
- `cross_series_misregistration` is locked to one invariant: geometry verification still passes, required series remain admissible, and the case fails because lesion corroboration fails.
