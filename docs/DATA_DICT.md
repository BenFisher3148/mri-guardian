# Data Dictionary

This document describes the main data structures emitted and consumed by the MRI Guardian Layer.

## Core Case State

The base case state is the canonical input to the verification harness.

| Field | Type | Description |
| --- | --- | --- |
| `case_id` | `str` | Stable case identifier used across artifacts. |
| `patient_pseudonym` | `str` | Pseudonymised patient identifier used in demo artifacts. |
| `study_uid` | `str` | Study identifier preserved across ingress and egress. |
| `series_uid` | `str` | Source series identifier for the AI candidate. |
| `scanner_id` | `str` | Scanner or acquisition source identifier. |
| `sequence_id` | `str` | MRI acquisition or protocol name. |
| `ai_model_id` | `str` | Identifier for the upstream AI model. |
| `release_status` | `str` | `pending_verification` or `released_to_radiologist`. |
| `raw_real` | `list[list[list[float]]]` | Real component of raw k-space. |
| `raw_imag` | `list[list[list[float]]]` | Imaginary component of raw k-space. |
| `ai_reconstruction` | `list[list[list[float]]]` | AI-proposed voxel intensities. |
| `tumor_mask` | `list[list[list[int]]]` | AI-proposed binary tumour mask. |
| `voxel_spacing_mm` | `list[float]` | Physical voxel spacing in millimetres. |
| `tumor_volume_mm3` | `float` | Claimed tumour volume. |
| `verification_profile_descriptor` | `dict` | Canonical runtime profile descriptor embedded with the case state and replayed on verification. |
| `verified_manifest` | `dict | None` | Deterministic proof manifest, populated only on release. |
| `workstation_payload` | `dict | None` | Verified downstream payload, populated only on release. |

## Released Workstation Payload

Current top-level keys from [`../demo_artifacts/workstation_payload.json`](../demo_artifacts/workstation_payload.json):

| Field | Type | Description |
| --- | --- | --- |
| `case_id` | `str` | Case identifier. |
| `study_uid` | `str` | Study identifier. |
| `series_uid` | `str` | Source series identifier. |
| `scanner_id` | `str` | Scanner identifier. |
| `sequence_id` | `str` | Acquisition or protocol identifier. |
| `patient_pseudonym` | `str` | Pseudonymised patient identifier. |
| `profile_name` | `str` | Guardian runtime profile used to derive the payload. |
| `profile_revision` | `str` | Revision scoped to the selected `profile_name`. |
| `profile_digest` | `str` | Canonical digest of the resolved runtime profile descriptor. |
| `pixel_source` | `str` | Source semantics for the released reconstruction. |
| `overlay_mode` | `str` | Source semantics for the released mask. |
| `reconstruction_hash` | `str` | Canonical hash of the released reconstruction. |
| `verified_mask_hash` | `str` | Canonical hash of the released mask. |
| `tumor_volume_mm3` | `float` | Verified tumour volume. |
| `reconstruction` | `list[list[list[float]]]` | Released raw-derived reconstruction. |
| `verified_mask` | `list[list[list[int]]]` | Released verified mask. |

## Evidence Envelope

Current top-level keys from [`../demo_artifacts/evidence.json`](../demo_artifacts/evidence.json):

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | `str` | ICSD evidence schema version. |
| `entity_type` | `str` | Entity type under transition. |
| `entity_id` | `str` | Entity identifier under transition. |
| `actor` | `str` | Actor applying the release transition. |
| `authority` | `str` | Authority or policy context. |
| `authority_sources` | `list[dict]` | Required upstream anchors for the release. |
| `timestamp` | `str` | Transition timestamp. |
| `transition_id` | `str` | Deterministic transition identifier. |
| `before_hash` | `str` | Canonical hash of pre-transition state. |
| `after_hash` | `str` | Canonical hash of post-transition state. |
| `provenance` | `list[dict]` | Transparency receipts and provenance entries. |

## Claim-to-Proof Contracts

Defined in [`../mri_guardian/contracts.py`](../mri_guardian/contracts.py).

| Interface | Purpose | Key fields |
| --- | --- | --- |
| `VerificationProfileDescriptor` | Canonical runtime profile contract | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, release semantics, heuristics, performance expectations |
| `AIClaimEnvelope` | AI-side claim over reconstruction, mask, and volume | case identity, `raw_kspace_hash`, AI output hashes, claimed volume |
| `MaskDerivationPolicy` | Deterministic mapping from raw-derived reconstruction to verified mask | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, threshold rule |
| `VerificationTolerancePolicy` | Explicit numeric tolerance contract | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, per-voxel, per-frequency, global-norm, imaginary leakage, and volume tolerances |
| `VerifiedReconstructionTrace` | Proof trace from raw k-space to verified reconstruction, mask, and volume | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, raw hash, derived hash, verified mask hash, measurement metrics, claim comparison metrics |
| `GuardianIntegrationNarrative` | Hospital-facing integration narrative | DICOM compliance story, hospital placement, fail-closed behaviour |

## Integration Artifacts

| Artifact | Purpose |
| --- | --- |
| [`../demo_artifacts/dicom_input.json`](../demo_artifacts/dicom_input.json) | Mocked DICOM ingress envelope carrying raw data, AI claim, and embedded Guardian case state |
| [`../demo_artifacts/pacs_submission.json`](../demo_artifacts/pacs_submission.json) | Mocked accepted PACS handoff with verified series and proof contracts |
| [`../demo_artifacts/tamper_fake_tumour_pacs_block.json`](../demo_artifacts/tamper_fake_tumour_pacs_block.json) | Mocked rejected PACS notice for a blocked case |
| [`../demo_artifacts/integration_trace.json`](../demo_artifacts/integration_trace.json) | Boundary trace from DICOM ingress through Guardian decision to PACS egress |
| [`../demo_artifacts/integration_narrative.json`](../demo_artifacts/integration_narrative.json) | Standalone DICOM, placement, and fail-closed narrative |

## Bundle Members

The portable bundle written by [`../mri_guardian/bundle.py`](../mri_guardian/bundle.py) contains:

- `bundle.json`
- `signatures.json`
- `evidence.json`
- `input_state.json`
- `released_state.json`
- `workstation_payload.json`
- `verification_report.json`
- `verification_profile_descriptor.json`
- `transcript.json`
- `replay.json`
- `transparency-log.jsonl`

## Notes on Determinism

- Hash-bearing fields assume canonical JSON serialisation.
- Arrays are represented as nested lists to keep bundle and evidence payloads deterministic.
- Changing field order, numeric rounding rules, or member ordering may break replay verification.

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`adr/0003-explicit-claim-proof-contracts.md`](adr/0003-explicit-claim-proof-contracts.md)
- [`adr/0004-replayable-evidence-bundles.md`](adr/0004-replayable-evidence-bundles.md)
