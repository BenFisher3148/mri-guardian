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

## Multi-Series Case State

`multi_series_v2` adds a study-level wrapper around the existing single-series fields.

| Field | Type | Description |
| --- | --- | --- |
| `case_kind` | `str` | Fixed to `multi_series_v2` for the study-level path. |
| `case_id` | `str` | Stable study identifier used across artifacts. |
| `patient_pseudonym` | `str` | Study-level pseudonymized patient identifier. |
| `study_uid` | `str` | Study identifier preserved across ingress and egress. |
| `scanner_id` | `str` | Study-level scanner identifier. |
| `release_status` | `str` | `pending_verification` or `released_to_radiologist`. |
| `series` | `list[dict]` | Ordered payload series. Each entry contains a unique `series_id` plus the existing per-series raw, AI, mask, volume, and embedded profile fields. |
| `cross_series_policy` | `dict` | Study-level gate selecting `primary_series_id`, `supporting_series_ids`, `required_series_ids`, the geometry rule, and the corroboration rule. |
| `verified_manifest` | `dict | None` | Released study manifest. Includes every payload series in proof-significant order. |
| `workstation_payload` | `dict | None` | Released study payload. Includes only downstream-allowed series plus an explicit withheld-series summary. |

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

For `multi_series_v2`, the top-level released payload contains:

- `series_payloads`: only the series that remain downstream-allowed after the merged study gate
- `withheld_series`: explicit summaries for payload series that were preserved in study artifacts but omitted from downstream release

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
| `VerificationProfileDescriptor` | Canonical runtime profile contract | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, release semantics, canonical `mask_rules`, heuristics, performance expectations |
| `AIClaimEnvelope` | AI-side claim over reconstruction, mask, and volume | case identity, `raw_kspace_hash`, AI output hashes, claimed volume |
| `MaskDerivationPolicy` | Deterministic mapping from raw-derived reconstruction to verified mask | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, proof-significant `algorithm`, and canonical resolved `parameters` |
| `VerificationTolerancePolicy` | Explicit numeric tolerance contract | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, per-voxel, per-frequency, global-norm, imaginary leakage, and volume tolerances |
| `VerifiedReconstructionTrace` | Proof trace from raw k-space to verified reconstruction, mask, and volume | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, raw hash, derived hash, verified mask hash, measurement metrics, claim comparison metrics |
| `CrossSeriesVerificationPolicy` | Study-level multi-series gate contract | `primary_series_id`, canonicalized `supporting_series_ids`, canonicalized `required_series_ids`, geometry rule, corroboration rule |
| `CrossSeriesVerificationTrace` | Study-level trace for geometry, corroboration, and merged release evaluation | `geometry_verification`, `lesion_corroboration`, `merged_release_evaluation` |
| `MultiSeriesVerificationProfileDescriptorSet` | Ordered wrapper around per-series runtime profile descriptors | proof-significant ordered `series_descriptors` list |
| `GuardianIntegrationNarrative` | Hospital-facing integration narrative | DICOM compliance story, hospital placement, fail-closed behaviour |

### Mask Policy Shape

- `VerificationProfileDescriptor.mask_rules` stores the canonical approved mask policy object used by the selected profile.
- `mask_rules.algorithm` is a proof-significant stable identifier, not a display label.
- `mask_rules.parameters` is resolved canonically before execution. Unknown or algorithm-inapplicable parameters are rejected rather than ignored.
- The connected-component family uses an explicit proof-significant `connectivity` field, but this milestone approves only `faces`; other values are rejected during canonical resolution.
- The slice-consistency family uses explicit proof-significant `consistency_unit`, `in_slice_connectivity`, `min_consecutive_slices`, and `overlap_rule` fields. This milestone approves only `in_slice_component`, `faces_2d`, and `any_overlap`.
- Morphology connectivity is encoded by the structuring element definition rather than a separate connectivity field.
- Component labels are deterministic internal execution state and are intentionally excluded from proof artifacts.
- Slice-consistency labels, overlap-graph node ids, and qualifying-run ids are deterministic internal execution state and are intentionally excluded from proof artifacts.
- Slice-consistency overlap is evaluated as non-empty intersection of adjacent components' `(y, x)` footprint sets on slices `z` and `z+1`; skipped-slice bridging is not permitted.
- Runtime metadata from mask execution, such as timing or internal operation counts, is intentionally excluded from proof artifacts.

### Mask Verification Shape

Released manifests expose `mask_verification` in two logical sections:

- `policy`: canonical `algorithm` plus canonical resolved `parameters`
- `comparison`: exact-match outcome over AI vs verified mask, including voxel counts and deltas

Flat compatibility aliases may still appear during the transition, but future non-threshold families should rely on canonical `mask_rules` and `policy`/`comparison` structure rather than `mask_threshold`.

### Multi-Series Gating Shape

- `required_series_ids` is the only authoritative merged-release gating set.
- `primary_series_id` must appear exactly once in `required_series_ids`.
- `supporting_series_ids` must not include `primary_series_id`.
- For this milestone, every `supporting_series_id` is required and every payload series outside `required_series_ids` is auxiliary.
- Duplicate payload `series_id` values are a structural study-shape failure and stop meaningful cross-series evaluation.
- Policy `series_id` references must be non-empty strings.
- `supporting_series_ids` must be non-empty in `multi_series_v2`; this is a hard validation error rather than an automatic downgrade to single-series.
- `required_series_ids` and `supporting_series_ids` are canonicalized into outer study order. Caller order is not proof-significant.
- Geometry equality is exact equality over canonicalized voxel spacing plus exact voxel shape.
- Positive-lesion corroboration uses exact coordinate intersection count `> 0`.
- If every required verified mask is empty, corroboration passes; lesion absence is not separately cross-series corroborated in this milestone.
- `series_manifests` includes all payload series in proof-significant order; `series_payloads` includes only downstream-allowed series and `withheld_series` records explicit omissions.

## Decision Object

`verification_report.json` now includes an additive canonical `decision` object alongside the legacy `decision_summary`.

| Field | Type | Description |
| --- | --- | --- |
| `final_disposition` | `str` | `released` or `blocked`. Mirrors the hard handoff outcome and does not replace `release_eligible`. |
| `decision_class` | `str` | Canonical semantic class: `verified_clean`, `verified_boundary_hugging`, `verified_with_advisory`, `blocked_structural`, `blocked_physical`, or `blocked_policy`. |
| `severity` | `str` | `info`, `warning`, `error`, or `critical`. |
| `evaluation_stage` | `str` | Fixed stage enum: `intake_validation`, `structural_validity`, `physical_consistency`, `clinical_plausibility`, `policy_evaluation`, `multi_series_aggregation`, or `release_control`. |
| `reason_codes` | `list[str]` | Deterministically ordered machine-readable reason codes. |
| `decision_reasons` | `list[dict]` | Structured reasons with `code`, `stage`, `severity`, and `message`. |
| `success_modifiers` | `dict` | Secondary flags such as `near_proof_threshold` and `has_non_blocking_advisory`, plus supporting code/id lists. |
| `decision_lineage` | `list[dict]` | Ordered stage/outcome trace used for replay, debugging, and viewer summaries. |

Ordering rule for `reason_codes` and `decision_reasons`:

- primary class-determining reason first
- same-stage supporting reasons next
- secondary modifiers after that
- advisory/info reasons last, with lexical tie-breaks inside each band

Near-boundary classification is also fixed: only passing measurement-consistency metrics, AI-reconstruction parity metrics, and the tumour-volume tolerance metric participate, and `near_boundary` means "inside the configured guard band of the relevant failure threshold." Exact mask parity remains binary.

For compatibility:

- `decision_summary` remains available
- new logic should use `decision.decision_class`

Minimal structural block reports guarantee at least:

- `decision`
- `decision_summary`
- `release_eligible`
- available case identifiers
- `analysis.input_state_hash` when it can be computed
- structural verification tiers

They do not guarantee full proof artifacts such as reconstruction traces or verified-mask hashes.

## Integration Artifacts

| Artifact | Purpose |
| --- | --- |
| [`../demo_artifacts/dicom_input.json`](../demo_artifacts/dicom_input.json) | Mocked DICOM ingress envelope carrying raw data, AI claim, and embedded Guardian case state |
| [`../demo_artifacts/pacs_submission.json`](../demo_artifacts/pacs_submission.json) | Mocked accepted PACS handoff with verified series and proof contracts |
| [`../demo_artifacts/tamper_fake_tumour_pacs_block.json`](../demo_artifacts/tamper_fake_tumour_pacs_block.json) | Mocked rejected PACS notice for a blocked case |
| [`../demo_artifacts/integration_trace.json`](../demo_artifacts/integration_trace.json) | Boundary trace from DICOM ingress through Guardian decision to PACS egress |
| [`../demo_artifacts/integration_narrative.json`](../demo_artifacts/integration_narrative.json) | Standalone DICOM, placement, and fail-closed narrative |

Mirrored integration surfaces now carry the canonical decision alongside the legacy compatibility fields:

- `pacs_submission.json -> verification_summary.canonical_decision`
- `pacs_block_notice.json -> verification_summary.canonical_decision`
- `integration_trace.json -> pipeline[guardian_layer].canonical_decision`

On fail-soft structural rejects, `dicom_input.json` may legitimately carry `ai_claim_envelope = null` because the mocked ingress artifact is preserved even when proof-only contract members could not be derived from the malformed input.

## Bundle Members

The portable bundle written by [`../mri_guardian/bundle.py`](../mri_guardian/bundle.py) contains:

- `bundle.json`
- `signatures.json`
- `evidence.json`
- `input_state.json`
- `released_state.json`
- `workstation_payload.json`
- `verification_report.json`
- `verification_profile.json`
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
