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
| `ai_stack_identity` | `dict | None` | Optional canonical declared AI stack identity bound into `AIClaimEnvelope` when present. This is claim provenance, not an independently verified physical invariant. |
| `release_status` | `str` | `pending_verification` or `released_to_radiologist`. |
| `raw_real` | `list[list[list[float]]]` | Real component of raw k-space. |
| `raw_imag` | `list[list[list[float]]]` | Imaginary component of raw k-space. |
| `ai_reconstruction` | `list[list[list[float]]]` | AI-proposed voxel intensities. |
| `tumor_mask` | `list[list[list[int]]]` | AI-proposed binary tumour mask. |
| `voxel_spacing_mm` | `list[float]` | Physical voxel spacing in millimetres. |
| `tumor_volume_mm3` | `float` | Claimed tumour volume. |
| `study_acquired_at` | `str | None` | Optional canonical UTC study timestamp (`YYYY-MM-DDTHH:MM:SSZ`) used for Level 3 temporal comparison. |
| `study_timestamp_provenance` | `str` | Optional current-case timestamp trust class: `trusted_study_metadata`, `missing`, or `untrusted`. |
| `spatial_geometry` | `dict | None` | Optional geometry object with `frame_of_reference_uid` plus canonical `affine_ijk_to_mm`. |
| `prior_verified_contexts` | `list[dict] | None` | Optional canonical source list of prior Guardian-verified comparators used by multi-prior longitudinal selection and trajectory analysis. |
| `prior_verified_context` | `dict | None` | Optional compact prior Guardian-verified comparator used only by single-series Level 3 longitudinal review. |
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
| `series` | `list[dict]` | Ordered payload series. Each entry contains a unique `series_id` plus the existing per-series raw, AI, mask, volume, embedded profile fields, and optional per-series `ai_stack_identity`. |
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
| `AIClaimEnvelope` | AI-side claim over reconstruction, mask, volume, and optional declared AI stack identity | case identity, `raw_kspace_hash`, AI output hashes, claimed volume, optional `ai_stack_identity` |
| `MaskDerivationPolicy` | Deterministic mapping from raw-derived reconstruction to verified mask | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, proof-significant `algorithm`, and canonical resolved `parameters` |
| `VerificationTolerancePolicy` | Explicit numeric tolerance contract | `profile_name`, `profile_revision`, `profile_digest`, `icsd_policy_id`, `icsd_policy_version`, per-voxel, per-frequency, global-norm, imaginary leakage, and volume tolerances |
| `MeasurementIntegrityProof` | Proof that the raw-derived reconstruction remains anchored to the measurement domain within policy | `raw_kspace_hash`, `derived_reconstruction_hash`, `forward_projection_hash`, `verified`, `metrics`, claim/profile linkage hashes |
| `ReconstructionParityProof` | Proof that the AI reconstruction matches the raw-derived reconstruction within policy | `ai_reconstruction_hash`, `derived_reconstruction_hash`, `verified`, `metrics`, claim/profile linkage hashes |
| `SegmentationPolicyProof` | Proof that the AI mask matches the policy-derived verified mask | `ai_mask_hash`, `verified_mask_hash`, `verified`, `comparison`, claim/profile linkage hashes |
| `VolumeComputationProof` | Proof that the claimed scalar volume matches verified-mask geometry within policy | claimed/computed volume, verified voxel count, voxel volume, tolerance metric, verified flag |
| `ProofCoreEvidenceTrace` | Strictly derived replay spine over the already-materialized proof-stage artefacts | `verification_profile_descriptor_hash`, `ai_claim_envelope_hash`, canonical proof-stage order, `proof_stage_hashes` |
| `PriorVerifiedContext` | Compact canonical summary of one admissible prior Guardian-verified comparator | prior bundle/report hashes, prior profile identity, prior release class, prior verified mask, verified volume, timestamp provenance, and geometry |
| `TemporalConsistencyTrace` | Deterministic single-series Level 3 temporal trace | `comparison_status`, `comparison_scope`, `prior_admissibility`, `chronology_assessment`, `geometry_assessment`, `computed_metrics`, `flags`, `skipped_metrics` |
| `LongitudinalSelectionTrace` | Deterministic selection trace over supplied prior contexts | supplied count, selected count, primary-selection mode, per-prior admissibility and selection outcomes |
| `TemporalTrajectoryTrace` | Deterministic multi-prior trajectory trace | selected count, usable interval count, trajectory class, aggregate metrics, interval summaries, flags, skipped metrics |
| `LongitudinalRegistrationTrace` | Deterministic current-vs-primary registration trace | algorithm identity, transform payload, blocking admissibility, quality metrics |
| `StudyTemporalAggregationTrace` | Deterministic study-level temporal aggregation for `multi_series_v2` | per-series temporal disposition, top flags/reasons, final study temporal disposition, gating summary |
| `CrossSeriesVerificationPolicy` | Study-level multi-series gate contract | `primary_series_id`, canonicalized `supporting_series_ids`, canonicalized `required_series_ids`, geometry rule, corroboration rule |
| `CrossSeriesVerificationTrace` | Study-level trace for geometry, corroboration, and merged release evaluation | `geometry_verification`, `lesion_corroboration`, `merged_release_evaluation` |
| `MultiSeriesVerificationProfileDescriptorSet` | Ordered wrapper around per-series runtime profile descriptors | proof-significant ordered `series_descriptors` list |
| `GuardianIntegrationNarrative` | Hospital-facing integration narrative | DICOM compliance story, hospital placement, fail-closed behaviour |

### AI Stack Identity Shape

- `ai_stack_identity` is optional as a whole, but mandatory in full when present.
- It is canonical claim provenance: the Guardian canonicalizes and binds the producer declaration, but does not independently prove that the declared model stack is true.
- Required fields:
  - `model_version_id`
  - `weights_sha256`
  - `preprocessing_pipeline_sha256`
  - `postprocessing_pipeline_sha256`
  - `input_normalization`
  - `sequence_model_compatibility`
- Declared `ai_stack_identity` string fields must be non-empty and must not carry leading or trailing whitespace.
- SHA-256 fields must be lowercase hex with no leading/trailing whitespace and no `0x` prefix.
- Empty strings are invalid for identifier and digest fields.
- `input_normalization.parameters` must be `JsonLike`:
  - scalar values: `str`, `bool`, `int`, finite `float`
  - composite values: `list[JsonLike]`, `dict[str, JsonLike]`
  - rejected values: `null`, non-finite floats, non-string keys, tuples, sets, bytes, numpy scalars, datetimes, and custom objects
- Floating-point values are serialized only through the repository's canonical JSON encoder. Producers must not rely on language-specific float string representations outside that encoder.
- Input dictionaries may be supplied in any order, but canonicalization relies on the repository serializer. Upstream ordering must not affect canonical output.
- `sequence_model_compatibility` is mandatory inside `ai_stack_identity`:
  - `compatibility_type = exact_sequence_id` requires exact equality with `state["sequence_id"]`
  - `compatibility_type = sequence_family` requires one of the registry-backed `VerificationProfileDescriptor.sequence_family` values and must match the active case/profile context

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

### Segmentation Policy Proof Shape

Released manifests expose `segmentation_policy_proof` in two logical sections:

- `policy`: canonical `algorithm` plus canonical resolved `parameters`
- `comparison`: exact-match outcome over AI vs verified mask, including voxel counts and deltas

The older `mask_verification` naming is retired from the canonical manifest schema.

### Longitudinal Verification Shape

- `PriorVerifiedContext` is admissible only when:
  - `prior_release.release_status = released_to_radiologist`
  - `prior_release.final_disposition = released`
  - `prior_release.decision_class` is one of `verified_clean`, `verified_boundary_hugging`, or `verified_with_advisory`
  - prior and current `sequence_family` match
- Pair timestamp provenance is derived deterministically:
  - any `untrusted` side dominates to `timestamp_untrusted`
  - otherwise any `missing` side dominates to `timestamp_missing`
  - both `verified_bundle` sides yield `verified_bundle_pair`
  - both `trusted_study_metadata` sides yield `trusted_metadata_pair`
  - otherwise mixed trusted classes yield `mixed_provenance`
- `comparison_status` is one of `not_evaluated`, `aligned`, `degraded`, or `unsuitable`.
- `comparison_scope` is one of `not_evaluated`, `full`, `volume_only`, or `none`.
- Interval bands are policy comparability bands, not clinical expectation bands:
  - `<= 730` days may be `aligned`
  - `731-3650` days is `degraded`
  - `> 3650` days is `unsuitable` and suppresses temporal comparison
- Geometry comparability classes are:
  - `exact_match`
  - `volume_basis_only`
  - `insufficient_geometry_trust`
- Spatial metrics use fixed definitions:
  - overlap is `intersection_over_union`
  - centroid shift is the mm distance between global positive-voxel centroids after applying `affine_ijk_to_mm`
- `skipped_metrics` is a closed `{metric, reason_code}` list and should be consumed as machine-readable data rather than free text.

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
| `pipeline_phase` | `str` | Fixed assurance-flow phase enum: `intake_validation`, `structural_validity`, `proof_core`, `clinical_plausibility`, `policy_evaluation`, `multi_series_aggregation`, or `release_control`. |
| `proof_stage` | `str \| null` | Optional proof-core stage identifier: `MeasurementIntegrityProof`, `ReconstructionParityProof`, `SegmentationPolicyProof`, or `VolumeComputationProof`. Present only when `pipeline_phase = proof_core`. |
| `reason_codes` | `list[str]` | Deterministically ordered machine-readable reason codes. |
| `decision_reasons` | `list[dict]` | Structured reasons with `code`, `pipeline_phase`, optional `proof_stage`, `severity`, and `message`. |
| `success_modifiers` | `dict` | Secondary flags such as `near_proof_threshold` and `has_non_blocking_advisory`, plus supporting code/id lists. |
| `decision_lineage` | `list[dict]` | Ordered phase/outcome trace used for replay, debugging, and viewer summaries; proof-core entries also carry `proof_stage`. |

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
- structural verification pipeline phases

They do not guarantee full proof-core artifacts such as stage contracts or verified-mask hashes.

## Operational Subphases

Single-series `verification_report.json` outputs include an additive derived
`verification_pipeline.operational_subphases` block.

- It is forbidden on parent `multi_series_v2` reports.
- It is an operational audit/performance view only.
- It must never alter primary decision ownership, which remains
  `decision.pipeline_phase` plus optional `decision.proof_stage`.
- `blocking = true` on both subphases means a non-passing subphase is
  release-preventing within operational flow. It does not replace canonical
  ownership.

Top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `schema` | `str` | Fixed derived schema identifier for operational subphases. |
| `ordering_semantics` | `str` | Explains that the block is derived and does not replace canonical decision ownership. |
| `subphase_order` | `list[str]` | Fixed to `["pre_check", "full_proof"]`. |
| `subphases` | `dict` | Frozen subphase mapping. |

`pre_check` fields:

- `title`
- `status`
- `summary`
- `blocking`
- `checks`

Frozen `pre_check.status` vocabulary:

- `passed`
- `failed`
- `advisory`

Frozen `pre_check.checks` order:

- `structure_validation`
- `hash_checks`
- `shape_consistency`
- `quick_anomaly_detection`

`hash_checks` means consistency checks over already-materialized deterministic
hash surfaces. It is not a free-form integrity-verification hook.

Frozen per-check statuses:

- `structure_validation`: `passed | failed`
- `hash_checks`: `passed | failed | not_evaluated`
- `shape_consistency`: `passed | failed | not_evaluated`
- `quick_anomaly_detection`: `passed | advisory | not_evaluated`

`quick_anomaly_detection` boundaries:

- It may consume only already-available low-cost intake-adjacent or structural
  report facts.
- It must not trigger heavy computation or recompute proof artifacts.
- It must not consume proof-core outputs, clinical-plausibility outputs, or
  downstream phase outputs.
- It must not introduce semantic comparison against AI-derived content.
- It must not introduce thresholds that duplicate or approximate downstream
  proof decisions.
- It must not create a new blocking path.

`full_proof` fields:

- `title`
- `status`
- `summary`
- `blocking`
- `proof_stage_order`
- `proof_stages`

Frozen `full_proof.status` vocabulary:

- `passed`
- `failed`
- `not_evaluated`

`full_proof.proof_stages` always materializes in canonical proof-stage order:

- `MeasurementIntegrityProof`
- `ReconstructionParityProof`
- `SegmentationPolicyProof`
- `VolumeComputationProof`

Frozen `full_proof.proof_stages[].status` vocabulary:

- `passed`
- `failed`
- `not_evaluated`

`not_evaluated` on `full_proof` means canonical `proof_core` was not entered or
not reached. It does not imply silent success or irrelevance.

The viewer-facing operational labels:

- `MeasurementIntegrityProof -> FFT / reprojection`
- `ReconstructionParityProof -> parity validation`
- `SegmentationPolicyProof -> segmentation proof`
- `VolumeComputationProof -> volume computation`

are presentation labels only. They must never replace canonical proof-stage
identifiers in reports, decisions, journals, or ADR terminology.

## Decision Journal

`decision_journal.json` is an additive operational audit artifact derived only from `verification_report.json` plus optional transparency receipt context.

- It is emitted wherever `verification_report.json` is emitted.
- It is bundled only when the case shape already produces a portable bundle.

Top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | `str` | Decision-journal schema version. |
| `journal_type` | `str` | Fixed to `DecisionJournal`. |
| `case_id` | `str` | Stable case or study identifier when available; falls back to `unknown-case` on minimal structural rejects with no usable identifier. |
| `case_kind` | `str` | `single_series_v1` or `multi_series_v2`. |
| `profile_identity` | `dict` | Single-profile identity or compact multi-series descriptor-set identity. |
| `verification_report_hash` | `str` | Canonical hash of the source `verification_report.json`. |
| `final_disposition` | `str` | `released` or `blocked`. Mirrors the canonical decision. |
| `decision_class` | `str` | Canonical case-level decision class copied from `decision.decision_class`. |
| `severity` | `str` | Canonical case-level severity copied from `decision.severity`. |
| `reason_codes` | `list[str]` | Canonical case-level reason summary copied from `decision.reason_codes`. |
| `journal_summary.transparency_status` | `str` | `recorded`, `absent`, or `not_applicable`. |
| `journal_summary.residual_risk` | `dict` | Structured case-level residual risk with fixed `classification` and bounded `summary`. |
| `entries` | `list[dict]` | Canonically ordered event summaries. |

Journal entry fields:

| Field | Type | Description |
| --- | --- | --- |
| `event_id` | `str` | Frozen event identifier. |
| `sequence` | `int` | Canonical event order. It is not a timestamp or discovery order. |
| `pipeline_phase` | `str` | Canonical assurance-flow phase reference for the journaled outcome. |
| `proof_stage` | `str` | Optional proof-stage identifier for proof-core outcome summaries only. |
| `status` | `str` | Frozen journal status enum. |
| `summary` | `str` | Canonical bounded event summary text. |
| `reason_codes` | `list[str]` | Entry-scoped subset of the case-level `reason_codes`. |
| `evidence_paths` | `list[str]` | Canonical references to already-emitted artifacts or `verification_report.json#/...` surfaces. Emit `[]` when no direct evidence surface exists. |
| `hazards_triggered` | `list[dict]` | Stable mapped hazards only. Emit `[]` when no stable mapping exists. |
| `mitigations_applied` | `list[dict]` | Stable mapped controls relevant to the surfaced path only. |
| `residual_risk` | `dict` | Structured event-level residual risk with fixed `classification` and bounded `summary`. |

Frozen `event_id` vocabulary:

- `input_received`
- `structural_validation`
- `reconstruction`
- `parity_checks`
- `segmentation_verification`
- `volume_proof`
- `advisory_heuristics`
- `multi_series_aggregation`
- `final_decision`
- `transparency_anchor`

Frozen `status` vocabulary:

- `not_evaluated`
- `passed`
- `failed`
- `advisory`
- `recorded`
- `absent`
- `not_applicable`

Status mapping notes:

- `final_decision` uses `advisory` for released `verified_boundary_hugging` and `verified_with_advisory` outcomes, `passed` for `verified_clean`, and `failed` for blocked outcomes.
- `multi_series_aggregation` uses `advisory` when the study is released but the aggregated required-series path still surfaces review-relevant advisory or near-boundary signals.

Residual-risk classifications:

- `bounded`
- `workflow_external`
- `operator_relevant`

`transparency_anchor` is always present as a journaled anchoring outcome. Its status records whether anchoring was present, absent from the current artifact surface, or not applicable to the current non-final path. It does not imply that anchoring positively occurred when the status is not `recorded`.

For `multi_series_v2`, only the `multi_series_aggregation` entry may include `child_outcomes`, and each child outcome is restricted to:

- `series_id`
- `decision_class`
- `final_disposition`
- `gating_status`

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

When temporal context is present on single-series cases, the mocked PACS and integration summaries also mirror:

- `temporal_comparison_status`
- `temporal_comparison_scope`
- `temporal_spatial_comparison_mode`
- `temporal_primary_selection_mode`
- `temporal_selected_prior_count`
- `temporal_trajectory_class`
- `temporal_registration_status`
- `temporal_top_flags`
- `temporal_trajectory_flags`
- `temporal_top_reason_codes`

On fail-soft structural rejects, `dicom_input.json` may legitimately carry `ai_claim_envelope = null` because the mocked ingress artifact is preserved even when proof-only contract members could not be derived from the malformed input.

## Bundle Members

The portable bundle written by [`../mri_guardian/bundle.py`](../mri_guardian/bundle.py) contains:

- `bundle.json`
- `input_state.json`
- `released_state.json`
- `workstation_payload.json`
- `verification_report.json`
- `decision_journal.json`
- `evidence.json`
- `transparency-prefix.jsonl`

Inside `bundle.json`, Guardian Bundle V1 carries these canonical top-level sections:

- `bundle_type = "dpa-guardian-bundle"`
- `bundle_schema_version = "1"`
- `claim`
- `proof_context`
- `replay_inputs`
- `release_artifacts`
- `integrity_manifest`
- `transparency_anchor`
- optional `attachments`

`integrity_manifest` binds all declared bundle members, while replay verification is evaluated only over the proof-critical deterministic surfaces plus the declared release artifacts. Auxiliary `attachments` are never proof-critical in V1.

## Notes on Determinism

- Hash-bearing fields assume canonical JSON serialisation.
- Arrays are represented as nested lists to keep bundle and evidence payloads deterministic.
- Changing field order, numeric rounding rules, or member ordering may break replay verification.

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`adr/0003-explicit-claim-proof-contracts.md`](adr/0003-explicit-claim-proof-contracts.md)
- [`adr/0004-replayable-evidence-bundles.md`](adr/0004-replayable-evidence-bundles.md)
