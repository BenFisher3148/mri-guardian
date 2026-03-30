# Architecture

This repository implements an MRI Guardian Layer: a deterministic verification proxy that sits between a raw MRI reconstruction AI and the downstream PACS or radiologist workstation.

## System Context

```text
MRI scanner / raw k-space
        |
        v
AI reconstruction + AI mask + AI volume claim
        |
        v
MRI Guardian Layer
  - reconstruct from raw k-space
  - verify structural validity
  - verify physical consistency
  - apply optional clinical heuristics
  - emit evidence, bundle, and verified workstation payload
        |
        v
PACS / VNA / radiologist workstation
```

The Guardian Layer is intentionally non-invasive. It does not replace the MRI scanner, the AI model, or the PACS. It controls only the verified handoff.

## Core Invariants

- The PACS/workstation path never receives AI-authored pixels directly from the Guardian path.
- The released reconstruction is always recomputed from raw k-space inside the verification layer.
- The verified mask is always derived from the raw-derived reconstruction by a deterministic runtime-profile mask rule.
- Mask derivation uses a fixed approved in-repo algorithm catalogue rather than environment-discovered plugins or runtime-swappable release engines.
- Release is fail-closed for the Guardian handoff. If blocking proof fails, nothing verified is forwarded downstream.
- Identical inputs must produce identical released state, evidence, bundle members, and replay hashes.
- The replay path must bind to the embedded runtime profile descriptor by `profile_name`, `profile_revision`, and `profile_digest`, not the process default.
- Canonical `ai_stack_identity` is claim-bound declared provenance when present. It affects claim identity hashes, but it is not an independently verified physical invariant of the proof process.
- New monitoring logic must treat the additive `decision` object as canonical; `decision_summary` remains compatibility-only.
- Single-series longitudinal verification is optional, non-blocking in V1, and must not change the Level 2 release gate.
- Longitudinal interval bands are comparability-confidence policy bands, not clinical expectation bands.
- In `multi_series_v2`, all payload series are individually verified and reported, but only `required_series_ids` participate in merged release gating.
- In `multi_series_v2`, study order is proof-significant for canonical serialization, stable diffs, and deterministic replay.

## Profile Registry

The architecture separates three identities:

- Guardian runtime profile identity: `profile_name`, `profile_revision`, `profile_digest`
- ICSD invariant policy identity: `icsd_policy_id`, `icsd_policy_version`
- Descriptive metadata: `sequence_family`, `clinical_mode`

The canonical `VerificationProfileDescriptor` is a first-class artifact. It is serialized with stable field ordering, hashed with SHA-256 over an explicit digest scope, and embedded in reports, released manifests, replay descriptors, and portable bundles.

The claim layer now carries two different AI identities:

- `ai_model_id`: coarse upstream model-family identifier
- `ai_stack_identity`: optional exact declared model-stack identity bound into `AIClaimEnvelope`

`ai_stack_identity` is canonicalized when present, but it remains a producer declaration. Guardian proves reconstruction, mask, and volume consistency against raw data; it does not independently verify the truth of the declared model stack itself.

## Approved Mask Algorithm Catalogue

The verified-mask layer uses a fixed approved algorithm catalogue and canonical dispatcher.

- No user-installed plugins, module scanning, or environment discovery is supported.
- The selected runtime profile freezes the algorithm family for a released case.
- Algorithm identifiers are proof-significant names and must remain stable once published.
- Canonical parameter resolution happens before execution, rejects unknown or algorithm-inapplicable parameters, and serializes one resolved parameter object into all proof surfaces.
- Runtime metadata such as timing, operation counts, and debug notes is explicitly non-canonical and excluded from manifests, bundle hashes, and replay material.
- The connected-component family uses explicit `faces` connectivity with fixed queue-based BFS traversal and a fixed face-neighbor order. Component labels are internal deterministic execution state only.
- The slice-consistency family uses deterministic 2D in-slice components, fixed queue-based BFS traversal with fixed face-neighbor order, and an undirected layered overlap graph whose edges only cross adjacent slices `z` and `z+1`.
- Slice-consistency overlap is defined by non-empty intersection of adjacent components' `(y, x)` footprint sets. Whole-component retention is intentional: survival is governed by qualifying-run membership, not by a shared-overlap core.

## Major Components

| Module | Responsibility |
| --- | --- |
| [`mri_guardian/harness.py`](../mri_guardian/harness.py) | Core verification transition, deterministic reconstruction, release gate, and manifest generation. |
| [`mri_guardian/mask_engine.py`](../mri_guardian/mask_engine.py) | Fixed approved mask-algorithm catalogue, canonical parameter normalization, and deterministic dispatcher. |
| [`mri_guardian/profiles.py`](../mri_guardian/profiles.py) | Runtime profile registry, canonical descriptor hashing, and mapping to ICSD invariant policy identities. |
| [`mri_guardian/contracts.py`](../mri_guardian/contracts.py) | Explicit claim-to-proof interfaces: `AIClaimEnvelope`, `MaskDerivationPolicy`, `VerificationTolerancePolicy`, `VerifiedReconstructionTrace`, `PriorVerifiedContext`, and `TemporalConsistencyTrace`. |
| [`mri_guardian/integration.py`](../mri_guardian/integration.py) | Mocked DICOM ingress, PACS egress, and hospital integration narrative. |
| [`mri_guardian/bundle.py`](../mri_guardian/bundle.py) | Portable `.mri-guard-bundle` writer and deterministic replay verifier. |
| [`mri_guardian/timing.py`](../mri_guardian/timing.py) | Benchmark ladder, worst-case bound derivation, and streaming-validation strategy. |
| [`mri_guardian/viewer.py`](../mri_guardian/viewer.py) | Human-facing HTML report for side-by-side review. |
| [`mri_guardian/transparency.py`](../mri_guardian/transparency.py) | Append-only transparency log used to anchor evidence receipts. |
| [`mri_guardian/demo.py`](../mri_guardian/demo.py) | Deterministic artifact generator for demo and documentation. |

## Verification Flow

1. Ingest raw k-space, AI reconstruction, AI mask, voxel spacing, and claimed tumour volume.
2. Run Level 1 structural validity checks over identifiers, tensor shape, numeric finiteness, and release-state shape.
3. Reconstruct the image from raw k-space with an inverse FFT.
4. Reproject that reconstruction into frequency space and compare it with the supplied measurements.
5. Compare the AI reconstruction with the raw-derived reconstruction using explicit per-voxel and global-norm tolerances.
6. Resolve the runtime profile's mask policy canonically, derive the verified mask with the approved dispatcher, and compare it with the AI mask.
7. Recompute tumour volume from verified voxels and voxel spacing.
8. Optionally evaluate Level 3 longitudinal context:
   - deterministically canonicalize `prior_verified_contexts` or the legacy single `prior_verified_context`
   - select an admissible primary prior with `LongitudinalSelectionTrace`
   - compute `TemporalTrajectoryTrace` over the selected multi-prior history
   - if needed, attempt deterministic registration for current-vs-primary spatial comparison and record `LongitudinalRegistrationTrace`
   - emit `TemporalConsistencyTrace` for the current-vs-primary contract surface
   - current-case timestamps accept `trusted_study_metadata`, `missing`, or `untrusted`
   - prior timestamps accept `verified_bundle`, `trusted_study_metadata`, `missing`, or `untrusted`
   - exact geometry requires matching shape, voxel spacing, frame of reference, and canonicalized affine
   - affine-only mismatches may upgrade into deterministic `registered` comparison when registration quality passes the profile floor
   - geometry-limited cases may otherwise fall back to scalar `volume_only` comparison
   - interval `> 3650` days suppresses all temporal comparison by policy
9. If deterministic proof passes and the selected profile permits final release, emit:
   - `verified_manifest`
   - `workstation_payload`
   - ICSD evidence
   - transparency receipt
   - portable bundle
10. If proof fails, a blocking heuristic fails, or the profile is non-final, block the Guardian handoff and emit explicit failure reasons with separate `proof_failure`, `heuristic_block`, or `profile_policy_block` categories.

## Decision Contract

The report contract now separates hard release semantics from richer monitoring semantics.

- `release_eligible` and the released/blocked handoff remain the only downstream release gate.
- `decision.decision_class` is the canonical semantic class for dashboards, summaries, and trend analysis.
- `decision_summary` remains available for compatibility, but it should not be the source of truth for new logic.

Fixed `decision.evaluation_stage` values:

- `intake_validation`
- `structural_validity`
- `physical_consistency`
- `clinical_plausibility`
- `policy_evaluation`
- `multi_series_aggregation`
- `release_control`

Deterministic reason ordering:

- primary class-determining reason
- same-stage supporting reasons
- secondary modifiers
- advisory/info reasons, then lexical tie-breaks

`verified_boundary_hugging` is restricted to passing proof metrics that land within the configured guard band of a failure threshold. The current eligible metrics are the measurement-consistency metrics, AI-reconstruction parity metrics, and the tumour-volume tolerance metric. Exact mask parity remains binary.

## Multi-Series V2

`multi_series_v2` adds a study-level container on top of the existing per-series proof path.

- one study contains an ordered `series` list plus one `cross_series_policy`
- all payload series are individually verified and reported
- `required_series_ids` is the only authoritative merged-release gating set
- duplicate payload `series_id` values are an early study-shape validation failure
- `supporting_series_ids` must be non-empty for `multi_series_v2`; callers must not auto-downgrade into the single-series path
- payload series outside `required_series_ids` are auxiliary for this milestone
- caller-provided `required_series_ids` and `supporting_series_ids` are accepted, then canonicalized into outer study order; caller order is not proof-significant
- required-series geometry is checked only on canonicalized voxel spacing plus exact voxel shape
- positive-lesion corroboration is a study-level gate only when required verified masks are non-empty
- corroboration overlap is exact coordinate intersection count `> 0`; adjacency without identical coordinates does not count

## Explainability Layers

| Tier | Purpose | Blocking |
| --- | --- | --- |
| Level 1 | Structural validity for ICSD and state-shape correctness | Yes |
| Level 2 | Physical consistency from raw k-space to reconstruction, mask, and volume | Yes |
| Level 3 | Profile-scoped clinical plausibility heuristics | Profile-dependent |

Within Level 3, longitudinal verification is now a first-class single-series sub-path. It consumes one or more `PriorVerifiedContext` payloads, produces `LongitudinalSelectionTrace`, `TemporalTrajectoryTrace`, `LongitudinalRegistrationTrace`, and `TemporalConsistencyTrace`, and surfaces `comparison_status`, `comparison_scope`, and `spatial_comparison_mode` explicitly so viewer, integration, and replay code do not infer temporal meaning from prose.

In `multi_series_v2`, each payload series now carries its own optional temporal inputs. The study path verifies those per-series longitudinal results first, then aggregates them into `StudyTemporalAggregationTrace`. Only required series participate in study-level temporal gating semantics; auxiliary series remain informational.

## State Model

The project operates on a simple two-state release model:

- `pending_verification`: input captured, no released payload is allowed.
- `released_to_radiologist`: `verified_manifest` and `workstation_payload` exist and must match the deterministic proof trace exactly.

In `multi_series_v2`, released study manifests include all payload series in proof-significant order and record each series role, gating status, proof status, and downstream release flag. Released study workstation payloads include only downstream-allowed series plus an explicit withheld-series summary.

Released cases are only valid if the mathematical trace still verifies when recomputed.

## Artifact Model

The Guardian Layer emits several artifact classes:

- state artifacts: `released_state.json`, `workstation_payload.json`
- proof artifacts: `evidence.json`, `bundle_verification.json`, `.mri-guard-bundle`
- contract artifacts: `verification_profile.json`, `verification_profile_descriptor.json`, `ai_claim_envelope.json`, `verified_reconstruction_trace.json`, `mask_derivation_policy.json`, `verification_tolerance_policy.json`, plus longitudinal artifacts such as `prior_verified_contexts.json`, `prior_verified_context.json`, `longitudinal_selection_trace.json`, `temporal_trajectory_trace.json`, `longitudinal_registration_trace.json`, and `temporal_consistency_trace.json` when longitudinal context is supplied
- integration artifacts: `dicom_input.json`, `integration_narrative.json`, `pacs_submission.json`, `integration_trace.json`
- performance artifacts: `performance_profile.json`

## Architecture References

- [`SECURITY.md`](SECURITY.md)
- [`PERFORMANCE.md`](PERFORMANCE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`DATA_DICT.md`](DATA_DICT.md)
- [`adr/README.md`](adr/README.md)
