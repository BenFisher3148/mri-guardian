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

## Profile Registry

The architecture separates three identities:

- Guardian runtime profile identity: `profile_name`, `profile_revision`, `profile_digest`
- ICSD invariant policy identity: `icsd_policy_id`, `icsd_policy_version`
- Descriptive metadata: `sequence_family`, `clinical_mode`

The canonical `VerificationProfileDescriptor` is a first-class artifact. It is serialized with stable field ordering, hashed with SHA-256 over an explicit digest scope, and embedded in reports, released manifests, replay descriptors, and portable bundles.

## Approved Mask Algorithm Catalogue

The verified-mask layer uses a fixed approved algorithm catalogue and canonical dispatcher.

- No user-installed plugins, module scanning, or environment discovery is supported.
- The selected runtime profile freezes the algorithm family for a released case.
- Algorithm identifiers are proof-significant names and must remain stable once published.
- Canonical parameter resolution happens before execution, rejects unknown or algorithm-inapplicable parameters, and serializes one resolved parameter object into all proof surfaces.
- Runtime metadata such as timing, operation counts, and debug notes is explicitly non-canonical and excluded from manifests, bundle hashes, and replay material.
- The connected-component family uses explicit `faces` connectivity with fixed queue-based BFS traversal and a fixed face-neighbor order. Component labels are internal deterministic execution state only.

## Major Components

| Module | Responsibility |
| --- | --- |
| [`mri_guardian/harness.py`](../mri_guardian/harness.py) | Core verification transition, deterministic reconstruction, release gate, and manifest generation. |
| [`mri_guardian/mask_engine.py`](../mri_guardian/mask_engine.py) | Fixed approved mask-algorithm catalogue, canonical parameter normalization, and deterministic dispatcher. |
| [`mri_guardian/profiles.py`](../mri_guardian/profiles.py) | Runtime profile registry, canonical descriptor hashing, and mapping to ICSD invariant policy identities. |
| [`mri_guardian/contracts.py`](../mri_guardian/contracts.py) | Explicit claim-to-proof interfaces: `AIClaimEnvelope`, `MaskDerivationPolicy`, `VerificationTolerancePolicy`, and `VerifiedReconstructionTrace`. |
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
8. If deterministic proof passes and the selected profile permits final release, emit:
   - `verified_manifest`
   - `workstation_payload`
   - ICSD evidence
   - transparency receipt
   - portable bundle
9. If proof fails, a blocking heuristic fails, or the profile is non-final, block the Guardian handoff and emit explicit failure reasons with separate `proof_failure`, `heuristic_block`, or `profile_policy_block` categories.

## Explainability Layers

| Tier | Purpose | Blocking |
| --- | --- | --- |
| Level 1 | Structural validity for ICSD and state-shape correctness | Yes |
| Level 2 | Physical consistency from raw k-space to reconstruction, mask, and volume | Yes |
| Level 3 | Profile-scoped clinical plausibility heuristics | Profile-dependent |

## State Model

The project operates on a simple two-state release model:

- `pending_verification`: input captured, no released payload is allowed.
- `released_to_radiologist`: `verified_manifest` and `workstation_payload` exist and must match the deterministic proof trace exactly.

Released cases are only valid if the mathematical trace still verifies when recomputed.

## Artifact Model

The Guardian Layer emits several artifact classes:

- state artifacts: `released_state.json`, `workstation_payload.json`
- proof artifacts: `evidence.json`, `bundle_verification.json`, `.mri-guard-bundle`
- contract artifacts: `verification_profile.json`, `verification_profile_descriptor.json`, `ai_claim_envelope.json`, `verified_reconstruction_trace.json`, `mask_derivation_policy.json`, `verification_tolerance_policy.json`
- integration artifacts: `dicom_input.json`, `integration_narrative.json`, `pacs_submission.json`, `integration_trace.json`
- performance artifacts: `performance_profile.json`

## Architecture References

- [`SECURITY.md`](SECURITY.md)
- [`PERFORMANCE.md`](PERFORMANCE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`DATA_DICT.md`](DATA_DICT.md)
- [`adr/README.md`](adr/README.md)
