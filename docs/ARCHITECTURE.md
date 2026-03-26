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
- The verified mask is always derived from the raw-derived reconstruction by a deterministic mask policy.
- Release is fail-closed for the Guardian handoff. If blocking proof fails, nothing verified is forwarded downstream.
- Identical inputs must produce identical released state, evidence, bundle members, and replay hashes.

## Major Components

| Module | Responsibility |
| --- | --- |
| [`mri_guardian/harness.py`](../mri_guardian/harness.py) | Core verification transition, invariant profile, deterministic reconstruction, release gate, and manifest generation. |
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
6. Derive the verified mask using the deterministic mask policy and compare it with the AI mask.
7. Recompute tumour volume from verified voxels and voxel spacing.
8. If Level 1 and Level 2 pass, emit:
   - `verified_manifest`
   - `workstation_payload`
   - ICSD evidence
   - transparency receipt
   - portable bundle
9. If Level 1 or Level 2 fails, block the Guardian handoff and emit explicit failure reasons.

## Explainability Layers

| Tier | Purpose | Blocking |
| --- | --- | --- |
| Level 1 | Structural validity for ICSD and state-shape correctness | Yes |
| Level 2 | Physical consistency from raw k-space to reconstruction, mask, and volume | Yes |
| Level 3 | Optional clinical plausibility heuristics | No |

## State Model

The project operates on a simple two-state release model:

- `pending_verification`: input captured, no released payload is allowed.
- `released_to_radiologist`: `verified_manifest` and `workstation_payload` exist and must match the deterministic proof trace exactly.

Released cases are only valid if the mathematical trace still verifies when recomputed.

## Artifact Model

The Guardian Layer emits several artifact classes:

- state artifacts: `released_state.json`, `workstation_payload.json`
- proof artifacts: `evidence.json`, `bundle_verification.json`, `.mri-guard-bundle`
- contract artifacts: `ai_claim_envelope.json`, `verified_reconstruction_trace.json`, `mask_derivation_policy.json`, `verification_tolerance_policy.json`
- integration artifacts: `dicom_input.json`, `integration_narrative.json`, `pacs_submission.json`, `integration_trace.json`
- performance artifacts: `performance_profile.json`

## Architecture References

- [`SECURITY.md`](SECURITY.md)
- [`PERFORMANCE.md`](PERFORMANCE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`DATA_DICT.md`](DATA_DICT.md)
- [`adr/README.md`](adr/README.md)
