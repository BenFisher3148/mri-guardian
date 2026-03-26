# ADR-0003: Explicit Claim-To-Proof Contracts

Status: Accepted
Date: 2026-03-26

## Context

The original prototype contained an implicit relation between what the AI claimed and what the Guardian proved. That made reuse and audit discussion harder.

## Decision

Promote the claim-to-proof boundary into explicit interfaces:

- `AIClaimEnvelope`
- `MaskDerivationPolicy`
- `VerificationTolerancePolicy`
- `VerifiedReconstructionTrace`

## Consequences

- Contract hashes can be included in manifests and bundles.
- Cross-domain reuse becomes easier because the proof interface is no longer hard-coded into MRI-specific prose.
- Documentation and data dictionaries can describe stable interfaces rather than inferred relationships.
