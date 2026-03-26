# ADR-0004: Replayable Evidence Bundles

Status: Accepted
Date: 2026-03-26

## Context

Hospital and regulatory reviewers need portable proof, not proof that only works while the original service is running.

## Decision

Emit deterministic `.mri-guard-bundle` archives that contain enough state, evidence, transcript, and transparency context to verify the handoff offline and replay it deterministically.

## Consequences

- Accepted cases become shareable proof artifacts.
- Verification can be repeated away from the live service.
- Bundle member ordering, canonical encoding, and replay inputs become compatibility-sensitive system contracts.
