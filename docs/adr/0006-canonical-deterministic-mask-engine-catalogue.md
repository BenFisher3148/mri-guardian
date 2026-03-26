# ADR-0006: Canonical Deterministic Mask Engine Catalogue

Status: Accepted
Date: 2026-03-26

## Context

The original prototype used one hard-coded threshold rule for verified-mask derivation. That proved the claim-to-proof concept, but it made the mask layer look like a one-off implementation detail rather than a policy family that could evolve while remaining deterministic and replayable.

At the same time, the repository must not drift toward plugin-style or runtime-swappable release semantics. A released case needs one fixed, policy-bound mask derivation path whose identity and resolved parameters are part of the proof surface.

## Decision

Introduce a fixed in-repo approved deterministic mask-engine catalogue and canonical dispatcher.

- Engine choice is frozen by the selected runtime profile.
- The abstraction supports multiple deterministic policy families over time, not user-selectable algorithm freedom within a released verification context.
- Algorithm identifiers are proof-significant canonical names and must remain stable once published in released profiles or replay artifacts.
- Canonical parameter resolution happens before execution, rejects unknown or algorithm-inapplicable parameters, and serializes one resolved parameter object into the profile, manifest, bundle, and replay surfaces.
- Non-canonical runtime metadata such as timing, internal operation counts, and debug notes is kept separate from proof-significant outputs and excluded from hashes and replay material.

The first approved algorithms are:

- `threshold_on_raw_derived_reconstruction`
- `morphology_constrained_threshold_on_raw_derived_reconstruction`

## Consequences

- The verified-mask layer can now grow across deterministic policy families without weakening replayability or the fail-closed release gate.
- Baseline profiles remain threshold-based, while `brain_t1_morphology_review` serves as the first showcase profile for a richer deterministic family.
- Future mask-policy work must go through the approved catalogue and canonical parameter resolution rather than ad hoc runtime wiring.
