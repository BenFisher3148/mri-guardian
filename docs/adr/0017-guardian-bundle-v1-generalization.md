# ADR-0017: Guardian Bundle V1 Generalization

Status: Accepted
Date: 2026-03-31

## Context

The original replayable evidence bundle was MRI-branded and coupled archive naming, transcript surfaces, and integrity catalog structure to the prototype's first domain. The repository now needs one canonical, replayable, integrity-bound evidence container that can carry Guardian-verified outputs across domains without changing proof contracts or decision logic.

## Decision

Adopt Guardian Bundle V1 as the canonical portable artifact format:

- archive suffix: `.guardian-bundle`
- embedded identity: `bundle_type = "dpa-guardian-bundle"`
- schema identity: `bundle_schema_version = "1"`

Guardian Bundle V1 keeps a strict surface split:

- `claim` is declarative and identity-focused
- `proof_context` records only proof-defining or proof-recording outputs
- `replay_inputs` carries only deterministic rerun inputs that are not themselves proof outcomes
- `release_artifacts` is only a descriptor table for emitted artifacts
- `attachments` are optional auxiliary members and are never proof-critical in V1

Guardian Bundle V1 also freezes these compatibility-sensitive rules:

- `transparency-prefix.jsonl` is the minimal canonical UTF-8 JSONL prefix required to recreate the anchored entry context
- `bundle_descriptor_hash` is computed from canonical `bundle.json` with only `integrity_manifest` redacted and no other exclusions
- archive member order, `integrity_manifest.member_order`, and `integrity_manifest.member_hashes` order must agree exactly

The canonical bundle removes `transcript.json`, `replay.json`, `signatures.json`, and the full bundled transparency-log snapshot. If `transcript.json` is retained at all, it is only an external or demo-oriented artifact rather than a canonical bundle member. Instead the bundle carries:

- canonical top-level sections in `bundle.json`
- `input_state.json`
- `released_state.json`
- `workstation_payload.json`
- `verification_report.json`
- `decision_journal.json`
- `evidence.json`
- `transparency-prefix.jsonl`
- `integrity_manifest` inside `bundle.json`

`integrity_manifest` binds all declared members, while replay verification is evaluated only over proof-critical deterministic surfaces plus reproducible release artifacts.

ADR-0004 remains the historical record of the earlier MRI-specific replayable bundle.

## Consequences

- Portable bundles are now domain-agnostic at the core schema level.
- Archive member ordering, descriptor hashing, and integrity/replay boundaries are explicit compatibility-sensitive contracts.
- Attachment mutation can invalidate whole-bundle integrity without changing proof replay results.
- Existing proof contracts, decision schemas, and verification logic remain unchanged.
