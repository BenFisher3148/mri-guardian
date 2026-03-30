# ADR-0015: Deterministic Decision Journal V1

Status: Accepted
Date: 2026-03-30

## Context

The repository already exposed:

- canonical proof contracts
- a canonical `decision` object
- deterministic bundle transcripts
- transparency-log anchoring

That was enough to prove replay integrity, but it still left an operational gap:
reviewers could see the proof surfaces and the final decision, yet there was no
single deterministic artifact that explained the ordered case outcome in an
audit-friendly way across clean releases, advisory releases, proof failures,
policy blocks, structural rejects, and study-level aggregation.

The repository needed one additive operational artifact that:

- is replayable and hash-stable
- does not become a shadow proof engine
- does not rewrite existing proof or decision contracts
- can be emitted for report-only flows as well as bundled release flows
- keeps hazard, mitigation, and residual-risk language bounded and queryable

## Decision

Introduce `decision_journal.json` as a strictly derived operational audit
artifact.

- Emit `decision_journal.json` everywhere `verification_report.json` is emitted.
- Include `decision_journal.json` in bundles only when a bundle already exists
  for that case shape.
- The journal builder is pure:
  - its admissible inputs are `verification_report` plus optional transparency
    receipt context
  - it must not read the filesystem, bundle members, runtime state, or ambient
    configuration
- Internal journal assembly must emit only the frozen event vocabulary:
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
- Entry order is canonical and phase-aligned. `sequence` is deterministic entry
  order, not timestamp order or discovery order.
- Journal entry status is a closed enum:
  - `not_evaluated`
  - `passed`
  - `failed`
  - `advisory`
  - `recorded`
  - `absent`
  - `not_applicable`
- `reason_codes` remain scoped:
  - top-level `reason_codes` are the canonical case-level summary
  - entry-level `reason_codes` must be a subset of the surfaced report reasons
    relevant to that entry
- `evidence_paths` may reference only:
  - already-emitted artifact names; or
  - canonical `verification_report.json#/...` surfaces
  - when no direct evidence surface exists, emit `[]`
- `hazards_triggered` and `mitigations_applied` are mapping-driven only:
  - if no stable hazard mapping exists for the surfaced path, emit `[]`
  - mitigations may list only controls relevant to the surfaced path
- Residual risk is structured, not free-form:
  - `classification`: `bounded | workflow_external | operator_relevant`
  - `summary`: fixed deterministic template text
  - classification selection is mapping-driven from final disposition and path,
    not hand-authored per case
- `transparency_anchor` is always present as a journaled anchoring outcome:
  - `recorded` when anchoring is present
  - `absent` when anchoring is expected but not present in the current artifact
    surface
  - `not_applicable` for non-final report outcomes where anchoring is not part
    of the path
- Multi-series parent journals may reference child outcomes only in compact
  form, with these fields only:
  - `series_id`
  - `decision_class`
  - `final_disposition`
  - `gating_status`
  - parent journals must not inline nested child journals or full child
    reasoning

## Consequences

- The repository now has one deterministic operational narrative surface that
  sits beside the proof contracts rather than competing with them.
- Report-only flows gain the same audit artifact shape as bundled release
  flows, without implying that every case produces a bundle.
- Replay verification becomes stricter because `decision_journal.json` must
  regenerate exactly from the canonical report plus optional receipt context.
- The event vocabulary, status vocabulary, residual-risk shape, and child
  outcome schema are now explicit invariants rather than presentation choices.
