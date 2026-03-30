# ADR-0016: Operational Pre-Check and Full-Proof Layer V1

Status: Accepted
Date: 2026-03-30

## Context

The repository already had a clear canonical assurance model:

- fixed `verification_pipeline` phases
- fixed `proof_core` stage order
- canonical decision ownership through `decision.pipeline_phase` plus optional
  `decision.proof_stage`
- a derived deterministic decision journal

That was enough for proof attribution and replay, but it still left one
review-facing gap: the runtime story did not explicitly distinguish a realistic
early-screening path from the heavier full-proof path.

The repository needed a way to make that operational flow visible without:

- renaming canonical phases
- introducing new proof stages
- turning the journal into a second schema
- inventing a fake study-level pre-check/full-proof narrative for parent
  multi-series reports

## Decision

Add `verification_pipeline.operational_subphases` as an additive derived block
on single-series reports only.

- It is forbidden on parent `multi_series_v2` reports.
- It is interpretive and performance-facing only.
- It must never alter primary decision ownership. Canonical ownership remains
  `decision.pipeline_phase` plus optional `decision.proof_stage`.

Frozen subphase order:

- `pre_check`
- `full_proof`

### Pre-Check

`pre_check` always materializes these frozen checks in this order:

- `structure_validation`
- `hash_checks`
- `shape_consistency`
- `quick_anomaly_detection`

Rules:

- `structure_validation` derives only from intake and structural-validity facts.
- `hash_checks` means consistency checks over already-materialized deterministic
  hash surfaces. It is not a free-form integrity hook.
- `shape_consistency` derives only from existing numeric-tensor, geometry,
  spacing, and release-shape checks.
- `quick_anomaly_detection` is bounded:
  - it may consume only already-available low-cost intake-adjacent or
    structural surfaces
  - it must not trigger heavy computation or recompute proof artifacts
  - it must not consume proof-core, clinical-plausibility, or downstream-phase
    outputs
  - it must not compare AI-derived content semantically
  - it must not introduce thresholds that approximate downstream proof
  - it must not create a new blocking path

Frozen status vocabularies:

- `pre_check.status`: `passed | failed | advisory`
- `structure_validation.status`: `passed | failed`
- `hash_checks.status`: `passed | failed | not_evaluated`
- `shape_consistency.status`: `passed | failed | not_evaluated`
- `quick_anomaly_detection.status`: `passed | advisory | not_evaluated`

### Full Proof

`full_proof` is a derived view over canonical `proof_core`. It is not a new
phase, not a new proof contract, and not a replacement name.

- `full_proof.proof_stages` always materializes all four canonical proof stages
  in canonical order.
- If canonical proof-core was not entered, every stage still materializes with
  `status = not_evaluated`.
- `not_evaluated` means proof-core was not entered or not reached, not that the
  stage passed silently or was irrelevant.

Frozen status vocabularies:

- `full_proof.status`: `passed | failed | not_evaluated`
- `full_proof.proof_stages[].status`: `passed | failed | not_evaluated`

Operational labels such as `FFT / reprojection` or `parity validation` are
viewer/documentation labels only. They must never replace canonical proof-stage
identifiers.

### Journal boundary

- `decision_journal.json` keeps its frozen event vocabulary unchanged.
- The journal may summarize operational flow, but distinct `pre_check` checks
  remain inspectable only through
  `verification_pipeline.operational_subphases.pre_check.checks`.

### Validation

- Generated reports must validate against the frozen subphase and check
  vocabulary.
- Non-conforming names, orders, or statuses are invalid generated output.
- Subphase and check summaries are bounded template text derived from outcomes,
  not authored free prose.

## Consequences

- Reviewers now get an explicit early-screening versus full-proof narrative
  without any canonical phase drift.
- Single-series structural rejects and full releases share one stable
  operational-subphase shape.
- Parent multi-series reports remain principled aggregations of child canonical
  outcomes rather than speculative study-level operational stories.
- The repository gains a stronger performance and audit narrative without
  creating a shadow assurance model.
