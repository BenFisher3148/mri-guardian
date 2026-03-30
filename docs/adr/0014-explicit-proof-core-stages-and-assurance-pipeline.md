# ADR-0014: Explicit Proof-Core Stages and Assurance Pipeline

Status: Accepted
Date: 2026-03-30

## Context

The repository originally described the hard verification layer as one implicit
Level 2 block. That was serviceable early on, but it blurred several distinct
things:

- end-to-end assurance flow versus internal execution topology
- proof-stage contracts versus replay/evidence aggregation
- hard proof semantics versus clinical or policy interpretation
- stage-level failure attribution versus broader release disposition

As the repo grew to include longitudinal review, multi-series aggregation,
portable bundles, viewer layers, and a formal threat corpus, the implicit shape
became too ambiguous. The code already performed distinct kinds of proof work,
but the public model did not state them clearly enough.

The repository needed one explicit architecture decision that fixes:

- the canonical assurance-flow schema
- the ordered hard-proof stages inside `proof_core`
- the role and limits of the aggregate evidence spine
- decision attribution inside proof-core failures
- the boundary between proof-core and non-proof phases

## Decision

Adopt a canonical `verification_pipeline` and make proof-core stages explicit.

### Assurance pipeline

- `verification_pipeline` is the canonical assurance-flow schema and
  evidence-ordering model.
- It does not mandate runtime execution topology.
- The canonical pipeline phases are:
  - `intake_validation`
  - `structural_validity`
  - `proof_core`
  - `clinical_plausibility`
  - `policy_evaluation`
  - `multi_series_aggregation`
  - `release_control`

### Proof core

- `proof_core` is one pipeline phase, not a synonym for the whole pipeline.
- The canonical proof stages are:
  - `MeasurementIntegrityProof`
  - `ReconstructionParityProof`
  - `SegmentationPolicyProof`
  - `VolumeComputationProof`
- Stage order is fixed:
  `MeasurementIntegrityProof -> ReconstructionParityProof -> SegmentationPolicyProof -> VolumeComputationProof`.
- Level 1/2/3 language is retained only as a derived viewer or documentation
  presentation layer. It is not the canonical report schema.

### Public proof contracts

- `MeasurementIntegrityProof` proves that the raw-derived reconstruction remains
  anchored to the measurement domain within policy.
- `ReconstructionParityProof` proves AI reconstruction parity against the
  raw-derived reconstruction under the declared tolerance policy.
- `SegmentationPolicyProof` proves the AI mask against the policy-derived
  verified mask and records the exact comparison outcome.
- `VolumeComputationProof` proves the claimed scalar volume against the
  verified-mask volume computed from voxel geometry.

### Aggregate evidence spine

- Retire `VerifiedReconstructionTrace` as the canonical top-level public proof
  contract.
- Preserve its evidentiary semantics in a derived aggregate replay surface named
  `ProofCoreEvidenceTrace`.
- `ProofCoreEvidenceTrace` is strictly derived-only:
  - it contains no independent verdicts
  - it contains no recomputed proof outcomes
  - it is a canonical aggregation of already-materialized proof-stage artefacts
    only
- `ProofCoreEvidenceTrace` is not a fifth proof stage and not a separate gating
  rule.
- Viewer-facing language such as "verified reconstruction" may remain as a
  derived presentation label where useful, but it is not the canonical contract
  name.

### Decision schema

- Replace `decision.evaluation_stage` with `decision.pipeline_phase`.
- Add optional `decision.proof_stage` for outcomes inside `proof_core`.
- Decision lineage and structured reasons must carry `pipeline_phase` plus
  optional `proof_stage`.
- A failed earlier proof stage fixes the earliest failing `proof_stage` for
  lineage and primary decision attribution within `proof_core`.

### Derived summaries

- Demote `no_invented_pixels` from a peer proof section to a derived summary or
  invariant, for example `proof_core_summary.no_invented_pixels`.
- Derived summaries must not be treated as additional proof contracts.

### Threat corpus

- Threat corpus entries keep `scenario_family` and `scenario_surface`.
- Threat corpus entries add canonical `pipeline_phase` and optional
  `proof_stage`.
- In the threat corpus, `proof_stage` denotes the intended primary exercised
  stage only.
- Downstream propagated failures must not cause the corpus entry itself to be
  reclassified.

## Proof-Core Invariants

- Final-release-capable flows must emit a canonical four-stage proof-core record
  in order.
- Stages not reached due to earlier proof failure must still materialize as
  explicit non-passing records rather than disappearing.
- A later proof stage cannot clear, downgrade, or bypass failure in an earlier
  proof stage.
- Non-final or advisory flows may change downstream release consequences only
  outside `proof_core`; they must not rewrite proof-stage results.

### Stage independence

- `MeasurementIntegrityProof` may depend only on raw measurements,
  deterministic reconstruction logic, and policy inputs.
- `ReconstructionParityProof` may depend on AI reconstruction plus
  `MeasurementIntegrityProof` outputs, but never on segmentation or volume
  outputs.
- `SegmentationPolicyProof` may depend on the verified reconstruction surface
  used for mask derivation and mask-policy outputs, but not on AI reconstruction
  parity outcomes beyond admissible upstream artefact availability.
- `VolumeComputationProof` may depend on verified mask and voxel geometry
  outputs, but never on clinical or multi-series outputs.
- `VolumeComputationProof` must not depend on AI-claimed semantic
  interpretation beyond the claimed scalar quantity being checked.

### Sacred boundary

- `clinical_plausibility` is interpretive and heuristic.
- It must never be described or surfaced as a proof stage.
- Temporal review, policy evaluation, multi-series aggregation, and release
  control remain outside the proof-core narrative.

## Consequences

- The hard-proof model is now explicit and layered in a way that matches the
  repository's real assurance structure.
- Replay, bundle, viewer, integration, and threat-corpus consumers share one
  canonical phase/stage vocabulary.
- `ProofCoreEvidenceTrace` remains useful as a replayable evidence spine without
  silently becoming a shadow proof engine.
- Failure reporting is sharper because `decision.pipeline_phase` and optional
  `decision.proof_stage` separate proof-stage attribution from broader release
  posture.
- Future extensions must preserve the proof-core boundary rather than smuggling
  heuristic, policy, or cross-series semantics into the hard-proof contracts.
