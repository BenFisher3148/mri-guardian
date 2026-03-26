# ADR-0008: Slice-Consistency-Constrained Mask Policy Family

Status: Accepted
Date: 2026-03-26

## Context

The approved mask-engine catalogue already supports threshold-local,
morphology-local-structural, and connected-component structural policy families.
The next useful deterministic family is slice-consistency filtering, which can
reject detached single-slice threshold-positive islands and other no-support
slice flicker without introducing probabilistic or learned behavior.

Because Guardian is replay-sensitive, slice consistency cannot be described with
loose language such as "similar across nearby slices". The survival rule has to
freeze the unit of continuity, the overlap definition, the allowed slice-step,
and the traversal discipline used for deterministic in-slice component labelling.

## Decision

Introduce the approved algorithm
`slice_consistency_constrained_threshold_on_raw_derived_reconstruction`.

- This family expands the deterministic policy catalogue; it does not introduce
  runtime-selectable algorithm freedom for released verification.
- The engine first computes the canonical threshold seed mask from the `float64`
  raw-derived reconstruction.
- The survival unit is the whole in-slice connected component.
- The canonical schema includes proof-significant `consistency_unit`,
  `in_slice_connectivity`, `min_consecutive_slices`, and `overlap_rule`
  parameters, but this milestone approves only:
  - `consistency_unit = in_slice_component`
  - `in_slice_connectivity = faces_2d`
  - `overlap_rule = any_overlap`
- In-slice traversal uses queue-based BFS with canonical `(y, x)` scan order and
  fixed 2D face-neighbor order:
  `(-1, 0)`, `(1, 0)`, `(0, -1)`, `(0, 1)`.
- The engine constructs an undirected layered overlap graph whose vertices are
  slice-labelled in-slice components. Each vertex belongs to exactly one slice,
  and edges are permitted only between vertices from slices `z` and `z+1`.
- Adjacent-slice overlap is defined as non-empty intersection of the two
  components' `(y, x)` footprint sets on those adjacent slices.
- A component survives iff it lies on at least one qualifying no-gap run of
  consecutive slice indices of length at least `min_consecutive_slices`.
- Whole-component retention is intentional for interpretability and replay
  simplicity. This family is a run-participation rule, not a shared-overlap-core
  rule.
- In-slice labels, overlap-graph node ids, and qualifying-run ids are
  deterministic internal execution state only and are excluded from the
  proof-significant output surface.

## Consequences

- Guardian now has a clean progression of deterministic mask-policy families:
  threshold-local, morphology-local-structural, connected-component structural,
  and slice-consistency cross-slice continuity.
- `brain_t1_slice_review` becomes the first showcase profile for the
  slice-consistency family while baseline release profiles remain unchanged.
- Future skipped-slice bridging, fuzzy overlap thresholds, centroid continuity,
  or ranking behavior must be introduced as separately specified extensions
  rather than silently bolted onto this family.
