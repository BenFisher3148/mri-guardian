# ADR-0007: Connected-Component-Constrained Mask Policy Family

Status: Accepted
Date: 2026-03-26

## Context

The approved mask-engine catalogue now supports more than one deterministic policy
family. The next useful family after threshold-only and morphology-constrained
thresholding is connected-component-constrained thresholding, which can suppress
detached threshold-positive micro-islands without introducing probabilistic or
learned behavior.

Because Guardian is replay-sensitive, the component engine cannot leave traversal
discipline or connectivity semantics implicit. Even when retained-mask output is
currently driven only by component size, future contributors must not treat label
assignment order, neighbor order, or traversal strategy as arbitrary.

## Decision

Introduce the approved algorithm
`connected_component_constrained_threshold_on_raw_derived_reconstruction`.

- The engine first computes the canonical threshold seed mask from the `float64`
  raw-derived reconstruction.
- The canonical schema includes a proof-significant `connectivity` field, but
  this milestone approves only `faces`; any other value is rejected during
  canonical resolution.
- Face connectivity uses this fixed neighbor order:
  `(-1, 0, 0)`, `(1, 0, 0)`, `(0, -1, 0)`, `(0, 1, 0)`, `(0, 0, -1)`,
  `(0, 0, 1)`.
- Component traversal uses queue-based BFS with canonical `(z, y, x)` scan order.
- Component labels are deterministic internal execution state only and are not
  part of the proof-significant output surface.
- The first approved selection rule keeps all connected components meeting
  `min_component_voxels`; ranking-based or top-N selection is deferred until a
  separate canonical tie-break specification exists.

## Consequences

- Guardian now has a clean progression of deterministic mask-policy families:
  threshold-local, morphology-local-structural, and connected-component
  structural.
- `brain_t1_component_review` becomes the first showcase profile for the
  connected-component family while baseline release profiles remain unchanged.
- Future component-ranking behavior must be introduced as a separately specified
  extension rather than silently bolted onto this family.
