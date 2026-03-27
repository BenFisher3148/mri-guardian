# ADR-0011: Longitudinal Temporal Consistency V1
Status: Accepted
Date: 2026-03-27

## Context

Single-scan correctness is necessary but not sufficient for many reviewers. A result can be internally provable and still look suspicious when compared with a prior verified trajectory.

The repository needed a first longitudinal layer that:

- remains deterministic and replay-safe
- does not weaken or complicate the existing hard release gate
- makes timestamp trust, geometry comparability, and skipped-metric reasons explicit
- stays narrow enough for a first implementation

## Decision

Add an optional single-series Level 3 longitudinal path built around two new canonical contracts:

- `PriorVerifiedContext`
- `TemporalConsistencyTrace`

V1 is intentionally constrained:

- exactly one prior comparator is supported
- only Guardian-verified prior artefacts are admissible
- temporal findings are non-blocking in V1
- Level 2 proof plus existing profile policy still determine release

Timestamp trust is explicit:

- current-case input accepts `trusted_study_metadata`, `missing`, or `untrusted`
- prior context accepts `verified_bundle`, `trusted_study_metadata`, `missing`, or `untrusted`
- pair provenance is derived deterministically with `untrusted` dominating `missing`, identical trusted classes mapping to `verified_bundle_pair` or `trusted_metadata_pair`, and mixed trusted classes mapping to `mixed_provenance`

Temporal outputs must expose:

- `comparison_status`: `not_evaluated`, `aligned`, `degraded`, `unsuitable`
- `comparison_scope`: `not_evaluated`, `full`, `volume_only`, `none`
- `prior_admissibility`
- `chronology_assessment`
- `geometry_assessment`
- `computed_metrics`
- `flags`
- `skipped_metrics`

Geometry comparability is policy-driven:

- `exact_match` permits full spatial comparison
- `volume_basis_only` permits scalar-only comparison
- `insufficient_geometry_trust` suppresses temporal comparison

Metric definitions are fixed in the canonical contract:

- signed volume delta and signed percent change
- signed normalized percent change per 30 days
- `intersection_over_union`
- global positive-voxel centroid shift in mm

Interval bands are policy comparability bands, not clinical expectation bands:

- `<= 730` days may be `aligned`
- `731-3650` days is `degraded`
- `> 3650` days is `unsuitable` and suppresses temporal comparison even if scalar or spatial comparison would otherwise be mathematically possible

When prior context is present, the demo/bundle surfaces may emit:

- `prior_verified_context.json`
- `temporal_consistency_trace.json`

## Consequences

Positive:

- The system now has explicit longitudinal memory without changing the hard release gate.
- Viewer, integration, and replay consumers receive machine-readable temporal status and scope instead of inferring from prose.
- Timestamp trust, geometry limitations, and skipped metrics are pinned as canonical enums, reducing replay drift risk.

Tradeoffs:

- V1 does not support registration, interpolation, multiple priors, or study-level longitudinal aggregation.
- `> 3650` day suppression is a policy choice, not a numerical limitation.
- Temporal warnings can enrich `verified_with_advisory` outcomes but cannot block release in this milestone.
