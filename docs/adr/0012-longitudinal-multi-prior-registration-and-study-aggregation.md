# ADR 0012: Longitudinal Multi-Prior Selection, Registration, and Study Aggregation

## Status

Accepted

## Context

ADR 0011 introduced the first longitudinal layer for one prior comparator and
explicitly deferred:

- multi-prior selection and trajectory analysis
- deterministic registration-aware spatial comparison
- study-level temporal aggregation in `multi_series_v2`
- any profile-controlled temporal blocking beyond the baseline advisory layer

Those features are now required without breaking the existing V1 report shape.

## Decision

Extend the existing longitudinal surfaces instead of replacing them.

### Single-series

- Accept optional `prior_verified_contexts` as the canonical multi-prior source list.
- Keep `prior_verified_context` as the selected-primary alias for compatibility.
- Emit:
  - `LongitudinalSelectionTrace`
  - `TemporalTrajectoryTrace`
  - `LongitudinalRegistrationTrace`
  - `TemporalConsistencyTrace`
- Select trajectory priors from admissible earlier priors only.
- Keep inadmissible priors visible in `LongitudinalSelectionTrace` but exclude them from selection and trajectory computation.
- Use only trusted, positive, non-zero intervals for trajectory slope and aggregate trend calculations.

### Registration

- Registration remains in-repo and deterministic.
- The shipped algorithm is bounded integer translation over the verified mask grid.
- Registration trace separates transform production from quality outcome and blocking admissibility.
- `TemporalConsistencyTrace.spatial_comparison_mode` is now one of:
  - `exact`
  - `registered`
  - `volume_only`
  - `none`

### Blocking

- Temporal blocking remains profile-controlled.
- In the shipped registry, `strict_research_mode` is the only profile with temporal blocking enabled.
- Blocking requires:
  - trusted chronology
  - a usable primary prior
  - evaluated blocking-relevant metrics
  - acceptable registration quality when registered comparison was used

### Study aggregation

- `multi_series_v2` now accepts per-series temporal inputs.
- Each series runs the same single-series longitudinal path first.
- The study report emits `StudyTemporalAggregationTrace`.
- Only `required_series_ids` participate in study-level temporal gating semantics.
- Auxiliary series remain informational.

## Consequences

- Longitudinal review now has deterministic memory, not just one-step comparison.
- Viewer, bundle, integration, and replay surfaces expose selection, trajectory,
  registration, and study aggregation explicitly.
- Affine-only geometry mismatches can upgrade into deterministic registered
  comparison instead of collapsing directly to `volume_only`.
- The hard proof gate remains unchanged. Temporal blocking is still opt-in by
  profile, not globally enabled.
