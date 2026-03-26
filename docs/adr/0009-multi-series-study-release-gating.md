# ADR-0009: Multi-Series Study Release Gating

Status: Accepted
Date: 2026-03-26

## Context

The original Guardian path verifies one MRI series cleanly, but real imaging
workflows often operate at study scope across multiple sequences such as T1, T2,
FLAIR, contrast, non-contrast, and derived outputs.

That creates a gap between per-series proof and study-level credibility. A
single verified raster is useful, but diagnostic workflows often depend on
evidence context across multiple series. If Guardian is going to make a stronger
claim in that setting, it needs one deterministic study contract that preserves
per-series proofs while adding explicit cross-series gating semantics.

Because Guardian is replay-sensitive, the study contract also needs to freeze:

- which series are release-gating versus auxiliary
- how geometry equality is decided
- how positive-lesion corroboration is decided
- what ordering is proof-significant
- how malformed study payloads fail

## Decision

Introduce an additive `multi_series_v2` study-level path alongside the existing
single-series flow.

- The outer study payload contains an ordered `series` list and one
  `cross_series_policy`.
- All payload series are individually verified and reported through the existing
  per-series proof pipeline.
- `required_series_ids` is the only authoritative merged-release gating set.
- `primary_series_id` must appear exactly once in `required_series_ids`.
- For this milestone, every `supporting_series_id` is required. Payload series
  outside `required_series_ids` are auxiliary and non-gating.
- `supporting_series_ids` must be non-empty in `multi_series_v2`. Invalid
  callers fail validation; the implementation does not auto-downgrade to the
  single-series path.
- Duplicate payload `series_id` values are an early structural study-shape
  failure because identity resolution would otherwise be ambiguous before any
  meaningful cross-series evaluation can occur.
- Caller-provided `required_series_ids` and `supporting_series_ids` are
  accepted, then canonicalized into the same order as the outer `series` list.
  Caller order is not proof-significant.
- Study order is proof-significant for deterministic serialization, stable
  report diffs, stable replay shape, and canonical bundle identity. Later
  "helpful" sorting by UID, sequence id, or series id is not allowed unless a
  future ADR intentionally changes the invariant.
- Required-series geometry uses exact equality over exact voxel shape plus
  canonicalized voxel spacing.
- Canonical voxel spacing is derived by coercing values to numeric floats and
  comparing the canonical rounded tuple, not incidental raw list formatting.
- Positive-lesion corroboration is defined as exact coordinate intersection on
  the shared voxel grid:
  `intersection_voxel_count = count_nonzero(primary_verified_mask & supporting_verified_mask)`.
- Corroboration passes iff:
  - every required verified mask is empty, or
  - the primary verified mask is non-empty and at least one supporting verified
    mask has `intersection_voxel_count > 0`.
- One shared voxel counts. Diagonal adjacency without identical coordinates does
  not count.
- If the primary verified mask is empty and any required supporting verified
  mask is non-empty, corroboration fails.
- Cross-series corroboration is intentionally a positive-lesion consistency
  gate. Negative studies may still release if each required series proves clean
  and required-series geometry verifies.
- Released study manifests include all payload series in proof-significant
  order, including auxiliaries and withheld series.
- Released study workstation payloads include only downstream-allowed series and
  an explicit `withheld_series` summary so omission does not look like
  disappearance.

## Consequences

- Guardian can now make a deterministic study-level release decision while
  preserving the existing single-series proof model unchanged.
- Auxiliary series remain visible and attributable without influencing the
  merged release gate in this milestone.
- Reviewers and replay code have one clear contract for malformed study shape,
  non-canonical caller ordering, exact-overlap corroboration, and proof-significant
  study order.
- Future registration-aware geometry, optional supporting series, or richer
  corroboration thresholds must be introduced as explicit extensions rather than
  drifting into this milestone's same-grid exact-overlap semantics.
