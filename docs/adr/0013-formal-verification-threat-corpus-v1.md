# ADR-0013: Formal Verification Threat Corpus V1

Status: Accepted
Date: 2026-03-30

## Context

The repository already had strong deterministic verification behaviour and a
small tamper-mode demo surface, but the adversarial and failure scenarios were
still fragmented across:

- ad hoc tamper builders
- individual tests
- documentation tables
- demo artefact expectations

That made the threat model harder to query, harder to extend safely, and harder
to align with the newer clinical-risk documentation. The project needed one
code-backed registry that could describe executable threats, keep the current
tamper-mode compatibility surface intact, and avoid changing the proof,
manifest, bundle, or decision contracts.

Two scenarios also needed explicit architectural boundaries:

- `stale_mask_replay` must remain a semantic state mismatch in normal
  verification, not drift into provenance or bundle-fraud semantics.
- `cross_series_misregistration` must exercise corroboration failure while
  geometry verification still passes, not collapse into the existing geometry
  mismatch path.

## Decision

Introduce an additive formal verification threat corpus in
`mri_guardian/tamper.py`.

- Export `build_verification_threat_corpus(...)` as the canonical corpus
  builder.
- Keep `build_tamper_scenarios()` as a curated compatibility adapter for the
  existing tamper-mode subset.
- Represent each corpus entry with stable structured metadata including:
  - stable `corpus_id`
  - human-facing `slug`, `title`, and `description`
  - `scenario_family`
  - controlled `scenario_surface`
  - `corpus_status`
  - `technical_failure_mechanism`
  - `clinical_hazard`
  - `downstream_effect`
  - corpus-only `hazard_severity`
  - structured `mitigation_mapping`
  - structured `determinism_contract`
  - normative invariants:
    - `expected_decision_class`
    - `expected_failure_route`
  - optional observational expectations:
    - `expected_block_category`
    - `expected_primary_reason_family`
- Keep mutation logic in small deterministic helpers. The corpus builder
  assembles entries; it does not become a second hidden execution engine.
- Add `verification_threat_corpus.json` as a non-proof demo artefact that
  serialises the corpus metadata together with generated tamper artefact paths.
- Keep existing `tamper_<slug>*` file naming unchanged for the legacy curated
  scenarios and add new `tamper_<slug>*` artefacts for the broader corpus.
- Add `docs/VERIFICATION_THREAT_CORPUS.md` as the threat-reference document and
  link it from the root README, `docs/SECURITY.md`, and
  `docs/CLINICAL_RISK_ASSURANCE_MAPPING.md`.

### Scenario constraints

- `component_micro_island` and `slice_single_slice_island` are
  `profile_scoped`, not general baseline threats.
- No `deferred` corpus entries are shipped in V1; every listed scenario is
  executable.
- `stale_mask_replay` remains:
  - syntactically valid
  - structurally admissible
  - caught by semantic verification rather than intake rejection
  - not a provenance, transport, or bundle replay threat
- `cross_series_misregistration` remains:
  - geometry-valid
  - structurally admissible
  - blocked because lesion corroboration fails
  - distinct from `required_series_geometry_mismatch`

## Consequences

- The threat model now has one executable source of truth shared by code, demo
  outputs, tests, and documentation.
- Threat metadata is queryable and stable enough for clinical-risk mapping and
  future coverage checks.
- Existing tamper-mode consumers keep the curated compatibility subset without
  needing proof-contract or decision-schema changes.
- Future threat additions must preserve the normative/observational expectation
  split and the deterministic-builder boundary rather than accumulating
  free-form scenario text.
