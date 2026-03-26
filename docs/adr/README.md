# Architecture Decision Records

This directory records architecture decisions that materially affect invariants, interfaces, or operational posture.

## Current ADRs

- [`0001-guardian-layer-as-verification-proxy.md`](0001-guardian-layer-as-verification-proxy.md)
- [`0002-fail-closed-guardian-handoff.md`](0002-fail-closed-guardian-handoff.md)
- [`0003-explicit-claim-proof-contracts.md`](0003-explicit-claim-proof-contracts.md)
- [`0004-replayable-evidence-bundles.md`](0004-replayable-evidence-bundles.md)
- [`0005-performance-bounds-and-streaming-validation.md`](0005-performance-bounds-and-streaming-validation.md)
- [`0006-canonical-deterministic-mask-engine-catalogue.md`](0006-canonical-deterministic-mask-engine-catalogue.md)

## ADR Rules

- Add a new ADR when a decision changes architecture, invariants, security posture, wire format, interoperability model, or release semantics.
- Do not rewrite accepted ADRs to hide history. Add a new ADR if the decision changes materially.
- Keep file names stable and numbered in ascending order.

## Template

```text
# ADR-XXXX: Title
Status: Accepted
Date: YYYY-MM-DD

## Context
## Decision
## Consequences
```
