# ADR-0005: Performance Bounds And Streaming Validation

Status: Accepted
Date: 2026-03-26

## Context

MRI reconstruction from raw k-space is computationally expensive. Reviewers need to know both that the Guardian Layer is fast enough and that its latency story is explicit rather than anecdotal.

## Decision

Publish a benchmark ladder, derive a shape-scoped worst-case bound from the largest profiled volume, and support streaming partial validation without partial downstream release.

## Consequences

- Performance claims are tied to documented probe shapes and benchmark methodology.
- The Guardian Layer can start detecting bad series before full-volume close.
- The release gate remains deterministic and fail-closed because downstream release still waits for the full proof.
