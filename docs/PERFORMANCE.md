# Performance

This document describes how the MRI Guardian Layer measures and communicates verification overhead.

## Performance Objective

The Guardian Layer must prove integrity without making the hospital handoff operationally impractical. The performance story therefore has three parts:

- representative benchmark timings
- a shape-scoped worst-case bound
- a streaming strategy that starts verification early without weakening the release gate

Each runtime profile also declares performance expectations. Those expectations are monitoring surfaces only; they do not change proof validity, replay validity, or final-release eligibility unless a separate operational policy chooses to act on them.

## Benchmark Method

Benchmarking is implemented in [`mri_guardian/timing.py`](../mri_guardian/timing.py).

Method:

- Use synthetic probes derived from the deterministic tamper baseline.
- Warm up the numerical path before measuring.
- Measure reconstruction and verification separately.
- Report medians for headline numbers.
- Derive worst-case bounds from the largest profiled volume using observed max plus a fixed safety margin.

Current parameters:

- warm-up runs: `2`
- measured runs per probe: `5`
- worst-case safety margin: `25%`

## Current Local Profile

The authoritative machine-readable record is [`../demo_artifacts/performance_profile.json`](../demo_artifacts/performance_profile.json).

On the current local run dated March 26, 2026, the generated `brain_t1_baseline` performance profile reported:

| Probe | Shape | Voxels | Reconstruction | Verification | Total | Overhead |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Streaming Slab | `12x96x96` | 110,592 | `3.00ms` | `0.94ms` | `3.96ms` | `+31%` |
| Representative Series | `24x264x264` | 1,672,704 | `50.45ms` | `16.85ms` | `67.13ms` | `+33%` |
| High-Load Bound Probe | `30x336x336` | 3,386,880 | `98.31ms` | `32.76ms` | `132.54ms` | `+33%` |

Derived worst-case bound on the largest profiled volume:

- reconstruction bound: `<=130.29ms`
- verification bound: `<=43.64ms`
- total bound: `<=173.93ms`

Backend on that run:

- `NumPy FFT core (2.4.3) with Numba available (0.64.0)`

The profile artifact also records:

- `profile_name`, `profile_revision`, and `profile_digest`
- the embedded `verification_profile_descriptor`
- `performance_expectations.normative = false`
- budget-versus-observed comparisons for representative and worst-case totals

## Worst-Case Bound Policy

The repository does not claim an unbounded global latency guarantee. Instead it publishes a shape-scoped bound:

- bound basis: observed max across warmed runs
- bound scope: largest profiled volume only
- padding: fixed 25% deterministic safety margin

This avoids hand-wavy claims while still giving reviewers a concrete latency ceiling for a known study size.

## Streaming / Partial Validation Strategy

Streaming is supported as an internal validation strategy, not as partial release.

Stages:

1. Ingress structural gate: reject malformed objects before FFT work starts.
2. Streaming k-space anchoring: canonicalise and hash slabs as they arrive.
3. Partial reconstruction parity: reconstruct completed slabs and compare them with AI slabs.
4. Whole-volume release commit: release only after the full-volume proof closes.

Release rule:

- no partial reconstruction or partial mask is released to PACS/workstation from the Guardian path

This preserves fail-closed semantics while reducing time-to-detection for bad studies.

## Performance Caveats

- These numbers come from synthetic probes, not a production scanner feed.
- FFT cost scales with study shape and hardware characteristics.
- The current benchmark path measures numerical verification only, not network transport, storage latency, or PACS import time.
- Production deployments should recalibrate bounds per scanner family, sequence family, and hardware tier.

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`adr/0005-performance-bounds-and-streaming-validation.md`](adr/0005-performance-bounds-and-streaming-validation.md)
