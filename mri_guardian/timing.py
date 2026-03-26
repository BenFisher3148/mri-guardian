"""Performance benchmark helpers for the MRI Guardian Layer."""

from __future__ import annotations

from functools import lru_cache
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np

from .harness import _RECONSTRUCTION_TOLERANCE, _VERIFIED_MASK_THRESHOLD, _round_float
from .tamper import build_tamper_mode_base_case

_TIMING_PROBE_SPECS = (
    (
        "Streaming Slab",
        2,
        4,
        "Small slab profile used to support early partial validation before the full volume closes.",
    ),
    (
        "Representative Series",
        4,
        11,
        "Representative radiology handoff profile for the side-by-side verification view.",
    ),
    (
        "High-Load Bound Probe",
        5,
        14,
        "Largest profiled volume used to derive a guarded worst-case latency bound.",
    ),
)
_REPRESENTATIVE_BENCHMARK_LABEL = "Representative Series"
_WORST_CASE_BENCHMARK_LABEL = "High-Load Bound Probe"
_TIMING_WARMUP_RUNS = 2
_TIMING_SAMPLE_RUNS = 5
_BOUND_SAFETY_MARGIN_FRACTION = 0.25
_STREAMING_SLAB_DEPTH_SLICES = 4


@lru_cache(maxsize=1)
def build_guardian_performance_profile() -> dict[str, Any]:
    """Return a cached benchmark ladder and performance envelope for the demo."""
    benchmarks = [
        _benchmark_probe(
            label=label,
            tile_z=tile_z,
            tile_xy=tile_xy,
            purpose=purpose,
        )
        for label, tile_z, tile_xy, purpose in _TIMING_PROBE_SPECS
    ]
    representative = next(
        benchmark
        for benchmark in benchmarks
        if benchmark["label"] == _REPRESENTATIVE_BENCHMARK_LABEL
    )
    worst_case_benchmark = next(
        benchmark
        for benchmark in benchmarks
        if benchmark["label"] == _WORST_CASE_BENCHMARK_LABEL
    )
    worst_case_bound = _build_worst_case_bound(worst_case_benchmark)
    return {
        "reconstruction_ms": representative["reconstruction_ms"],
        "verification_ms": representative["verification_ms"],
        "total_pipeline_ms": representative["total_pipeline_ms"],
        "overhead_percent": representative["overhead_percent"],
        "probe_shape": representative["probe_shape"],
        "probe_voxel_count": representative["probe_voxel_count"],
        "warmup_runs": _TIMING_WARMUP_RUNS,
        "sample_runs": _TIMING_SAMPLE_RUNS,
        "backend": _backend_label(),
        "benchmarks": benchmarks,
        "worst_case_bound": worst_case_bound,
        "streaming_strategy": _build_streaming_strategy(),
    }


def build_timing_overlay_snapshot() -> dict[str, Any]:
    """Return the cached performance profile used by the HTML timing overlay."""
    return build_guardian_performance_profile()


def _build_timing_probe(*, tile_z: int, tile_xy: int) -> dict[str, np.ndarray]:
    base_state = build_tamper_mode_base_case()
    return {
        "raw_real": np.tile(
            np.asarray(base_state["raw_real"], dtype=np.float64),
            (tile_z, tile_xy, tile_xy),
        ),
        "raw_imag": np.tile(
            np.asarray(base_state["raw_imag"], dtype=np.float64),
            (tile_z, tile_xy, tile_xy),
        ),
        "ai_reconstruction": np.tile(
            np.asarray(base_state["ai_reconstruction"], dtype=np.float64),
            (tile_z, tile_xy, tile_xy),
        ),
        "ai_mask": np.tile(
            np.asarray(base_state["tumor_mask"], dtype=np.int64),
            (tile_z, tile_xy, tile_xy),
        ),
        "voxel_spacing_mm": np.asarray(base_state["voxel_spacing_mm"], dtype=np.float64),
    }


def _warm_up_probe(probe: dict[str, np.ndarray]) -> None:
    for _ in range(_TIMING_WARMUP_RUNS):
        derived_reconstruction = _run_reconstruction_stage(probe)
        _run_verification_stage(probe, derived_reconstruction)


def _run_reconstruction_stage(probe: dict[str, np.ndarray]) -> np.ndarray:
    raw_kspace = probe["raw_real"] + (1j * probe["raw_imag"])
    return np.fft.ifft2(raw_kspace, axes=(-2, -1)).real


def _run_verification_stage(
    probe: dict[str, np.ndarray],
    derived_reconstruction: np.ndarray,
) -> None:
    ai_reconstruction = probe["ai_reconstruction"]
    ai_mask = probe["ai_mask"]
    intensity_mismatch_mask = (
        np.abs(ai_reconstruction - derived_reconstruction) > _RECONSTRUCTION_TOLERANCE
    )
    verified_mask = (derived_reconstruction >= _VERIFIED_MASK_THRESHOLD).astype(np.int64)
    rejected_mask = (ai_mask == 1) & (verified_mask == 0)
    missing_mask = (ai_mask == 0) & (verified_mask == 1)
    voxel_volume_mm3 = float(np.prod(probe["voxel_spacing_mm"]))
    _ = int(np.count_nonzero(intensity_mismatch_mask))
    _ = int(np.count_nonzero(rejected_mask | missing_mask))
    _ = int(np.count_nonzero(rejected_mask))
    _ = int(np.count_nonzero(missing_mask))
    _ = _round_float(int(np.count_nonzero(verified_mask)) * voxel_volume_mm3)


def _benchmark_probe(
    *,
    label: str,
    tile_z: int,
    tile_xy: int,
    purpose: str,
) -> dict[str, Any]:
    probe = _build_timing_probe(tile_z=tile_z, tile_xy=tile_xy)
    _warm_up_probe(probe)
    reconstruction_samples_ms: list[float] = []
    verification_samples_ms: list[float] = []
    total_samples_ms: list[float] = []
    for _ in range(_TIMING_SAMPLE_RUNS):
        reconstruction_start = perf_counter_ns()
        derived_reconstruction = _run_reconstruction_stage(probe)
        reconstruction_end = perf_counter_ns()
        _run_verification_stage(probe, derived_reconstruction)
        verification_end = perf_counter_ns()
        reconstruction_ms = (reconstruction_end - reconstruction_start) / 1_000_000
        verification_ms = (verification_end - reconstruction_end) / 1_000_000
        total_ms = (verification_end - reconstruction_start) / 1_000_000
        reconstruction_samples_ms.append(reconstruction_ms)
        verification_samples_ms.append(verification_ms)
        total_samples_ms.append(total_ms)

    reconstruction_ms = median(reconstruction_samples_ms)
    verification_ms = median(verification_samples_ms)
    total_pipeline_ms = median(total_samples_ms)
    overhead_percent = 0.0
    if reconstruction_ms > 0:
        overhead_percent = (verification_ms / reconstruction_ms) * 100
    return {
        "label": label,
        "purpose": purpose,
        "probe_shape": list(probe["raw_real"].shape),
        "probe_voxel_count": int(np.prod(probe["raw_real"].shape)),
        "reconstruction_ms": _round_float(reconstruction_ms),
        "verification_ms": _round_float(verification_ms),
        "total_pipeline_ms": _round_float(total_pipeline_ms),
        "overhead_percent": _round_float(overhead_percent),
        "max_reconstruction_ms": _round_float(max(reconstruction_samples_ms)),
        "max_verification_ms": _round_float(max(verification_samples_ms)),
        "max_total_pipeline_ms": _round_float(max(total_samples_ms)),
    }


def _build_worst_case_bound(benchmark: dict[str, Any]) -> dict[str, Any]:
    reconstruction_bound_ms = _apply_safety_margin(benchmark["max_reconstruction_ms"])
    verification_bound_ms = _apply_safety_margin(benchmark["max_verification_ms"])
    total_bound_ms = max(
        _apply_safety_margin(benchmark["max_total_pipeline_ms"]),
        _round_float(reconstruction_bound_ms + verification_bound_ms),
    )
    overhead_percent = 0.0
    if reconstruction_bound_ms > 0:
        overhead_percent = (verification_bound_ms / reconstruction_bound_ms) * 100
    return {
        "label": benchmark["label"],
        "probe_shape": benchmark["probe_shape"],
        "probe_voxel_count": benchmark["probe_voxel_count"],
        "reconstruction_ms": reconstruction_bound_ms,
        "verification_ms": verification_bound_ms,
        "total_pipeline_ms": total_bound_ms,
        "overhead_percent": _round_float(overhead_percent),
        "safety_margin_fraction": _BOUND_SAFETY_MARGIN_FRACTION,
        "basis": (
            f"Observed max across {_TIMING_SAMPLE_RUNS} warmed runs on the largest profiled "
            f"volume, padded by {int(_BOUND_SAFETY_MARGIN_FRACTION * 100)}%."
        ),
    }


def _build_streaming_strategy() -> dict[str, Any]:
    return {
        "mode": "streaming_partial_validation",
        "release_policy": (
            "No partial reconstruction or mask is released to PACS/workstation. "
            "Streaming only advances internal verification state until the full series closes."
        ),
        "window_unit": "z_slab",
        "recommended_slab_depth_slices": _STREAMING_SLAB_DEPTH_SLICES,
        "stages": [
            {
                "title": "Ingress structural gate",
                "window": "per DICOM object",
                "action": (
                    "Validate identifiers, tensor shapes, finite numeric fields, and voxel "
                    "geometry before any FFT work starts."
                ),
                "benefit": "Malformed or geometry-drifting series fail before expensive reconstruction.",
            },
            {
                "title": "Streaming k-space anchoring",
                "window": "per received slab",
                "action": (
                    "Canonicalise and hash raw k-space slabs as they arrive while enforcing a "
                    "stable voxel grid for the whole study."
                ),
                "benefit": "Audit anchoring and geometry drift detection begin before the full series is present.",
            },
            {
                "title": "Partial reconstruction parity",
                "window": f"per {_STREAMING_SLAB_DEPTH_SLICES}-slice slab",
                "action": (
                    "Reconstruct each completed slab from raw k-space and compare it with the "
                    "AI slab under the same explicit tolerance policy."
                ),
                "benefit": "Invented pixels and local intensity drift are caught before final series close.",
            },
            {
                "title": "Whole-volume release commit",
                "window": "final series close",
                "action": (
                    "Assemble the verified mask, recompute tumour volume, sign the bundle, "
                    "and decide whether the PACS/workstation handoff is allowed."
                ),
                "benefit": "Final downstream release still depends on one deterministic full-volume proof.",
            },
        ],
        "early_block_conditions": [
            "DICOM or geometry mismatch",
            "Per-slab raw k-space inconsistency",
            "Per-slab AI reconstruction parity failure",
            "Completed-slab mask divergence",
        ],
    }


def _apply_safety_margin(value: float) -> float:
    return _round_float(float(value) * (1.0 + _BOUND_SAFETY_MARGIN_FRACTION))


def _backend_label() -> str:
    numpy_version = np.__version__
    numba_version = _optional_module_version("numba")
    if numba_version is None:
        return f"NumPy FFT core ({numpy_version}); Numba not installed"
    return f"NumPy FFT core ({numpy_version}) with Numba available ({numba_version})"


def _optional_module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")
