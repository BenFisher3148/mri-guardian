"""Performance timing helpers for the MRI guardian visual overlay."""

from __future__ import annotations

from functools import lru_cache
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np

from .harness import _RECONSTRUCTION_TOLERANCE, _VERIFIED_MASK_THRESHOLD, _round_float
from .tamper import build_tamper_mode_base_case

_PROBE_TILE_XY = 11
_PROBE_TILE_Z = 4
_TIMING_WARMUP_RUNS = 2
_TIMING_SAMPLE_RUNS = 7


@lru_cache(maxsize=1)
def build_timing_overlay_snapshot() -> dict[str, Any]:
    """Return cached timing data for the HTML timing overlay."""
    probe = _build_timing_probe()
    _warm_up_probe(probe)
    reconstruction_samples_ms: list[float] = []
    verification_samples_ms: list[float] = []
    for _ in range(_TIMING_SAMPLE_RUNS):
        reconstruction_start = perf_counter_ns()
        derived_reconstruction = _run_reconstruction_stage(probe)
        reconstruction_end = perf_counter_ns()
        _run_verification_stage(probe, derived_reconstruction)
        verification_end = perf_counter_ns()
        reconstruction_samples_ms.append((reconstruction_end - reconstruction_start) / 1_000_000)
        verification_samples_ms.append((verification_end - reconstruction_end) / 1_000_000)

    reconstruction_ms = median(reconstruction_samples_ms)
    verification_ms = median(verification_samples_ms)
    total_pipeline_ms = reconstruction_ms + verification_ms
    overhead_percent = 0.0
    if reconstruction_ms > 0:
        overhead_percent = (verification_ms / reconstruction_ms) * 100
    return {
        "reconstruction_ms": _round_float(reconstruction_ms),
        "verification_ms": _round_float(verification_ms),
        "total_pipeline_ms": _round_float(total_pipeline_ms),
        "overhead_percent": _round_float(overhead_percent),
        "probe_shape": list(probe["raw_real"].shape),
        "warmup_runs": _TIMING_WARMUP_RUNS,
        "sample_runs": _TIMING_SAMPLE_RUNS,
        "backend": _backend_label(),
    }


def _build_timing_probe() -> dict[str, np.ndarray]:
    base_state = build_tamper_mode_base_case()
    return {
        "raw_real": np.tile(
            np.asarray(base_state["raw_real"], dtype=np.float64),
            (_PROBE_TILE_Z, _PROBE_TILE_XY, _PROBE_TILE_XY),
        ),
        "raw_imag": np.tile(
            np.asarray(base_state["raw_imag"], dtype=np.float64),
            (_PROBE_TILE_Z, _PROBE_TILE_XY, _PROBE_TILE_XY),
        ),
        "ai_reconstruction": np.tile(
            np.asarray(base_state["ai_reconstruction"], dtype=np.float64),
            (_PROBE_TILE_Z, _PROBE_TILE_XY, _PROBE_TILE_XY),
        ),
        "ai_mask": np.tile(
            np.asarray(base_state["tumor_mask"], dtype=np.int64),
            (_PROBE_TILE_Z, _PROBE_TILE_XY, _PROBE_TILE_XY),
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
