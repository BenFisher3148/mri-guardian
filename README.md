# Verifiable MRI Guardian

This repository contains a proof-of-concept ICSD harness that sits between a raw MRI reconstruction AI and the radiologist workstation.

The core rule is simple: the radiologist never receives AI-supplied pixels. The harness treats the AI reconstruction as a claim, recomputes the displayed reconstruction from raw k-space itself, and only releases a workstation payload when all of the following are true:

- the reconstructed voxel grid is mathematically consistent with the raw sensor data
- the AI reconstruction matches the harness-derived reconstruction within a fixed tolerance
- the tumor mask lives on the same voxel grid as the reconstruction
- the tumor volume exactly equals `mask_voxel_count * voxel_spacing`

The ICSD layer makes the release state invariant-closed. A case is either `pending_verification` with no released payload, or `released_to_radiologist` with a `verified_manifest` and `workstation_payload` that exactly match the expected mathematical trace. If the AI invents pixels or misstates the volume, the released state becomes unrepresentable and the transition fails.

## Structure

- [mri_guardian/harness.py](mri_guardian/harness.py) defines the policy profile, invariants, deterministic reconstruction checks, and release transition.
- [mri_guardian/transparency.py](mri_guardian/transparency.py) implements a file-backed append-only hash chain for ICSD transparency receipts.
- [mri_guardian/demo.py](mri_guardian/demo.py) writes a deterministic demo evidence bundle and workstation payload.
- [tests/test_guardian.py](tests/test_guardian.py) covers the happy path and two rejection paths.

## Run

Run the deterministic demo:

```powershell
.\.venv\Scripts\python.exe -m mri_guardian.demo --output-dir demo_artifacts
```

Verify the emitted ICSD evidence envelope:

```powershell
.\.venv\Scripts\icsd-dpa verify demo_artifacts\evidence.json
```

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Why this satisfies the challenge

The challenge is not only to log what the AI said, but to prevent unverifiable output from ever reaching the workstation.

This prototype does that by making the workstation payload a deterministic function of:

- raw k-space
- voxel spacing
- the AI mask and volume claim
- the active ICSD policy profile

The AI can suggest a reconstruction, but it cannot inject a different pixel raster into the released state. The released view is always harness-reconstructed from raw data, hashed into ICSD evidence, and appended to a transparency log with a verifiable receipt.
