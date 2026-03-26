# Verifiable MRI Guardian

This repository contains a proof-of-concept ICSD harness that sits between a raw MRI reconstruction AI and the radiologist workstation.

The core rule is simple: the radiologist never receives AI-supplied pixels. The harness treats the AI reconstruction as a claim, recomputes the displayed reconstruction from raw k-space itself, and only releases a workstation payload when all of the following are true:

- the reconstructed voxel grid is mathematically consistent with the raw sensor data
- the AI reconstruction matches the harness-derived reconstruction within a fixed tolerance
- the AI tumour mask exactly matches a deterministic policy mask derived from the raw-derived reconstruction
- the tumour volume exactly equals `verified_mask_voxel_count * voxel_spacing`

The ICSD layer makes the release state invariant-closed. A case is either `pending_verification` with no released payload, or `released_to_radiologist` with a `verified_manifest` and `workstation_payload` that exactly match the expected mathematical trace. If the AI invents pixels or misstates the volume, the released state becomes unrepresentable and the transition fails.

## Structure

- [mri_guardian/harness.py](mri_guardian/harness.py) defines the policy profile, invariants, deterministic reconstruction checks, and release transition.
- [mri_guardian/tamper.py](mri_guardian/tamper.py) builds deterministic fake-tumour, shifted-mask, and intensity-manipulation scenarios.
- [mri_guardian/transparency.py](mri_guardian/transparency.py) implements a file-backed append-only hash chain for ICSD transparency receipts.
- [mri_guardian/demo.py](mri_guardian/demo.py) writes a deterministic demo evidence bundle and workstation payload.
- [mri_guardian/viewer.py](mri_guardian/viewer.py) renders the side-by-side "AI vs Verified Reality" review page with mismatch and rejected-region highlights.
- [tests/test_guardian.py](tests/test_guardian.py) covers the happy path and two rejection paths.

## Run

Run the deterministic demo:

```powershell
.\.venv\Scripts\python.exe -m mri_guardian.demo --output-dir demo_artifacts --tamper-mode
```

Verify the emitted ICSD evidence envelope:

```powershell
.\.venv\Scripts\icsd-dpa verify demo_artifacts\evidence.json
```

Open the visual comparison outputs:

- `demo_artifacts/comparison.html`
- `demo_artifacts/tamper_baseline.html`
- `demo_artifacts/tamper_fake_tumour.html`
- `demo_artifacts/tamper_shifted_mask.html`
- `demo_artifacts/tamper_intensity_manipulation.html`

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Why this satisfies the challenge

The challenge is not only to log what the AI said, but to prevent unverifiable output from ever reaching the workstation.

This prototype does that by making the workstation payload a deterministic function of:

- raw k-space
- voxel spacing
- a policy-defined verified mask derived from the raw-derived reconstruction
- the AI mask and volume claim
- the active ICSD policy profile

The AI can suggest a reconstruction, but it cannot inject a different pixel raster into the released state. The released view is always harness-reconstructed from raw data, hashed into ICSD evidence, and appended to a transparency log with a verifiable receipt.

The added viewer makes that boundary human-visible. It renders:

- left: AI reconstruction plus the AI tumour mask
- right: deterministic reconstruction plus the admissible mask
- highlighted mismatched voxels
- highlighted rejected mask regions where the AI overlay rests on unsupported pixels

The tamper-mode reports make failure detection explicit:

- `fake_tumour` injects 342 unsupported tumour voxels and fails with `mask_voxel_mismatch`
- `shifted_mask` moves the overlay off the verified anatomy and fails with `mask_voxel_mismatch`
- `intensity_manipulation` alters the AI image while raw k-space stays fixed and fails with `intensity_manipulation_detected`

This segmentation rule is intentionally simple for the prototype: the verified mask is derived by a deterministic threshold over the raw-derived reconstruction. The point of the harness is not clinical segmentation quality; it is proving that the workstation view and volume estimate are mathematically anchored to measured data and that tampering is detectable.
