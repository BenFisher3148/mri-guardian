# Verifiable MRI Guardian

This repository contains a proof-of-concept ICSD harness that sits between a raw MRI reconstruction AI and the radiologist workstation.

The core rule is simple: the radiologist never receives AI-supplied pixels. The harness treats the AI reconstruction as a claim, recomputes the displayed reconstruction from raw k-space itself, and only releases a workstation payload when all of the following are true:

- the reconstructed voxel grid is mathematically consistent with the raw sensor data
- the AI reconstruction matches the harness-derived reconstruction within a fixed tolerance
- the AI tumour mask exactly matches a deterministic policy mask derived from the raw-derived reconstruction
- the tumour volume exactly equals `verified_mask_voxel_count * voxel_spacing`

The verification envelope is now split into three explainability tiers:

- Level 1: Structural validity. ICSD-facing checks over identifiers, numeric tensors, voxel geometry, and release-state shape.
- Level 2: Physical consistency. Raw k-space reprojection, deterministic reconstruction parity, mask parity, and tumour-volume proof.
- Level 3: Clinical plausibility. Advisory-only heuristics over the verified anatomy, intended to prompt review without blocking release.

The ICSD layer makes the release state invariant-closed. A case is either `pending_verification` with no released payload, or `released_to_radiologist` with a `verified_manifest` and `workstation_payload` that exactly match the expected mathematical trace. If the AI invents pixels or misstates the volume, the released state becomes unrepresentable and the transition fails.

## Structure

- [mri_guardian/bundle.py](mri_guardian/bundle.py) builds and verifies portable `.mri-guard-bundle` proof artifacts.
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

Inspect the portable proof artifact:

- `demo_artifacts/mri-case-0001.mri-guard-bundle`
- `demo_artifacts/bundle_verification.json`

Verify the archive or an extracted `bundle.json` replay package in Python:

```python
from mri_guardian import verify_bundle

result = verify_bundle("demo_artifacts/mri-case-0001.mri-guard-bundle")
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

The portable bundle pushes that one step further. Each accepted case now emits a shareable `.mri-guard-bundle` archive containing:

- the ICSD evidence envelope
- the released state and workstation payload
- a deterministic verification transcript
- the original input state plus replay context
- the transparency-log snapshot that anchored the release
- portable integrity signatures over the bundle members
- top-level hashes for raw k-space, reconstruction, and mask state

That means a reviewer can carry the artifact elsewhere, verify it offline, and still check that the scan remained anchored to the same measured data and release proof.

The replay layer makes the bundle regulator-friendly: `verify_bundle(...)` does not just check static integrity. It replays the release from the bundled `input_state.json`, `replay.json`, and transparency-log prefix, then confirms that the regenerated released state, evidence payload, and bundle members are identical.

The added viewer makes that boundary human-visible. It renders:

- left: AI reconstruction plus the AI tumour mask
- right: deterministic reconstruction plus the admissible mask
- Level 1 / Level 2 / Level 3 verification cards so reviewers can see which trust layer passed, failed, or only warned
- highlighted mismatched voxels
- highlighted rejected mask regions where the AI overlay rests on unsupported pixels
- a timing overlay that measures reconstruction, verification, and total overhead on a cached in-memory performance probe

The tamper-mode reports make failure detection explicit:

- `fake_tumour` injects 342 unsupported tumour voxels and fails with `mask_voxel_mismatch`
- `shifted_mask` moves the overlay off the verified anatomy and fails with `mask_voxel_mismatch`
- `intensity_manipulation` alters the AI image while raw k-space stays fixed and fails with `intensity_manipulation_detected`

This segmentation rule is intentionally simple for the prototype: the verified mask is derived by a deterministic threshold over the raw-derived reconstruction. The point of the harness is not clinical segmentation quality; it is proving that the workstation view and volume estimate are mathematically anchored to measured data and that tampering is detectable.

Level 3 should be read in that light: it does not claim diagnosis quality. It is an optional, deterministic heuristics layer that helps frame whether the verified anatomy still looks clinically reasonable enough for human follow-up.

The timing overlay uses a cached synthetic probe derived from the deterministic tamper baseline, scaled to a larger in-memory volume. It reports median reconstruction and verification timings across repeated runs and includes the active NumPy/Numba backend so reviewers can see the numerical path is fast enough to be practical.
