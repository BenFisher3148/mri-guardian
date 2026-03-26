# MRI Guardian Layer

This repository contains a proof-of-concept MRI Guardian Layer: an ICSD verification proxy that sits between a raw MRI reconstruction AI and the downstream PACS or radiologist workstation.

It is intentionally non-invasive. The reconstruction AI and the PACS/workstation stay in place; the Guardian Layer only verifies the handoff and blocks any output that cannot be proven from the measured MRI data.

The core rule is simple: the downstream PACS/workstation never receives AI-supplied pixels directly. The layer treats the AI reconstruction as a claim, recomputes the displayed reconstruction from raw k-space itself, and only releases a workstation payload when all of the following are true:

- the reconstructed voxel grid is mathematically consistent with the raw sensor data
- the AI reconstruction matches the harness-derived reconstruction under an explicit tolerance policy:
  spatial per-voxel max-abs error plus a global relative L2 bound
- the AI tumour mask exactly matches a deterministic policy mask derived from the raw-derived reconstruction
- the tumour volume exactly equals `verified_mask_voxel_count * voxel_spacing`

The verification envelope is now split into three explainability tiers:

- Level 1: Structural validity. ICSD-facing checks over identifiers, numeric tensors, voxel geometry, and release-state shape.
- Level 2: Physical consistency. Raw k-space reprojection, deterministic reconstruction parity, mask parity, and tumour-volume proof.
- Level 3: Clinical plausibility. Advisory-only heuristics over the verified anatomy, intended to prompt review without blocking release.

The ICSD verification layer makes the release state invariant-closed. A case is either `pending_verification` with no released payload, or `released_to_radiologist` with a `verified_manifest` and `workstation_payload` that exactly match the expected mathematical trace. If the AI invents pixels or misstates the volume, the released state becomes unrepresentable and the transition fails.

## Structure

- [mri_guardian/bundle.py](mri_guardian/bundle.py) builds and verifies portable `.mri-guard-bundle` proof artifacts.
- [mri_guardian/contracts.py](mri_guardian/contracts.py) defines the reusable `AIClaimEnvelope`, `VerifiedReconstructionTrace`, and `MaskDerivationPolicy` interfaces.
- [mri_guardian/harness.py](mri_guardian/harness.py) defines the policy profile, invariants, deterministic reconstruction checks, and release transition.
- [mri_guardian/integration.py](mri_guardian/integration.py) mocks the DICOM ingress and PACS egress contracts around the verification proxy.
- [mri_guardian/tamper.py](mri_guardian/tamper.py) builds deterministic fake-tumour, shifted-mask, and intensity-manipulation scenarios.
- [mri_guardian/timing.py](mri_guardian/timing.py) profiles benchmark ladders, worst-case bounds, and the streaming-validation strategy for the verification path.
- [mri_guardian/transparency.py](mri_guardian/transparency.py) implements a file-backed append-only hash chain for ICSD transparency receipts.
- [mri_guardian/demo.py](mri_guardian/demo.py) writes a deterministic demo evidence bundle and downstream workstation payload.
- [mri_guardian/viewer.py](mri_guardian/viewer.py) renders the side-by-side Guardian Layer review page with mismatch and rejected-region highlights.
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

Inspect the mocked integration artifacts:

- `demo_artifacts/dicom_input.json`
- `demo_artifacts/integration_narrative.json`
- `demo_artifacts/pacs_submission.json`
- `demo_artifacts/integration_trace.json`
- `demo_artifacts/tamper_fake_tumour_pacs_block.json`
- `demo_artifacts/tamper_shifted_mask_pacs_block.json`
- `demo_artifacts/tamper_intensity_manipulation_pacs_block.json`

Inspect the explicit claim-to-proof interfaces:

- `demo_artifacts/ai_claim_envelope.json`
- `demo_artifacts/verified_reconstruction_trace.json`
- `demo_artifacts/mask_derivation_policy.json`
- `demo_artifacts/verification_tolerance_policy.json`

Inspect the performance evidence:

- `demo_artifacts/performance_profile.json`

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

The challenge is not only to log what the AI said, but to prevent unverifiable output from ever reaching the PACS/workstation boundary.

This prototype does that by making the Guardian Layer handoff payload a deterministic function of:

- raw k-space
- voxel spacing
- a policy-defined verified mask derived from the raw-derived reconstruction
- the AI mask and volume claim
- the active ICSD policy profile

The AI can suggest a reconstruction, but it cannot inject a different pixel raster into the released state. The released view is always recomputed inside the verification layer from raw data, hashed into ICSD evidence, and appended to a transparency log with a verifiable receipt.

That interface is now explicit rather than implied. The harness emits:

- `AIClaimEnvelope`: the AI-side reconstruction, mask, and volume claim anchored to the raw-input hash
- `MaskDerivationPolicy`: the deterministic rule that turns a raw-derived reconstruction into the admissible mask
- `VerificationTolerancePolicy`: the attack-aware tolerance contract that states which metrics are per-voxel, global-norm, and frequency-domain
- `VerifiedReconstructionTrace`: the proof object that links the raw measurements, derived reconstruction, mask policy, and verified volume back to the original claim

Tolerance handling is also explicit rather than hand-wavy. The verifier now defines:

- spatial AI parity by both per-voxel max-abs error and global relative L2 error
- frequency-domain measurement consistency by both per-frequency max-abs error and global relative L2 error
- a separate imaginary-leakage bound on the inverse FFT output
- a deterministic guard band so near-threshold cases are marked as boundary-hugging rather than looking identical to comfortably passing cases

The portable bundle pushes that one step further. Each accepted case now emits a shareable `.mri-guard-bundle` archive containing:

- the ICSD evidence envelope
- the released state and workstation payload
- a deterministic verification transcript
- the original input state plus replay context
- the transparency-log snapshot that anchored the release
- portable integrity signatures over the bundle members
- top-level hashes for raw k-space, reconstruction, and mask state

That means a reviewer can carry the artifact elsewhere, verify it offline, and still check that the scan remained anchored to the same measured data and Guardian Layer release proof.

The replay layer makes the bundle regulator-friendly: `verify_bundle(...)` does not just check static integrity. It replays the release from the bundled `input_state.json`, `replay.json`, and transparency-log prefix, then confirms that the regenerated released state, evidence payload, and bundle members are identical.

The added viewer makes that boundary human-visible. It renders the verification proxy exactly as a reviewer would experience it:

- a clinician-facing summary panel with status, verified tumour volume, raw-data confidence wording, and a deterministic audit ID
- a mocked `DICOM Input -> MRI Guardian Layer -> PACS Output` boundary strip so the plug-in point is explicit
- left: AI reconstruction plus the AI tumour mask proposed for downstream handoff
- right: Guardian Layer reconstruction plus the admissible verified mask
- Level 1 / Level 2 / Level 3 verification cards so reviewers can see which trust layer passed, failed, or only warned
- highlighted mismatched voxels
- highlighted rejected mask regions where the AI overlay rests on unsupported pixels
- a timing overlay that measures representative reconstruction latency, verification latency, total overhead, and a profiled worst-case bound

The tamper-mode reports make failure detection explicit:

- `fake_tumour` injects 342 unsupported tumour voxels and fails with `mask_voxel_mismatch`
- `shifted_mask` moves the overlay off the verified anatomy and fails with `mask_voxel_mismatch`
- `intensity_manipulation` alters the AI image while raw k-space stays fixed and fails with `intensity_manipulation_detected`

The fake integration layer answers the operational question directly. The demo now shows:

- mocked DICOM ingress carrying raw k-space plus the AI claim
- the Guardian Layer acting as a verification proxy rather than a replacement system
- mocked PACS egress that either emits verified series or records a blocked handoff with the exact failure reason

The PACS integration narrative is now explicit rather than conceptual:

- the DICOM compliance story states that the Guardian Layer receives DICOM-wrapped MRI evidence and AI candidate objects, preserves study and patient context, and emits only verified derived series back onto the hospital imaging network
- the hospital-placement story states that it sits after the reconstruction AI and before PACS/workstation import, behaving like a gateway or sidecar rather than a new clinical system
- the failure behaviour is fail-closed for the Guardian-verified handoff: if proof fails or the proxy is unavailable, no verified reconstruction, segmentation, manifest, or evidence series are auto-forwarded from the Guardian path

This segmentation rule is intentionally simple for the prototype: the verified mask is derived by a deterministic threshold over the raw-derived reconstruction. The point of the verification layer is not clinical segmentation quality; it is proving that the workstation view and volume estimate are mathematically anchored to measured data and that tampering is detectable.

Level 3 should be read in that light: it does not claim diagnosis quality. It is an optional, deterministic heuristics layer that helps frame whether the verified anatomy still looks clinically reasonable enough for human follow-up.

The performance story is now explicit rather than anecdotal. The timing layer uses cached synthetic probes derived from the deterministic tamper baseline and profiles three shapes:

- a streaming slab for early partial validation
- a representative review volume for the headline timing cards
- a larger high-load probe used to derive a guarded worst-case latency bound

The emitted `performance_profile.json` records the benchmark ladder, the profiled worst-case bound, and the active NumPy/Numba backend.

The worst-case bound is intentionally shape-scoped rather than hand-wavy. It is computed from the largest profiled volume, using the observed max across warmed runs plus a deterministic safety margin, so reviewers can see exactly what latency claim is being made and for which volume size.

The streaming / partial validation strategy is also explicit:

- per-object structural checks run before FFT work starts
- raw k-space slabs are canonicalised and hashed as they arrive
- completed slabs can be reconstructed and parity-checked before the full series closes
- final PACS/workstation release still waits for the full-volume proof, so streaming reduces latency without weakening the release gate
