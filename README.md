# MRI Guardian Layer

This repository contains a proof-of-concept MRI Guardian Layer: an ICSD verification proxy that sits between a raw MRI reconstruction AI and the downstream PACS or radiologist workstation.

It is intentionally non-invasive. The reconstruction AI and the PACS/workstation stay in place; the Guardian Layer only verifies the handoff and blocks any output that cannot be proven from the measured MRI data.

## At A Glance

- What it is: a verification layer between a raw MRI AI and PACS/workstation handoff.
- What it prevents: invented pixels, unsupported tumour masks, and incorrect tumour-volume claims moving through the Guardian path.
- What the demo shows: accepted verification, live tamper detection, portable evidence bundles, mocked DICOM/PACS integration, bounded performance, additive `multi_series_v2` study gating, and optional Level 3 longitudinal verification against one prior Guardian-verified comparator.
- What it is not: a replacement PACS, scanner, or reporting workstation.

## Quick Review Path

If you are short on time, the fastest way to understand the repo is:

1. Install the general environment from [requirements.txt](requirements.txt) or the exact reviewed stack from [requirements-review.txt](requirements-review.txt), then install the local package with `pip install -e .`.
2. Run the deterministic demo with `mri-guardian demo --output-dir demo_artifacts --profile brain_t1_baseline --tamper-mode`.
   Or run the additive study-level path with `mri-guardian demo --output-dir demo_artifacts --case-kind multi_series_v2`.
3. Open `demo_artifacts/comparison.html` and one of the `tamper_*.html` reports.
4. If you want the deeper model, then read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/INTEROPERABILITY.md](docs/INTEROPERABILITY.md), and [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Happy Path Commands

After `pip install -e .`, the repository exposes a single CLI:

```powershell
mri-guardian demo --output-dir demo_artifacts --profile brain_t1_baseline --tamper-mode
mri-guardian demo --output-dir demo_artifacts --case-kind multi_series_v2
mri-guardian verify demo_artifacts
mri-guardian test
```

The same commands also work as a module entrypoint:

```powershell
python -m mri_guardian demo --output-dir demo_artifacts --profile brain_t1_baseline --tamper-mode
python -m mri_guardian demo --output-dir demo_artifacts --case-kind multi_series_v2
python -m mri_guardian verify demo_artifacts
python -m mri_guardian test
```

## Verification Model

The core rule is simple: the downstream PACS/workstation never receives AI-supplied pixels directly. The layer treats the AI reconstruction as a claim, recomputes the displayed reconstruction from raw k-space itself, and only releases a workstation payload when all of the following are true:

- the reconstructed voxel grid is mathematically consistent with the raw sensor data
- the AI reconstruction matches the harness-derived reconstruction under an explicit tolerance profile:
  spatial per-voxel max-abs error plus a global relative L2 bound
- the AI tumour mask exactly matches a deterministic profile mask derived from the raw-derived reconstruction
- the tumour volume exactly equals `verified_mask_voxel_count * voxel_spacing`

The verification envelope is now split into three explainability tiers:

- Level 1: Structural validity. ICSD-facing checks over identifiers, numeric tensors, voxel geometry, and release-state shape.
- Level 2: Physical consistency. Raw k-space reprojection, deterministic reconstruction parity, mask parity, and tumour-volume proof.
- Level 3: Clinical plausibility. Profile-scoped heuristics over the verified anatomy, with centrally defined `disabled`, `advisory`, `warning`, and `blocking` semantics.

The repository now also supports an additive `multi_series_v2` study container. In that path:

- every payload series is individually verified and reported
- `required_series_ids` is the only authoritative merged-release gating set
- auxiliary series are preserved and reported but remain non-gating for this milestone
- duplicate payload `series_id` values and empty `supporting_series_ids` are hard study-shape failures; the path does not auto-downgrade to single-series
- same-grid geometry uses exact equality over canonicalized voxel spacing plus voxel shape
- positive-lesion corroboration uses exact coordinate intersection count `> 0`

Single-series Level 3 now also supports an optional longitudinal layer. When callers supply one admissible `prior_verified_context`, the report emits a canonical `TemporalConsistencyTrace` with explicit:

- timestamp trust classes and deterministic pair-provenance derivation
- `comparison_status` (`not_evaluated`, `aligned`, `degraded`, `unsuitable`)
- `comparison_scope` (`not_evaluated`, `full`, `volume_only`, `none`)
- chronology, geometry, metric, flag, and skipped-metric detail

V1 longitudinal rules are intentionally narrow:

- temporal findings are advisory only and do not change the hard release gate
- exactly one prior comparator is supported
- `<= 730` days is the preferred aligned band, `731-3650` days is degraded-but-usable, and `> 3650` days suppresses temporal comparison by policy
- these interval bands express comparability confidence only, not clinical expectation
- missing or untrusted timestamps suppress only time-dependent metrics; exact-geometry or scalar-volume comparison may still run as degraded

## Runtime Profiles

The Guardian Layer now has a runtime profile registry rather than a single verification family.

- `profile_name` selects the Guardian runtime profile.
- `profile_revision` is scoped to `profile_name` and must change whenever that named profile's canonical descriptor body changes.
- `profile_digest` is a canonical SHA-256 digest over the resolved `VerificationProfileDescriptor` with stable field ordering and an explicit digest scope.
- `icsd_policy_id` and `icsd_policy_version` identify the underlying ICSD invariant policy separately from the runtime profile.

Built-in profiles:

- `brain_t1_baseline`
- `brain_t1_morphology_review`
- `brain_t1_component_review`
- `brain_t1_slice_review`
- `brain_t2_baseline`
- `contrast_enhanced_review`
- `strict_research_mode`
- `fast_triage_mode`

`fast_triage_mode` is intentionally non-final. It can produce a deterministic proof result for pre-screening, but it cannot release a final Guardian state or portable evidence bundle.

For future additions, prefer profile names shaped like `<domain>_<sequence>_<policy_family>_<posture>`. Existing names are kept for continuity.

The ICSD verification layer makes the release state invariant-closed. A case is either `pending_verification` with no released payload, or `released_to_radiologist` with a `verified_manifest` and `workstation_payload` that exactly match the expected mathematical trace. If the AI invents pixels or misstates the volume, the released state becomes unrepresentable and the transition fails.

## Approved Mask Algorithm Catalogue

Verified-mask derivation uses a fixed in-repo approved algorithm catalogue, not a plugin system:

- no environment discovery
- no user-installed engines
- no module scanning
- no runtime algorithm override for released verification

Algorithm identifiers are proof-significant names and must remain stable once published in released profiles or replay artifacts.

Canonical parameter resolution happens before execution. The dispatcher:

- rejects unknown or algorithm-inapplicable parameters
- preserves integers as integers and booleans as booleans
- normalises enum-like values to canonical lowercase strings
- removes ambiguity between omitted defaults and explicit defaults by serialising one resolved parameter object
- executes from that resolved object and serialises that same object into the profile, manifest, bundle, and replay surfaces

Only proof-significant outputs are serialized:

- verified mask array
- canonical algorithm identity
- canonical resolved parameters

Runtime metadata such as timing, internal operation counts, or debug notes stays out of manifests, bundle hashes, and replay material.

Built-in approved algorithms:

- `threshold_on_raw_derived_reconstruction`
- `morphology_constrained_threshold_on_raw_derived_reconstruction`
- `connected_component_constrained_threshold_on_raw_derived_reconstruction`
- `slice_consistency_constrained_threshold_on_raw_derived_reconstruction`

For the component-constrained family, `connectivity` is an explicit proof-significant
parameter. This milestone approves only `faces`; any other value is rejected during
canonical resolution. Component labels remain internal deterministic execution
state and are not part of the proof-significant output surface.

For the slice-consistency family, each vertex belongs to exactly one slice index and
edges are permitted only between vertices from slices `z` and `z+1`. Adjacent-slice
overlap uses non-empty intersection of the two components' `(y, x)` footprint sets.
Traversal order is fixed for reproducibility only; survival is defined by membership
in a qualifying no-gap run of consecutive slices, not by label rank.

## Structure

- [mri_guardian/bundle.py](mri_guardian/bundle.py) builds and verifies portable `.mri-guard-bundle` proof artifacts.
- [mri_guardian/contracts.py](mri_guardian/contracts.py) defines the reusable claim/proof and longitudinal interfaces, including `AIClaimEnvelope`, `VerifiedReconstructionTrace`, `PriorVerifiedContext`, and `TemporalConsistencyTrace`.
- [mri_guardian/harness.py](mri_guardian/harness.py) defines the release transition, invariants, deterministic reconstruction checks, and manifest generation.
- [mri_guardian/integration.py](mri_guardian/integration.py) mocks the DICOM ingress and PACS egress contracts around the verification proxy.
- [mri_guardian/mask_engine.py](mri_guardian/mask_engine.py) defines the fixed approved mask-algorithm catalogue, canonical parameter resolution, and deterministic dispatcher.
- [mri_guardian/profiles.py](mri_guardian/profiles.py) defines the runtime profile registry, canonical descriptor hashing, and the mapping from runtime profiles to ICSD policy identities.
- [mri_guardian/tamper.py](mri_guardian/tamper.py) builds deterministic fake-tumour, shifted-mask, and intensity-manipulation scenarios.
- [mri_guardian/timing.py](mri_guardian/timing.py) profiles benchmark ladders, worst-case bounds, and the streaming-validation strategy for the verification path.
- [mri_guardian/transparency.py](mri_guardian/transparency.py) implements a file-backed append-only hash chain for ICSD transparency receipts.
- [mri_guardian/demo.py](mri_guardian/demo.py) writes a deterministic demo evidence bundle and downstream workstation payload.
- [mri_guardian/viewer.py](mri_guardian/viewer.py) renders the side-by-side Guardian Layer review page with mismatch and rejected-region highlights.
- [tests/test_guardian.py](tests/test_guardian.py) covers release, bundle replay, tamper rejection, mocked DICOM/PACS integration, viewer rendering, and performance-profile regressions.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/SECURITY.md](docs/SECURITY.md)
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- [docs/INTEROPERABILITY.md](docs/INTEROPERABILITY.md)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/DATA_DICT.md](docs/DATA_DICT.md)
- [docs/adr/README.md](docs/adr/README.md)
- [CITATION.cff](CITATION.cff)

## Fresh Setup

For a cold-start reviewer, the repository now has both package metadata and two dependency surfaces:

- [pyproject.toml](pyproject.toml) defines the general package metadata and dependency ranges
- [requirements.txt](requirements.txt) is the general development environment
- [requirements-review.txt](requirements-review.txt) is the pinned reproducible review stack used for the documented benchmark/demo environment

Recommended general clone -> install -> test path:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\mri-guardian.exe test
```

If you want the exact reviewed environment used in the current docs and demo artifacts, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-review.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Optional extra acceleration dependencies are exposed through the package extras:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[speed]"
```

## Run

Run the deterministic demo:

```powershell
mri-guardian demo --output-dir demo_artifacts --profile brain_t1_baseline --tamper-mode
```

Verify the emitted evidence and bundle artifacts:

```powershell
mri-guardian verify demo_artifacts
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
- `demo_artifacts/verification_profile.json`
- `demo_artifacts/verification_profile_descriptor.json`
- `demo_artifacts/verified_reconstruction_trace.json`
- `demo_artifacts/mask_derivation_policy.json`
- `demo_artifacts/verification_tolerance_policy.json`
- `demo_artifacts/prior_verified_context.json` when a prior comparator is supplied
- `demo_artifacts/temporal_consistency_trace.json` when a prior comparator is supplied

Inspect the performance evidence:

- `demo_artifacts/performance_profile.json`

For a non-final triage run:

```powershell
mri-guardian demo --output-dir demo_artifacts --profile fast_triage_mode
```

That path emits the report, profile descriptor, performance profile, and HTML view only. It does not emit a released state, evidence payload, transparency log, or portable bundle.

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
mri-guardian test
```

## Why this satisfies the challenge

The challenge is not only to log what the AI said, but to prevent unverifiable output from ever reaching the PACS/workstation boundary.

This prototype does that by making the Guardian Layer handoff payload a deterministic function of:

- raw k-space
- voxel spacing
- a profile-defined verified mask derived from the raw-derived reconstruction
- the AI mask and volume claim
- the active Guardian runtime profile plus its underlying ICSD invariant policy identity

The AI can suggest a reconstruction, but it cannot inject a different pixel raster into the released state. The released view is always recomputed inside the verification layer from raw data, hashed into ICSD evidence, and appended to a transparency log with a verifiable receipt.

That interface is now explicit rather than implied. The harness emits:

- `AIClaimEnvelope`: the AI-side reconstruction, mask, and volume claim anchored to the raw-input hash, with optional canonical declared `ai_stack_identity`
- `MaskDerivationPolicy`: the approved deterministic algorithm identity and canonical resolved parameters that turn a raw-derived reconstruction into the admissible mask
- `VerificationTolerancePolicy`: the attack-aware tolerance contract that states which metrics are per-voxel, global-norm, and frequency-domain
- `VerificationProfileDescriptor`: the canonical runtime profile contract with `profile_name`, `profile_revision`, `profile_digest`, release semantics, authority requirements, heuristics, and performance expectations
- `VerifiedReconstructionTrace`: the proof object that links the raw measurements, derived reconstruction, mask policy, and verified volume back to the original claim
- `PriorVerifiedContext`: the compact canonical summary of one admissible prior Guardian-verified comparator
- `TemporalConsistencyTrace`: the deterministic Level 3 longitudinal trace with chronology, geometry, signed change metrics, flags, and skipped-metric reasons

When present, `ai_stack_identity` is canonical claim provenance rather than a new physical proof. Guardian canonicalizes and hashes the producer declaration into the claim identity, but it does not independently verify that the declared model stack itself is true.

Tolerance handling is also explicit rather than hand-wavy. The verifier now defines:

- spatial AI parity by both per-voxel max-abs error and global relative L2 error
- frequency-domain measurement consistency by both per-frequency max-abs error and global relative L2 error
- a separate imaginary-leakage bound on the inverse FFT output
- a deterministic guard band so near-threshold cases are marked as boundary-hugging rather than looking identical to comfortably passing cases

Verification reports now expose two decision layers:

- the hard handoff gate remains `release_eligible` plus the final released/blocked disposition
- the additive canonical `decision` object carries the richer monitoring class used by new dashboards and summaries

The canonical `decision` object fixes `evaluation_stage` to one of:

- `intake_validation`
- `structural_validity`
- `physical_consistency`
- `clinical_plausibility`
- `policy_evaluation`
- `multi_series_aggregation`
- `release_control`

The canonical `decision.decision_class` values are:

- `verified_clean`
- `verified_boundary_hugging`
- `verified_with_advisory`
- `blocked_structural`
- `blocked_physical`
- `blocked_policy`

The report keeps `decision_summary` for compatibility, but that legacy block is no longer the canonical contract for new monitoring logic. New consumers should use `decision.decision_class`.

Reason ordering inside `decision.reason_codes` and `decision.decision_reasons` is deterministic:

- the primary class-determining reason appears first
- same-stage supporting reasons follow
- secondary modifiers follow after that
- advisory and info reasons come last, with lexical tie-breaks inside each band

`verified_boundary_hugging` is also pinned down explicitly: only passing proof metrics with `boundary_status = near_boundary` are eligible, and `near_boundary` means "within the configured guard band of the relevant failure threshold." The current eligible metrics are the measurement-consistency metrics, AI-reconstruction parity metrics, and the tumour-volume tolerance metric. Exact mask equality is still binary and does not participate in guard-band classification.

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

Replay identity now includes the embedded runtime profile descriptor. A verifier can therefore prove not just "the profile name was `brain_t1_baseline`" but "the exact canonical descriptor body used at runtime had this digest."

The same distinction applies to declared AI stack provenance: if `ai_stack_identity` is present, replay can prove that the released artifacts were bound to that exact declaration. It does not elevate that declaration into an independently proven model-validity statement.

The added viewer makes that boundary human-visible. It renders the verification proxy exactly as a reviewer would experience it:

- a clinician-facing summary panel with status, verified tumour volume, raw-data confidence wording, and a deterministic audit ID
- a mocked `DICOM Input -> MRI Guardian Layer -> PACS Output` boundary strip so the plug-in point is explicit
- left: AI reconstruction plus the AI tumour mask proposed for downstream handoff
- right: Guardian Layer reconstruction plus the admissible verified mask
- Level 1 / Level 2 / Level 3 verification cards so reviewers can see which trust layer passed, failed, or only warned
- a longitudinal summary card with comparison status, comparison scope, signed change, normalized rate, IoU, and centroid shift when prior context is available
- highlighted mismatched voxels
- highlighted rejected mask regions where the AI overlay rests on unsupported pixels
- a faded prior-mask overlay on the verified panel only when exact geometry comparability passed
- a timing overlay that measures representative reconstruction latency, verification latency, total overhead, and a profiled worst-case bound

The tamper-mode reports make failure detection explicit:

- `fake_tumour` injects 342 unsupported tumour voxels and fails with `mask_voxel_mismatch`
- `shifted_mask` moves the overlay off the verified anatomy and fails with `mask_voxel_mismatch`
- `intensity_manipulation` alters the AI image while raw k-space stays fixed and fails with `intensity_manipulation_detected`
- `component_micro_island` appears under `brain_t1_component_review` and fails when the AI restores a detached threshold-positive island that the component policy rejects
- `slice_single_slice_island` appears under `brain_t1_slice_review` and fails when the AI restores a detached single-slice threshold-positive island that the slice-consistency policy rejects

The fake integration layer answers the operational question directly. The demo now shows:

- mocked DICOM ingress carrying raw k-space plus the AI claim
- the Guardian Layer acting as a verification proxy rather than a replacement system
- mocked PACS egress that either emits verified series or records a blocked handoff with the exact failure reason

The PACS integration narrative is now explicit rather than conceptual:

- the DICOM compliance story states that the Guardian Layer receives DICOM-wrapped MRI evidence and AI candidate objects, preserves study and patient context, and emits only verified derived series back onto the hospital imaging network
- the hospital-placement story states that it sits after the reconstruction AI and before PACS/workstation import, behaving like a gateway or sidecar rather than a new clinical system
- the failure behaviour is fail-closed for the Guardian-verified handoff: if proof fails or the proxy is unavailable, no verified reconstruction, segmentation, manifest, or evidence series are auto-forwarded from the Guardian path

The baseline release profiles still use a deterministic threshold rule because it is easy to audit and keeps the original proof-of-concept intact. The repository now also includes morphology-constrained, connected-component-constrained, and slice-consistency-constrained deterministic showcase profiles to demonstrate that the verification layer can support multiple policy families without turning mask derivation into runtime-selectable freedom.

The point of the verification layer is still not clinical segmentation quality by itself. It is proving that the workstation view and volume estimate are mathematically anchored to measured data and that tampering is detectable.

Level 3 should be read in that light: it does not claim diagnosis quality. It is an optional, deterministic heuristics layer that helps frame whether the verified anatomy still looks clinically reasonable enough for human follow-up.

The performance story is now explicit rather than anecdotal. The timing layer uses cached synthetic probes derived from the deterministic tamper baseline and profiles three shapes:

- a streaming slab for early partial validation
- a representative review volume for the headline timing cards
- a larger high-load probe used to derive a guarded worst-case latency bound

The emitted `performance_profile.json` records the benchmark ladder, the profiled worst-case bound, the active profile descriptor, and budget-versus-observed comparisons. Those budgets are monitoring surfaces, not proof gates.

The worst-case bound is intentionally shape-scoped rather than hand-wavy. It is computed from the largest profiled volume, using the observed max across warmed runs plus a deterministic safety margin, so reviewers can see exactly what latency claim is being made and for which volume size.

The streaming / partial validation strategy is also explicit:

- per-object structural checks run before FFT work starts
- raw k-space slabs are canonicalised and hashed as they arrive
- completed slabs can be reconstructed and parity-checked before the full series closes
- final PACS/workstation release still waits for the full-volume proof, so streaming reduces latency without weakening the release gate
