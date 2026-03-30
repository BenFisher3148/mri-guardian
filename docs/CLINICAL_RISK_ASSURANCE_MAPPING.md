# Clinical Risk & Assurance Mapping (Prototype Alignment)

This document translates the MRI Guardian Layer prototype into clinical risk and assurance language for review. It uses NHS/DCB0129-style reasoning to explain controls, evidence, and residual risk without claiming compliance, certification, or diagnostic correctness.

The Guardian Layer is a post-processing verification and release-gating layer applied to AI-derived imaging outputs. It does not originate clinical findings and does not replace clinician judgement.

## Purpose and Audience

This document is intended for:

- clinical-safety-minded technical reviewers
- engineering reviewers who need a safety-language translation of the repository
- interview or prototype reviewers who need a concise assurance narrative

In this document, "Prototype Alignment" means alignment in language, structure, and safety reasoning style, not formal NHS approval.

## System Boundary

```text
[Scanner / acquisition] -> out of scope for acquisition correctness, calibration, and raw-data truth
        |
        v
[AI reconstruction + segmentation + volume claim] -> input claim surface
        |
        v
[Guardian Layer] -> in scope
  - intake_validation / structural_validity
  - physical_consistency
  - optional clinical_plausibility, temporal, and multi-series checks
  - release_control
        |
        v
[PACS / viewer / clinician] -> downstream consumer
```

- The prototype verifies whether downstream outputs can be justified from the supplied data and the active policy.
- It does not validate whether upstream acquisition itself was correct, whether scanner calibration was correct, or whether a clinically useful finding exists.

## Explicit Non-Claims

- This is not a DCB0129 safety case.
- This is not a compliance or regulatory submission.
- This is not a claim of diagnostic correctness.
- This is not a replacement for clinician judgement.
- This does not guarantee scanner behavior, acquisition fidelity, or institutional workflow safety outside the Guardian path.

## Control Behavior Notes

- When a blocking structural, physical, or policy control fails, the Guardian path transitions fail-closed; only controls intentionally configured as advisory can record findings without blocking release.
- Residual risk in this document means what could still go wrong after the listed controls pass or the system responds as designed. It does not restate hazards that have already been blocked.
- Tests are treated as executable evidence of control behaviour, not illustrative examples.

## Safety-Language Translation

### Input Integrity Controls

| Repo surface | Safety-language interpretation | Control type | Assurance contribution |
| --- | --- | --- | --- |
| `intake_validation` | Input acceptance gate | Preventative | Rejects malformed or internally inconsistent case state before proof construction proceeds. |
| `structural_validity` | Structural safety check over required fields, shapes, and canonical state | Preventative | Prevents unsafe release decisions from being built on malformed identifiers, geometry, or profile state. |
| `AIClaimEnvelope` | Explicit upstream claim boundary | Detective | Makes the AI-supplied reconstruction, mask, volume, and declared provenance explicit rather than implicit. |
| `VerificationProfileDescriptor` | Approved configuration baseline | Preventative | Binds verification to a canonical profile body and prevents silent runtime policy drift. |

### Reconstruction Verification Controls

| Repo surface | Safety-language interpretation | Control type | Assurance contribution |
| --- | --- | --- | --- |
| `physical_consistency` | Hard proof layer from raw measurements to releasable output | Both | Detects mismatch and blocks release when reconstruction, mask, or volume cannot be justified from supplied data. |
| `VerifiedReconstructionTrace` | Traceable reconstruction assurance record | Detective | Provides a machine-readable proof trace from raw k-space to verified reconstruction, verified mask, and verified volume. |
| `VerificationTolerancePolicy` | Explicit mismatch acceptance envelope | Both | Pins down per-voxel, per-frequency, global relative L2, and volume tolerances so borderline cases are governed explicitly rather than heuristically. |
| `manifest.ai_reconstruction_match` | Image-parity evidence surface | Detective | Records whether the AI reconstruction matches the raw-derived reconstruction and which metrics passed, neared, or exceeded thresholds. |

### AI Output Constraints

| Repo surface | Safety-language interpretation | Control type | Assurance contribution |
| --- | --- | --- | --- |
| `MaskDerivationPolicy` | Approved deterministic segmentation constraint | Both | Constrains verified masks to one canonical algorithm identity plus resolved parameters bound to the active profile. |
| `manifest.mask_verification` | Segmentation parity evidence surface | Detective | Records exact AI-versus-verified mask comparison, including voxel deltas and canonical policy identity. |
| `manifest.tumor_volume_proof` | Quantitative measurement validation | Detective | Confirms whether the claimed tumour volume equals the verified mask volume on the measured grid. |
| `decision.success_modifiers.near_proof_threshold` | Near-miss escalation signal | Detective | Preserves evidence that proof passed close to a configured threshold rather than treating all passes as equally comfortable. |

### Aggregation and Temporal Controls

| Repo surface | Safety-language interpretation | Control type | Assurance contribution |
| --- | --- | --- | --- |
| `clinical_plausibility` | Optional higher-layer plausibility review | Both | Adds profile-scoped heuristic review without changing the underlying Level 2 proof model. |
| `TemporalConsistencyTrace` | Longitudinal comparison assurance record | Detective | Makes prior admissibility, chronology trust, geometry scope, and skipped metrics explicit so temporal findings are reviewable. |
| `LongitudinalRegistrationTrace` | Registration quality assurance record | Detective | Separates the fact of using registered comparison from the quality and admissibility of that registration. |
| `StudyTemporalAggregationTrace` | Study-level temporal coverage summary | Detective | Shows which required series have usable temporal context and how that affects study-level temporal posture. |
| `multi_series_aggregation` | Study-level release gate over required series | Both | Prevents study release when required series are missing, structurally incompatible, not release-eligible, or not corroborated. |

### Release Gating Controls

| Repo surface | Safety-language interpretation | Control type | Assurance contribution |
| --- | --- | --- | --- |
| `policy_evaluation` | Configured release-policy gate | Preventative | Converts allowed advisory behavior, blocking heuristics, and non-final profile semantics into explicit release posture. |
| `release_control` | Final downstream release gate | Preventative | Ensures only Guardian-verified payloads move through the Guardian path to downstream consumers. |
| `decision.decision_class` | Operational risk-posture label | Detective | Gives downstream dashboards and reviewers a stable semantic class for the case outcome. |

## Assurance Matrix

| Failure Mode | Adversarial Scenario (if applicable) | Clinical hazard | Potential clinical impact | System control(s) | Control type | System response (decision class) | Residual risk | Evidence (tests / traces) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False positive lesion propagated to downstream handoff | Fake tumour injection | False positive imaging finding | Unnecessary diagnostic escalation; possible invasive follow-on; increased workload | `MaskDerivationPolicy`; `manifest.mask_verification.comparison`; `VerifiedReconstructionTrace`; `release_control` | Both | `blocked_physical` | Low in the Guardian path; residual risk remains if upstream raw data or acquisition is wrong but still internally consistent | `test_rejects_ai_mask_with_fake_tumour_voxel`; `manifest.mask_verification`; `VerifiedReconstructionTrace` |
| Unsupported image content accepted as measured reality | Invented pixels in AI reconstruction | False or misleading image appearance | False positive or false negative interpretation; inappropriate review burden | `VerificationTolerancePolicy`; `manifest.ai_reconstruction_match`; `VerifiedReconstructionTrace`; `release_control` | Both | `blocked_physical` | Low in the Guardian path; residual risk depends on correctness of supplied raw data and tolerance calibration | `test_rejects_ai_reconstruction_with_invented_pixels`; `manifest.ai_reconstruction_match`; `VerifiedReconstructionTrace` |
| Incorrect quantitative lesion burden reported | Incorrect tumour-volume claim | False quantitative finding | Incorrect trend interpretation; inappropriate follow-up or treatment framing | `manifest.tumor_volume_proof`; `manifest.mask_verification`; `VerifiedReconstructionTrace`; `release_control` | Both | `blocked_physical` | Residual risk remains if upstream voxel spacing or acquisition geometry is wrong but structurally valid | `test_rejects_incorrect_tumor_volume_claim`; `manifest.tumor_volume_proof`; `VerifiedReconstructionTrace` |
| Subtle numeric manipulation not obvious on per-voxel inspection | Distributed low-amplitude attack | Hidden image alteration close to tolerance boundary | Missed manipulation or overconfidence in borderline data | `VerificationTolerancePolicy`; `manifest.ai_reconstruction_match`; `decision.success_modifiers.near_proof_threshold`; `release_control` | Both | `blocked_physical` when tolerance is exceeded | Residual risk depends on scanner-specific tolerance suitability and profile calibration | `test_rejects_distributed_low_amplitude_attack_by_global_l2_bound`; `manifest.ai_reconstruction_match.metrics`; `decision.success_modifiers.near_proof_threshold` |
| Unsafe or malformed profile/input state accepted into the proof path | Not attack-specific | Misconfigured verification process | False assurance from a structurally invalid case or profile | `intake_validation`; `structural_validity`; `VerificationProfileDescriptor`; `release_control` | Preventative | `blocked_structural` | Residual risk remains if upstream acquisition is semantically wrong but still passes structural checks | `test_invalid_embedded_profile_descriptor_returns_structural_block_report`; `decision.decision_class`; `decision_summary.block_category` |
| Non-final output released as if it were final | Non-final profile selection | Premature downstream reliance on non-final output | Inappropriate clinical action on triage-only output | `policy_evaluation`; `VerificationProfileDescriptor.release_semantics`; `release_control` | Preventative | `blocked_policy` | Residual risk sits in external workflows outside the Guardian path, not in the Guardian release decision | `test_fast_triage_profile_is_non_final_and_rejected_for_release`; `decision_summary.block_category = profile_policy_block`; `decision.decision_class` |
| Configured blocking heuristic ignored after proof passes | Blocking heuristic failure in `strict_research_mode` | Clinically concerning verified pattern allowed through despite configured block | Unsafe automation or undertriaged output under a stricter profile | `clinical_plausibility`; `policy_evaluation`; `decision.decision_class`; `release_control` | Both | `blocked_policy` | Residual risk depends on heuristic specificity and the narrowness of the configured blocking rules | `test_strict_research_mode_blocks_on_heuristic_failure`; `manifest.clinical_plausibility.blocking_failures`; `decision.decision_class` |
| Concerning longitudinal growth not acted on when temporal policy is enabled | Temporal growth warning/block in `strict_research_mode` | Under-recognized longitudinal deterioration | Delayed escalation or false reassurance in longitudinal review | `PriorVerifiedContext`; `TemporalConsistencyTrace`; `TemporalTrajectoryTrace`; `clinical_plausibility`; `release_control` | Both | `blocked_policy` in `strict_research_mode`; otherwise advisory-only | Residual risk depends on prior admissibility, timestamp trust, geometry quality, and profile policy bands | `test_strict_research_profile_can_block_on_temporal_growth`; `TemporalConsistencyTrace`; `TemporalTrajectoryTrace` |
| Unsuitable prior context used to drive temporal interpretation | Not attack-specific | Misleading longitudinal comparison | False trend signal or spurious reassurance | `PriorVerifiedContext` admissibility; `TemporalConsistencyTrace.prior_admissibility`; `TemporalConsistencyTrace.comparison_status`; `TemporalConsistencyTrace.comparison_scope` | Detective | No automatic block by itself; final disposition remains driven by Level 2 and profile logic | Residual risk is absence of reliable temporal context rather than release of an unproven current case | `test_temporal_missing_prior_bundle_identity_is_inadmissible_but_non_blocking`; `TemporalConsistencyTrace` |
| Cross-series study release built from incompatible required-series geometry | Not attack-specific | False corroboration across sequences | Misleading study-level confidence or aggregation | `multi_series_aggregation`; `CrossSeriesVerificationTrace.geometry_verification`; `required_series_ids`; `release_control` | Both | `blocked_structural` | Residual risk remains if acquisition quality is poor but geometry still matches structurally | `test_multi_series_geometry_mismatch_blocks_merged_release`; `CrossSeriesVerificationTrace`; `decision_summary.block_category` |
| Study released even though a required supporting series failed its own proof | Not attack-specific | Unsupported study-level release | False confidence in a corroborated study result | `multi_series_aggregation`; per-series `decision.decision_class`; `required_series_ids`; `release_control` | Both | Usually inherited `blocked_physical` from the failing required series | Residual risk depends on the correctness of required-series selection policy | `test_multi_series_required_supporting_series_proof_failure_blocks_release`; per-series `decision.decision_class`; study `decision.decision_class` |
| Positive lesion treated as corroborated without required supporting overlap | Not attack-specific | False corroborative confidence across sequences | Escalation based on a single unsupported positive signal | `CrossSeriesVerificationTrace.lesion_corroboration`; `multi_series_aggregation`; `release_control` | Both | `blocked_physical` | Residual risk remains if all required verified masks are empty, because absence is intentionally not separately corroborated in this milestone | `test_multi_series_primary_positive_without_any_supporting_overlap_blocks_release`; `analysis.lesion_corroboration`; `CrossSeriesVerificationTrace` |

## Decision Class to Risk Posture

| `decision.decision_class` | Plain-language risk posture | Downstream meaning |
| --- | --- | --- |
| `verified_clean` | Verified and released without advisory modifiers | Guardian-verified payload can move downstream through the Guardian path. |
| `verified_boundary_hugging` | Verified but close to at least one configured proof boundary | Release is allowed, but reviewers should treat the case as close to a failure threshold. |
| `verified_with_advisory` | Verified and released with non-blocking advisory findings | Release is allowed, but the case carries review-relevant warnings or temporal/plausibility findings. |
| `blocked_structural` | Blocked before trustworthy proof construction | No verified downstream payload is released because the case or configuration is structurally unsafe. |
| `blocked_physical` | Blocked because the AI claim failed measured-data proof | No verified downstream payload is released because reconstruction, mask, volume, or corroboration failed the proof model. |
| `blocked_policy` | Blocked by configured release policy after or alongside proof evaluation | No final downstream payload is released because the profile or heuristic policy intentionally denies release. |

## Worked Example: Fake Tumour Injection

- Failure mode: false positive lesion propagation into the downstream handoff.
- Adversarial scenario: the AI tumour mask contains unsupported positive voxels.
- Clinical hazard: a false positive clinical finding is presented as if it were supported by measured data.
- Potential clinical impact: unnecessary escalation, potentially invasive follow-on work, and increased clinical workload.
- System controls: `MaskDerivationPolicy`, `manifest.mask_verification.comparison`, `VerifiedReconstructionTrace`, and `release_control`.
- Control type: Both. The system detects the mismatch and prevents downstream release through the Guardian path.
- System response: the case is blocked with `decision.decision_class = blocked_physical`.
- Concrete trace and manifest references: `manifest.mask_verification.comparison.verified = false`, a non-zero voxel delta in `manifest.mask_verification.comparison`, and the verified-mask linkage carried in `VerifiedReconstructionTrace`.
- Executable evidence reference: `test_rejects_ai_mask_with_fake_tumour_voxel`.

## Longitudinal Controls

- `TemporalConsistencyTrace` is additive to the hard Level 2 proof gate. It does not replace reconstruction, mask, or volume proof.
- Prior use is explicit and bounded: `PriorVerifiedContext` admissibility, chronology provenance, geometry comparability, and skipped metrics are all surfaced rather than inferred.
- Registered comparison is explicit rather than hidden. `LongitudinalRegistrationTrace` records whether registration was used and whether spatial use was allowed.
- Temporal findings are profile-controlled. In the shipped profiles, `strict_research_mode` can promote selected temporal findings into blocking behavior, while other profiles keep temporal findings advisory.
- `StudyTemporalAggregationTrace` extends that logic to `multi_series_v2` by showing which required series have usable temporal context and which do not.

Representative evidence:

- `test_temporal_affine_geometry_mismatch_uses_registered_full_comparison`
- `test_temporal_missing_prior_bundle_identity_is_inadmissible_but_non_blocking`
- `test_strict_research_profile_can_block_on_temporal_growth`
- `test_multi_series_temporal_aggregation_tracks_required_series_coverage`

## Multi-Series V2 Controls

- `required_series_ids` is the only authoritative gating set for merged study release.
- Required-series geometry mismatch is treated as a structural block rather than an advisory inconsistency.
- Required-series proof failure is inherited into the study-level decision so the study cannot appear stronger than its gating series.
- Positive-lesion corroboration is explicit: when the primary verified mask is positive, at least one required supporting series must show overlap under the configured corroboration rule.
- Auxiliary series remain visible and reviewable, but they do not become hidden gates by accident.

Representative evidence:

- `test_multi_series_missing_required_series_is_reported_explicitly`
- `test_multi_series_required_supporting_series_proof_failure_blocks_release`
- `test_multi_series_primary_positive_without_any_supporting_overlap_blocks_release`

## Limitations and Residual Risk

- The DICOM objects are mocked and do not claim standards certification.
- The baseline release profiles still use deterministic threshold masks; the morphology-constrained, component-constrained, and slice-consistency-constrained profiles are deterministic showcases rather than production tumour segmentation models.
- The prototype proves integrity of the handoff path; it does not by itself prove diagnostic correctness.
- The prototype does not validate correctness of upstream acquisition or scanner calibration.
- Tolerance policies are explicit but still static and not yet calibrated per scanner family, sequence family, and reconstruction mode.
- Evidence integrity is hash-based in-repo; it is not a substitute for institutional signing, production key management, or external timestamp authority.
- External fallback workflows, local hospital policy, identity management, privacy controls, and operational monitoring outside the Guardian path remain outside the scope of this prototype.
