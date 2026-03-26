# Security

This document describes the current security posture of the MRI Guardian Layer prototype.

## Security Objective

The primary security goal is integrity of the clinical handoff. The Guardian Layer must detect when the AI reconstruction, segmentation, or volume claim diverges from what can be mathematically derived from the supplied raw MRI data.

## Security Boundary

The trust boundary sits at the Guardian Layer:

- Upstream systems may provide incorrect, manipulated, or adversarial AI outputs.
- The Guardian Layer treats AI outputs as claims rather than trusted facts.
- Downstream PACS/workstation systems should receive only Guardian-verified payloads from the Guardian path.

## Threat Model

| Threat | Current control | Residual gap |
| --- | --- | --- |
| Invented pixels in the AI reconstruction | Deterministic reconstruction from raw k-space plus explicit parity checks | No production-grade DICOM transport hardening is implemented yet. |
| Fake tumour mask regions | Deterministic profile-scoped mask derivation, fixed approved algorithm catalogue, and exact voxel match policy | Baseline profiles remain intentionally simple and not clinically rich. |
| Boundary-hugging numeric attacks | Explicit `VerificationTolerancePolicy` with per-voxel, per-frequency, and global relative L2 checks | Tolerances are profile-scoped but still static rather than scanner-calibrated. |
| Bundle tampering | Canonical hashes, signatures catalog, transparency anchoring, deterministic replay | Signatures are integrity hashes, not asymmetric institutional signatures. |
| Silent PACS bypass from the Guardian path | Fail-closed Guardian handoff behaviour | External fallback workflows are not governed by this prototype. |
| Replay drift | Deterministic replay using bundled input state, transparency prefix, and canonical profile digest | No external time-stamping authority is used. |

## Security Controls Present Today

- Deterministic reconstruction from raw k-space.
- Deterministic verified mask derivation.
- Fixed approved mask-algorithm catalogue with canonical parameter resolution before execution.
- Explicit connected-component policy support with proof-significant `faces` connectivity and deterministic queue-based traversal.
- Canonical runtime profile descriptors with embedded digest identity.
- Explicit claim-to-proof interfaces with stable hashes.
- Fail-closed Guardian release gate for blocking proof failures.
- Distinct `proof_failure`, `heuristic_block`, and `profile_policy_block` outcomes in reports and bundles.
- Transparency log anchoring for evidence receipts.
- Portable bundle integrity catalog.
- Deterministic replay that verifies released state, evidence payload, and bundle members.
- Attack-aware tolerance policy with guard-band reporting.
- Exclusion of non-canonical mask-engine runtime metadata from proof hashes and replay surfaces.

## Fail-Closed Behaviour

The Guardian Layer is fail-closed for the verified handoff:

- If Level 1 or Level 2 verification fails, no verified reconstruction, verified mask, manifest, or evidence series are released from the Guardian path.
- If a runtime profile is non-final, proof may still pass, but no final released state or portable bundle is produced from the Guardian path.
- If the Guardian service is unavailable, the Guardian path does not automatically pass AI outputs through.
- Any operational fail-open workflow would need to be an explicit external process outside this repository's Guardian model.

## Sensitive Data Assumptions

The demo uses pseudonymised study identifiers and does not model a full hospital privacy posture. Production deployment would still need:

- DICOM transport security
- authentication and AE-title allow-listing
- encryption at rest
- role-based access controls
- retention and audit policies
- PHI handling procedures

## Recommended Production Hardening

- Publish a formal DICOM Conformance Statement.
- Use mutually authenticated DICOM transport or a protected network segment.
- Sign evidence and bundles with institutional keys rather than hash-only catalogs.
- Separate operator privileges for AI operation, Guardian verification, and PACS release.
- Push transparency and verification events into an external audit or SIEM system.
- Calibrate tolerance policies per scanner family, sequence family, and reconstruction mode.

## Known Limits

- The DICOM objects are mocked and do not claim standards certification.
- The baseline release profiles still use deterministic threshold masks; the morphology-constrained and component-constrained profiles are deterministic showcases rather than production tumour segmentation models.
- The prototype proves integrity of the handoff path; it does not by itself prove diagnostic correctness.

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`INTEROPERABILITY.md`](INTEROPERABILITY.md)
- [`adr/0002-fail-closed-guardian-handoff.md`](adr/0002-fail-closed-guardian-handoff.md)
