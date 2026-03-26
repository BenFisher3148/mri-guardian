# Interoperability

This document explains how the MRI Guardian Layer fits into hospital imaging infrastructure and how the current repository models DICOM and PACS interaction.

## Interoperability Goal

The Guardian Layer should be understandable to hospital stakeholders as a verification proxy, not as a replacement clinical system. It must therefore answer three questions clearly:

- what DICOM-shaped objects move through the boundary
- where the Guardian Layer sits in the network
- what happens when verification fails or the Guardian path is unavailable

## Placement in Hospital Infrastructure

```text
MRI scanner --> reconstruction AI --> MRI Guardian Layer --> PACS / VNA --> workstation
```

Expected operational role:

- upstream: MRI scanner and raw MRI reconstruction AI
- midstream: Guardian Layer verification proxy AE on the imaging network
- downstream: PACS, VNA, and radiologist workstation

The Guardian Layer is intended to sit after the AI and before downstream import. That makes it a gateway or sidecar service rather than a replacement PACS or workstation.

## DICOM Compliance Story

The repository currently uses mocked DICOM narratives in [`mri_guardian/integration.py`](../mri_guardian/integration.py). The story is:

- ingress transport: `dicom-c-store`
- egress transport: `dicom-c-store`
- study and patient context are preserved across ingress and egress
- only Guardian-verified derived series are emitted from the Guardian path
- in `multi_series_v2`, the ingress envelope carries one study-shaped Guardian case with ordered payload series and one study-level cross-series policy
- in `multi_series_v2`, the egress package includes one study-level manifest/evidence summary plus only downstream-allowed released series; withheld auxiliary series are explicit in the study payload rather than silently disappearing

Current mocked object mapping:

| Direction | Role | Mock SOP class |
| --- | --- | --- |
| Ingress | raw k-space | `MR Raw Data Storage (mock)` |
| Ingress | AI reconstruction | `MR Image Storage (mock)` |
| Ingress | AI segmentation | `Segmentation Storage (mock)` |
| Ingress | AI volume claim | `Comprehensive SR Storage (mock)` |
| Egress | verified reconstruction | `MR Image Storage (mock)` |
| Egress | verified segmentation | `Segmentation Storage (mock)` |
| Egress | verification manifest | `Comprehensive SR Storage (mock)` |
| Egress | ICSD evidence | `Encapsulated JSON Storage (mock)` |

For `multi_series_v2`, the mocked egress submission keeps two levels distinct:

- `series_manifests` covers every payload series in proof-significant order, including failed auxiliaries
- `series_payloads` covers only the released downstream subset
- `withheld_series` records series omitted from downstream release together with their role and proof outcome

## Current Interoperability Artifacts

Generated demo artifacts:

- [`../demo_artifacts/dicom_input.json`](../demo_artifacts/dicom_input.json)
- [`../demo_artifacts/integration_narrative.json`](../demo_artifacts/integration_narrative.json)
- [`../demo_artifacts/pacs_submission.json`](../demo_artifacts/pacs_submission.json)
- [`../demo_artifacts/integration_trace.json`](../demo_artifacts/integration_trace.json)

These are narrative and schema examples, not a formal DICOM conformance package.

## Fail-Open vs Fail-Closed

The Guardian Layer is fail-closed for the verified handoff.

Fail-closed means:

- if blocking verification fails, no verified reconstruction, mask, manifest, or evidence series are forwarded from the Guardian path
- if the Guardian Layer is unavailable, the Guardian path does not auto-pass AI outputs through
- any emergency or business-continuity bypass must be an explicit external workflow, not an implicit fallback inside this repository

This design protects the integrity claim. A fail-open Guardian would undermine the meaning of "verified downstream payload."

## Productionisation Path

To move from mock interoperability to hospital-grade integration, the next layer would typically include:

- formal DICOM Conformance Statement
- AE-title registration and allow-listing
- protected network placement and transport security
- PACS/VNA import rules for verified derived series
- operational policy for external fallback workflows
- audit integration for Guardian decisions and downstream releases

## Non-Goals of the Current Prototype

- full DICOM standards certification
- modality worklist, storage commitment, or IHE profile implementation
- enterprise identity, tenancy, or scheduling workflows
- replacement of PACS, RIS, or reporting systems

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`SECURITY.md`](SECURITY.md)
- [`adr/0001-guardian-layer-as-verification-proxy.md`](adr/0001-guardian-layer-as-verification-proxy.md)
- [`adr/0002-fail-closed-guardian-handoff.md`](adr/0002-fail-closed-guardian-handoff.md)
