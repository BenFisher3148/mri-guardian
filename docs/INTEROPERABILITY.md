# Interoperability

This document explains how the MRI Guardian Layer fits into hospital imaging infrastructure and how the repository now models mocked DICOM and PACS interaction.

## Interoperability Goal

The Guardian Layer should be understandable to hospital stakeholders as a verification proxy, not as a replacement clinical system. The interoperability model therefore focuses on three questions:

- what DICOM-shaped objects move through the boundary
- where the Guardian Layer sits in the network
- what happens when verification fails or the Guardian path is unavailable

## Placement in Hospital Infrastructure

```text
MRI scanner --> reconstruction AI --> MRI Guardian Layer --> PACS / VNA --> workstation
```

Expected operational role:

- upstream: MRI scanner and reconstruction AI
- midstream: Guardian Layer verification proxy AE on the imaging network
- downstream: PACS, VNA, and radiologist workstation

The Guardian Layer is intended to sit after the AI and before downstream import. That makes it a gateway or sidecar service rather than a replacement PACS or workstation.

## DICOM-Shaped Model

The repository now uses a deterministic study-centred graph in [`mri_guardian/dicom_model.py`](../mri_guardian/dicom_model.py) and [`mri_guardian/integration.py`](../mri_guardian/integration.py). The current mock schema version is `0.2`.

Ingress and egress payloads are shaped as:

- `study`
- `source_series`
- `released_series` on accepted egress
- `withheld_outputs` on blocked or partially withheld egress
- `relationships`
- `sidecars`
- optional `debug`

The older flat `objects` list is retired. The current model keeps study, series, instances, documents, sidecars, and lineage objects structurally distinct.

## Standard-Like Boundary

The mock payloads are reviewer-facing and standards-shaped, but they are not a claim of certified byte-level DICOM conformance.

Rules:

- every study, series, instance, document, sidecar, and acquisition-provenance object carries exactly one custom namespace: `extensions.guardian_v1`
- Guardian-specific fields are not allowed anywhere else inside standard-like objects
- unknown keys inside `extensions.guardian_v1` are rejected
- SR-like and SEG-like objects follow the same rule as image-like series and manifest-like outputs

This keeps the mocked interoperability surface honest and prevents Guardian-specific semantics from leaking through ad hoc fields.

## Object Kinds and Relationships

Closed object-kind vocabulary:

- `study`
- `series`
- `instance`
- `document`
- `sidecar`
- `acquisition_provenance`

Closed relationship vocabulary:

- `contains_series`
- `contains_instance`
- `derived_from`
- `references`
- `same_frame_of_reference`
- `blocked_output_for`
- `sidecar_for`

Directionality is explicit in the model:

- `derived_from`, `references`, `blocked_output_for`, and `sidecar_for` are directional
- `same_frame_of_reference` is symmetric
- source and target object kinds are validated for each relationship type

## UID and Frame-of-Reference Rules

Study, series, and instance identities use deterministic mock DICOM UIDs.

UID invariants:

- the seed includes schema version, UID namespace version, object kind, and canonical source identity
- child objects include parent identity in the seed where relevant
- identical logical seeds produce identical UIDs
- different object identity classes do not share UID space
- seeds do not depend on display labels or incidental caller ordering

Frame-of-reference handling is intentionally strict:

- a `frame_of_reference_uid` appears on a derived or source series only when there is explicit upstream lineage for it
- if upstream lineage is absent or non-DICOM-shaped, the model records availability in series lineage metadata instead of fabricating a pseudo UID
- mixed or missing frame-of-reference state is reported in integration traces through `frame_of_reference_summary`

## Source and Derived Objects

Ingress modelling uses three source series per logical payload series:

- MR-image-oriented reconstruction series
- SEG-like segmentation series
- SR-like AI claim series

Raw-input lineage is represented as `acquisition_provenance`, not as a pretend image or document instance. It records source lineage metadata only and must not be treated as a standards-shaped image/document instance.

Egress modelling uses:

- verified reconstruction series
- verified segmentation series
- Guardian verification manifest SR-like series
- optional withheld-output groups when downstream release is blocked or reduced

For `multi_series_v2`, one study-level manifest SR-like series is emitted alongside the ordered released per-series outputs.

## SR-Like, SEG-Like, and Sidecar Surfaces

SEG-like instances carry explicit segment metadata:

- `segment_number`
- `segment_label`
- `algorithm_type`
- `algorithm_name`
- referenced source instance UIDs

SR-like documents use fixed reviewer-facing `document_kind` values:

- `ai_volume_claim_sr_like`
- `guardian_verification_manifest_sr_like`
- `guardian_study_verification_manifest_sr_like`

These documents are structured review artefacts shaped for interoperability preparation. They are not a claim of exchangeable, standards-certified DICOM SR payloads.

Evidence linkage is modelled as a typed non-DICOM sidecar:

- `sidecar_type` is fixed to `icsd_evidence_json`
- the sidecar references the manifest document instance UID
- the sidecar may also record the Guardian bundle path and specific bundle member
- the sidecar must not become the sole carrier of proof-critical semantics already represented elsewhere

## Proof-State Boundary

The proof system and replay model do not move into the DICOM-shaped layer.

Rules:

- `debug.guardian_case_snapshot` exists only as a restore-only quarantine surface
- batch queue quarantine is an operational filesystem bucket and is distinct from the restore-only `debug.guardian_case_snapshot` proof-state surface
- `restore_case_state_from_dicom_input(...)` restores from `debug.guardian_case_snapshot` only
- standard-like DICOM-shaped objects are ignored for proof-state recovery
- if the debug snapshot is absent or malformed, restoration fails hard
- reviewer-facing objects, relationships, and sidecars do not reference the debug snapshot

This preserves the boundary between believable interoperability modelling and Guardian proof semantics.

## Integration Traces

`integration_trace.json` now summarizes the study-centred graph rather than the retired flat object list.

Key operational summaries include:

- released series and instance counts
- withheld output counts
- sidecar count
- `frame_of_reference_summary`
- `lineage_completeness_status`

For single-series cases with temporal context, temporal comparison summaries remain mirrored into the trace. For `multi_series_v2`, the trace also records study-level temporal disposition summaries.

## Fail-Closed vs Fail-Open

The Guardian Layer is fail-closed for the verified handoff.

Fail-closed means:

- if blocking verification fails, no verified reconstruction, segmentation, manifest, or evidence sidecar is forwarded from the Guardian path
- if the Guardian Layer is unavailable, the Guardian path does not auto-pass AI outputs through
- any emergency or business-continuity bypass must be an explicit external workflow, not an implicit fallback inside this repository

This design protects the meaning of "verified downstream payload."

## Productionisation Path

To move from mocked interoperability to hospital-grade integration, the next layer would typically include:

- formal DICOM Conformance Statement work
- AE-title registration and allow-listing
- protected network placement and transport security
- PACS/VNA import rules for verified derived series
- operator procedures for explicit fallback workflows
- audit integration for Guardian decisions and downstream releases

## Non-Goals of the Current Prototype

- full DICOM standards certification
- real DICOM file or network IO
- modality worklist, storage commitment, or IHE profile implementation
- enterprise identity, tenancy, or scheduling workflows
- replacement of PACS, RIS, or reporting systems

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`DATA_DICT.md`](DATA_DICT.md)
- [`SECURITY.md`](SECURITY.md)
- [`adr/0001-guardian-layer-as-verification-proxy.md`](adr/0001-guardian-layer-as-verification-proxy.md)
- [`adr/0002-fail-closed-guardian-handoff.md`](adr/0002-fail-closed-guardian-handoff.md)
- [`adr/0018-proper-dicom-object-modelling.md`](adr/0018-proper-dicom-object-modelling.md)
