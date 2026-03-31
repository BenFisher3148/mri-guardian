# ADR-0018: Proper DICOM Object Modelling
Status: Accepted
Date: 2026-03-31

## Context

The repository originally used a flat mocked DICOM/PACS narrative built around an `objects` list and loosely typed role labels. That was acceptable for an early demo, but it was not a credible long-term interoperability story for reviewers who expect study, series, instance, provenance, and withheld-output boundaries to be explicit.

At the same time, the Guardian proof system must not drift into the interoperability layer. The DICOM-shaped surfaces are reviewer-facing transport and audit artefacts, not the source of truth for proof replay or decision semantics.

## Decision

The repository now uses a deterministic study-centred DICOM-shaped model in [`mri_guardian/dicom_model.py`](../../mri_guardian/dicom_model.py) and [`mri_guardian/integration.py`](../../mri_guardian/integration.py).

Core decisions:

- adopt integration schema version `0.2`
- retire the old flat mock integration schemas in favour of a study-centred graph
- freeze the object-kind vocabulary to `study`, `series`, `instance`, `document`, `sidecar`, and `acquisition_provenance`
- freeze the relationship vocabulary to `contains_series`, `contains_instance`, `derived_from`, `references`, `same_frame_of_reference`, `blocked_output_for`, and `sidecar_for`
- require all Guardian-specific fields inside standard-like objects to live only in `extensions.guardian_v1`
- reject unknown Guardian fields outside that namespace and reject unknown extension keys inside it
- generate deterministic mock DICOM UIDs from a canonical seed that includes schema version, UID namespace version, object kind, and parent identity where relevant
- model raw-input lineage as `acquisition_provenance`, not as a pretend DICOM instance
- model manifest and claim outputs as SR-like reviewer artefacts, not as a claim of standards-certified DICOM SR encoding
- model evidence linkage as a typed non-DICOM sidecar with fixed `sidecar_type = icsd_evidence_json`
- quarantine `debug.guardian_case_snapshot` as restore-only and non-interoperable
- require `restore_case_state_from_dicom_input(...)` to restore from the debug snapshot only and to fail hard when it is absent or malformed
- preserve proof contracts, decision logic, bundle replay, and proof-critical surfaces unchanged

`pydicom` is used only as a UID-validation and standards-vocabulary helper. The repository-owned JSON schemas remain authoritative.

## Consequences

Positive consequences:

- the mocked interoperability layer is now much closer to how reviewers expect DICOM-shaped studies, derived series, withheld outputs, and provenance links to look
- frame-of-reference handling is stricter and more honest because missing lineage is represented as missing lineage, not fabricated UIDs
- SR-like, SEG-like, and sidecar artefacts are easier to inspect and reason about
- the extension boundary is explicit and testable
- proof-state recovery remains isolated from the interoperability model

Tradeoffs:

- the integration payloads are larger and more structured than the old flat object list
- older flat mock integration schemas are retired rather than retained
- this is still not a claim of DICOM certification, file-format conformance, or network-service implementation

This decision improves reviewer credibility without changing Guardian proof semantics.
