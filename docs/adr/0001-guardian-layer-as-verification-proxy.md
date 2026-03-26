# ADR-0001: Guardian Layer As Verification Proxy

Status: Accepted
Date: 2026-03-26

## Context

Hospital stakeholders need to understand where this system plugs in. Replacing PACS or the radiologist workstation would create unnecessary operational and regulatory scope.

## Decision

Position the MRI Guardian Layer as a verification proxy between the reconstruction AI and downstream PACS/workstation import.

## Consequences

- The system is framed as a non-invasive gateway rather than a new clinical platform.
- Integration narratives focus on handoff verification, not clinical system replacement.
- Mocked DICOM and PACS artifacts preserve upstream and downstream context while keeping the Guardian in the middle of the path.
