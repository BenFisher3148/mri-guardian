# Development

This document is the developer-facing operating guide for the MRI Guardian Layer repository.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`../mri_guardian/`](../mri_guardian/) | Core package |
| [`../tests/`](../tests/) | Unit and regression tests |
| [`../demo_artifacts/`](../demo_artifacts/) | Generated demo outputs |
| [`../docs/`](../docs/) | Documentation suite |
| [`../docs/adr/`](../docs/adr/) | Architecture Decision Records |
| [`../README.md`](../README.md) | Project overview and usage |
| [`../AGENTS.md`](../AGENTS.md) | Repository operating rules for human and agent contributors |

## Local Workflow

Run the deterministic demo:

```powershell
.\.venv\Scripts\python.exe -m mri_guardian.demo --output-dir demo_artifacts --tamper-mode
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run ICSD lint:

```powershell
.\.venv\Scripts\icsd-dpa lint mri_guardian --profile policy-v0.1 --format json
.\.venv\Scripts\icsd-dpa lint tests --profile policy-v0.1 --format json
```

Verify emitted evidence:

```powershell
.\.venv\Scripts\icsd-dpa verify demo_artifacts\evidence.json --format json
```

Verify the portable bundle:

```powershell
@'
from mri_guardian import verify_bundle
print(verify_bundle("demo_artifacts/mri-case-0001.mri-guard-bundle"))
'@ | .\.venv\Scripts\python.exe -
```

## Change Discipline

When modifying code or schemas:

- preserve deterministic outputs and stable canonical ordering
- update tests alongside behavioural changes
- update documentation when interfaces, guarantees, or workflows change
- avoid unrelated cleanup
- keep commits narrow and attributable

The repository-level process is defined in [`../AGENTS.md`](../AGENTS.md).

## Documentation Maintenance

Update these documents when the corresponding surface changes:

| Document | Update when |
| --- | --- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | modules, release flow, or verification tiers change |
| [`SECURITY.md`](SECURITY.md) | threat model, fail-closed behaviour, or trust boundary changes |
| [`PERFORMANCE.md`](PERFORMANCE.md) | benchmark methodology, bounds, or streaming strategy changes |
| [`INTEROPERABILITY.md`](INTEROPERABILITY.md) | DICOM/PACS narrative or hospital placement changes |
| [`DATA_DICT.md`](DATA_DICT.md) | state shape, contract shape, or artifact keys change |
| [`../CITATION.cff`](../CITATION.cff) | project title, version, or attribution changes |

## ADR Process

Architecture decisions belong in [`adr/`](adr/).

Rule of thumb:

- add a new ADR for any decision that changes architecture, invariants, security posture, wire format, interoperability model, or release semantics
- update an existing ADR only for clarifications that do not materially change the decision

Suggested template:

```text
# ADR-XXXX: Title
Status: Accepted
Date: YYYY-MM-DD

## Context
## Decision
## Consequences
```

## High-Sensitivity Areas

Take extra care when touching:

- canonical hashes and serialised outputs
- tolerance policy and verification semantics
- released state shape
- bundle member ordering and replayability
- integration envelopes and PACS handoff behaviour

## Related Material

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`DATA_DICT.md`](DATA_DICT.md)
- [`adr/README.md`](adr/README.md)
