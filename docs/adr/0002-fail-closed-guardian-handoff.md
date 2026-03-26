# ADR-0002: Fail-Closed Guardian Handoff

Status: Accepted
Date: 2026-03-26

## Context

The value of the Guardian Layer depends on the integrity of its downstream claim. A fail-open design would allow unverified AI output to move through the same path that is supposed to provide proof.

## Decision

Make the Guardian-verified handoff fail-closed:

- blocking verification failure prevents verified release
- Guardian unavailability does not auto-pass AI outputs through the Guardian path
- bypass requires an explicit external workflow outside the repository's proof model

## Consequences

- Downstream consumers can interpret Guardian-released payloads as integrity-gated outputs.
- Operational teams must define a separate contingency process if they require emergency bypass.
- Documentation and integration artifacts must make this behaviour explicit.
