# ADR-0010: Canonical AI Claim Provenance Split

Status: Accepted
Date: 2026-03-27

## Context

The original `AIClaimEnvelope` identified the coarse upstream AI family through
`ai_model_id`, but it did not let auditors distinguish which exact declared
model stack produced a claim.

That is enough to prove reconstruction, mask, and volume parity against raw
data, but it is weak for:

- audit review
- incident investigation
- model-family comparison
- drift analysis across deployments

At the same time, the repository already draws a hard line between
proof-significant canonical identity and operational runtime metadata. We do not
want runtime/device observations to drift into proof hashes or for readers to
misinterpret canonical declared provenance as if Guardian had independently
validated the model stack itself.

## Decision

Extend the canonical claim surface so a claim can explicitly declare the exact
model stack identity that produced it.

- `AIClaimEnvelope` gains optional `ai_stack_identity`.
- `ai_stack_identity` is optional as a whole, but mandatory in full when
  present.
- `ai_model_id` remains the coarse model-family identifier. It is not replaced
  or reinterpreted by `ai_stack_identity`.
- `ai_stack_identity` is canonical claim provenance. Guardian canonicalizes and
  binds the producer declaration into the claim identity, but does not
  independently prove that the declared model stack is true.
- The claim contract remains byte-for-byte identical when `ai_stack_identity`
  is absent. `AIClaimEnvelope.interface_version` therefore remains unchanged for
  this additive change.

Required `ai_stack_identity` fields:

- `model_version_id`
- `weights_sha256`
- `preprocessing_pipeline_sha256`
- `postprocessing_pipeline_sha256`
- `input_normalization`
- `sequence_model_compatibility`

Field semantics:

- `weights_sha256` is a declared SHA-256 digest over immutable model-weights
  artifact bytes. It is not an opaque deployment label.
- `preprocessing_pipeline_sha256` and `postprocessing_pipeline_sha256` are
  declared SHA-256 digests over immutable pipeline artifact representations.
- `sequence_model_compatibility` is mandatory inside `ai_stack_identity`.
  - `compatibility_type = exact_sequence_id` requires exact equality with the
    case `sequence_id`.
  - `compatibility_type = sequence_family` requires one of the registry-backed
    `VerificationProfileDescriptor.sequence_family` values and must match the
    active case/profile context.

Canonical value-domain rules:

- Declared `ai_stack_identity` string fields must be non-empty and must not
  carry leading or trailing whitespace.
- Empty strings are invalid for identifier and digest fields.
- SHA-256 fields must be lowercase hex with no leading/trailing whitespace and
  no `0x` prefix.
- `input_normalization.parameters` must be `JsonLike`:
  - scalars: `str`, `bool`, `int`, finite `float`
  - composites: `list[JsonLike]`, `dict[str, JsonLike]`
  - rejected: `null`, non-finite floats, non-string keys, tuples, sets, bytes,
    numpy scalars, datetimes, and custom objects
- No implicit type coercion is allowed between `bool`, `int`, `float`, and
  `str`.
- Floating-point values are serialized only through the repository's canonical
  JSON encoder. Producers must not rely on language-specific float string
  representations outside that encoder.
- Input dictionaries may arrive in any order, but canonical output is defined
  only by the repository serializer. Upstream ordering must not affect
  canonicalized claim bytes.

Operational metadata remains separate:

- optional `ai_runtime_context` may appear only on mocked interoperability
  envelopes
- it includes `inference_runtime_version` and `inference_device_class`
- it is excluded from core Guardian state, manifests, evidence, report hashes,
  bundle hashes, and replay material

## Consequences

- Claim hashes, manifests, reports, released state hashes, and replay hashes can
  now distinguish materially different declared AI stacks.
- Auditors gain direct visibility into canonical declared provenance through the
  claim contract and single-series proof summaries.
- Runtime/device observations remain operational-only and cannot silently drift
  into proof identity.
- Invalid declared provenance is treated as Level 1 structural invalidity. It is
  not silently dropped, defaulted, coerced, or sanitized.
- Multi-series handling stays per-series. No study-level AI provenance aggregate
  is introduced, so mixed per-series provenance remains explicit rather than
  being collapsed into an ambiguous study-level summary.
