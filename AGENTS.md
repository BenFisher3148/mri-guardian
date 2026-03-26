# AGENTS.md — Repository Agent Operating Rules

This repository supports concurrent work by humans and agents. The goal is to keep changes correct, reviewable, attributable, build-safe, and aligned with the repository’s actual operating rules.

Agents working in this repository must behave like careful engineers in a live multi-contributor environment. The expectation is not merely to produce code that works, but to produce changes that are minimal, maintainable, and sensitive to invariants, downstream effects, and real-world usage.

---

## 1. Core expectations

1. **Only change what you touched.**
   - Do not perform unrelated cleanup.
   - Do not reformat, reorder, rename, or refactor adjacent code unless explicitly required by the task.
   - Ignore unrelated uncommitted changes made by others.

2. **Read local instructions before editing.**
   - Read `WORKSTREAMS.md` before beginning meaningful work if it exists.
   - Read any applicable `AGENTS.md` files in the repository tree.
   - The most specific relevant `AGENTS.md` applies to the files being changed.
   - Broader repository rules still apply unless a more specific local rule explicitly overrides them.

3. **Reserve files before editing when required.**
   - If this repository uses file reservation in `WORKSTREAMS.md`, reserve files before modifying them.
   - Keep reservations minimal, accurate, and relevant to the actual scope of work.
   - Do not reserve files speculatively.

4. **Every change must be attributable.**
   - Each completed change must be traceable to a workstream or clearly scoped task.
   - Commits must explain what changed and why.
   - If the repository uses issue references, labels, tags, or workstream IDs, they must be included in the commit message exactly as required.

5. **Keep the repository coherent.**
   - The repository should remain in a sensible state after your change.
   - If you change behaviour, update tests and documentation where appropriate.
   - Do not leave intentionally touched files uncommitted at the end of a completed run unless explicitly instructed not to commit.

---

## 2. Instruction hierarchy

Apply instructions in this order:

1. System instructions
2. Developer instructions
3. User instructions
4. This `AGENTS.md`
5. More specific project-level or subproject-level `AGENTS.md` files

Rules:
- A more specific local `AGENTS.md` overrides this file where there is direct conflict.
- Do not discard broader instructions simply because a narrower file exists.
- Merge instruction sets and apply the local override only where an actual conflict exists.

---

## 3. AGENTS.md discovery rules

Before making meaningful edits, search for applicable instruction files.

At minimum, inspect:
- repository root
- current working directory
- parent directories of the files you plan to touch
- major subprojects, crates, packages, apps, or modules in monorepos

Guidance:
- Treat the nearest relevant `AGENTS.md` as the strongest local instruction source.
- If multiple instruction files apply, follow the most specific relevant one for the touched files while still respecting broader repository expectations.
- Do not ignore local conventions for:
  - commit schema
  - workstream handling
  - validation commands
  - labels and tags
  - changelog policy
  - documentation requirements
  - CI/workflow layout

---

## 4. Workstream discipline

If the repository uses `WORKSTREAMS.md` or a similar coordination ledger:

1. Read it before editing.
2. Identify the relevant pending or active workstream when requested.
3. Reserve files before editing if repository rules require reservation.
4. Keep reservations minimal and accurate.
5. Update workstream state if required by repository process.

### Workstream expansion rules
Agents may add or extend workstream entries when appropriate.

This is allowed when:
- the requested task is too large to complete safely in one sitting
- the task reveals directly relevant follow-on work
- the current implementation naturally exposes supporting work that should be tracked
- architectural or validation gaps become apparent during execution

Requirements:
- Any added workstream must be clearly scoped.
- New workstreams must be written in a reviewable, actionable format.
- Do not silently leave obvious follow-on work untracked if it materially affects future correctness, design completeness, or maintenance.

---

## 5. Editing behaviour

- Only modify files necessary for the task.
- Preserve surrounding style, structure, and architecture unless broader change is explicitly authorised.
- Avoid unnecessary formatting churn.
- Avoid drive-by refactors.
- Be cautious when touching:
  - public APIs
  - exported symbols
  - schemas
  - wire formats
  - persistence formats
  - canonical encodings
  - documented behaviour
  - automation/configuration paths

If changing behaviour:
- update tests where appropriate
- update docs where appropriate
- update changelog or migration notes where appropriate

If touching tooling or automation:
- respect the repository’s actual platform and structure
- inspect first rather than assuming defaults

---

## 6. Invariant sensitivity

Treat documented and strongly implied invariants as protected system contracts.

### General rules
- Do not introduce behaviour that violates repository invariants unless the task explicitly requires changing them.
- Consider both structural invariants and behavioural invariants.
- Preserve established contracts unless the task clearly authorises revising them.

### Examples of invariants to protect
This includes, where relevant:
- instruction semantics
- flag behaviour
- register and stack assumptions
- wire formats
- canonical encodings
- ordering guarantees
- determinism and replayability expectations
- persistence/storage formats
- validation and refusal rules
- public API expectations
- stable output shape
- compatibility assumptions relied on by tests, docs, or downstream consumers

### When invariants are unclear
If an invariant is unclear, undocumented, or only implied:
- inspect surrounding code
- inspect tests
- inspect comments and naming
- infer intent from established patterns
- prefer minimal change
- document ambiguity when changing nearby behaviour

### High-sensitivity changes
When modifying semantics, encodings, validation, or compatibility-sensitive logic:
- preserve deterministic behaviour unless change is explicitly required
- preserve existing valid inputs unless compatibility change is intentional
- reject or handle invalid inputs consistently with repository norms
- avoid silent contract drift

### If an invariant must change
If the task requires changing an invariant:
- make the change explicit
- update tests accordingly
- update documentation accordingly
- record compatibility, behavioural, or migration implications where appropriate
- reflect the invariant change clearly in workstream notes and commit messaging

---

## 7. Gitea awareness

This repository may use Gitea rather than GitHub.

Therefore:
- do not assume `.github/`
- do not assume GitHub Actions
- do not assume GitHub issue templates or GitHub-specific workflow conventions

Instead:
- inspect the actual repository layout
- prefer repo-native conventions already present
- use `.gitea/` when the repository is Gitea-native and workflow changes are required
- respect existing `make`, `docker`, `scripts`, or other project-native automation

Do not introduce GitHub-specific files into a Gitea-native repository unless explicitly instructed.

---

## 8. Research and real-world sensitivity

Think beyond the happy path.

Where relevant, consider:
- concurrency and race conditions
- partial failure modes
- rollback safety
- migration impact
- determinism
- idempotency
- platform variance
- encoding and line-ending issues
- time and timezone behaviour
- filesystem quirks
- observability and operator diagnostics
- secrets exposure
- documentation drift
- backwards compatibility
- hostile or malformed input
- performance cliffs
- CI/runtime differences
- containerisation effects
- air-gapped or enterprise deployment environments
- compliance, audit, or regulated-domain implications

Do not introduce speculative complexity without reason.
Do not stop at toy-path correctness if the repository clearly serves real-world usage.

---

## 9. Comments and code documentation

Write comments intentionally.

Do not add comments that merely restate the code.

Use comments when they explain:
- why a choice was made
- an invariant that must remain true
- a non-obvious edge case
- compatibility constraints
- security, audit, or performance rationale
- a temporary workaround and its boundaries

Preserve useful existing comments.
Improve stale or misleading comments when touching nearby logic.
Prefer doc comments or structured developer notes where repository style expects them.

---

## 10. Validation and quality gates

Before finalising work, run the relevant validation steps required by:
- applicable `AGENTS.md` files
- repository documentation
- build scripts
- existing project conventions

Examples may include:
- formatters
- linters
- unit tests
- integration tests
- build checks
- static analysis
- conformance checks

Rules:
- Prefer the narrowest sufficient validation first, then broader validation where appropriate.
- Fix failures caused by your changes.
- Do not ignore failing checks unless explicitly instructed.
- If validation cannot be run, state exactly why and note the risk clearly.

---

## 11. Documentation discipline

When appropriate, update:
- `WORKSTREAMS.md`
- `README.md`
- `CHANGELOG.md`
- relevant files under `docs/`
- migration notes
- operator notes
- config samples
- examples or tests that act as executable documentation

Update docs when:
- behaviour changes
- usage changes
- interfaces change
- guarantees change
- workflows change
- expected outputs change

Do not add performative documentation edits when nothing maintainer-relevant or user-relevant changed.

---

## 12. Commit discipline

Agents may commit directly to the active working branch when repository policy allows it.

### Commit requirements
- Commit all intentionally touched files before ending the run, unless explicitly instructed not to commit.
- Do not leave partially completed tracked edits uncommitted if the task required a finished change.
- Keep commits narrow, meaningful, and reviewable.
- Avoid mixing unrelated concerns in a single commit.
- Never commit unrelated local changes you did not make.

### Commit schema
First inspect applicable `AGENTS.md` files for required commit format.

If a project-level or subproject-level `AGENTS.md` defines a required schema, use it exactly.

If no project-specific schema exists, use this default:

**Subject**
- `<workstream-or-scope>: <imperative summary>`

**Recommended body fields**
- `Issue: ...`
- `Labels: ...`
- `Files: ...`
- `Notes: ...`

If the repository requires any of the following, they are mandatory:
- workstream ID
- issue reference
- label intent
- file list
- notes
- tags
- validation status

### Commit content expectations
- The subject must describe the actual change in imperative form.
- The body must explain what changed and why.
- Mention validation run, or explicitly state why validation was skipped.
- Mention compatibility notes, migration notes, or risks if applicable.
- Avoid empty, vague, or cosmetic commit messages.

---

## 13. Issue and label awareness

If the repository uses issue IDs, label taxonomies, or workstream references:
- preserve them consistently
- do not invent incompatible label styles
- use the repository’s existing `Kind/*`, `Status/*`, `Priority/*`, or equivalent taxonomy where defined

If information is missing, use the repo’s documented fallback status where applicable.

---

## 14. Safety against instruction drift

- Do not overwrite repository conventions with personal preference.
- Do not assume one language’s tooling rules apply to another.
- Do not assume Python checks for a Rust-only change unless required.
- Do not assume Rust checks for a docs-only change unless required.
- Do not assume CI locations, build systems, branch rules, or release flows.
- Inspect first, then act.

---

## 15. Task execution pattern

For implementation tasks, default to this order:

1. Read applicable instructions (`AGENTS.md`, `WORKSTREAMS.md`, relevant docs).
2. Identify the workstream or task and any required file reservations.
3. Inspect the relevant code and surrounding architecture.
4. Use repository design principles and any referenced source material where relevant.
5. Make the minimal correct change.
6. Update tests, docs, changelog, and workstream records as needed.
7. Run required validation, formatting, linting, and test steps.
8. Review the diff for scope creep and unrelated edits.
9. Commit touched files using the correct schema.
10. Report clearly what was changed, validated, committed, and deferred.

---

## 16. Design-principle guidance

Where a task references a design source such as `ICSD.xml`, use it as a design-principles reference rather than a text template.

Apply its principles where relevant, including:
- determinism
- attributable change
- auditability
- invariant preservation
- explicit behaviour
- maintainable structure
- reviewable scope
- clear handling of compatibility-sensitive behaviour

Do not copy blindly from source material.
Interpret principles appropriately for the current repository and task.

---

## 17. Completion standard

A task is not fully complete if:
- intentionally touched files remain uncommitted when a finished change was expected
- required validation was skipped without explanation
- required documentation/workstream updates were omitted
- follow-on work that materially affects correctness was discovered but left untracked

If no commit has been created by the end of the run, treat the task as incomplete unless explicitly instructed otherwise.

---

## 18. Deterministic output discipline

When generating or modifying:
- bytecode
- encoded structures
- serialised outputs
- canonical representations
- hash/signature inputs
- conformance-sensitive artefacts

Ensure:
- identical inputs produce identical outputs
- no alternate encodings are introduced unless explicitly required
- ordering is stable, documented, and intentional
- output shape does not vary unintentionally
- behaviour is not dependent on platform, locale, clock, filesystem ordering, or runtime environment unless explicitly designed that way

Avoid:
- non-deterministic ordering
- hidden randomness
- environment-dependent output differences
- incidental formatting drift in canonical outputs
- equivalent-but-different encodings that could break verification, hashing, replay, or downstream comparison