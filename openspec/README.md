# OpenSpec workflow

This directory holds the spec-first contract for larql.

## Layout

```
openspec/
  config.yaml                       # OpenSpec project config
  specs/<capability>/spec.md        # canonical specs, one per capability
  changes/<change-id>/              # in-flight proposals
    proposal.md                     # why + what
    design.md                       # how (cross-cutting changes only)
    tasks.md                        # implementation backlog
    specs/<capability>/spec.md      # delta with ## ADDED / MODIFIED / REMOVED
    gaps-unbacked-scenarios.md      # auto-generated; lists scenarios with no test
    gaps-untested-code.md           # auto-generated; lists code without tests
  changes/archive/                  # archived (merged) changes
  coverage/
    traceability.md                 # requirement → test matrix
    traceability.json               # machine-readable trace data
```

## Authoring rules

1. **One capability per coherent feature area.** Use kebab-case names.
2. **Each spec is a delta when in `changes/<id>/`.** Top-level header is
   exactly `## ADDED Requirements` (or `MODIFIED` / `REMOVED` / `RENAMED`) —
   **no `: <capability>` suffix** (the capability comes from the folder name;
   the suffix makes the `openspec` v1.4.1 parser see zero deltas).
3. **Each requirement heading is `### Requirement: <name>`** — the literal
   `Requirement:` prefix is required; a bare `### REQ-XXX: ...` parses as zero
   requirements. Keep the `REQ-XXX` id in the name, e.g.
   `### Requirement: REQ-QB-001 — Dicke state constructor`.
4. **Every Requirement contains SHALL or MUST** (or `SHALL NOT`) **on its first
   body line** — the validator checks the leading line, so don't let `SHALL`
   wrap onto a later line.
5. **Every Requirement has at least one Scenario.** Scenarios use four
   hashes (`#### Scenario:`).
6. **Every Scenario references at least one test.** Annotate with an HTML
   comment: `<!-- test: path/to/test.rs::test_name -->`.
7. **`proposal.md` has a `## What Changes` section**, and **every change must
   pass `openspec validate <id> --strict`** before commit — this is the gate, not
   an afterthought. (`openspec` lives at
   `~/.local/share/pi-node/current/bin/openspec`; v1.4.1.)
