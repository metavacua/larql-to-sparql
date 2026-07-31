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
   `## ADDED Requirements` (or `MODIFIED` / `REMOVED` / `RENAMED`).
3. **Every Requirement contains SHALL or MUST** (or `SHALL NOT`).
4. **Every Requirement has at least one Scenario.** Scenarios use four
   hashes (`####`).
5. **Every Scenario references at least one test.** Annotate with an HTML
   comment: `<!-- test: path/to/test.rs::test_name -->`.
