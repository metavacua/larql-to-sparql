# Legal Transition Plan

## Current State (as of 2026-05-06)

The larql-to-sparql project operates under **Apache License 2.0** for all first-party source code and documentation. This applies to:

- Original LARQL codebase (Chris Hay, https://github.com/chrishayuk/larql) — Apache-2.0
- Fork toolchain and compliance infrastructure (Ian Douglas Lawrence Norman McLean) — Apache-2.0
- Licensing audit deliverables (`audit/`) — CC-BY-SA-4.0 (by fork)

Copyright attribution is explicit and tracked in `REUSE.toml` per the REUSE Specification. Every file has a documented copyright holder and license identifier.

## Future Transition (Conditional)

**Timeline**: After achieving the following stability milestones:
1. All platform builds pass (Ubuntu, macOS, and beyond)
2. The dependency tree is free of foundational bugs
3. The codebase is minimal and maintainable
4. CI/CD infrastructure is complete and production-ready

**Proposed licensing posture**:

- **Source code** (crates/larql-*/src/) → AGPL-3.0-or-later
  - Rationale: Protects against proprietary forks of derivative works while remaining compatible with Apache-2.0 upstream
  - Mechanism: Each source file will be re-licensed by adding an AGPL-3.0 header and updating `REUSE.toml` annotations
  
- **Documentation and specifications** (docs/, audit/new content) → CC-BY-SA-4.0
  - Rationale: Encourages collaborative knowledge-sharing while requiring attribution and reciprocal sharing
  - Existing audit/ (already CC-BY-SA-4.0) will remain; new docs will be explicitly licensed
  
- **Build infrastructure and config** (.github/, scripts/, deny.toml, cliff.toml) → Apache-2.0 (unchanged)
  - Rationale: Keeps tools redistributable and vendor-agnostic

## Transition Procedure

When ready to execute:

1. **Pre-licensing audit**: `reuse lint` must pass before any license changes
2. **No active annotations for future licenses**: AGPL-3.0 and CC-BY-SA-4.0 annotations will NOT be added to `REUSE.toml` until the corresponding files are re-licensed
3. **Atomic PR**: All files transitioning to AGPL-3.0 must be updated in a single PR with a clear commit message
4. **Backwards compatibility**: The Apache-2.0 license text will remain in LICENSES/ alongside new license texts
5. **Dependent projects**: Public notice period before the transition, allowing downstreams to fork or adjust their license policies

## Provenance Principles

Throughout this transition, the following non-negotiable principles apply:

- **No copyright laundering**: Every copyright attribution is accurate and traceable to Git history
- **No retroactive re-licensing**: Files will only be re-licensed forward; existing commits remain under their original license
- **Explicit attribution**: If you can't tell where a file came from, the provenance system has failed
- **Third-party isolation**: Copied files from upstream projects retain their original copyright and license, in discrete REUSE.toml annotation blocks

## Questions or Disputes

Contact: Ian Douglas Lawrence Norman McLean (project maintainer)
