# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reference the `./AGENTS.md` file for the broader context of this project.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## native-cli-ai (nca)

The project can be queried via the native-cli-ai (nca) toolchain:

- `larql serve --no-infer --port 8181 /path/to/code-base-vindex` — serves the code-base-vindex (Vindexfile at repo root) via HTTP on `/v1/describe`, `/v1/walk`, `/v1/stats`
- `nca index build` — indexes this workspace for nca agent context (writes to `~/.nca/workspaces/larql-main-*/cli-index.json`)
- `nca run --prompt "..."` — one-shot agent query; uses Anthropic or MiniMax provider

The `Vindexfile` at the repo root defines the code-base-vindex: qwen3-0.6b base + 84 triples encoding the Cargo crate dependency graph and architectural roles from AGENTS.md.