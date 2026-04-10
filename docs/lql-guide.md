# LQL Quick Start Guide

**LQL** (Lazarus Query Language) is the query language for neural network weights treated as a graph database. One binary, no Python, no GPU.

## Launch the REPL

```bash
cargo run -p larql-cli -- repl
```

Features: arrow keys, command history (`~/.larql_history`), Ctrl-R search, Ctrl-C cancel, Ctrl-D exit.

Single statement:

```bash
cargo run -p larql-cli -- lql 'SHOW MODELS;'
```

## Getting Started

### 1. Extract a model

```sql
-- Browse-only (~3 GB f16 / ~6 GB f32, fast queries, no inference)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";

-- With inference support (~6 GB f16 / ~12 GB f32, enables INFER)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE;

-- Full (~10 GB f16 / ~18 GB f32, enables COMPILE for recompilation)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;
```

CLI equivalent: `larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16`

### 2. Connect

```sql
-- Use a pre-extracted vindex (fast, all operations)
USE "gemma3-4b.vindex";
STATS;

-- Or point directly at model weights (no extraction needed)
USE MODEL "google/gemma-3-4b-it";
STATS;
-- Supports: INFER, EXPLAIN INFER, STATS
-- For WALK/DESCRIBE/SELECT/INSERT: extract into a vindex first
```

### 3. Browse knowledge

```sql
-- What does the model know about France?
-- Verbose by default: relation labels, also-tokens, layer ranges
DESCRIBE "France";

-- Compact view: top edges, primary layer only
DESCRIBE "France" BRIEF;

-- No labels — pure model signal
DESCRIBE "France" RAW;

-- Show all layer bands (syntax + knowledge + output)
DESCRIBE "France" ALL LAYERS;

-- Single layer
DESCRIBE "Mozart" AT LAYER 26;

-- Feature scan: which features fire for a prompt?
WALK "The capital of France is" TOP 10;

-- Per-layer trace
EXPLAIN WALK "The capital of France is" LAYERS 24-33;

-- SQL-style edge queries
SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 10;
```

### 4. Run inference

Requires model weights: either a vindex built with `WITH INFERENCE` / `WITH ALL`,
or a `USE MODEL` session (direct weight access).

```sql
-- Next-token prediction with attention
INFER "The capital of France is" TOP 5;

-- Compare walk (no attention) vs infer (with attention)
INFER "The capital of France is" TOP 5 COMPARE;

-- Full inference trace
EXPLAIN INFER "The capital of France is" TOP 5;
```

### 5. Edit knowledge

```sql
-- Insert a fact (multi-layer constellation install, alpha=0.25 default)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

-- Insert with all knobs: center the span on a specific layer, set
-- confidence, dial the override strength
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30;

-- Verify
DESCRIBE "John Coyle";

-- Delete
DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";

-- Update by entity (works on both heap and mmap-loaded vindexes)
UPDATE EDGES SET target = "London"
    WHERE entity = "John Coyle" AND relation = "lives-in";

-- Update by (layer, feature) — fast-path, bypasses the entity scan
UPDATE EDGES SET target = "London", confidence = 0.95
    WHERE layer = 26 AND feature = 8821;
```

INSERT is a multi-layer constellation install (8 layers × `alpha=0.25` is the
validated regime — see `docs/training-free-insert.md`). The defaults are
deliberately conservative; raise `ALPHA` for stubborn facts at the cost of
nudging neighbouring facts.

### 6. Patches

Patches are lightweight knowledge diffs — portable JSON files that modify a vindex without touching the base files.

```sql
-- Start recording a patch
BEGIN PATCH "medical-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

-- Save (base vindex NOT modified)
SAVE PATCH;

-- Apply a patch
APPLY PATCH "medical-knowledge.vlp";

-- Stack multiple patches
APPLY PATCH "fix-hallucinations.vlp";

-- See active patches
SHOW PATCHES;

-- Remove a patch (instantly reverts)
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

### 7. Recompile

```sql
-- See what changed
DIFF "gemma3-4b.vindex" CURRENT;

-- Bake the patches into a fresh standalone vindex (instant on APFS:
-- weight files hardlinked from source, only down_weights.bin gets the
-- override columns rewritten in place).
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

-- Use the compiled vindex like any other — INFER produces the new
-- facts with no patch overlay loaded.
USE "gemma3-4b-medical.vindex";
INFER "The capital of Atlantis is" TOP 5;

-- Or compile back to HuggingFace format. The constellation is in the
-- standard down_proj tensors, so loading in Transformers / GGUF
-- runtimes Just Works.
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

## Residual Stream Trace

Trace decomposes a forward pass into attention and FFN contributions at every layer.

```sql
-- What does the model predict at each layer?
TRACE "The capital of France is";

-- Track a specific answer through all layers
TRACE "The capital of France is" FOR "Paris";
-- Shows rank, probability, attn/FFN logit contribution, who pushes the answer

-- Attention vs FFN decomposition at the phase transition
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Save the trace to an mmap'd file
TRACE "The capital of France is" SAVE "france.trace";

-- Trace all token positions (not just last)
TRACE "The capital of France is" POSITIONS ALL SAVE "france_all.trace";
```

TRACE requires model weights (`WITH ALL` or `WITH INFERENCE` during EXTRACT). It uses the same WalkFfn as INFER — INSERT/DELETE mutations are reflected.

## Introspection

```sql
-- Discovered relation types
SHOW RELATIONS WITH EXAMPLES;

-- Layer summary
SHOW LAYERS;
SHOW LAYERS RANGE 14-27;

-- Feature details
SHOW FEATURES 26 LIMIT 20;

-- Available vindexes in current directory
SHOW MODELS;

-- Active patches
SHOW PATCHES;

-- Knowledge graph coverage
STATS;
```

## Layer Bands

DESCRIBE groups features into three bands based on the model's layer structure:

| Band | Gemma 3 4B | Llama 3 8B | What it contains |
|------|-----------|-----------|-----------------|
| Syntax | L0-13 | L0-12 | Morphological, syntactic, code |
| Knowledge | L14-27 | L13-25 | Factual relations (default view) |
| Output | L28-33 | L26-31 | Formatting, token selection |

```sql
DESCRIBE "France";              -- Verbose: relation labels, also-tokens, layer ranges (default)
DESCRIBE "France" BRIEF;        -- Compact: top edges, primary layer only
DESCRIBE "France" RAW;          -- No labels, pure model signal
DESCRIBE "France" SYNTAX;       -- Syntax band only
DESCRIBE "France" OUTPUT;       -- Output band only
DESCRIBE "France" ALL LAYERS;   -- All three bands
```

Bands are model-specific — computed automatically during EXTRACT from known architecture boundaries.

## Statement Reference

| Category | Statements |
|----------|-----------|
| Lifecycle | EXTRACT, COMPILE, DIFF, USE |
| Browse | WALK, DESCRIBE, SELECT, EXPLAIN WALK |
| Inference | INFER, EXPLAIN INFER |
| Trace | TRACE (with FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE) |
| Mutation | INSERT, DELETE, UPDATE, MERGE |
| Patches | BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH |
| Introspection | SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS |
| Pipe | `\|>` chains two statements |

See [lql-spec.md](lql-spec.md) for the full language specification.
