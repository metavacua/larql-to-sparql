# Experiment 05: Syntax-Circuit Routing

## Hypothesis

Syntax features (L0-12) predict which attention circuits (L13-26) activate,
analogous to how trigram types predicted MoE expert routing in GPT-OSS.

If true, attention can be replaced with a cached routing table:
syntax feature -> circuit -> cached pattern. No QK computation needed.

## Connection to GPT-OSS MoE

| GPT-OSS (MoE) | Gemma 3-4B (dense + vindex) |
|---|---|
| Trigram type detected -> routes to expert | Syntax features in L0-12 detect pattern |
| ADJ->SYN->ADJ -> E5 (synonym expert) | `wn:synonym` features fire -> synonym circuit |
| NOUN->AS->NOUN -> E25 (analogy expert) | `wn:hypernym` features fire -> hypernym circuit |
| Explicit expert selection | Implicit attention head combinations |

## Scripts

### Minimal (start here)

```bash
python3 experiments/05_syntax_circuit_routing/syntax_attention_minimal.py \
    --model google/gemma-3-4b-it \
    --vindex output/gemma3-4b-f16.vindex
```

20 prompts, 4 categories. Captures syntax gates + attention head activity.
Prints a table. If categories cluster -> routing exists. ~2 min.

### Full experiment

```bash
python3 experiments/05_syntax_circuit_routing/syntax_circuit_routing.py \
    --model google/gemma-3-4b-it \
    --vindex output/gemma3-4b-f16.vindex \
    --circuits output/gemma3-4b-f16.vindex/circuits.json \
    --output output/syntax_circuit_routing/
```

240 prompts, 16 template families. Builds co-occurrence matrix.
Circuit matching optional (works without circuits.json).

## Success criteria

- **Sparsity > 0.8**: Most syntax features map to <=3 circuits/heads
- **Category separation**: Different trigram types route to different head clusters
- **Labeled features in top rules**: `wn:hypernym`, `wn:synonym` etc. have clean mappings
- **Code vs knowledge separation**: `python:*` features -> different heads than `entity_predicate`

## LARQL integration

Once routing table is confirmed, it enables:

```
larql> WALK "France" -> capital;
  Syntax gates (0.12ms) -> Circuit lookup (0.00ms) -> Graph walk (0.98ms) -> Output (0.05ms)
  Total: 1.15ms vs ~45ms neural = 39x speedup
```
