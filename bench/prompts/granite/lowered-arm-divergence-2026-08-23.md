# Obligation: the lowered arms diverge from the Granite oracle

Found 2026-08-23 while building the REPRESENT quality gate. **Not a
REPRESENT defect** — it reproduces on the canonical container with no
compiled representation involved, and it is independent of the
`logits_scaling` fix (it reproduces after that too, and a monotonic logit
scale could not have caused it).

`bench/prompts/granite/vindex3-oracle-2026-08-19.txt` banks an external
transformers/torch ground truth and records which backends were verified
against it: `reference`, `production`, `metal`. The **lowered** arms were
never covered by that oracle, and they disagree with it.

## Reproducer

Prompt is the oracle's own chat-wrapped 15 ids.

```bash
TOK="100264,882,100265,3923,374,279,6864,315,9822,30,100257,198,100264,78191,100265"
G=~/chris-models/granite-4.1-3b-fixed.vindex3

# oracle-verified path — agrees
larql vindex3 exec $G --tokens "$TOK" --backend metal --generate 8
#   generated ids: 791,6864,315,9822,374,12366,13,100257   "The capital of France is Paris."

# lowered path — disagrees
larql vindex3 exec $G --tokens "$TOK" --backend metal-lowered-f16 --generate 8
#   generated ids: 198,791,827,55436,198,94447,198,827
```

Expected (HF/PyTorch ground truth, `do_sample=False`):
`[791, 6864, 315, 9822, 374, 12366, 13, 100257]`

## Why it stayed hidden

The banked oracle predates the lowered path and says so. Nothing since has
run Granite's external ground truth through `metal-lowered-*`, so the two
never met. The lowered arms have been exercised on gpt-oss and Glimmer,
where their own oracles pass — so this is plausibly Granite-specific
carriage (`embed_scale`, `residual_scale` and `logit_scale` are all
Granite-shaped scalars the lowering must honour) rather than a general
lowering defect.

That is a hypothesis, not a finding. It has not been investigated.

## What it does not affect

- REPRESENT: the stored/transient parity gate ran on the interpreter arm
  (`metal-nvfp4-no-head`) and is bitwise exact.
- The quality gate: `metal` and `metal-nvfp4-no-head` are the same arm
  family, and both reproduce the external oracle.
