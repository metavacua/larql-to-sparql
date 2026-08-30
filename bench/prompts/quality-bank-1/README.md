# Q-BANK-1 — a frozen bank for characterising representation quality

One prompt is an anecdote. Granite measured KL 0.331 bits/token and
Glimmer 0.019 on single prompts — a 17x spread that says nothing about
either model except that single prompts do not characterise anything.

This bank exists to turn that into a distribution.

## What it is

`prompts.json` — heterogeneous text, deliberately spanning the regimes
where a quantiser behaves differently:

| category    | why it is here |
|-------------|----------------|
| `factual`   | sharp distributions; an argmax flip is expensive |
| `prose`     | ordinary continuation, moderate entropy |
| `code`      | low-entropy structure, long-tail identifiers |
| `arithmetic`| sharp, and sensitive to accumulated error |
| `structured`| format tokens with near-deterministic successors |
| `uncertain` | deliberately high-entropy: KL means something different here |
| `longform`  | natural text, teacher-forced — many observations, no generation drift |

The `longform` entries carry most of the positions. Teacher forcing over
real text gives a large sample cheaply and, crucially, without generation
drift: every position sees identical context on both arms, so a
divergence is attributable to the representation rather than to the two
arms having wandered apart.

## Method

Both arms are teacher-forced over the same ids. Per position:

- KL(BF16 ‖ NVFP4) in bits
- ΔNLL of the actually-next token
- top-1 agreement, top-5 set overlap
- max / mean |Δlogit|
- **BF16 top-1 margin** — `p1 - p2` of the reference
- **BF16 entropy** in bits

The last two are the interpretation. An argmax flip where the reference
separated its first and second choice by 0.001 is a different event from
one where it was certain, and a KL of 0.05 means something different over
a sharp distribution than over a flat one. Reporting the mean alone hides
both.

## What this bank does NOT cover

**Multimodal.** `vindex3 exec` takes token ids and has no image path, so
the perception tower is protected by policy but not exercised here.
Protecting it is only *demonstrably* right once something runs through it;
until then the claim is structural, not measured.

**Thresholds.** None are set. The point of a first bank is to learn the
shape of the distribution, not to ratify a number chosen before seeing it.

## Frozen

Editing `prompts.json` invalidates comparison against banked runs. Add a
`quality-bank-2` instead.
