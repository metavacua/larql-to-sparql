# KV Cache Benchmark: Accuracy Test Suite

## Purpose

Prove that memory savings don't come at the cost of correctness.
Every strategy must demonstrate it preserves the information
that the baseline (Standard KV) has access to. Without this,
the memory numbers are marketing, not engineering.

The accuracy tests answer three questions:
1. Does each strategy produce the SAME output as baseline?
2. At what context length does accuracy degrade, if ever?
3. Which failure modes are graceful (coherence loss) vs catastrophic (wrong answer)?

---

## Test Categories

### Category 1: Top-1 Token Match

The simplest correctness test. Run the same prompt through all five
strategies, compare the next-token prediction against Standard KV baseline.

Test prompts (diverse categories):
- Factual:        "The capital of France is"                    → Paris
- Factual:        "The currency of Japan is the"                → yen
- Factual:        "Mozart was born in"                          → Salzburg
- Arithmetic:     "25 × 4 ="                                   → 100
- Completion:     "To be or not to be, that is the"             → question
- Grammar:        "She don't like → She doesn't"                → (grammatical)
- Code:           "def fibonacci(n):\n    if n <= 1:\n        return" → n
- Conversation:   "How are you today? I'm doing"                → well/great/fine
- Scientific:     "Water freezes at"                            → 0/32
- Geographic:     "The longest river in Africa is the"          → Nile

Expected results:
  Standard KV      100%           (IS the baseline)
  TurboQuant 4b    >98%           near-lossless at 4-bit
  TurboQuant 3b    >95%           slight degradation
  Markov RS        100%           (KL=0.0, proven bit-perfect)
  Hybrid RS+CA     >99%           dynamic heads preserve precision
  Graph Walk       >95%           factual queries only, fallback for rest

### Category 2: KL Divergence on Output Distribution

Compare the full softmax distribution over the vocabulary, not just argmax.

Expected results:
  Standard KV      0.0              (IS the baseline)
  TurboQuant 4b    <0.01            near-lossless
  TurboQuant 3b    <0.05            measurable but small
  Markov RS        0.0              (proven bit-perfect)
  Hybrid RS+CA     <0.001           dynamic heads at full precision
  Graph Walk       N/A              (different prediction mechanism)

### Category 3: Needle-in-a-Haystack

Plant a specific, retrievable fact at a known position in the context.

Test 3a: Short Context Needle (within window) - 500 tokens
Test 3b: Long Context Needle (outside window) - 4000 tokens
Test 3c: Very Long Context Needle (stress test) - 32K/64K/128K/370K
Test 3d: Multi-Needle - 5 facts at different positions in 32K

Key finding to demonstrate:
"At 370K tokens, Standard KV and TurboQuant FAIL to find the needle
 because of softmax dilution. Markov RS and Hybrid RS SUCCEED because
 they route to the relevant window and attend over 512 positions."

### Category 4: Multi-Turn Fact Retention

Simulate real conversation where facts established early must be remembered.
Score: facts retained at turn 15 (out of 3) and turn 25 (out of 5).

### Category 5: Multi-Token Generation Coherence

Generate 50 tokens from the same prompt through each strategy.
Measure first divergence point, token match rate, BLEU score.

### Category 6: Adversarial Accuracy

Test 6a: Entity Confusion (attacks template caching)
Test 6b: Rare Token Recovery (attacks quantization)
Test 6c: Context-Dependent Meaning (attacks graph walk)
Test 6d: Long-Range Dependency (attacks bounded window)
Test 6e: Attention Head Stress (attacks hybrid caching)

---

## What This Proves for the Video

Frame A — "Same output, less memory"
Frame B — "We find what they can't" (needle at 370K)
Frame C — "Accuracy table" (five strategies, five metrics)

The counterintuitive finding:
"Storing less can be more accurate than storing everything,
 because bounded attention over 512 tokens beats diluted
 attention over 370,000 tokens."
