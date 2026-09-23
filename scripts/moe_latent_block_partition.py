"""Step 2 — per-layer block clustering for MoE latent-axis sparsity.

Closes (or revives) Route 2 at the level of the policy it actually requires:
per-layer channel permutations, optimised against the EXACT induced loss of
16-channel blocking, evaluated on held-out tokens through the HF reference.

Why exact loss and not pairwise disagreement. For block `g` on token `t`, with
`s_{t,g}` = how many of its 16 channels the block-1 mask retains, the best
block approximation keeps the top-`K` blocks by `s`, and its Hamming
distortion is

    L_t = sum_{g kept} (16 - s_{t,g}) + sum_{g dropped} s_{t,g}

which prices BOTH within-block disagreement AND the competition between blocks
under the fixed budget. Pairwise `D_ij = P(m_i != m_j)` is a good clustering
INITIALISER but two partitions can share pairwise disagreement and differ in
`L_t`.

Permutation exactness: this never moves a weight. It groups channel INDICES by
a permuted order and zeroes the losing channels of `z`. That is arithmetically
identical to permuting `W` and taking contiguous blocks (`z'=Pz`, `W'=W·Pᵀ`),
so the "permuted dense" control is exact by construction rather than
numerically — there is nothing to drift.

Calibration and held-out are DISJOINT CONTIGUOUS spans, not interleaved.
"""

import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe import modeling_olmoe as M

MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
# Absolute: this script gets launched from varying cwds and a relative
# path has now killed two runs at startup.
CORPUS = "/Users/christopherhay/chris-source/larql/data/gutenberg/frankenstein.txt"
SPAN = 512           # tokens per span; calibration = span 0, held-out = span 1
BLOCK = 16
DTYPE = torch.bfloat16
RETENTIONS = [0.875, 0.75, 0.625, 0.5]
# Reduced from 20000 after the r=0.875 run showed sharply diminishing returns:
# 2545 accepted swaps bought only 0.4% calibration L_t over D-clustering and
# made held-out bits/token WORSE. More budget buys overfitting, not structure.
SWAP_CANDIDATES = 8000
RANDOM_SEEDS = [0, 1, 2]
LN2 = 0.6931471805599453

# Runtime config consumed by the patched forward.
CFG = {"retention": None, "mode": "magnitude", "block": 1, "perms": None,
       "capture": None, "layer_counter": 0}


def block_mask_from(z: torch.Tensor, r: float, block: int, perm) -> torch.Tensor:
    """Retained-channel boolean mask [tokens, hidden] for one layer."""
    n = z.shape[-1]
    if block == 1:
        keep = max(1, min(n, int(np.ceil(n * r))))
        idx = z.abs().float().topk(keep, dim=-1).indices
        m = torch.zeros_like(z, dtype=torch.bool)
        m.scatter_(-1, idx, True)
        return m
    order = torch.as_tensor(perm, dtype=torch.long) if perm is not None \
        else torch.arange(n)
    nblocks = n // block
    keep_b = max(1, min(nblocks, int(np.ceil(nblocks * r))))
    # Block score = summed |z| mass, matching the Rust implementation.
    zb = z.abs().float()[:, order].reshape(-1, nblocks, block).sum(-1)
    top = zb.topk(keep_b, dim=-1).indices                      # [tokens, keep_b]
    m = torch.zeros(z.shape[0], nblocks, block, dtype=torch.bool)
    m.scatter_(1, top.unsqueeze(-1).expand(-1, -1, block), True)
    m = m.reshape(z.shape[0], n)
    out = torch.zeros_like(m)
    out[:, order] = m
    # INVARIANT: exactly keep_b whole blocks retained, every token.
    assert torch.all(out.sum(-1) == keep_b * block), "block budget violated"
    return out


def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch, seq, hidden = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden)
    # Routing on the UNMASKED input — same experts, same weights.
    _, top_k_weights, top_k_index = self.gate(hidden_states)

    layer = CFG["layer_counter"]
    CFG["layer_counter"] += 1
    r = CFG["retention"]
    if r is not None:
        perm = CFG["perms"][layer] if CFG["perms"] is not None else None
        m = block_mask_from(hidden_states, r, CFG["block"], perm)
        if CFG["capture"] is not None:
            CFG["capture"].setdefault(layer, []).append(m.cpu().numpy())
        hidden_states = torch.where(m, hidden_states, torch.zeros_like(hidden_states))

    out = self.experts(hidden_states, top_k_index, top_k_weights)
    return out.reshape(batch, seq, hidden)


# ── Partition construction ───────────────────────────────────────────────

def exact_loss(masks: np.ndarray, groups: np.ndarray, keep_b: int) -> float:
    """Mean L_t over tokens for a partition. `groups` is [nblocks, BLOCK]."""
    s = masks[:, groups].sum(-1)                       # [tokens, nblocks]
    order = np.argsort(-s, axis=-1)
    kept = np.zeros_like(s, dtype=bool)
    np.put_along_axis(kept, order[:, :keep_b], True, axis=-1)
    return float((np.where(kept, BLOCK - s, s)).sum(-1).mean())


def cluster_by_disagreement(masks: np.ndarray) -> np.ndarray:
    """Greedy balanced partition minimising within-group disagreement."""
    n = masks.shape[1]
    p = masks.mean(0)
    co = (masks.astype(np.float32).T @ masks.astype(np.float32)) / masks.shape[0]
    D = p[:, None] + p[None, :] - 2 * co               # P(m_i != m_j)
    np.fill_diagonal(D, np.inf)
    unassigned = np.ones(n, dtype=bool)
    groups = []
    while unassigned.any():
        seed = int(np.flatnonzero(unassigned)[0])
        grp = [seed]
        unassigned[seed] = False
        while len(grp) < BLOCK and unassigned.any():
            cand = np.flatnonzero(unassigned)
            best = cand[np.argmin(D[np.ix_(grp, cand)].sum(0))]
            grp.append(int(best))
            unassigned[best] = False
        groups.append(grp)
    return np.array(groups)


def refine_by_exact_loss(masks, groups, keep_b, rng, budget=SWAP_CANDIDATES):
    """Swap channels between groups, accepting strict improvements in L."""
    groups = groups.copy()
    cur = exact_loss(masks, groups, keep_b)
    nb = groups.shape[0]
    accepted = 0
    for _ in range(budget):
        g1, g2 = rng.integers(0, nb, 2)
        if g1 == g2:
            continue
        i, j = rng.integers(0, BLOCK, 2)
        groups[g1, i], groups[g2, j] = groups[g2, j], groups[g1, i]
        new = exact_loss(masks, groups, keep_b)
        if new < cur:
            cur, accepted = new, accepted + 1
        else:
            groups[g1, i], groups[g2, j] = groups[g2, j], groups[g1, i]
    return groups, cur, accepted


def main() -> int:
    M.OlmoeSparseMoeBlock.forward = patched_forward
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(CORPUS, "rb").read(65536).decode("utf-8", errors="ignore")
    all_ids = tok(text, return_tensors="pt").input_ids[0]
    calib = all_ids[:SPAN].unsqueeze(0)
    heldout = all_ids[SPAN:2 * SPAN].unsqueeze(0)
    print(f"calibration tokens {calib.shape[1]}, held-out {heldout.shape[1]} "
          f"(disjoint contiguous spans)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval()

    def score(ids, retention, block, perms, capture=None) -> float:
        CFG.update(retention=retention, block=block, perms=perms,
                   capture=capture, layer_counter=0)
        with torch.no_grad():
            out = model(ids, labels=ids)
        return float(out.loss) / LN2

    results = {}
    for r in RETENTIONS:
        print(f"\n{'='*72}\nretention r={r}", flush=True)
        # Phase A — capture block-1 masks on CALIBRATION only.
        cap = {}
        score(calib, r, 1, None, capture=cap)
        masks = {l: np.concatenate(v, 0) for l, v in cap.items()}
        nlayers = len(masks)
        n = masks[0].shape[1]
        keep_b = max(1, min(n // BLOCK, int(np.ceil((n // BLOCK) * r))))

        # Phase B — per-layer partition, offline.
        perms_D, perms_ref = [], []
        stats = []
        rng = np.random.default_rng(0)
        for l in range(nlayers):
            mk = masks[l]
            gD = cluster_by_disagreement(mk)
            lD = exact_loss(mk, gD, keep_b)
            gR, lR, acc = refine_by_exact_loss(mk, gD, keep_b, rng)
            gN = np.arange(n).reshape(-1, BLOCK)
            lN = exact_loss(mk, gN, keep_b)
            perms_D.append(gD.ravel())
            perms_ref.append(gR.ravel())
            stats.append((lN, lD, lR, acc))
        ln, ld, lr = (float(np.mean([s[i] for s in stats])) for i in range(3))
        acc = int(np.sum([s[3] for s in stats]))
        print(f"  calib mean L_t/token  native {ln:.1f}  D-cluster {ld:.1f}  "
              f"exact-refined {lr:.1f}   ({acc} swaps accepted)", flush=True)
        print(f"  (L_t is Hamming distance to the block-1 mask; "
              f"{keep_b*BLOCK} of {n} channels kept)", flush=True)

        # Phase C — held-out evaluation.
        base = score(heldout, None, 1, None)
        b1 = score(heldout, r, 1, None)
        bn = score(heldout, r, BLOCK, None)
        bd = score(heldout, r, BLOCK, perms_D)
        br = score(heldout, r, BLOCK, perms_ref)
        rnds = []
        for seed in RANDOM_SEEDS:
            g = np.random.default_rng(seed)
            rp = [g.permutation(n) for _ in range(nlayers)]
            rnds.append(score(heldout, r, BLOCK, rp))
        # Held-out L_t — the diagnostic separating "partition overfit the
        # calibration masks" from "Hamming objective misses functional cost".
        cap_h = {}
        score(heldout, r, 1, None, capture=cap_h)
        masks_h = {l: np.concatenate(v, 0) for l, v in cap_h.items()}
        rng_ctl = np.random.default_rng(7)
        parts = {"native": [np.arange(n).reshape(-1, BLOCK) for _ in range(nlayers)],
                 "D-cluster": [p.reshape(-1, BLOCK) for p in perms_D],
                 "refined": [p.reshape(-1, BLOCK) for p in perms_ref],
                 "random": [rng_ctl.permutation(n).reshape(-1, BLOCK)
                            for _ in range(nlayers)]}
        print("  L_t/token (Hamming vs block-1)   calib -> held-out", flush=True)
        per_layer = {}
        for name, gs in parts.items():
            lc = [exact_loss(masks[l], gs[l], keep_b) for l in range(nlayers)]
            lh = [exact_loss(masks_h[l], gs[l], keep_b) for l in range(nlayers)]
            per_layer[name] = (lc, lh)
            print(f"    {name:<12} {np.mean(lc):>7.1f} -> {np.mean(lh):>7.1f}"
                  f"   worst layer held-out {int(np.argmax(lh))} @ {max(lh):.1f}",
                  flush=True)
        np.savez_compressed(
            f"/private/tmp/claude-501/-Users-christopherhay-chris-source-larql/90cb8359-7e34-44a7-a889-c755caff8f26/scratchpad/step2_r{r}.npz",
            masks_calib=np.stack([masks[l] for l in range(nlayers)]),
            masks_heldout=np.stack([masks_h[l] for l in range(nlayers)]),
            **{f"part_{k}": np.stack(v) for k, v in parts.items()},
            **{f"Lt_{k}_{sp}": np.array(v[i])
               for k, v in per_layer.items() for i, sp in enumerate(["calib", "held"])},
        )
        results[r] = (base, b1, bn, bd, br, rnds)
        print(f"  held-out bits/token   dense {base:.3f}", flush=True)
        for name, v in [("block=1", b1), ("block=16 native", bn),
                        ("block=16 D-cluster", bd),
                        ("block=16 exact-refined", br)]:
            print(f"    {name:<24} {v:>8.3f}  {v-base:>+7.3f}  "
                  f"{100*(v-base)/base:>+7.2f}%", flush=True)
        print(f"    {'block=16 random x3':<24} "
              f"{np.mean(rnds):>8.3f}  {np.mean(rnds)-base:>+7.3f}  "
              f"(sd {np.std(rnds):.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
