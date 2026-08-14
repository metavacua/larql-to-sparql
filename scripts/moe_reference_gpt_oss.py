"""Reference GPT-OSS MoE block output, for the Rust differential test.

Weights come from the same LCG the Rust test uses, drawn in the same order,
so neither side needs a fixture file. The maths is transcribed from
`transformers/models/gpt_oss/modeling_gpt_oss.py` (GptOssTopKRouter.forward
and GptOssExperts._apply_gate / forward).

Tensor orientations match the ON-DISK layout larql loads:
  fused gate_up : [2*inter, hidden]   (reference param is its transpose)
  down          : [hidden, inter]     (reference param is its transpose)
"""

import numpy as np

HIDDEN = 16
INTER = 16
EXPERTS = 4
TOP_K = 2
TOKENS = 3

ALPHA = 1.702
LIMIT = 7.0

MUL = 6364136223846793005
INC = 1442695040888963407
MASK = (1 << 64) - 1
U32_MAX = 0xFFFFFFFF


class Lcg:
    """Mirrors the Rust test's generator exactly."""

    def __init__(self, seed):
        self.state = seed

    def next(self, scale):
        self.state = (self.state * MUL + INC) & MASK
        u32 = self.state & U32_MAX
        return np.float32(u32 / U32_MAX * 2.0 * scale - scale)

    def mat(self, rows, cols, scale):
        return np.array(
            [[self.next(scale) for _ in range(cols)] for _ in range(rows)],
            dtype=np.float32,
        )

    def vec(self, n, scale):
        return np.array([self.next(scale) for _ in range(n)], dtype=np.float32)


def main():
    rng = Lcg(0x5EED_1234)

    # Draw order must match the Rust test line for line.
    x = rng.mat(TOKENS, HIDDEN, 0.5)
    router_w = rng.mat(EXPERTS, HIDDEN, 0.4)
    router_b = rng.vec(EXPERTS, 0.3)

    fused = [rng.mat(2 * INTER, HIDDEN, 0.3) for _ in range(EXPERTS)]
    fused_b = [rng.vec(2 * INTER, 0.2) for _ in range(EXPERTS)]
    down = [rng.mat(HIDDEN, INTER, 0.3) for _ in range(EXPERTS)]
    down_b = [rng.vec(HIDDEN, 0.2) for _ in range(EXPERTS)]

    # ── router: bias, then softmax over the SELECTED top-k logits ──────────
    logits = x @ router_w.T + router_b            # [T, E]
    order = np.argsort(-logits, axis=-1, kind="stable")[:, :TOP_K]
    out = np.zeros((TOKENS, HIDDEN), dtype=np.float32)

    for t in range(TOKENS):
        sel = order[t]
        vals = logits[t, sel]
        vals = vals - vals.max()
        w = np.exp(vals)
        w = w / w.sum()

        for expert, weight in zip(sel, w):
            # gate_up = x @ gate_up_proj[e] + bias, with gate_up_proj = fused.T
            gate_up = x[t] @ fused[expert].T + fused_b[expert]     # [2*inter]
            gate = gate_up[0::2]                                   # even rows
            up = gate_up[1::2]                                     # odd rows

            g = np.minimum(gate, LIMIT)
            u = np.clip(up, -LIMIT, LIMIT)
            glu = g * (1.0 / (1.0 + np.exp(-g * ALPHA)))
            act = (u + 1.0) * glu

            # out = act @ down_proj[e] + bias, with down_proj = down.T
            out[t] += weight * (act @ down[expert].T + down_b[expert])

    print("selected experts per token:", order.tolist())
    print("routing weights per token:")
    for t in range(TOKENS):
        vals = logits[t, order[t]]
        vals = vals - vals.max()
        w = np.exp(vals)
        print("   ", (w / w.sum()).tolist())
    print()
    print("expected output, row-major [TOKENS x HIDDEN]:")
    flat = out.reshape(-1)
    for i in range(0, flat.size, 4):
        chunk = ", ".join(f"{v:.7f}" for v in flat[i : i + 4])
        print(f"        {chunk},")


if __name__ == "__main__":
    main()
