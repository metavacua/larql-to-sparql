"""Reference-forward gate for the MoE latent-axis sparsity result.

The larql measurement rests on an OLMoE forward known to diverge from the HF
reference (k3-funnel §4.7: larql 1.901 bits/char vs reference 0.390). An A/B
cancels engine drift, but it cannot guarantee the observed channel-importance
geometry matches the true model. This repeats only the decisive points through
transformers' own implementation.

Patch point mirrors larql exactly: routing is computed on the UNMASKED hidden
states, then the shared expert input is masked, then the experts run. So the
trained router still selects the same experts with the same weights.

Arms: baseline / magnitude r=0.5 / random r=0.5 / magnitude r=0.375.
The premise under test is the magnitude-vs-random separation, not the absolute.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe import modeling_olmoe as M

MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
CORPUS = "data/gutenberg/frankenstein.txt"
N_TOKENS = 512  # one window; the effect under test is ~7 bits, not marginal
DTYPE = torch.bfloat16

# Set by each arm before the forward.
CFG = {"retention": None, "mode": "magnitude"}


def mask_(z: torch.Tensor) -> torch.Tensor:
    """Zero all but the retained channels of each row of `z` [tokens, hidden]."""
    r = CFG["retention"]
    if r is None:
        return z
    n = z.shape[-1]
    keep = max(1, min(n, int(n * r + 0.999)))
    if CFG["mode"] == "random":
        # Fresh uniform subset per token — the control for magnitude.
        score = torch.rand(z.shape, device=z.device)
    else:
        score = z.abs().float()
    idx = score.topk(keep, dim=-1).indices
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, z.gather(-1, idx))
    return out


def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # Routing on the UNMASKED input — same experts, same weights.
    _, top_k_weights, top_k_index = self.gate(hidden_states)
    hidden_states = mask_(hidden_states)
    final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights).reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states


def main() -> int:
    M.OlmoeSparseMoeBlock.forward = patched_forward

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open(CORPUS, "rb").read(16384).decode("utf-8", errors="ignore")
    ids = tok(text, return_tensors="pt").input_ids[:, :N_TOKENS]
    print(f"scoring {ids.shape[1]} tokens through the HF reference ({DTYPE})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE)
    model.eval()

    def bits_per_token() -> float:
        with torch.no_grad():
            out = model(ids, labels=ids)
        # HF returns mean NLL in nats over the shifted targets.
        return float(out.loss) / 0.6931471805599453

    arms = [
        ("baseline", None, "magnitude"),
        ("magnitude r=0.938", 0.938, "magnitude"),
        ("magnitude r=0.875", 0.875, "magnitude"),
        ("magnitude r=0.750", 0.75, "magnitude"),
        ("magnitude r=0.625", 0.625, "magnitude"),
        ("magnitude r=0.500", 0.5, "magnitude"),
        ("random    r=0.500", 0.5, "random"),
    ]
    base = None
    print(f"\n{'arm':<20} {'bits/token':>12} {'delta':>10} {'rel':>10}")
    for name, r, mode in arms:
        CFG["retention"], CFG["mode"] = r, mode
        torch.manual_seed(0)
        v = bits_per_token()
        if base is None:
            base = v
        print(f"{name:<20} {v:>12.3f} {v - base:>+10.3f} {100*(v-base)/base:>+9.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
