#!/usr/bin/env python3
"""Upstream golden trace for Muse-Glimmer (VINDEX3 G5b-3 step 1).

Captures the *upstream* `transformers` forward over a tiny fixture so the
VINDEX3 plan executor can be diffed against it layer by layer. Upstream is
the semantic authority here: everything the container had to *judge* —
where `qk_scale_factor` multiplies, which layers are NoPE, what the
attention gate multiplies and when, which epsilon each of the four norms
uses, the order of the head's multiplier and softcap — is stated by the
modeling code and by nothing else.

    python scripts/capture_glimmer_oracle.py ~/chris-models/Muse-Glimmer-30B \\
        --out dumps/glimmer-oracle --prompt "The capital of France is"

Writes the same on-disk format as `larql shannon layer-dump`, so the
existing `larql shannon layer-diff` produces the L0..L51 table without a
new comparator:

  plane 000   the residual entering layer 0 (post embedding RMS norm --
              `MuseGlimmerTextNormedEmbedding` normalises weightlessly, and
              a pre-hook on layer 0 captures after that, not before)
  plane i+1   the residual leaving layer i

plus `final_norm.f32` (after `model.norm`) and `logits.f32` (after the
`output_multiplier` * tanh-softcap head), which the layer table does not
cover but the executor also produces.

**Two deliberate constraints, both load-bearing.**

*Token ids are recorded, never re-derived.* The prompt is tokenised at most
once, here, and every later consumer reads `token_ids` from the manifest.
A tokenizer is part of the fixture; only one side may choose it.

*The compute dtype is in the engine tag.* Weights are bf16 by default
because f32 for a 30B model is ~120 GB and will not fit alongside the OS in
128 GB. bf16 is adequate for what this trace is *for* -- a misplaced scale,
a wrong NoPE layer, a gate on the wrong operand, or a swapped epsilon all
diverge O(1), far above bf16 noise -- but it cannot certify agreement below
roughly bf16 epsilon accumulated over 52 layers. Recording the dtype in
`engine` means a bf16 trace can never be quietly read as an f32 one. Pass
`--dtype float32` (with disk offload) when a tighter anchor is wanted.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_layers_hf import (  # noqa: E402
    MANIFEST_NAME,
    PLANE_DTYPE,
    find_layers,
    plane_name,
)

# Named so the dtype travels with the trace. `layer-diff` prints it, so a
# margin can always be read against the arithmetic that produced it.
ENGINE_TEMPLATE = "hf-torch-{dtype}-upstream"

FINAL_NORM_PLANE = "final_norm.f32"
LOGITS_PLANE = "logits.f32"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Eager, not sdpa: the reference is `eager_attention_forward`, and a fused
# kernel is free to reassociate. A golden trace should be the arithmetic the
# modeling code spells out.
ATTN_IMPLEMENTATION = "eager"


def read_token_file(path: Path) -> list[int]:
    """Token ids from a comma- or whitespace-separated file.

    Long-context fixtures run to thousands of ids, which is past the
    point where passing them as argv is either readable or portable —
    zsh does not word-split unquoted expansions, so a shell-built list
    silently arrives as one argument.
    """
    text = path.read_text().replace(",", " ")
    return [int(tok) for tok in text.split()]


def resolve_token_ids(model_path: str, prompt: str | None, token_ids: list[int] | None) -> list[int]:
    """The fixture's token ids, chosen exactly once.

    Explicit ids win over a prompt: once a trace exists, later captures must
    be able to reproduce its fixture without depending on a tokenizer
    revision.
    """
    if token_ids:
        return token_ids
    if prompt is None:
        raise SystemExit("need either --prompt or --token-ids")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ids = tokenizer(prompt, return_tensors=None)["input_ids"]
    print(f"tokenised {prompt!r} -> {ids}", file=sys.stderr)
    return [int(i) for i in ids]


def load_model(model_path: str, torch_dtype):
    """Load whichever auto-class actually owns the head.

    Muse-Glimmer is a multimodal checkpoint: it registers under
    image-text-to-text, not causal-LM, and `MuseGlimmerForConditionalGeneration`
    is where `output_multiplier` and the tanh softcap are applied. Loading a
    bare text model would silently drop both and produce logits that differ
    from the deployed ones by more than a scale.
    """
    errors = []
    for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            return auto_cls.from_pretrained(
                model_path, dtype=torch_dtype, attn_implementation=ATTN_IMPLEMENTATION
            )
        except (ValueError, KeyError) as exc:
            errors.append(f"{auto_cls.__name__}: {exc}")
    raise SystemExit("no auto-class could load this checkpoint:\n  " + "\n  ".join(errors))


def write_flat(path: Path, tensor: torch.Tensor) -> None:
    """Write any shape as little-endian f32, row-major.

    Goes through numpy rather than `struct.pack` on a Python list. The
    list form costs ~28 bytes per element, so a long-context capture
    would need tens of GB of interpreter objects to write a few GB of
    floats — at 2052 positions the logits alone are 414M values.
    """
    array = tensor.detach().to(torch.float32).contiguous().view(-1).numpy()
    array.astype("<f4", copy=False).tofile(path)


def find_final_norm(model):
    """The norm applied after the last block, before the head."""
    for path in ("model.norm", "model.model.norm", "model.language_model.norm",
                 "model.model.language_model.norm"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise SystemExit("could not locate the final norm on this model")


def capture(model_path: str, out_dir: Path, prompt: str | None,
            token_ids: list[int] | None, dtype_name: str, device: str) -> None:
    ids = resolve_token_ids(model_path, prompt, token_ids)
    if len(ids) < 2:
        raise SystemExit(f"fixture needs >= 2 tokens to exercise RoPE, got {len(ids)}")

    torch_dtype = DTYPES[dtype_name]
    print(f"loading {model_path} on {device} in {dtype_name} "
          f"(attn={ATTN_IMPLEMENTATION})...", file=sys.stderr)
    model = load_model(model_path, torch_dtype)
    print(f"loaded {type(model).__name__}", file=sys.stderr)
    model.eval()
    model.to(torch.device(device))

    dtypes = {p.dtype for p in model.parameters() if p.dtype.is_floating_point}
    if dtypes != {torch_dtype}:
        raise SystemExit(f"model is not uniformly {dtype_name}: {dtypes}")

    layers = find_layers(model)
    print(f"found {len(layers)} decoder layers", file=sys.stderr)
    captures: list[torch.Tensor | None] = [None] * (len(layers) + 1)
    final_norm_out: list[torch.Tensor] = []

    def first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise SystemExit(f"unexpected layer output type: {type(value)}")

    def pre_hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None:
            hidden = args[0]
        captures[0] = hidden[0].detach().clone()

    def make_post_hook(idx: int):
        def hook(_module, _args, output):
            captures[idx + 1] = first_tensor(output)[0].detach().clone()

        return hook

    def final_norm_hook(_module, _args, output):
        final_norm_out.append(first_tensor(output)[0].detach().clone())

    handles = [layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    handles += [layers[i].register_forward_hook(make_post_hook(i)) for i in range(len(layers))]
    handles.append(find_final_norm(model).register_forward_hook(final_norm_hook))

    print(f"forward over {len(ids)} tokens...", file=sys.stderr)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=torch.device(device)))
    for h in handles:
        h.remove()

    missing = [i for i, c in enumerate(captures) if c is None]
    if missing:
        raise SystemExit(f"captures {missing} never fired — hook points are wrong for this model")
    if not final_norm_out:
        raise SystemExit("final norm hook never fired")

    out_dir.mkdir(parents=True, exist_ok=True)
    planes = []
    for idx, tensor in enumerate(captures):
        name = plane_name(idx)
        write_flat(out_dir / name, tensor)
        planes.append(name)

    write_flat(out_dir / FINAL_NORM_PLANE, final_norm_out[-1])
    logits = out.logits[0]
    write_flat(out_dir / LOGITS_PLANE, logits)

    seq_len, hidden_size = captures[0].shape
    manifest = {
        "engine": ENGINE_TEMPLATE.format(dtype=dtype_name),
        "model": model_path,
        "num_layers": len(layers),
        "seq_len": int(seq_len),
        "hidden_size": int(hidden_size),
        "token_ids": ids,
        "planes": planes,
        "dtype": PLANE_DTYPE,
        # Extras beyond the layer-diff schema; unknown fields are ignored by
        # the Rust reader, so one manifest still describes the whole capture.
        "compute_dtype": dtype_name,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "final_norm_plane": FINAL_NORM_PLANE,
        "logits_plane": LOGITS_PLANE,
        "logits_shape": list(logits.shape),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(planes)} planes of [{seq_len}, {hidden_size}] "
          f"+ final norm + logits {tuple(logits.shape)} to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="checkpoint directory or HuggingFace model id")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--prompt", default=None, help="fixture text; tokenised once, then recorded")
    ap.add_argument("--token-ids", type=int, nargs="+", default=None,
                    help="explicit fixture ids; reproduces a capture without a tokenizer")
    ap.add_argument("--tokens-file", type=Path, default=None,
                    help="file of comma/whitespace-separated ids, for long fixtures")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16",
                    help="compute dtype (default bfloat16; float32 needs ~120GB for a 30B model)")
    ap.add_argument("--device", default="cpu", help="torch device")
    args = ap.parse_args()
    token_ids = args.token_ids
    if args.tokens_file is not None:
        token_ids = read_token_file(args.tokens_file)
    capture(args.model, args.out, args.prompt, token_ids, args.dtype, args.device)


if __name__ == "__main__":
    main()
