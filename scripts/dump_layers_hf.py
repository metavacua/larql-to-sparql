#!/usr/bin/env python3
"""HF/PyTorch reference side of `larql shannon layer-dump`.

Gate B at rungs R1/R2 is "a layer-by-layer f32 diff against the local
reference implementation" (docs/k3-funnel.md §4). This writes the reference
half of that diff in the same on-disk format the Rust side writes, so
`larql shannon layer-diff` can compare them.

    larql shannon layer-dump MODEL --corpus c.txt --bytes 384 --out dumps/rust
    python scripts/dump_layers_hf.py MODEL --ref dumps/rust --out dumps/hf
    larql shannon layer-diff --a dumps/rust --b dumps/hf

**Token ids come from the Rust manifest, not from re-tokenising.** §4.7.7
withdrew a whole verdict because two engines scored different text; a
tokenizer is part of the fixture, so only one side is allowed to choose it.

Capture convention, matching `forward_hidden_all_layers`:
  plane 000 -- the residual *entering* layer 0 (embedding output, including
               any architecture-specific embedding scale)
  plane i+1 -- the residual leaving layer i

Layer 0's input is captured with a pre-hook rather than by reading
`embed_tokens`, so an architecture that scales its embeddings is captured
after the scale, the same point the Rust side records.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Must match `MANIFEST_NAME` / `PLANE_DTYPE` / `plane_name` in
# crates/larql-cli/src/commands/primary/shannon_trace/dump.rs.
MANIFEST_NAME = "layer_dump.json"
PLANE_DTYPE = "f32-le"
ENGINE_HF_F32 = "hf-torch-f32"


def plane_name(idx: int) -> str:
    return f"layer_{idx:03d}.f32"


def write_plane(path: Path, tensor: torch.Tensor) -> None:
    """Write `[seq, hidden]` as little-endian f32, row-major."""
    flat = tensor.detach().to(torch.float32).contiguous().view(-1).tolist()
    path.write_bytes(struct.pack(f"<{len(flat)}f", *flat))


def find_layers(model):
    """The decoder block list, across the layouts transformers uses."""
    for path in (
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "model.decoder.layers",
        # Multimodal wrappers keep the text stack under `language_model`
        # (e.g. MuseGlimmerForConditionalGeneration).
        "model.language_model.layers",
        "model.model.language_model.layers",
    ):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj
    raise SystemExit("could not locate the decoder layer list on this model")


def cast_everything_to_f32(model) -> int:
    """Force every parameter and buffer to f32, returning how many were cast.

    `dtype=torch.float32` is a *request*, not a guarantee. GPT-OSS's MXFP4
    experts are materialised by `convert_moe_packed_tensors`, which
    dequantises to bf16 whatever dtype was asked for; the f32 activations then
    meet bf16 weights inside `_grouped_mm_fallback`'s `torch.mm` and the
    forward dies with `float != c10::BFloat16`. docs/k3-funnel.md §4.7.6
    recorded that crash as a hole in the ladder's premise -- a reference you
    cannot run is not a reference. This is the hole, closed.

    Casting after load rather than patching the dequantiser keeps the fix
    outside `transformers` internals, and it reports a count so a silent no-op
    cannot be mistaken for "the model was already f32".
    """
    cast = 0
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.dtype.is_floating_point and param.dtype != torch.float32:
                setattr(module, name, torch.nn.Parameter(param.data.to(torch.float32),
                                                         requires_grad=False))
                cast += 1
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.dtype.is_floating_point and buf.dtype != torch.float32:
                module.register_buffer(name, buf.to(torch.float32), persistent=False)
                cast += 1
    return cast


def attach_router_trace(layers, sink: list) -> list:
    """Record each MoE layer's expert selection and the margin behind it.

    A layer-diff residual that grows with depth on a mixture-of-experts model
    has two very different explanations — accumulated arithmetic, or the two
    engines *selecting different experts* — and they are not distinguishable
    from the residual alone. This captures the selections so the question can
    be settled rather than argued (`docs/k3-funnel.md` §4.9.3).

    `margin` is the gap between the last selected router logit and the best
    unselected one. If a disagreement is a tie-break, its margin is ~0; if
    disagreements appear at wide margins, the router itself is wrong and the
    tie-break story is false.
    """
    handles = []

    def make_hook(idx: int):
        def hook(_module, _args, output):
            # GptOssTopKRouter returns (router_logits, router_scores, indices).
            logits, _scores, indices = output
            logits = logits.detach().float()
            indices = indices.detach()
            top_k = indices.shape[-1]
            # Sorted logits let us read both the k-th selected and the best
            # unselected value without re-running topk.
            ordered, _ = torch.sort(logits, dim=-1, descending=True)
            margin = ordered[:, top_k - 1] - ordered[:, top_k]
            sink.append(
                {
                    "layer": idx,
                    "experts": indices.tolist(),
                    "margin": margin.tolist(),
                }
            )

        return hook

    for i, layer in enumerate(layers):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is not None:
            handles.append(router.register_forward_hook(make_hook(i)))
    return handles


def dump(
    model_id: str,
    ref_dir: Path,
    out_dir: Path,
    device: str,
    router_trace: Path | None = None,
) -> None:
    ref = json.loads((ref_dir / MANIFEST_NAME).read_text())
    token_ids = ref["token_ids"]
    print(f"fixture: {len(token_ids)} tokens from {ref_dir / MANIFEST_NAME}", file=sys.stderr)

    print(f"loading {model_id} on {device} in float32...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    cast = cast_everything_to_f32(model)
    print(f"cast {cast} non-f32 tensors to f32", file=sys.stderr)

    model.to(torch.device(device))

    stragglers = {p.dtype for p in model.parameters() if p.dtype.is_floating_point}
    if stragglers != {torch.float32}:
        raise SystemExit(f"model is not uniformly f32 after cast: {stragglers}")

    layers = find_layers(model)
    captures: list[torch.Tensor | None] = [None] * (len(layers) + 1)

    def first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise SystemExit(f"unexpected layer output type: {type(value)}")

    def make_pre_hook():
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states") if kwargs else None
            if hidden is None:
                hidden = args[0]
            captures[0] = hidden[0].detach().clone()

        return hook

    def make_post_hook(idx: int):
        def hook(_module, _args, output):
            captures[idx + 1] = first_tensor(output)[0].detach().clone()

        return hook

    handles = [layers[0].register_forward_pre_hook(make_pre_hook(), with_kwargs=True)]
    handles += [layers[i].register_forward_hook(make_post_hook(i)) for i in range(len(layers))]

    routing: list = []
    if router_trace is not None:
        handles += attach_router_trace(layers, routing)
        if not routing and not handles:
            raise SystemExit("no MoE routers found — --router-trace needs a mixture-of-experts model")

    with torch.no_grad():
        model(torch.tensor([token_ids], device=torch.device(device)))
    for h in handles:
        h.remove()

    missing = [i for i, c in enumerate(captures) if c is None]
    if missing:
        raise SystemExit(f"captures {missing} never fired — hook points are wrong for this model")

    out_dir.mkdir(parents=True, exist_ok=True)
    planes = []
    for idx, tensor in enumerate(captures):
        name = plane_name(idx)
        write_plane(out_dir / name, tensor)
        planes.append(name)

    seq_len, hidden_size = captures[0].shape
    manifest = {
        "engine": ENGINE_HF_F32,
        "model": model_id,
        "num_layers": len(layers),
        "seq_len": int(seq_len),
        "hidden_size": int(hidden_size),
        "token_ids": token_ids,
        "planes": planes,
        "dtype": PLANE_DTYPE,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(planes)} planes of [{seq_len}, {hidden_size}] to {out_dir}")

    if router_trace is not None:
        if not routing:
            raise SystemExit("--router-trace requested but no router hook fired")
        routing.sort(key=lambda r: r["layer"])
        with router_trace.open("w") as fh:
            for rec in routing:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {len(routing)} router records to {router_trace}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model path or HuggingFace model ID")
    ap.add_argument("--ref", type=Path, required=True,
                    help="a `larql shannon layer-dump` directory; its token ids define the fixture")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--device", default="cpu",
                    help="torch device (cpu recommended: mps lacks f32 parity for some ops)")
    ap.add_argument("--router-trace", type=Path, default=None, metavar="FILE",
                    help="also write per-layer expert selections + tie margins as JSON lines "
                         "(MoE models only); pairs with LARQL_MOE_ROUTE_TRACE on the Rust side")
    args = ap.parse_args()
    dump(args.model, args.ref, args.out, args.device, args.router_trace)


if __name__ == "__main__":
    main()
