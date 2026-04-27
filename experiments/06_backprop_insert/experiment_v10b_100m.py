#!/usr/bin/env python3
"""
v10b-100M: Attention Transfer from Gemma 3 at 100M Scale

The 20M model (dim=256) failed to produce fluent text — the 4.5x compression
from Gemma's dim=1152 destroyed the geometric structure of attention.

At 100M (dim=512), the compression is only 2.25x. More of Gemma's compositional
structure should survive projection.

Architecture: 95M params
  - dim=512, ffn_dim=2048, 20 layers
  - 8 heads, 4 KV heads, head_dim=64
  - Gemma 3 tokenizer (clamped 32K vocab)

Also augments training data with compositional sentences (~4K samples)
to give attention more diverse patterns to learn from.
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from model import TinyGemma
from synth_data_v2 import (
    build_mixed_corpus, build_knowledge_graph, build_wordnet_pairs,
    GroundTruth, COUNTRIES, CAPITALS, PRESIDENTS, CURRENCIES,
    CODE_SNIPPETS,
)

# ---------------------------------------------------------------------------
# Config — 100M scale
# ---------------------------------------------------------------------------

MODEL_DIM = 512
FFN_DIM = 2048
N_LAYERS = 20
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = MODEL_DIM // N_HEADS  # 64
MAX_SEQ = 128
BATCH_SIZE = 8
LR = 2e-4
SEED = 42
VOCAB = 32000

BASELINE_EPOCHS = 15
FINETUNE_EPOCHS = 5

GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-pt")
OUTPUT_DIR = "results_v10b_100m"


# ---------------------------------------------------------------------------
# Data augmentation: compositional sentences
# ---------------------------------------------------------------------------

def build_compositional_data(seed: int = 42) -> List[GroundTruth]:
    """Generate compositional sentences that cross relation boundaries."""
    rng = random.Random(seed)
    samples = []
    n = min(50, len(COUNTRIES))

    # Cross-relation compositions
    for i in range(n):
        c, cap, pres, cur = COUNTRIES[i], CAPITALS[i], PRESIDENTS[i], CURRENCIES[i]

        templates = [
            f"The capital of {c} is {cap}, and the president is {pres}.",
            f"President {pres} governs {c} from the capital city of {cap}.",
            f"In {c}, the currency is the {cur} and the capital is {cap}.",
            f"{cap} is the capital of {c}, which is led by President {pres}.",
            f"People in {c} use the {cur}. The capital of {c} is {cap}.",
            f"{pres} is the president of {c}. The capital is {cap}.",
            f"The country of {c} has its capital at {cap} and uses the {cur}.",
            f"President {pres} of {c} works in {cap}, the capital city.",
        ]

        for tid, tmpl in enumerate(templates):
            samples.append(GroundTruth(
                text=tmpl, band="knowledge", relation="compositional",
                subject=c, object=f"{cap}/{pres}/{cur}", template_id=tid,
            ))

    # Simple English sentence patterns (teaches compositional generation)
    nouns = ["dog", "cat", "bird", "fish", "tree", "house", "river", "mountain",
             "city", "book", "star", "cloud", "stone", "flower", "leaf"]
    adjs = ["big", "small", "old", "new", "bright", "dark", "quiet", "loud",
            "fast", "slow", "tall", "short", "warm", "cold", "deep"]
    verbs = ["runs", "jumps", "falls", "grows", "shines", "moves", "rests",
             "flows", "rises", "fades", "turns", "holds", "stands", "breaks"]

    for _ in range(200):
        n1, n2 = rng.sample(nouns, 2)
        adj = rng.choice(adjs)
        verb = rng.choice(verbs)
        patterns = [
            f"The {adj} {n1} {verb} near the {n2}.",
            f"A {n1} is {adj} and {verb} quietly.",
            f"The {n1} and the {n2} are both {adj}.",
            f"Every {adj} {n1} eventually {verb}.",
            f"Near the {n2}, a {adj} {n1} {verb}.",
        ]
        tmpl = rng.choice(patterns)
        samples.append(GroundTruth(
            text=tmpl, band="syntax", relation="compositional_syntax",
            subject=n1, object=n2, template_id=0,
        ))

    # Narrative fragments (teaches sentence continuity)
    starts = [
        "Once upon a time, there was a", "In a small village, the", "The old",
        "Long ago, a", "Deep in the forest, a", "By the river, the",
        "On a cold night, the", "At the edge of town, a", "The last",
    ]
    middles = [
        f"{rng.choice(nouns)} {rng.choice(verbs)} beside the {rng.choice(nouns)}"
        for _ in range(20)
    ]
    for start in starts:
        for mid in rng.sample(middles, 5):
            text = f"{start} {mid}."
            samples.append(GroundTruth(
                text=text, band="syntax", relation="narrative",
                subject="narrative", object="fragment", template_id=0,
            ))

    rng.shuffle(samples)
    return samples


def build_augmented_corpus(seed: int = 42):
    """Build larger corpus for 100M model."""
    base_samples, ground_truth = build_mixed_corpus(n_countries=50, seed=seed)
    comp_samples = build_compositional_data(seed=seed)

    all_samples = base_samples + comp_samples
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    ground_truth["counts"]["compositional"] = len(comp_samples)
    ground_truth["counts"]["total"] = len(all_samples)

    return all_samples, ground_truth


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

class ClampedTokenizer:
    def __init__(self, tok, vocab):
        self.tok = tok
        self.vocab = vocab
        self.pad_token_id = tok.pad_token_id or 0

    def encode(self, text, **kwargs):
        ids = self.tok.encode(text, **kwargs)
        return [min(i, self.vocab - 1) for i in ids]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def decode_token(self, tid):
        return self.tok.decode([tid]).strip()


class TokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   max_length=max_len, truncation=True)
            self.encodings.append(ids)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def collate_fn(batch, pad_id=0):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


def make_model(device):
    """Create 100M TinyGemma."""
    return TinyGemma(
        vocab_size=VOCAB, dim=MODEL_DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq=MAX_SEQ,
    ).to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(loader, tokenizer, device, epochs=BASELINE_EPOCHS):
    print(f"\n  Training 100M baseline ({epochs} epochs)...")
    torch.manual_seed(SEED)
    model = make_model(device)
    print(f"  Parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        eloss = 0; n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); n += 1
        scheduler.step()
        avg = eloss / n
        if (epoch + 1) % 3 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.1e} {elapsed:.0f}s")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"  Baseline done: loss={avg:.4f}, {elapsed:.0f}s")
    return model


def evaluate(model, loader, tokenizer, device, label=""):
    model.eval()
    total = 0; n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id)
            total += loss.item(); n += 1
    avg = total / n
    if label:
        print(f"  {label}: {avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# FFN compilation (v6 approach adapted for 100M)
# ---------------------------------------------------------------------------

def compile_ffn(trained_model, samples, tokenizer, device):
    """Compile FFN weights from trained model (v6 approach)."""
    print(f"\n  Compiling FFN from trained model...")
    trained_model.eval()

    # Extract relation templates
    layer_residuals = [None] * N_LAYERS
    layer_gate_acts = [None] * N_LAYERS
    hooks = []

    def make_res_hook(li):
        def hook(m, inp, out): layer_residuals[li] = inp[0].detach()
        return hook

    def make_gate_hook(li):
        def hook(m, inp, out): layer_gate_acts[li] = out.detach()
        return hook

    for i, layer in enumerate(trained_model.layers):
        hooks.append(layer.ffn_norm.register_forward_hook(make_res_hook(i)))
        hooks.append(layer.ffn.gate.register_forward_hook(make_gate_hook(i)))

    rel_samples = defaultdict(list)
    for s in samples:
        rel_samples[s.relation].append(s)

    templates = {}
    with torch.no_grad():
        for relation, samps in rel_samples.items():
            rel_res = [[] for _ in range(N_LAYERS)]
            rel_gate = [[] for _ in range(N_LAYERS)]
            for s in samps[:20]:
                ids = tokenizer.encode(s.text, add_special_tokens=True,
                                       max_length=MAX_SEQ, truncation=True)
                _ = trained_model(torch.tensor([ids], dtype=torch.long, device=device))
                for li in range(N_LAYERS):
                    if layer_residuals[li] is not None:
                        rel_res[li].append(layer_residuals[li].mean(dim=1).squeeze(0).cpu())
                    if layer_gate_acts[li] is not None:
                        rel_gate[li].append(layer_gate_acts[li].mean(dim=1).squeeze(0).cpu())

            templates[relation] = {
                "residuals": [torch.stack(r).mean(0) if r else torch.zeros(MODEL_DIM)
                              for r in rel_res],
                "gate_patterns": [torch.stack(g).mean(0) if g else torch.zeros(FFN_DIM)
                                  for g in rel_gate],
            }

    for h in hooks:
        h.remove()

    # Build compiled model
    torch.manual_seed(SEED)
    compiled = make_model(device)

    # Transfer top-K features per relation
    claimed = [set() for _ in range(N_LAYERS)]
    feats_per_rel = 20
    for relation, tmpl in templates.items():
        for li in range(N_LAYERS):
            activated = F.silu(tmpl["gate_patterns"][li]).abs()
            topk = activated.topk(min(feats_per_rel * 2, FFN_DIM))
            copied = 0
            for fi in topk.indices.tolist():
                if fi in claimed[li] or copied >= feats_per_rel:
                    continue
                with torch.no_grad():
                    compiled.layers[li].ffn.gate.weight.data[fi] = \
                        trained_model.layers[li].ffn.gate.weight.data[fi].clone()
                    compiled.layers[li].ffn.up.weight.data[fi] = \
                        trained_model.layers[li].ffn.up.weight.data[fi].clone()
                    compiled.layers[li].ffn.down.weight.data[:, fi] = \
                        trained_model.layers[li].ffn.down.weight.data[:, fi].clone()
                claimed[li].add(fi)
                copied += 1

    total_claimed = sum(len(c) for c in claimed)
    total_feats = N_LAYERS * FFN_DIM
    print(f"    Transferred {total_claimed}/{total_feats} features "
          f"({100*total_claimed/total_feats:.1f}%)")

    # Align unclaimed gates with residuals
    relations = list(templates.keys())
    for li in range(N_LAYERS):
        unclaimed = [f for f in range(FFN_DIM) if f not in claimed[li]]
        if not unclaimed or not relations:
            continue
        per_rel = max(1, len(unclaimed) // len(relations))
        for ri, rel in enumerate(relations):
            start = ri * per_rel
            end = min(start + per_rel, len(unclaimed))
            residual = templates[rel]["residuals"][li].to(device)
            for fi in unclaimed[start:end]:
                with torch.no_grad():
                    gn = compiled.layers[li].ffn.gate.weight.data[fi].norm()
                    rn = residual.norm()
                    if rn > 0:
                        scaled = residual * (gn / rn)
                        compiled.layers[li].ffn.gate.weight.data[fi] = (
                            0.7 * scaled + 0.3 * compiled.layers[li].ffn.gate.weight.data[fi])

    # Copy embeddings and norms
    with torch.no_grad():
        compiled.embed.weight.data.copy_(trained_model.embed.weight.data)
        compiled.norm.weight.data.copy_(trained_model.norm.weight.data)
        for li in range(N_LAYERS):
            compiled.layers[li].attn_norm.weight.data.copy_(
                trained_model.layers[li].attn_norm.weight.data)
            compiled.layers[li].ffn_norm.weight.data.copy_(
                trained_model.layers[li].ffn_norm.weight.data)

    print(f"  FFN compilation done.")
    return compiled


# ---------------------------------------------------------------------------
# Gemma attention projection (adapted for 512-dim)
# ---------------------------------------------------------------------------

def load_gemma_attention(model_name: str) -> Dict:
    """Load Gemma 3 and extract attention weights."""
    print(f"\n  Loading {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    gemma_dim = config.hidden_size
    gemma_heads = config.num_attention_heads
    gemma_kv_heads = getattr(config, 'num_key_value_heads', gemma_heads)
    gemma_head_dim = getattr(config, 'head_dim', gemma_dim // gemma_heads)
    gemma_layers = config.num_hidden_layers

    print(f"  Gemma: dim={gemma_dim}, heads={gemma_heads}, kv={gemma_kv_heads}, "
          f"hd={gemma_head_dim}, layers={gemma_layers}")

    t0 = time.time()
    gemma = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    result = {
        "config": {
            "dim": gemma_dim, "heads": gemma_heads, "kv_heads": gemma_kv_heads,
            "head_dim": gemma_head_dim, "layers": gemma_layers,
        },
        "embed": gemma.model.embed_tokens.weight.data.float().cpu(),
        "attention_weights": [],
    }

    for li in range(gemma_layers):
        attn = gemma.model.layers[li].self_attn
        result["attention_weights"].append({
            "q_proj": attn.q_proj.weight.data.float().cpu(),
            "k_proj": attn.k_proj.weight.data.float().cpu(),
            "v_proj": attn.v_proj.weight.data.float().cpu(),
            "o_proj": attn.o_proj.weight.data.float().cpu(),
        })

    del gemma
    import gc; gc.collect()
    print(f"  Extracted {gemma_layers} layers, freed model")
    return result


def build_input_projection(gemma_embed, target_dim):
    """SVD of Gemma embeddings → projection to target_dim."""
    print(f"  Building SVD projection: {gemma_embed.shape[1]}→{target_dim}...")
    norms = gemma_embed.norm(dim=1)
    top_idx = norms.topk(min(10000, gemma_embed.shape[0])).indices
    subset = gemma_embed[top_idx]
    U, S, Vh = torch.linalg.svd(subset, full_matrices=False)
    P = Vh[:target_dim, :]
    var_explained = S[:target_dim].sum() / S.sum()
    print(f"    Variance explained: {var_explained:.1%}")
    return P


def compress_heads(W, P_in, n_src, src_hd, n_tgt, tgt_hd):
    """Project weight matrix and compress heads."""
    W_in = W @ P_in.T  # project input dim
    W_heads = W_in.view(n_src, src_hd, -1)

    # Select/duplicate heads
    head_norms = W_heads.flatten(1).norm(dim=1)
    if n_src <= n_tgt:
        selected = list(range(n_src))
        ranked = head_norms.argsort(descending=True).tolist()
        while len(selected) < n_tgt:
            for idx in ranked:
                if len(selected) >= n_tgt:
                    break
                selected.append(idx)
    else:
        selected = head_norms.topk(n_tgt).indices.sort().values.tolist()

    # Compress head dim via SVD
    compressed = []
    for hi in selected:
        Wh = W_heads[hi]  # (src_hd, target_dim)
        if src_hd <= tgt_hd:
            padded = torch.zeros(tgt_hd, Wh.shape[1])
            padded[:src_hd] = Wh
            compressed.append(padded)
        else:
            U, S_vals, Vh = torch.linalg.svd(Wh, full_matrices=False)
            k = min(tgt_hd, len(S_vals))
            compressed.append(U[:tgt_hd, :k] @ torch.diag(S_vals[:k]) @ Vh[:k, :])

    return torch.cat(compressed, dim=0), selected


def project_attention(gemma_info, model, device):
    """Project Gemma attention into 100M model."""
    print(f"\n  Projecting Gemma attention → {MODEL_DIM}-dim...")
    t0 = time.time()

    gc = gemma_info["config"]
    P_in = build_input_projection(gemma_info["embed"], MODEL_DIM)

    # Layer mapping
    gemma_n = gc["layers"]
    if gemma_n <= N_LAYERS:
        layer_map = list(range(gemma_n))
        while len(layer_map) < N_LAYERS:
            layer_map.append(gemma_n - 1)
    else:
        layer_map = [round(i * (gemma_n - 1) / (N_LAYERS - 1)) for i in range(N_LAYERS)]

    print(f"  Layer map: {layer_map}")

    for ti in range(N_LAYERS):
        gi = layer_map[ti]
        gw = gemma_info["attention_weights"][gi]

        W_q, sel_q = compress_heads(
            gw["q_proj"], P_in,
            gc["heads"], gc["head_dim"], N_HEADS, HEAD_DIM)

        W_k, _ = compress_heads(
            gw["k_proj"], P_in,
            gc["kv_heads"], gc["head_dim"], N_KV_HEADS, HEAD_DIM)

        W_v, _ = compress_heads(
            gw["v_proj"], P_in,
            gc["kv_heads"], gc["head_dim"], N_KV_HEADS, HEAD_DIM)

        # O projection: (gemma_dim, heads*hd) → (MODEL_DIM, N_HEADS*HEAD_DIM)
        W_o_full = gw["o_proj"]  # (gemma_dim, heads*hd)
        W_o_out = P_in @ W_o_full  # (MODEL_DIM, heads*hd)
        W_o_heads = W_o_out.view(MODEL_DIM, gc["heads"], gc["head_dim"])

        # Select same Q heads for O
        if gc["heads"] <= N_HEADS:
            o_sel = list(range(gc["heads"]))
            ranked = W_o_heads.flatten(0, 1).view(gc["heads"], -1).norm(dim=1).argsort(descending=True).tolist()
            while len(o_sel) < N_HEADS:
                for idx in ranked:
                    if len(o_sel) >= N_HEADS:
                        break
                    o_sel.append(idx)
        else:
            o_sel = sel_q

        o_selected = W_o_heads[:, o_sel, :]  # (MODEL_DIM, N_HEADS, gemma_hd)

        if gc["head_dim"] <= HEAD_DIM:
            o_comp = torch.zeros(MODEL_DIM, N_HEADS, HEAD_DIM)
            o_comp[:, :, :gc["head_dim"]] = o_selected
        else:
            o_parts = []
            for h in range(N_HEADS):
                Woh = o_selected[:, h, :]
                U, S_vals, Vh = torch.linalg.svd(Woh, full_matrices=False)
                k = min(HEAD_DIM, len(S_vals))
                o_parts.append(U[:, :k] @ torch.diag(S_vals[:k]) @ Vh[:k, :HEAD_DIM])
            o_comp = torch.stack(o_parts, dim=1)

        W_o = o_comp.reshape(MODEL_DIM, N_HEADS * HEAD_DIM)

        # Scale to match TinyGemma activation magnitudes
        for W_src, param in [
            (W_q, model.layers[ti].attn.q_proj.weight),
            (W_k, model.layers[ti].attn.k_proj.weight),
            (W_v, model.layers[ti].attn.v_proj.weight),
            (W_o, model.layers[ti].attn.o_proj.weight),
        ]:
            tgt_norm = param.data.norm().item()
            src_norm = W_src.norm().item()
            scale = tgt_norm / src_norm if src_norm > 0 else 1.0
            with torch.no_grad():
                param.data.copy_((W_src * scale).to(device))

    print(f"  Projection done in {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def freeze_ffn_and_norms(model):
    """Freeze FFN + embeddings + norms. Only attention Q/K/V/O trainable."""
    frozen = 0
    for layer in model.layers:
        for p in layer.ffn.parameters():
            p.requires_grad = False
            frozen += p.numel()
        for p in layer.attn_norm.parameters():
            p.requires_grad = False
            frozen += p.numel()
        for p in layer.ffn_norm.parameters():
            p.requires_grad = False
            frozen += p.numel()
    for p in model.embed.parameters():
        p.requires_grad = False
        frozen += p.numel()
    for p in model.norm.parameters():
        p.requires_grad = False
        frozen += p.numel()
    return frozen


def finetune_attention(model, loader, tokenizer, device, epochs, label=""):
    """Fine-tune only attention weights."""
    n_frozen = freeze_ffn_and_norms(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Fine-tuning: {label}")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR * 0.3, weight_decay=0.01,
    )
    t0 = time.time()
    history = []

    for epoch in range(epochs):
        model.train()
        eloss = 0; n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); n += 1
        avg = eloss / n
        history.append({"epoch": epoch + 1, "loss": avg})
        print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()

    return history


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=50, temperature=0.8,
             top_k=40, device=None):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=MAX_SEQ,
                           truncation=True)
    generated = list(ids)

    for _ in range(max_new):
        if len(generated) >= MAX_SEQ:
            break
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            vals, indices = next_logits.topk(top_k)
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(0, indices, vals)
            next_logits = mask

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)

        if next_token in (0, 1):
            break

    return tokenizer.decode(generated)


def generation_test(model, tokenizer, device, label=""):
    print(f"\n  Generation test: {label}")
    print(f"  {'─'*70}")

    prompts = [
        "The capital of Freedonia is",
        "The president of Sylvania is",
        "The currency of Genovia is the",
        "Freedonia is a country whose capital",
        "President Albrecht governs Freedonia from",
        "A dog is a type of",
        "The opposite of big is",
        "The big dog runs near the",
        "In the city of",
        "Once upon a time, there was a",
        "The most important thing about",
        "def add(a, b):",
    ]

    results = []
    for prompt in prompts:
        for temp in [0.1, 0.7]:
            text = generate(model, tokenizer, prompt, max_new=40,
                          temperature=temp, device=device)
            gen_part = text[len(prompt):] if text.startswith(prompt) else text
            temp_label = "greedy" if temp <= 0.2 else f"t={temp}"
            print(f"  [{temp_label:>7}] {prompt}  →  {gen_part[:80]}")
            results.append({"prompt": prompt, "temperature": temp, "output": text})

    print(f"  {'─'*70}")
    return results


def score_fluency(results):
    """Score generation quality."""
    scores = []
    for r in results:
        gen = r["output"][len(r["prompt"]):].strip() if r["output"].startswith(r["prompt"]) else r["output"].strip()
        words = gen.split()
        score = 0

        if len(words) > 0: score += 1
        if len(words) >= 3: score += 1

        if words:
            uniq = len(set(words)) / len(words)
            if uniq > 0.5: score += 1

        common = {"the", "a", "is", "of", "in", "to", "and", "it", "for", "that",
                  "was", "on", "are", "with", "as", "at", "be", "this", "have", "from"}
        gen_words = set(w.lower().strip(".,!?;:") for w in words)
        if len(gen_words & common) >= 1: score += 1

        garbage = sum(1 for w in words if len(w) > 20 or not any(c.isalpha() for c in w))
        if garbage == 0: score += 1

        scores.append(score)

    total = sum(scores)
    maximum = len(scores) * 5
    return total, maximum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  v10b-100M: ATTENTION TRANSFER AT 100M SCALE")
    print("  dim=512, 20 layers, 95M params")
    print("  Gemma 1152→512 = 2.25x compression (vs 4.5x at 20M)")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n  Device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"\n  Device: CPU")

    # Tokenizer + data
    print("  Loading tokenizer...")
    raw_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if raw_tok.pad_token_id is None:
        raw_tok.pad_token_id = 0
    tokenizer = ClampedTokenizer(raw_tok, VOCAB)

    print("  Building augmented corpus...")
    samples, ground_truth = build_augmented_corpus(seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    dataset = TokenDataset([s.text for s in samples], tokenizer, MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Train 100M baseline + compile FFN
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Train 100M baseline + compile FFN")
    print(f"{'='*70}")

    baseline = train_baseline(loader, tokenizer, device)
    baseline_loss = evaluate(baseline, loader, tokenizer, device, "Baseline")

    print(f"\n  Baseline generation:")
    baseline_gen = generation_test(baseline, tokenizer, device, "BASELINE (100M)")
    base_score, base_max = score_fluency(baseline_gen)

    compiled = compile_ffn(baseline, samples, tokenizer, device)

    # Save baseline norms/embeds for later random-attn comparison, then free it
    baseline_state = {
        "embed": baseline.embed.weight.data.cpu().clone(),
        "norm": baseline.norm.weight.data.cpu().clone(),
        "attn_norms": [baseline.layers[li].attn_norm.weight.data.cpu().clone()
                       for li in range(N_LAYERS)],
        "ffn_norms": [baseline.layers[li].ffn_norm.weight.data.cpu().clone()
                      for li in range(N_LAYERS)],
    }
    del baseline
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"  Freed baseline model to save memory")

    compiled_pre = evaluate(compiled, loader, tokenizer, device, "Compiled FFN (pre-transfer)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Project Gemma attention
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Project Gemma 3 attention → 512-dim")
    print(f"{'='*70}")

    # Move compiled to CPU during Gemma load to save GPU memory
    compiled.cpu()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gemma_info = load_gemma_attention(GEMMA_MODEL)
    transfer_model = project_attention(gemma_info, compiled, torch.device("cpu"))
    del gemma_info
    gc.collect()

    # Move back to device
    transfer_model.to(device)

    transfer_pre = evaluate(transfer_model, loader, tokenizer, device,
                           "After projection (pre fine-tune)")

    print(f"\n  Pre-finetune generation:")
    pre_gen = generation_test(transfer_model, tokenizer, device, "PROJECTED (raw)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Fine-tune projected attention
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Fine-tune projected attention ({FINETUNE_EPOCHS} epochs)")
    print(f"{'='*70}")

    ft_history = finetune_attention(
        transfer_model, loader, tokenizer, device,
        epochs=FINETUNE_EPOCHS, label="GEMMA TRANSFER",
    )
    transfer_post = evaluate(transfer_model, loader, tokenizer, device, "After fine-tune")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Fluency test
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 4: FLUENCY TEST")
    print(f"{'='*70}")

    post_gen = generation_test(transfer_model, tokenizer, device,
                              "TRANSFERRED (after fine-tune)")
    post_score, post_max = score_fluency(post_gen)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: Random attention comparison
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PHASE 5: Random attention comparison")
    print(f"{'='*70}")

    # Move transfer model to CPU to free GPU for random model
    transfer_model.cpu()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    torch.manual_seed(SEED + 1)
    random_model = make_model(device)

    # Copy compiled FFN + saved baseline embeddings/norms
    with torch.no_grad():
        random_model.embed.weight.data.copy_(baseline_state["embed"].to(device))
        random_model.norm.weight.data.copy_(baseline_state["norm"].to(device))
        for li in range(N_LAYERS):
            random_model.layers[li].ffn.gate.weight.data.copy_(
                compiled.layers[li].ffn.gate.weight.data)
            random_model.layers[li].ffn.up.weight.data.copy_(
                compiled.layers[li].ffn.up.weight.data)
            random_model.layers[li].ffn.down.weight.data.copy_(
                compiled.layers[li].ffn.down.weight.data)
            random_model.layers[li].attn_norm.weight.data.copy_(
                baseline_state["attn_norms"][li].to(device))
            random_model.layers[li].ffn_norm.weight.data.copy_(
                baseline_state["ffn_norms"][li].to(device))

    random_pre = evaluate(random_model, loader, tokenizer, device,
                         "Random attn + compiled FFN (pre)")

    rand_history = finetune_attention(
        random_model, loader, tokenizer, device,
        epochs=FINETUNE_EPOCHS, label="RANDOM ATTENTION",
    )
    random_post = evaluate(random_model, loader, tokenizer, device,
                          "Random attn + compiled FFN (post)")

    rand_gen = generation_test(random_model, tokenizer, device,
                              "RANDOM ATTENTION (after fine-tune)")
    rand_score, rand_max = score_fluency(rand_gen)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY: 100M ATTENTION TRANSFER")
    print(f"{'='*70}")

    print(f"\n  {'Configuration':<55} {'Loss':>8} {'Fluency':>10}")
    print(f"  {'─'*75}")
    print(f"  {'Baseline (100M, full training)':<55} {baseline_loss:>8.4f} "
          f"{base_score}/{base_max} ({base_score/base_max:.0%})")
    print(f"  {'Compiled FFN + Gemma attn (pre-tune)':<55} "
          f"{transfer_pre:>8.4f}")
    print(f"  {'Compiled FFN + Gemma attn (post-tune)':<55} "
          f"{transfer_post:>8.4f} {post_score}/{post_max} ({post_score/post_max:.0%})")
    print(f"  {'Compiled FFN + random attn (post-tune)':<55} "
          f"{random_post:>8.4f} {rand_score}/{rand_max} ({rand_score/rand_max:.0%})")

    print(f"\n  Fine-tune curves:")
    print(f"  {'Epoch':>6} {'Transfer':>11} {'Random':>11}")
    print(f"  {'─'*30}")
    for i in range(FINETUNE_EPOCHS):
        tl = ft_history[i]["loss"] if i < len(ft_history) else 0
        rl = rand_history[i]["loss"] if i < len(rand_history) else 0
        print(f"  {i+1:>6} {tl:>11.4f} {rl:>11.4f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT: 100M SCALE")
    print(f"{'='*70}")

    tf = post_score / post_max if post_max else 0
    rf = rand_score / rand_max if rand_max else 0
    bf = base_score / base_max if base_max else 0

    # Compare transfer vs random vs baseline
    if tf > rf + 0.05:
        print(f"\n  ✓ GEMMA TRANSFER HELPS: {tf:.0%} vs random {rf:.0%}")
        print(f"    Compositional structure survives 2.25x projection.")
    elif abs(tf - rf) <= 0.05:
        print(f"\n  ~ TRANSFER ≈ RANDOM: {tf:.0%} vs {rf:.0%}")
        print(f"    Projection still loses too much structure.")
    else:
        print(f"\n  ✗ RANDOM BEATS TRANSFER: random {rf:.0%} vs transfer {tf:.0%}")

    if tf >= 0.6 or bf >= 0.6:
        print(f"\n  ✓ 100M CAN COMPOSE SENTENCES")
        if tf >= bf - 0.05:
            print(f"    Transfer matches baseline fluency.")
        else:
            print(f"    Baseline more fluent ({bf:.0%} vs {tf:.0%})")
            print(f"    → More fine-tune epochs or curriculum needed")
    else:
        print(f"\n  ✗ 100M STILL NOT FLUENT")
        print(f"    Baseline: {bf:.0%}, Transfer: {tf:.0%}, Random: {rf:.0%}")
        print(f"    → Skip to Gemma 3-4B validation (the thesis doesn't need")
        print(f"      a small model to talk — it needs to prove scaling)")

    print(f"\n  Key insight: dim compression")
    print(f"    20M (256-dim): 4.5x compression → transfer failed")
    print(f"    100M (512-dim): 2.25x compression → {'improved' if tf > 0.5 else 'still insufficient'}")
    if tf < 0.5:
        print(f"    → head_dim compression (256→64 = 4x) may be the bottleneck")

    # Save
    results = {
        "model_params": "95M",
        "gemma_model": GEMMA_MODEL,
        "baseline_loss": baseline_loss,
        "transfer_pre": transfer_pre,
        "transfer_post": transfer_post,
        "random_pre": random_pre,
        "random_post": random_post,
        "baseline_fluency": bf,
        "transfer_fluency": tf,
        "random_fluency": rf,
        "finetune_curve": ft_history,
        "random_curve": rand_history,
        "samples": {
            "baseline": baseline_gen[:5],
            "post_transfer": post_gen[:5],
            "random": rand_gen[:5],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
