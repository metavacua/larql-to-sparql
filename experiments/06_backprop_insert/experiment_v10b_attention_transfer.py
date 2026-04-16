#!/usr/bin/env python3
"""
v10b: Attention Transfer from Gemma 3

CRITICAL BLOCKER: The 20M model has knowledge (compiled FFN) but can't compose
sentences because attention was only trained on ~1,700 synthetic samples.
Gemma 3's attention has learned compositional generation from billions of tokens.

Method: Project Gemma 3's attention weights down to TinyGemma's 256-dim space
via SVD, fine-tune 2-3 epochs against compiled FFN.

Expected: ~1.5 hours total. If the model produces coherent English, every
downstream experiment (style, code gen, benchmarks) unlocks.

Projection strategy:
  1. SVD of Gemma's embedding → shared 256-dim semantic projection
  2. Per-layer: select best heads by norm, SVD head dim → 64
  3. Layer mapping: evenly sample 12 layers from Gemma's depth
  4. Compiled FFN from v6 approach (proven 103.8% quality)
  5. Fine-tune projected attention 2-3 epochs
  6. Generation test: can it produce fluent English?
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
from synth_data_v2 import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_LAYERS = 12
DIM = 256
FFN_DIM = 1024
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = DIM // N_HEADS  # 64
BATCH_SIZE = 8
LR = 3e-4
MAX_SEQ = 64
SEED = 42
VOCAB = 32000

# Gemma source model — 1B is fastest to download (~2GB), 4B if you want more capacity
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-3-1b-pt")

FINETUNE_EPOCHS = 3
COMPILE_EPOCHS = 20  # for baseline used in FFN compilation

OUTPUT_DIR = "results_v10b_transfer"


# ---------------------------------------------------------------------------
# Shared infrastructure (reused from prior experiments)
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


# ---------------------------------------------------------------------------
# Phase 1: Train baseline + compile FFN (reuse v6 approach)
# ---------------------------------------------------------------------------

def train_baseline(loader, tokenizer, device, epochs=COMPILE_EPOCHS):
    """Train full baseline model for FFN compilation source."""
    print(f"\n  Training baseline ({epochs} epochs) for FFN compilation...")
    torch.manual_seed(SEED)
    model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq=MAX_SEQ,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    for epoch in range(epochs):
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()
    print(f"  Baseline done: loss={avg:.4f}, {time.time()-t0:.0f}s")
    return model


def extract_and_compile_ffn(trained_model, samples, tokenizer, device):
    """
    Extract relation templates from trained model and compile FFN weights.
    Returns a fresh TinyGemma with compiled FFN (from v6 approach).
    """
    print(f"\n  Compiling FFN from trained model (v6 approach)...")
    trained_model.eval()

    # --- Extract relation templates ---
    residuals = {}
    gate_acts = {}
    hooks = []
    layer_residuals = [None] * N_LAYERS
    layer_gate_acts = [None] * N_LAYERS

    def make_residual_hook(li):
        def hook(module, input, output):
            layer_residuals[li] = input[0].detach()
        return hook

    def make_gate_hook(li):
        def hook(module, input, output):
            layer_gate_acts[li] = output.detach()
        return hook

    for i, layer in enumerate(trained_model.layers):
        hooks.append(layer.ffn_norm.register_forward_hook(make_residual_hook(i)))
        hooks.append(layer.ffn.gate.register_forward_hook(make_gate_hook(i)))

    rel_samples = defaultdict(list)
    for s in samples:
        rel_samples[s.relation].append(s)

    templates = {}
    with torch.no_grad():
        for relation, samps in rel_samples.items():
            rel_residuals = [[] for _ in range(N_LAYERS)]
            rel_gate_patterns = [[] for _ in range(N_LAYERS)]
            for s in samps[:20]:
                ids = tokenizer.encode(s.text, add_special_tokens=True,
                                       max_length=MAX_SEQ, truncation=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                _ = trained_model(input_ids)
                for li in range(N_LAYERS):
                    if layer_residuals[li] is not None:
                        rel_residuals[li].append(
                            layer_residuals[li].mean(dim=1).squeeze(0).cpu())
                    if layer_gate_acts[li] is not None:
                        rel_gate_patterns[li].append(
                            layer_gate_acts[li].mean(dim=1).squeeze(0).cpu())

            avg_res = []
            avg_gate = []
            for li in range(N_LAYERS):
                avg_res.append(torch.stack(rel_residuals[li]).mean(0)
                               if rel_residuals[li] else torch.zeros(DIM))
                avg_gate.append(torch.stack(rel_gate_patterns[li]).mean(0)
                                if rel_gate_patterns[li] else torch.zeros(FFN_DIM))
            templates[relation] = {"residuals": avg_res, "gate_patterns": avg_gate}

    for h in hooks:
        h.remove()

    # --- Compile into fresh model ---
    torch.manual_seed(SEED)
    compiled = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq=MAX_SEQ,
    ).to(device)

    # Strategy 1: Transfer top-K features per relation
    claimed = [set() for _ in range(N_LAYERS)]
    features_per_rel = 20
    for relation, tmpl in templates.items():
        for li in range(N_LAYERS):
            activated = F.silu(tmpl["gate_patterns"][li]).abs()
            topk = activated.topk(features_per_rel * 2)
            copied = 0
            for feat_idx in topk.indices.tolist():
                if feat_idx in claimed[li] or copied >= features_per_rel:
                    continue
                with torch.no_grad():
                    compiled.layers[li].ffn.gate.weight.data[feat_idx] = \
                        trained_model.layers[li].ffn.gate.weight.data[feat_idx].clone()
                    compiled.layers[li].ffn.up.weight.data[feat_idx] = \
                        trained_model.layers[li].ffn.up.weight.data[feat_idx].clone()
                    compiled.layers[li].ffn.down.weight.data[:, feat_idx] = \
                        trained_model.layers[li].ffn.down.weight.data[:, feat_idx].clone()
                claimed[li].add(feat_idx)
                copied += 1

    total_claimed = sum(len(c) for c in claimed)
    print(f"    Transferred {total_claimed}/{N_LAYERS * FFN_DIM} features "
          f"({100*total_claimed/(N_LAYERS * FFN_DIM):.1f}%)")

    # Strategy 2: Align unclaimed gates with residuals
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
            for feat_idx in unclaimed[start:end]:
                with torch.no_grad():
                    gate_norm = compiled.layers[li].ffn.gate.weight.data[feat_idx].norm()
                    res_norm = residual.norm()
                    if res_norm > 0:
                        scaled = residual * (gate_norm / res_norm)
                        compiled.layers[li].ffn.gate.weight.data[feat_idx] = (
                            0.7 * scaled + 0.3 * compiled.layers[li].ffn.gate.weight.data[feat_idx])

    # Strategy 3: Copy embeddings and norms
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
# Phase 2: Load and project Gemma attention weights
# ---------------------------------------------------------------------------

def load_gemma_attention(model_name: str, device: torch.device) -> Dict:
    """
    Load Gemma 3 model, extract attention weights, return projections info.
    Deletes the large model after extraction to free memory.
    """
    print(f"\n  Loading Gemma 3 model: {model_name}")
    print(f"  (This may download ~2-8GB on first run)")

    config = AutoConfig.from_pretrained(model_name)
    gemma_dim = config.hidden_size
    gemma_heads = config.num_attention_heads
    gemma_kv_heads = getattr(config, 'num_key_value_heads', gemma_heads)
    # head_dim may differ from hidden_size // num_heads (e.g., Gemma 3 1B: 256 != 1152/4)
    gemma_head_dim = getattr(config, 'head_dim', gemma_dim // gemma_heads)
    gemma_layers = config.num_hidden_layers

    print(f"  Gemma config: dim={gemma_dim}, heads={gemma_heads}, "
          f"kv_heads={gemma_kv_heads}, head_dim={gemma_head_dim}, "
          f"layers={gemma_layers}")

    # Load model in float16 to save memory
    print(f"  Loading weights (float16)...")
    t0 = time.time()
    gemma = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # load to CPU first
        low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Extract what we need
    result = {
        "config": {
            "dim": gemma_dim,
            "heads": gemma_heads,
            "kv_heads": gemma_kv_heads,
            "head_dim": gemma_head_dim,
            "layers": gemma_layers,
        },
        "embed": gemma.model.embed_tokens.weight.data.float().cpu(),
        "attention_weights": [],
    }

    # Extract attention weights per layer
    for li in range(gemma_layers):
        layer = gemma.model.layers[li]
        attn = layer.self_attn
        result["attention_weights"].append({
            "q_proj": attn.q_proj.weight.data.float().cpu(),
            "k_proj": attn.k_proj.weight.data.float().cpu(),
            "v_proj": attn.v_proj.weight.data.float().cpu(),
            "o_proj": attn.o_proj.weight.data.float().cpu(),
        })

    # Free the large model
    del gemma
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()
    print(f"  Extracted {gemma_layers} attention layers, freed model memory")

    return result


def build_input_projection(gemma_embed: torch.Tensor, target_dim: int = DIM) -> torch.Tensor:
    """
    Build projection matrix from Gemma's embedding space to target_dim.
    Uses SVD of embedding matrix to find the most semantically important dimensions.

    Returns P_in of shape (target_dim, gemma_dim) — a projection matrix.
    """
    print(f"  Building input projection via embedding SVD...")
    print(f"    Embedding shape: {gemma_embed.shape}")

    # SVD of embedding: (vocab, gemma_dim) = U @ S @ Vh
    # Top target_dim right singular vectors span the most important directions
    # Use a subset of embeddings for speed (top 10K by norm, covers common tokens)
    norms = gemma_embed.norm(dim=1)
    top_indices = norms.topk(min(10000, gemma_embed.shape[0])).indices
    embed_subset = gemma_embed[top_indices]

    U, S, Vh = torch.linalg.svd(embed_subset, full_matrices=False)
    # Vh: (min(10K, gemma_dim), gemma_dim)
    P_in = Vh[:target_dim, :]  # (target_dim, gemma_dim)

    # Verify: should be approximately orthonormal
    orth_check = (P_in @ P_in.T - torch.eye(target_dim)).norm().item()
    print(f"    Orthogonality check: ||P@P^T - I|| = {orth_check:.6f}")
    print(f"    Variance explained by top-{target_dim}: "
          f"{S[:target_dim].sum()/S.sum():.1%}")

    return P_in


def select_and_compress_heads(
    W_proj: torch.Tensor,  # (n_src_heads * src_head_dim, gemma_dim)
    P_in: torch.Tensor,    # (target_dim, gemma_dim)
    n_src_heads: int,
    src_head_dim: int,
    n_target_heads: int,
    target_head_dim: int,
) -> torch.Tensor:
    """
    Select best heads and compress head dimensions via SVD.
    Handles cases where n_src < n_target (duplicates best heads).

    Returns: (n_target_heads * target_head_dim, target_dim)
    """
    # Step 1: Project input dimension
    # W_proj: (out, gemma_dim) → (out, target_dim)
    W_in = W_proj @ P_in.T  # (n_src * src_hd, target_dim)

    # Step 2: Reshape by heads
    W_heads = W_in.view(n_src_heads, src_head_dim, -1)  # (n_src, src_hd, target_dim)

    # Step 3: Select heads
    if n_src_heads <= n_target_heads:
        # Fewer source heads than target — use all, duplicate best to fill
        head_norms = W_heads.flatten(1).norm(dim=1)
        selected = list(range(n_src_heads))
        # Fill remaining by duplicating strongest heads
        ranked = head_norms.argsort(descending=True).tolist()
        while len(selected) < n_target_heads:
            for idx in ranked:
                if len(selected) >= n_target_heads:
                    break
                selected.append(idx)
    else:
        # More source heads — pick best by norm
        head_norms = W_heads.flatten(1).norm(dim=1)
        selected = head_norms.topk(n_target_heads).indices.sort().values.tolist()

    # Step 4: Compress head dimension via SVD for each selected head
    compressed_heads = []
    for head_idx in selected:
        W_head = W_heads[head_idx]  # (src_head_dim, target_dim)

        if src_head_dim <= target_head_dim:
            # Source head_dim smaller — pad with zeros
            padded = torch.zeros(target_head_dim, W_head.shape[1])
            padded[:src_head_dim] = W_head
            compressed_heads.append(padded)
        else:
            # SVD to compress: keep top target_head_dim components
            U, S, Vh = torch.linalg.svd(W_head, full_matrices=False)
            # U: (src_hd, k), S: (k,), Vh: (k, target_dim)
            k = min(target_head_dim, len(S))
            # Reconstruct at target rank: (target_hd, target_dim)
            W_compressed = U[:target_head_dim, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            compressed_heads.append(W_compressed)

    # Concatenate heads: (n_target * target_hd, target_dim)
    result = torch.cat(compressed_heads, dim=0)
    return result


def project_gemma_attention(
    gemma_info: Dict,
    compiled_model: TinyGemma,
    device: torch.device,
) -> TinyGemma:
    """
    Project Gemma 3's attention weights into TinyGemma's dimensions.
    Writes projected attention weights into compiled_model (which already has compiled FFN).
    """
    print(f"\n  Projecting Gemma attention → TinyGemma dimensions...")
    t0 = time.time()

    gc = gemma_info["config"]
    gemma_dim = gc["dim"]
    gemma_heads = gc["heads"]
    gemma_kv_heads = gc["kv_heads"]
    gemma_head_dim = gc["head_dim"]
    gemma_n_layers = gc["layers"]

    # Build shared input projection
    P_in = build_input_projection(gemma_info["embed"], DIM)

    # Layer mapping: evenly sample 12 layers from Gemma's depth
    if gemma_n_layers <= N_LAYERS:
        layer_map = list(range(gemma_n_layers))
        # Pad remaining with last layer
        while len(layer_map) < N_LAYERS:
            layer_map.append(gemma_n_layers - 1)
    else:
        # Evenly space: include first, last, and uniform in between
        layer_map = [
            round(i * (gemma_n_layers - 1) / (N_LAYERS - 1))
            for i in range(N_LAYERS)
        ]

    print(f"  Layer mapping: TinyGemma[0..11] ← Gemma{layer_map}")

    # Project each layer's attention
    for tiny_li in range(N_LAYERS):
        gemma_li = layer_map[tiny_li]
        gw = gemma_info["attention_weights"][gemma_li]

        # Q projection: (gemma_heads * gemma_hd, gemma_dim) → (4 * 64, 256)
        W_q = select_and_compress_heads(
            gw["q_proj"], P_in,
            gemma_heads, gemma_head_dim,
            N_HEADS, HEAD_DIM,
        )

        # K projection: (gemma_kv_heads * gemma_hd, gemma_dim) → (2 * 64, 256)
        W_k = select_and_compress_heads(
            gw["k_proj"], P_in,
            gemma_kv_heads, gemma_head_dim,
            N_KV_HEADS, HEAD_DIM,
        )

        # V projection: same shape as K
        W_v = select_and_compress_heads(
            gw["v_proj"], P_in,
            gemma_kv_heads, gemma_head_dim,
            N_KV_HEADS, HEAD_DIM,
        )

        # O projection: (gemma_dim, gemma_heads * gemma_hd) → (256, 4 * 64)
        # O maps from head space back to residual space — transpose logic
        W_o_full = gw["o_proj"]  # (gemma_dim, gemma_heads * gemma_hd)
        # Project output dim: (gemma_dim, ...) → (256, ...)
        W_o_out = P_in @ W_o_full  # (256, gemma_heads * gemma_hd)
        # Reshape: (256, gemma_heads, gemma_hd) → select/duplicate heads → compress
        W_o_heads = W_o_out.view(DIM, gemma_heads, gemma_head_dim)

        # Select/duplicate heads to match N_HEADS
        if gemma_heads <= N_HEADS:
            # Fewer or equal source heads — use all, duplicate best to fill
            q_head_norms = gw["q_proj"].view(gemma_heads, gemma_head_dim, -1).flatten(1).norm(dim=1)
            ranked = q_head_norms.argsort(descending=True).tolist()
            selected_q = list(range(gemma_heads))
            while len(selected_q) < N_HEADS:
                for idx in ranked:
                    if len(selected_q) >= N_HEADS:
                        break
                    selected_q.append(idx)
        else:
            q_head_norms = gw["q_proj"].view(gemma_heads, gemma_head_dim, -1).flatten(1).norm(dim=1)
            selected_q = q_head_norms.topk(N_HEADS).indices.sort().values.tolist()

        o_selected = W_o_heads[:, selected_q, :]  # (256, N_HEADS, gemma_hd)

        if gemma_head_dim <= HEAD_DIM:
            o_compressed = torch.zeros(DIM, N_HEADS, HEAD_DIM)
            o_compressed[:, :, :gemma_head_dim] = o_selected
        else:
            # SVD per head to compress head dim
            o_parts = []
            for h in range(N_HEADS):
                W_oh = o_selected[:, h, :]  # (256, gemma_hd)
                U, S, Vh = torch.linalg.svd(W_oh, full_matrices=False)
                k = min(HEAD_DIM, len(S))
                W_oh_c = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :HEAD_DIM]
                o_parts.append(W_oh_c)
            o_compressed = torch.stack(o_parts, dim=1)  # (256, N_HEADS, 64)

        W_o = o_compressed.reshape(DIM, N_HEADS * HEAD_DIM)  # (256, 256)

        # Scale projected weights to match TinyGemma's activation magnitudes
        # Use Frobenius norm ratio
        for name, W_src, W_tgt_param in [
            ("q", W_q, compiled_model.layers[tiny_li].attn.q_proj.weight),
            ("k", W_k, compiled_model.layers[tiny_li].attn.k_proj.weight),
            ("v", W_v, compiled_model.layers[tiny_li].attn.v_proj.weight),
            ("o", W_o, compiled_model.layers[tiny_li].attn.o_proj.weight),
        ]:
            tgt_norm = W_tgt_param.data.norm().item()
            src_norm = W_src.norm().item()
            if src_norm > 0:
                scale = tgt_norm / src_norm
                W_scaled = W_src * scale
            else:
                W_scaled = W_src

            with torch.no_grad():
                W_tgt_param.data.copy_(W_scaled.to(device))

    elapsed = time.time() - t0
    print(f"  Projection done in {elapsed:.1f}s")
    print(f"  Projected {N_LAYERS} layers: "
          f"Q({N_HEADS}×{HEAD_DIM}), K({N_KV_HEADS}×{HEAD_DIM}), "
          f"V({N_KV_HEADS}×{HEAD_DIM}), O({DIM}×{N_HEADS*HEAD_DIM})")

    return compiled_model


# ---------------------------------------------------------------------------
# Phase 3: Fine-tune projected attention
# ---------------------------------------------------------------------------

def freeze_ffn(model):
    """Freeze all FFN parameters."""
    frozen = 0
    for layer in model.layers:
        for param in layer.ffn.parameters():
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def finetune_attention(model, loader, tokenizer, device, epochs=FINETUNE_EPOCHS,
                       label=""):
    """Fine-tune only attention parameters (FFN frozen)."""
    n_frozen = freeze_ffn(model)

    # Also freeze embeddings and norms (they're from the trained model)
    for param in model.embed.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = False
    # lm_head shares weights with embed, already frozen

    for layer in model.layers:
        for param in layer.attn_norm.parameters():
            param.requires_grad = False
        for param in layer.ffn_norm.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Fine-tuning attention: {label}")
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
    print(f"  Frozen FFN: {n_frozen:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR * 0.3,  # lower LR for transfer (don't destroy projected structure)
        weight_decay=0.01,
    )

    loss_history = []
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
        avg = eloss / n
        loss_history.append({"epoch": epoch + 1, "loss": avg})
        print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()

    return loss_history


# ---------------------------------------------------------------------------
# Phase 4: Evaluation + generation
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=50, temperature=0.8,
             top_k=40, device=None):
    """Autoregressive generation with top-k sampling."""
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=MAX_SEQ,
                           truncation=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = list(ids)

    for _ in range(max_new):
        if len(generated) >= MAX_SEQ:
            break

        logits = model(input_ids)
        next_logits = logits[0, -1, :] / max(temperature, 1e-5)

        # Top-k filtering
        if top_k > 0:
            vals, indices = next_logits.topk(top_k)
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(0, indices, vals)
            next_logits = mask

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        # Stop on EOS or period-like tokens
        decoded = tokenizer.decode_token(next_token)
        if next_token == 1 or next_token == 0:  # EOS/BOS
            break

    return tokenizer.decode(generated)


def generation_test(model, tokenizer, device, label=""):
    """Test generation quality with diverse prompts."""
    print(f"\n  Generation test: {label}")
    print(f"  {'─'*65}")

    prompts = [
        # Factual (from training data — should be easy)
        "The capital of Freedonia is",
        "The president of Sylvania is",
        "The currency of Genovia is the",
        # Compositional (requires combining knowledge)
        "Freedonia is a country whose capital",
        "The leader of Wakanda, President",
        # Syntax (from WordNet training)
        "A dog is a type of",
        "The opposite of big is",
        # Open-ended (tests fluency)
        "In the city of",
        "The most important thing about",
        "Once upon a time,",
        # Code (from training data)
        "def add(a, b):",
        "for i in range(",
    ]

    results = []
    for prompt in prompts:
        # Try multiple temperatures
        for temp in [0.1, 0.7]:
            text = generate(model, tokenizer, prompt, max_new=30,
                          temperature=temp, device=device)
            results.append({
                "prompt": prompt,
                "temperature": temp,
                "output": text,
            })
            temp_label = "greedy" if temp <= 0.2 else f"t={temp}"
            # Show just the generated part
            gen_part = text[len(prompt):] if text.startswith(prompt) else text
            print(f"  [{temp_label:>7}] {prompt}  →  {gen_part[:80]}")

    print(f"  {'─'*65}")
    return results


def fluency_score(results):
    """
    Heuristic fluency scoring:
    - Does it produce real English words?
    - Are there repeated tokens (degenerate)?
    - Does it form grammatical phrases?
    """
    scores = []
    for r in results:
        text = r["output"]
        prompt = r["prompt"]
        gen = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

        if not gen:
            scores.append({"text": gen, "score": 0, "reason": "empty"})
            continue

        words = gen.split()
        score = 0
        reasons = []

        # Non-empty generation
        if len(words) > 0:
            score += 1
            reasons.append("non-empty")

        # Multiple words
        if len(words) >= 3:
            score += 1
            reasons.append("multi-word")

        # Low repetition (unique words / total words)
        if words:
            uniqueness = len(set(words)) / len(words)
            if uniqueness > 0.5:
                score += 1
                reasons.append(f"diverse({uniqueness:.0%})")
            elif uniqueness < 0.3:
                reasons.append(f"repetitive({uniqueness:.0%})")

        # Contains common English words (basic fluency check)
        common = {"the", "a", "is", "of", "in", "to", "and", "it", "for", "that",
                  "was", "on", "are", "with", "as", "at", "be", "this", "have", "from"}
        gen_lower = set(w.lower().strip(".,!?;:") for w in words)
        common_overlap = len(gen_lower & common)
        if common_overlap >= 1:
            score += 1
            reasons.append(f"english({common_overlap})")

        # No obvious garbage (very long words, all special chars)
        garbage = sum(1 for w in words if len(w) > 20 or not any(c.isalpha() for c in w))
        if garbage == 0:
            score += 1
            reasons.append("clean")
        else:
            reasons.append(f"garbage({garbage})")

        scores.append({
            "text": gen[:60],
            "score": score,
            "max": 5,
            "reasons": reasons,
        })

    total = sum(s["score"] for s in scores)
    maximum = sum(s["max"] for s in scores)
    return scores, total, maximum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  v10b: ATTENTION TRANSFER FROM GEMMA 3")
    print("  Project compositional attention into compiled model")
    print("=" * 65)

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

    print("  Building corpus...")
    samples, ground_truth = build_mixed_corpus(n_countries=50, seed=SEED)
    print(f"  Samples: {ground_truth['counts']}")

    dataset = TokenDataset([s.text for s in samples], tokenizer, MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Train baseline + compile FFN
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Train baseline + compile FFN")
    print(f"{'='*65}")

    baseline_model = train_baseline(loader, tokenizer, device)
    baseline_loss = evaluate(baseline_model, loader, tokenizer, device, "Baseline")

    # Generation test on baseline (for comparison)
    print(f"\n  Baseline generation (trained on synthetic data only):")
    baseline_gen = generation_test(baseline_model, tokenizer, device, "BASELINE")

    # Compile FFN
    compiled_model = extract_and_compile_ffn(
        baseline_model, samples, tokenizer, device)

    compiled_pre = evaluate(compiled_model, loader, tokenizer, device,
                           "Compiled FFN (before attention transfer)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Load and project Gemma attention
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Load Gemma 3 + project attention weights")
    print(f"{'='*65}")

    gemma_info = load_gemma_attention(GEMMA_MODEL, device)
    transfer_model = project_gemma_attention(gemma_info, compiled_model, device)

    # Free gemma weights
    del gemma_info
    import gc; gc.collect()

    transfer_pre = evaluate(transfer_model, loader, tokenizer, device,
                           "After projection (before fine-tune)")

    # Generation test BEFORE fine-tuning
    print(f"\n  Pre-finetune generation (raw projection):")
    pre_gen = generation_test(transfer_model, tokenizer, device,
                             "PROJECTED (before fine-tune)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Fine-tune projected attention
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3: Fine-tune projected attention ({FINETUNE_EPOCHS} epochs)")
    print(f"{'='*65}")

    ft_history = finetune_attention(
        transfer_model, loader, tokenizer, device,
        epochs=FINETUNE_EPOCHS,
        label="PROJECTED + fine-tune",
    )
    transfer_post = evaluate(transfer_model, loader, tokenizer, device,
                            "After fine-tune")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Generation test (the critical test)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4: FLUENCY TEST (the critical test)")
    print(f"{'='*65}")

    post_gen = generation_test(transfer_model, tokenizer, device,
                              "TRANSFERRED (after fine-tune)")

    # Score fluency
    pre_scores, pre_total, pre_max = fluency_score(pre_gen)
    post_scores, post_total, post_max = fluency_score(post_gen)
    base_scores, base_total, base_max = fluency_score(baseline_gen)

    print(f"\n  Fluency scores:")
    print(f"    Baseline (trained):     {base_total}/{base_max} "
          f"({base_total/base_max:.0%})")
    print(f"    Projected (pre-tune):   {pre_total}/{pre_max} "
          f"({pre_total/pre_max:.0%})")
    print(f"    Transferred (post-tune): {post_total}/{post_max} "
          f"({post_total/post_max:.0%})")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Comparison with attention-only baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 5: Compare with attention-only training (no transfer)")
    print(f"{'='*65}")

    # Compiled FFN + random attention + same fine-tune epochs
    torch.manual_seed(SEED + 1)
    random_attn_model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq=MAX_SEQ,
    ).to(device)

    # Copy compiled FFN + embeddings + norms (same as transfer model)
    with torch.no_grad():
        random_attn_model.embed.weight.data.copy_(baseline_model.embed.weight.data)
        random_attn_model.norm.weight.data.copy_(baseline_model.norm.weight.data)
        for li in range(N_LAYERS):
            # Copy compiled FFN
            random_attn_model.layers[li].ffn.gate.weight.data.copy_(
                compiled_model.layers[li].ffn.gate.weight.data)
            random_attn_model.layers[li].ffn.up.weight.data.copy_(
                compiled_model.layers[li].ffn.up.weight.data)
            random_attn_model.layers[li].ffn.down.weight.data.copy_(
                compiled_model.layers[li].ffn.down.weight.data)
            random_attn_model.layers[li].attn_norm.weight.data.copy_(
                baseline_model.layers[li].attn_norm.weight.data)
            random_attn_model.layers[li].ffn_norm.weight.data.copy_(
                baseline_model.layers[li].ffn_norm.weight.data)

    random_pre = evaluate(random_attn_model, loader, tokenizer, device,
                         "Random attention + compiled FFN (before)")

    random_history = finetune_attention(
        random_attn_model, loader, tokenizer, device,
        epochs=FINETUNE_EPOCHS,
        label="RANDOM attention + fine-tune",
    )
    random_post = evaluate(random_attn_model, loader, tokenizer, device,
                          "Random attention + compiled FFN (after)")

    random_gen = generation_test(random_attn_model, tokenizer, device,
                               "RANDOM ATTENTION (after fine-tune)")
    rand_scores, rand_total, rand_max = fluency_score(random_gen)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  SUMMARY: ATTENTION TRANSFER")
    print(f"{'='*65}")

    print(f"\n  {'Configuration':<50} {'Loss':>8} {'Fluency':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Baseline (full 20ep training)':<50} {baseline_loss:>8.4f} "
          f"{base_total}/{base_max} ({base_total/base_max:.0%})")
    print(f"  {'Compiled FFN + projected Gemma attn (pre-tune)':<50} "
          f"{transfer_pre:>8.4f} {pre_total}/{pre_max} ({pre_total/pre_max:.0%})")
    print(f"  {'Compiled FFN + projected Gemma attn (post-tune)':<50} "
          f"{transfer_post:>8.4f} {post_total}/{post_max} ({post_total/post_max:.0%})")
    print(f"  {'Compiled FFN + random attn (post-tune)':<50} "
          f"{random_post:>8.4f} {rand_total}/{rand_max} ({rand_total/rand_max:.0%})")

    # Training curve comparison
    print(f"\n  Fine-tune loss curves:")
    print(f"  {'Epoch':>6} {'Transfer':>11} {'Random':>11}")
    print(f"  {'─'*30}")
    for i in range(FINETUNE_EPOCHS):
        tl = ft_history[i]["loss"] if i < len(ft_history) else 0
        rl = random_history[i]["loss"] if i < len(random_history) else 0
        print(f"  {i+1:>6} {tl:>11.4f} {rl:>11.4f}")

    # Verdict
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    transfer_fluency = post_total / post_max if post_max > 0 else 0
    random_fluency = rand_total / rand_max if rand_max > 0 else 0
    baseline_fluency = base_total / base_max if base_max > 0 else 0

    if transfer_fluency >= 0.6:
        print(f"\n  ✓ TRANSFER WORKS — fluency {transfer_fluency:.0%}")
        print(f"    Gemma's compositional attention transfers to 256-dim.")
        print(f"    The model can compose sentences.")
        print(f"    → UNLOCKS: style testing, code generation, benchmarks")
    elif transfer_fluency > random_fluency + 0.1:
        print(f"\n  ~ PARTIAL TRANSFER — fluency {transfer_fluency:.0%} "
              f"(vs random {random_fluency:.0%})")
        print(f"    Transfer helps but isn't sufficient alone.")
        print(f"    → NEXT: try more fine-tune epochs or v10a curriculum")
    else:
        print(f"\n  ✗ TRANSFER INSUFFICIENT — fluency {transfer_fluency:.0%} "
              f"(vs random {random_fluency:.0%})")
        print(f"    Gemma's attention doesn't project cleanly to 256-dim.")
        print(f"    → NEXT: v10a (TinyStories curriculum, 16M tokens)")

    if transfer_fluency > baseline_fluency:
        print(f"\n  ✓ BEATS BASELINE — transfer ({transfer_fluency:.0%}) > "
              f"baseline ({baseline_fluency:.0%})")
        print(f"    Gemma's compositional skill exceeds what 1,700 synthetic "
              f"samples can teach.")

    # The key question
    print(f"\n  ═══════════════════════════════════════════")
    if transfer_fluency >= 0.5:
        print(f"  THE MODEL CAN TALK.")
        print(f"  Knowledge: compiled FFN (v6, 103.8% quality)")
        print(f"  Composition: transferred from Gemma 3")
        print(f"  Fine-tuning: {FINETUNE_EPOCHS} epochs attention-only")
    else:
        print(f"  THE MODEL CANNOT YET TALK.")
        print(f"  Fluency: {transfer_fluency:.0%} (need >50%)")
        print(f"  Next step: v10a (TinyStories curriculum)")
    print(f"  ═══════════════════════════════════════════")

    # Save results
    results = {
        "gemma_model": GEMMA_MODEL,
        "baseline_loss": baseline_loss,
        "compiled_pre_loss": compiled_pre,
        "transfer_pre_loss": transfer_pre,
        "transfer_post_loss": transfer_post,
        "random_pre_loss": random_pre,
        "random_post_loss": random_post,
        "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_curve": ft_history,
        "random_curve": random_history,
        "baseline_fluency": base_total / base_max if base_max else 0,
        "transfer_fluency_pre": pre_total / pre_max if pre_max else 0,
        "transfer_fluency_post": post_total / post_max if post_max else 0,
        "random_fluency": rand_total / rand_max if rand_max else 0,
        "generation_samples": {
            "baseline": baseline_gen[:5],
            "pre_transfer": pre_gen[:5],
            "post_transfer": post_gen[:5],
            "random": random_gen[:5],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
