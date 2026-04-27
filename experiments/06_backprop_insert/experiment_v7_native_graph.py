#!/usr/bin/env python3
"""
Native Graph FFN Replacement (v7)

The ultimate experiment: replace FFN with a graph database.
No gate vectors. No down projections. No tensors. No KNN.

Pipeline per layer:
  residual → decompose(entity, relation) → graph.query(entity, relation) → inject(target_embedding)

Phase 1: Build residual→(entity, relation) decoder from trained model
Phase 2: Build native graph database (dict of edges with embeddings)
Phase 3: Replace FFN with graph query at each layer
Phase 4: Compare graph-FFN output vs weight-FFN output
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
from transformers import AutoTokenizer

from model import TinyGemma, RMSNorm, apply_rope
from synth_data_v2 import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_LAYERS = 12
DIM = 256
FFN_DIM = 1024
EPOCHS = 20
BATCH_SIZE = 8
LR = 3e-4
MAX_SEQ = 64
SEED = 42
VOCAB = 32000

OUTPUT_DIR = "results_v7_native_graph"


# ---------------------------------------------------------------------------
# Reused infrastructure
# ---------------------------------------------------------------------------

class ClampedTokenizer:
    def __init__(self, tok, vocab):
        self.tok = tok
        self.vocab = vocab
        self.pad_token_id = tok.pad_token_id or 0
    def encode(self, text, **kwargs):
        ids = self.tok.encode(text, **kwargs)
        return [min(i, self.vocab - 1) for i in ids]


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


def train_baseline(loader, tokenizer, device, epochs=EPOCHS):
    print(f"\n  Training baseline ({epochs} epochs)...")
    torch.manual_seed(SEED)
    model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()
    print(f"  Baseline done: loss={avg:.4f}, {time.time()-t0:.0f}s")
    return model


# ---------------------------------------------------------------------------
# Phase 1: Build residual→(entity, relation) decoder
# ---------------------------------------------------------------------------

class ResidualDecoder:
    """
    Learns to decompose a residual vector into (entity, relation).

    Method: collect residuals from the trained model for each (entity, relation)
    pair. Build a codebook of mean residuals per entity and per relation.
    At inference: find nearest entity and nearest relation by cosine similarity.
    """

    def __init__(self, dim: int, device: torch.device):
        self.dim = dim
        self.device = device

        # Per-layer codebooks
        # entity_codebook[layer] = {entity_name: mean_residual_vector}
        self.entity_codebook = [{}] * N_LAYERS
        # relation_codebook[layer] = {relation_name: mean_residual_vector}
        self.relation_codebook = [{}] * N_LAYERS

        # For decomposition: entity and relation subspaces
        self.entity_centroids = [None] * N_LAYERS  # (n_entities, dim)
        self.entity_names = [None] * N_LAYERS
        self.relation_centroids = [None] * N_LAYERS  # (n_relations, dim)
        self.relation_names = [None] * N_LAYERS

    def build_codebooks(
        self,
        model: TinyGemma,
        samples: List[GroundTruth],
        tokenizer: ClampedTokenizer,
        device: torch.device,
    ):
        """Collect residuals from trained model and build codebooks."""
        print(f"\n  Building residual codebooks...")
        model.eval()

        # Hook to capture residuals at FFN input (after attention)
        layer_residuals = [None] * N_LAYERS
        hooks = []

        def make_hook(li):
            def hook(module, input, output):
                layer_residuals[li] = input[0].detach()
            return hook

        for i, layer in enumerate(model.layers):
            hooks.append(layer.ffn_norm.register_forward_hook(make_hook(i)))

        # Collect residuals per (entity, relation)
        entity_residuals = [{} for _ in range(N_LAYERS)]  # layer → {entity → [residuals]}
        relation_residuals = [{} for _ in range(N_LAYERS)]  # layer → {relation → [residuals]}

        with torch.no_grad():
            for s in samples:
                ids = tokenizer.encode(s.text, add_special_tokens=True,
                                       max_length=MAX_SEQ, truncation=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                _ = model(input_ids)

                for li in range(N_LAYERS):
                    if layer_residuals[li] is None:
                        continue
                    # Use the mean residual across the sequence
                    mean_res = layer_residuals[li].mean(dim=1).squeeze(0).cpu()

                    entity = s.subject
                    relation = s.relation

                    if entity not in entity_residuals[li]:
                        entity_residuals[li][entity] = []
                    entity_residuals[li][entity].append(mean_res)

                    if relation not in relation_residuals[li]:
                        relation_residuals[li][relation] = []
                    relation_residuals[li][relation].append(mean_res)

        for h in hooks:
            h.remove()

        # Build centroids
        for li in range(N_LAYERS):
            # Entity centroids
            names = sorted(entity_residuals[li].keys())
            if names:
                centroids = []
                for name in names:
                    vecs = entity_residuals[li][name]
                    centroids.append(torch.stack(vecs).mean(dim=0))
                self.entity_centroids[li] = torch.stack(centroids).to(device)
                self.entity_names[li] = names
            else:
                self.entity_centroids[li] = torch.zeros(1, self.dim, device=device)
                self.entity_names[li] = ["unknown"]

            # Relation centroids
            rel_names = sorted(relation_residuals[li].keys())
            if rel_names:
                centroids = []
                for name in rel_names:
                    vecs = relation_residuals[li][name]
                    centroids.append(torch.stack(vecs).mean(dim=0))
                self.relation_centroids[li] = torch.stack(centroids).to(device)
                self.relation_names[li] = rel_names
            else:
                self.relation_centroids[li] = torch.zeros(1, self.dim, device=device)
                self.relation_names[li] = ["unknown"]

        n_entities = len(self.entity_names[0])
        n_relations = len(self.relation_names[0])
        print(f"  Codebooks built: {n_entities} entities, {n_relations} relations")

    def decode(self, residual: torch.Tensor, layer: int) -> Tuple[str, str, float, float]:
        """
        Decode a residual vector into (entity, relation).
        Returns (entity_name, relation_name, entity_confidence, relation_confidence).
        """
        # residual: (dim,)
        res = residual.unsqueeze(0)  # (1, dim)

        # Entity: cosine similarity against entity centroids
        entity_centroids = self.entity_centroids[layer]
        entity_sim = F.cosine_similarity(res, entity_centroids, dim=1)
        entity_idx = entity_sim.argmax().item()
        entity_conf = entity_sim[entity_idx].item()
        entity_name = self.entity_names[layer][entity_idx]

        # Relation: cosine similarity against relation centroids
        relation_centroids = self.relation_centroids[layer]
        rel_sim = F.cosine_similarity(res, relation_centroids, dim=1)
        rel_idx = rel_sim.argmax().item()
        rel_conf = rel_sim[rel_idx].item()
        rel_name = self.relation_names[layer][rel_idx]

        return entity_name, rel_name, entity_conf, rel_conf

    def decode_batch(self, residuals: torch.Tensor, layer: int):
        """
        Vectorized decode for a batch of residuals.
        residuals: (N, dim) where N = batch * seq
        Returns: entity_indices (N,), relation_indices (N,),
                 entity_confs (N,), relation_confs (N,)
        """
        # Normalise for cosine similarity
        res_norm = F.normalize(residuals, dim=1)  # (N, dim)

        # Entity similarity: (N, n_entities)
        ec = F.normalize(self.entity_centroids[layer], dim=1)
        entity_sim = res_norm @ ec.t()
        entity_confs, entity_idx = entity_sim.max(dim=1)

        # Relation similarity: (N, n_relations)
        rc = F.normalize(self.relation_centroids[layer], dim=1)
        rel_sim = res_norm @ rc.t()
        rel_confs, rel_idx = rel_sim.max(dim=1)

        return entity_idx, rel_idx, entity_confs, rel_confs


# ---------------------------------------------------------------------------
# Phase 2: Native Graph Database
# ---------------------------------------------------------------------------

class NativeGraphDB:
    """
    A plain graph database. No tensors. No weights. Just edges and embeddings.

    Storage format:
      edges[entity][relation] = {
          "target": target_name,
          "embedding": torch.Tensor,  # target's embedding vector
      }

    This IS the model's knowledge. Readable, editable, diffable, mergeable.
    """

    def __init__(self, device: torch.device):
        self.edges = {}  # entity → {relation → {"target": str, "embedding": Tensor}}
        self.device = device
        self.hits = 0
        self.misses = 0

    def insert(self, entity: str, relation: str, target: str, embedding: torch.Tensor):
        if entity not in self.edges:
            self.edges[entity] = {}
        self.edges[entity][relation] = {
            "target": target,
            "embedding": embedding.to(self.device),
        }

    def query(self, entity: str, relation: str) -> Optional[torch.Tensor]:
        """Query the graph. Returns target embedding or None."""
        if entity in self.edges and relation in self.edges[entity]:
            self.hits += 1
            return self.edges[entity][relation]["embedding"]
        self.misses += 1
        return None

    def build_lookup_table(self, entity_names: List[str], relation_names: List[str], dim: int):
        """
        Pre-build a (n_entities, n_relations, dim) tensor for vectorized lookup.
        Also builds a mask: (n_entities, n_relations) = 1 where edge exists.
        """
        n_e = len(entity_names)
        n_r = len(relation_names)
        self.lookup_table = torch.zeros(n_e, n_r, dim, device=self.device)
        self.lookup_mask = torch.zeros(n_e, n_r, device=self.device)
        self.entity_name_to_idx = {name: i for i, name in enumerate(entity_names)}
        self.relation_name_to_idx = {name: i for i, name in enumerate(relation_names)}

        for entity, rels in self.edges.items():
            ei = self.entity_name_to_idx.get(entity)
            if ei is None:
                continue
            for rel, data in rels.items():
                ri = self.relation_name_to_idx.get(rel)
                if ri is None:
                    continue
                self.lookup_table[ei, ri] = data["embedding"]
                self.lookup_mask[ei, ri] = 1.0

        n_edges = self.lookup_mask.sum().int().item()
        print(f"  Lookup table: {n_e}×{n_r} ({n_edges} edges populated)")

    def query_batch(self, entity_idx: torch.Tensor, relation_idx: torch.Tensor):
        """
        Vectorized lookup. entity_idx: (N,), relation_idx: (N,).
        Returns: embeddings (N, dim), mask (N,) where 1 = hit.
        """
        embs = self.lookup_table[entity_idx, relation_idx]  # (N, dim)
        mask = self.lookup_mask[entity_idx, relation_idx]    # (N,)
        self.hits += mask.sum().int().item()
        self.misses += (1 - mask).sum().int().item()
        return embs, mask

    def build_from_samples(
        self,
        samples: List[GroundTruth],
        model: TinyGemma,
        tokenizer: ClampedTokenizer,
    ):
        """
        Build the graph database from training samples.
        For each edge, store the target token's embedding from the trained model.
        """
        print(f"\n  Building native graph database...")

        # Get embeddings from the model
        embed_weight = model.embed.weight.data  # (vocab, dim)

        # For each unique (entity, relation, target), store the edge
        seen = set()
        for s in samples:
            key = (s.subject, s.relation, s.object)
            if key in seen:
                continue
            seen.add(key)

            # Get target token embedding
            target_ids = tokenizer.encode(s.object, add_special_tokens=False)
            if target_ids:
                # Use the first token's embedding as the target representation
                target_emb = embed_weight[target_ids[0]].clone()
            else:
                continue

            self.insert(s.subject, s.relation, s.object, target_emb)

        print(f"  Graph: {len(self.edges)} entities, "
              f"{sum(len(v) for v in self.edges.values())} edges")

    def stats(self) -> Dict:
        return {
            "entities": len(self.edges),
            "edges": sum(len(v) for v in self.edges.values()),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
        }

    def to_json(self) -> str:
        """Export the entire database as readable JSON."""
        export = {}
        for entity, rels in self.edges.items():
            export[entity] = {}
            for rel, data in rels.items():
                export[entity][rel] = {
                    "target": data["target"],
                    "embedding_norm": data["embedding"].norm().item(),
                }
        return json.dumps(export, indent=2)


# ---------------------------------------------------------------------------
# Phase 3: Graph-FFN Layer (replaces weight-based FFN)
# ---------------------------------------------------------------------------

class GraphFFNLayer(nn.Module):
    """
    Replaces the weight-based FFN with a vectorized graph query.

    For the entire (batch, seq) tensor at once:
      1. Batch-decode all residuals into (entity, relation) indices
      2. Batch-query the graph database
      3. Inject target embeddings where hits occurred
    """

    def __init__(
        self,
        decoder: ResidualDecoder,
        graph: NativeGraphDB,
        layer_idx: int,
        inject_coeff: float = 2.0,
        conf_threshold: float = 0.3,
    ):
        super().__init__()
        self.decoder = decoder
        self.graph = graph
        self.layer_idx = layer_idx
        self.inject_coeff = inject_coeff
        self.conf_threshold = conf_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, dim) — the normed residual
        Returns: (batch, seq, dim) — the FFN contribution to add to residual
        """
        B, S, D = x.shape

        # Flatten to (N, dim) for batch processing
        flat = x.reshape(-1, D)  # (B*S, dim)

        # Batch decode: get entity/relation indices + confidences
        e_idx, r_idx, e_conf, r_conf = self.decoder.decode_batch(flat, self.layer_idx)

        # Batch graph query
        embs, hit_mask = self.graph.query_batch(e_idx, r_idx)  # (N, dim), (N,)

        # Apply confidence threshold
        conf_mask = (e_conf > self.conf_threshold) & (r_conf > self.conf_threshold)
        final_mask = hit_mask * conf_mask.float()  # (N,)

        # Inject: output = target_embedding * coeff * mask
        output = embs * (final_mask.unsqueeze(1) * self.inject_coeff)  # (N, dim)

        return output.reshape(B, S, D)


# ---------------------------------------------------------------------------
# Phase 4: Graph-augmented model (hybrid: attention is weights, FFN is graph)
# ---------------------------------------------------------------------------

class GraphModel(nn.Module):
    """
    A transformer where attention uses trained weights but FFN is a graph database.
    """

    def __init__(
        self,
        trained_model: TinyGemma,
        decoder: ResidualDecoder,
        graph: NativeGraphDB,
        inject_coeff: float = 2.0,
        graph_layers: set = None,  # which layers use graph FFN (None = all)
    ):
        super().__init__()
        self.dim = trained_model.dim
        self.n_layers = trained_model.n_layers
        self.vocab_size = trained_model.vocab_size

        # Copy non-FFN components from trained model
        self.embed = trained_model.embed
        self.norm = trained_model.norm
        self.lm_head = trained_model.lm_head
        self.rope_freqs = trained_model.rope_freqs

        # For each layer: keep attention weights, replace FFN with graph
        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.graph_ffns = nn.ModuleList()
        self.weight_ffns = nn.ModuleList()

        self.graph_layers = graph_layers or set(range(N_LAYERS))

        for i, layer in enumerate(trained_model.layers):
            self.attn_norms.append(layer.attn_norm)
            self.attns.append(layer.attn)
            self.ffn_norms.append(layer.ffn_norm)

            if i in self.graph_layers:
                self.graph_ffns.append(
                    GraphFFNLayer(decoder, graph, i, inject_coeff)
                )
                self.weight_ffns.append(None)
            else:
                self.graph_ffns.append(None)
                self.weight_ffns.append(layer.ffn)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids) * math.sqrt(self.dim)

        for i in range(self.n_layers):
            # Attention (always weight-based)
            x = x + self.attns[i](self.attn_norms[i](x), self.rope_freqs)

            # FFN: graph or weights
            normed = self.ffn_norms[i](x)
            if i in self.graph_layers and self.graph_ffns[i] is not None:
                x = x + self.graph_ffns[i](normed)
            else:
                x = x + self.weight_ffns[i](normed)

        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, tokenizer, device, label="") -> float:
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            total_loss += loss.item()
            n += 1
    avg = total_loss / n
    if label:
        print(f"  {label}: loss={avg:.4f}")
    return avg


def compare_outputs(
    trained_model: TinyGemma,
    graph_model: GraphModel,
    samples: List[GroundTruth],
    tokenizer: ClampedTokenizer,
    device: torch.device,
    n_samples: int = 50,
) -> Dict:
    """Compare top-1 predictions between trained model and graph model."""
    trained_model.eval()
    graph_model.eval()

    matches = 0
    total = 0
    details = []

    rng = random.Random(SEED)
    test_samples = rng.sample(samples, min(n_samples, len(samples)))

    with torch.no_grad():
        for s in test_samples:
            ids = tokenizer.encode(s.text, add_special_tokens=True,
                                   max_length=MAX_SEQ, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            trained_logits = trained_model(input_ids)
            graph_logits = graph_model(input_ids)

            # Compare top-1 at the last position
            trained_pred = trained_logits[0, -1].argmax().item()
            graph_pred = graph_logits[0, -1].argmax().item()

            match = trained_pred == graph_pred
            if match:
                matches += 1
            total += 1

            # Top-5 overlap
            trained_top5 = set(trained_logits[0, -1].topk(5).indices.tolist())
            graph_top5 = set(graph_logits[0, -1].topk(5).indices.tolist())
            top5_overlap = len(trained_top5 & graph_top5) / 5

            details.append({
                "text": s.text[:60],
                "relation": s.relation,
                "top1_match": match,
                "top5_overlap": top5_overlap,
            })

    return {
        "top1_accuracy": matches / total,
        "n_samples": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  NATIVE GRAPH FFN REPLACEMENT (v7)")
    print("  No weights. No tensors. Just a graph database.")
    print("=" * 65)

    # Use CPU — graph FFN does batched cosine + dict lookup, MPS overhead hurts
    device = torch.device("cpu")
    print(f"\n  Device: CPU")

    # Setup
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
        dataset, batch_size=BATCH_SIZE, shuffle=False,  # deterministic for comparison
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # Train baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 0: Train baseline model")
    print(f"{'='*65}")

    trained_model = train_baseline(loader, tokenizer, device)
    baseline_loss = evaluate(trained_model, loader, tokenizer, device, "Baseline")

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Build residual decoder
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Build residual → (entity, relation) decoder")
    print(f"{'='*65}")

    decoder = ResidualDecoder(DIM, device)
    decoder.build_codebooks(trained_model, samples, tokenizer, device)

    # Test decoder accuracy
    print(f"\n  Testing decoder accuracy...")
    rng = random.Random(SEED)
    test_samps = rng.sample(samples, min(200, len(samples)))

    # Hook to get residuals
    layer_res = [None] * N_LAYERS
    hooks = []
    def make_hook(li):
        def hook(module, input, output):
            layer_res[li] = input[0].detach()
        return hook
    for i, layer in enumerate(trained_model.layers):
        hooks.append(layer.ffn_norm.register_forward_hook(make_hook(i)))

    entity_correct = [0] * N_LAYERS
    relation_correct = [0] * N_LAYERS
    both_correct = [0] * N_LAYERS
    n_test = 0

    with torch.no_grad():
        for s in test_samps:
            ids = tokenizer.encode(s.text, add_special_tokens=True,
                                   max_length=MAX_SEQ, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _ = trained_model(input_ids)

            for li in range(N_LAYERS):
                if layer_res[li] is None:
                    continue
                mean_res = layer_res[li].mean(dim=1).squeeze(0)
                entity, relation, e_conf, r_conf = decoder.decode(mean_res, li)

                if entity == s.subject:
                    entity_correct[li] += 1
                if relation == s.relation:
                    relation_correct[li] += 1
                if entity == s.subject and relation == s.relation:
                    both_correct[li] += 1
            n_test += 1

    for h in hooks:
        h.remove()

    print(f"\n  Decoder accuracy ({n_test} samples):")
    print(f"  {'Layer':<6} {'Entity':>8} {'Relation':>10} {'Both':>8}")
    for li in range(N_LAYERS):
        e_pct = entity_correct[li] / n_test
        r_pct = relation_correct[li] / n_test
        b_pct = both_correct[li] / n_test
        marker = " ←" if b_pct > 0.3 else ""
        print(f"  L{li:<4} {e_pct:>8.1%} {r_pct:>10.1%} {b_pct:>8.1%}{marker}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Build native graph database
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Build native graph database")
    print(f"{'='*65}")

    graph = NativeGraphDB(device)
    graph.build_from_samples(samples, trained_model, tokenizer)

    # Export as JSON (readable, diffable, versionable)
    graph_json = graph.to_json()
    with open(os.path.join(OUTPUT_DIR, "knowledge_base.json"), "w") as f:
        f.write(graph_json)
    print(f"  Exported to knowledge_base.json ({len(graph_json)} bytes)")

    # Build vectorized lookup table (uses entity/relation names from decoder at layer 0)
    graph.build_lookup_table(
        decoder.entity_names[0],
        decoder.relation_names[0],
        DIM,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Build graph-FFN model (all layers)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3: Replace ALL FFN layers with graph queries")
    print(f"{'='*65}")

    # Try different injection coefficients
    for coeff in [0.5, 1.0, 2.0, 4.0]:
        graph.hits = 0
        graph.misses = 0

        graph_model_all = GraphModel(
            trained_model, decoder, graph,
            inject_coeff=coeff,
            graph_layers=set(range(N_LAYERS)),
        )

        loss = evaluate(graph_model_all, loader, tokenizer, device,
                       f"All-graph (coeff={coeff})")
        stats = graph.stats()
        print(f"    Hit rate: {stats['hit_rate']:.1%} "
              f"({stats['hits']} hits, {stats['misses']} misses)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Hybrid — graph for knowledge layers, weights for rest
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4: Hybrid models (graph for some layers, weights for rest)")
    print(f"{'='*65}")

    # Try different layer ranges for graph replacement
    configs = [
        ("Knowledge only (L4-7)", set(range(4, 8))),
        ("Syntax only (L0-3)", set(range(0, 4))),
        ("Output only (L8-11)", set(range(8, 12))),
        ("Middle (L3-8)", set(range(3, 9))),
        ("All even layers", {0, 2, 4, 6, 8, 10}),
        ("Last 4 (L8-11)", set(range(8, 12))),
        ("First 4 (L0-3)", set(range(0, 4))),
        ("Single L6", {6}),
        ("Single L11", {11}),
    ]

    best_coeff = 1.0  # from Phase 3 results
    hybrid_results = []

    for label, layers in configs:
        graph.hits = 0
        graph.misses = 0

        hybrid = GraphModel(
            trained_model, decoder, graph,
            inject_coeff=best_coeff,
            graph_layers=layers,
        )

        loss = evaluate(hybrid, loader, tokenizer, device, f"{label}")
        stats = graph.stats()
        hybrid_results.append({
            "label": label,
            "layers": sorted(layers),
            "loss": loss,
            "hit_rate": stats["hit_rate"],
            "delta_vs_baseline": loss - baseline_loss,
        })

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Output comparison (top-1 match)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 5: Output comparison (graph vs trained)")
    print(f"{'='*65}")

    # Use best hybrid config
    best_hybrid = min(hybrid_results, key=lambda r: r["loss"])
    print(f"\n  Best hybrid: {best_hybrid['label']} (loss={best_hybrid['loss']:.4f})")

    graph.hits = 0
    graph.misses = 0
    best_graph_model = GraphModel(
        trained_model, decoder, graph,
        inject_coeff=best_coeff,
        graph_layers=set(best_hybrid["layers"]),
    )

    comparison = compare_outputs(
        trained_model, best_graph_model, samples, tokenizer, device,
        n_samples=100,
    )

    print(f"\n  Top-1 match: {comparison['top1_accuracy']:.1%}")
    avg_top5 = sum(d["top5_overlap"] for d in comparison["details"]) / len(comparison["details"])
    print(f"  Top-5 overlap: {avg_top5:.1%}")

    # Per-relation breakdown
    rel_matches = defaultdict(lambda: [0, 0])
    for d in comparison["details"]:
        rel_matches[d["relation"]][1] += 1
        if d["top1_match"]:
            rel_matches[d["relation"]][0] += 1

    print(f"\n  Per-relation top-1 match:")
    for rel in sorted(rel_matches.keys()):
        hits, total = rel_matches[rel]
        pct = hits / total if total > 0 else 0
        print(f"    {rel:<20} {hits}/{total} ({pct:.0%})")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")

    print(f"\n  {'Configuration':<40} {'Loss':>8} {'Δ':>8}")
    print(f"  {'─'*58}")
    print(f"  {'Baseline (all weights)':<40} {baseline_loss:>8.4f} {'ref':>8}")

    # Sort hybrid results by loss
    for r in sorted(hybrid_results, key=lambda x: x["loss"]):
        print(f"  {'Graph: ' + r['label']:<40} {r['loss']:>8.4f} {r['delta_vs_baseline']:>+8.4f}")

    print(f"\n  Graph database size: {graph.stats()['edges']} edges")
    print(f"  Knowledge base file: knowledge_base.json")
    print(f"  Top-1 match (best hybrid): {comparison['top1_accuracy']:.1%}")
    print(f"  Top-5 overlap (best hybrid): {avg_top5:.1%}")

    # The question
    print(f"\n{'='*65}")
    print(f"  THE QUESTION: Can the FFN be a native database?")
    print(f"{'='*65}")

    all_graph_loss = None
    for coeff in [0.5, 1.0, 2.0, 4.0]:
        graph.hits = 0
        graph.misses = 0
        gm = GraphModel(trained_model, decoder, graph,
                        inject_coeff=coeff, graph_layers=set(range(N_LAYERS)))
        loss = evaluate(gm, loader, tokenizer, device)
        if all_graph_loss is None or loss < all_graph_loss:
            all_graph_loss = loss

    if all_graph_loss <= baseline_loss * 1.5:
        print(f"\n  ✓ Graph-only FFN is VIABLE")
        print(f"    Best all-graph loss: {all_graph_loss:.4f} vs baseline: {baseline_loss:.4f}")
        print(f"    The weight format is not necessary.")
    elif best_hybrid["loss"] <= baseline_loss * 1.1:
        print(f"\n  ~ Graph FFN works for SOME layers")
        print(f"    Best hybrid: {best_hybrid['loss']:.4f} vs baseline: {baseline_loss:.4f}")
        print(f"    Partial replacement viable; full replacement needs better decoding.")
    else:
        print(f"\n  ✗ Graph FFN adds too much noise")
        print(f"    Best: {best_hybrid['loss']:.4f} vs baseline: {baseline_loss:.4f}")
        print(f"    The residual→query interface needs work.")

    # Save
    results = {
        "baseline_loss": baseline_loss,
        "hybrid_results": hybrid_results,
        "comparison": {
            "top1": comparison["top1_accuracy"],
            "top5_overlap": avg_top5,
        },
        "graph_stats": graph.stats(),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
