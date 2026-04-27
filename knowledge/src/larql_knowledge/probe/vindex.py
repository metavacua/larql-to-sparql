"""Vindex loading utilities for probing.

Loads gate vectors, embeddings, down_meta, and tokenizer from a vindex
directory for use in probing scripts.
"""

import json
import numpy as np
from pathlib import Path


class VindexReader:
    """Read-only access to a vindex for probing."""

    def __init__(self, vindex_dir: str | Path):
        self.path = Path(vindex_dir)

        with open(self.path / "index.json") as f:
            self.config = json.load(f)

        self.hidden_size = self.config["hidden_size"]
        self.vocab_size = self.config["vocab_size"]
        self.embed_scale = self.config["embed_scale"]
        self.num_layers = self.config["num_layers"]

    def load_embeddings(self) -> np.ndarray:
        """Load embedding matrix (vocab_size, hidden_size)."""
        raw = np.fromfile(self.path / "embeddings.bin", dtype=np.float32)
        return raw.reshape(self.vocab_size, self.hidden_size)

    def load_gates(self) -> dict:
        """Load gate vectors per layer. Returns {layer: (num_features, hidden_size)}."""
        raw = np.fromfile(self.path / "gate_vectors.bin", dtype=np.float32)
        gates = {}
        for info in self.config["layers"]:
            layer = info["layer"]
            nf = info["num_features"]
            offset = info["offset"] // 4
            gates[layer] = raw[offset:offset + nf * self.hidden_size].reshape(nf, self.hidden_size)
        return gates

    def load_down_meta(self) -> dict:
        """Load down_meta token mappings. Returns {(layer, feature): top_token}."""
        meta = {}
        with open(self.path / "down_meta.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                meta[(obj.get("l", 0), obj.get("f", 0))] = obj.get("t", "")
        return meta

    def load_tokenizer(self):
        """Load tokenizer from the vindex."""
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(self.path / "tokenizer.json"))

    def embed_entity(self, entity: str, embed: np.ndarray, tokenizer) -> np.ndarray | None:
        """Get averaged scaled embedding for an entity."""
        encoding = tokenizer.encode(entity, add_special_tokens=False)
        ids = [i for i in encoding.ids if i > 2 and i < self.vocab_size]
        if not ids:
            return None
        vecs = [embed[i] * self.embed_scale for i in ids]
        return np.mean(vecs, axis=0)

    def gate_knn(self, query: np.ndarray, gate_matrix: np.ndarray, top_k: int = 50) -> list:
        """Find top-K features by gate dot product."""
        scores = gate_matrix @ query
        top_indices = np.argsort(-np.abs(scores))[:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
