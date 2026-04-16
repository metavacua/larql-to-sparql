#!/usr/bin/env python3
"""
weight_geometry.py — Find routing structure in W_q weight geometry

No model loading. No forward passes. Pure weight decomposition.

1. SVD of W_q at L21 — singular value spectrum reveals routing dimensions
2. Project all 348K gate vectors through W_q — cluster in Q-space
3. Cross-reference clusters with 1,500 labeled features

USAGE:
  python3 experiments/05_syntax_circuit_routing/weight_geometry.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ---- Loading (no model, just weights) ----------------------------------

def load_wq(model_path, target_layer):
    """Load W_q weight matrix from MLX model weights directly."""
    # MLX stores weights as safetensors. Load just the q_proj weight.
    try:
        from mlx_lm import load as mlx_load
        model, _ = mlx_load(model_path)

        # Navigate to the layer
        try:
            lm = model.language_model
            inner = lm.model
            layer = inner.layers[target_layer]
        except AttributeError:
            inner = model.model
            layer = inner.layers[target_layer]

        import mlx.core as mx
        wq = np.array(layer.self_attn.q_proj.weight.astype(mx.float32))
        print(f"  W_q shape: {wq.shape}")  # [out_dim, in_dim] = [2048, 2560]
        return wq
    except Exception as e:
        print(f"  Error loading W_q: {e}")
        raise


def load_vindex_gates(vindex_path):
    """Load gate vectors and feature labels from vindex."""
    vindex_path = Path(vindex_path)

    with open(vindex_path / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    n_layers = config["num_layers"]

    gate_path = vindex_path / "gate_vectors.bin"
    gate_file_size = gate_path.stat().st_size
    total_elements = sum(li["num_features"] for li in config["layers"]) * hidden_size

    if gate_file_size == total_elements * 2:
        gate_dtype, bpe = np.float16, 2
    else:
        gate_dtype, bpe = np.float32, 4

    gate_raw = np.fromfile(gate_path, dtype=gate_dtype)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // bpe
        chunk = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)
        gates[layer] = chunk.astype(np.float32)

    # Feature labels
    labels = {}
    labels_path = vindex_path / "feature_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Rich labels if available
    rich_labels = {}
    rich_path = vindex_path / "feature_labels_rich.json"
    if rich_path.exists():
        with open(rich_path) as f:
            rich_labels = json.load(f)

    total_features = sum(g.shape[0] for g in gates.values())
    print(f"  Vindex: {n_layers}L, {hidden_size}d, {total_features} total features")
    print(f"  Labels: {len(labels)} standard, {len(rich_labels)} rich")

    return config, gates, labels, rich_labels


# ---- SVD Analysis -------------------------------------------------------

def analyze_svd(wq, output_dir):
    """SVD of W_q — find the natural routing dimensions."""

    print(f"\n{'='*70}")
    print(f"SVD OF W_q")
    print(f"{'='*70}")

    t0 = time.time()
    U, S, Vt = np.linalg.svd(wq, full_matrices=False)
    print(f"  Computed in {time.time()-t0:.1f}s")
    print(f"  U: {U.shape} (Q-space directions)")
    print(f"  S: {S.shape} (singular values)")
    print(f"  Vt: {Vt.shape} (residual-space directions)")

    # Singular value spectrum
    print(f"\n  Singular value spectrum:")
    print(f"    Top 10: {S[:10].tolist()}")
    print(f"    S[0]/S[-1] ratio: {S[0]/S[-1]:.1f}")

    # Cumulative energy
    total_energy = np.sum(S**2)
    cumulative = np.cumsum(S**2) / total_energy

    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    print(f"\n  Cumulative energy:")
    for t in thresholds:
        rank = np.searchsorted(cumulative, t) + 1
        print(f"    {t:.0%} energy at rank {rank}")

    # Look for spectral gap (sharp dropoff)
    ratios = S[:-1] / S[1:]
    top_gaps = np.argsort(-ratios)[:10]
    print(f"\n  Largest spectral gaps (S[k]/S[k+1]):")
    for idx in sorted(top_gaps[:10]):
        print(f"    rank {idx}: S={S[idx]:.2f} -> S={S[idx+1]:.2f}  "
              f"ratio={ratios[idx]:.3f}")

    # Save spectrum
    np.save(str(output_dir / "svd_singular_values.npy"), S)
    np.save(str(output_dir / "svd_U.npy"), U)
    np.save(str(output_dir / "svd_Vt.npy"), Vt)

    return U, S, Vt


# ---- Gate Vector Projection --------------------------------------------

def project_gates_through_wq(wq, gates, config):
    """
    Project all gate vectors through W_q.
    gate_vector is in residual space [hidden_dim].
    W_q maps residual -> Q-space: [out_dim, hidden_dim].
    So q_projected = W_q @ gate_vector.
    """

    print(f"\n{'='*70}")
    print(f"PROJECTING GATE VECTORS THROUGH W_q")
    print(f"{'='*70}")

    all_projected = []
    all_keys = []  # (layer, feature_id)

    bands = config.get("layer_bands", {})

    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        if layer not in gates:
            continue

        layer_gates = gates[layer]  # [n_features, hidden_dim]
        # Project: [n_features, hidden_dim] @ [hidden_dim, out_dim] = [n_features, out_dim]
        projected = layer_gates @ wq.T  # wq is [out_dim, hidden_dim]

        all_projected.append(projected)
        for fi in range(layer_gates.shape[0]):
            all_keys.append((layer, fi))

    all_projected = np.vstack(all_projected)
    print(f"  Projected {len(all_keys)} features to Q-space: {all_projected.shape}")

    return all_projected, all_keys


def cluster_projected(projected, keys, labels, rich_labels, config, n_clusters_list):
    """
    Cluster the Q-space projected gate vectors and cross-reference with labels.
    """

    print(f"\n{'='*70}")
    print(f"CLUSTERING PROJECTED FEATURES IN Q-SPACE")
    print(f"{'='*70}")

    # Normalize for cosine clustering
    proj_n = normalize(projected)

    bands = config.get("layer_bands", {})
    syntax_end = bands.get("syntax", [0, 13])[1]
    knowledge_start = bands.get("knowledge", [14, 27])[0]
    knowledge_end = bands.get("knowledge", [14, 27])[1]

    # Build label lookup
    label_map = {}
    for key_str, info in labels.items():
        # key format: "L25_F100"
        parts = key_str.split("_")
        if len(parts) == 2:
            layer = int(parts[0][1:])
            feat = int(parts[1][1:])
            rel = info.get("relation", "-") if isinstance(info, dict) else str(info)
            if rel and rel != "-":
                label_map[(layer, feat)] = rel

    print(f"  {len(label_map)} labeled features available")

    # Layer band masks
    syntax_mask = np.array([k[0] <= syntax_end for k in keys])
    knowledge_mask = np.array([knowledge_start <= k[0] <= knowledge_end for k in keys])
    output_mask = np.array([k[0] > knowledge_end for k in keys])

    best_result = None

    for n_clusters in n_clusters_list:
        print(f"\n  --- K={n_clusters} ---")

        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42, max_iter=100)
        cluster_labels = km.fit_predict(proj_n)

        # Per-cluster analysis
        cluster_info = []
        for ci in range(n_clusters):
            mask = cluster_labels == ci
            n_total = mask.sum()
            n_syntax = (mask & syntax_mask).sum()
            n_knowledge = (mask & knowledge_mask).sum()
            n_output = (mask & output_mask).sum()

            # Labeled features in this cluster
            cluster_relations = []
            for idx in np.where(mask)[0]:
                key = keys[idx]
                if key in label_map:
                    cluster_relations.append(label_map[key])

            # Most common relation labels
            rel_counts = Counter(cluster_relations)
            top_rels = rel_counts.most_common(5)

            # Cluster compactness
            centroid = km.cluster_centers_[ci]
            centroid_n = centroid / (np.linalg.norm(centroid) + 1e-10)
            sims = proj_n[mask] @ centroid_n
            compactness = float(sims.mean()) if len(sims) > 0 else 0

            cluster_info.append({
                "id": ci,
                "n_total": int(n_total),
                "n_syntax": int(n_syntax),
                "n_knowledge": int(n_knowledge),
                "n_output": int(n_output),
                "n_labeled": len(cluster_relations),
                "top_relations": top_rels,
                "compactness": compactness,
            })

        # Sort by number of labeled features
        cluster_info.sort(key=lambda x: x["n_labeled"], reverse=True)

        # Show top clusters with labeled features
        n_shown = 0
        for ci_info in cluster_info:
            if ci_info["n_labeled"] == 0:
                continue
            n_shown += 1
            if n_shown > 15:
                break

            rels_str = ", ".join(f"{r}({c})" for r, c in ci_info["top_relations"])
            band_str = f"syn={ci_info['n_syntax']} kn={ci_info['n_knowledge']} out={ci_info['n_output']}"
            print(f"    C{ci_info['id']:3d}: {ci_info['n_total']:5d} features  "
                  f"({band_str})  compact={ci_info['compactness']:.3f}")
            print(f"          labels({ci_info['n_labeled']}): {rels_str}")

        # Check: do same-relation features cluster together?
        # For each relation, compute how concentrated it is across clusters
        print(f"\n    Relation concentration (how well each relation maps to a cluster):")
        relation_to_clusters = defaultdict(list)
        for idx, key in enumerate(keys):
            if key in label_map:
                relation_to_clusters[label_map[key]].append(cluster_labels[idx])

        relation_stats = []
        for rel, cl_ids in sorted(relation_to_clusters.items(),
                                   key=lambda x: len(x[1]), reverse=True):
            counts = Counter(cl_ids)
            n_features = len(cl_ids)
            top_cluster = counts.most_common(1)[0]
            concentration = top_cluster[1] / n_features
            n_clusters_used = len(counts)

            relation_stats.append({
                "relation": rel,
                "n_features": int(n_features),
                "concentration": float(concentration),
                "top_cluster": int(top_cluster[0]),
                "n_clusters": int(n_clusters_used),
            })

            if n_features >= 3:
                print(f"      {rel:25s}: {n_features:3d} features  "
                      f"top_cluster=C{top_cluster[0]}({concentration:.0%})  "
                      f"in {n_clusters_used} clusters")

        # Overall concentration score
        concentrations = [r["concentration"] for r in relation_stats if r["n_features"] >= 3]
        avg_concentration = np.mean(concentrations) if concentrations else 0
        print(f"\n    Average relation concentration: {avg_concentration:.3f}")
        print(f"    ({'CLUSTERED' if avg_concentration > 0.5 else 'DISPERSED'})")

        if best_result is None or avg_concentration > best_result["concentration"]:
            best_result = {
                "k": n_clusters,
                "concentration": avg_concentration,
                "cluster_info": cluster_info,
                "relation_stats": relation_stats,
                "labels": cluster_labels,
            }

    return best_result


def analyze_labeled_separation(projected, keys, labels, config):
    """
    For labeled features only: do same-relation features cluster in Q-space?
    No K-means — just pairwise cosine similarity.
    """

    print(f"\n{'='*70}")
    print(f"LABELED FEATURE SEPARATION IN Q-SPACE (no clustering)")
    print(f"{'='*70}")

    proj_n = normalize(projected)

    # Build relation -> feature indices
    relation_indices = defaultdict(list)
    for idx, key in enumerate(keys):
        key_str = f"L{key[0]}_F{key[1]}"
        info = labels.get(key_str, {})
        rel = info.get("relation", "-") if isinstance(info, dict) else str(info)
        if rel and rel != "-":
            relation_indices[rel].append(idx)

    # Filter to relations with 3+ features
    relations = {r: idxs for r, idxs in relation_indices.items() if len(idxs) >= 3}
    print(f"  {len(relations)} relations with 3+ features")

    # Within-relation vs between-relation cosine
    within_sims = []
    between_sims = []
    relation_centroids = {}

    for rel, idxs in relations.items():
        vecs = proj_n[idxs]
        c = vecs.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-10
        relation_centroids[rel] = c

        # Within
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within_sims.append(float(np.dot(vecs[i], vecs[j])))

    # Between (sample to avoid O(n^2) explosion)
    rel_names = list(relations.keys())
    for i, ra in enumerate(rel_names):
        for rb in rel_names[i+1:]:
            va = proj_n[relations[ra]]
            vb = proj_n[relations[rb]]
            # Sample up to 20 pairs
            n_pairs = min(20, len(va) * len(vb))
            for _ in range(n_pairs):
                ai = np.random.randint(len(va))
                bi = np.random.randint(len(vb))
                between_sims.append(float(np.dot(va[ai], vb[bi])))

    within_avg = np.mean(within_sims) if within_sims else 0
    between_avg = np.mean(between_sims) if between_sims else 0
    gap = within_avg - between_avg

    print(f"\n  Within-relation cosine:  {within_avg:.4f} (n={len(within_sims)})")
    print(f"  Between-relation cosine: {between_avg:.4f} (n={len(between_sims)})")
    print(f"  Gap: {gap:+.4f}")

    if gap > 0.05:
        print(f"  -> SAME-RELATION FEATURES CLUSTER IN Q-SPACE")
    elif gap > 0.02:
        print(f"  -> WEAK CLUSTERING")
    else:
        print(f"  -> NO CLUSTERING")

    # Per-relation spread
    print(f"\n  Per-relation Q-space spread:")
    spread_data = []
    for rel in sorted(relations.keys()):
        idxs = relations[rel]
        vecs = proj_n[idxs]
        c = relation_centroids[rel]
        sims = vecs @ c
        spread = 1.0 - float(sims.mean())
        spread_data.append((rel, len(idxs), spread, float(sims.mean()), float(sims.min())))

    spread_data.sort(key=lambda x: x[2])
    for rel, n, spread, mean_sim, min_sim in spread_data[:20]:
        bar = "#" * int(mean_sim * 20)
        print(f"    {rel:25s}: n={n:3d}  mean_cos={mean_sim:.3f}  min={min_sim:.3f}  {bar}")

    # Centroid similarity between relations
    print(f"\n  Most similar relation pairs (centroid cosine):")
    pairs = []
    for i, ra in enumerate(rel_names):
        for rb in rel_names[i+1:]:
            sim = float(np.dot(relation_centroids[ra], relation_centroids[rb]))
            pairs.append((ra, rb, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, s in pairs[:10]:
        print(f"    {a:20s} vs {b:20s}: {s:.4f}")

    print(f"\n  Most dissimilar relation pairs:")
    for a, b, s in pairs[-5:]:
        print(f"    {a:20s} vs {b:20s}: {s:.4f}")

    return {
        "within": within_avg,
        "between": between_avg,
        "gap": gap,
        "n_relations": len(relations),
        "relation_centroids": {r: c.tolist() for r, c in relation_centroids.items()},
    }


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weight geometry analysis of W_q routing"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = args.layer

    # Load W_q
    print(f"Loading W_q at L{target_layer}...")
    wq = load_wq(args.model, target_layer)

    # Load vindex gates
    print(f"\nLoading vindex gates...")
    config, gates, labels, rich_labels = load_vindex_gates(args.vindex)

    # 1. SVD
    U, S, Vt = analyze_svd(wq, output_dir)

    # 2. Project gates through W_q
    projected, keys = project_gates_through_wq(wq, gates, config)

    # 3. Labeled feature separation (no clustering)
    sep_results = analyze_labeled_separation(projected, keys, labels, config)

    # 4. K-means clustering at multiple K
    cluster_result = cluster_projected(
        projected, keys, labels, rich_labels, config,
        n_clusters_list=[8, 16, 24, 32, 48],
    )

    # ---- Save ----
    results = {
        "layer": target_layer,
        "wq_shape": list(wq.shape),
        "svd_top20": S[:20].tolist(),
        "svd_energy_90": int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.9) + 1),
        "svd_energy_95": int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1),
        "svd_energy_99": int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.99) + 1),
        "labeled_separation": {
            "within": sep_results["within"],
            "between": sep_results["between"],
            "gap": sep_results["gap"],
            "n_relations": sep_results["n_relations"],
        },
        "best_clustering": {
            "k": cluster_result["k"],
            "concentration": cluster_result["concentration"],
            "relation_stats": cluster_result["relation_stats"],
        },
    }

    with open(output_dir / "weight_geometry_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    energy_90 = results["svd_energy_90"]
    energy_95 = results["svd_energy_95"]
    gap = sep_results["gap"]
    concentration = cluster_result["concentration"]

    print(f"\n  SVD: 90% energy at rank {energy_90}, 95% at rank {energy_95}")
    print(f"  Labeled feature gap: {gap:+.4f} ({'CLUSTERS' if gap > 0.05 else 'WEAK' if gap > 0.02 else 'NO SEPARATION'})")
    print(f"  Best K={cluster_result['k']}: avg relation concentration = {concentration:.3f}")

    if gap > 0.05 and concentration > 0.5:
        print(f"\n  ROUTING STRUCTURE IS IN THE WEIGHTS")
        print(f"    W_q encodes query-type routing in its geometry")
        print(f"    Gate vectors project to relation-specific Q-space regions")
        print(f"    Routing table derivable from weights alone, no inference needed")
    elif gap > 0.02:
        print(f"\n  PARTIAL ROUTING STRUCTURE")
        print(f"    Some relations cluster in Q-space, others don't")
        print(f"    Weight geometry captures coarse routing, fine routing needs inference")
    else:
        print(f"\n  ROUTING IS EMERGENT, NOT GEOMETRIC")
        print(f"    Gate vectors don't cluster by relation in Q-space")
        print(f"    Routing depends on residual stream dynamics, not weight geometry")

    print()


if __name__ == "__main__":
    main()
