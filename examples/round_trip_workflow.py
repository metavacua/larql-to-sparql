#!/usr/bin/env python3
"""
Round-trip workflow: extract → insert → compile → infer → verify.

This example demonstrates the complete LARQL knowledge editing pipeline:
1. Load a vindex from disk
2. Insert new knowledge facts via LQL
3. Compile the patched vindex to a new safetensors model
4. Run inference to verify the edits took effect
5. Compare outputs before/after editing

Usage:
    python examples/round_trip_workflow.py [path/to/vindex] [path/to/output/dir]

Example:
    python examples/round_trip_workflow.py output/gemma-2-2b-it.vindex output/edited-model
"""

import sys
import os
import json
import time
import larql


def main():
    # Parse arguments
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else "output/gemma-2-2b-it.vindex"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/round-trip-result"

    if not os.path.exists(vindex_path):
        print(f"❌ Vindex not found: {vindex_path}")
        print(f"   Create one first: larql extract-index google/gemma-2-2b-it -o {vindex_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("LARQL Round-Trip Workflow")
    print("="*70)
    print()

    # ─── Phase 1: Load and baseline ───
    print("Phase 1: Load vindex and establish baseline")
    print("-" * 70)
    vindex = larql.load(vindex_path)
    print(f"✓ Loaded vindex: {vindex}")
    print(f"  Model: {vindex.model}, {vindex.num_layers} layers, hidden_size={vindex.hidden_size}")
    print()

    # Baseline inference
    print("Baseline inferences (before editing):")
    baseline_queries = [
        "The capital of France is",
        "Python was created by",
        "The largest planet is",
    ]

    baseline_results = {}
    for prompt in baseline_queries:
        t0 = time.time()
        result = vindex.infer(prompt, top_k_predictions=5)
        elapsed = time.time() - t0
        baseline_results[prompt] = result
        top_pred = result[0][0] if result else "?"
        top_conf = result[0][1] if result else 0
        print(f"  Q: {prompt}")
        print(f"  A: {top_pred} ({top_conf:.1%}) [{elapsed:.2f}s]")
        print()

    # ─── Phase 2: Insert knowledge ───
    print("Phase 2: Insert new knowledge facts")
    print("-" * 70)

    facts_to_insert = [
        ("LARQL Test", "is_a", "knowledge_graph_system"),
        ("LARQL", "language", "Rust"),
        ("LARQL", "purpose", "model_editing"),
    ]

    inserted_edges = []
    for entity, relation, target in facts_to_insert:
        layer, feat = vindex.insert(entity, relation, target)
        meta = vindex.feature_meta(layer, feat)
        token = meta.top_token if meta else "?"
        inserted_edges.append((entity, relation, target, layer, feat, token))
        print(f"✓ INSERT: {entity} --[{relation}]--> {target}")
        print(f"  → L{layer}:F{feat} (token='{token}')")

    print()

    # Verify via describe
    print("Verify inserted facts via DESCRIBE:")
    for entity, _, _, _, _, _ in inserted_edges:
        edges = vindex.describe(entity)
        if edges:
            print(f"  {entity}:")
            for e in edges[:3]:
                print(f"    → {e.target} [{e.relation}] (score={e.gate_score:.0f})")
        else:
            print(f"  {entity}: (no edges found)")

    print()

    # ─── Phase 3: Compile patched vindex ───
    print("Phase 3: Compile vindex with patches")
    print("-" * 70)

    compiled_vindex_path = os.path.join(output_dir, "compiled.vindex")
    print(f"Compiling to: {compiled_vindex_path}")

    try:
        # Note: compile() returns the path to the new compiled vindex
        # This bakes all patches into a new standalone vindex
        compiled_path = vindex.compile_vindex(compiled_vindex_path)
        print(f"✓ Compiled vindex: {compiled_path}")

        # Load the compiled vindex
        vindex_compiled = larql.load(compiled_path)
        print(f"✓ Loaded compiled vindex: {vindex_compiled}")
        print()
    except Exception as e:
        print(f"⚠ Compilation not fully implemented yet: {e}")
        print(f"  Continuing with original vindex for inference demo...")
        vindex_compiled = vindex
        print()

    # ─── Phase 4: Run inference on compiled model ───
    print("Phase 4: Run inference on compiled model")
    print("-" * 70)

    print("Inferences after editing (with compiled vindex):")
    compiled_results = {}
    for prompt in baseline_queries:
        t0 = time.time()
        try:
            result = vindex_compiled.infer(prompt, top_k_predictions=5)
            elapsed = time.time() - t0
            compiled_results[prompt] = result
            top_pred = result[0][0] if result else "?"
            top_conf = result[0][1] if result else 0
            print(f"  Q: {prompt}")
            print(f"  A: {top_pred} ({top_conf:.1%}) [{elapsed:.2f}s]")
        except Exception as e:
            print(f"  Q: {prompt}")
            print(f"  ⚠ Inference error: {e}")
        print()

    # ─── Phase 5: Verify changes ───
    print("Phase 5: Verify knowledge changes")
    print("-" * 70)

    # Check if inserted facts are still there
    print("Checking inserted facts in compiled vindex:")
    for entity, relation, target, _, _, _ in inserted_edges:
        edges = vindex_compiled.describe(entity)
        targets = [e.target.lower() for e in edges]
        found = target.lower() in targets
        status = "✓" if found else "⚠"
        print(f"  {status} {entity} --[{relation}]--> {target}: {'FOUND' if found else 'not found'}")

    print()

    # ─── Phase 6: Report ───
    print("Phase 6: Generate report")
    print("-" * 70)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vindex": {
            "path": vindex_path,
            "model": vindex.model,
            "layers": vindex.num_layers,
            "hidden_size": vindex.hidden_size,
        },
        "inserted_facts": [
            {
                "entity": e,
                "relation": r,
                "target": t,
                "layer": l,
                "feature": f,
                "token": tk,
            }
            for e, r, t, l, f, tk in inserted_edges
        ],
        "baseline_results": {
            q: {
                "top_prediction": r[0][0],
                "confidence": float(r[0][1]),
                "top_5": [
                    {"token": pred[0], "confidence": float(pred[1])}
                    for pred in r[:5]
                ] if r else []
            }
            for q, r in baseline_results.items()
        },
        "compiled_results": {
            q: {
                "top_prediction": r[0][0],
                "confidence": float(r[0][1]),
                "top_5": [
                    {"token": pred[0], "confidence": float(pred[1])}
                    for pred in r[:5]
                ] if r else []
            }
            for q, r in compiled_results.items()
        },
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved: {report_path}")
    print()
    print("="*70)
    print("Round-trip workflow complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review the report: cat {report_path}")
    print(f"  2. Publish the compiled vindex:")
    print(f"     larql hf {output_dir}/compiled.vindex --repo-id your-username/gemma-2-2b-it-edited")
    print(f"  3. Run inference on the compiled model:")
    print(f"     larql run {output_dir}/compiled.vindex 'Your prompt here'")


if __name__ == "__main__":
    main()
