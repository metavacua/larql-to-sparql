#!/usr/bin/env python3
"""
Reading the Query Assembly

Train tiny linear probes to read structured information from Gemma 3-4B's
residual stream at every layer. Instead of token-space projection (which
produces noise), ask: "what does the model KNOW at this layer?"

Probes:
  1. Task type (factual/code/narrative/reasoning/grammar/etc.)
  2. Relation type (capital_of/president_of/language_of/etc.)
  3. Entity presence (has entity identified?)
  4. Entity identity (which entity: France/Japan/Germany?)

Each probe is a single linear layer — if it reads the info, it's
explicitly encoded in the residual.
"""

import os
import sys
import json
import time
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-4b-pt"
MAX_SEQ = 64
OUTPUT_DIR = "results_probing"
PROBE_EPOCHS = 200
PROBE_LR = 1e-3

# ---------------------------------------------------------------------------
# Labelled prompts
# ---------------------------------------------------------------------------

LABELLED = {
    "factual_capital": {
        "prompts": [
            "The capital of France is", "The capital of Japan is",
            "The capital of Germany is", "The capital of Brazil is",
            "The capital of Egypt is", "The capital of Australia is",
            "The capital of Canada is", "The capital of India is",
            "The capital of Spain is", "The capital of Italy is",
        ],
        "task_type": "factual", "relation": "capital_of",
        "entity_list": ["France","Japan","Germany","Brazil","Egypt",
                        "Australia","Canada","India","Spain","Italy"],
    },
    "factual_president": {
        "prompts": [
            "The president of France is", "The president of Brazil is",
            "The president of Mexico is", "The president of Russia is",
            "The president of the United States is",
        ],
        "task_type": "factual", "relation": "president_of",
        "entity_list": ["France","Brazil","Mexico","Russia","United States"],
    },
    "factual_language": {
        "prompts": [
            "The official language of France is", "The official language of Japan is",
            "The official language of Brazil is", "The official language of Germany is",
            "The official language of China is",
        ],
        "task_type": "factual", "relation": "language_of",
        "entity_list": ["France","Japan","Brazil","Germany","China"],
    },
    "factual_currency": {
        "prompts": [
            "The currency of Japan is the", "The currency of India is the",
            "The currency of Brazil is the", "The currency of Mexico is the",
            "The currency of Russia is the",
        ],
        "task_type": "factual", "relation": "currency_of",
        "entity_list": ["Japan","India","Brazil","Mexico","Russia"],
    },
    "factual_science": {
        "prompts": [
            "Water boils at", "The speed of light is approximately",
            "The chemical symbol for gold is", "The Earth orbits the",
            "Photosynthesis converts sunlight into",
        ],
        "task_type": "factual", "relation": "science",
        "entity_list": [None,None,None,None,None],
    },
    "code": {
        "prompts": [
            "def fibonacci(n):\n    if n <= 1:\n        return",
            "for i in range(", "while x > 0:\n    x =",
            "class Node:\n    def __init__(self,",
            "if __name__ == '__main__':\n    ",
            "try:\n    result = api.get(url)\nexcept",
            "with open('file.txt') as f:\n    data =",
            "x = [i for i in range(",
            "import os\nif os.path.exists(",
            "async def fetch(url):\n    return await",
        ],
        "task_type": "code", "relation": "none",
        "entity_list": [None]*10,
    },
    "grammar": {
        "prompts": [
            "The big dog runs near the", "She quickly ran to the",
            "They have been", "The children are playing with their",
            "An extremely", "He carefully opened the old wooden",
            "The birds in the tree were", "I haven't seen them since",
            "Neither the cat nor the dog", "All of the students have already",
        ],
        "task_type": "grammar", "relation": "none",
        "entity_list": [None]*10,
    },
    "narrative": {
        "prompts": [
            "Once upon a time, there was a",
            "The detective opened the door and saw",
            "She picked up the old violin and began to",
            "The last human on Earth sat alone when suddenly",
            "In a world where dreams were currency,",
            "The old lighthouse keeper had a secret that",
            "Rain fell on the empty city as",
            "He reached into his pocket and found",
            "The letter was dated 1847 and read",
            "Somewhere in the distance, a bell",
        ],
        "task_type": "narrative", "relation": "none",
        "entity_list": [None]*10,
    },
    "reasoning": {
        "prompts": [
            "If all cats are mammals and Whiskers is a cat, then",
            "Given that x plus y equals 10 and x equals 3, then y equals",
            "The probability of rolling two sixes in a row is",
            "All roses are flowers. Some flowers fade quickly. Therefore",
            "2 plus 3 times 4 equals",
            "If A implies B and B implies C, then A implies",
            "The sequence 2, 4, 8, 16, so the next number is",
            "If the train leaves at 3pm at 60mph for 2 hours, it arrives at",
        ],
        "task_type": "reasoning", "relation": "none",
        "entity_list": [None]*8,
    },
    "conversational": {
        "prompts": [
            "I think the best approach would be to",
            "That is an interesting point, but",
            "Thank you for helping me with",
            "Could you explain what you mean by",
            "I have been thinking about this and",
            "The problem with that idea is",
            "What do you think about",
            "I completely agree that",
        ],
        "task_type": "conversational", "relation": "none",
        "entity_list": [None]*8,
    },
    "instructional": {
        "prompts": [
            "To make scrambled eggs, first",
            "The most effective way to learn a language is",
            "When debugging a program, start by",
            "The first rule of good writing is to",
            "To change a flat tire, begin by",
            "The key to effective communication is",
            "Before starting any exercise routine, you should",
            "Step 1: Install Python. Step 2:",
        ],
        "task_type": "instructional", "relation": "none",
        "entity_list": [None]*8,
    },
}


# ---------------------------------------------------------------------------
# Phase 1: Collect residuals
# ---------------------------------------------------------------------------

def collect_residuals(model, tokenizer, device, n_layers):
    """Extract prediction-position residuals at every layer."""
    print(f"\n  Phase 1: Collecting residuals...")

    pred_vecs = [None] * n_layers
    hooks = []

    for li in range(n_layers):
        layer = model.model.language_model.layers[li]

        def make_post_ffn(idx):
            def hook(module, args, output):
                pred_vecs[idx] = output[0, -1].detach().float().cpu()
            return hook

        hooks.append(layer.mlp.register_forward_hook(make_post_ffn(li)))

    samples = []  # (residuals_per_layer, task_type, relation, entity)
    t0 = time.time()
    total = sum(len(g["prompts"]) for g in LABELLED.values())
    done = 0

    with torch.no_grad():
        for group_name, group in LABELLED.items():
            for pi, prompt in enumerate(group["prompts"]):
                inputs = tokenizer(prompt, return_tensors="pt",
                                 max_length=MAX_SEQ, truncation=True).to(device)
                _ = model(**inputs)

                residuals = [pred_vecs[li].clone() if pred_vecs[li] is not None
                           else torch.zeros(model.config.text_config.hidden_size)
                           for li in range(n_layers)]

                entity = group["entity_list"][pi] if pi < len(group["entity_list"]) else None

                samples.append({
                    "residuals": residuals,
                    "task_type": group["task_type"],
                    "relation": group["relation"],
                    "entity": entity,
                    "prompt": prompt,
                })
                done += 1
                if done % 20 == 0:
                    print(f"    {done}/{total} ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()

    print(f"  Collected {len(samples)} samples in {time.time()-t0:.0f}s")
    return samples


# ---------------------------------------------------------------------------
# Phase 2: Train probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(X_train, y_train, X_test, y_test, n_classes, epochs=PROBE_EPOCHS):
    """Train a linear probe and return test accuracy."""
    in_dim = X_train.shape[1]
    probe = LinearProbe(in_dim, n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    for epoch in range(epochs):
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = probe(X_test)
        preds = test_logits.argmax(dim=-1)
        acc = (preds == y_test).float().mean().item()

    return acc, probe


def run_probes(samples, n_layers, hidden_dim):
    """Train all probes at all layers."""
    print(f"\n  Phase 2: Training probes...")

    # Build label sets
    task_types = sorted(set(s["task_type"] for s in samples))
    relations = sorted(set(s["relation"] for s in samples))
    entities = sorted(set(s["entity"] for s in samples if s["entity"] is not None))

    task_to_idx = {t: i for i, t in enumerate(task_types)}
    rel_to_idx = {r: i for i, r in enumerate(relations)}
    ent_to_idx = {e: i for i, e in enumerate(entities)}

    print(f"  Task types: {task_types}")
    print(f"  Relations: {relations}")
    print(f"  Entities: {len(entities)}")

    # Stratified train/test split — ensure every class appears in both sets
    import random
    rng = random.Random(42)

    # Group by task_type for stratification
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task_type"]].append(s)

    train_samples = []
    test_samples = []
    for task, task_samples in by_task.items():
        rng.shuffle(task_samples)
        n_test = max(2, len(task_samples) // 5)  # at least 2 per class in test
        test_samples.extend(task_samples[:n_test])
        train_samples.extend(task_samples[n_test:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")
    print(f"  Train tasks: {Counter(s['task_type'] for s in train_samples)}")
    print(f"  Test tasks: {Counter(s['task_type'] for s in test_samples)}")

    results = {}

    # --- Probe 1: Task type ---
    print(f"\n  Training TASK TYPE probes...")
    task_accs = []
    for li in range(n_layers):
        X_tr = torch.stack([s["residuals"][li] for s in train_samples])
        y_tr = torch.tensor([task_to_idx[s["task_type"]] for s in train_samples])
        X_te = torch.stack([s["residuals"][li] for s in test_samples])
        y_te = torch.tensor([task_to_idx[s["task_type"]] for s in test_samples])

        acc, _ = train_probe(X_tr, y_tr, X_te, y_te, len(task_types))
        task_accs.append(acc)

    results["task_type"] = {"accuracies": task_accs, "classes": task_types}

    # --- Probe 2: Relation type ---
    print(f"  Training RELATION probes...")
    rel_accs = []
    for li in range(n_layers):
        X_tr = torch.stack([s["residuals"][li] for s in train_samples])
        y_tr = torch.tensor([rel_to_idx[s["relation"]] for s in train_samples])
        X_te = torch.stack([s["residuals"][li] for s in test_samples])
        y_te = torch.tensor([rel_to_idx[s["relation"]] for s in test_samples])

        acc, _ = train_probe(X_tr, y_tr, X_te, y_te, len(relations))
        rel_accs.append(acc)

    results["relation"] = {"accuracies": rel_accs, "classes": relations}

    # --- Probe 3: Entity presence ---
    print(f"  Training ENTITY PRESENCE probes...")
    pres_accs = []
    for li in range(n_layers):
        X_tr = torch.stack([s["residuals"][li] for s in train_samples])
        y_tr = torch.tensor([1 if s["entity"] is not None else 0
                           for s in train_samples])
        X_te = torch.stack([s["residuals"][li] for s in test_samples])
        y_te = torch.tensor([1 if s["entity"] is not None else 0
                           for s in test_samples])

        acc, _ = train_probe(X_tr, y_tr, X_te, y_te, 2)
        pres_accs.append(acc)

    results["entity_presence"] = {"accuracies": pres_accs, "classes": ["no_entity", "has_entity"]}

    # --- Probe 4: Entity identity (factual only) ---
    print(f"  Training ENTITY IDENTITY probes...")
    factual_train = [s for s in train_samples if s["entity"] is not None]
    factual_test = [s for s in test_samples if s["entity"] is not None]

    # Only use entities that appear in BOTH train and test
    train_entities = set(s["entity"] for s in factual_train)
    test_entities = set(s["entity"] for s in factual_test)
    shared_entities = sorted(train_entities & test_entities)
    print(f"  Shared entities (train∩test): {len(shared_entities)}/{len(entities)}")

    if shared_entities and len(shared_entities) >= 2:
        shared_ent_to_idx = {e: i for i, e in enumerate(shared_entities)}
        factual_train = [s for s in factual_train if s["entity"] in shared_ent_to_idx]
        factual_test = [s for s in factual_test if s["entity"] in shared_ent_to_idx]

    ent_accs = []
    if factual_train and factual_test and len(shared_entities) >= 2:
        for li in range(n_layers):
            X_tr = torch.stack([s["residuals"][li] for s in factual_train])
            y_tr = torch.tensor([shared_ent_to_idx[s["entity"]] for s in factual_train])
            X_te = torch.stack([s["residuals"][li] for s in factual_test])
            y_te = torch.tensor([shared_ent_to_idx[s["entity"]] for s in factual_test])

            acc, _ = train_probe(X_tr, y_tr, X_te, y_te, len(shared_entities))
            ent_accs.append(acc)
        entities_used = shared_entities
    else:
        ent_accs = [0] * n_layers
        entities_used = []
        print(f"  WARNING: Not enough shared entities for identity probe")

    results["entity_identity"] = {"accuracies": ent_accs,
                                  "classes": entities_used if entities_used else entities}

    return results


# ---------------------------------------------------------------------------
# Phase 3: Information landscape
# ---------------------------------------------------------------------------

def print_landscape(probe_results, n_layers):
    """Display the information emergence map."""
    print(f"\n{'='*70}")
    print(f"  INFORMATION LANDSCAPE")
    print(f"{'='*70}")

    for probe_name, data in probe_results.items():
        accs = data["accuracies"]
        n_classes = len(data["classes"])
        chance = 1.0 / n_classes

        print(f"\n  {probe_name.upper()} ({n_classes} classes, chance={chance:.0%})")
        print(f"  {'L':>3} {'Accuracy':>10} {'Bar':>12} {'Phase'}")
        print(f"  {'─'*45}")

        first_90 = None
        first_80 = None

        for li in range(n_layers):
            acc = accs[li]
            bar = "█" * int(acc * 10)

            if acc >= 0.9 and first_90 is None:
                first_90 = li
            if acc >= 0.8 and first_80 is None:
                first_80 = li

            phase = ""
            if acc >= 0.9:
                phase = "KNOWN"
            elif acc >= 0.7:
                phase = "emerging"
            elif acc > chance + 0.1:
                phase = "weak signal"

            if li % 3 == 0 or li == n_layers - 1 or phase == "KNOWN" and (first_90 == li):
                print(f"  L{li:>2} {acc:>10.2f} {bar:>12s} {phase}")

        print(f"  First ≥80%: L{first_80}" if first_80 is not None else "  Never reaches 80%")
        print(f"  First ≥90%: L{first_90}" if first_90 is not None else "  Never reaches 90%")

    # Combined timeline
    print(f"\n  {'─'*70}")
    print(f"  QUERY ASSEMBLY TIMELINE:")
    for probe_name, data in probe_results.items():
        accs = data["accuracies"]
        first_80 = next((li for li, a in enumerate(accs) if a >= 0.8), None)
        first_90 = next((li for li, a in enumerate(accs) if a >= 0.9), None)
        print(f"    {probe_name:<20} emerges: L{first_80 or '?':<3} "
              f"confident: L{first_90 or '?':<3} "
              f"(final: {accs[-1]:.0%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  READING THE QUERY ASSEMBLY")
    print("  Linear probes on Gemma 3-4B residual stream")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"\n  Device: CPU (float32)")

    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()

    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    hidden_dim = tc.hidden_size
    print(f"  Loaded in {time.time()-t0:.0f}s: {n_layers}L, hidden={hidden_dim}")

    # Phase 1: Collect
    samples = collect_residuals(model, tokenizer, device, n_layers)

    # Free model to save memory for probe training
    del model
    import gc; gc.collect()
    print(f"  Freed model memory")

    # Phase 2: Train probes
    probe_results = run_probes(samples, n_layers, hidden_dim)

    # Phase 3: Display
    print_landscape(probe_results, n_layers)

    # ═══════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT: QUERY ASSEMBLY MAP")
    print(f"{'='*70}")

    task_first = next((li for li, a in enumerate(probe_results["task_type"]["accuracies"])
                      if a >= 0.8), None)
    rel_first = next((li for li, a in enumerate(probe_results["relation"]["accuracies"])
                     if a >= 0.8), None)
    ent_pres_first = next((li for li, a in enumerate(probe_results["entity_presence"]["accuracies"])
                          if a >= 0.8), None)
    ent_id_first = next((li for li, a in enumerate(probe_results["entity_identity"]["accuracies"])
                        if a >= 0.8), None)

    print(f"\n  Task type classified at: L{task_first or '?'}")
    print(f"  Relation identified at: L{rel_first or '?'}")
    print(f"  Entity presence at: L{ent_pres_first or '?'}")
    print(f"  Entity identity at: L{ent_id_first or '?'}")

    if task_first is not None and task_first <= 6:
        print(f"\n  ✓ EARLY TASK CLASSIFICATION (by L{task_first})")
        print(f"    The model knows what KIND of query this is within 6 layers.")
        print(f"    → Route to compute engine possible by L{task_first}")

    if rel_first is not None and ent_id_first is not None:
        if rel_first < ent_id_first:
            print(f"\n  ✓ RELATION BEFORE ENTITY (L{rel_first} < L{ent_id_first})")
            print(f"    The model classifies the relation type before identifying the entity.")
            print(f"    → Template first, parameters second. Like a SQL prepared statement.")
        else:
            print(f"\n  Entity before relation (L{ent_id_first} < L{rel_first})")
            print(f"    The model identifies the entity before classifying the relation.")

    # The big picture
    if (task_first and task_first <= 5 and
        rel_first and rel_first <= 10 and
        ent_id_first and ent_id_first <= 15):
        print(f"\n  ═══════════════════════════════════════════════")
        print(f"  THE QUERY ASSEMBLES IN LAYERS:")
        print(f"    L{task_first}: 'This is a factual query'     (task type)")
        print(f"    L{rel_first}: 'About capital_of'             (relation)")
        print(f"    L{ent_id_first}: 'For France'                  (entity)")
        print(f"    L{ent_id_first}+: FFN retrieves 'Paris'        (answer)")
        print(f"  ═══════════════════════════════════════════════")

    # Save
    save_data = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_samples": len(samples),
        "probes": {
            name: {
                "accuracies": data["accuracies"],
                "classes": data["classes"],
                "first_80": next((li for li, a in enumerate(data["accuracies"])
                                 if a >= 0.8), None),
                "first_90": next((li for li, a in enumerate(data["accuracies"])
                                 if a >= 0.9), None),
                "final_acc": data["accuracies"][-1],
            }
            for name, data in probe_results.items()
        },
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
