#!/usr/bin/env python3
"""
LQL corpus generator — codebase as training data.

Reads the LARQL source tree (ast.rs, parser tests, docs, lql-guide.md)
and generates a systematic NL→LQL corpus covering every Statement variant
in the grammar.

The key principle: the grammar, parser tests, and documentation are the
ground truth. We do not invent examples — we extract them from the source
and apply template expansions grounded in the formal grammar.

Sources read:
  crates/larql-lql/src/ast.rs             — Statement variant taxonomy
  crates/larql-lql/src/parser/tests.rs    — canonical LQL syntax examples
  docs/lql-guide.md                       — human descriptions + usage
  docs/training-free-insert.md            — INSERT workflow examples
  crates/larql-wasm/tools/lql_corpus.json — existing seed corpus

Output:
  crates/larql-wasm/tools/lql_corpus_full.json

Usage:
  python lql_corpus_gen.py [--repo-root /path/to/larql-to-sparql]
"""
import argparse
import json
import re
import sys
from pathlib import Path


# ── NL templates per statement type ──────────────────────────────────────────
# Each entry: (first_token, statement_type, category, [NL prompts], [lql_examples])
#
# NL prompts are grounded in the lql-guide.md descriptions + grammar patterns.
# LQL examples come from parser/tests.rs and the guide.
#
# Entities and values are varied to help the KNN keys span the prompt space.

TEMPLATES = [
    # ── QUERY ─────────────────────────────────────────────────────────────────
    ("DESCRIBE", "Describe", "query", [
        "Describe what the model knows about {entity}",
        "What does the model know about {entity}?",
        "Show me the knowledge graph for {entity}",
        "Tell me about {entity} in the vindex",
        "What relations does the model have for {entity}?",
        "Look up {entity} in the vindex",
        "Show all edges involving {entity}",
        "DESCRIBE {entity}",
        "Describe {entity} at layer {layer}",
        "Show relations only for {entity}",
        "Describe {entity} in the knowledge band",
        "What does the model associate with {entity}?",
    ], [
        'DESCRIBE "{entity}";',
        'DESCRIBE "{entity}" AT LAYER {layer};',
        'DESCRIBE "{entity}" RELATIONS ONLY;',
        'DESCRIBE "{entity}" KNOWLEDGE;',
        'DESCRIBE "{entity}" VERBOSE;',
        'DESCRIBE "{entity}" BRIEF;',
        'DESCRIBE "{entity}" ALL LAYERS;',
    ]),

    ("WALK", "Walk", "query", [
        "Walk through the model for the prompt '{prompt}'",
        "Show me which features activate for '{prompt}'",
        "Which gate features fire for the sentence '{prompt}'?",
        "Trace the feature activations for '{prompt}'",
        "Run a sparse walk for '{prompt}'",
        "What features does the model activate for '{prompt}'?",
        "WALK '{prompt}' TOP 10",
        "Do a gate KNN walk on '{prompt}'",
        "Show top activating features for '{prompt}'",
        "Feature scan for the prompt '{prompt}'",
        "Walk the FFN graph for '{prompt}'",
        "Which vindex features respond to '{prompt}'?",
    ], [
        'WALK "{prompt}";',
        'WALK "{prompt}" TOP 10;',
        'WALK "{prompt}" TOP 5 LAYERS 20-33;',
        'WALK "{prompt}" MODE pure;',
        'WALK "{prompt}" MODE hybrid;',
        'WALK "{prompt}" TOP 10 COMPARE;',
    ]),

    ("INFER", "Infer", "query", [
        "Run inference on the prompt '{prompt}'",
        "What would the model predict to complete '{prompt}'?",
        "INFER '{prompt}'",
        "Get the model's top predictions for '{prompt}'",
        "Complete the sentence '{prompt}'",
        "What does the model output for '{prompt}'?",
        "Run a full forward pass on '{prompt}'",
        "Predict the next token after '{prompt}'",
        "What is the model's top-5 for '{prompt}'?",
        "Run attention-based inference on '{prompt}'",
        "What does the model think comes after '{prompt}'?",
        "Inference result for '{prompt}'",
    ], [
        'INFER "{prompt}";',
        'INFER "{prompt}" TOP 5;',
        'INFER "{prompt}" TOP 10;',
        'INFER "{prompt}" COMPARE;',
    ]),

    ("SELECT", "Select", "query", [
        "Find all edges where relation is '{relation}'",
        "List edges with entity '{entity}'",
        "Query the edge graph for '{entity}'",
        "SELECT from edges where layer is {layer}",
        "Find the top edges by confidence",
        "Get all edges for '{entity}' with confidence above {threshold}",
        "Show edges near '{entity}' at layer {layer}",
        "Query features at layer {layer}",
        "List all knowledge graph edges",
        "Find edges where target is '{entity}'",
        "SELECT edges filtered by relation",
        "SQL query over the vindex edges",
        "Get edges ordered by confidence",
        "Find edges matching entity like '{entity}%'",
        "List features at layer {layer} with high confidence",
    ], [
        'SELECT * FROM EDGES;',
        'SELECT * FROM EDGES WHERE relation = "{relation}";',
        'SELECT * FROM EDGES WHERE entity = "{entity}";',
        'SELECT * FROM EDGES WHERE layer = {layer} LIMIT 20;',
        'SELECT * FROM EDGES WHERE confidence > {threshold};',
        'SELECT entity, target FROM EDGES NEAREST TO "{entity}" AT LAYER {layer} LIMIT 20;',
        'SELECT * FROM EDGES ORDER BY confidence DESC LIMIT 10;',
        'SELECT * FROM EDGES WHERE entity LIKE "{entity}%";',
    ]),

    ("EXPLAIN", "Explain", "query", [
        "Explain how the model processes '{prompt}'",
        "Show the per-layer feature trace for '{prompt}'",
        "Explain the walk for '{prompt}'",
        "EXPLAIN WALK '{prompt}'",
        "Explain the inference for '{prompt}'",
        "Show layer-by-layer breakdown for '{prompt}'",
        "What features and layers activate for '{prompt}'?",
        "Debug the model's processing of '{prompt}'",
        "Show verbose feature trace for '{prompt}'",
        "Explain INFER for '{prompt}'",
        "Decompose the model's computation for '{prompt}'",
        "Show per-layer walk explanation for '{prompt}'",
    ], [
        'EXPLAIN WALK "{prompt}";',
        'EXPLAIN INFER "{prompt}";',
        'EXPLAIN WALK "{prompt}" VERBOSE;',
        'EXPLAIN WALK "{prompt}" LAYERS 20-33;',
        'EXPLAIN WALK "{prompt}" TOP 10;',
    ]),

    # ── TRACE ─────────────────────────────────────────────────────────────────
    ("TRACE", "Trace", "trace", [
        "Trace the residual stream for '{prompt}'",
        "Capture the per-layer residuals for '{prompt}'",
        "TRACE '{prompt}'",
        "Get a residual stream decomposition for '{prompt}'",
        "Record the attention and FFN deltas for '{prompt}'",
        "Capture the residual trace for '{prompt}'",
        "Decompose the forward pass for '{prompt}'",
        "Show the residual at each layer for '{prompt}'",
        "Trace and save the residual stream for '{prompt}'",
        "Capture per-position residuals for '{prompt}'",
    ], [
        'TRACE "{prompt}";',
        'TRACE "{prompt}" FOR "{entity}";',
        'TRACE "{prompt}" DECOMPOSE;',
        'TRACE "{prompt}" LAYERS 20-33;',
        'TRACE "{prompt}" SAVE "trace.bin";',
    ]),

    # ── MUTATION ──────────────────────────────────────────────────────────────
    ("INSERT", "Insert", "mutation", [
        "Insert the fact that {entity} {relation} {target}",
        "Add a knowledge edge: {entity} {relation} {target}",
        "Install the fact that {entity} {relation} is {target}",
        "Store the relation {entity} → {relation} → {target}",
        "Add a new edge connecting {entity} to {target}",
        "INSERT INTO EDGES: {entity} {relation} {target}",
        "Put the fact {entity} {relation} {target} into the vindex",
        "Record that {entity} has relation {relation} to {target}",
        "Install knowledge: {entity} is related to {target} by {relation}",
        "Add the edge ({entity}, {relation}, {target}) to the graph",
        "Store new knowledge: {relation} of {entity} is {target}",
        "Write the fact that {entity}'s {relation} is {target}",
    ], [
        'INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}");',
        'INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") AT LAYER {layer};',
        'INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") CONFIDENCE 0.9;',
        'INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") MODE KNN;',
        'INSERT INTO EDGES (entity, relation, target) VALUES ("{entity}", "{relation}", "{target}") MODE COMPOSE ALPHA 0.25;',
    ]),

    ("DELETE", "Delete", "mutation", [
        "Delete the edge where entity is '{entity}'",
        "Remove edges where relation is '{relation}'",
        "Delete all edges with confidence below {threshold}",
        "Erase the knowledge about {entity}",
        "Remove the {entity} → {relation} edge",
        "Delete edges at layer {layer}",
        "Clear edges matching entity '{entity}'",
        "Remove the fact connecting {entity} to {target}",
        "Delete edges where confidence < {threshold}",
        "Erase all edges for entity '{entity}'",
    ], [
        'DELETE FROM EDGES WHERE entity = "{entity}";',
        'DELETE FROM EDGES WHERE relation = "{relation}";',
        'DELETE FROM EDGES WHERE confidence < {threshold};',
        'DELETE FROM EDGES WHERE entity = "{entity}" AND relation = "{relation}";',
        'DELETE FROM EDGES WHERE layer = {layer};',
    ]),

    ("UPDATE", "Update", "mutation", [
        "Update the confidence of edges for '{entity}' to {threshold}",
        "Change the target of the '{entity}' {relation} edge",
        "Set confidence to {threshold} for all edges in the knowledge band",
        "Modify the relation label for '{entity}'",
        "Update edges where entity is '{entity}'",
        "Set layer to {layer} for edges matching '{entity}'",
        "Change confidence values for '{relation}' edges",
        "Update the target for '{entity}'s '{relation}' edge to '{target}'",
        "Edit the edge ({entity}, {relation}) to point to {target}",
        "Adjust confidence scores for edges above layer {layer}",
    ], [
        'UPDATE EDGES SET confidence = {threshold} WHERE entity = "{entity}";',
        'UPDATE EDGES SET target = "{target}" WHERE entity = "{entity}" AND relation = "{relation}";',
        'UPDATE EDGES SET confidence = {threshold} WHERE layer >= {layer};',
        'UPDATE EDGES SET relation = "{relation}" WHERE entity = "{entity}";',
    ]),

    ("MERGE", "Merge", "mutation", [
        "Merge the patch from another vindex into the current one",
        "Combine the current vindex with {vindex}",
        "Merge {vindex} into the active session",
        "Integrate changes from {vindex} into the current model",
        "Join two vindexes together",
        "Combine vindexes keeping highest confidence on conflicts",
        "Merge knowledge from {vindex} keeping the source on conflicts",
        "Apply a knowledge merge from {vindex}",
        "Merge {vindex} into current, resolving conflicts by confidence",
        "Blend two vindexes together",
    ], [
        'MERGE SOURCE "{vindex}";',
        'MERGE SOURCE "{vindex}" ON CONFLICT HIGHEST_CONFIDENCE;',
        'MERGE SOURCE "{vindex}" ON CONFLICT KEEP_SOURCE;',
        'MERGE SOURCE "{vindex}" ON CONFLICT KEEP_TARGET;',
    ]),

    ("REBALANCE", "Rebalance", "mutation", [
        "Rebalance the inserted facts until they converge",
        "Run a rebalance pass with floor {threshold} and ceiling {ceiling}",
        "Calibrate inserted knowledge confidence scores",
        "Rebalance until converged, max {n} iterations",
        "Balance the confidence of installed knowledge edges",
        "Run REBALANCE to fix confidence drift",
        "Rebalance with floor probability {threshold}",
        "Calibrate all compose-mode insertions",
        "Run REBALANCE UNTIL CONVERGED",
        "Adjust installed edge confidence scores to target range",
    ], [
        'REBALANCE;',
        'REBALANCE UNTIL CONVERGED;',
        'REBALANCE MAX {n};',
        'REBALANCE FLOOR {threshold} CEILING {ceiling};',
        'REBALANCE UNTIL CONVERGED MAX {n} FLOOR {threshold} CEILING {ceiling};',
    ]),

    # ── LIFECYCLE ─────────────────────────────────────────────────────────────
    ("EXTRACT", "Extract", "lifecycle", [
        "Extract a vindex from the model {model}",
        "Download and index {model} for browsing",
        "Build a vindex from {model} with inference support",
        "Extract {model} into a new vindex with all weights",
        "Create a vindex from {model}",
        "Index {model} for WALK and DESCRIBE",
        "Extract {model} with INFERENCE level",
        "Build a browse-only vindex from {model}",
        "EXTRACT MODEL {model} INTO a new vindex",
        "Download {model} weights and build the vindex",
        "Index the model {model} into {output}",
        "Create a full extraction of {model} including lm_head",
    ], [
        'EXTRACT MODEL "{model}" INTO "{output}";',
        'EXTRACT MODEL "{model}" INTO "{output}" WITH INFERENCE;',
        'EXTRACT MODEL "{model}" INTO "{output}" WITH ALL;',
        'EXTRACT MODEL "{model}" INTO "{output}" COMPONENTS FFN_GATE, FFN_DOWN;',
    ]),

    ("COMPILE", "Compile", "lifecycle", [
        "Compile the current vindex with patches baked in",
        "Bake the current patches into a new standalone vindex",
        "Export the patched model as safetensors",
        "COMPILE CURRENT INTO VINDEX at {output}",
        "Bake all changes into a clean vindex",
        "Compile the model to GGUF format",
        "Export the current model as a safetensors file",
        "Compile the patched vindex to {output}",
        "Bake patches into {output}",
        "COMPILE to produce a standalone model file",
        "Create a compiled vindex with all edits baked in",
        "Export compiled model weights",
    ], [
        'COMPILE CURRENT INTO VINDEX "{output}";',
        'COMPILE CURRENT INTO MODEL "{output}";',
        'COMPILE CURRENT INTO MODEL "{output}" FORMAT GGUF;',
        'COMPILE CURRENT INTO MODEL "{output}" FORMAT SAFETENSORS;',
    ]),

    ("DIFF", "Diff", "lifecycle", [
        "Show the differences between two vindexes",
        "Compare {vindex_a} and {vindex_b}",
        "Diff {vindex_a} against {vindex_b} at layer {layer}",
        "What changed between {vindex_a} and {vindex_b}?",
        "Show the delta between two vindex versions",
        "Compare two vindexes and save as a patch",
        "DIFF {vindex_a} {vindex_b}",
        "Find what edges differ between {vindex_a} and {vindex_b}",
        "Generate a diff of two vindexes",
        "Show changes from {vindex_a} to {vindex_b}",
    ], [
        'DIFF "{vindex_a}" "{vindex_b}";',
        'DIFF "{vindex_a}" "{vindex_b}" AT LAYER {layer};',
        'DIFF "{vindex_a}" CURRENT;',
        'DIFF "{vindex_a}" "{vindex_b}" INTO PATCH "{output}";',
    ]),

    ("USE", "Use", "lifecycle", [
        "Load a vindex to start querying it",
        "Switch to using {vindex}",
        "Connect to the vindex at {vindex}",
        "USE {vindex}",
        "Open {vindex} for queries",
        "Switch to a remote LQL server at {url}",
        "Use the model {model} directly",
        "Load {vindex} as the active vindex",
        "Connect to {url} for remote inference",
        "USE MODEL {model}",
        "Switch the active vindex to {vindex}",
        "Open the vindex {vindex} for browsing",
    ], [
        'USE "{vindex}";',
        'USE MODEL "{model}";',
        'USE REMOTE "{url}";',
        'USE "{vindex}" AUTO EXTRACT;',
    ]),

    # ── INTROSPECTION ─────────────────────────────────────────────────────────
    ("SHOW", "ShowRelations", "introspection", [
        "Show all relations in the model",
        "List the available relations in the vindex",
        "What relations exist at layer {layer}?",
        "Show relations with examples",
        "List all edge relation types",
        "SHOW RELATIONS",
        "What relation labels are in the vindex?",
        "Show verbose relation listing",
        "Show raw relation data",
        "List relations with examples at layer {layer}",
    ], [
        'SHOW RELATIONS;',
        'SHOW RELATIONS AT LAYER {layer};',
        'SHOW RELATIONS VERBOSE;',
        'SHOW RELATIONS VERBOSE WITH EXAMPLES;',
        'SHOW RELATIONS WITH EXAMPLES;',
        'SHOW RELATIONS RAW;',
    ]),

    ("SHOW", "ShowLayers", "introspection", [
        "List all layers in the vindex",
        "Show the layer range of the vindex",
        "SHOW LAYERS",
        "How many layers does the vindex have?",
        "List layers in range {layer_start}-{layer_end}",
        "Show available layers",
        "What layers are in the vindex?",
        "Display the layer structure",
    ], [
        'SHOW LAYERS;',
        'SHOW LAYERS {layer_start}-{layer_end};',
        'SHOW LAYERS RANGE 0-33;',
    ]),

    ("SHOW", "ShowFeatures", "introspection", [
        "Show features at layer {layer}",
        "List features at layer {layer} with high confidence",
        "SHOW FEATURES {layer}",
        "What features exist at layer {layer}?",
        "Show features at layer {layer} for relation '{relation}'",
        "Display feature metadata at layer {layer}",
        "List all features at layer {layer}",
    ], [
        'SHOW FEATURES {layer};',
        'SHOW FEATURES {layer} WHERE relation = "{relation}";',
        'SHOW FEATURES {layer} WHERE confidence > {threshold} LIMIT 20;',
    ]),

    ("SHOW", "ShowModels", "introspection", [
        "Show all loaded models",
        "List available models",
        "SHOW MODELS",
        "What models are loaded?",
        "Display loaded model list",
        "Which models are available in this session?",
    ], [
        'SHOW MODELS;',
    ]),

    ("STATS", "Stats", "introspection", [
        "Show statistics about the current vindex",
        "Print a summary of vindex storage and feature counts",
        "STATS",
        "How large is the current vindex?",
        "Display vindex storage statistics",
        "Show vindex stats",
        "How many features and layers does the vindex have?",
        "Report storage usage for the current vindex",
        "Print storage stats",
        "Vindex statistics summary",
    ], [
        'STATS;',
        'STATS "{vindex}";',
    ]),

    ("COMPACT", "CompactMinor", "introspection", [
        "Run minor compaction to reclaim space in the vindex",
        "COMPACT MINOR",
        "Do a minor compaction",
        "Reclaim deleted feature space",
        "Run a small compaction pass",
        "Minor compact the vindex",
    ], [
        'COMPACT MINOR;',
    ]),

    ("COMPACT", "CompactMajor", "introspection", [
        "Run a full major compaction on the vindex",
        "COMPACT MAJOR FULL",
        "Do a full compaction with lambda tuning",
        "Run major compaction to rebuild the vindex",
        "Compact major with lambda {lambda_val}",
        "Rebuild the vindex through major compaction",
    ], [
        'COMPACT MAJOR;',
        'COMPACT MAJOR FULL;',
        'COMPACT MAJOR WITH LAMBDA = {lambda_val};',
    ]),

    # ── PATCH ─────────────────────────────────────────────────────────────────
    ("BEGIN", "BeginPatch", "patch", [
        "Start a new patch session to track mutations",
        "BEGIN PATCH '{patch_file}'",
        "Start recording changes to {patch_file}",
        "Open a new patch file for edits",
        "Begin recording a patch",
        "Start a patch session",
        "Create a new patch {patch_file}",
        "Begin patch to record upcoming mutations",
    ], [
        'BEGIN PATCH "{patch_file}";',
    ]),

    ("SAVE", "SavePatch", "patch", [
        "Save the current patch to disk",
        "SAVE PATCH",
        "Persist the current patch",
        "Write the active patch to disk",
        "Save all recorded mutations to the patch file",
        "Flush the patch to disk",
    ], [
        'SAVE PATCH;',
    ]),

    ("APPLY", "ApplyPatch", "patch", [
        "Apply an existing patch file to the current vindex",
        "APPLY PATCH '{patch_file}'",
        "Load and apply {patch_file}",
        "Apply changes from {patch_file}",
        "Install patch {patch_file} onto the current vindex",
        "Load {patch_file} and apply its changes",
        "Stack patch {patch_file} onto the current session",
        "Apply knowledge patch from {patch_file}",
    ], [
        'APPLY PATCH "{patch_file}";',
    ]),

    ("SHOW", "ShowPatches", "patch", [
        "List all patches currently loaded",
        "Show active patches",
        "SHOW PATCHES",
        "What patches are applied?",
        "List loaded patch files",
        "Display the patch stack",
    ], [
        'SHOW PATCHES;',
    ]),

    ("REMOVE", "RemovePatch", "patch", [
        "Remove a patch from the current session",
        "REMOVE PATCH '{patch_file}'",
        "Unload patch {patch_file}",
        "Revert the changes from {patch_file}",
        "Remove {patch_file} from the active patch stack",
        "Discard patch {patch_file}",
    ], [
        'REMOVE PATCH "{patch_file}";',
    ]),
]

# Template variable expansions — varied entities, layers, etc.
EXPANSIONS = {
    "entity": ["France", "Germany", "Japan", "Mozart", "Einstein", "Atlantis",
                "Python", "calculus", "aspirin", "Newton", "Marie Curie", "Rome",
                "the Eiffel Tower", "relativity", "DNA", "quantum mechanics"],
    "relation": ["capital-of", "capital", "invented-by", "treats", "side-effect",
                 "located-in", "part-of", "created-by", "instance-of", "author-of"],
    "target": ["Paris", "Berlin", "Poseidon", "Neptune", "Newton", "bleeding",
               "headache", "France", "Europe", "Marie Curie"],
    "prompt": ["The capital of France is", "The largest planet is",
               "Water boils at", "The inventor of calculus was",
               "Mount Everest is the", "The speed of light is",
               "The capital of Germany is", "DNA stands for"],
    "layer": ["24", "26", "20", "28", "16"],
    "layer_start": ["0", "10", "20"],
    "layer_end": ["10", "20", "33"],
    "threshold": ["0.5", "0.7", "0.8", "0.3"],
    "ceiling": ["0.9", "0.95"],
    "n": ["16", "32", "64"],
    "model": ["google/gemma-3-4b-it", "meta-llama/Llama-3.2-1B-Instruct",
              "microsoft/Phi-3-mini-4k-instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
    "output": ["output/compiled.vindex", "output/model.safetensors",
               "output/model.gguf", "output/edited.vindex"],
    "vindex": ["gemma3-4b.vindex", "base.vindex", "v2.vindex", "edited.vindex"],
    "vindex_a": ["v1.vindex", "base.vindex", "original.vindex"],
    "vindex_b": ["v2.vindex", "patched.vindex", "edited.vindex"],
    "url": ["http://localhost:8080", "https://larql.example.com"],
    "patch_file": ["medical.vlp", "facts.vlp", "edits.vlp", "knowledge.vlp"],
    "lambda_val": ["0.001", "0.01", "0.1"],
}


def expand(template: str, idx: int = 0) -> str:
    """Fill template variables using round-robin from EXPANSIONS."""
    result = template
    for key, values in EXPANSIONS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, values[idx % len(values)])
    return result


VALID_FIRST_TOKENS = {
    "DESCRIBE", "WALK", "INFER", "SELECT", "EXPLAIN",
    "TRACE", "INSERT", "DELETE", "UPDATE", "MERGE", "REBALANCE",
    "EXTRACT", "COMPILE", "DIFF", "USE",
    "SHOW", "STATS", "COMPACT",
    "BEGIN", "SAVE", "APPLY", "REMOVE",
}


def extract_lql_examples_from_tests(repo_root: Path) -> list[dict]:
    """Pull canonical LQL examples directly from parser/tests.rs."""
    test_file = repo_root / "crates/larql-lql/src/parser/tests.rs"
    examples = []
    if not test_file.exists():
        return examples

    text = test_file.read_text()
    # Match parse("...") and parse(r#"..."#) patterns — require complete statements
    for m in re.finditer(r'parse\(r?#?"([A-Z][^"#]{4,})"', text):
        lql = m.group(1).strip()
        if not lql.endswith(";"):
            lql += ";"
        first_word = lql.split()[0].rstrip(";")
        if first_word not in VALID_FIRST_TOKENS:
            continue
        examples.append({"source": "parser_test", "lql": lql, "first_token": first_word})
    return examples


def extract_examples_from_guide(repo_root: Path) -> list[dict]:
    """Pull LQL examples from the lql-guide.md documentation."""
    guide = repo_root / "docs/lql-guide.md"
    examples = []
    if not guide.exists():
        return examples

    text = guide.read_text()
    # Match SQL code blocks
    for block in re.finditer(r"```sql\n(.*?)```", text, re.DOTALL):
        for line in block.group(1).splitlines():
            line = line.strip()
            if line and not line.startswith("--") and not line.startswith("larql>"):
                first_word = line.split()[0] if line.split() else ""
                if first_word.isupper() and len(first_word) >= 2:
                    examples.append({
                        "source": "lql_guide",
                        "lql": line if line.endswith(";") else line + ";",
                        "first_token": first_word,
                    })
    return examples


def generate_corpus(repo_root: Path) -> list[dict]:
    """Generate the full NL→LQL corpus from the codebase."""
    corpus = []
    seen_prompts = set()

    # ── Source 1: Template-generated entries ─────────────────────────────────
    for first_token, stmt_type, category, nl_templates, lql_templates in TEMPLATES:
        # Generate multiple expansions of each NL template
        for i, nl_tmpl in enumerate(nl_templates):
            # Use two different entity/variable expansions per template
            for offset in range(3):
                prompt = expand(nl_tmpl, i + offset)
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                lql = expand(lql_templates[i % len(lql_templates)], i + offset)
                # Determine train/eval split: first 2 expansions → train, rest → eval
                split = "train" if offset < 2 else "eval"
                corpus.append({
                    "prompt": prompt,
                    "expected_lql": lql,
                    "first_token": first_token,
                    "statement_type": stmt_type,
                    "category": category,
                    "split": split,
                    "source": "template",
                })

    # ── Source 2: Parser test examples as eval prompts ────────────────────────
    # These are canonical LQL statements from the test suite.
    # We generate NL descriptions from them.
    stmt_to_nl = {
        "DESCRIBE": "Show the model's knowledge for this DESCRIBE statement",
        "WALK": "Run this WALK feature scan",
        "INFER": "Run this INFER statement",
        "SELECT": "Execute this SELECT query over the vindex edges",
        "EXPLAIN": "Run this EXPLAIN command",
        "TRACE": "Run this TRACE command",
        "INSERT": "Execute this INSERT statement to add knowledge",
        "DELETE": "Run this DELETE to remove edges",
        "UPDATE": "Execute this UPDATE statement",
        "MERGE": "Run this MERGE command",
        "REBALANCE": "Run this REBALANCE pass",
        "EXTRACT": "Run this EXTRACT command",
        "COMPILE": "Execute this COMPILE command",
        "DIFF": "Run this DIFF command",
        "USE": "Execute this USE statement",
        "SHOW": "Run this SHOW command",
        "STATS": "Show these statistics",
        "COMPACT": "Run this COMPACT command",
        "BEGIN": "Execute this BEGIN PATCH command",
        "SAVE": "Run SAVE PATCH",
        "APPLY": "Run this APPLY PATCH command",
        "REMOVE": "Run this REMOVE PATCH command",
    }

    # ── Source 3: Existing seed corpus ────────────────────────────────────────
    seed_path = repo_root / "crates/larql-wasm/tools/lql_corpus.json"
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
        for entry in seed:
            if entry["prompt"] not in seen_prompts:
                seen_prompts.add(entry["prompt"])
                corpus.append({**entry, "source": "seed"})

    # ── Source 4: Direct LQL-as-prompt entries ────────────────────────────────
    # The LQL statement itself is a valid (if terse) prompt.
    lql_examples = extract_lql_examples_from_tests(repo_root)
    lql_examples += extract_examples_from_guide(repo_root)

    for ex in lql_examples:
        lql = ex["lql"]
        first_token = ex["first_token"]
        if first_token not in VALID_FIRST_TOKENS:
            continue
        # Trim long LQL to use as a prompt
        prompt = lql[:80].rstrip(";").strip()
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        corpus.append({
            "prompt": prompt,
            "expected_lql": lql,
            "first_token": first_token,
            "statement_type": first_token.capitalize(),
            "category": _category(first_token),
            "split": "eval",
            "source": ex["source"],
        })

    return corpus


def _category(first_token: str) -> str:
    query = {"WALK", "INFER", "SELECT", "DESCRIBE", "EXPLAIN"}
    mutation = {"INSERT", "DELETE", "UPDATE", "MERGE", "REBALANCE"}
    lifecycle = {"EXTRACT", "COMPILE", "DIFF", "USE"}
    patch = {"BEGIN", "SAVE", "APPLY", "REMOVE"}
    trace = {"TRACE"}
    if first_token in query:
        return "query"
    if first_token in mutation:
        return "mutation"
    if first_token in lifecycle:
        return "lifecycle"
    if first_token in patch:
        return "patch"
    if first_token in trace:
        return "trace"
    return "introspection"


def main():
    ap = argparse.ArgumentParser(description="Generate LQL NL→LQL corpus from the codebase")
    ap.add_argument("--repo-root", default=".", help="Path to the larql-to-sparql repo root")
    ap.add_argument("--output", default="crates/larql-wasm/tools/lql_corpus_full.json")
    ap.add_argument("--stats", action="store_true", help="Print per-category statistics")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    corpus = generate_corpus(repo)

    # Write output relative to repo root
    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2))

    # Summary
    from collections import Counter
    by_type = Counter(e["first_token"] for e in corpus)
    by_split = Counter(e["split"] for e in corpus)
    by_source = Counter(e["source"] for e in corpus)

    print(f"Generated {len(corpus)} corpus entries → {out_path}")
    print(f"  Split:  train={by_split['train']}  eval={by_split['eval']}")
    print(f"  Source: {dict(by_source)}")
    if args.stats:
        print("\nPer statement type:")
        for tok, count in sorted(by_type.items()):
            print(f"  {tok:12s}  {count}")


if __name__ == "__main__":
    main()
