# LARQL Demo: The Model is a Database

## Introduction

10,240 features per layer. 34 layers. 348,160 edges total in Gemma 4B. Every transformer is a graph database that forgot it was a graph database. LARQL lets you query it.

---

## Act 1: Entities — What Does the Model Know?

Start with something familiar. What does the model know about France?

```sql
DESCRIBE "France";
```

Three stages come back, matching the residual stream architecture:

**Syntax (L0-13)** — The model hasn't figured out what France means yet. Intro, Spanish, internasional — surface co-occurrence statistics, not knowledge. This is the dark accumulation phase.

**Edges (L14-27)** — Where knowledge lives. francais at L23, Europe and Italy tagged as nationality, country tagged as borders, Spain at L18. But also Australia at 25.1, CEO, Fountain — polysemantic noise sharing the same slots. The facts are here, buried among unrelated concepts.

**Output (L28-33)** — The model commits to an answer. French dominates at 35.2. German, European, Western — all plausible next tokens depending on context. Wilk at L31 is noise that won't survive to the final prediction.

Syntax gathers context. Edges store knowledge. Output commits to a token. That's the three-stage architecture of every transformer, visible in one query.

Now pull on one of those relations:

```sql
SELECT * FROM EDGES WHERE entity = "France" AND relation = "borders" AND score > 5;
```

L25 F5067 — country [Country, country, Country], score 10.3. Three capitalisation variants of the same token. This is a real fact — France borders other countries — stored in one feature at one layer.

The score filter cuts the noise. Real edges score 5+. Probe artifacts hover near zero or go negative. LARQL doesn't hide the noise — it gives you the score so you can filter it yourself.

Drill into nationality:

```sql
SELECT * FROM EDGES WHERE entity = "France" AND relation = "nationality" LIMIT 5;
```

Italy [Germany, Sweden, Italy] at L19-L20. USA [USA, България, Germany] at L20. The model groups countries by the nationality relation — a real pattern it learned from text.

Compare with Einstein:

```sql
DESCRIBE "Einstein";
```

Same three-stage structure. Physics, gravity, atom in the knowledge band. Nobel at L26. German at L23. Different facts, same architecture. The model learned a schema — not just individual facts.

What entities sit nearest to France in the feature space?

```sql
SELECT * FROM EDGES NEAREST TO "France" AT LAYER 26 LIMIT 10;
```

Australia [Italy, Germany, Spain] — a countries cluster. French [French, french, FRENCH]. euros [€, EU, Euros]. Channel [Channel, channel, channel]. Geographic and cultural neighbours, found by gate-vector cosine similarity.

---

## Act 2: The Raw Graph — What It Actually Looks Like

That was the clean view. Now here's what the FFN actually looks like under the hood.

```sql
SHOW FEATURES 26;
```

34 features shown by default. Most look like noise — F3 is Cyrillic, F12 is Gujarati numerals, F15 is Hebrew. But look closer.

F2 — five. Also: 5, फाइव, cinq, vijf. English, digit, Hindi, French, Dutch. Five different representations of the same number. This feature fires for the *concept* of five, not the token.

F9 — runners. Also: Runner, Runners, Runner, runner. Every capitalisation variant. A morphological cluster.

F11 — disipl. Also: discipline, Self, disciplined, Discipline. A concept cluster with a misspelling as the top token — the gate direction matched closest to the partial token, but the feature genuinely encodes discipline.

F21 — color. Also: color, colour, COLOR, colors. American, British, uppercase, plural. Same concept, all surface forms.

F25 — Article. Score 0.12 — one of the highest. Also: Article, article, articles, Articles. A formatting feature — this fires when the model predicts a section header or reference.

The rest? F7 is Unternehmer (German for entrepreneur) next to 混凝土 (Chinese for concrete) next to ListView (a UI component). Genuine noise — polysemantic compression at its most extreme.

This is what the FFN looks like without a query. 10,240 slots, each one a gate-times-down edge. Some encode clean concepts. Most are polysemantic collisions. The model packed 348,160 edges into the cheapest representation it could find.

DESCRIBE cut through this. It filtered 10,240 features down to the ones that matter for France. That's the difference between browsing the raw graph and querying it.

---

## Act 3: Features — How Does the Model Store It?

A feature is a single column in the FFN — one gate vector that decides when it fires, one down vector that decides what it outputs. Gate times down. That's the edge.

The gate is a direction in the residual stream. When the model's internal state at that layer has high cosine similarity with the gate direction, the feature activates. The down vector then adds its contribution to the output — pushing the next-token prediction toward a specific answer.

We already saw that France's borders relation lives at L25 F5067. Let's look at that feature directly:

```sql
SELECT * FROM FEATURES WHERE layer = 25 AND feature = 5067;
```

France is one of many entities passing through this slot. The feature doesn't belong to France — France is just one node connected by this edge.

We saw the countries cluster near France in Act 1 — Australia [Italy, Germany, Spain] at feature 9348. Now drill into it:

```sql
SELECT * FROM EDGES WHERE layer = 26 AND feature = 9348;
```

This feature outputs Australia, but also fires for Italy, Germany, and Spain. It's not an Australia feature — it's a Western nations feature. The model compressed multiple countries into one slot because they appear in similar contexts. That's polysemanticity.

Now trace that feature index through all 34 layers:

```sql
SELECT * FROM FEATURES WHERE feature = 9348;
```

Different concept at every layer. The index is reused — the knowledge is independent. Each layer has its own gate and down matrices. Feature 9348 at layer 2 and feature 9348 at layer 26 are completely different neurons sharing an index number.

Compare with feature 1484:

```sql
SELECT * FROM FEATURES WHERE feature = 1484;
```

Layer 2: planet — also 惑星, the Japanese word for planet. Same concept, multiple languages. Layer 6: foods. Layer 15: recognize. Layer 22: addition. Completely unrelated concepts sharing the same slot.

Then layer 23 — Arizona. Also: Arizona, Phoenix, Phoenix. And the probe labeled it *capital*. The model knows Phoenix is the capital of Arizona. A real fact, stored in one feature at one layer, surrounded by 33 layers of unrelated concepts.

Layer 33: Score. Nothing to do with Arizona.

That's what polysemantic means in practice. The model reuses 10,240 slots at every layer for different knowledge. LARQL shows you this — one query, all 34 layers, the full life of a feature index through the network.

---

## Act 4: Relations — The Model Learned a Schema

Entities are nodes. Features are edges. Relations are the labels on those edges. The probe discovered them automatically — and LARQL lets you query by relation across the entire graph.

```sql
SHOW RELATIONS;
```

1,489 probe-confirmed relation labels. The top 30 read like a knowledge graph schema: manufacturer (76 features), league (60), founder (52), genre (52), language (46), capital (32), currency (29), spouse (25), continent (25).

Nobody taught the model this schema. The model learned these categories because they're how the world is structured — things have makers, places have capitals, people have occupations. The FFN reinvented a relational schema from raw text.

Browse one relation:

```sql
SELECT * FROM EDGES WHERE relation = "capital" LIMIT 10;
```

32 features across the network store capital-city knowledge. Washington [Washington, Canberra, Brasilia] — a capital-cities cluster. Arizona [Arizona, Phoenix, Phoenix] — the state capital fact from feature 1484. Look for it:

```sql
SELECT * FROM FEATURES WHERE layer = 23 AND token = "Arizona";
```

Arizona, Phoenix, capital. A clean, verifiable fact.

Try another:

```sql
SELECT * FROM EDGES WHERE relation = "founder" LIMIT 10;
```

52 features encoding who founded what.

```sql
SELECT * FROM EDGES WHERE relation = "spouse" LIMIT 10;
```

25 features. The model knows who married whom.

What about Nobel features near Einstein?

```sql
SELECT * FROM EDGES NEAREST TO "Einstein" AT LAYER 26 LIMIT 10;
```

Nob [Academy, Nob, Noble] — the Nobel prize feature, labeled `award`. brain [brains, Brain, Brain]. particle [particles, Particle, particle]. The physics cluster.

Drill into the Nobel feature:

```sql
SELECT * FROM EDGES WHERE layer = 26 AND feature = 4874;
```

Browse, zoom in, trace across layers, filter by relation. Each query reveals a different facet of the same underlying structure.

---

## Act 5: What Entities Does the Model Know?

Browse the named entities the model has stored:

```sql
SELECT * FROM ENTITIES LIMIT 20;
```

Australia (20 features), Google (17), Microsoft (17), John (18), Chinese (19). These are the entities that appear most frequently as feature outputs across all 34 layers.

Filter to a single layer:

```sql
SELECT * FROM ENTITIES WHERE layer = 26 LIMIT 20;
```

The entity list changes at every layer — different knowledge is stored at different depths.

Find a specific token across the graph:

```sql
SELECT * FROM FEATURES WHERE token = "Paris";
```

Every feature in the model that outputs "Paris" — across all layers, with the Also column showing what else each feature produces.

---

## Act 6: The Dimensionality Gap — Why Features Aren't Enough

Everything so far has been browsing the graph. And the graph is messy. Feature 9348 fires for Australia, Italy, Germany, and Spain. The capital relation returns Washington, Canberra, and Brasilia in one cluster. DESCRIBE "France" shows CEO and Fountain alongside real facts. Why?

Because each feature is one-dimensional. One gate vector, one score. A single scalar activation. When the gate fires, it can't distinguish *why* it fired — was the input about France-the-country, France-the-language-origin, or France-the-neighbour-of-Germany? The feature compresses all of those contexts into one number.

That's the polysemanticity problem. It's not a bug — it's a dimensionality constraint. 10,240 features per layer is a lot, but the residual stream is 2,560-dimensional. The model has to project a high-dimensional space down to scalar activations. Every projection loses information. Every feature is a shadow of the full representation.

So how does the model actually answer "The capital of France is" correctly?

Attention. Attention operates in the full-dimensional residual stream — not in the 1D feature space. The query "capital of France" creates a specific pattern across all 2,560 dimensions. Attention heads at each layer match that pattern against the key vectors, selecting *which* features to route signal through and *which* to suppress. The polysemantic noise — CEO, Fountain, Australia — gets low attention weight because the query pattern doesn't align with those directions.

One feature can't tell you the capital of France. 34 layers of attention-weighted features can. The features are the edges. Attention is the routing. You need both.

---

## Act 7: INFER — It's a Graph Walk

Here's the thing most people miss. When LARQL runs INFER, the FFN isn't doing matrix multiplication. The vindex format decomposed the FFN matrices into graph structure. The knowledge half of each layer is a graph walk.

INFER is a graph walk. Literally.

At each layer, the inference engine takes the current residual stream state and does a KNN lookup against the gate vectors — finding which features are nearest neighbours to the current state. The matched features fire. Their down vectors accumulate into the residual stream. Then the walk moves to the next layer.

Attention still uses matrix multiplies — QKV projections and output projections remain as dense matrices. Attention is the routing mechanism that selects which path the walk takes through the graph. The FFN is the graph. Attention is the navigator. Together they produce the forward pass.

```sql
INFER "The capital of France is" TOP 5;
```

Paris. 80%.

That's not a dense matrix multiply producing a result that we then interpret as a graph. The FFN portion is a walk through 34 layers of edges — KNN match on gates, accumulate downs, attention routes between layers — producing Paris at top-1.

DESCRIBE and INFER are the same operation on the same graph. DESCRIBE walks all edges touching an entity. INFER walks the edges that attention routes a specific query through. Same graph, same KNN lookup, different selection criteria.

The FFN was always a graph. The matrix was just an inefficient encoding of it. LARQL removed the encoding and queries the structure directly.

DESCRIBE showed you every edge touching France — messy, polysemantic, overlapping. INFER walked the same graph with attention routing the path — and the accumulated signal across 34 layers of KNN-matched edges produces Paris at 80%.

The graph is messy. The walk is precise. They're the same structure.

---

## Act 8: INSERT — Writing New Knowledge

So far we've been reading the graph. Now we write to it.

Atlantis has no capital in the model's training data. Let's give it one:

```sql
INFER "The capital of Atlantis is" TOP 5;
```

The model guesses — "believed", "said", "a". It has no idea. There's nothing in the graph for Atlantis.

Now insert a fact:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");
```

One statement. The INSERT pipeline captures the model's residual at L26 for the canonical prompt "The capital of Atlantis is", engineers a gate vector from that direction, synthesises a down vector pointing toward "Poseidon", and installs the (gate, up, down) triple into a free feature slot. A balancer then scales the down vector so the fact lands at the right strength — strong enough to be top-1 on the canonical prompt, not so strong it hijacks other capital queries.

Verify it worked:

```sql
INFER "The capital of Atlantis is" TOP 5;
```

Poseidon. 99.98%. The fact is installed.

But did we break anything? The critical question with any knowledge edit:

```sql
INFER "The capital of France is" TOP 5;
```

Paris. 81%. Preserved. The balancer scaled the Atlantis install so it doesn't bleed onto the France query. Zero regression on existing knowledge.

Check the new edge in the graph:

```sql
DESCRIBE "Atlantis";
```

The insert is visible — Poseidon appears in the knowledge band at L26.

---

## Act 9: COMPILE — From Graph to Weights

The INSERT lives in a patch overlay — a runtime edit on top of the base vindex. To make it permanent, compile it:

```sql
COMPILE CURRENT INTO VINDEX "/tmp/atlantis.vindex";
```

This bakes the patch into a standalone vindex — the inserted gate, up, and down vectors are written into the canonical weight files. No overlay, no sidecar, no special loader. The compiled vindex is a normal vindex.

Load it in a fresh session:

```sql
USE "/tmp/atlantis.vindex";
INFER "The capital of Atlantis is" TOP 5;
```

Poseidon. Still there. The fact survived the compile — it's in the bytes now.

```sql
INFER "The capital of France is" TOP 5;
```

Paris. Still preserved. The compiled vindex behaves exactly like the patched session.

That's the full loop: **query the graph -> insert new knowledge -> verify no regression -> compile to permanent weights**. Same query language for reading and writing. The model is a database you can edit.

---

## Summary

| Command | What it does |
|---|---|
| `DESCRIBE "entity"` | The node view — everything the model knows about one entity |
| `SHOW FEATURES <layer>` | Browse the raw FFN — 10,240 slots per layer |
| `SHOW RELATIONS` | The emergent schema — 1,489 relation types the model taught itself |
| `SELECT * FROM EDGES` | Query edges with WHERE, NEAREST TO, score filters |
| `SELECT * FROM FEATURES` | Query features by layer, feature index, or token |
| `SELECT * FROM ENTITIES` | Browse named entities the model has stored |
| `INFER "prompt"` | Graph walk — the precise answer from the messy graph |
| `INSERT INTO EDGES` | Write a new fact into the graph |
| `COMPILE INTO VINDEX` | Bake patches into a standalone vindex |

One engine. One query language. 348,160 edges — and you can add more. The model is a database.
