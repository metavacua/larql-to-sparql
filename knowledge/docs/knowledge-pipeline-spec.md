# larql-knowledge — Knowledge Pipeline for LARQL

**Version:** 0.1
**Author:** Chris Hay
**Date:** 2026-03-31
**Status:** Draft
**Companion to:** LQL Language Specification v0.1

---

## 1. Purpose

larql-knowledge is the data pipeline that produces reference databases and probe labels for LARQL. It is separate from the LARQL engine — different repo, different release cadence, different contributors.

The LARQL engine reads JSON files. This project produces them.

```
larql-knowledge (this project)        larql (the engine)
  ┌──────────────────────┐              ┌──────────────────────┐
  │ Ingest               │              │                      │
  │   DBpedia            │    JSON      │  extract-index       │
  │   Wikidata           │──────────►   │  label               │
  │   WordNet            │   files      │  describe            │
  │   AST corpora        │              │  walk                │
  │                      │              │                      │
  │ Probe                │              │                      │
  │   MLX inference      │──────────►   │  feature_labels.json │
  │   Template probing   │   labels     │                      │
  └──────────────────────┘              └──────────────────────┘
```

---

## 2. Output Artifacts

The project produces three categories of artifacts:

### 2.1 Reference Triples

Structured (subject, object) pairs grouped by relation type. Model-agnostic — the same triples work for any model.

**Target: 200+ relations, 100K+ pairs across all domains.**

```
data/triples/

  # ══════════════════════════════════════
  # GEOGRAPHY & COUNTRIES
  # ══════════════════════════════════════
  capital.json              # France→Paris, Germany→Berlin (200+ countries)
  language.json             # France→French, Germany→German (200+ countries)
  continent.json            # France→Europe, Japan→Asia (200+ countries)
  borders.json              # France→Spain, France→Germany (500+ border pairs)
  currency.json             # France→euro, Japan→yen (200+ countries)
  government_type.json      # France→republic, UK→monarchy
  head_of_state.json        # France→Macron, UK→Charles
  head_of_government.json   # UK→Starmer, Canada→Carney
  population.json           # China→billion, Monaco→thousands
  area.json                 # Russia→largest, Vatican→smallest
  gdp.json                  # US→largest, Luxembourg→highest per capita
  calling_code.json         # US→1, UK→44, France→33
  driving_side.json         # UK→left, France→right
  flag_colors.json          # France→blue/white/red, Japan→white/red

  # ══════════════════════════════════════
  # CITIES & PLACES
  # ══════════════════════════════════════
  located_in.json           # Paris→France, Tokyo→Japan (5000+ cities)
  city_country.json         # London→UK, Sydney→Australia
  city_state.json           # LA→California, Mumbai→Maharashtra
  landmark.json             # Paris→Eiffel Tower, NYC→Statue of Liberty
  timezone.json             # London→GMT, Tokyo→JST
  river.json                # London→Thames, Paris→Seine, Cairo→Nile
  elevation.json            # Denver→high, Death Valley→low

  # ══════════════════════════════════════
  # PEOPLE (GENERAL)
  # ══════════════════════════════════════
  occupation.json           # Einstein→physicist, Mozart→composer (2000+ people)
  birthplace.json           # Einstein→Ulm, Mozart→Salzburg (2000+ people)
  deathplace.json           # Mozart→Vienna, Einstein→Princeton
  nationality.json          # Einstein→German, Picasso→Spanish (2000+ people)
  birth_year.json           # Einstein→1879, Mozart→1756
  death_year.json           # Mozart→1791, Einstein→1955
  spouse.json               # Obama→Michelle, Einstein→Mileva
  alma_mater.json           # Obama→Harvard, Zuckerberg→Harvard
  religion.json             # Gandhi→Hindu, Bach→Lutheran
  award.json                # Einstein→Nobel, Obama→Nobel

  # ══════════════════════════════════════
  # POLITICS & GOVERNMENT
  # ══════════════════════════════════════
  party.json                # Obama→Democrat, Thatcher→Conservative (1000+ politicians)
  position.json             # Obama→President, Merkel→Chancellor
  country_leader.json       # US→Biden, France→Macron
  political_ideology.json   # Marx→communism, Hayek→liberalism
  cabinet_position.json     # Secretary of State, Chancellor of Exchequer

  # ══════════════════════════════════════
  # MUSIC & MUSICIANS
  # ══════════════════════════════════════
  genre.json                # Beatles→rock, Mozart→classical (2000+ artists/works)
  instrument.json           # Hendrix→guitar, Coltrane→saxophone (1000+ musicians)
  record_label.json         # Beatles→Apple, Drake→OVO
  band_member.json          # Beatles→Lennon, Queen→Mercury
  album_artist.json         # Thriller→Michael Jackson, Abbey Road→Beatles
  song_artist.json          # Bohemian Rhapsody→Queen, Imagine→Lennon
  music_era.json            # Mozart→Classical, Beethoven→Romantic
  producer_artist.json      # Dr. Dre→Eminem, George Martin→Beatles
  composer.json             # Symphony No. 5→Beethoven, Magic Flute→Mozart

  # ══════════════════════════════════════
  # FILM & TELEVISION
  # ══════════════════════════════════════
  director.json             # Jaws→Spielberg, Psycho→Hitchcock (2000+ films)
  starring.json             # Godfather→Pacino, Titanic→DiCaprio (5000+ film→actor)
  film_genre.json           # Godfather→crime, Alien→sci-fi
  film_year.json            # Godfather→1972, Titanic→1997
  film_studio.json          # Avengers→Marvel, Star Wars→Lucasfilm
  film_country.json         # Parasite→South Korea, Amelie→France
  tv_network.json           # Breaking Bad→AMC, Game of Thrones→HBO
  tv_creator.json           # Breaking Bad→Vince Gilligan, The Wire→David Simon
  screenwriter.json         # Pulp Fiction→Tarantino, Chinatown→Robert Towne
  cinematographer.json      # Blade Runner→Jordan Cronenweth
  film_award.json           # Parasite→Oscar, Godfather→Oscar

  # ══════════════════════════════════════
  # BOOKS & LITERATURE
  # ══════════════════════════════════════
  author.json               # Hamlet→Shakespeare, 1984→Orwell (2000+ books)
  literary_genre.json       # 1984→dystopian, LOTR→fantasy
  book_year.json            # 1984→1949, LOTR→1954
  publisher.json            # Harry Potter→Bloomsbury
  book_series.json          # LOTR→Middle-earth, Narnia→Chronicles
  poet.json                 # The Raven→Poe, Iliad→Homer
  playwright.json           # Hamlet→Shakespeare, Waiting for Godot→Beckett
  literary_movement.json    # Kafka→modernism, Dickens→realism
  book_character.json       # Harry Potter→Hogwarts, LOTR→Frodo

  # ══════════════════════════════════════
  # SPORTS
  # ══════════════════════════════════════
  team.json                 # Messi→Barcelona, Jordan→Bulls (5000+ player→team)
  league.json               # Lakers→NBA, Man United→Premier League (1000+ teams)
  sport.json                # Messi→football, Jordan→basketball
  team_city.json            # Lakers→Los Angeles, Yankees→New York
  team_stadium.json         # Man United→Old Trafford, Lakers→Crypto.com Arena
  team_coach.json           # Man City→Guardiola, Patriots→Belichick
  player_position.json      # Messi→forward, Ronaldo→forward
  player_nationality.json   # Messi→Argentine, Ronaldo→Portuguese
  championship.json         # Man City→Premier League, Lakers→NBA
  sports_award.json         # Messi→Ballon d'Or, Jordan→MVP
  olympic_sport.json        # Bolt→sprinting, Phelps→swimming
  team_color.json           # Man United→red, Chelsea→blue
  team_rival.json           # Real Madrid→Barcelona, Yankees→Red Sox
  team_founded.json         # Man United→1878, Lakers→1947

  # ══════════════════════════════════════
  # COMPANIES & BUSINESS
  # ══════════════════════════════════════
  founder.json              # Apple→Jobs, Microsoft→Gates (1000+ companies)
  headquarters.json         # Apple→Cupertino, Google→Mountain View
  ceo.json                  # Apple→Tim Cook, Microsoft→Satya Nadella
  industry.json             # Apple→technology, Toyota→automotive
  parent_company.json       # Instagram→Meta, YouTube→Google
  subsidiary.json           # WhatsApp→Meta, AWS→Amazon
  stock_exchange.json       # Apple→NASDAQ, Toyota→Tokyo Stock Exchange
  ticker.json               # Apple→AAPL, Google→GOOGL, Tesla→TSLA
  brand_product.json        # Apple→iPhone, Google→Search, Tesla→Model 3
  company_country.json      # Apple→US, Samsung→South Korea, Toyota→Japan
  competitor.json           # Apple→Samsung, Google→Microsoft, Coke→Pepsi
  year_founded.json         # Apple→1976, Google→1998, Amazon→1994
  company_revenue.json      # Apple→largest, Walmart→highest revenue
  designer.json             # iPhone→Jony Ive
  developer.json            # Linux→Torvalds, Python→Guido

  # ══════════════════════════════════════
  # SCIENCE & TECHNOLOGY
  # ══════════════════════════════════════
  inventor.json             # telephone→Bell, light bulb→Edison
  discovery.json            # penicillin→Fleming, radium→Curie
  field_of_study.json       # Einstein→physics, Darwin→biology
  chemical_symbol.json      # gold→Au, iron→Fe, oxygen→O
  planet.json               # Mars→fourth, Jupiter→largest
  element_number.json       # hydrogen→1, carbon→6, oxygen→8
  SI_unit.json              # length→meter, mass→kilogram
  programming_language.json # Python→Guido, C→Ritchie, Rust→Mozilla
  operating_system.json     # macOS→Apple, Windows→Microsoft, Linux→Torvalds
  framework.json            # React→Meta, Angular→Google, PyTorch→Meta

  # ══════════════════════════════════════
  # FOOD & DRINK
  # ══════════════════════════════════════
  ingredient.json           # cheese→milk, bread→flour, wine→grapes
  cuisine_origin.json       # pizza→Italy, sushi→Japan, tacos→Mexico
  food_category.json        # cheese→dairy, apple→fruit, rice→grain
  drink_type.json           # wine→alcoholic, coffee→caffeine, juice→non-alcoholic
  dish_country.json         # paella→Spain, ramen→Japan, curry→India
  food_animal.json          # beef→cow, pork→pig, chicken→chicken

  # ══════════════════════════════════════
  # ART & CULTURE
  # ══════════════════════════════════════
  painter.json              # Mona Lisa→Da Vinci, Starry Night→Van Gogh
  art_movement.json         # Picasso→cubism, Monet→impressionism
  art_museum.json           # Mona Lisa→Louvre, Starry Night→MoMA
  architect.json            # Sagrada Familia→Gaudi, Fallingwater→Wright
  sculpture.json            # David→Michelangelo, Thinker→Rodin

  # ══════════════════════════════════════
  # HISTORY & EVENTS
  # ══════════════════════════════════════
  event_year.json           # WW2→1939, Moon landing→1969
  event_country.json        # French Revolution→France, Meiji→Japan
  battle_war.json           # Normandy→WW2, Gettysburg→Civil War
  historical_figure.json    # Cleopatra→Egypt, Caesar→Rome
  dynasty.json              # Tudor→England, Ming→China
  era.json                  # Renaissance→Europe, Edo→Japan

  # ══════════════════════════════════════
  # ANIMALS & NATURE
  # ══════════════════════════════════════
  animal_class.json         # dog→mammal, eagle→bird, shark→fish
  animal_habitat.json       # penguin→Antarctica, camel→desert
  animal_diet.json          # lion→carnivore, cow→herbivore
  animal_sound.json         # dog→bark, cat→meow, lion→roar
  plant_type.json           # oak→tree, rose→flower, wheat→grass
  endangered.json           # panda→endangered, dodo→extinct

  # ══════════════════════════════════════
  # EDUCATION
  # ══════════════════════════════════════
  university_city.json      # Harvard→Cambridge, Oxford→Oxford, MIT→Cambridge
  university_country.json   # Harvard→US, Oxford→UK, Tokyo→Japan
  university_type.json      # MIT→private, UCLA→public
  academic_field.json       # MIT→engineering, Harvard→law, Oxford→humanities

  # ══════════════════════════════════════
  # RELIGION & PHILOSOPHY
  # ══════════════════════════════════════
  religion_founder.json     # Christianity→Jesus, Islam→Muhammad, Buddhism→Buddha
  religion_text.json        # Christianity→Bible, Islam→Quran, Judaism→Torah
  philosopher_era.json      # Plato→ancient, Kant→Enlightenment
  philosophy_school.json    # Plato→idealism, Nietzsche→existentialism

  # ══════════════════════════════════════
  # TRANSPORT & VEHICLES
  # ══════════════════════════════════════
  manufacturer.json         # Model 3→Tesla, Corolla→Toyota, 747→Boeing
  vehicle_type.json         # 747→airplane, Corolla→car, Titanic→ship
  airline_country.json      # Lufthansa→Germany, JAL→Japan, Emirates→UAE
  airport_city.json         # Heathrow→London, JFK→New York, Narita→Tokyo

  # ══════════════════════════════════════
  # LANGUAGE & LINGUISTICS
  # ══════════════════════════════════════
  language_family.json      # French→Romance, Japanese→Japonic
  language_script.json      # Japanese→kanji, Arabic→Arabic script
  language_speakers.json    # English→most spoken, Mandarin→most native
```

**Format per file:**

```json
{
  "relation": "capital",
  "pid": "P36",
  "description": "Capital city of a country",
  "source": "hand-curated + dbpedia",
  "pairs": [
    ["France", "Paris"],
    ["Germany", "Berlin"],
    ["Japan", "Tokyo"]
  ]
}
```

**Assembled output:**

```
data/wikidata_triples.json    # Combined: all relations in one file
```

### 2.2 Linguistic Databases

Structured linguistic relationships. Model-agnostic.

```
data/
  wordnet_relations.json      # Synonyms, hypernyms, antonyms, meronyms, derivations
  english_grammar.json        # Determiner→noun, preposition→object, copula→adjective
  ast/
    python_ast.json           # def→identifier, import→module, return→expression
    rust_ast.json             # fn→identifier, let→identifier, use→module
    javascript_ast.json       # function→identifier, const→identifier, require→module
    typescript_ast.json       # interface→identifier, type→identifier, enum→identifier
    java_ast.json             # class→identifier, import→package, void→method
    go_ast.json               # func→identifier, import→package, var→identifier
    c_ast.json                # int→identifier, #include→header, struct→identifier
    sql_ast.json              # SELECT→column, FROM→table, WHERE→condition
```

### 2.3 Probe Labels

Per-feature labels confirmed by running entities through actual model inference. Model-specific — each model gets its own probe results.

```
probes/
  gemma-3-4b-it/
    feature_labels.json       # Per-feature confirmed labels
    probe_meta.json           # Metadata: when, how many, templates used
  llama-3-8b/
    feature_labels.json
    probe_meta.json
  mistral-7b/
    feature_labels.json
    probe_meta.json
  ... (one directory per model)
```

**feature_labels.json format:**

```json
[
  {
    "layer": 27,
    "feature": 9515,
    "relation": "capital",
    "source": "probe",
    "confidence": 0.97,
    "examples": [
      {"entity": "France", "target": "Paris", "gate_score": 1436.9},
      {"entity": "Germany", "target": "Berlin", "gate_score": 1289.3},
      {"entity": "Japan", "target": "Tokyo", "gate_score": 1156.7}
    ]
  },
  {
    "layer": 24,
    "feature": 4532,
    "relation": "language",
    "source": "probe",
    "confidence": 0.95,
    "examples": [
      {"entity": "France", "target": "French", "gate_score": 26.1},
      {"entity": "Germany", "target": "German", "gate_score": 24.8}
    ]
  }
]
```

**probe_meta.json format:**

```json
{
  "model": "google/gemma-3-4b-it",
  "date": "2026-03-31",
  "num_entities": 15983,
  "num_templates": 13,
  "num_probes": 16000,
  "num_features_labeled": 112,
  "probe_time_seconds": 1020,
  "top_k_per_layer": 50,
  "min_gate_score": 5.0,
  "templates": {
    "capital": "The capital of {X} is",
    "language": "The official language of {X} is",
    "continent": "{X} is located in",
    "borders": "{X} shares a border with",
    "occupation": "{X} was a",
    "birthplace": "{X} was born in",
    "currency": "The currency of {X} is",
    "located_in": "{X} is located in",
    "author": "The author of {X} is",
    "director": "{X} was directed by",
    "genre": "The genre of {X} is",
    "founder": "{X} was founded by",
    "nationality": "{X} is from"
  }
}
```

---

## 3. Data Sources

### 3.1 Wikidata / DBpedia (Factual Relations, L14-27)

**Purpose:** Provide ground truth (subject, object) pairs for factual knowledge the model learned from Wikipedia.

**Source hierarchy:**

| Tier | Source | Pairs | Quality | Method |
|------|--------|-------|---------|--------|
| 1 | Hand-curated | ~500 | Gold | Manual JSON files per relation |
| 2 | DBpedia | ~16K | High | SPARQL queries + API, filtered to single/few-token entities |
| 3 | Wikidata dump | ~500K+ | Medium | Full dump filtered to top properties, common entities |

**Ingestion pipeline:**

```bash
# Tier 1: Hand-curate core relations
# Edit data/triples/*.json directly

# Tier 2: Pull from DBpedia
python3 scripts/ingest_dbpedia.py \
  --properties capital,language,continent,borders,occupation,... \
  --max-per-relation 500 \
  --output data/triples/

# Tier 3: Pull from Wikidata dump (future)
python3 scripts/ingest_wikidata_dump.py \
  --dump wikidata-latest-truthy.nt.gz \
  --properties P36,P37,P30,P47,P106,... \
  --max-per-relation 5000 \
  --output data/triples/

# Assemble into combined file
python3 scripts/assemble_triples.py
```

**Entity filtering rules:**
- Prefer single-token entities ("France" over "Republic of France")
- Include common multi-token entities ("United States", "New York", "Ice cream")
- Exclude entities with IDs or codes ("Q12345", "ISO 3166-1")
- Exclude fictional entities from factual relations (no "Gohan→Earth")
- Lowercase normalize for matching, preserve original case for display

**Relation selection criteria:**
- Must appear in >10,000 Wikidata items (high frequency)
- Must involve entity types the model commonly encounters (countries, people, works, companies, cities)
- Must produce single/few-token objects the model can output
- Exclude media properties (image, audio, video links)
- Exclude identifier properties (ISNI, VIAF, GND)

### 3.2 WordNet (Semantic Relations, L0-13)

**Purpose:** Provide ground truth (word, related_word) pairs for semantic relationships the model learned from language.

**Relations extracted:**

| Relation | Description | Expected pairs | Example |
|----------|-------------|----------------|---------|
| synonym | Same meaning | 5,000 | big→large |
| hypernym | Is-a (parent category) | 3,000 | dog→animal |
| antonym | Opposite meaning | 2,000 | hot→cold |
| meronym | Part-of | 2,000 | wheel→car |
| derivation | Derived form | 5,000 | able→ability |

**Ingestion pipeline:**

```bash
python3 scripts/fetch_wordnet_relations.py
# Requires: pip install nltk
# Downloads WordNet data on first run
# Output: data/wordnet_relations.json
```

**Quality rules:**
- Only include pairs where both words are common English (frequency > 1000 in Brown corpus)
- Exclude technical/archaic terms
- Exclude multi-word expressions for now
- Validate with lemminflect for morphological pairs

### 3.3 Morphological Lexicon (Form Relations, L0-13)

**Purpose:** Provide ground truth (base_form, inflected_form) pairs for morphological patterns.

**Relations extracted:**

| Relation | Description | Example |
|----------|-------------|---------|
| plural | Singular→plural | dog→dogs |
| gerund | Base→-ing form | run→running |
| past_tense | Base→past | run→ran |
| third_person | Base→3rd person | run→runs |
| comparative | Base→-er form | big→bigger |
| superlative | Base→-est form | big→biggest |
| agent_noun | Verb→-er noun | run→runner |
| nominalization | Adj→-ness noun | happy→happiness |
| adverb | Adj→-ly adverb | happy→happily |
| negation_prefix | Base→un- form | happy→unhappy |

**Ingestion pipeline:**

```bash
python3 scripts/fetch_morphological.py
# Requires: pip install lemminflect
# Handles irregular forms correctly (ran, not runned)
# Output: integrated into data/wordnet_relations.json
```

**Quality rules:**
- Use lemminflect for all inflections (handles irregulars)
- Validate every generated form exists in a word frequency list
- Exclude forms that don't appear in common English text
- Focus on the 500 most common verbs, 500 most common adjectives, 500 most common nouns

### 3.4 English Grammar (Syntactic Relations, L0-13)

**Purpose:** Provide ground truth (function_word, following_word_type) pairs for syntactic patterns.

**Relations extracted:**

| Relation | Description | Example |
|----------|-------------|---------|
| determiner→noun | Article predicts noun | the→dog, a→cat |
| preposition→noun | Prep predicts noun | in→London, of→France |
| copula→adjective | Be-verb predicts adj | is→big, was→born |
| auxiliary→verb | Aux predicts verb | will→go, can→see |
| conjunction→clause | Conj predicts clause start | and→the, but→it |
| pronoun→verb | Pronoun predicts verb | he→said, they→went |

**Ingestion pipeline:**

```bash
python3 scripts/extract_grammar_pairs.py \
  --corpus data/corpora/english_sample.txt \
  --output data/english_grammar.json
```

**Method:**
- Parse a large English corpus (Wikipedia text dump, ~1M sentences)
- Extract bigram co-occurrences at syntactic boundaries
- Filter to function_word→content_word pairs
- Group by syntactic relation type
- Top 200 pairs per relation

### 3.5 AST Pairs (Code Structure, L0-13)

**Purpose:** Provide ground truth (keyword, following_token) pairs for code syntax patterns the model learned from code corpora.

**Languages supported:**

| Language | Parser | Key relations |
|----------|--------|---------------|
| **Systems** | | |
| Python | `ast` module | def→identifier, class→identifier, import→module, return→expression, for→identifier, if→condition, with→expression, yield→expression, async→def, lambda→expression, try→block, except→exception, raise→exception |
| Rust | tree-sitter-rust | fn→identifier, let→identifier, use→module, impl→type, struct→identifier, enum→identifier, match→expression, trait→identifier, pub→fn, mod→identifier, unsafe→block, async→fn, move→closure, where→constraint |
| C | tree-sitter-c | int→identifier, #include→header, struct→identifier, void→function, typedef→type, malloc→size, printf→format, #define→macro, switch→variable, goto→label, sizeof→type, static→type, extern→type |
| C++ | tree-sitter-cpp | class→identifier, template→type, namespace→identifier, virtual→method, override→method, auto→variable, std→container, new→type, delete→pointer, const→type, friend→class, operator→symbol |
| **Web** | | |
| JavaScript | tree-sitter-javascript | function→identifier, const→identifier, let→identifier, require→module, class→identifier, import→module, export→declaration, async→function, await→promise, yield→value, new→constructor, this→property, throw→error |
| TypeScript | tree-sitter-typescript | interface→identifier, type→identifier, enum→identifier, extends→type, implements→type, readonly→property, generic→type, as→type, keyof→type, typeof→expression, declare→type, abstract→class, namespace→identifier |
| HTML | regex/tree-sitter | div→class, span→class, a→href, img→src, input→type, form→action, table→class, script→src, link→href, meta→content, button→onclick, select→name, style→type, head→meta, body→div |
| CSS/SCSS | regex patterns | color→value, font→value, display→value, margin→value, padding→value, background→value, border→value, position→value, width→value, height→value, flex→value, grid→value, @media→query, @import→url, :hover→property |
| **JVM** | | |
| Java | tree-sitter-java | class→identifier, import→package, void→method, public→class, interface→identifier, extends→class, implements→interface, new→constructor, throws→exception, synchronized→block, static→method, final→variable, abstract→method, enum→identifier, package→name |
| Kotlin | tree-sitter-kotlin | fun→identifier, val→identifier, var→identifier, class→identifier, object→identifier, data→class, sealed→class, when→expression, suspend→fun, companion→object, inline→fun, lateinit→var |
| Scala | tree-sitter-scala | def→identifier, val→identifier, var→identifier, class→identifier, object→identifier, trait→identifier, case→class, sealed→trait, implicit→value, lazy→val, match→expression |
| **Scripting** | | |
| Ruby | tree-sitter-ruby | def→identifier, class→identifier, module→identifier, require→string, attr→symbol, do→block, end→statement, yield→value, begin→block, rescue→exception, include→module |
| PHP | tree-sitter-php | function→identifier, class→identifier, namespace→identifier, use→class, echo→expression, require→path, public→function, private→function, try→block, throw→exception, interface→identifier |
| Perl | regex patterns | sub→identifier, my→variable, use→module, foreach→variable, unless→condition, die→message, bless→reference, package→name |
| Lua | tree-sitter-lua | function→identifier, local→identifier, require→module, for→variable, while→condition, return→value, table→constructor, nil→value |
| **Functional** | | |
| Haskell | tree-sitter-haskell | data→type, class→typeclass, instance→typeclass, where→definition, let→binding, import→module, type→alias, newtype→wrapper, do→monad, case→expression, deriving→typeclass |
| OCaml | tree-sitter-ocaml | let→identifier, type→identifier, module→identifier, match→expression, fun→parameter, val→identifier, open→module, sig→signature |
| Elixir | tree-sitter-elixir | def→identifier, defmodule→identifier, defp→identifier, do→block, end→statement, use→module, import→module, alias→module, case→expression, with→pattern |
| Clojure | regex patterns | defn→identifier, def→identifier, ns→namespace, require→module, let→binding, fn→parameter, if→condition, cond→expression |
| **Data/Query** | | |
| SQL | regex patterns | SELECT→column, FROM→table, WHERE→condition, JOIN→table, INSERT→table, CREATE→table, UPDATE→table, DELETE→table, ALTER→table, DROP→table, GROUP→BY, ORDER→BY, HAVING→condition, INDEX→column, GRANT→privilege |
| R | tree-sitter-r | function→identifier, library→package, data→frame, plot→variable, for→variable, if→condition, return→value, source→file |
| MATLAB | regex patterns | function→identifier, for→variable, while→condition, switch→variable, class→identifier, end→statement |
| **Shell/Config** | | |
| Bash | tree-sitter-bash | function→identifier, if→condition, for→variable, while→condition, case→variable, export→variable, source→file, alias→name, echo→string, cd→path, chmod→permissions |
| PowerShell | regex patterns | function→identifier, param→parameter, foreach→variable, if→condition, Write-Host→string, Get→object, Set→object, New→object |
| YAML | regex patterns | key→value, list→item, map→key, include→file, env→variable |
| JSON | regex patterns | key→value, array→element, object→key, string→value, number→value |
| TOML | regex patterns | key→value, section→name, array→element |
| **Markup** | | |
| LaTeX | regex patterns | \begin→environment, \section→title, \usepackage→package, \cite→reference, \ref→label, \label→name, \textbf→text, \emph→text, \frac→numerator |
| Markdown | regex patterns | #→heading, *→emphasis, [→link_text, ```→language, -→list_item, >→blockquote, |→table_cell |
| XML | regex patterns | tag→attribute, xmlns→namespace, xsl→template, schema→element |
| **Mobile** | | |
| Swift | tree-sitter-swift | func→identifier, class→identifier, struct→identifier, enum→identifier, let→identifier, var→identifier, import→module, protocol→identifier, extension→type, guard→condition, @→attribute |
| Dart | tree-sitter-dart | class→identifier, void→method, import→package, final→variable, const→variable, async→function, await→future, extends→class, implements→interface, Widget→build |

**Ingestion pipeline:**

```bash
# Parse code corpora and extract AST boundary pairs
python3 scripts/extract_ast_pairs.py \
  --language python \
  --corpus data/corpora/python_files/ \
  --max-pairs 500 \
  --output data/ast/python_ast.json

# Or parse all supported languages at once
python3 scripts/extract_all_ast_pairs.py \
  --corpus-dir data/corpora/ \
  --output-dir data/ast/
```

**AST pair format:**

```json
{
  "language": "python",
  "relations": {
    "py:function_def": {
      "description": "Function definition: def keyword followed by function name",
      "keyword": "def",
      "pairs": [
        ["def", "__init__"], ["def", "forward"], ["def", "main"],
        ["def", "train"], ["def", "test"], ["def", "setup"],
        ["def", "get"], ["def", "set"], ["def", "update"],
        ["def", "process"], ["def", "run"], ["def", "load"]
      ]
    },
    "py:class_def": {
      "description": "Class definition: class keyword followed by class name",
      "keyword": "class",
      "pairs": [
        ["class", "Model"], ["class", "Dataset"], ["class", "Config"],
        ["class", "Module"], ["class", "Layer"], ["class", "Block"],
        ["class", "Trainer"], ["class", "Optimizer"], ["class", "Scheduler"]
      ]
    },
    "py:import": {
      "description": "Import statement: import keyword followed by module name",
      "keyword": "import",
      "pairs": [
        ["import", "torch"], ["import", "numpy"], ["import", "os"],
        ["import", "json"], ["import", "sys"], ["import", "typing"],
        ["import", "pathlib"], ["import", "collections"], ["import", "math"]
      ]
    }
  }
}
```

**Corpus sources:**
- Python: top 100 PyPI packages source code
- Rust: top 100 crates.io packages
- JavaScript: top 100 npm packages
- TypeScript: top TypeScript repos on GitHub
- Java: top Maven packages
- Go: top Go modules
- C/C++: Linux kernel headers, popular C libraries
- SQL: StackOverflow SQL examples, database documentation
- HTML/CSS: top websites source, MDN examples

**Quality rules:**
- Only pairs that appear 5+ times in the corpus (not one-off variable names)
- Exclude generated code (node_modules, build artifacts)
- Exclude comments and strings
- Focus on keyword→first_meaningful_token at AST boundaries
- Normalize: lowercase identifiers, strip decorators/modifiers

---

## 4. Probe Pipeline

### 4.1 Overview

The probe runs actual model inference to confirm which features encode which facts. It is the highest-confidence labelling method — ground truth from the model itself.

```
Triples + Templates → Model Inference → Feature Activations → Match → Labels
```

### 4.2 Templates

Each relation has one or more prompt templates. Multiple variants per relation improve probe coverage — different phrasings activate different features.

**Target: 200+ relations x 2-3 templates each = 500+ templates.**

```json
{
  "// === GEOGRAPHY & COUNTRIES ===": "",

  "capital": [
    "The capital of {X} is",
    "The capital city of {X} is",
    "{X}'s capital is"
  ],
  "language": [
    "The official language of {X} is",
    "The language spoken in {X} is",
    "People in {X} speak"
  ],
  "continent": [
    "{X} is located in",
    "{X} is a country in",
    "The continent of {X} is"
  ],
  "borders": [
    "{X} shares a border with",
    "{X} is bordered by",
    "A country next to {X} is"
  ],
  "currency": [
    "The currency of {X} is",
    "{X} uses the",
    "The money used in {X} is"
  ],
  "government_type": [
    "{X} is a",
    "The government of {X} is a",
    "The political system of {X} is"
  ],
  "head_of_state": [
    "The president of {X} is",
    "The head of state of {X} is",
    "The leader of {X} is"
  ],
  "head_of_government": [
    "The prime minister of {X} is",
    "The head of government of {X} is"
  ],
  "flag_colors": [
    "The flag of {X} is",
    "The colors of {X}'s flag are"
  ],
  "driving_side": [
    "In {X}, people drive on the",
    "{X} drives on the"
  ]
}
```

(Full template set continues for all 200+ relations as specified above.)

### 4.3 Probe Execution

```bash
python3 scripts/probe_mlx.py \
  --model google/gemma-3-4b-it \
  --vindex output/gemma3-4b-full.vindex \
  --triples data/wikidata_triples.json \
  --templates data/probe_templates.json \
  --output probes/gemma-3-4b-it/ \
  --top-k 50 \
  --min-gate-score 5.0 \
  --max-entities-per-relation 500
```

**Algorithm per probe:**

```
1. Format prompt: template.replace("{X}", entity)
2. Run forward pass through model (MLX/PyTorch)
3. Capture residual at each knowledge layer (L14-27)
4. For each layer:
   a. Compute gate scores: gates[layer] @ residual
   b. Take top-K features by |gate_score|
   c. For each top feature:
      - Look up its output token from down_meta
      - Check if (entity, output_token) matches any Wikidata triple
      - If match: record (layer, feature, relation, entity, target, gate_score)
5. Filter: only keep features that match for 2+ entities (not one-off activations)
```

### 4.4 Incremental Probing

```bash
# First run: probe all entities
python3 scripts/probe_mlx.py --output probes/gemma-3-4b-it/

# Add new triples, probe only new entities
python3 scripts/probe_mlx.py \
  --output probes/gemma-3-4b-it/ \
  --incremental \
  --add-triples data/triples/sports_teams.json

# Add new templates, re-probe affected entities
python3 scripts/probe_mlx.py \
  --output probes/gemma-3-4b-it/ \
  --incremental \
  --add-templates data/new_templates.json
```

The probe stores which (entity, template) pairs have been run. Incremental mode only runs new combinations.

### 4.5 Multi-Model Support

```bash
# Probe Gemma
python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --output probes/gemma-3-4b-it/

# Probe Llama (same triples, same templates, different model)
python3 scripts/probe_mlx.py --model meta-llama/Llama-3-8B --output probes/llama-3-8b/

# Probe Mistral
python3 scripts/probe_mlx.py --model mistralai/Mistral-7B --output probes/mistral-7b/

# Compare: which relations does each model encode?
python3 scripts/compare_probes.py probes/gemma-3-4b-it/ probes/llama-3-8b/ probes/mistral-7b/
```

---

## 5. Label Merging

### 5.1 Priority Order

Labels come from multiple sources. Higher priority overrides lower:

```
1. Probe-confirmed (highest)    — model inference confirmed this feature encodes this relation
2. Wikidata output matching     — cluster outputs match Wikidata objects
3. WordNet output matching      — cluster outputs match WordNet pairs (L0-13 only)
4. AST output matching          — cluster outputs match AST pairs (L0-13 only)
5. Entity pattern detection     — cluster members match known entity lists (country, language, month, number)
6. Morphological detection      — cluster members are short suffixes/prefixes
7. TF-IDF top tokens (lowest)   — fallback: most distinctive tokens in the cluster
```

### 5.2 Layer-Aware Matching

| Layer Range | Source Databases | Label Types |
|-------------|-----------------|-------------|
| L0-7 | Morphological lexicon, WordNet derivations | plural, gerund, past_tense, derivation |
| L4-13 | WordNet (synonym, hypernym, antonym, meronym), English grammar, AST pairs | synonym, determiner→noun, py:function_def |
| L14-27 | Wikidata triples, probe labels | capital, language, continent, occupation |
| L28-33 | None (output formatting) | TF-IDF fallback only |

### 5.3 Merge Command

```bash
# Merge all sources into the vindex
larql label gemma3-4b.vindex \
  --triples data/wikidata_triples.json \
  --wordnet data/wordnet_relations.json \
  --ast data/ast/ \
  --probes probes/gemma-3-4b-it/feature_labels.json
```

**Or from the LARQL engine:**

```bash
larql label <vindex_path> \
  --knowledge-dir <path_to_larql_knowledge_repo>
```

### 5.4 Output

The merge produces `feature_labels.json` in the vindex directory:

```json
[
  {"l": 27, "f": 9515, "rel": "capital", "src": "probe", "conf": 0.97},
  {"l": 24, "f": 4532, "rel": "language", "src": "probe", "conf": 0.95},
  {"l": 25, "f": 4207, "rel": "continent", "src": "probe", "conf": 0.92},
  {"l": 18, "f": 3629, "rel": "borders", "src": "probe", "conf": 0.89},
  {"l": 26, "f": 9348, "rel": "country", "src": "cluster", "conf": 0.61},
  {"l":  3, "f": 1204, "rel": "plural", "src": "wordnet", "conf": 0.85},
  {"l":  8, "f": 5621, "rel": "synonym", "src": "wordnet", "conf": 0.78},
  {"l": 10, "f": 2305, "rel": "py:function_def", "src": "ast", "conf": 0.82}
]
```

---

## 6. Directory Structure

```
larql-knowledge/
  README.md
  LICENSE

  # Reference databases (model-agnostic)
  data/
    triples/                          # Wikidata relation pairs
      capital.json
      language.json
      continent.json
      borders.json
      occupation.json
      genre.json
      author.json
      director.json
      birthplace.json
      currency.json
      located_in.json
      founder.json
      nationality.json
      spouse.json
      instrument.json
      league.json
      team.json
      starring.json
      producer.json
      record_label.json
      designer.json
      developer.json
      manufacturer.json
      subsidiary.json
      parent_company.json
      religion.json
      party.json
      alma_mater.json
      composer.json
      deathplace.json

    wikidata_triples.json             # Combined: all triples in one file
    wordnet_relations.json            # WordNet pairs
    english_grammar.json              # Syntactic pairs

    ast/                              # AST pairs per language
      python_ast.json
      rust_ast.json
      javascript_ast.json
      typescript_ast.json
      java_ast.json
      go_ast.json
      c_ast.json
      sql_ast.json
      html_css_ast.json

    probe_templates.json              # Prompt templates per relation

    corpora/                          # Raw text/code for extraction (gitignored)
      english_sample.txt
      python_files/
      rust_files/
      javascript_files/

  # Model-specific probe results
  probes/
    gemma-3-4b-it/
      feature_labels.json
      probe_meta.json
    llama-3-8b/
      feature_labels.json
      probe_meta.json

  # Ingestion and probe scripts
  scripts/
    # Data ingestion
    ingest_dbpedia.py                 # Pull from DBpedia SPARQL endpoint
    ingest_wikidata_dump.py           # Parse Wikidata dump file
    fetch_wordnet_relations.py        # Extract WordNet relations via NLTK
    fetch_morphological.py            # Generate morphological pairs via lemminflect
    extract_grammar_pairs.py          # Extract syntactic pairs from English corpus
    extract_ast_pairs.py              # Extract AST pairs from code corpus
    extract_all_ast_pairs.py          # Extract all language ASTs at once
    assemble_triples.py               # Combine all triples into one file
    build_core_triples.py             # Seed core hand-curated triples

    # Probing
    probe_mlx.py                      # Run MLX inference probes
    probe_pytorch.py                  # Run PyTorch inference probes (future)
    build_feature_labels.py           # Gate KNN probes (no model needed)

    # Analysis
    compare_probes.py                 # Compare probe results across models
    coverage_report.py                # Report which relations/entities are covered
    quality_check.py                  # Validate triples quality

    # Utilities
    filter_entities.py                # Filter entities to single/few-token forms
    normalize_triples.py              # Case normalize, deduplicate

  # Tests
  tests/
    test_triples_format.py            # Validate all triples JSON files
    test_wordnet_quality.py           # Check WordNet pairs quality
    test_ast_coverage.py              # Check AST pairs coverage
    test_probe_output.py              # Validate probe output format

  # CI/CD
  .github/
    workflows/
      validate_data.yml               # Check triples format on PR
      run_probes.yml                  # Run probes on new models (GPU runner)
```

---

## 7. Contributing

### 7.1 Adding Triples

The easiest way to contribute. Create a JSON file in `data/triples/`:

```json
{
  "relation": "habitat",
  "pid": "P2974",
  "description": "Natural habitat of an animal or plant species",
  "source": "hand-curated",
  "pairs": [
    ["polar bear", "Arctic"],
    ["penguin", "Antarctica"],
    ["kangaroo", "Australia"],
    ["panda", "China"],
    ["elephant", "Africa"]
  ]
}
```

Run `python3 scripts/assemble_triples.py` to rebuild the combined file.

### 7.2 Adding AST Languages

1. Create a parser script for the language
2. Parse a corpus of 100+ files
3. Extract keyword→following_token pairs at AST boundaries
4. Filter to pairs appearing 5+ times
5. Save to `data/ast/<language>_ast.json`

### 7.3 Adding Templates

Add to `data/probe_templates.json`:

```json
{
  "habitat": [
    "The natural habitat of a {X} is",
    "{X} are found in",
    "The {X} lives in"
  ]
}
```

### 7.4 Running Probes for a New Model

```bash
# 1. Build the vindex
larql extract-index <model_id> -o <vindex_path>

# 2. Run the probe
python3 scripts/probe_mlx.py \
  --model <model_id> \
  --vindex <vindex_path> \
  --output probes/<model_name>/

# 3. Merge labels into the vindex
larql label <vindex_path> --probes probes/<model_name>/
```

---

## 8. Scaling Roadmap

| Phase | Triples | Relations | AST Languages | Probe Coverage | WordNet | Timeline |
|-------|---------|-----------|---------------|----------------|---------|----------|
| 1 (now) | 16K | 32 | 0 | 112 features | 18K pairs | Done |
| 2 | 100K | 100+ | 5 (Py/Rust/JS/TS/Java) | 1,000+ features | 25K pairs | 1 week |
| 3 | 500K | 150+ | 15 languages | 5,000+ features | 30K pairs + grammar | 1 month |
| 4 | 2M+ | 200+ | 30+ languages | 20,000+ features | Full WordNet + FrameNet | 3 months |

**Phase 2 -- Demo Ready:**
- Expand DBpedia to 1000+ pairs per relation for top 30 relations
- Add 70 more relations: sports (team_city, player_position, championship, team_rival, team_coach), entertainment (film_studio, tv_network, song_artist, album_artist, music_era), business (ticker, industry, headquarters, brand_product, competitor), science (chemical_symbol, planet, programming_language, operating_system), food (ingredient, cuisine_origin, food_category), history (event_year, dynasty, historical_figure), animals (habitat, diet, classification), education (university_city, academic_field)
- AST pairs for Python, Rust, JavaScript, TypeScript, Java (500+ pairs each)
- Full MLX probe run on Gemma 3 4B: all 16K entities x 32 templates x full model inference
- English grammar from parsed Wikipedia (10K+ syntactic pairs)
- Target: DESCRIBE any common entity -> 3+ correctly labelled edges

**Phase 3 -- Broad Coverage:**
- Ingest full Wikidata dump filtered to top 500 properties and entities with Wikipedia articles
- Add 15 more AST languages: Go, C, C++, Ruby, PHP, Kotlin, Swift, Scala, Haskell, Elixir, Bash, SQL, R, Lua, Dart
- Add FrameNet for richer syntactic frame pairs
- Run probes on Llama 3 8B, Mistral 7B, DeepSeek, Qwen -- cross-model comparison
- Target: DESCRIBE any Wikipedia entity -> 5+ correctly labelled edges
- Publish pre-labelled vindexes for top 5 models on HuggingFace

**Phase 4 -- Community Scale:**
- Open contribution pipeline: PR a JSON file, CI validates format, automated quality checks
- Community-contributed domain-specific triple sets: medical (ICD codes, drug interactions), legal (case citations, statutes), financial (company filings, market data)
- Automated probe runner: new model on HuggingFace -> CI triggers probe -> publishes labelled vindex
- Cross-lingual triples: French, German, Spanish, Chinese, Japanese, Korean Wikipedia infoboxes
- Multi-modal: image caption pairs for vision-language models
- Target: Any model, any entity, any language -> rich labelled knowledge profile
- The knowledge database becomes a shared resource with 10K+ GitHub stars

---

## 9. Integration with LARQL

The LARQL engine consumes artifacts from this project:

```bash
# At vindex build time -- cluster-based labels
larql extract-index <model> -o <vindex> \
  --triples data/wikidata_triples.json \
  --wordnet data/wordnet_relations.json

# After probe -- merge probe labels into vindex
larql label <vindex> \
  --probes probes/<model>/feature_labels.json

# At query time -- DESCRIBE uses merged labels
larql> DESCRIBE "France";
France
  capital        -> Paris           (probe, 0.97)
  language       -> French          (probe, 0.95)
  continent      -> Europe          (probe, 0.92)
  borders        -> Spain           (probe, 0.89)
  country        -> Australia       (cluster, 0.61)
```

The engine does not import or depend on any ingestion code. It reads JSON files. This project produces those files.
