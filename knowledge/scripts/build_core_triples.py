#!/usr/bin/env python3
"""Build dense, hand-curated Wikidata triples for the top relations.

These are facts every language model knows from Wikipedia. Dense coverage
(50+ pairs per relation) ensures distinctive matching — "Paris, Berlin,
Tokyo, Rome, Madrid" only appears together in the capital relation.

Output: data/wikidata_triples.json (merges with existing)

Usage:
    python3 scripts/build_core_triples.py
"""

import json
from pathlib import Path

CORE_TRIPLES = {
    "capital": {
        "pid": "P36",
        "pairs": [
            ["France", "Paris"], ["Germany", "Berlin"], ["Japan", "Tokyo"],
            ["Italy", "Rome"], ["Spain", "Madrid"], ["United Kingdom", "London"],
            ["United States", "Washington"], ["Russia", "Moscow"],
            ["China", "Beijing"], ["India", "New Delhi"], ["Australia", "Canberra"],
            ["Canada", "Ottawa"], ["Brazil", "Brasília"], ["Mexico", "Mexico City"],
            ["Egypt", "Cairo"], ["South Korea", "Seoul"], ["Turkey", "Ankara"],
            ["Thailand", "Bangkok"], ["Poland", "Warsaw"], ["Sweden", "Stockholm"],
            ["Norway", "Oslo"], ["Austria", "Vienna"], ["Greece", "Athens"],
            ["Portugal", "Lisbon"], ["Argentina", "Buenos Aires"],
            ["Indonesia", "Jakarta"], ["Nigeria", "Abuja"], ["Iran", "Tehran"],
            ["Iraq", "Baghdad"], ["Saudi Arabia", "Riyadh"],
            ["South Africa", "Pretoria"], ["Kenya", "Nairobi"],
            ["Colombia", "Bogotá"], ["Peru", "Lima"], ["Chile", "Santiago"],
            ["Netherlands", "Amsterdam"], ["Belgium", "Brussels"],
            ["Switzerland", "Bern"], ["Denmark", "Copenhagen"],
            ["Finland", "Helsinki"], ["Ireland", "Dublin"],
            ["Czech Republic", "Prague"], ["Hungary", "Budapest"],
            ["Romania", "Bucharest"], ["Ukraine", "Kyiv"],
            ["Philippines", "Manila"], ["Vietnam", "Hanoi"],
            ["Malaysia", "Kuala Lumpur"], ["Singapore", "Singapore"],
            ["New Zealand", "Wellington"],
        ],
    },
    "official language": {
        "pid": "P37",
        "pairs": [
            ["France", "French"], ["Germany", "German"], ["Japan", "Japanese"],
            ["Italy", "Italian"], ["Spain", "Spanish"], ["Portugal", "Portuguese"],
            ["Russia", "Russian"], ["China", "Chinese"], ["Brazil", "Portuguese"],
            ["South Korea", "Korean"], ["Turkey", "Turkish"], ["Poland", "Polish"],
            ["Sweden", "Swedish"], ["Norway", "Norwegian"], ["Greece", "Greek"],
            ["Thailand", "Thai"], ["India", "Hindi"], ["Egypt", "Arabic"],
            ["Iran", "Persian"], ["Israel", "Hebrew"], ["Indonesia", "Indonesian"],
            ["Vietnam", "Vietnamese"], ["Netherlands", "Dutch"],
            ["Denmark", "Danish"], ["Finland", "Finnish"], ["Czech Republic", "Czech"],
            ["Hungary", "Hungarian"], ["Romania", "Romanian"],
            ["Ukraine", "Ukrainian"], ["Ireland", "Irish"],
            ["Malaysia", "Malay"], ["Philippines", "Filipino"],
            ["Saudi Arabia", "Arabic"], ["Iraq", "Arabic"],
            ["United States", "English"], ["United Kingdom", "English"],
            ["Australia", "English"], ["Canada", "English"],
            ["New Zealand", "English"], ["Nigeria", "English"],
            ["South Africa", "English"], ["Kenya", "Swahili"],
            ["Argentina", "Spanish"], ["Colombia", "Spanish"],
            ["Peru", "Spanish"], ["Chile", "Spanish"], ["Mexico", "Spanish"],
            ["Austria", "German"], ["Switzerland", "German"],
            ["Belgium", "Dutch"], ["Singapore", "English"],
        ],
    },
    "continent": {
        "pid": "P30",
        "pairs": [
            ["France", "Europe"], ["Germany", "Europe"], ["Italy", "Europe"],
            ["Spain", "Europe"], ["United Kingdom", "Europe"],
            ["Poland", "Europe"], ["Sweden", "Europe"], ["Norway", "Europe"],
            ["Greece", "Europe"], ["Portugal", "Europe"], ["Austria", "Europe"],
            ["Switzerland", "Europe"], ["Netherlands", "Europe"],
            ["Belgium", "Europe"], ["Denmark", "Europe"], ["Finland", "Europe"],
            ["Ireland", "Europe"], ["Czech Republic", "Europe"],
            ["Hungary", "Europe"], ["Romania", "Europe"],
            ["Japan", "Asia"], ["China", "Asia"], ["India", "Asia"],
            ["South Korea", "Asia"], ["Thailand", "Asia"], ["Vietnam", "Asia"],
            ["Indonesia", "Asia"], ["Malaysia", "Asia"], ["Philippines", "Asia"],
            ["Turkey", "Asia"], ["Iran", "Asia"], ["Iraq", "Asia"],
            ["Saudi Arabia", "Asia"], ["Israel", "Asia"], ["Singapore", "Asia"],
            ["United States", "North America"], ["Canada", "North America"],
            ["Mexico", "North America"],
            ["Brazil", "South America"], ["Argentina", "South America"],
            ["Colombia", "South America"], ["Peru", "South America"],
            ["Chile", "South America"],
            ["Egypt", "Africa"], ["Nigeria", "Africa"], ["South Africa", "Africa"],
            ["Kenya", "Africa"],
            ["Australia", "Oceania"], ["New Zealand", "Oceania"],
            ["Russia", "Europe"],
        ],
    },
    "country": {
        "pid": "P17",
        "pairs": [
            ["Paris", "France"], ["Berlin", "Germany"], ["Tokyo", "Japan"],
            ["Rome", "Italy"], ["Madrid", "Spain"], ["London", "United Kingdom"],
            ["Moscow", "Russia"], ["Beijing", "China"], ["Seoul", "South Korea"],
            ["Bangkok", "Thailand"], ["Warsaw", "Poland"], ["Stockholm", "Sweden"],
            ["Oslo", "Norway"], ["Athens", "Greece"], ["Lisbon", "Portugal"],
            ["Vienna", "Austria"], ["Amsterdam", "Netherlands"],
            ["Brussels", "Belgium"], ["Copenhagen", "Denmark"],
            ["Helsinki", "Finland"], ["Dublin", "Ireland"],
            ["Prague", "Czech Republic"], ["Budapest", "Hungary"],
            ["Bucharest", "Romania"], ["Kyiv", "Ukraine"],
            ["Cairo", "Egypt"], ["Nairobi", "Kenya"],
            ["New Delhi", "India"], ["Jakarta", "Indonesia"],
            ["Manila", "Philippines"], ["Hanoi", "Vietnam"],
            ["Ottawa", "Canada"], ["Canberra", "Australia"],
            ["Wellington", "New Zealand"], ["Lima", "Peru"],
            ["Santiago", "Chile"], ["Bogotá", "Colombia"],
        ],
    },
    "occupation": {
        "pid": "P106",
        "pairs": [
            ["Mozart", "composer"], ["Beethoven", "composer"], ["Bach", "composer"],
            ["Einstein", "physicist"], ["Newton", "physicist"], ["Curie", "physicist"],
            ["Shakespeare", "playwright"], ["Dickens", "novelist"],
            ["Picasso", "painter"], ["Van Gogh", "painter"], ["Rembrandt", "painter"],
            ["Darwin", "naturalist"], ["Aristotle", "philosopher"],
            ["Plato", "philosopher"], ["Socrates", "philosopher"],
            ["Napoleon", "military leader"], ["Caesar", "politician"],
            ["Galileo", "astronomer"], ["Copernicus", "astronomer"],
            ["Tesla", "inventor"], ["Edison", "inventor"],
            ["Hemingway", "writer"], ["Tolkien", "writer"], ["Twain", "writer"],
            ["Churchill", "politician"], ["Lincoln", "politician"],
            ["Gandhi", "activist"], ["Mandela", "activist"],
            ["Bolt", "sprinter"], ["Phelps", "swimmer"],
            ["Federer", "tennis player"], ["Ronaldo", "footballer"],
            ["Messi", "footballer"], ["Jordan", "basketball player"],
            ["Elvis", "singer"], ["Madonna", "singer"],
            ["Spielberg", "film director"], ["Hitchcock", "film director"],
            ["Kubrick", "film director"], ["Disney", "animator"],
        ],
    },
    "place of birth": {
        "pid": "P19",
        "pairs": [
            ["Mozart", "Salzburg"], ["Einstein", "Ulm"], ["Shakespeare", "Stratford"],
            ["Darwin", "Shrewsbury"], ["Newton", "Woolsthorpe"],
            ["Picasso", "Málaga"], ["Napoleon", "Ajaccio"],
            ["Beethoven", "Bonn"], ["Bach", "Eisenach"],
            ["Galileo", "Pisa"], ["Copernicus", "Toruń"],
            ["Tesla", "Smiljan"], ["Edison", "Milan"],
            ["Van Gogh", "Zundert"], ["Rembrandt", "Leiden"],
            ["Hemingway", "Oak Park"], ["Tolkien", "Bloemfontein"],
            ["Churchill", "Blenheim Palace"], ["Gandhi", "Porbandar"],
            ["Mandela", "Mvezo"], ["Elvis", "Tupelo"],
        ],
    },
    "currency": {
        "pid": "P38",
        "pairs": [
            ["United States", "dollar"], ["United Kingdom", "pound"],
            ["Japan", "yen"], ["European Union", "euro"],
            ["China", "yuan"], ["India", "rupee"], ["Russia", "ruble"],
            ["South Korea", "won"], ["Brazil", "real"], ["Mexico", "peso"],
            ["Turkey", "lira"], ["Switzerland", "franc"],
            ["Sweden", "krona"], ["Norway", "krone"], ["Denmark", "krone"],
            ["Poland", "zloty"], ["Czech Republic", "koruna"],
            ["Hungary", "forint"], ["Thailand", "baht"],
            ["Malaysia", "ringgit"], ["Indonesia", "rupiah"],
            ["Philippines", "peso"], ["South Africa", "rand"],
            ["Nigeria", "naira"], ["Egypt", "pound"], ["Israel", "shekel"],
            ["Saudi Arabia", "riyal"], ["Australia", "dollar"],
            ["Canada", "dollar"], ["New Zealand", "dollar"],
            ["Singapore", "dollar"],
        ],
    },
    "shares border with": {
        "pid": "P47",
        "pairs": [
            ["France", "Spain"], ["France", "Germany"], ["France", "Italy"],
            ["France", "Belgium"], ["France", "Switzerland"],
            ["Germany", "Poland"], ["Germany", "Austria"], ["Germany", "Netherlands"],
            ["Germany", "Denmark"], ["Germany", "Czech Republic"],
            ["Spain", "Portugal"], ["Italy", "Austria"], ["Italy", "Switzerland"],
            ["China", "India"], ["China", "Russia"], ["China", "Vietnam"],
            ["United States", "Canada"], ["United States", "Mexico"],
            ["Russia", "Ukraine"], ["Russia", "Finland"], ["Russia", "Norway"],
            ["India", "Pakistan"], ["India", "China"], ["India", "Nepal"],
            ["Brazil", "Argentina"], ["Brazil", "Colombia"], ["Brazil", "Peru"],
        ],
    },
    "genre": {
        "pid": "P136",
        "pairs": [
            ["Mozart", "classical"], ["Beethoven", "classical"], ["Bach", "baroque"],
            ["Beatles", "rock"], ["Led Zeppelin", "rock"], ["Pink Floyd", "rock"],
            ["Elvis", "rock and roll"], ["Chuck Berry", "rock and roll"],
            ["Miles Davis", "jazz"], ["Louis Armstrong", "jazz"],
            ["Bob Marley", "reggae"], ["Eminem", "hip hop"],
            ["Tupac", "hip hop"], ["Beyoncé", "pop"], ["Madonna", "pop"],
            ["Metallica", "heavy metal"], ["Iron Maiden", "heavy metal"],
            ["Chopin", "romantic"], ["Tchaikovsky", "romantic"],
        ],
    },
    "instance of": {
        "pid": "P31",
        "pairs": [
            ["France", "country"], ["Germany", "country"], ["Japan", "country"],
            ["Italy", "country"], ["Spain", "country"], ["China", "country"],
            ["India", "country"], ["Brazil", "country"], ["Russia", "country"],
            ["Paris", "city"], ["Berlin", "city"], ["Tokyo", "city"],
            ["London", "city"], ["Rome", "city"], ["Madrid", "city"],
            ["Mozart", "human"], ["Einstein", "human"], ["Shakespeare", "human"],
            ["Google", "company"], ["Apple", "company"], ["Microsoft", "company"],
            ["Amazon", "company"], ["Facebook", "company"], ["Tesla", "company"],
            ["Toyota", "company"], ["Samsung", "company"], ["BMW", "company"],
            ["Earth", "planet"], ["Mars", "planet"], ["Jupiter", "planet"],
            ["Atlantic", "ocean"], ["Pacific", "ocean"], ["Indian", "ocean"],
            ["Sahara", "desert"], ["Amazon", "river"], ["Nile", "river"],
            ["Everest", "mountain"], ["Alps", "mountain range"],
        ],
    },
    "located in": {
        "pid": "P131",
        "pairs": [
            ["Paris", "France"], ["Lyon", "France"], ["Marseille", "France"],
            ["Berlin", "Germany"], ["Munich", "Germany"], ["Hamburg", "Germany"],
            ["Tokyo", "Japan"], ["Osaka", "Japan"], ["Kyoto", "Japan"],
            ["Rome", "Italy"], ["Milan", "Italy"], ["Naples", "Italy"],
            ["Madrid", "Spain"], ["Barcelona", "Spain"],
            ["London", "England"], ["Manchester", "England"],
            ["Moscow", "Russia"], ["Beijing", "China"], ["Shanghai", "China"],
            ["New York", "United States"], ["Los Angeles", "United States"],
            ["Sydney", "Australia"], ["Melbourne", "Australia"],
            ["Toronto", "Canada"], ["Vancouver", "Canada"],
            ["Mumbai", "India"], ["Delhi", "India"],
            ["São Paulo", "Brazil"], ["Cairo", "Egypt"],
        ],
    },
    "author": {
        "pid": "P50",
        "pairs": [
            ["Hamlet", "Shakespeare"], ["Romeo and Juliet", "Shakespeare"],
            ["The Odyssey", "Homer"], ["The Iliad", "Homer"],
            ["Don Quixote", "Cervantes"], ["War and Peace", "Tolstoy"],
            ["Crime and Punishment", "Dostoevsky"], ["1984", "Orwell"],
            ["Harry Potter", "Rowling"], ["The Lord of the Rings", "Tolkien"],
            ["Pride and Prejudice", "Austen"], ["Great Expectations", "Dickens"],
            ["The Great Gatsby", "Fitzgerald"], ["Moby Dick", "Melville"],
            ["The Divine Comedy", "Dante"], ["Faust", "Goethe"],
        ],
    },
    "performer": {
        "pid": "P175",
        "pairs": [
            ["Yesterday", "Beatles"], ["Bohemian Rhapsody", "Queen"],
            ["Imagine", "John Lennon"], ["Like a Virgin", "Madonna"],
            ["Thriller", "Michael Jackson"], ["Purple Rain", "Prince"],
            ["Smells Like Teen Spirit", "Nirvana"],
            ["Stairway to Heaven", "Led Zeppelin"],
            ["Hotel California", "Eagles"],
        ],
    },
}


def main() -> None:
    """Build and merge hand-curated core triples into the combined file."""
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "wikidata_triples.json"

    # Load existing data
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    # Merge core triples (core takes priority for overlap)
    for rel_name, rel_data in CORE_TRIPLES.items():
        if rel_name not in existing:
            existing[rel_name] = rel_data
        else:
            # Add new pairs, keep existing
            existing_pairs = set(tuple(p) for p in existing[rel_name]["pairs"])
            for pair in rel_data["pairs"]:
                key = tuple(pair)
                if key not in existing_pairs:
                    existing[rel_name]["pairs"].append(pair)
                    existing_pairs.add(key)
            if not existing[rel_name].get("pid"):
                existing[rel_name]["pid"] = rel_data["pid"]

    # Save
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    # Summary
    total = sum(len(v["pairs"]) for v in existing.values())
    print(f"Saved {len(existing)} properties, {total} total pairs to {output_path}")
    print()
    for name, data in sorted(existing.items(), key=lambda x: -len(x[1]["pairs"])):
        n = len(data["pairs"])
        if n >= 10:
            examples = ", ".join(f"{s}→{o}" for s, o in data["pairs"][:3])
            print(f"  {name:<25s} {n:4d} pairs  [{examples}]")


if __name__ == "__main__":
    main()
