//! # Element expert
//!
//! Periodic-table lookups. Name matching uses lowercase IUPAC identifiers
//! (e.g. `"gold"`, `"sulfur"`, `"aluminium"`); symbol matching is
//! case-insensitive against the chemical symbol (`"Au"`, `"au"`).
//!
//! ## Ops
//!
//! - `by_number {z: 1..=118} → {z, symbol, name, mass} | null`
//! - `by_symbol {symbol: string} → {z, symbol, name, mass} | null`
//! - `by_name {name: string} → {z, symbol, name, mass} | null`
//! - `list {} → [{z, symbol, name, mass}]`

use expert_interface::{arg_str, arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "element",
    tier = 1,
    description = "Periodic table: lookup by IUPAC name, symbol, or atomic number",
    version = "0.2.0",
    ops = ["by_number", "by_symbol", "by_name", "list"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "by_number" => by_number(arg_u64(args, "z")? as u8).map(to_value),
        "by_symbol" => by_symbol(arg_str(args, "symbol")?).map(to_value),
        "by_name" => by_name(arg_str(args, "name")?).map(to_value),
        "list" => Some(json!(ELEMENTS.iter().map(to_value).collect::<Vec<_>>())),
        _ => None,
    }
}

fn to_value(e: &Element) -> Value {
    json!({
        "z": e.z,
        "symbol": e.symbol,
        "name": e.name,
        "mass": e.mass_m1000 as f64 / 1000.0,
    })
}

struct Element {
    z: u8,
    symbol: &'static str,
    name: &'static str,
    mass_m1000: u32,
}

const ELEMENTS: &[Element] = &[
    Element { z: 1,   symbol: "H",  name: "hydrogen",      mass_m1000: 1008 },
    Element { z: 2,   symbol: "He", name: "helium",        mass_m1000: 4003 },
    Element { z: 3,   symbol: "Li", name: "lithium",       mass_m1000: 6941 },
    Element { z: 4,   symbol: "Be", name: "beryllium",     mass_m1000: 9012 },
    Element { z: 5,   symbol: "B",  name: "boron",         mass_m1000: 10811 },
    Element { z: 6,   symbol: "C",  name: "carbon",        mass_m1000: 12011 },
    Element { z: 7,   symbol: "N",  name: "nitrogen",      mass_m1000: 14007 },
    Element { z: 8,   symbol: "O",  name: "oxygen",        mass_m1000: 15999 },
    Element { z: 9,   symbol: "F",  name: "fluorine",      mass_m1000: 18998 },
    Element { z: 10,  symbol: "Ne", name: "neon",          mass_m1000: 20180 },
    Element { z: 11,  symbol: "Na", name: "sodium",        mass_m1000: 22990 },
    Element { z: 12,  symbol: "Mg", name: "magnesium",     mass_m1000: 24305 },
    Element { z: 13,  symbol: "Al", name: "aluminium",     mass_m1000: 26982 },
    Element { z: 14,  symbol: "Si", name: "silicon",       mass_m1000: 28086 },
    Element { z: 15,  symbol: "P",  name: "phosphorus",    mass_m1000: 30974 },
    Element { z: 16,  symbol: "S",  name: "sulfur",        mass_m1000: 32065 },
    Element { z: 17,  symbol: "Cl", name: "chlorine",      mass_m1000: 35453 },
    Element { z: 18,  symbol: "Ar", name: "argon",         mass_m1000: 39948 },
    Element { z: 19,  symbol: "K",  name: "potassium",     mass_m1000: 39098 },
    Element { z: 20,  symbol: "Ca", name: "calcium",       mass_m1000: 40078 },
    Element { z: 21,  symbol: "Sc", name: "scandium",      mass_m1000: 44956 },
    Element { z: 22,  symbol: "Ti", name: "titanium",      mass_m1000: 47867 },
    Element { z: 23,  symbol: "V",  name: "vanadium",      mass_m1000: 50942 },
    Element { z: 24,  symbol: "Cr", name: "chromium",      mass_m1000: 51996 },
    Element { z: 25,  symbol: "Mn", name: "manganese",     mass_m1000: 54938 },
    Element { z: 26,  symbol: "Fe", name: "iron",          mass_m1000: 55845 },
    Element { z: 27,  symbol: "Co", name: "cobalt",        mass_m1000: 58933 },
    Element { z: 28,  symbol: "Ni", name: "nickel",        mass_m1000: 58693 },
    Element { z: 29,  symbol: "Cu", name: "copper",        mass_m1000: 63546 },
    Element { z: 30,  symbol: "Zn", name: "zinc",          mass_m1000: 65380 },
    Element { z: 31,  symbol: "Ga", name: "gallium",       mass_m1000: 69723 },
    Element { z: 32,  symbol: "Ge", name: "germanium",     mass_m1000: 72640 },
    Element { z: 33,  symbol: "As", name: "arsenic",       mass_m1000: 74922 },
    Element { z: 34,  symbol: "Se", name: "selenium",      mass_m1000: 78960 },
    Element { z: 35,  symbol: "Br", name: "bromine",       mass_m1000: 79904 },
    Element { z: 36,  symbol: "Kr", name: "krypton",       mass_m1000: 83798 },
    Element { z: 37,  symbol: "Rb", name: "rubidium",      mass_m1000: 85468 },
    Element { z: 38,  symbol: "Sr", name: "strontium",     mass_m1000: 87620 },
    Element { z: 39,  symbol: "Y",  name: "yttrium",       mass_m1000: 88906 },
    Element { z: 40,  symbol: "Zr", name: "zirconium",     mass_m1000: 91224 },
    Element { z: 41,  symbol: "Nb", name: "niobium",       mass_m1000: 92906 },
    Element { z: 42,  symbol: "Mo", name: "molybdenum",    mass_m1000: 95960 },
    Element { z: 43,  symbol: "Tc", name: "technetium",    mass_m1000: 98000 },
    Element { z: 44,  symbol: "Ru", name: "ruthenium",     mass_m1000: 101070 },
    Element { z: 45,  symbol: "Rh", name: "rhodium",       mass_m1000: 102906 },
    Element { z: 46,  symbol: "Pd", name: "palladium",     mass_m1000: 106420 },
    Element { z: 47,  symbol: "Ag", name: "silver",        mass_m1000: 107868 },
    Element { z: 48,  symbol: "Cd", name: "cadmium",       mass_m1000: 112411 },
    Element { z: 49,  symbol: "In", name: "indium",        mass_m1000: 114818 },
    Element { z: 50,  symbol: "Sn", name: "tin",           mass_m1000: 118710 },
    Element { z: 51,  symbol: "Sb", name: "antimony",      mass_m1000: 121760 },
    Element { z: 52,  symbol: "Te", name: "tellurium",     mass_m1000: 127600 },
    Element { z: 53,  symbol: "I",  name: "iodine",        mass_m1000: 126904 },
    Element { z: 54,  symbol: "Xe", name: "xenon",         mass_m1000: 131293 },
    Element { z: 55,  symbol: "Cs", name: "caesium",       mass_m1000: 132905 },
    Element { z: 56,  symbol: "Ba", name: "barium",        mass_m1000: 137327 },
    Element { z: 57,  symbol: "La", name: "lanthanum",     mass_m1000: 138905 },
    Element { z: 58,  symbol: "Ce", name: "cerium",        mass_m1000: 140116 },
    Element { z: 59,  symbol: "Pr", name: "praseodymium",  mass_m1000: 140908 },
    Element { z: 60,  symbol: "Nd", name: "neodymium",     mass_m1000: 144242 },
    Element { z: 61,  symbol: "Pm", name: "promethium",    mass_m1000: 145000 },
    Element { z: 62,  symbol: "Sm", name: "samarium",      mass_m1000: 150360 },
    Element { z: 63,  symbol: "Eu", name: "europium",      mass_m1000: 151964 },
    Element { z: 64,  symbol: "Gd", name: "gadolinium",    mass_m1000: 157250 },
    Element { z: 65,  symbol: "Tb", name: "terbium",       mass_m1000: 158925 },
    Element { z: 66,  symbol: "Dy", name: "dysprosium",    mass_m1000: 162500 },
    Element { z: 67,  symbol: "Ho", name: "holmium",       mass_m1000: 164930 },
    Element { z: 68,  symbol: "Er", name: "erbium",        mass_m1000: 167259 },
    Element { z: 69,  symbol: "Tm", name: "thulium",       mass_m1000: 168934 },
    Element { z: 70,  symbol: "Yb", name: "ytterbium",     mass_m1000: 173054 },
    Element { z: 71,  symbol: "Lu", name: "lutetium",      mass_m1000: 174967 },
    Element { z: 72,  symbol: "Hf", name: "hafnium",       mass_m1000: 178490 },
    Element { z: 73,  symbol: "Ta", name: "tantalum",      mass_m1000: 180948 },
    Element { z: 74,  symbol: "W",  name: "tungsten",      mass_m1000: 183840 },
    Element { z: 75,  symbol: "Re", name: "rhenium",       mass_m1000: 186207 },
    Element { z: 76,  symbol: "Os", name: "osmium",        mass_m1000: 190230 },
    Element { z: 77,  symbol: "Ir", name: "iridium",       mass_m1000: 192217 },
    Element { z: 78,  symbol: "Pt", name: "platinum",      mass_m1000: 195084 },
    Element { z: 79,  symbol: "Au", name: "gold",          mass_m1000: 196967 },
    Element { z: 80,  symbol: "Hg", name: "mercury",       mass_m1000: 200590 },
    Element { z: 81,  symbol: "Tl", name: "thallium",      mass_m1000: 204383 },
    Element { z: 82,  symbol: "Pb", name: "lead",          mass_m1000: 207200 },
    Element { z: 83,  symbol: "Bi", name: "bismuth",       mass_m1000: 208980 },
    Element { z: 84,  symbol: "Po", name: "polonium",      mass_m1000: 209000 },
    Element { z: 85,  symbol: "At", name: "astatine",      mass_m1000: 210000 },
    Element { z: 86,  symbol: "Rn", name: "radon",         mass_m1000: 222000 },
    Element { z: 87,  symbol: "Fr", name: "francium",      mass_m1000: 223000 },
    Element { z: 88,  symbol: "Ra", name: "radium",        mass_m1000: 226000 },
    Element { z: 89,  symbol: "Ac", name: "actinium",      mass_m1000: 227000 },
    Element { z: 90,  symbol: "Th", name: "thorium",       mass_m1000: 232038 },
    Element { z: 91,  symbol: "Pa", name: "protactinium",  mass_m1000: 231036 },
    Element { z: 92,  symbol: "U",  name: "uranium",       mass_m1000: 238029 },
    Element { z: 93,  symbol: "Np", name: "neptunium",     mass_m1000: 237000 },
    Element { z: 94,  symbol: "Pu", name: "plutonium",     mass_m1000: 244000 },
    Element { z: 95,  symbol: "Am", name: "americium",     mass_m1000: 243000 },
    Element { z: 96,  symbol: "Cm", name: "curium",        mass_m1000: 247000 },
    Element { z: 97,  symbol: "Bk", name: "berkelium",     mass_m1000: 247000 },
    Element { z: 98,  symbol: "Cf", name: "californium",   mass_m1000: 251000 },
    Element { z: 99,  symbol: "Es", name: "einsteinium",   mass_m1000: 252000 },
    Element { z: 100, symbol: "Fm", name: "fermium",       mass_m1000: 257000 },
    Element { z: 101, symbol: "Md", name: "mendelevium",   mass_m1000: 258000 },
    Element { z: 102, symbol: "No", name: "nobelium",      mass_m1000: 259000 },
    Element { z: 103, symbol: "Lr", name: "lawrencium",    mass_m1000: 262000 },
    Element { z: 104, symbol: "Rf", name: "rutherfordium", mass_m1000: 265000 },
    Element { z: 105, symbol: "Db", name: "dubnium",       mass_m1000: 268000 },
    Element { z: 106, symbol: "Sg", name: "seaborgium",    mass_m1000: 271000 },
    Element { z: 107, symbol: "Bh", name: "bohrium",       mass_m1000: 270000 },
    Element { z: 108, symbol: "Hs", name: "hassium",       mass_m1000: 277000 },
    Element { z: 109, symbol: "Mt", name: "meitnerium",    mass_m1000: 276000 },
    Element { z: 110, symbol: "Ds", name: "darmstadtium",  mass_m1000: 281000 },
    Element { z: 111, symbol: "Rg", name: "roentgenium",   mass_m1000: 280000 },
    Element { z: 112, symbol: "Cn", name: "copernicium",   mass_m1000: 285000 },
    Element { z: 113, symbol: "Nh", name: "nihonium",      mass_m1000: 284000 },
    Element { z: 114, symbol: "Fl", name: "flerovium",     mass_m1000: 289000 },
    Element { z: 115, symbol: "Mc", name: "moscovium",     mass_m1000: 288000 },
    Element { z: 116, symbol: "Lv", name: "livermorium",   mass_m1000: 293000 },
    Element { z: 117, symbol: "Ts", name: "tennessine",    mass_m1000: 294000 },
    Element { z: 118, symbol: "Og", name: "oganesson",     mass_m1000: 294000 },
];

fn by_number(z: u8) -> Option<&'static Element> { ELEMENTS.iter().find(|e| e.z == z) }

fn by_symbol(sym: &str) -> Option<&'static Element> {
    let s = sym.trim();
    ELEMENTS.iter().find(|e| e.symbol.eq_ignore_ascii_case(s))
}

fn by_name(name: &str) -> Option<&'static Element> {
    let n = name.trim().to_ascii_lowercase();
    ELEMENTS.iter().find(|e| e.name == n)
}
