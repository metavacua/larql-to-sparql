//! Prompt sets for accuracy testing.
//!
//! 100 diverse prompts spanning: factual, completion, reasoning,
//! code, arithmetic, scientific, geographic, conversational.

/// A test prompt with expected top-1 token (or prefix thereof).
#[derive(Debug, Clone)]
pub struct TestPrompt {
    pub text: &'static str,
    pub expected_contains: &'static str,
    pub category: &'static str,
}

/// The Paris test — single pass/fail sanity check.
pub fn paris_test() -> TestPrompt {
    TestPrompt {
        text: "The capital of France is",
        expected_contains: "Paris",
        category: "factual",
    }
}

/// 100 diverse prompts for top-1 match rate testing.
pub fn diverse_100() -> Vec<TestPrompt> {
    vec![
        // Factual: capitals (20)
        TestPrompt { text: "The capital of France is", expected_contains: "Paris", category: "factual" },
        TestPrompt { text: "The capital of Germany is", expected_contains: "Berlin", category: "factual" },
        TestPrompt { text: "The capital of Japan is", expected_contains: "Tokyo", category: "factual" },
        TestPrompt { text: "The capital of Italy is", expected_contains: "Rome", category: "factual" },
        TestPrompt { text: "The capital of Spain is", expected_contains: "Madrid", category: "factual" },
        TestPrompt { text: "The capital of Brazil is", expected_contains: "Bras", category: "factual" },
        TestPrompt { text: "The capital of Australia is", expected_contains: "Canberra", category: "factual" },
        TestPrompt { text: "The capital of Canada is", expected_contains: "Ottawa", category: "factual" },
        TestPrompt { text: "The capital of Egypt is", expected_contains: "Cairo", category: "factual" },
        TestPrompt { text: "The capital of India is", expected_contains: "Delhi", category: "factual" },
        TestPrompt { text: "The capital of Mexico is", expected_contains: "Mexico", category: "factual" },
        TestPrompt { text: "The capital of Russia is", expected_contains: "Moscow", category: "factual" },
        TestPrompt { text: "The capital of China is", expected_contains: "Beijing", category: "factual" },
        TestPrompt { text: "The capital of South Korea is", expected_contains: "Seoul", category: "factual" },
        TestPrompt { text: "The capital of Turkey is", expected_contains: "Ankara", category: "factual" },
        TestPrompt { text: "The capital of Thailand is", expected_contains: "Bangkok", category: "factual" },
        TestPrompt { text: "The capital of Argentina is", expected_contains: "Buenos", category: "factual" },
        TestPrompt { text: "The capital of Sweden is", expected_contains: "Stockholm", category: "factual" },
        TestPrompt { text: "The capital of Norway is", expected_contains: "Oslo", category: "factual" },
        TestPrompt { text: "The capital of Poland is", expected_contains: "Warsaw", category: "factual" },

        // Factual: people (10)
        TestPrompt { text: "Mozart was born in", expected_contains: "Salzburg", category: "factual" },
        TestPrompt { text: "Einstein was born in", expected_contains: "Ulm", category: "factual" },
        TestPrompt { text: "Shakespeare was born in", expected_contains: "Strat", category: "factual" },
        TestPrompt { text: "The Mona Lisa was painted by", expected_contains: "Leonardo", category: "factual" },
        TestPrompt { text: "The theory of relativity was developed by", expected_contains: "Einstein", category: "factual" },
        TestPrompt { text: "The first president of the United States was", expected_contains: "George", category: "factual" },
        TestPrompt { text: "Apple Inc. was co-founded by Steve", expected_contains: "Jobs", category: "factual" },
        TestPrompt { text: "The author of Harry Potter is J.K.", expected_contains: "Rowling", category: "factual" },
        TestPrompt { text: "Beethoven's first name was", expected_contains: "Ludwig", category: "factual" },
        TestPrompt { text: "Isaac Newton discovered", expected_contains: "grav", category: "factual" },

        // Factual: science (10)
        TestPrompt { text: "Water freezes at", expected_contains: "0", category: "scientific" },
        TestPrompt { text: "The chemical symbol for gold is", expected_contains: "Au", category: "scientific" },
        TestPrompt { text: "The chemical formula for water is", expected_contains: "H", category: "scientific" },
        TestPrompt { text: "The speed of light is approximately", expected_contains: "3", category: "scientific" },
        TestPrompt { text: "The largest planet in our solar system is", expected_contains: "Jupiter", category: "scientific" },
        TestPrompt { text: "DNA stands for deoxyribonucle", expected_contains: "ic", category: "scientific" },
        TestPrompt { text: "The atomic number of carbon is", expected_contains: "6", category: "scientific" },
        TestPrompt { text: "Photosynthesis converts sunlight into", expected_contains: "energy", category: "scientific" },
        TestPrompt { text: "The boiling point of water is", expected_contains: "100", category: "scientific" },
        TestPrompt { text: "The nearest star to Earth is the", expected_contains: "Sun", category: "scientific" },

        // Factual: geography (10)
        TestPrompt { text: "The longest river in Africa is the", expected_contains: "Nile", category: "geographic" },
        TestPrompt { text: "The tallest mountain in the world is", expected_contains: "Everest", category: "geographic" },
        TestPrompt { text: "The largest ocean is the", expected_contains: "Pacific", category: "geographic" },
        TestPrompt { text: "The Amazon River flows through", expected_contains: "Brazil", category: "geographic" },
        TestPrompt { text: "The Sahara Desert is located in", expected_contains: "Africa", category: "geographic" },
        TestPrompt { text: "The Great Wall of China is located in", expected_contains: "China", category: "geographic" },
        TestPrompt { text: "The currency of Japan is the", expected_contains: "yen", category: "geographic" },
        TestPrompt { text: "The currency of the United Kingdom is the", expected_contains: "pound", category: "geographic" },
        TestPrompt { text: "The official language of Brazil is", expected_contains: "Portug", category: "geographic" },
        TestPrompt { text: "The smallest continent is", expected_contains: "Australia", category: "geographic" },

        // Completion (10)
        TestPrompt { text: "To be or not to be, that is the", expected_contains: "question", category: "completion" },
        TestPrompt { text: "I think, therefore I", expected_contains: "am", category: "completion" },
        TestPrompt { text: "All that glitters is not", expected_contains: "gold", category: "completion" },
        TestPrompt { text: "A journey of a thousand miles begins with a single", expected_contains: "step", category: "completion" },
        TestPrompt { text: "The early bird catches the", expected_contains: "worm", category: "completion" },
        TestPrompt { text: "Actions speak louder than", expected_contains: "words", category: "completion" },
        TestPrompt { text: "Rome was not built in a", expected_contains: "day", category: "completion" },
        TestPrompt { text: "Knowledge is", expected_contains: "power", category: "completion" },
        TestPrompt { text: "Practice makes", expected_contains: "perfect", category: "completion" },
        TestPrompt { text: "Where there is smoke, there is", expected_contains: "fire", category: "completion" },

        // Arithmetic (10)
        TestPrompt { text: "2 + 2 =", expected_contains: "4", category: "arithmetic" },
        TestPrompt { text: "10 × 10 =", expected_contains: "100", category: "arithmetic" },
        TestPrompt { text: "100 / 4 =", expected_contains: "25", category: "arithmetic" },
        TestPrompt { text: "The square root of 144 is", expected_contains: "12", category: "arithmetic" },
        TestPrompt { text: "15 + 27 =", expected_contains: "42", category: "arithmetic" },
        TestPrompt { text: "One dozen equals", expected_contains: "12", category: "arithmetic" },
        TestPrompt { text: "A century is", expected_contains: "100", category: "arithmetic" },
        TestPrompt { text: "One kilometer equals", expected_contains: "1", category: "arithmetic" },
        TestPrompt { text: "There are 60 seconds in a", expected_contains: "minute", category: "arithmetic" },
        TestPrompt { text: "There are 24 hours in a", expected_contains: "day", category: "arithmetic" },

        // Code (10)
        TestPrompt { text: "In Python, to print 'hello' you write print(", expected_contains: "'", category: "code" },
        TestPrompt { text: "In JavaScript, a variable is declared with let, const, or", expected_contains: "var", category: "code" },
        TestPrompt { text: "HTML stands for Hyper", expected_contains: "Text", category: "code" },
        TestPrompt { text: "The HTTP status code for 'Not Found' is", expected_contains: "404", category: "code" },
        TestPrompt { text: "In SQL, to select all columns you use SELECT", expected_contains: "*", category: "code" },
        TestPrompt { text: "Git is a distributed version", expected_contains: "control", category: "code" },
        TestPrompt { text: "JSON stands for JavaScript Object", expected_contains: "Notation", category: "code" },
        TestPrompt { text: "The file extension for Python files is .", expected_contains: "py", category: "code" },
        TestPrompt { text: "In CSS, to make text bold you use font-weight:", expected_contains: "bold", category: "code" },
        TestPrompt { text: "The command to list files in Linux is", expected_contains: "ls", category: "code" },

        // Conversational (10)
        TestPrompt { text: "How are you today? I'm doing", expected_contains: "well", category: "conversational" },
        TestPrompt { text: "Thank you very much! You're", expected_contains: "welcome", category: "conversational" },
        TestPrompt { text: "Good morning! How did you", expected_contains: "sleep", category: "conversational" },
        TestPrompt { text: "See you later! Have a great", expected_contains: "day", category: "conversational" },
        TestPrompt { text: "Happy birthday! How old are", expected_contains: "you", category: "conversational" },
        TestPrompt { text: "Sorry for the delay. I was", expected_contains: "busy", category: "conversational" },
        TestPrompt { text: "What do you think about", expected_contains: "the", category: "conversational" },
        TestPrompt { text: "Let me know if you need any", expected_contains: "help", category: "conversational" },
        TestPrompt { text: "I completely agree with", expected_contains: "you", category: "conversational" },
        TestPrompt { text: "That's a really good", expected_contains: "point", category: "conversational" },

        // Reasoning (10)
        TestPrompt { text: "If it rains, the ground gets", expected_contains: "wet", category: "reasoning" },
        TestPrompt { text: "The opposite of hot is", expected_contains: "cold", category: "reasoning" },
        TestPrompt { text: "The color of grass is", expected_contains: "green", category: "reasoning" },
        TestPrompt { text: "The day after Monday is", expected_contains: "Tuesday", category: "reasoning" },
        TestPrompt { text: "Ice is the solid form of", expected_contains: "water", category: "reasoning" },
        TestPrompt { text: "The month after January is", expected_contains: "February", category: "reasoning" },
        TestPrompt { text: "Cats are a type of", expected_contains: "animal", category: "reasoning" },
        TestPrompt { text: "The sun rises in the", expected_contains: "east", category: "reasoning" },
        TestPrompt { text: "The plural of child is", expected_contains: "children", category: "reasoning" },
        TestPrompt { text: "A triangle has three", expected_contains: "side", category: "reasoning" },
    ]
}

/// Short prompt set for quick validation (20 prompts).
pub fn quick_20() -> Vec<TestPrompt> {
    diverse_100().into_iter().step_by(5).collect()
}
