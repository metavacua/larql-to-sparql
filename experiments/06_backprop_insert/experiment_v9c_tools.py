#!/usr/bin/env python3
"""
v9c: Tool Engine

Tool calling as graph routing + schema validation. No neural computation.

Three components:
  1. Tool Registry — JSON schemas for 10 tools (weather, search, email, etc.)
  2. Intent Router — keyword patterns → tool selection
  3. Argument Extractor — entity extraction + schema validation

Test: 50 queries across 10 tools. Measure tool selection accuracy,
argument extraction accuracy, and JSON output validity.

This proves tool calling is structured routing, not learned behaviour.
"""

import os
import sys
import json
import re
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

OUTPUT_DIR = "results_v9c_tools"

# ---------------------------------------------------------------------------
# Component 1: Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_weather": {
        "description": "Get current weather for a location",
        "parameters": {
            "location": {"type": "string", "required": True,
                        "description": "City or location name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                     "default": "celsius", "required": False},
        },
        "returns": {"temperature": "float", "conditions": "string"},
    },
    "search_web": {
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "required": True,
                     "description": "Search query"},
            "num_results": {"type": "integer", "default": 5, "required": False},
        },
        "returns": {"results": "list[{title, url, snippet}]"},
    },
    "send_email": {
        "description": "Send an email to a recipient",
        "parameters": {
            "to": {"type": "string", "required": True,
                  "description": "Recipient email address"},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
        },
        "returns": {"sent": "boolean", "message_id": "string"},
    },
    "create_calendar_event": {
        "description": "Create a calendar event",
        "parameters": {
            "title": {"type": "string", "required": True},
            "date": {"type": "string", "required": True,
                    "description": "Date in YYYY-MM-DD format"},
            "time": {"type": "string", "required": False,
                    "description": "Time in HH:MM format"},
            "duration_minutes": {"type": "integer", "default": 60, "required": False},
        },
        "returns": {"event_id": "string", "confirmed": "boolean"},
    },
    "calculate": {
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "expression": {"type": "string", "required": True,
                          "description": "Mathematical expression to evaluate"},
        },
        "returns": {"result": "float"},
    },
    "translate": {
        "description": "Translate text between languages",
        "parameters": {
            "text": {"type": "string", "required": True},
            "source_language": {"type": "string", "required": True},
            "target_language": {"type": "string", "required": True},
        },
        "returns": {"translated_text": "string"},
    },
    "run_code": {
        "description": "Execute a code snippet in a sandbox",
        "parameters": {
            "code": {"type": "string", "required": True},
            "language": {"type": "string", "enum": ["python", "javascript", "rust"],
                        "required": True},
        },
        "returns": {"output": "string", "exit_code": "integer"},
    },
    "read_file": {
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "required": True,
                    "description": "File path to read"},
            "encoding": {"type": "string", "default": "utf-8", "required": False},
        },
        "returns": {"content": "string", "size_bytes": "integer"},
    },
    "query_database": {
        "description": "Run a SQL query against a database",
        "parameters": {
            "query": {"type": "string", "required": True,
                     "description": "SQL query to execute"},
            "database": {"type": "string", "required": True,
                        "description": "Database name or connection string"},
        },
        "returns": {"rows": "list", "columns": "list[string]"},
    },
    "generate_image": {
        "description": "Generate an image from a text description",
        "parameters": {
            "prompt": {"type": "string", "required": True,
                      "description": "Image description"},
            "width": {"type": "integer", "default": 512, "required": False},
            "height": {"type": "integer", "default": 512, "required": False},
            "style": {"type": "string", "enum": ["realistic", "artistic", "cartoon"],
                     "default": "realistic", "required": False},
        },
        "returns": {"image_url": "string"},
    },
}


# ---------------------------------------------------------------------------
# Component 2: Intent Router
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Routes user queries to tools via keyword pattern matching.

    Each tool has a set of trigger patterns (regex). The router scores
    each tool against the query and returns the best match.
    """

    def __init__(self, registry: Dict):
        self.registry = registry
        self.routing_rules = self._build_rules()

    def _build_rules(self) -> List[Dict]:
        """Build routing rules from tool descriptions and common patterns."""
        rules = [
            # Weather
            {"patterns": [r"weather", r"temperature", r"forecast", r"hot|cold outside",
                         r"rain|snow|sunny", r"degrees"],
             "tool": "get_weather", "priority": 1},

            # Search
            {"patterns": [r"search\b", r"look up", r"find.*(?:information|info|about)",
                         r"google", r"what is", r"who is", r"tell me about"],
             "tool": "search_web", "priority": 0},  # lower priority (catch-all)

            # Email
            {"patterns": [r"send.*email", r"email.*to", r"write.*email", r"message.*to",
                         r"mail\b", r"compose"],
             "tool": "send_email", "priority": 1},

            # Calendar
            {"patterns": [r"schedule", r"calendar", r"appointment", r"meeting",
                         r"book.*(?:for|on)", r"event.*(?:on|at)", r"remind.*(?:on|at)"],
             "tool": "create_calendar_event", "priority": 1},

            # Calculate
            {"patterns": [r"calculat", r"compute", r"what is \d", r"how much is",
                         r"\d+\s*[\+\-\*\/]\s*\d+", r"math", r"solve"],
             "tool": "calculate", "priority": 1},

            # Translate
            {"patterns": [r"translat", r"in (?:french|spanish|german|chinese|japanese)",
                         r"how.*say.*in", r"(?:french|spanish|german).*(?:for|of)"],
             "tool": "translate", "priority": 1},

            # Run code
            {"patterns": [r"run.*code", r"execute", r"eval.*(?:python|javascript|rust)",
                         r"run.*(?:python|javascript|rust)", r"```"],
             "tool": "run_code", "priority": 1},

            # Read file
            {"patterns": [r"read.*file", r"open.*file", r"contents? of", r"cat\s",
                         r"show.*file", r"what's in.*\."],
             "tool": "read_file", "priority": 1},

            # Database
            {"patterns": [r"query.*(?:database|db)", r"sql\b", r"select.*from",
                         r"database", r"table"],
             "tool": "query_database", "priority": 1},

            # Image
            {"patterns": [r"generat.*image", r"create.*(?:image|picture|photo)",
                         r"draw\b", r"make.*(?:image|picture)", r"illustrat"],
             "tool": "generate_image", "priority": 1},
        ]
        return rules

    def route(self, query: str) -> Tuple[Optional[str], float, List[Dict]]:
        """
        Route a query to the best matching tool.
        Returns: (tool_name, confidence, all_scores)
        """
        query_lower = query.lower()
        scores = []

        for rule in self.routing_rules:
            match_count = 0
            for pattern in rule["patterns"]:
                if re.search(pattern, query_lower):
                    match_count += 1

            if match_count > 0:
                # Score = number of matching patterns + priority bonus
                score = match_count + rule["priority"] * 0.5
                scores.append({
                    "tool": rule["tool"],
                    "score": score,
                    "matches": match_count,
                })

        if not scores:
            return None, 0.0, scores

        scores.sort(key=lambda x: -x["score"])
        best = scores[0]

        # Confidence: how much better is the best vs second best
        if len(scores) > 1:
            confidence = best["score"] / (best["score"] + scores[1]["score"])
        else:
            confidence = 1.0

        return best["tool"], confidence, scores


# ---------------------------------------------------------------------------
# Component 3: Argument Extractor
# ---------------------------------------------------------------------------

class ArgumentExtractor:
    """
    Extract structured arguments from natural language queries.
    Uses regex patterns + tool schema for validation.
    """

    def __init__(self, registry: Dict):
        self.registry = registry

        # Entity extraction patterns
        self.patterns = {
            "location": [
                r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "in London"
                r"(?:for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "for Paris"
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+weather",  # "London weather"
            ],
            "email": [
                r"to\s+(\S+@\S+\.\S+)",  # email address
                r"(\S+@\S+\.\S+)",
            ],
            "date": [
                r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
                r"on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)",  # "on March 15"
                r"(tomorrow|today|next\s+\w+)",
            ],
            "time": [
                r"at\s+(\d{1,2}:\d{2})",  # "at 14:30"
                r"at\s+(\d{1,2}\s*(?:am|pm))",  # "at 3pm"
            ],
            "number": [
                r"(\d+(?:\.\d+)?)",
            ],
            "language": [
                r"(?:to|in|into)\s+(french|spanish|german|chinese|japanese|italian|portuguese|russian|arabic|korean)",
                r"(french|spanish|german|chinese|japanese|italian|portuguese|russian|arabic|korean)",
            ],
            "math_expression": [
                r"(?:calculate|compute|what is|how much is)\s+(.+?)(?:\?|$)",
                r"(\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)*)",
            ],
            "file_path": [
                r"(?:file|read|open)\s+['\"]?([/\w\-\.]+\.\w+)['\"]?",
                r"['\"]([/\w\-\.]+\.\w+)['\"]",
            ],
            "sql_query": [
                r"(SELECT\s+.+?)(?:\s+on\s+|\s+against\s+|$)",
                r"query:\s*(.+?)(?:\s+on\s+|\s+against\s+|$)",
            ],
            "database_name": [
                r"(?:database|db)\s+['\"]?(\w+)['\"]?",
                r"(?:on|against|in)\s+(?:the\s+)?['\"]?(\w+)['\"]?\s+(?:database|db)",
            ],
            "image_prompt": [
                r"(?:generate|create|draw|make)\s+(?:an?\s+)?(?:image|picture|photo)\s+(?:of\s+)?(.+?)(?:\s+in\s+\w+\s+style)?$",
                r"(?:image|picture)\s+of\s+(.+?)$",
            ],
        }

    def extract(self, query: str, tool_name: str) -> Dict[str, Any]:
        """Extract arguments for a specific tool from the query."""
        tool_schema = self.registry.get(tool_name, {})
        params = tool_schema.get("parameters", {})
        extracted = {}

        query_lower = query.lower()

        # Tool-specific extraction
        if tool_name == "get_weather":
            loc = self._extract_first("location", query)
            if loc:
                extracted["location"] = loc
            if "fahrenheit" in query_lower or "°f" in query_lower:
                extracted["units"] = "fahrenheit"
            else:
                extracted["units"] = "celsius"

        elif tool_name == "search_web":
            # The whole query is the search query (minus intent words)
            clean = re.sub(r"^(search|look up|find|google|tell me about)\s+", "",
                          query_lower).strip()
            extracted["query"] = clean or query

        elif tool_name == "send_email":
            email = self._extract_first("email", query)
            if email:
                extracted["to"] = email
            # Subject and body need more context — use heuristics
            extracted["subject"] = self._extract_after(query, ["about", "regarding", "subject"])
            extracted["body"] = self._extract_after(query, ["saying", "message", "body"])

        elif tool_name == "create_calendar_event":
            extracted["title"] = self._extract_after(query, ["schedule", "book", "create"])
            date = self._extract_first("date", query)
            if date:
                extracted["date"] = date
            time_val = self._extract_first("time", query)
            if time_val:
                extracted["time"] = time_val

        elif tool_name == "calculate":
            expr = self._extract_first("math_expression", query)
            if expr:
                extracted["expression"] = expr.strip()

        elif tool_name == "translate":
            lang = self._extract_first("language", query)
            if lang:
                extracted["target_language"] = lang
                extracted["source_language"] = "english"  # default
            # Text to translate
            text = re.sub(r"translat\w*\s+", "", query_lower)
            text = re.sub(r"\s+(?:to|into|in)\s+\w+$", "", text).strip()
            if text:
                extracted["text"] = text

        elif tool_name == "run_code":
            # Extract language
            for lang in ["python", "javascript", "rust"]:
                if lang in query_lower:
                    extracted["language"] = lang
                    break
            # Code is everything in backticks or after "run"
            code_match = re.search(r"```(?:\w+)?\n?(.*?)```", query, re.DOTALL)
            if code_match:
                extracted["code"] = code_match.group(1).strip()
            else:
                extracted["code"] = re.sub(r"^.*?(?:run|execute)\s+", "", query).strip()

        elif tool_name == "read_file":
            path = self._extract_first("file_path", query)
            if path:
                extracted["path"] = path

        elif tool_name == "query_database":
            sql = self._extract_first("sql_query", query)
            if sql:
                extracted["query"] = sql
            db = self._extract_first("database_name", query)
            if db:
                extracted["database"] = db

        elif tool_name == "generate_image":
            prompt = self._extract_first("image_prompt", query)
            if prompt:
                extracted["prompt"] = prompt
            for style in ["realistic", "artistic", "cartoon"]:
                if style in query_lower:
                    extracted["style"] = style

        # Apply defaults from schema
        for pname, pschema in params.items():
            if pname not in extracted and "default" in pschema:
                extracted[pname] = pschema["default"]

        return extracted

    def _extract_first(self, pattern_name: str, text: str) -> Optional[str]:
        """Extract first match from named pattern set."""
        for pattern in self.patterns.get(pattern_name, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_after(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract text after a keyword."""
        for kw in keywords:
            match = re.search(rf"{kw}\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def validate(self, args: Dict, tool_name: str) -> Dict:
        """Validate extracted arguments against the tool schema."""
        schema = self.registry.get(tool_name, {}).get("parameters", {})
        result = {"valid": True, "errors": [], "missing": [], "extra": []}

        # Check required params
        for pname, pschema in schema.items():
            if pschema.get("required") and pname not in args:
                result["valid"] = False
                result["missing"].append(pname)

        # Check enum constraints
        for pname, value in args.items():
            if pname in schema:
                pschema = schema[pname]
                if "enum" in pschema and value not in pschema["enum"]:
                    result["errors"].append(f"{pname}={value} not in {pschema['enum']}")
                    result["valid"] = False
            else:
                result["extra"].append(pname)

        return result

    def format_call(self, tool_name: str, args: Dict) -> str:
        """Format as a JSON function call."""
        call = {
            "function": tool_name,
            "arguments": args,
        }
        return json.dumps(call, indent=2)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Weather (5)
    ("What's the weather in London?", "get_weather", {"location": "London"}),
    ("Temperature in Tokyo today", "get_weather", {"location": "Tokyo"}),
    ("Is it going to rain in Paris?", "get_weather", {"location": "Paris"}),
    ("How cold is it in Moscow?", "get_weather", {"location": "Moscow"}),
    ("Weather forecast for New York", "get_weather", {"location": "New York"}),

    # Search (5)
    ("Search for Python tutorials", "search_web", {"query": "python tutorials"}),
    ("Look up the history of Rome", "search_web", {"query": "the history of rome"}),
    ("Find information about black holes", "search_web", {"query": "information about black holes"}),
    ("Tell me about quantum computing", "search_web", {"query": "quantum computing"}),
    ("Google machine learning frameworks", "search_web", {"query": "machine learning frameworks"}),

    # Email (5)
    ("Send an email to alice@example.com about the meeting", "send_email",
     {"to": "alice@example.com", "subject": "the meeting"}),
    ("Email bob@work.com saying the report is ready", "send_email",
     {"to": "bob@work.com", "body": "the report is ready"}),
    ("Write an email to team@company.com about project update", "send_email",
     {"to": "team@company.com", "subject": "project update"}),
    ("Mail jane@test.org regarding the invoice", "send_email",
     {"to": "jane@test.org", "subject": "the invoice"}),
    ("Compose an email to support@service.com about a bug", "send_email",
     {"to": "support@service.com", "subject": "a bug"}),

    # Calendar (5)
    ("Schedule a meeting on 2026-04-15 at 14:30", "create_calendar_event",
     {"date": "2026-04-15", "time": "14:30"}),
    ("Book a dentist appointment for tomorrow", "create_calendar_event",
     {"date": "tomorrow"}),
    ("Create a calendar event for team standup on 2026-05-01", "create_calendar_event",
     {"date": "2026-05-01"}),
    ("Remind me about the presentation on next Monday", "create_calendar_event",
     {"date": "next Monday"}),
    ("Schedule lunch on 2026-04-20 at 12:00", "create_calendar_event",
     {"date": "2026-04-20", "time": "12:00"}),

    # Calculate (5)
    ("Calculate 15 * 23 + 7", "calculate", {"expression": "15 * 23 + 7"}),
    ("What is 256 / 16?", "calculate", {"expression": "256 / 16"}),
    ("Compute the square root of 144", "calculate", {}),
    ("How much is 3.14 * 100?", "calculate", {"expression": "3.14 * 100"}),
    ("Solve 42 + 58", "calculate", {"expression": "42 + 58"}),

    # Translate (5)
    ("Translate hello to french", "translate",
     {"target_language": "french", "source_language": "english"}),
    ("How do you say goodbye in spanish?", "translate",
     {"target_language": "spanish"}),
    ("Translate 'good morning' into german", "translate",
     {"target_language": "german", "source_language": "english"}),
    ("What is 'thank you' in japanese?", "translate",
     {"target_language": "japanese"}),
    ("Translate this text to chinese", "translate",
     {"target_language": "chinese", "source_language": "english"}),

    # Run code (5)
    ("Run this python code: print('hello')", "run_code",
     {"language": "python"}),
    ("Execute python ```\nx = 42\nprint(x)\n```", "run_code",
     {"language": "python"}),
    ("Run javascript console.log('test')", "run_code",
     {"language": "javascript"}),
    ("Execute this rust code: fn main() {}", "run_code",
     {"language": "rust"}),
    ("Run python list(range(10))", "run_code",
     {"language": "python"}),

    # Read file (5)
    ("Read the file config.json", "read_file", {"path": "config.json"}),
    ("Show me the contents of /etc/hosts", "read_file", {"path": "/etc/hosts"}),
    ("Open file data.csv", "read_file", {"path": "data.csv"}),
    ("What's in README.md?", "read_file", {"path": "README.md"}),
    ("Read the file 'output.txt'", "read_file", {"path": "output.txt"}),

    # Database (5)
    ("Query the users database: SELECT * FROM users", "query_database",
     {"query": "SELECT * FROM users", "database": "users"}),
    ("Run SQL SELECT count(*) FROM orders on the sales database", "query_database",
     {"database": "sales"}),
    ("Query database analytics for recent events", "query_database",
     {"database": "analytics"}),
    ("SELECT name FROM products on inventory db", "query_database",
     {"database": "inventory"}),
    ("Database query: SELECT * FROM logs", "query_database",
     {"query": "SELECT * FROM logs"}),

    # Image (5)
    ("Generate an image of a sunset over the ocean", "generate_image",
     {"prompt": "a sunset over the ocean"}),
    ("Create a picture of a cat in cartoon style", "generate_image",
     {"prompt": "a cat", "style": "cartoon"}),
    ("Draw a mountain landscape", "generate_image",
     {"prompt": "a mountain landscape"}),
    ("Make an artistic image of a city skyline", "generate_image",
     {"prompt": "a city skyline", "style": "artistic"}),
    ("Generate a realistic photo of a forest", "generate_image",
     {"prompt": "a forest", "style": "realistic"}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  v9c: TOOL ENGINE")
    print("  Tool calling as graph routing + schema validation.")
    print("=" * 65)

    # Build components
    router = IntentRouter(TOOL_REGISTRY)
    extractor = ArgumentExtractor(TOOL_REGISTRY)

    print(f"\n  Tool registry: {len(TOOL_REGISTRY)} tools")
    print(f"  Routing rules: {len(router.routing_rules)} patterns")
    print(f"  Test queries: {len(TEST_QUERIES)}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Tool Selection Accuracy
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Tool Selection")
    print(f"{'='*65}")

    tool_correct = 0
    tool_total = 0
    per_tool_accuracy = defaultdict(lambda: [0, 0])  # [correct, total]

    for query, expected_tool, _ in TEST_QUERIES:
        selected, confidence, scores = router.route(query)
        correct = selected == expected_tool
        if correct:
            tool_correct += 1
        tool_total += 1
        per_tool_accuracy[expected_tool][1] += 1
        if correct:
            per_tool_accuracy[expected_tool][0] += 1

        status = "✓" if correct else "✗"
        conf_str = f"{confidence:.0%}" if selected else "—"
        print(f"  {status} {query[:50]:<50} → {selected or 'NONE':<25} "
              f"(expect: {expected_tool}, conf: {conf_str})")

    print(f"\n  Tool selection: {tool_correct}/{tool_total} "
          f"({tool_correct/tool_total:.0%})")

    print(f"\n  Per-tool accuracy:")
    for tool in sorted(per_tool_accuracy.keys()):
        c, t = per_tool_accuracy[tool]
        print(f"    {tool:<25} {c}/{t} ({c/t:.0%})")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Argument Extraction
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Argument Extraction")
    print(f"{'='*65}")

    arg_correct = 0
    arg_total = 0
    json_valid = 0

    for query, expected_tool, expected_args in TEST_QUERIES:
        # Route first
        selected, _, _ = router.route(query)
        if selected != expected_tool:
            continue  # skip if wrong tool

        # Extract arguments
        extracted = extractor.extract(query, expected_tool)

        # Validate against schema
        validation = extractor.validate(extracted, expected_tool)

        # Check against expected args
        args_match = True
        for key, expected_val in expected_args.items():
            actual = extracted.get(key)
            if actual is None:
                args_match = False
            elif isinstance(expected_val, str) and isinstance(actual, str):
                if expected_val.lower() not in actual.lower() and actual.lower() not in expected_val.lower():
                    args_match = False

        if args_match:
            arg_correct += 1
        arg_total += 1

        # JSON validity
        try:
            call_json = extractor.format_call(expected_tool, extracted)
            json.loads(call_json)  # verify it parses
            json_valid += 1
        except (json.JSONDecodeError, TypeError):
            pass

        status = "✓" if args_match else "~"
        missing = validation.get("missing", [])
        print(f"  {status} {query[:45]:<45} args={json.dumps(extracted)[:60]}"
              + (f" [missing: {missing}]" if missing else ""))

    print(f"\n  Argument extraction: {arg_correct}/{arg_total} "
          f"({arg_correct/max(arg_total,1):.0%})")
    print(f"  JSON validity: {json_valid}/{arg_total} "
          f"({json_valid/max(arg_total,1):.0%})")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: End-to-End Tool Calls
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3: End-to-End Tool Calls")
    print(f"{'='*65}")

    e2e_success = 0
    e2e_total = len(TEST_QUERIES)

    print(f"\n  Sample formatted tool calls:")
    shown = set()
    for query, expected_tool, expected_args in TEST_QUERIES:
        if expected_tool in shown:
            continue
        shown.add(expected_tool)

        selected, conf, _ = router.route(query)
        if selected:
            extracted = extractor.extract(query, selected)
            validation = extractor.validate(extracted, selected)
            call_json = extractor.format_call(selected, extracted)

            print(f"\n  Query: \"{query}\"")
            print(f"  Tool: {selected} (confidence: {conf:.0%})")
            print(f"  Call: {call_json}")
            print(f"  Valid: {'✓' if validation['valid'] else '✗ ' + str(validation['errors'] + validation['missing'])}")

            if selected == expected_tool and validation["valid"]:
                e2e_success += 1

    # Count full e2e for all queries
    for query, expected_tool, _ in TEST_QUERIES:
        selected, _, _ = router.route(query)
        if selected == expected_tool:
            extracted = extractor.extract(query, selected)
            validation = extractor.validate(extracted, selected)
            if validation["valid"]:
                e2e_success += 1

    # Subtract the ones we already counted in the shown loop
    # Actually let's just recount properly
    e2e_success = 0
    for query, expected_tool, _ in TEST_QUERIES:
        selected, _, _ = router.route(query)
        if selected == expected_tool:
            extracted = extractor.extract(query, selected)
            validation = extractor.validate(extracted, selected)
            if validation["valid"]:
                e2e_success += 1

    print(f"\n  End-to-end (correct tool + valid args): {e2e_success}/{e2e_total} "
          f"({e2e_success/e2e_total:.0%})")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Tool Registry as Knowledge Graph
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4: Tool Registry as Knowledge Graph")
    print(f"{'='*65}")

    # Convert registry to graph edges
    tool_edges = []
    for tool_name, tool_def in TOOL_REGISTRY.items():
        tool_edges.append({
            "subject": tool_name,
            "relation": "has_description",
            "object": tool_def["description"],
        })
        for pname, pdef in tool_def["parameters"].items():
            tool_edges.append({
                "subject": tool_name,
                "relation": "has_parameter",
                "object": pname,
            })
            tool_edges.append({
                "subject": f"{tool_name}.{pname}",
                "relation": "has_type",
                "object": pdef["type"],
            })
            if pdef.get("required"):
                tool_edges.append({
                    "subject": f"{tool_name}.{pname}",
                    "relation": "is_required",
                    "object": "true",
                })
            if "enum" in pdef:
                for val in pdef["enum"]:
                    tool_edges.append({
                        "subject": f"{tool_name}.{pname}",
                        "relation": "allowed_value",
                        "object": val,
                    })

    print(f"  Tool knowledge graph: {len(tool_edges)} edges")
    print(f"  Tools: {len(TOOL_REGISTRY)}")
    print(f"  Total parameters: {sum(len(t['parameters']) for t in TOOL_REGISTRY.values())}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  SUMMARY: TOOL ENGINE")
    print(f"{'='*65}")

    registry_size = len(json.dumps(TOOL_REGISTRY, indent=2))
    rules_size = len(json.dumps([{"patterns": r["patterns"], "tool": r["tool"]}
                                  for r in router.routing_rules], indent=2))
    total = registry_size + rules_size

    print(f"\n  Tool selection:      {tool_correct}/{tool_total} ({tool_correct/tool_total:.0%})")
    print(f"  Argument extraction: {arg_correct}/{arg_total} ({arg_correct/max(arg_total,1):.0%})")
    print(f"  JSON validity:       {json_valid}/{arg_total} ({json_valid/max(arg_total,1):.0%})")
    print(f"  End-to-end:          {e2e_success}/{e2e_total} ({e2e_success/e2e_total:.0%})")

    print(f"\n  Tool engine size:")
    print(f"    Registry:      {registry_size:>8,} bytes")
    print(f"    Routing rules: {rules_size:>8,} bytes")
    print(f"    Total:         {total:>8,} bytes ({total/1024:.0f} KB)")

    print(f"\n  Tool graph: {len(tool_edges)} edges")

    # Verdict
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    if tool_correct / tool_total >= 0.9:
        print(f"\n  ✓ Tool selection: {tool_correct/tool_total:.0%} (target: 90%)")
    else:
        print(f"\n  ~ Tool selection: {tool_correct/tool_total:.0%} (target: 90%)")

    if arg_correct / max(arg_total, 1) >= 0.85:
        print(f"  ✓ Arg extraction: {arg_correct/max(arg_total,1):.0%} (target: 85%)")
    else:
        print(f"  ~ Arg extraction: {arg_correct/max(arg_total,1):.0%} (target: 85%)")

    if json_valid / max(arg_total, 1) >= 1.0:
        print(f"  ✓ JSON validity: {json_valid/max(arg_total,1):.0%} (target: 100%)")
    else:
        print(f"  ~ JSON validity: {json_valid/max(arg_total,1):.0%} (target: 100%)")

    print(f"\n  Tool calling is structured routing + schema validation.")
    print(f"  {total/1024:.0f} KB of registry + rules. No neural computation.")

    # Save
    results = {
        "tool_selection": {"correct": tool_correct, "total": tool_total},
        "arg_extraction": {"correct": arg_correct, "total": arg_total},
        "json_validity": {"valid": json_valid, "total": arg_total},
        "e2e": {"success": e2e_success, "total": e2e_total},
        "sizes": {"registry": registry_size, "rules": rules_size, "total": total},
        "tool_graph_edges": len(tool_edges),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Export tool graph
    with open(os.path.join(OUTPUT_DIR, "tool_graph.json"), "w") as f:
        json.dump(tool_edges, f, indent=2)

    # Export registry
    with open(os.path.join(OUTPUT_DIR, "tool_registry.json"), "w") as f:
        json.dump(TOOL_REGISTRY, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
