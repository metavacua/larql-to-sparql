#!/usr/bin/env python3
"""Legs 5-6 (native-template-wiring) measurement, per
docs/specs/2026-07-16-larql-goose-toolcalling-design.md ADR-6.

Honesty note (see the research residual, Q5): Goose's actual
`template_result_supports_native_tool_calling` predicate depends on llama.cpp's own
C++ capability-detection dry-run (`common_chat_templates_apply` via the `llama-cpp-2`
FFI binding) -- that is not something this standalone script can faithfully
reproduce without also linking llama.cpp. This script instead checks a real, cheap,
and honestly-labeled PROXY question: does the model's own `tokenizer_config.json`
chat_template even reference tool-calling constructs (a `tools` variable, a
`tool_calls` role/field) at all? A template with zero such references categorically
cannot support native tool-calling regardless of llama.cpp's exact runtime detection;
a template that does reference them is a necessary-but-not-sufficient signal, not a
confirmed "yes." Both facts are recorded plainly, not conflated with the real
predicate this script cannot run.
"""
import argparse
import json
import sys

from huggingface_hub import hf_hub_download


def load_chat_template(model: str) -> str | None:
    path = hf_hub_download(repo_id=model, filename="tokenizer_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("chat_template")


TOOL_REFERENCE_MARKERS = ["tools", "tool_call", "tool_calls", "function_call"]


def proxy_supports_tool_calling(template_src: str) -> bool:
    return any(marker in template_src for marker in TOOL_REFERENCE_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--schema-mode", required=True, choices=["compact-tools", "full-schema"])
    ap.add_argument("--leg-id", required=True)
    ap.add_argument("--approach-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    template = load_chat_template(args.model)
    if template is None:
        result = {
            "leg_id": args.leg_id,
            "approach_id": args.approach_id,
            "outcome": "inconclusive",
            "detail": f"{args.model} has no chat_template in tokenizer_config.json — "
            "native tool-calling categorically not supported (no template to detect "
            "capability from).",
        }
    else:
        proxy_positive = proxy_supports_tool_calling(template)
        result = {
            "leg_id": args.leg_id,
            "approach_id": args.approach_id,
            "outcome": "proxy_positive" if proxy_positive else "proxy_negative",
            "detail": (
                f"schema_mode={args.schema_mode}. PROXY check only (see script "
                "docstring) — the real llama.cpp capability-detection dry-run this "
                "would need to faithfully answer Q5 is not run here. Raw template "
                f"{'DOES' if proxy_positive else 'does NOT'} reference a tool-calling "
                "construct in its source."
            ),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
