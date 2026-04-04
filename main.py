"""
Multi-Agent Course Finder v2 — NCU CSIE
Midterm Project
"""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

from agents.OrchestratorAgent import CourseFinderOrchestrator
from config.main import GROQ_DEFAULT_MODEL

load_dotenv()

try:
    import chromadb
except ImportError:
    sys.exit("pip install chromadb")

SUPPORTED_PROVIDERS = {
    "groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "default": GROQ_DEFAULT_MODEL,
    },
    "gemini": {
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "default": "gemini-2.5-flash",
    },
}


def get_default_model(provider: str) -> str:
    return SUPPORTED_PROVIDERS[provider]["default"]


def list_models(provider: str) -> list[str]:
    return SUPPORTED_PROVIDERS[provider]["models"]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Course Finder v2 (Groq / Gemini powered)"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Free-form student query. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="groq",
        choices=list(SUPPORTED_PROVIDERS.keys()),
        help="LLM provider to use: groq or gemini (default: groq)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Model name to use. Defaults to provider's default model.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models for the selected provider and exit.",
    )
    args = parser.parse_args()

    # ── list models and exit ──────────────────────────────────────────
    if args.list_models:
        print(f"\nAvailable models for provider '{args.provider}':")
        for m in list_models(args.provider):
            default_marker = " (default)" if m == get_default_model(args.provider) else ""
            print(f"  • {m}{default_marker}")
        return

    # ── resolve model ─────────────────────────────────────────────────
    model = args.model or get_default_model(args.provider)

    if model not in list_models(args.provider):
        print(
            f"⚠️  Warning: '{model}' is not in the known model list for '{args.provider}'. "
            f"Proceeding anyway."
        )

    print(f"Provider : {args.provider}")
    print(f"Model    : {model}")

    # ── build orchestrator ────────────────────────────────────────────
    orchestrator = CourseFinderOrchestrator(model=model, provider=args.provider)

    # ── single-shot mode ──────────────────────────────────────────────
    if args.query:
        output, _ = orchestrator.run(args.query)
        print(output)
        return

    # ── REPL mode ─────────────────────────────────────────────────────
    print("\n╔═══════════════════════════════════════════════════╗")
    print(f"║  NCU Course Finder v2  [{args.provider.upper():^6}] REPL mode      ║")
    print("║  Type 'exit' to quit  |  'model' to show config  ║")
    print("╚═══════════════════════════════════════════════════╝\n")

    profile = None
    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if raw.lower() == "model":
            print(f"  Provider : {args.provider}")
            print(f"  Model    : {model}\n")
            continue

        output, new_profile = orchestrator.run(raw, profile=profile)
        if new_profile is not None:
            profile = new_profile
        print(output)


if __name__ == "__main__":
    main()