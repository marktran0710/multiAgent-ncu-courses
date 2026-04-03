"""
Multi-Agent Course Finder v2 — NCU CSIE
Midterm Project

NEW in v2:
  • IntakeAgent   — uses Groq function calling to extract a structured
                    UserProfile (year, completed, goals, constraints) from
                    free-form conversation before any retrieval happens
  • JudgeAgent    — after RRF fusion, calls Groq with the top-N candidates
                    and the full UserProfile to reason and pick the single
                    best course with an explanation

Full pipeline:
  IntakeAgent  → structured UserProfile via function call
      │
      ▼
  [BM25Agent ‖ VectorAgent]  (parallel retrieval, query built from profile)
      │
      ▼
  FusionAgent  → RRF-fused ranked list
      │
      ▼
  JudgeAgent   → single best pick + reasoning  (LLM function call)
      │
      ▼
  ResponseAgent → final formatted output

Usage:
  python main.py                      # interactive REPL
  python main.py -q "some free text"  # single-shot CLI
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
    from chromadb.config import Settings
except ImportError:
    sys.exit("pip install chromadb")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Course Finder v2 (Groq-powered IntakeAgent + JudgeAgent)"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Free-form student query. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=GROQ_DEFAULT_MODEL,
        help=f"Groq model to use (default: {GROQ_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print(f"Using Groq model: {args.model}")
    orchestrator = CourseFinderOrchestrator(model=args.model)

    if args.query:
        print(orchestrator.run(args.query))
    else:
        print("\n╔═══════════════════════════════════════════════════╗")
        print("║  NCU Multi-Agent Course Finder v2  (REPL mode)   ║")
        print("║  Type 'exit' to quit                              ║")
        print("╚═══════════════════════════════════════════════════╝\n")
        # In main(), replace the while-loop body:
        profile = None
        while True:
            try:
                raw = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not raw or raw.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            output, new_profile = orchestrator.run(raw, profile=profile)
            if new_profile is not None:          # only update on valid turns
                profile = new_profile
            print(output)


if __name__ == "__main__":
    main()