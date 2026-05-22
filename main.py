"""
CheapestSolver — Main Entry Point
-----------------------------------
Wires up all solvers, configures the router, and runs the benchmark.

Usage:
    python main.py                   # run full benchmark
    python main.py --verbose         # per-task output
    python main.py --tier-max 2      # only use tiers 0-2
    python main.py --threshold 0.75  # lower confidence threshold
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator.router import Router
from orchestrator.base import Tier
from solvers.tier0_regex import RegexSolver
from solvers.tier1_lookup import LookupSolver
from solvers.tier2_classical_ml import ClassicalMLSolver
from benchmark.evaluate import run_benchmark, print_report, save_results


def build_router(
    confidence_threshold: float = 0.80,
    max_tier: int = 6,
    verbose: bool = False,
) -> Router:
    """Instantiate all available solvers and wire them into a Router."""
    solvers = [
        RegexSolver(),
        LookupSolver(),
        ClassicalMLSolver(),
        # Fine-tuned (Tier 3): placeholder — add your BERT-class model here
        # RAG (Tier 4): placeholder — add your RAG pipeline here
        # SLM (Tier 5): placeholder — Ollama / HuggingFace local model
        # LLM (Tier 6): placeholder — OpenAI / Anthropic API
    ]

    return Router(
        solvers=solvers,
        confidence_threshold=confidence_threshold,
        max_tier=Tier(max_tier),
        verbose=verbose,
    )


def demo_single_task():
    """Quick demo: route a few example tasks and print traces."""
    router = build_router(verbose=True)

    demo_tasks = [
        {
            "input": "alice@wonderland.org",
            "task_type": "validate_email",
        },
        {
            "input": "germany",
            "task_type": "country_code",
        },
        {
            "input": "The product is absolutely fantastic, best I have ever used!",
            "task_type": "sentiment",
        },
        {
            "input": "Click here to earn free money fast, guaranteed no risk!",
            "task_type": "spam_detect",
        },
    ]

    print("\n" + "=" * 60)
    print("  CHEAPESTSOLVER — DEMO")
    print("=" * 60)

    for task in demo_tasks:
        trace = router.route(task)
        print(f"\nInput:  {task['input']}")
        print(trace.summary())


def main():
    parser = argparse.ArgumentParser(description="CheapestSolver benchmark runner")
    parser.add_argument("--demo", action="store_true", help="Run quick demo instead of full benchmark")
    parser.add_argument("--verbose", action="store_true", help="Per-task output")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold (default 0.80)")
    parser.add_argument("--tier-max", type=int, default=6, help="Max tier to use (default 6)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    if args.demo:
        demo_single_task()
        return

    print(f"\nRunning CheapestSolver benchmark...")
    print(f"  Confidence threshold: {args.threshold}")
    print(f"  Max tier: {args.tier_max} ({['Regex','Lookup','Classical ML','Fine-tuned','RAG','SLM','LLM'][args.tier_max]})")

    router = build_router(
        confidence_threshold=args.threshold,
        max_tier=args.tier_max,
        verbose=args.verbose,
    )

    summary = run_benchmark(router, verbose=args.verbose)
    print_report(summary)

    if args.save:
        save_results(summary)


if __name__ == "__main__":
    main()
