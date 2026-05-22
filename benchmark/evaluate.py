"""
Benchmark Evaluator
--------------------
Runs the full benchmark task set through the CheapestSolver router
and produces a detailed report:

- Per-task accuracy
- Tier resolution distribution
- Energy and cost savings vs always-LLM baseline
- Pareto frontier: quality vs energy
"""

import json
import time
from collections import defaultdict
from typing import Optional

from orchestrator.base import Tier, TIER_LABELS, TIER_ENERGY_MJ, TIER_COST_USD
from orchestrator.router import Router, RoutingTrace
from benchmark.tasks.benchmark_tasks import BENCHMARK_TASKS


def _answers_match(predicted, ground_truth) -> bool:
    """Flexible answer matching: exact, case-insensitive, or substring."""
    if ground_truth is None:
        return True  # open-ended tasks: not evaluated here
    if predicted is None:
        return False
    p = str(predicted).strip().lower()
    g = str(ground_truth).strip().lower()
    return p == g or g in p or p in g


def run_benchmark(
    router: Router,
    tasks: list = None,
    verbose: bool = False,
) -> dict:
    """
    Run the benchmark and return a results dict.

    Args:
        router:  A configured Router instance with solvers loaded
        tasks:   Task list (defaults to full BENCHMARK_TASKS)
        verbose: Print per-task trace

    Returns:
        results dict suitable for reporting and JSON export
    """
    if tasks is None:
        tasks = BENCHMARK_TASKS

    results = []
    tier_counts = defaultdict(int)
    tier_correct = defaultdict(int)
    category_counts = defaultdict(int)
    category_correct = defaultdict(int)

    total_energy = 0.0
    total_cost = 0.0
    baseline_energy = 0.0   # if every task went to LLM
    baseline_cost = 0.0

    t_start = time.perf_counter()

    for task in tasks:
        trace: RoutingTrace = router.route(task)

        ground_truth = task.get("ground_truth")
        predicted = trace.final_result.answer if trace.final_result else None
        correct = _answers_match(predicted, ground_truth)
        expected_tier = task.get("expected_tier", -1)
        winning_tier = trace.winning_tier if trace.winning_tier is not None else -1
        category = task.get("category", "unknown")

        tier_counts[winning_tier] += 1
        category_counts[category] += 1
        if correct:
            tier_correct[winning_tier] += 1
            category_correct[category] += 1

        total_energy += trace.total_energy_mj
        total_cost += trace.total_cost_usd
        baseline_energy += TIER_ENERGY_MJ[Tier.LLM]
        baseline_cost += TIER_COST_USD[Tier.LLM]

        result = {
            "id": task.get("id", "?"),
            "category": category,
            "description": task.get("description", ""),
            "input_snippet": str(task.get("input", ""))[:60],
            "ground_truth": str(ground_truth),
            "predicted": str(predicted),
            "correct": correct,
            "expected_tier": expected_tier,
            "winning_tier": winning_tier,
            "winning_tier_label": TIER_LABELS.get(winning_tier, "unknown"),
            "tiers_tried": trace.tiers_tried,
            "confidence": trace.final_result.confidence if trace.final_result else 0.0,
            "energy_mj": trace.total_energy_mj,
            "cost_usd": trace.total_cost_usd,
            "latency_ms": trace.total_latency_ms,
        }

        if verbose:
            status = "✓" if correct else "✗"
            print(f"[{status}] {task['id']} | tier={TIER_LABELS.get(winning_tier,'?')[:25]} | "
                  f"conf={result['confidence']:.2f} | {task.get('description','')}")

        results.append(result)

    elapsed = (time.perf_counter() - t_start) * 1000

    # ── Aggregate stats ───────────────────────────────────────────────────────
    n_tasks = len(tasks)
    n_correct = sum(1 for r in results if r["correct"])
    n_open_ended = sum(1 for t in tasks if t.get("ground_truth") is None)
    n_evaluable = n_tasks - n_open_ended
    accuracy = n_correct / n_evaluable if n_evaluable > 0 else 0.0

    energy_saving_pct = (1 - total_energy / baseline_energy) * 100 if baseline_energy > 0 else 0
    cost_saving_pct = (1 - total_cost / baseline_cost) * 100 if baseline_cost > 0 else 0

    tier_distribution = {
        TIER_LABELS.get(tier, f"tier_{tier}"): {
            "count": count,
            "correct": tier_correct.get(tier, 0),
            "pct_of_tasks": round(count / n_tasks * 100, 1),
        }
        for tier, count in sorted(tier_counts.items())
    }

    summary = {
        "n_tasks": n_tasks,
        "n_evaluable": n_evaluable,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "total_energy_mj": round(total_energy, 3),
        "baseline_energy_mj": round(baseline_energy, 3),
        "energy_saving_pct": round(energy_saving_pct, 1),
        "total_cost_usd": round(total_cost, 6),
        "baseline_cost_usd": round(baseline_cost, 6),
        "cost_saving_pct": round(cost_saving_pct, 1),
        "elapsed_ms": round(elapsed, 1),
        "tier_distribution": tier_distribution,
        "category_accuracy": {
            cat: round(category_correct[cat] / category_counts[cat], 3)
            for cat in category_counts
        },
        "tasks": results,
    }

    return summary


def print_report(summary: dict):
    """Pretty-print a benchmark summary."""
    print("\n" + "=" * 65)
    print("  CHEAPESTSOLVER BENCHMARK REPORT")
    print("=" * 65)
    print(f"  Tasks:        {summary['n_tasks']} total, {summary['n_evaluable']} evaluable")
    print(f"  Accuracy:     {summary['accuracy'] * 100:.1f}%")
    print(f"  Energy:       {summary['total_energy_mj']:.2f} mJ  "
          f"(baseline {summary['baseline_energy_mj']:.2f} mJ, "
          f"saving {summary['energy_saving_pct']:.1f}%)")
    print(f"  Cost:         ${summary['total_cost_usd']:.6f}  "
          f"(baseline ${summary['baseline_cost_usd']:.6f}, "
          f"saving {summary['cost_saving_pct']:.1f}%)")
    print(f"  Latency:      {summary['elapsed_ms']:.0f} ms total")

    print("\n  TIER DISTRIBUTION")
    print("  " + "-" * 55)
    for tier_label, stats in summary["tier_distribution"].items():
        bar = "█" * int(stats["pct_of_tasks"] / 3)
        print(f"  {tier_label[:30]:30s} {stats['count']:3d} tasks ({stats['pct_of_tasks']:5.1f}%) {bar}")

    print("\n  ACCURACY BY CATEGORY")
    print("  " + "-" * 55)
    for cat, acc in summary["category_accuracy"].items():
        print(f"  {cat:20s}  {acc * 100:.1f}%")

    print("\n  FAILED TASKS")
    print("  " + "-" * 55)
    failed = [r for r in summary["tasks"] if not r["correct"] and r["ground_truth"] != "None"]
    if not failed:
        print("  All evaluable tasks answered correctly.")
    for r in failed:
        print(f"  [{r['id']}] {r['description']}")
        print(f"         Expected: {r['ground_truth']}  |  Got: {r['predicted']}")

    print("=" * 65)


def save_results(summary: dict, path: str = "benchmark/results/latest.json"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {path}")
