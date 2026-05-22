"""
Router
------
The heart of CheapestSolver.

Traverses solver tiers from cheapest to most expensive.
Stops as soon as a solver returns confidence >= threshold.
Records the full trace for benchmarking and analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from orchestrator.base import BaseSolver, SolverResult, Tier, TIER_LABELS
from orchestrator.characteriser import Characteriser, TaskProfile


@dataclass
class RoutingTrace:
    """Full record of a routing decision — used for benchmarking."""
    task: dict
    profile: TaskProfile
    attempts: list[SolverResult] = field(default_factory=list)
    final_result: Optional[SolverResult] = None
    total_energy_mj: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    winning_tier: Optional[Tier] = None
    tiers_tried: int = 0

    def summary(self) -> str:
        if not self.final_result:
            return "No solver could answer this task."
        lines = [
            f"Answer:      {self.final_result.answer}",
            f"Solved by:   {TIER_LABELS[self.winning_tier]}",
            f"Confidence:  {self.final_result.confidence:.2f}",
            f"Tiers tried: {self.tiers_tried}",
            f"Energy:      {self.total_energy_mj:.3f} mJ",
            f"Cost:        ${self.total_cost_usd:.6f}",
            f"Latency:     {self.total_latency_ms:.1f} ms",
        ]
        return "\n".join(lines)


class Router:
    """
    Given a list of solvers (one per tier, in ascending order),
    routes each task to the cheapest solver that meets the
    confidence threshold.
    """

    def __init__(
        self,
        solvers: list[BaseSolver],
        confidence_threshold: float = 0.80,
        max_tier: Tier = Tier.LLM,
        verbose: bool = False,
    ):
        # Sort solvers by tier to guarantee cheapest-first traversal
        self.solvers = sorted(solvers, key=lambda s: s.tier)
        self.threshold = confidence_threshold
        self.max_tier = max_tier
        self.characteriser = Characteriser()
        self.verbose = verbose

    def route(self, task: dict) -> RoutingTrace:
        """
        Main entry point. Returns a RoutingTrace containing
        the final answer and the full decision history.
        """
        profile = self.characteriser.characterise(task)
        trace = RoutingTrace(task=task, profile=profile)

        if self.verbose:
            print(f"\n[Router] Task type: {profile.task_type}")
            print(f"[Router] Suggested min tier: {profile.suggested_min_tier}")
            print(f"[Router] Signals: {profile.signals}")

        for solver in self.solvers:
            # Skip tiers below suggested minimum (characteriser hint)
            if solver.tier < profile.suggested_min_tier:
                continue

            # Stop if we've exceeded the max tier limit
            if solver.tier > self.max_tier:
                break

            # Quick capability check — no inference cost
            if not solver.can_attempt(task):
                if self.verbose:
                    print(f"[Router] Tier {solver.tier} ({TIER_LABELS[solver.tier]}): skipped (can_attempt=False)")
                continue

            # Attempt to solve
            t0 = time.perf_counter()
            result = solver.solve(task)
            result.latency_ms = (time.perf_counter() - t0) * 1000

            trace.attempts.append(result)
            trace.tiers_tried += 1
            trace.total_energy_mj += result.energy_mj
            trace.total_cost_usd += result.cost_usd
            trace.total_latency_ms += result.latency_ms

            if self.verbose:
                print(
                    f"[Router] Tier {solver.tier} ({TIER_LABELS[solver.tier]}): "
                    f"confidence={result.confidence:.2f}, "
                    f"answer={str(result.answer)[:60]}"
                )

            if result.confidence >= self.threshold:
                trace.final_result = result
                trace.winning_tier = solver.tier
                if self.verbose:
                    print(f"[Router] Solved at tier {solver.tier}. Stopping.")
                return trace

        # If no solver met threshold, return best available
        if trace.attempts:
            best = max(trace.attempts, key=lambda r: r.confidence)
            trace.final_result = best
            trace.winning_tier = best.tier
            if self.verbose:
                print(f"[Router] No solver met threshold. Best: tier {best.tier}, confidence={best.confidence:.2f}")

        return trace
