"""
Base interface for all solvers in the CheapestSolver framework.
Every solver tier implements this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class Tier(IntEnum):
    REGEX = 0
    LOOKUP = 1
    CLASSICAL_ML = 2
    FINETUNED = 3
    RAG = 4
    SLM = 5
    LLM = 6


TIER_LABELS = {
    Tier.REGEX:        "Regex / Pattern Match",
    Tier.LOOKUP:       "Lookup / Mapping Table",
    Tier.CLASSICAL_ML: "Classical ML (XGBoost / sklearn)",
    Tier.FINETUNED:    "Fine-tuned Small Model (BERT-class)",
    Tier.RAG:          "RAG (Retrieval-Augmented Generation)",
    Tier.SLM:          "Small Language Model (Phi-4 / Qwen-2.5-7B)",
    Tier.LLM:          "Large Language Model (GPT-4 class)",
}

# Approximate energy cost per query in millijoules (illustrative, not measured)
TIER_ENERGY_MJ = {
    Tier.REGEX:        0.0,
    Tier.LOOKUP:       0.0,
    Tier.CLASSICAL_ML: 0.5,
    Tier.FINETUNED:    5.0,
    Tier.RAG:          20.0,
    Tier.SLM:          200.0,
    Tier.LLM:          2000.0,
}

# Approximate cost per query in USD (illustrative)
TIER_COST_USD = {
    Tier.REGEX:        0.0,
    Tier.LOOKUP:       0.0,
    Tier.CLASSICAL_ML: 0.000001,
    Tier.FINETUNED:    0.00001,
    Tier.RAG:          0.0001,
    Tier.SLM:          0.0005,
    Tier.LLM:          0.01,
}


@dataclass
class SolverResult:
    """Standardised output from any solver tier."""
    answer: Any
    confidence: float          # 0.0 – 1.0
    tier: Tier
    energy_mj: float = 0.0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.answer is not None

    def __repr__(self):
        return (
            f"SolverResult(tier={TIER_LABELS[self.tier]}, "
            f"confidence={self.confidence:.2f}, "
            f"energy={self.energy_mj:.2f}mJ, "
            f"cost=${self.cost_usd:.6f})"
        )


class BaseSolver(ABC):
    """Abstract base class for all solver tiers."""

    tier: Tier

    @abstractmethod
    def can_attempt(self, task: dict) -> bool:
        """
        Quick check: can this solver even attempt this task type?
        Should be very cheap — no actual solving.
        """
        ...

    @abstractmethod
    def solve(self, task: dict) -> SolverResult:
        """
        Attempt to solve the task.
        Returns a SolverResult with confidence score.
        If the solver cannot produce an answer, return confidence=0.0.
        """
        ...
