"""
Tier 5: Small Language Model Solver
Uses Ollama local inference — no API key, no cloud cost.
"""

import requests
from orchestrator.base import BaseSolver, SolverResult, Tier, TIER_ENERGY_MJ, TIER_COST_USD

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"

SUPPORTED = {"reasoning", "generation", "multi_step", "arithmetic"}


class SLMSolver(BaseSolver):
    tier = Tier.SLM

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def can_attempt(self, task: dict) -> bool:
        return task.get("task_type", "") in SUPPORTED

    def solve(self, task: dict) -> SolverResult:
        prompt = str(task.get("input", ""))
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }, timeout=60)
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
            return SolverResult(
                answer=answer,
                confidence=0.82,
                tier=self.tier,
                energy_mj=TIER_ENERGY_MJ[self.tier],
                cost_usd=TIER_COST_USD[self.tier],
            )
        except Exception as e:
            return SolverResult(answer=None, confidence=0.0, tier=self.tier,
                                metadata={"error": str(e)})
