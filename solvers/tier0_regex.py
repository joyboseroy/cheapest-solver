"""
Tier 0: Regex / Pattern Match Solver
-------------------------------------
Handles tasks solvable by deterministic pattern matching.
Zero energy cost. Zero API cost. Instant.

Examples:
- Email validation
- Date / phone / URL extraction
- Format checking
- Simple string transformation
"""

import re
from orchestrator.base import BaseSolver, SolverResult, Tier, TIER_ENERGY_MJ, TIER_COST_USD


# Built-in pattern registry
# Each entry: name -> (pattern, extractor_fn or None)
_BUILTIN_PATTERNS = {
    "validate_email": (
        r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',
        lambda m: "valid" if m else "invalid",
    ),
    "extract_email": (
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        lambda m: m.group(0) if m else None,
    ),
    "extract_url": (
        r'https?://[^\s]+',
        lambda m: m.group(0) if m else None,
    ),
    "extract_date": (
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        lambda m: m.group(1) if m else None,
    ),
    "extract_phone": (
        r'\b[\+]?[\d\s\-\(\)]{7,15}\b',
        lambda m: m.group(0).strip() if m else None,
    ),
    "extract_number": (
        r'\b\d+\.?\d*\b',
        lambda m: m.group(0) if m else None,
    ),
    "validate_postcode_uk": (
        r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$',
        lambda m: "valid" if m else "invalid",
    ),
    "extract_ip": (
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        lambda m: m.group(0) if m else None,
    ),
}


class RegexSolver(BaseSolver):
    """
    Tier 0 solver. Handles pattern-matching tasks.

    Supports:
    - Built-in named patterns (see _BUILTIN_PATTERNS above)
    - Custom patterns passed in the task dict
    - Direct regex task_type tasks

    Task dict:
    {
        "input":        str,              # text to match against
        "task_type":    str,              # e.g. "validate_email", "extract_date"
        "pattern":      str (optional),   # custom regex
        "flags":        int (optional),   # re flags
    }
    """

    tier = Tier.REGEX

    def __init__(self, custom_patterns: dict = None):
        self.patterns = dict(_BUILTIN_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def can_attempt(self, task: dict) -> bool:
        task_type = task.get("task_type", "")
        has_custom_pattern = "pattern" in task
        return task_type in self.patterns or has_custom_pattern

    def solve(self, task: dict) -> SolverResult:
        text = str(task.get("input", ""))
        task_type = task.get("task_type", "")
        flags = task.get("flags", re.IGNORECASE)

        # Custom pattern takes precedence
        if "pattern" in task:
            pattern = task["pattern"]
            m = re.search(pattern, text, flags)
            answer = m.group(0) if m else None
            confidence = 1.0 if answer is not None else 0.0
        elif task_type in self.patterns:
            pattern, extractor = self.patterns[task_type]
            m = re.search(pattern, text, flags)
            answer = extractor(m)
            # For validation tasks, a non-match is still a valid answer ("invalid")
            is_validation = task_type.startswith("validate_")
            if is_validation:
                confidence = 1.0  # we always know the answer for validation
            else:
                confidence = 1.0 if answer is not None else 0.0
        else:
            return SolverResult(
                answer=None,
                confidence=0.0,
                tier=self.tier,
                energy_mj=0.0,
                cost_usd=0.0,
            )

        return SolverResult(
            answer=answer,
            confidence=confidence,
            tier=self.tier,
            energy_mj=TIER_ENERGY_MJ[self.tier],
            cost_usd=TIER_COST_USD[self.tier],
            metadata={"pattern_used": task.get("pattern", task_type)},
        )
