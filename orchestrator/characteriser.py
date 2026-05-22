"""
Problem Characteriser
---------------------
A lightweight meta-layer that inspects an incoming task and produces
a TaskProfile — a set of signals used by the Router to decide where
to start in the tier hierarchy.

Deliberately cheap: no neural network inference here.
Uses structural heuristics, keyword signals, and optional schema hints.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskProfile:
    """
    Lightweight description of a task's structural properties.
    Used by the Router to pick an entry tier.
    """
    task_type: str                    # e.g. "classification", "lookup", "generation"
    is_structured: bool = False       # input has clear schema / tabular form
    has_pattern: bool = False         # answer likely extractable by regex
    has_lookup_key: bool = False      # answer likely in a mapping table
    needs_knowledge: bool = False     # requires external knowledge retrieval
    needs_reasoning: bool = False     # multi-step logical reasoning required
    needs_generation: bool = False    # free-form text generation required
    input_length: int = 0
    suggested_min_tier: int = 0       # lowest tier worth trying
    signals: dict = field(default_factory=dict)


# Patterns that strongly suggest tier-0 solvability
_REGEX_PATTERNS = {
    "email":    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "url":      r'https?://[^\s]+',
    "date":     r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    "phone":    r'\b[\+]?[\d\s\-\(\)]{7,15}\b',
    "number":   r'\b\d+\.?\d*\b',
    "postcode": r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b',
}

# Keywords suggesting the task needs reasoning / generation
_REASONING_KEYWORDS = [
    "explain", "why", "how does", "compare", "analyse", "analyze",
    "summarise", "summarize", "write", "generate", "create", "draft",
    "what would happen", "what if", "reason", "argue", "debate",
]

_LOOKUP_KEYWORDS = [
    "what is the capital", "convert", "translate to", "what does",
    "definition of", "synonym", "antonym", "currency", "country code",
    "what year", "who invented", "what is the formula",
]

_KNOWLEDGE_KEYWORDS = [
    "according to", "based on", "in the document", "find in",
    "search for", "retrieve", "what does the policy say",
    "in the paper", "from the database",
]


class Characteriser:
    """
    Inspects a task dict and returns a TaskProfile.

    Task dict schema (flexible, not all fields required):
    {
        "input":      str,            # the query or input text
        "task_type":  str (optional), # hint from caller
        "schema":     dict (optional),# if input is structured
        "context":    str (optional), # additional context
    }
    """

    def characterise(self, task: dict) -> TaskProfile:
        text = str(task.get("input", "")).strip().lower()
        task_type_hint = task.get("task_type", "unknown")
        schema = task.get("schema", None)

        profile = TaskProfile(
            task_type=task_type_hint,
            input_length=len(text),
        )

        # --- Structural signals ---
        if schema or self._looks_tabular(task):
            profile.is_structured = True
            profile.signals["structured"] = True

        # --- Regex / pattern signals ---
        for name, pat in _REGEX_PATTERNS.items():
            if re.search(pat, task.get("input", ""), re.IGNORECASE):
                profile.has_pattern = True
                profile.signals[f"pattern_{name}"] = True

        # Explicit pattern task types
        if task_type_hint in ("regex", "pattern_match", "extraction",
                               "validation", "format_check"):
            profile.has_pattern = True

        # --- Lookup signals ---
        if any(kw in text for kw in _LOOKUP_KEYWORDS):
            profile.has_lookup_key = True
            profile.signals["lookup_hint"] = True

        if task_type_hint in ("lookup", "mapping", "translation",
                               "conversion", "faq",
                               "country_code", "currency_code",
                               "http_status", "si_unit"):
            profile.has_lookup_key = True

        # --- Knowledge / RAG signals ---
        if any(kw in text for kw in _KNOWLEDGE_KEYWORDS):
            profile.needs_knowledge = True
            profile.signals["knowledge_hint"] = True

        if task_type_hint in ("rag", "document_qa", "knowledge_retrieval"):
            profile.needs_knowledge = True

        # --- Reasoning / generation signals ---
        if any(kw in text for kw in _REASONING_KEYWORDS):
            profile.needs_reasoning = True
            profile.signals["reasoning_hint"] = True

        if task_type_hint in ("generation", "summarisation", "reasoning",
                               "creative", "multi_step"):
            profile.needs_generation = True
            profile.needs_reasoning = True

        # --- Derive suggested minimum tier ---
        profile.suggested_min_tier = self._derive_min_tier(profile, task_type_hint)

        return profile

    def _looks_tabular(self, task: dict) -> bool:
        """Heuristic: does the input look like CSV or key-value pairs?"""
        text = str(task.get("input", ""))
        comma_lines = [l for l in text.split("\n") if l.count(",") >= 2]
        return len(comma_lines) >= 2

    def _derive_min_tier(self, profile: TaskProfile, hint: str) -> int:
        if hint.startswith("validate_") or hint.startswith("extract_"):
            return 0
        if profile.has_pattern and not profile.needs_reasoning:
            return 0
        if profile.has_lookup_key and not profile.needs_reasoning:
            return 1
        if profile.is_structured and not profile.needs_generation:
            return 2
        if profile.needs_knowledge and not profile.needs_reasoning:
            return 4
        if profile.needs_reasoning and not profile.needs_generation:
            return 5
        if profile.needs_generation:
            return 5
        return 2
