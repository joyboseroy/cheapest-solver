"""
Tier 1: Lookup / Mapping Table Solver
--------------------------------------
Handles tasks solvable by exact or fuzzy dictionary lookup.
Near-zero cost. No ML inference.

Examples:
- Currency codes / country codes / language codes
- FAQ answers
- Unit conversion tables
- Simple synonym / definition lookups
- Code → description mappings
"""

from difflib import get_close_matches
from orchestrator.base import BaseSolver, SolverResult, Tier, TIER_ENERGY_MJ, TIER_COST_USD


# ── Built-in lookup tables ────────────────────────────────────────────────────

COUNTRY_CODES = {
    "india": "IN", "united states": "US", "united kingdom": "GB",
    "germany": "DE", "france": "FR", "japan": "JP", "china": "CN",
    "brazil": "BR", "canada": "CA", "australia": "AU", "russia": "RU",
    "south korea": "KR", "italy": "IT", "spain": "ES", "mexico": "MX",
}

CURRENCY_CODES = {
    "indian rupee": "INR", "us dollar": "USD", "euro": "EUR",
    "british pound": "GBP", "japanese yen": "JPY", "chinese yuan": "CNY",
    "australian dollar": "AUD", "canadian dollar": "CAD",
    "swiss franc": "CHF", "swedish krona": "SEK",
}

HTTP_STATUS_CODES = {
    "200": "OK", "201": "Created", "204": "No Content",
    "301": "Moved Permanently", "302": "Found",
    "400": "Bad Request", "401": "Unauthorized", "403": "Forbidden",
    "404": "Not Found", "405": "Method Not Allowed",
    "500": "Internal Server Error", "502": "Bad Gateway",
    "503": "Service Unavailable", "504": "Gateway Timeout",
}

SI_UNITS = {
    "kilometer": 1000, "metre": 1, "centimetre": 0.01, "millimetre": 0.001,
    "kilogram": 1000, "gram": 1, "milligram": 0.001,
    "hour": 3600, "minute": 60, "second": 1,
}

# Registry maps task_type → lookup table
_DEFAULT_TABLES = {
    "country_code":    COUNTRY_CODES,
    "currency_code":   CURRENCY_CODES,
    "http_status":     HTTP_STATUS_CODES,
    "si_unit":         SI_UNITS,
}


class LookupSolver(BaseSolver):
    """
    Tier 1 solver. Handles exact and fuzzy dictionary lookups.

    Task dict:
    {
        "input":        str,              # the key to look up
        "task_type":    str,              # must match a registered table name
        "table":        dict (optional),  # custom lookup table
        "fuzzy":        bool (optional),  # use fuzzy matching (default True)
        "cutoff":       float (optional), # fuzzy match cutoff (default 0.6)
    }
    """

    tier = Tier.LOOKUP

    def __init__(self, custom_tables: dict = None):
        self.tables = dict(_DEFAULT_TABLES)
        if custom_tables:
            self.tables.update(custom_tables)

    def register_table(self, name: str, table: dict):
        self.tables[name] = table

    def can_attempt(self, task: dict) -> bool:
        has_inline_table = "table" in task
        task_type = task.get("task_type", "")
        return has_inline_table or task_type in self.tables

    def solve(self, task: dict) -> SolverResult:
        key = str(task.get("input", "")).strip().lower()
        task_type = task.get("task_type", "")
        fuzzy = task.get("fuzzy", True)
        cutoff = task.get("cutoff", 0.6)

        # Determine which table to use
        table = task.get("table") or self.tables.get(task_type, {})

        if not table:
            return SolverResult(answer=None, confidence=0.0, tier=self.tier)

        # Exact match first
        if key in table:
            return SolverResult(
                answer=table[key],
                confidence=1.0,
                tier=self.tier,
                energy_mj=TIER_ENERGY_MJ[self.tier],
                cost_usd=TIER_COST_USD[self.tier],
                metadata={"match_type": "exact", "key": key},
            )

        # Fuzzy match
        if fuzzy:
            matches = get_close_matches(key, table.keys(), n=1, cutoff=cutoff)
            if matches:
                matched_key = matches[0]
                # Confidence proportional to string similarity
                from difflib import SequenceMatcher
                sim = SequenceMatcher(None, key, matched_key).ratio()
                return SolverResult(
                    answer=table[matched_key],
                    confidence=round(sim * 0.9, 3),  # cap at 0.9 for fuzzy
                    tier=self.tier,
                    energy_mj=TIER_ENERGY_MJ[self.tier],
                    cost_usd=TIER_COST_USD[self.tier],
                    metadata={"match_type": "fuzzy", "key": matched_key, "similarity": sim},
                )

        return SolverResult(answer=None, confidence=0.0, tier=self.tier)
