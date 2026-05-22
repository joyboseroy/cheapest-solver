"""
CheapestSolver Custom Benchmark
---------------------------------
A diverse task set designed to exercise all solver tiers fairly.

Design principles:
1. Each task has a ground_truth answer for evaluation
2. Tasks are labelled with their expected_tier (the cheapest correct solver)
3. Tasks span the full spectrum: regex → lookup → ML → RAG → SLM → LLM
4. 10 task categories, ~10 instances each = ~100 tasks total

This benchmark intentionally avoids NLP-only benchmarks (GLUE, SuperGLUE)
which are biased toward neural models. The full spectrum includes tasks
where regex or a lookup table is objectively the best solution.

Categories:
  A. Format validation          (expected tier: 0 - regex)
  B. Information extraction     (expected tier: 0 - regex)
  C. Code/standard lookups      (expected tier: 1 - lookup)
  D. Unit / currency mapping    (expected tier: 1 - lookup)
  E. Sentiment analysis         (expected tier: 2 - classical ML)
  F. Intent classification      (expected tier: 2 - classical ML)
  G. Topic classification       (expected tier: 2 - classical ML)
  H. Spam detection             (expected tier: 2 - classical ML)
  I. Simple factual QA          (expected tier: 4 - RAG or lookup)
  J. Multi-step reasoning       (expected tier: 5/6 - SLM/LLM)
"""

BENCHMARK_TASKS = [

    # ── Category A: Format Validation (Tier 0) ────────────────────────────────
    {
        "id": "A01",
        "input": "user@example.com",
        "task_type": "validate_email",
        "ground_truth": "valid",
        "expected_tier": 0,
        "category": "format_validation",
        "description": "Valid email address",
    },
    {
        "id": "A02",
        "input": "not-an-email",
        "task_type": "validate_email",
        "ground_truth": "invalid",
        "expected_tier": 0,
        "category": "format_validation",
        "description": "Invalid email address",
    },
    {
        "id": "A03",
        "input": "joy.bose@ericsson.com",
        "task_type": "validate_email",
        "ground_truth": "valid",
        "expected_tier": 0,
        "category": "format_validation",
        "description": "Corporate email",
    },
    {
        "id": "A04",
        "input": "SW1A 1AA",
        "task_type": "validate_postcode_uk",
        "ground_truth": "valid",
        "expected_tier": 0,
        "category": "format_validation",
        "description": "Valid UK postcode",
    },
    {
        "id": "A05",
        "input": "99999",
        "task_type": "validate_postcode_uk",
        "ground_truth": "invalid",
        "expected_tier": 0,
        "category": "format_validation",
        "description": "Invalid UK postcode",
    },

    # ── Category B: Information Extraction (Tier 0) ───────────────────────────
    {
        "id": "B01",
        "input": "Please contact us at support@company.org for help.",
        "task_type": "extract_email",
        "ground_truth": "support@company.org",
        "expected_tier": 0,
        "category": "extraction",
        "description": "Extract email from sentence",
    },
    {
        "id": "B02",
        "input": "Visit us at https://www.example.com/products for more info.",
        "task_type": "extract_url",
        "ground_truth": "https://www.example.com/products",
        "expected_tier": 0,
        "category": "extraction",
        "description": "Extract URL from sentence",
    },
    {
        "id": "B03",
        "input": "The event is scheduled for 15/03/2026 in Bengaluru.",
        "task_type": "extract_date",
        "ground_truth": "15/03/2026",
        "expected_tier": 0,
        "category": "extraction",
        "description": "Extract date from sentence",
    },
    {
        "id": "B04",
        "input": "Server IP address is 192.168.1.100",
        "task_type": "extract_ip",
        "ground_truth": "192.168.1.100",
        "expected_tier": 0,
        "category": "extraction",
        "description": "Extract IP address",
    },
    {
        "id": "B05",
        "input": "Call us on +91 98765 43210 for support.",
        "task_type": "extract_phone",
        "ground_truth": "+91 98765 43210",
        "expected_tier": 0,
        "category": "extraction",
        "description": "Extract phone number",
    },

    # ── Category C: Code / Standard Lookups (Tier 1) ─────────────────────────
    {
        "id": "C01",
        "input": "india",
        "task_type": "country_code",
        "ground_truth": "IN",
        "expected_tier": 1,
        "category": "lookup",
        "description": "Country code for India",
    },
    {
        "id": "C02",
        "input": "germany",
        "task_type": "country_code",
        "ground_truth": "DE",
        "expected_tier": 1,
        "category": "lookup",
        "description": "Country code for Germany",
    },
    {
        "id": "C03",
        "input": "404",
        "task_type": "http_status",
        "ground_truth": "Not Found",
        "expected_tier": 1,
        "category": "lookup",
        "description": "HTTP 404 status meaning",
    },
    {
        "id": "C04",
        "input": "500",
        "task_type": "http_status",
        "ground_truth": "Internal Server Error",
        "expected_tier": 1,
        "category": "lookup",
        "description": "HTTP 500 status meaning",
    },
    {
        "id": "C05",
        "input": "200",
        "task_type": "http_status",
        "ground_truth": "OK",
        "expected_tier": 1,
        "category": "lookup",
        "description": "HTTP 200 status meaning",
    },
    {
        "id": "C06",
        "input": "japanese yen",
        "task_type": "currency_code",
        "ground_truth": "JPY",
        "expected_tier": 1,
        "category": "lookup",
        "description": "Currency code for Japanese yen",
    },
    {
        "id": "C07",
        "input": "indian rupee",
        "task_type": "currency_code",
        "ground_truth": "INR",
        "expected_tier": 1,
        "category": "lookup",
        "description": "Currency code for Indian rupee",
    },

    # ── Category E: Sentiment Analysis (Tier 2) ───────────────────────────────
    {
        "id": "E01",
        "input": "This product is absolutely amazing! I love it.",
        "task_type": "sentiment",
        "ground_truth": "positive",
        "expected_tier": 2,
        "category": "sentiment",
        "description": "Clear positive review",
    },
    {
        "id": "E02",
        "input": "Terrible experience. The software is broken and useless.",
        "task_type": "sentiment",
        "ground_truth": "negative",
        "expected_tier": 2,
        "category": "sentiment",
        "description": "Clear negative review",
    },
    {
        "id": "E03",
        "input": "The product arrived on time.",
        "task_type": "sentiment",
        "ground_truth": "neutral",
        "expected_tier": 2,
        "category": "sentiment",
        "description": "Neutral statement",
    },
    {
        "id": "E04",
        "input": "Great features but the performance is poor and frustrating.",
        "task_type": "sentiment",
        "ground_truth": "mixed",
        "expected_tier": 2,
        "category": "sentiment",
        "description": "Mixed sentiment",
    },
    {
        "id": "E05",
        "input": "Best purchase I've made this year. Outstanding quality!",
        "task_type": "sentiment",
        "ground_truth": "positive",
        "expected_tier": 2,
        "category": "sentiment",
        "description": "Strongly positive",
    },

    # ── Category F: Intent Classification (Tier 2) ────────────────────────────
    {
        "id": "F01",
        "input": "Hello, how are you doing today?",
        "task_type": "intent_classify",
        "ground_truth": "greeting",
        "expected_tier": 2,
        "category": "intent",
        "description": "Greeting intent",
    },
    {
        "id": "F02",
        "input": "I want to cancel my subscription and get a refund.",
        "task_type": "intent_classify",
        "ground_truth": "cancel",
        "expected_tier": 2,
        "category": "intent",
        "description": "Cancellation intent",
    },
    {
        "id": "F03",
        "input": "Thank you so much for your help, I really appreciate it.",
        "task_type": "intent_classify",
        "ground_truth": "thanks",
        "expected_tier": 2,
        "category": "intent",
        "description": "Thanks intent",
    },
    {
        "id": "F04",
        "input": "The app keeps crashing and nothing is working properly.",
        "task_type": "intent_classify",
        "ground_truth": "complaint",
        "expected_tier": 2,
        "category": "intent",
        "description": "Complaint intent",
    },
    {
        "id": "F05",
        "input": "What is the price for the enterprise plan?",
        "task_type": "intent_classify",
        "ground_truth": "question",
        "expected_tier": 2,
        "category": "intent",
        "description": "Question intent",
    },

    # ── Category G: Topic Classification (Tier 2) ─────────────────────────────
    {
        "id": "G01",
        "input": "The neural network model achieved state of the art results on the benchmark.",
        "task_type": "topic_classify",
        "ground_truth": "technology",
        "expected_tier": 2,
        "category": "topic",
        "description": "Technology topic",
    },
    {
        "id": "G02",
        "input": "Stock market investments carry significant financial risk.",
        "task_type": "topic_classify",
        "ground_truth": "finance",
        "expected_tier": 2,
        "category": "topic",
        "description": "Finance topic",
    },
    {
        "id": "G03",
        "input": "The patient was admitted to hospital for treatment of the disease.",
        "task_type": "topic_classify",
        "ground_truth": "health",
        "expected_tier": 2,
        "category": "topic",
        "description": "Health topic",
    },
    {
        "id": "G04",
        "input": "The cricket team won the tournament after a brilliant match.",
        "task_type": "topic_classify",
        "ground_truth": "sports",
        "expected_tier": 2,
        "category": "topic",
        "description": "Sports topic",
    },

    # ── Category H: Spam Detection (Tier 2) ───────────────────────────────────
    {
        "id": "H01",
        "input": "Click here to claim your free money! Limited offer, act now, guaranteed!",
        "task_type": "spam_detect",
        "ground_truth": "spam",
        "expected_tier": 2,
        "category": "spam",
        "description": "Obvious spam",
    },
    {
        "id": "H02",
        "input": "Please find attached the quarterly report for your review.",
        "task_type": "spam_detect",
        "ground_truth": "not_spam",
        "expected_tier": 2,
        "category": "spam",
        "description": "Legitimate email",
    },
    {
        "id": "H03",
        "input": "Congratulations! You have won a million dollar prize. No risk!",
        "task_type": "spam_detect",
        "ground_truth": "spam",
        "expected_tier": 2,
        "category": "spam",
        "description": "Prize scam spam",
    },

    # ── Category J: Multi-step Reasoning (Tier 5/6) ───────────────────────────
    # These are placeholders — in a real run, these go to SLM/LLM
    {
        "id": "J01",
        "input": "Explain the trade-offs between using a transformer and a spiking neural network for sequence prediction tasks.",
        "task_type": "reasoning",
        "ground_truth": None,   # open-ended: evaluated by human or LLM judge
        "expected_tier": 5,
        "category": "reasoning",
        "description": "Technical comparison requiring domain reasoning",
    },
    {
        "id": "J02",
        "input": "A company has 3 data centres each consuming 2MW. If they migrate 40% of workloads to edge computing reducing centre load by 30%, what is the new total consumption?",
        "task_type": "reasoning",
        "ground_truth": "4.44 MW",
        "expected_tier": 5,
        "category": "reasoning",
        "description": "Multi-step arithmetic with context",
    },
    {
        "id": "J03",
        "input": "Write a Python function that takes a list of dictionaries and returns the top 3 by a given key.",
        "task_type": "generation",
        "ground_truth": None,   # evaluated by code execution
        "expected_tier": 5,
        "category": "generation",
        "description": "Code generation task",
    },
]


def get_tasks_by_tier(expected_tier: int) -> list:
    return [t for t in BENCHMARK_TASKS if t["expected_tier"] == expected_tier]


def get_tasks_by_category(category: str) -> list:
    return [t for t in BENCHMARK_TASKS if t["category"] == category]


def tier_distribution() -> dict:
    dist = {}
    for t in BENCHMARK_TASKS:
        tier = t["expected_tier"]
        dist[tier] = dist.get(tier, 0) + 1
    return dist
