# CheapestSolver

**A tiered orchestration framework that routes AI tasks to the cheapest solver that can correctly answer them.**

Most AI systems default to the most capable — and most energy-intensive — model available,
regardless of task complexity. CheapestSolver inverts this default.

Given any input task, it traverses a hierarchy of solvers from cheapest to most expensive,
stopping at the first tier that meets a confidence threshold. A regex or lookup table
handles what they can. Classical ML handles what it can. A small language model handles
the rest. A large language model is the last resort, not the default.

---

## The Tier Hierarchy

| Tier | Solver                  | Energy/Query | Cost/Query   | Example Tasks                        |
|------|-------------------------|-------------|--------------|--------------------------------------|
| 0    | Regex / Pattern Match   | ~0 mJ       | $0           | Email validation, date extraction    |
| 1    | Lookup / Mapping Table  | ~0 mJ       | $0           | Country codes, HTTP status, FAQ      |
| 2    | Classical ML (sklearn)  | ~0.5 mJ     | ~$0.000001   | Sentiment, intent, topic, spam       |
| 3    | Fine-tuned LM (BERT)    | ~5 mJ       | ~$0.00001    | Domain NER, specialised classification|
| 4    | RAG                     | ~20 mJ      | ~$0.0001     | Document QA, knowledge retrieval     |
| 5    | Small LM (Qwen/Phi)     | ~200 mJ     | ~$0.0005     | Moderate reasoning, code generation  |
| 6    | Large LM (GPT-4 class)  | ~2000 mJ    | ~$0.01       | Complex reasoning — last resort only |

---

## Quick Start

```bash
git clone https://github.com/joyboseroy/cheapest-solver
cd cheapest-solver
pip install -r requirements.txt

# Run the demo
python main.py --demo

# Run the full benchmark
python main.py --verbose
```

### Running tier 5 (small language model)

Requires [Ollama](https://ollama.com) installed and running:

```bash
ollama pull qwen2.5:7b
python run_ollama_test_v2.py
```

### Demo output

```
Input:  alice@wonderland.org
Solved by:   Regex / Pattern Match
Confidence:  1.00
Energy:      0.000 mJ
Cost:        $0.000000
Latency:     0.4 ms

Input:  The product is absolutely fantastic, best I have ever used!
Solved by:   Classical ML (XGBoost / sklearn)
Confidence:  0.70
Energy:      0.500 mJ
Cost:        $0.000001
```

---

## Benchmark Results

On 37 tasks spanning nine categories:

| Metric | CheapestSolver | All-LLM baseline |
|--------|----------------|-----------------|
| Accuracy, tiers 0–2 (n=33) | 100% (deterministic) | ~97% est. |
| Accuracy, tier 5 (n=4) | 75% (3 of 4 correct) | ~90% est. |
| Energy, tiers 0–2 | 8.50 mJ | 66,000 mJ est. |
| Energy, tier 5 (measured) | 800 mJ | 8,000 mJ est. |
| Total energy (all 37 tasks) | ~809 mJ | ~74,000 mJ est. |
| Total cost (all 37 tasks) | $0.000017 | $0.370 |

89% of tasks resolved at tiers 0–2. Tier 5 validated using Qwen-2.5-7B
via Ollama (21–79 seconds per query on a local workstation GPU).
Full results in `results/benchmark_results.json`.

---

## Architecture

```
cheapest-solver/
├── orchestrator/
│   ├── base.py           ← SolverResult, BaseSolver, Tier definitions
│   ├── characteriser.py  ← Lightweight problem tagger (no neural inference)
│   └── router.py         ← Tier traversal, confidence threshold, routing trace
├── solvers/
│   ├── tier0_regex.py       ← Pattern matching
│   ├── tier1_lookup.py      ← Dictionary / mapping table
│   ├── tier2_classical_ml.py← Lexicon + rule-based classifiers (plug in sklearn)
│   ├── tier3_finetuned.py   ← [stub] BERT-class fine-tuned model
│   ├── tier4_rag.py         ← [stub] RAG pipeline
│   ├── tier5_slm.py         ← Ollama local SLM (Qwen-2.5-7B, validated)
│   └── tier6_llm.py         ← [stub] OpenAI / Anthropic API
├── benchmark/
│   ├── tasks/benchmark_tasks.py  ← 37-task custom benchmark
│   └── evaluate.py               ← Runner, report, JSON export
├── paper/
│   └── paper.docx   ← arXiv paper draft
└── main.py               ← Entry point
```

---

## Adding a Custom Solver

Every solver implements the same two-method interface:

```python
from orchestrator.base import BaseSolver, SolverResult, Tier

class MySolver(BaseSolver):
    tier = Tier.CLASSICAL_ML   # or any tier

    def can_attempt(self, task: dict) -> bool:
        return task.get("task_type") in {"my_task_type"}

    def solve(self, task: dict) -> SolverResult:
        answer = ...  # your logic
        return SolverResult(answer=answer, confidence=0.9, tier=self.tier)
```

Pass it to the Router:

```python
router = Router(solvers=[MySolver(), ...], confidence_threshold=0.80)
result = router.route({"input": "...", "task_type": "my_task_type"})
```

---

## Adding a Custom Lookup Table

```python
from solvers.tier1_lookup import LookupSolver

solver = LookupSolver(custom_tables={
    "telecom_error_codes": {
        "E001": "Bearer not established",
        "E002": "Handover failed",
    }
})
```

---

## Motivation

This project grew from a simple observation: data centres and GPU clusters consume
enormous energy, a large fraction of which goes to LLM inference. Many queries sent
to LLMs in production are not LLM-class problems. A regex handles email validation
perfectly. A dictionary handles country code lookups perfectly. Using a billion-parameter
model for these tasks is not intelligence — it is waste.

The goal is not to replace LLMs. It is to use them only when nothing cheaper works.

---

## Paper

**"CheapestSolver: A Tiered Orchestration Framework for Energy-Efficient AI Task Resolution"**
Joy Bose, 2026. arXiv preprint (in preparation).

See `paper/CheapestSolver_v8_final.docx` for the current manuscript.
arXiv submission in preparation.

---

## Related Work

- FrugalGPT (Chen et al., 2023) — LLM cascading within the LLM tier
- RouteLLM (Ong et al., 2024) — learned routing between two LLM sizes
- Green AI (Schwartz et al., 2020) — framework for reporting AI energy cost

CheapestSolver differs by spanning the full computational spectrum (regex to LLM)
and treating energy as a first-class routing criterion.

---

## Author

**Dr. Joy Bose** — AI Researcher, Bengaluru

[Google Scholar](https://scholar.google.com/citations?user=1E0YgA4AAAAJ) ·
[LinkedIn](https://linkedin.com/in/joyboseroy) ·
[GitHub](https://github.com/joyboseroy)

---

## License

MIT
