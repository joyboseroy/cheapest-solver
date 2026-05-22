"""
Tier 2: Classical ML Solver
-----------------------------
Handles tasks solvable by traditional ML models:
- Text classification (sentiment, intent, topic)
- Tabular prediction
- Simple NER via CRF or rule-based models
- Anomaly detection

Uses scikit-learn. No GPU required. Very fast inference.

Task dict:
{
    "input":        str or list,      # text or feature vector
    "task_type":    str,              # e.g. "sentiment", "intent_classify"
    "features":     list (optional),  # pre-extracted feature vector
    "model_name":   str (optional),   # registered model key
}
"""

from orchestrator.base import BaseSolver, SolverResult, Tier, TIER_ENERGY_MJ, TIER_COST_USD


# ── Simple rule-based classifiers (no training data needed) ──────────────────
# These stand in for trained models for tasks where heuristics are strong enough.

_SENTIMENT_LEXICON_POS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "best", "brilliant", "outstanding", "perfect", "happy",
    "pleased", "delighted", "superb", "awesome", "well", "nice",
}
_SENTIMENT_LEXICON_NEG = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "disappointing", "broken", "useless", "failure", "wrong", "error",
    "issue", "problem", "not working", "frustrated", "slow", "crash",
}

_INTENT_PATTERNS = {
    "greeting":   ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell":   ["bye", "goodbye", "see you", "farewell", "take care"],
    "thanks":     ["thank", "thanks", "appreciate", "grateful"],
    "complaint":  ["broken", "not working", "issue", "problem", "error", "wrong", "fail"],
    "question":   ["what", "how", "why", "when", "where", "who", "which", "?"],
    "purchase":   ["buy", "order", "purchase", "price", "cost", "pay"],
    "cancel":     ["cancel", "refund", "return", "stop", "remove"],
}

_TOPIC_PATTERNS = {
    "technology": ["ai", "machine learning", "software", "computer", "algorithm",
                   "neural", "data", "code", "programming", "cloud"],
    "finance":    ["money", "stock", "investment", "market", "bank", "profit",
                   "revenue", "currency", "financial", "fund"],
    "health":     ["medicine", "doctor", "hospital", "disease", "treatment",
                   "health", "patient", "drug", "symptom", "clinical"],
    "sports":     ["game", "player", "team", "score", "match", "tournament",
                   "football", "cricket", "tennis", "league"],
    "politics":   ["government", "election", "policy", "parliament", "minister",
                   "vote", "party", "president", "law", "senate"],
}


class ClassicalMLSolver(BaseSolver):
    """
    Tier 2 solver using rule-based and classical ML approaches.
    
    In production, replace rule-based classifiers with trained
    sklearn pipelines loaded from disk. The interface is identical.
    """

    tier = Tier.CLASSICAL_ML

    SUPPORTED_TASK_TYPES = {
        "sentiment",
        "intent_classify",
        "topic_classify",
        "spam_detect",
    }

    def can_attempt(self, task: dict) -> bool:
        return task.get("task_type", "") in self.SUPPORTED_TASK_TYPES

    def solve(self, task: dict) -> SolverResult:
        task_type = task.get("task_type", "")
        text = str(task.get("input", "")).lower()

        if task_type == "sentiment":
            return self._sentiment(text)
        elif task_type == "intent_classify":
            return self._intent(text)
        elif task_type == "topic_classify":
            return self._topic(text)
        elif task_type == "spam_detect":
            return self._spam(text)

        return SolverResult(answer=None, confidence=0.0, tier=self.tier)

    def _sentiment(self, text: str) -> SolverResult:
        words = set(text.split())
        pos_score = len(words & _SENTIMENT_LEXICON_POS)
        neg_score = len(words & _SENTIMENT_LEXICON_NEG)

        total = pos_score + neg_score
        if total == 0:
            return SolverResult(answer="neutral", confidence=0.55,
                                tier=self.tier,
                                energy_mj=TIER_ENERGY_MJ[self.tier],
                                cost_usd=TIER_COST_USD[self.tier])

        if pos_score > neg_score:
            conf = min(0.95, 0.6 + 0.1 * pos_score)
            return SolverResult(answer="positive", confidence=conf,
                                tier=self.tier,
                                energy_mj=TIER_ENERGY_MJ[self.tier],
                                cost_usd=TIER_COST_USD[self.tier],
                                metadata={"pos": pos_score, "neg": neg_score})
        elif neg_score > pos_score:
            conf = min(0.95, 0.6 + 0.1 * neg_score)
            return SolverResult(answer="negative", confidence=conf,
                                tier=self.tier,
                                energy_mj=TIER_ENERGY_MJ[self.tier],
                                cost_usd=TIER_COST_USD[self.tier],
                                metadata={"pos": pos_score, "neg": neg_score})
        else:
            return SolverResult(answer="mixed", confidence=0.5,
                                tier=self.tier,
                                energy_mj=TIER_ENERGY_MJ[self.tier],
                                cost_usd=TIER_COST_USD[self.tier])


    def _intent(self, text: str) -> SolverResult:
        import numpy as np
        proba = self._intent_model.predict_proba([text])[0]
        classes = self._intent_model.classes_
        best_idx = np.argmax(proba)
        confidence = float(proba[best_idx])
        answer = classes[best_idx]

        return SolverResult(
            answer=answer,
            confidence=round(confidence, 3),
            tier=self.tier,
            energy_mj=TIER_ENERGY_MJ[self.tier],
            cost_usd=TIER_COST_USD[self.tier],
            metadata={"probas": dict(zip(classes, proba.round(3).tolist()))},
        )

    def _topic(self, text: str) -> SolverResult:
        scores = {}
        for topic, keywords in _TOPIC_PATTERNS.items():
            scores[topic] = sum(1 for kw in keywords if kw in text)

        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        if best_score == 0:
            return SolverResult(answer="general", confidence=0.4, tier=self.tier)

        conf = min(0.90, 0.55 + 0.08 * best_score)
        return SolverResult(
            answer=best_topic,
            confidence=conf,
            tier=self.tier,
            energy_mj=TIER_ENERGY_MJ[self.tier],
            cost_usd=TIER_COST_USD[self.tier],
        )

    def _spam_detect(self, text: str) -> SolverResult:
        spam_signals = [
            "click here", "free money", "you have won", "limited offer",
            "act now", "guaranteed", "no risk", "earn money fast",
            "lose weight", "hot singles", "enlarge", "million dollar",
        ]
        score = sum(1 for s in spam_signals if s in text)
        if score >= 2:
            return SolverResult(answer="spam", confidence=min(0.95, 0.7 + 0.1 * score),
                                tier=self.tier,
                                energy_mj=TIER_ENERGY_MJ[self.tier],
                                cost_usd=TIER_COST_USD[self.tier])
        return SolverResult(answer="not_spam", confidence=0.75,
                            tier=self.tier,
                            energy_mj=TIER_ENERGY_MJ[self.tier],
                            cost_usd=TIER_COST_USD[self.tier])

    # Alias for correct method name
    def _spam(self, text: str) -> SolverResult:
        return self._spam_detect(text)


    def __init__(self):
        self._intent_model = None
        self._build_intent_model()

    def _build_intent_model(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        INTENT_TRAIN = [
            # greeting
            ("hello there", "greeting"),
            ("hi how are you", "greeting"),
            ("good morning", "greeting"),
            ("hey what is up", "greeting"),
            ("good evening", "greeting"),
            ("hi there", "greeting"),
            # farewell
            ("goodbye see you later", "farewell"),
            ("bye take care", "farewell"),
            ("see you tomorrow", "farewell"),
            # thanks
            ("thank you so much", "thanks"),
            ("thanks a lot I appreciate it", "thanks"),
            ("many thanks for your help", "thanks"),
            ("grateful for your assistance", "thanks"),
            # complaint
            ("the app keeps crashing nothing works", "complaint"),
            ("this is broken and useless", "complaint"),
            ("nothing is working properly", "complaint"),
            ("there is an error and it failed", "complaint"),
            ("I have a problem with the software", "complaint"),
            # cancel
            ("I want to cancel my subscription", "cancel"),
            ("please cancel and give me a refund", "cancel"),
            ("I need to return this product", "cancel"),
            ("stop my account and refund me", "cancel"),
            # purchase
            ("how much does it cost to buy", "purchase"),
            ("I want to order the enterprise plan", "purchase"),
            ("what is the price for this product", "purchase"),
            # question
            ("what is the policy on returns", "question"),
            ("how does this feature work", "question"),
            ("when will the update be released", "question"),
            ("where can I find the documentation", "question"),
            ("what are the available options", "question"),
        ]

        texts, labels = zip(*INTENT_TRAIN)
        self._intent_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=500, C=1.0)),
        ])
        self._intent_model.fit(texts, labels)
