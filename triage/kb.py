"""Vector search over the help-centre knowledge base."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_PATH = Path(__file__).resolve().parent.parent / "kb" / "articles.json"
INTENT_BOOST = 0.15  # nudge towards the article mapped to the classifier's predicted intent

Encoder = Callable[[list[str]], np.ndarray]


@dataclass
class Article:
    id: str
    title: str
    intents: list[str]
    body: str


@dataclass
class Hit:
    article: Article
    score: float


def load_kb(path: Path = KB_PATH) -> tuple[list[Article], set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = [Article(a["id"], a["title"], a["intents"], a["body"]) for a in raw["articles"]]
    return arts, set(raw["always_human"]["intents"])


class TfidfEncoder:
    """Lightweight encoder used in tests and as a no-download fallback."""

    def __init__(self, corpus: list[str]):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(corpus)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(list(texts)).toarray().astype(np.float32)


class KnowledgeBase:
    def __init__(self, articles: list[Article], encoder: Encoder):
        self.articles = articles
        self.encoder = encoder
        self.by_intent = {i: a for a in articles for i in a.intents}
        self.matrix = encoder([f"{a.title}. {a.body}" for a in articles])

    def search(self, query: str, k: int = 3, intent: str | None = None) -> list[Hit]:
        q = self.encoder([query])[0]
        scores = self.matrix @ q
        if intent and intent in self.by_intent:
            scores = scores.copy()
            scores[self.articles.index(self.by_intent[intent])] += INTENT_BOOST
        top = np.argsort(-scores)[:k]
        return [Hit(self.articles[i], float(scores[i])) for i in top]
