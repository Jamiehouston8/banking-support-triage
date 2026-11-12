"""classify -> gate -> retrieve -> draft -> grounding check -> decision."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import drift
from .answer import draft_reply, make_client
from .gate import route
from .kb import KnowledgeBase, TfidfEncoder, load_kb
from .models import Embedder, load_model

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "best"


class Triage:
    def __init__(self, model, meta: dict, kb: KnowledgeBase, always_human: set[str], client=None):
        self.model, self.meta, self.kb, self.always_human, self.client = model, meta, kb, always_human, client
        self.threshold = float(meta["threshold"])
        self.risk_cutoff = float(meta.get("risk_cutoff", 1.01))
        self.classes = model.classes_
        self.risk_idx = [i for i, c in enumerate(self.classes) if c in always_human]

    @classmethod
    def from_dir(cls, model_dir: Path = DEFAULT_MODEL_DIR, use_embeddings_for_kb: bool | None = None):
        model, meta = load_model(Path(model_dir))
        articles, always_human = load_kb()
        if use_embeddings_for_kb is None:
            use_embeddings_for_kb = meta["kind"] != "tfidf_logreg"
        enc = Embedder() if use_embeddings_for_kb else TfidfEncoder([f"{a.title}. {a.body}" for a in articles])
        return cls(model, meta, KnowledgeBase(articles, enc), always_human, make_client())

    def classify(self, texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(texts)
        idx = proba.argmax(1)
        return [self.classes[i] for i in idx], proba.max(1), proba

    def triage(self, text: str) -> dict:
        t0 = time.perf_counter()
        labels, conf, proba = self.classify([text])
        intent, confidence = labels[0], float(conf[0])
        top3 = [
            {"intent": self.classes[i], "p": round(float(proba[0, i]), 4)} for i in np.argsort(-proba[0])[:3]
        ]
        risk_mass = float(proba[0, self.risk_idx].sum())
        decision, reason = route(intent, confidence, self.threshold, self.always_human, risk_mass, self.risk_cutoff)
        hits = self.kb.search(text, k=3, intent=intent)
        out = {
            "intent": intent,
            "confidence": round(confidence, 4),
            "top3": top3,
            "fraud_risk": round(risk_mass, 4),
            "decision": decision,
            "reason": reason,
            "retrieved": [{"id": h.article.id, "title": h.article.title, "score": round(h.score, 3)} for h in hits],
            "reply": None,
            "citations": [],
        }
        if decision == "auto_answer":
            d = draft_reply(text, hits, self.client)
            if d.problems:
                out["decision"] = "human_review"
                out["reason"] = "draft failed the grounding check: " + "; ".join(d.problems)
            else:
                out["reply"], out["citations"] = d.answer, d.citations
            out["draft_mode"] = d.mode
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def drift_report(self, texts: list[str]) -> dict:
        labels, conf, _ = self.classify(texts)
        return drift.report(self.meta["reference"], labels, conf, self.classes, self.threshold)


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())
