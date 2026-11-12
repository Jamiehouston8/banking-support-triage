import pytest

from triage.data import load
from triage.kb import KnowledgeBase, TfidfEncoder, load_kb
from triage.models import TfidfLogReg
from triage.pipeline import Triage


@pytest.fixture(scope="session")
def splits():
    return load()


@pytest.fixture(scope="session")
def small_triage(splits):
    """A real (TF-IDF) model trained on the full train split - fast, no downloads, no LLM."""
    train, _, _ = splits
    model = TfidfLogReg().fit(train.texts, train.labels)
    articles, always_human = load_kb()
    kb = KnowledgeBase(articles, TfidfEncoder([f"{a.title}. {a.body}" for a in articles]))
    meta = {
        "kind": model.kind, "params": model.params, "threshold": 0.5, "target_accuracy": 0.98,
        "reference": {"intent_distribution": [1 / 77] * 77, "below_gate_rate": 0.2, "mean_confidence": 0.7, "n": 1},
    }
    return Triage(model, meta, kb, always_human, client=None)
