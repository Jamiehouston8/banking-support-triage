"""Three intent classifiers behind one interface.

1. tfidf_logreg  - classic baseline: word + char n-gram TF-IDF, logistic regression
2. emb_logreg    - frozen Hugging Face sentence embeddings + logistic regression
3. emb_mlp       - frozen embeddings + a small PyTorch MLP head (early stopping on val)

Every model exposes `classes_` and `predict_proba(texts) -> (n, k) array`, and can be
saved/loaded from a directory, so the API and the gate never care which one won.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Lazy wrapper around a sentence-transformers model; returns L2-normalised vectors."""

    _cache: dict[str, object] = {}

    def __init__(self, name: str = EMBED_MODEL):
        self.name = name

    def _model(self):
        if self.name not in Embedder._cache:
            from sentence_transformers import SentenceTransformer

            Embedder._cache[self.name] = SentenceTransformer(self.name, device="cpu")
        return Embedder._cache[self.name]

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model().encode(
                list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )


class TfidfLogReg:
    kind = "tfidf_logreg"

    def __init__(self, C: float = 10.0):
        self.params = {"C": C, "word_ngrams": "1-2", "char_ngrams": "3-5"}
        self.pipe = Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=C, max_iter=3000)),
            ]
        )

    def fit(self, texts, labels, **_):
        self.pipe.fit(texts, labels)
        return self

    @property
    def classes_(self) -> list[str]:
        return list(self.pipe.classes_)

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))

    def save(self, d: Path):
        joblib.dump(self.pipe, d / "model.joblib")

    @classmethod
    def load(cls, d: Path, meta: dict):
        m = cls.__new__(cls)
        m.params = meta.get("params", {})
        m.pipe = joblib.load(d / "model.joblib")
        return m


class EmbLogReg:
    kind = "emb_logreg"

    def __init__(self, C: float = 20.0, embedder: Embedder | None = None):
        self.params = {"C": C, "embedder": EMBED_MODEL}
        self.embedder = embedder or Embedder()
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def fit(self, texts, labels, X=None, **_):
        self.clf.fit(X if X is not None else self.embedder(texts), labels)
        return self

    @property
    def classes_(self) -> list[str]:
        return list(self.clf.classes_)

    def predict_proba(self, texts, X=None) -> np.ndarray:
        return self.clf.predict_proba(X if X is not None else self.embedder(texts))

    def save(self, d: Path):
        joblib.dump(self.clf, d / "model.joblib")

    @classmethod
    def load(cls, d: Path, meta: dict):
        m = cls.__new__(cls)
        m.params = meta.get("params", {})
        m.embedder = Embedder(m.params.get("embedder", EMBED_MODEL))
        m.clf = joblib.load(d / "model.joblib")
        return m


class EmbMLP:
    """PyTorch MLP head on frozen sentence embeddings."""

    kind = "emb_mlp"

    def __init__(self, hidden: int = 512, dropout: float = 0.3, lr: float = 1e-3,
                 epochs: int = 60, patience: int = 6, embedder: Embedder | None = None):
        self.params = {"hidden": hidden, "dropout": dropout, "lr": lr, "max_epochs": epochs,
                       "patience": patience, "embedder": EMBED_MODEL}
        self.embedder = embedder or Embedder()
        self._classes: list[str] = []
        self.net = None
        self.history: list[dict] = []

    def _build(self, dim: int, k: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(dim, self.params["hidden"]), nn.ReLU(), nn.Dropout(self.params["dropout"]),
            nn.Linear(self.params["hidden"], k),
        )

    def fit(self, texts, labels, X=None, val_X=None, val_labels=None, **_):
        import torch

        torch.manual_seed(42)
        X = X if X is not None else self.embedder(texts)
        self._classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(self._classes)}
        y = torch.tensor([idx[l] for l in labels])
        Xt = torch.tensor(X)
        self.net = self._build(X.shape[1], len(self._classes))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"], weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.params["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                b = perm[i : i + 64]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]), y[b])
                loss.backward()
                opt.step()
            if val_X is None:
                continue
            acc = float((np.array(self._classes)[self.predict_proba(None, X=val_X).argmax(1)] == np.array(val_labels)).mean())
            self.history.append({"epoch": epoch + 1, "val_acc": acc})
            if acc > best:
                best, bad = acc, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.params["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @property
    def classes_(self) -> list[str]:
        return list(self._classes)

    def predict_proba(self, texts, X=None) -> np.ndarray:
        import torch

        X = X if X is not None else self.embedder(texts)
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X)), dim=1).numpy()

    def save(self, d: Path):
        import torch

        torch.save(self.net.state_dict(), d / "mlp.pt")
        (d / "classes.json").write_text(json.dumps(self._classes))

    @classmethod
    def load(cls, d: Path, meta: dict):
        import torch

        m = cls.__new__(cls)
        m.params = meta.get("params", {})
        m.embedder = Embedder(m.params.get("embedder", EMBED_MODEL))
        m._classes = json.loads((d / "classes.json").read_text())
        m.history = []
        sd = torch.load(d / "mlp.pt", map_location="cpu")
        m.net = m._build(sd["0.weight"].shape[1], len(m._classes))
        m.net.load_state_dict(sd)
        return m


REGISTRY = {c.kind: c for c in (TfidfLogReg, EmbLogReg, EmbMLP)}


def save_model(model, d: Path, extra: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    model.save(d)
    meta = {"kind": model.kind, "params": model.params, **extra}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_model(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    return REGISTRY[meta["kind"]].load(d, meta), meta
