"""Load the Banking77 dataset (PolyAI, CC-BY-4.0) and make a fixed validation split."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEED = 42


@dataclass
class Split:
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def _read(path: Path) -> Split:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return Split([r["text"].strip() for r in rows], [r["category"] for r in rows])


def load(val_size: float = 0.15) -> tuple[Split, Split, Split]:
    """Return (train, val, test).

    The official test split is never touched during model or threshold selection.
    Validation is a stratified slice of the official train split.
    """
    full = _read(RAW / "train.csv")
    test = _read(RAW / "test.csv")
    tr_x, va_x, tr_y, va_y = train_test_split(
        full.texts, full.labels, test_size=val_size, stratify=full.labels, random_state=SEED
    )
    return Split(tr_x, tr_y), Split(va_x, va_y), test
