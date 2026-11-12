"""The confidence gate: decide which messages are safe to auto-answer.

A classifier that is 93% accurate is still wrong on 1 in 14 customers. Instead of
shipping that error rate, we pick a confidence threshold on the validation set so the
messages we DO auto-answer meet a target accuracy, and send the rest to a person.
Fraud/security intents bypass the gate entirely and always go to a person.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GateResult:
    threshold: float
    coverage: float          # share of messages auto-answered
    auto_accuracy: float     # accuracy on the auto-answered share


def coverage_at(conf: np.ndarray, correct: np.ndarray, t: float) -> GateResult:
    mask = conf >= t
    cov = float(mask.mean())
    acc = float(correct[mask].mean()) if mask.any() else 1.0
    return GateResult(float(t), cov, acc)


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_accuracy: float = 0.98,
                     min_support: int = 50) -> GateResult:
    """Lowest threshold (= highest coverage) whose auto-answered accuracy >= target.

    `min_support` stops us picking a threshold justified by a handful of examples.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)
    c_sorted, ok_sorted = conf[order], correct[order]
    running_acc = np.cumsum(ok_sorted) / np.arange(1, len(ok_sorted) + 1)
    best = None
    for i in range(min_support - 1, len(c_sorted)):
        # only cut between distinct confidence values
        if i + 1 < len(c_sorted) and c_sorted[i + 1] == c_sorted[i]:
            continue
        if running_acc[i] >= target_accuracy:
            best = i
    if best is None:
        return GateResult(1.01, 0.0, 1.0)  # nothing is safe: everything goes to a person
    return coverage_at(conf, correct, c_sorted[best])


def curve(conf: np.ndarray, correct: np.ndarray, points=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    return [coverage_at(np.asarray(conf), np.asarray(correct, dtype=bool), t).__dict__ for t in points]


def choose_risk_cutoff(risk_mass, pred_is_risky, true_is_risky, conf, threshold,
                       candidates=(0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01)) -> float:
    """Largest cutoff on total fraud-intent probability that lets ZERO true fraud/security
    messages be auto-answered on the validation set. Chosen on validation only."""
    risk_mass, conf = np.asarray(risk_mass), np.asarray(conf)
    base_auto = (conf >= threshold) & ~np.asarray(pred_is_risky)
    for c in candidates:
        if not (base_auto & (risk_mass < c) & np.asarray(true_is_risky)).any():
            return float(c)
    return float(candidates[-1])


def route(intent: str, confidence: float, threshold: float, always_human: set[str],
          risk_mass: float = 0.0, risk_cutoff: float = 1.01) -> tuple[str, str]:
    """Return (decision, reason). decision is 'auto_answer' or 'human_review'.

    risk_mass = the model's total probability across ALL fraud/security intents. Even when
    the top prediction is harmless, a meaningful chance of fraud sends it to a person.
    """
    if intent in always_human:
        return "human_review", f"'{intent}' is a fraud/security intent: always handled by a person"
    if risk_mass >= risk_cutoff:
        return "human_review", f"{risk_mass:.0%} probability of a fraud/security intent (cutoff {risk_cutoff:.0%})"
    if confidence < threshold:
        return "human_review", f"confidence {confidence:.2f} is below the gate ({threshold:.2f})"
    return "auto_answer", f"confidence {confidence:.2f} clears the gate ({threshold:.2f})"
