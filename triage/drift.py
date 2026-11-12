"""Production monitoring: is today's traffic still like the traffic we trained on?

Two cheap, label-free signals:
- PSI (population stability index) on the predicted-intent distribution
- the share of messages falling below the confidence gate (rising = model is less sure)
"""
from __future__ import annotations

import numpy as np

# Absolute floors. The real thresholds come from `noise_floor`: with 77 intents and a few
# hundred messages, sampling noise alone produces PSI around 0.2, so a fixed 0.25 rule
# raises false alarms on unchanged traffic (we hit exactly that - see reports/RESULTS.md).
PSI_WARN, PSI_ALERT = 0.05, 0.10


def distribution(labels: list[str], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.zeros(len(classes))
    for l in labels:
        counts[idx[l]] += 1
    return counts / max(counts.sum(), 1)


def psi(ref: np.ndarray, cur: np.ndarray, eps: float = 1e-4) -> float:
    ref = np.clip(np.asarray(ref, float), eps, None)
    cur = np.clip(np.asarray(cur, float), eps, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def noise_floor(ref: np.ndarray, n_ref: int, n_cur: int, sims: int = 400, seed: int = 0) -> tuple[float, float]:
    """95th/99th percentile PSI between two samples drawn from the SAME distribution,
    at these sample sizes. Anything below this is indistinguishable from chance."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, float) / np.sum(ref)
    vals = [psi(rng.multinomial(n_ref, ref) / n_ref, rng.multinomial(n_cur, ref) / n_cur) for _ in range(sims)]
    return float(np.percentile(vals, 95)), float(np.percentile(vals, 99))


def reference_stats(pred_labels: list[str], conf: np.ndarray, classes: list[str], threshold: float) -> dict:
    return {
        "intent_distribution": distribution(pred_labels, classes).tolist(),
        "below_gate_rate": float((np.asarray(conf) < threshold).mean()),
        "mean_confidence": float(np.mean(conf)),
        "n": len(pred_labels),
    }


def report(ref: dict, pred_labels: list[str], conf: np.ndarray, classes: list[str], threshold: float) -> dict:
    cur = distribution(pred_labels, classes)
    score = psi(np.array(ref["intent_distribution"]), cur)
    below = float((np.asarray(conf) < threshold).mean())
    p95, p99 = noise_floor(np.array(ref["intent_distribution"]), int(ref["n"]), len(pred_labels))
    warn_at, alert_at = max(p95, PSI_WARN), max(p99, PSI_ALERT)
    alerts = []
    if score >= alert_at:
        alerts.append(f"intent mix has shifted (PSI {score:.2f} >= {alert_at:.2f}, the 99th percentile of chance at this sample size)")
    elif score >= warn_at:
        alerts.append(f"intent mix is moving (PSI {score:.2f} >= {warn_at:.2f}, the 95th percentile of chance)")
    if below > ref["below_gate_rate"] + 0.10:
        alerts.append(
            f"{below:.0%} of messages fall below the confidence gate vs {ref['below_gate_rate']:.0%} at training time"
        )
    ref_d = np.array(ref["intent_distribution"])
    top_moves = sorted(range(len(classes)), key=lambda i: -abs(cur[i] - ref_d[i]))[:5]
    return {
        "n": len(pred_labels),
        "psi": round(score, 4),
        "psi_warn_at": round(warn_at, 4),
        "psi_alert_at": round(alert_at, 4),
        "below_gate_rate": round(below, 4),
        "reference_below_gate_rate": round(ref["below_gate_rate"], 4),
        "mean_confidence": round(float(np.mean(conf)), 4),
        "status": "alert" if any("shifted" in a or "below" in a for a in alerts) else ("warn" if alerts else "ok"),
        "alerts": alerts,
        "biggest_moves": [
            {"intent": classes[i], "reference": round(float(ref_d[i]), 4), "current": round(float(cur[i]), 4)}
            for i in top_moves
        ],
    }
