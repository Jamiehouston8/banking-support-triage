"""Train, compare and select the intent model; calibrate the gate; write the report.

    python -m triage.train            # all three models
    python -m triage.train --fast     # TF-IDF only (no model download)

Model AND threshold selection use the validation split only. The official Banking77
test split is scored once, at the end, for the report.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import drift
from .data import ROOT, load
from .gate import choose_risk_cutoff, choose_threshold, coverage_at, curve
from .kb import KnowledgeBase, TfidfEncoder, load_kb
from .models import EmbLogReg, EmbMLP, Embedder, TfidfLogReg, save_model

TARGET_ACC = 0.98
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"


def embed_cached(embedder: Embedder, name: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.npy"
    if p.exists():
        X = np.load(p)
        if len(X) == len(texts):
            return X
    X = embedder(texts)
    np.save(p, X)
    return X


def evaluate(model, split, X=None) -> dict:
    t0 = time.perf_counter()
    proba = model.predict_proba(split.texts, X=X) if X is not None else model.predict_proba(split.texts)
    ms = (time.perf_counter() - t0) * 1000 / len(split)
    classes = np.array(model.classes_)
    pred = classes[proba.argmax(1)]
    y = np.array(split.labels)
    return {
        "pred": pred, "conf": proba.max(1), "correct": pred == y,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ms_per_msg": ms,
    }


def main(fast: bool = False):
    random.seed(42)
    np.random.seed(42)
    train, val, test = load()
    articles, always_human = load_kb()
    print(f"train={len(train)} val={len(val)} test={len(test)} intents={len(set(train.labels))}")

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("banking77-triage")

    feats = {}
    candidates = [TfidfLogReg()]
    if not fast:
        emb = Embedder()
        feats = {s: embed_cached(emb, s, d.texts) for s, d in (("train", train), ("val", val), ("test", test))}
        candidates += [EmbLogReg(embedder=emb), EmbMLP(embedder=emb)]

    runs = []
    for m in candidates:
        uses_emb = m.kind != "tfidf_logreg"
        with mlflow.start_run(run_name=m.kind) as run:
            t0 = time.perf_counter()
            if uses_emb:
                m.fit(train.texts, train.labels, X=feats["train"], val_X=feats["val"], val_labels=val.labels)
            else:
                m.fit(train.texts, train.labels)
            fit_s = time.perf_counter() - t0
            v = evaluate(m, val, feats.get("val") if uses_emb else None)
            te = evaluate(m, test, feats.get("test") if uses_emb else None)
            gate = choose_threshold(v["conf"], v["correct"], TARGET_ACC)
            gate_test = coverage_at(te["conf"], te["correct"], gate.threshold)
            sample = test.texts[:300]  # end-to-end latency INCLUDING embedding, on raw text
            m.predict_proba(sample[:10])
            t1 = time.perf_counter()
            for i in range(0, len(sample), 32):
                m.predict_proba(sample[i : i + 32])
            te["ms_per_msg"] = (time.perf_counter() - t1) * 1000 / len(sample)
            metrics = {
                "val_accuracy": v["accuracy"], "val_macro_f1": v["macro_f1"],
                "test_accuracy": te["accuracy"], "test_macro_f1": te["macro_f1"],
                "gate_threshold": gate.threshold,
                "val_coverage": gate.coverage, "val_auto_accuracy": gate.auto_accuracy,
                "test_coverage": gate_test.coverage, "test_auto_accuracy": gate_test.auto_accuracy,
                "fit_seconds": fit_s, "ms_per_msg": te["ms_per_msg"],
            }
            mlflow.log_params({"model": m.kind, **{k: str(v_) for k, v_ in m.params.items()}, "target_accuracy": TARGET_ACC})
            mlflow.log_metrics(metrics)
            if getattr(m, "history", None):
                for h in m.history:
                    mlflow.log_metric("epoch_val_acc", h["val_acc"], step=h["epoch"])
            runs.append({"model": m, "kind": m.kind, "run_id": run.info.run_id, "metrics": metrics, "val": v, "test": te})
            print(f"{m.kind:13s} val_f1={v['macro_f1']:.4f} test_acc={te['accuracy']:.4f} "
                  f"gate={gate.threshold:.2f} test_cov={gate_test.coverage:.1%} test_auto_acc={gate_test.auto_accuracy:.2%}")

    best = max(runs, key=lambda r: r["metrics"]["val_macro_f1"])  # selected on VALIDATION only
    bm = best["model"]
    print(f"selected: {best['kind']}")

    # ---- fraud safety net: cutoff on total fraud-intent probability, chosen on VALIDATION ----
    thr = best["metrics"]["gate_threshold"]
    risk_cols = [i for i, c in enumerate(bm.classes_) if c in always_human]

    def proba_of(split, name):
        return bm.predict_proba(split.texts, X=feats[name]) if best["kind"] != "tfidf_logreg" else bm.predict_proba(split.texts)

    pv = proba_of(val, "val")
    risk_cutoff = choose_risk_cutoff(
        pv[:, risk_cols].sum(1), [p in always_human for p in best["val"]["pred"]],
        [l in always_human for l in val.labels], best["val"]["conf"], thr)
    print(f"fraud-risk cutoff (from validation): {risk_cutoff}")

    # ---- pipeline-level evaluation on test (gate + fraud rules + retrieval) ----
    te = best["test"]
    y = np.array(test.labels)
    is_human_intent = np.array([p in always_human for p in te["pred"]])
    risk_test = proba_of(test, "test")[:, risk_cols].sum(1)
    auto_no_net = (te["conf"] >= thr) & ~is_human_intent
    auto = auto_no_net & (risk_test < risk_cutoff)
    true_fraud = np.array([l in always_human for l in y])
    fraud_auto_answered = int((auto & true_fraud).sum())
    fraud_missed_examples = [
        {"text": test.texts[i], "true": y[i], "predicted": te["pred"][i],
         "confidence": round(float(te["conf"][i]), 3), "fraud_risk": round(float(risk_test[i]), 3)}
        for i in np.where(auto & true_fraud)[0]
    ]

    enc = Embedder() if best["kind"] != "tfidf_logreg" else TfidfEncoder([f"{a.title}. {a.body}" for a in articles])
    kb = KnowledgeBase(articles, enc)
    art_of = kb.by_intent
    r1 = r3 = 0
    auto_idx = np.where(auto)[0]
    for i in auto_idx:
        hits = kb.search(test.texts[i], k=3, intent=te["pred"][i])
        ids = [h.article.id for h in hits]
        gold = art_of[y[i]].id
        r1 += ids[0] == gold
        r3 += gold in ids
    pipeline = {
        "auto_answered_share": float(auto.mean()),
        "auto_answered_intent_accuracy": float(te["correct"][auto].mean()) if auto.any() else None,
        "sent_to_person_share": float(1 - auto.mean()),
        "fraud_security_messages_in_test": int(true_fraud.sum()),
        "fraud_security_messages_auto_answered": fraud_auto_answered,
        "fraud_security_auto_answered_without_safety_net": int((auto_no_net & true_fraud).sum()),
        "fraud_risk_cutoff": risk_cutoff,
        "fraud_missed_examples": fraud_missed_examples,
        "retrieval_hit_at_1": r1 / max(len(auto_idx), 1),
        "retrieval_hit_at_3": r3 / max(len(auto_idx), 1),
    }

    # ---- confusions ----
    classes = bm.classes_
    cm = confusion_matrix(y, te["pred"], labels=classes)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:8]
    confusions = [{"true": classes[i // len(classes)], "predicted": classes[i % len(classes)], "count": int(cm.flat[i])}
                  for i in flat if cm.flat[i] > 0]

    # ---- drift demo ----
    ref = drift.reference_stats(list(best["val"]["pred"]), best["val"]["conf"], classes, thr)
    rng = np.random.default_rng(0)
    same = rng.choice(len(test), 1000, replace=False)
    card_heavy = [i for i, l in enumerate(test.labels) if l.startswith(("card_", "lost_", "compromised", "declined_card"))]
    shifted = np.concatenate([rng.choice(card_heavy, 700), rng.choice(len(test), 300, replace=False)])
    vp, vc = best["val"]["pred"], best["val"]["conf"]
    half = len(vp) // 2
    ref_a = drift.reference_stats(list(vp[:half]), vc[:half], classes, thr)
    drift_demo = {
        "like_for_like": drift.report(ref_a, list(vp[half:]), vc[half:], classes, thr),
        "balanced_test_split": drift.report(ref, list(te["pred"][same]), te["conf"][same], classes, thr),
        "card_incident_spike": drift.report(ref, list(te["pred"][shifted]), te["conf"][shifted], classes, thr),
    }

    # ---- save the winner ----
    out = ROOT / "models" / "best"
    save_model(bm, out, {
        "threshold": thr, "target_accuracy": TARGET_ACC, "risk_cutoff": risk_cutoff,
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "mlflow_run_id": best["run_id"], "reference": ref,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    REPORTS.mkdir(exist_ok=True)
    summary = {
        "dataset": {"name": "Banking77 (PolyAI, CC-BY-4.0)", "train": len(train), "val": len(val), "test": len(test), "intents": len(classes)},
        "target_accuracy": TARGET_ACC,
        "selected_model": best["kind"],
        "models": [{"model": r["kind"], **{k: round(v, 4) for k, v in r["metrics"].items()}} for r in runs],
        "gate_curve_test": curve(te["conf"], te["correct"]),
        "pipeline_test": pipeline,
        "top_confusions_test": confusions,
        "drift_demo": drift_demo,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(json.dumps(pipeline, indent=2))


def write_report(s: dict):
    L = ["# Results\n", f"Dataset: {s['dataset']['name']}: {s['dataset']['train']} train / {s['dataset']['val']} validation / "
         f"{s['dataset']['test']} test, {s['dataset']['intents']} intents. Model and gate chosen on validation; test scored once.\n",
         "## Model comparison (official test split)\n",
         "| Model | Test accuracy | Test macro-F1 | Gate threshold | Auto-answered | Accuracy when auto-answered | ms / message (CPU, batch 32) |",
         "|---|---|---|---|---|---|---|"]
    for m in s["models"]:
        star = " **(selected)**" if m["model"] == s["selected_model"] else ""
        L.append(f"| `{m['model']}`{star} | {m['test_accuracy']:.2%} | {m['test_macro_f1']:.4f} | {m['gate_threshold']:.2f} | "
                 f"{m['test_coverage']:.1%} | {m['test_auto_accuracy']:.2%} | {m['ms_per_msg']:.2f} |")
    p = s["pipeline_test"]
    L += ["", f"## End-to-end pipeline (test, target {s['target_accuracy']:.0%} accuracy on auto-answers)\n",
          f"- Auto-answered: **{p['auto_answered_share']:.1%}** of messages, intent correct on **{p['auto_answered_intent_accuracy']:.2%}** of those",
          f"- Sent to a person: {p['sent_to_person_share']:.1%}",
          f"- Fraud/security messages in test: {p['fraud_security_messages_in_test']}. Auto-answered by mistake: **{p['fraud_security_messages_auto_answered']}** "
          f"(was {p['fraud_security_auto_answered_without_safety_net']} before the fraud-probability safety net, cutoff {p['fraud_risk_cutoff']:.0%} chosen on validation)",
          *[f"  - Still missed: \"{e['text']}\" (true `{e['true']}`, predicted `{e['predicted']}` at {e['confidence']:.0%}, fraud risk {e['fraud_risk']:.0%})" for e in p['fraud_missed_examples']],
          f"- Retrieval on auto-answered messages: hit@1 **{p['retrieval_hit_at_1']:.1%}**, hit@3 {p['retrieval_hit_at_3']:.1%}",
          "", "## Coverage vs accuracy (selected model, test)\n", "| Threshold | Auto-answered | Accuracy on those |", "|---|---|---|"]
    for c in s["gate_curve_test"]:
        L.append(f"| {c['threshold']:.2f} | {c['coverage']:.1%} | {c['auto_accuracy']:.2%} |")
    L += ["", "## Most common mistakes (test)\n", "| True intent | Predicted | Count |", "|---|---|---|"]
    for c in s["top_confusions_test"]:
        L.append(f"| {c['true']} | {c['predicted']} | {c['count']} |")
    d = s["drift_demo"]
    L += ["", "## Drift check demo\n",
          f"- Like-for-like (validation half A as reference, half B as live traffic): status **{d['like_for_like']['status']}**, PSI {d['like_for_like']['psi']} (alert line {d['like_for_like']['psi_alert_at']}). With the original fixed 0.25 threshold this unchanged traffic raised a false alarm; thresholds are now calibrated to sample size.",
          f"- 1,000 messages from the official test split, which is class-balanced unlike the training traffic: status **{d['balanced_test_split']['status']}**, PSI {d['balanced_test_split']['psi']} (alert line at this sample size: {d['balanced_test_split']['psi_alert_at']}). A mild mix change like this is within chance at 1,000 messages, so no alarm.",
          f"- 1,000 messages during a simulated card-incident spike: status **{d['card_incident_spike']['status']}**, "
          f"PSI {d['card_incident_spike']['psi']}; alerts: {'; '.join(d['card_incident_spike']['alerts'])}", ""]
    (REPORTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="TF-IDF model only")
    main(ap.parse_args().fast)
