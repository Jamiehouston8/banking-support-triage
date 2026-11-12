# Banking Support Triage

An ML + GenAI service that reads a bank customer's message, works out what they need, and decides whether it is **safe to answer automatically** or must go to a person. If it answers, the reply is grounded in help-centre articles and cited.

Trained and evaluated on **Banking77** (PolyAI, CC-BY-4.0): 13,083 real customer banking queries across 77 intents.

```
message ─▶ intent classifier ─▶ confidence gate ─▶ fraud safety net ─▶ vector search ─▶ grounded reply ─▶ grounding check
            (scikit-learn /       (threshold set       (any meaningful      (Hugging Face     (Claude, or       (fails closed:
             PyTorch on HF         on validation for    fraud probability    embeddings)       extractive        any doubt → person)
             embeddings)           98% accuracy)        → person)                              fallback)
```

## Results (official Banking77 test split, scored once)

| Model | Test accuracy | Macro-F1 | Auto-answered | Accuracy when auto-answered | ms / msg (CPU) |
|---|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 90.97% | 0.910 | 75.5% | 98.28% | 0.17 |
| **MiniLM embeddings + logistic regression (selected)** | **92.66%** | **0.927** | 85.7% | 97.54% | 3.6 |
| MiniLM embeddings + PyTorch MLP | 92.01% | 0.920 | 82.2% | 97.75% | 3.9 |

End-to-end, with every safety rule on:

- **70.9%** of messages auto-answered, with the intent right on **97.6%** of those. The other 29.1% go to a person.
- **Fraud/security messages auto-answered by mistake: 1 of 480** (5 before the fraud safety net).
- Retrieval finds the right help article first time on **92.7%** of auto-answered messages, and in the top 3 on **98.4%**.

Full numbers, the coverage/accuracy curve, the most common mistakes and the drift demo are in [`reports/RESULTS.md`](reports/RESULTS.md).

### What the numbers honestly say

- The gate was tuned on validation for 98% accuracy on auto-answers; on the test set it delivered **97.5%**. That gap is the normal cost of choosing a threshold on 1,501 examples, and it is reported rather than hidden.
- The selected model was picked on **validation macro-F1** only. The TF-IDF baseline actually has higher accuracy on the messages it auto-answers, but answers far fewer of them.
- The one fraud miss is a real mistake: *"Will declined funds I tried to withdraw be returned to me?"* (true intent `wrong_amount_of_cash_received`), confidently read as an ordinary declined withdrawal.
- Known limitation: "can I pay with my iPhone" goes to a person, because "phone" pulls probability toward `lost_or_stolen_phone`. It fails in the safe direction, but it costs coverage.

## Design decisions

**1. A confidence gate, not a raw classifier.** A 92.7%-accurate model is still wrong for 1 in 14 customers. The gate picks the lowest confidence threshold where auto-answers hit 98% accuracy on validation, which maximises how many we can answer while keeping the error rate fixed. Everything below it goes to a person. See `triage/gate.py`.

**2. Fraud never gets an automated reply.** Twelve fraud, security and vulnerability intents (stolen card, unrecognised payment, compromised details and so on) always go to a person. On top of that there's a **safety net**: if the model's *total* probability across those intents is at least 5%, the message goes to a person even when the top guess looks harmless. The 5% cutoff was the largest value that let zero fraud messages through on validation. It cut test misses from 5 to 1 for 1.4 points of coverage.

**3. Grounded replies that fail closed.** Retrieval is cosine similarity over MiniLM embeddings, nudged toward the article mapped to the predicted intent. With an `ANTHROPIC_API_KEY` set, Claude drafts the reply as structured JSON (`answer`, `citations`, `handoff`) using only the retrieved articles. Without one, the best article is returned verbatim. Either way, a draft that cites nothing, cites an article that wasn't retrieved, asks for a handoff, is refused or truncated, or errors is **not sent**, and the message goes to a person.

**4. Drift monitoring without labels.** `/drift` takes a batch of recent messages and compares the predicted-intent mix to the reference using PSI, plus the share falling below the gate. My first version used the textbook fixed thresholds (0.1 / 0.25), and it **raised a false alarm on unchanged traffic**: with 77 intents and ~750 messages, chance alone produces PSI around 0.25. Thresholds are now the 95th/99th percentile PSI between two same-distribution samples of the actual sizes, simulated per request. Unchanged traffic now reads `ok`, and a simulated card-incident spike reads `alert` (PSI 1.77 against a chance level of 0.19).

**5. Selection discipline.** Model choice, gate threshold and fraud cutoff all use a stratified validation slice of the official train split. The official test split is only used for the final report.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m triage.train            # trains 3 models, logs to MLflow, writes models/best + reports/
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs
uvicorn triage.api:app --port 8000                     # serve
pytest -q                          # 24 tests, no downloads or API calls needed
```

Docker (the embedding model is baked into the image, so it starts offline):

```bash
docker build -t banking-support-triage .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... banking-support-triage   # key optional
```

### API

| Endpoint | What it does |
|---|---|
| `POST /triage` `{"text": "..."}` | intent, confidence, top-3, fraud risk, decision + reason, retrieved articles, reply + citations |
| `POST /drift` `{"texts": [...]}` (20–5,000) | PSI, calibrated warn/alert lines, below-gate rate, biggest intent moves |
| `GET /model` | model card: kind, params, threshold, metrics, training time |
| `GET /health` | status, model, whether LLM drafting is on |

```json
POST /triage {"text": "Why was I charged a fee for taking out cash?"}
{
  "intent": "cash_withdrawal_charge", "confidence": 0.9956, "fraud_risk": 0.0004,
  "decision": "auto_answer", "reason": "confidence 1.00 clears the gate (0.71)",
  "reply": "Card payments in pounds are free. Cash withdrawals are free up to 200 pounds per month; after that a 2% fee applies...",
  "citations": ["KB-11"], "latency_ms": 22.8
}
```

## Layout

```
triage/data.py      Banking77 loading + stratified validation split
triage/models.py    TF-IDF+LR, embeddings+LR, embeddings+PyTorch MLP behind one interface
triage/train.py     training, MLflow tracking, selection, gate + fraud cutoff, evaluation, report
triage/gate.py      confidence gate, fraud safety net, routing
triage/kb.py        knowledge base + vector search
triage/answer.py    Claude structured-output drafting, extractive fallback, grounding check
triage/drift.py     PSI drift monitor with sample-size-calibrated thresholds
triage/api.py       FastAPI service
kb/articles.json    22 help articles covering all 77 intents (fictional bank, illustrative policy)
```

## Limits and next steps

- The help articles are written for this demo and describe a fictional bank. Fees and limits are illustrative.
- The Claude drafting path is covered by unit tests using a fake client (good drafts, bad citations, refusals, API errors). The reported metrics come from the extractive path, which needs no key. The next step is an LLM eval set: real questions, checking that each reply is faithful to its cited article.
- Next: fine-tune the embedding model on Banking77 (a contrastive objective typically adds a few points), calibrate probabilities (temperature scaling) before gating, and log live triage decisions for a weekly drift and accuracy review.

## Credits

Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, 2020, PolyAI, CC-BY-4.0. Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
