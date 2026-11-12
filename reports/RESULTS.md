# Results

Dataset: Banking77 (PolyAI, CC-BY-4.0): 8502 train / 1501 validation / 3080 test, 77 intents. Model and gate chosen on validation; test scored once.

## Model comparison (official test split)

| Model | Test accuracy | Test macro-F1 | Gate threshold | Auto-answered | Accuracy when auto-answered | ms / message (CPU, batch 32) |
|---|---|---|---|---|---|---|
| `tfidf_logreg` | 90.97% | 0.9100 | 0.70 | 75.5% | 98.28% | 0.21 |
| `emb_logreg` **(selected)** | 92.66% | 0.9265 | 0.71 | 85.7% | 97.54% | 3.80 |
| `emb_mlp` | 92.01% | 0.9201 | 0.85 | 82.2% | 97.75% | 4.09 |

## End-to-end pipeline (test, target 98% accuracy on auto-answers)

- Auto-answered: **70.9%** of messages, intent correct on **97.57%** of those
- Sent to a person: 29.1%
- Fraud/security messages in test: 480. Auto-answered by mistake: **1** (was 5 before the fraud-probability safety net, cutoff 5% chosen on validation)
  - Still missed: "Will declined funds I tried to withdraw be returned to me?" (true `wrong_amount_of_cash_received`, predicted `declined_cash_withdrawal` at 96%, fraud risk 1%)
- Retrieval on auto-answered messages: hit@1 **92.7%**, hit@3 98.4%

## Coverage vs accuracy (selected model, test)

| Threshold | Auto-answered | Accuracy on those |
|---|---|---|
| 0.00 | 100.0% | 92.66% |
| 0.30 | 98.8% | 93.49% |
| 0.50 | 93.8% | 95.71% |
| 0.60 | 90.2% | 96.87% |
| 0.70 | 86.0% | 97.47% |
| 0.80 | 80.7% | 98.43% |
| 0.90 | 69.6% | 99.16% |
| 0.95 | 55.6% | 99.59% |

## Most common mistakes (test)

| True intent | Predicted | Count |
|---|---|---|
| pending_transfer | transfer_timing | 5 |
| pending_transfer | transfer_not_received_by_recipient | 5 |
| card_payment_not_recognised | compromised_card | 5 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | 5 |
| declined_transfer | declined_card_payment | 4 |
| why_verify_identity | verify_my_identity | 4 |
| card_arrival | card_delivery_estimate | 4 |
| Refund_not_showing_up | reverted_card_payment? | 3 |

## Drift check demo

- Like-for-like (validation half A as reference, half B as live traffic): status **ok**, PSI 0.2516 (alert line 0.3111). With the original fixed 0.25 threshold this unchanged traffic raised a false alarm; thresholds are now calibrated to sample size.
- 1,000 messages from the official test split, which is class-balanced unlike the training traffic: status **ok**, PSI 0.1482 (alert line at this sample size: 0.192). A mild mix change like this is within chance at 1,000 messages, so no alarm.
- 1,000 messages during a simulated card-incident spike: status **alert**, PSI 1.77; alerts: intent mix has shifted (PSI 1.77 >= 0.19, the 99th percentile of chance at this sample size)
