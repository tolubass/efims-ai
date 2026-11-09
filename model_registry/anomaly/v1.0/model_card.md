# Model Card: anomaly (v1.0)

Generated 2026-10-30 22:02:58 WAT

## Purpose

Scores transactions for anomalous behaviour (0-1; higher means more suspicious).

## Training data

Synthetic transactions from data/synthetic/08_transactions.csv, cleaned via src.preprocessing.cleaning.clean_transactions and featurised via src.features.transaction_features.

## Performance

| Metric | Value |
|---|---|
| roc_auc | 0.9921893848009651 |
| precision_at_k | 0.7 |
| k_value | 50 |
| false_positive_rate | 0.0 |
| threshold_used | 0.75 |

## Fairness

Flagged columns: gender, age_group, customer_segment, state

> Small groups can produce unreliable gaps. Treat flags as prompts for human review, not proof of bias.

## Limitations

- Unsupervised model: fraud_label was used only for this evaluation, never during training.
- Evaluated against synthetically injected fraud patterns — real-world performance is unverified.
- Trained and evaluated on synthetic data only; not yet validated on real transactions.

## Human oversight

All outputs are advisory. A human must review and approve before any action is taken on a case.

## Status

Trained on SYNTHETIC data only. Not validated on real bank data. Retraining on real data is required during the pilot.
