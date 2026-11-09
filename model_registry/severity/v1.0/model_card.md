# Model Card: severity (v1.0)

Generated 2026-10-30 22:02:56 WAT

## Purpose

Classifies incoming incidents as S1-S4 (most to least critical) to prioritise triage.

## Training data

Synthetic incidents from data/synthetic/10_incidents.csv, cleaned via src.preprocessing.cleaning.clean_incidents and featurised via src.features.incident_features.

## Performance

| Metric | Value |
|---|---|
| accuracy | 0.375 |
| precision | 0.125 |
| recall | 0.1667 |
| f1 | 0.1429 |
| roc_auc | 0.6108 |

## Fairness

No columns flagged.

> Small groups can produce unreliable gaps. Treat flags as prompts for human review, not proof of bias.

## Limitations

- Only 3 S1 (most critical) examples exist in the full dataset — S1 performance cannot be trusted on this evidence alone.
- The validation set is tiny (16 rows), so metrics (including this fairness report) are noisy and can shift a lot with more data.
- Trained and evaluated on synthetic data only; not yet validated on real incidents.

## Human oversight

All outputs are advisory. A human must review and approve before any action is taken on a case.

## Status

Trained on SYNTHETIC data only. Not validated on real bank data. Retraining on real data is required during the pilot.
