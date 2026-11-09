# Model Card: recovery (v1.0)

Generated 2026-10-30 22:02:58 WAT

## Purpose

Predicts the probability that a confirmed-loss case will recover more than 50% of its gross loss, to help prioritise recovery actions.

## Training data

Synthetic recoveries, cases, losses, and incidents joined and cleaned per src.features.recovery_features.build_recovery_features.

## Performance

| Metric | Value |
|---|---|
| accuracy | 0.6129 |
| precision | 0.5 |
| recall | 0.3333 |
| f1 | 0.4 |
| roc_auc | 0.6272 |
| pr_auc | 0.5848 |
| brier_score | 0.2558 |
| cv_folds | 5 |

## Fairness

Not run: only 31 labelled cases exist, too few to slice reliably by demographic group.

## Limitations

- Cross-validated ROC-AUC is weak (0.627) — only a little better than chance; not reliable enough to act on without a human reviewing every case.
- Only 31 labelled recovery cases exist in total — far too few to slice reliably by demographic group, which is why no fairness report was generated for this model.
- Trained and evaluated on synthetic data only; not yet validated on real recoveries.

## Human oversight

All outputs are advisory. A human must review and approve before any action is taken on a case.

## Status

Trained on SYNTHETIC data only. Not validated on real bank data. Retraining on real data is required during the pilot.
