# EFIMS AI — Handoff (v1.0)

One page: what v1.0 actually delivers, what's deliberately deferred, and
what must be resolved before this touches production traffic.

## Delivered in v1.0

- **Three working models behind a live API**: severity (LightGBM
  multiclass), anomaly (Isolation Forest), recovery (LightGBM binary).
  Trained, evaluated, saved to `model_registry/`, served through
  `POST /api/v1/{severity,anomaly,recovery}/...`.
- **A full request pipeline per prediction**: pydantic validation → feature
  construction → prediction → SHAP explanation → human-review threshold
  check → audit log entry → typed response. See README's "How a request
  flows".
- **Governance artefacts for all three models**: `model_card.md` (purpose,
  training data, performance from `metrics.json`, limitations, human
  oversight) and a `fairness_report.json` for severity and anomaly
  (recovery's sample is too small — see its model card).
- **An append-only, PII-masked audit trail** (`logs/audit_YYYYMMDD.jsonl`)
  that never raises, so a logging failure can't take down a prediction.
- **A developer handoff package** (this file, the README, Docker files,
  trimmed `requirements.txt`, and real request/response samples in
  `data/samples/`).

## Deferred to v1.1

Not started — schemas and config scaffolding exist for some of these, but
no trainer, inference service, or route:

- **Duplicate-incident detection** (`src/api/schemas/duplicate_schema.py`
  exists; nothing behind it).
- **Entity extraction (NLP)** — narrative parsing for amounts, accounts,
  dates, etc.
- **Next-action recommendation.**
- **Summarisation.**
- **More training data for severity and recovery.** Both are usable for
  demos, neither for decisions, purely because of sample size (3 S1
  examples; 31 recovery cases) — this isn't a modelling problem, it's a
  data problem, and no amount of further model tuning fixes it.
- **Investigating the anomaly model's fairness flags.** All four group
  columns were flagged with no determined cause (see README). Someone
  needs to look at whether this is a real disparity or an artifact of how
  synthetic fraud was generated, before this model is trusted on real
  transactions.

## Open items before production

In roughly the order they'd bite:

1. **Restrict CORS.** `src/api/main.py` currently sets
   `allow_origins=["*"]`, explicitly flagged in that file's own comment as
   development-only. Anyone, from anywhere, can currently call this API
   cross-origin. Replace with the actual list of origins allowed to call
   this service before it's reachable from outside your own machine.

2. **Add API-key (or equivalent) authentication.** There is currently none.
   Every endpoint — including `/api/v1/severity/predict`,
   `/api/v1/anomaly/score`, `/api/v1/recovery/predict` — is open to anyone
   who can reach the port. `BaseRequest.requester_id` already exists in the
   schema and flows through to the audit log, but nothing currently
   enforces or verifies it; it's an unauthenticated free-text field, not
   an identity.

3. **FIXED.** Category mappings for severity, anomaly and recovery are now
   persisted as `model_registry/{model}/v1.0/category_mappings.json`
   (generated once from the exact training-time join — for recovery, this
   also corrected a real bug: the old code rebuilt the mapping from the
   full 70-case table, which doesn't match the 31-row subset the model was
   actually trained on, silently shifting `classification_encoded`'s
   integer codes). `SeverityService`/`AnomalyService`/`RecoveryService` now
   load this file in `load_model()` instead of rebuilding from
   `data/synthetic/*.csv` at startup; a missing/invalid file now fails that
   model loudly (`/api/v1/health` reports `model_not_loaded`, that
   endpoint 503s) instead of silently falling back to `-1` for every
   category. Verified by running the full test suite with `data/synthetic/`
   moved aside entirely.

4. **FIXED.** `import mlflow` moved from module level in
   `src/models/{severity,anomaly,recovery}/trainer.py` into the one
   function in each that actually calls it (`_log_to_mlflow`, training-time
   only). `mlflow` is no longer in `requirements.txt`. Verified by blocking
   `mlflow` from being importable at all (`sys.modules["mlflow"] = None`)
   and separately by installing only the updated `requirements.txt` into a
   brand-new venv (confirmed `mlflow` absent via `pip list`) and running a
   real `uvicorn` server against it — all three models loaded and a real
   prediction returned 200.

5. **Build-test and smoke-test the Docker image on a machine with a
   reachable Docker daemon.** Covered above in the README; repeating it
   here because it's a hard prerequisite for everything else — nothing in
   `docker/` has actually been built or run yet, only checked for
   structural correctness.

6. **Decide what to do about duplicate/NLP/next-action/summarisation**
   before anyone outside this handoff assumes they exist — the schemas are
   present and might look "ready" to someone skimming `src/api/schemas/`.
