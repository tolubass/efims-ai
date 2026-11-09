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

3. **Persist the severity and recovery category mappings in the model
   registry instead of rebuilding them from CSVs at startup.**
   `SeverityService` and `RecoveryService` currently reconstruct their
   categorical-feature encodings (source/channel/incident_type/
   reported_by_type for severity; classification/channel for recovery) by
   reading and re-cleaning `data/synthetic/10_incidents.csv` and
   `data/synthetic/11_cases.csv` once at service startup (see
   `_build_category_mappings()` in both files). **This will not work on
   the bank's server**, which won't have `data/synthetic/` — only
   `model_registry/`. When that happens, the mapping build fails silently
   (caught, logged as a warning), and every categorical feature falls back
   to `-1` ("unknown") for every request, degrading both models without
   throwing an error anyone would notice. Fix: save each mapping as a JSON
   artifact in `model_registry/{model}/v1.0/` at training time (the
   pattern `features.json` already uses for the feature-column list), and
   have the services load that instead of the raw CSVs.

4. **Lazy-import `mlflow` in the three trainers.**
   `src/models/{severity,anomaly,recovery}/trainer.py` each do
   `import mlflow` at module level. Because `joblib.load()` on a saved
   model re-imports its trainer module to reconstruct the pickled class,
   this means the API pulls in all of mlflow's own dependency tree (Flask,
   SQLAlchemy, matplotlib, pillow, opentelemetry, databricks-sdk,
   google-auth, ...) at startup, purely as a side effect of unpickling —
   inference never calls an mlflow function. Moving `import mlflow` inside
   each trainer's `run_*_training()` function (so it's only imported during
   training, never at inference-time unpickling) would let `mlflow` be
   dropped from the API's runtime `requirements.txt` entirely, meaningfully
   shrinking the production image. Flagged, not fixed here, since it means
   touching `src/models/*/trainer.py`, which is outside this handoff task's
   file list.

5. **Build-test and smoke-test the Docker image on a machine with a
   reachable Docker daemon.** Covered above in the README; repeating it
   here because it's a hard prerequisite for everything else — nothing in
   `docker/` has actually been built or run yet, only checked for
   structural correctness.

6. **Decide what to do about duplicate/NLP/next-action/summarisation**
   before anyone outside this handoff assumes they exist — the schemas are
   present and might look "ready" to someone skimming `src/api/schemas/`.
