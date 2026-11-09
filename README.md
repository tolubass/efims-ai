# EFIMS AI

The AI layer of an Enterprise Fraud and Incident Management System (EFIMS)
for Nigerian banks. It is a FastAPI service that wraps three trained models
— incident **severity** classification, transaction **anomaly** scoring, and
fraud-loss **recovery** propensity — behind a consistent, audited API.

**Every prediction is advisory.** No endpoint in this service approves,
blocks, or executes anything. Nigerian banking AI practice requires a human
to review and approve any action taken on a case; this service exists to
inform that reviewer, not to replace them. Every response carries a
`disclaimer` field saying so, and low-confidence predictions are explicitly
flagged with `requires_human_review: true` (see **Human-review thresholds**
below).

Built entirely on **synthetic data** — see **Known limitations**.

## Endpoints

| Method | Path | Request schema | Response schema | Purpose |
|---|---|---|---|---|
| GET | `/` | — | — | Service info, links to docs and health |
| GET | `/api/v1/health` | — | — | Per-model load status; `healthy` only if all three models loaded |
| POST | `/api/v1/severity/predict` | `SeverityRequest` | `SeverityResponse` | Classify an incident as S1 (most critical) – S4 |
| POST | `/api/v1/anomaly/score` | `AnomalyRequest` | `AnomalyResponse` | Score a transaction 0–1 for anomalous behaviour (higher = more suspicious) |
| POST | `/api/v1/recovery/predict` | `RecoveryRequest` | `RecoveryResponse` | Predict the probability a confirmed-loss case recovers >50% of its gross loss |

Interactive docs (request/response schemas, "Try it out"): `/docs` once the
service is running. Full request/response examples, captured from a real
run of this service, are in [`data/samples/`](data/samples/).

**Not built yet**: duplicate-incident detection, entity extraction (NLP),
next-action recommendation, and summarisation. Their schemas
(`src/api/schemas/duplicate_schema.py` etc.) and config scaffolding exist,
but there are no trainers, no inference services, and no routes for them —
see **Known limitations**.

## Running locally

Requires Python 3.14 (the venv this project was built and tested against).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs`, or check
`http://localhost:8000/api/v1/health`.

`requirements.txt` lists only what's needed to **run** the API — the exact
set of packages the process actually imports when it starts up and serves a
request, verified by running every endpoint once and recording what got
loaded (see the comments in that file, which also explain two non-obvious
inclusions: `pyarrow`, needed for pandas's string-handling to behave as
tested even though pandas doesn't require it; and `mlflow`, needed only
because unpickling a saved model indirectly re-imports its trainer module,
which imports `mlflow` at the top — not because inference calls it).

Training scripts (`run_severity_training()` etc.) and the governance
reports script (`run_governance_reports.py`) need more than this —
`lightgbm`, `scikit-learn`, and `mlflow` are already covered above, but
re-running training end to end also expects the raw CSVs under
`data/synthetic/` to be present (see `configs/models/*.yaml`).

## Running with Docker

```bash
cd docker
docker compose up --build
```

This builds from `python:3.14-slim` (matching the dev venv's Python exactly)
and serves on `http://localhost:8000`. Audit logs under `logs/` are mounted
as a volume, so they persist across container restarts and rebuilds instead
of disappearing with the container.

> **Not build-tested.** Docker is installed on the machine this was built
> on, but its daemon wasn't reachable, so `docker/Dockerfile` and
> `docker/docker-compose.yml` were checked for correctness (`docker compose
> config` parses and resolves them cleanly; the base image tag was confirmed
> to exist on Docker Hub) but never actually built or run. **Build and
> smoke-test both files before relying on them** — see `HANDOFF.md` for
> what to check first.

### Where `model_registry/` must be placed

Both the Dockerfile and the local-run instructions above assume
`model_registry/` already exists at the project root, containing each
model's champion version:

```
model_registry/
  severity/champion.txt          # currently: "v1.0"
  severity/v1.0/model.pkl
  anomaly/champion.txt
  anomaly/v1.0/model.pkl
  recovery/champion.txt
  recovery/v1.0/model.pkl
```

The API loads these once at startup (`src/api/lifespan.py`) and never
retrains. If `model_registry/` is missing or a `model.pkl` is absent, that
model's service reports `is_loaded: false` — `/api/v1/health` returns
`"status": "degraded"` and that model's endpoint returns `503` — but the
service as a whole still starts and serves whichever models *did* load.

## How a request flows

Using `POST /api/v1/severity/predict` as the example (anomaly and recovery
follow the same shape):

1. **Validation** — FastAPI validates the JSON body against `SeverityRequest`
   (`src/api/schemas/severity_schema.py`). A malformed or out-of-range field
   (e.g. a negative `financial_exposure_ngn`) is rejected with `422` before
   any model code runs.
2. **Prediction** — `SeverityService.predict()` (`src/inference/severity_service.py`)
   builds the exact feature row the model was trained on and calls the
   loaded LightGBM classifier.
3. **SHAP explanation** — the top 3 features behind that specific prediction
   are extracted via `shap.TreeExplainer` and returned as `top_factors` /
   `explanation`. If SHAP fails for any reason, the response still succeeds
   with an empty explanation list rather than erroring out.
4. **Human-review flag** — `requires_human_review()`
   (`src/governance/human_review_flag.py`) compares the prediction's
   confidence against both the model's own threshold and a global floor
   (see table below) and sets `requires_human_review` accordingly.
5. **Audit log** — `log_prediction()`
   (`src/governance/prediction_audit_logger.py`) appends one JSON line to
   `logs/audit_YYYYMMDD.jsonl` with a masked input summary (phone numbers,
   account-like numbers, and emails are redacted; narratives are never
   stored raw) — this never raises, so a logging failure can't break a
   prediction.
6. **Response** — a `SeverityResponse` is returned, always including the
   `disclaimer` field.

## Human-review thresholds

From `configs/base/thresholds.yaml`. A prediction is flagged for mandatory
human review if its confidence is below **either** its model's own
threshold **or** the global floor, whichever is stricter.

| Model | Threshold | What it means |
|---|---|---|
| Global floor | 0.60 | Applies on top of every model-specific threshold below |
| severity | 0.70 | Below 70% confidence, a triage officer must confirm the severity before it appears on the case |
| anomaly | 0.75 *(score, not confidence — higher is worse)* | Transactions scoring ≥0.75 are surfaced to fraud analysts |
| recovery | 0.65 | Below 65% confidence, recovery decisions must be reviewed (legal/financial commitments) |
| duplicate | 0.85 | Not yet wired to an endpoint — see Known limitations |
| nlp | 0.60 | Not yet wired to an endpoint — see Known limitations |

`auto_action_threshold` (0.90) exists in config for a future "high
confidence" UI badge; it does **not** mean any action is automatic — the
disclaimer and human-review requirement apply regardless.

## Known limitations

Stated plainly, because they materially affect how much weight to put on
any single prediction from this service:

- **All three trained models were trained and evaluated on synthetic data
  only.** Nothing here has been validated against real bank data.
- **The severity model has only 3 S1 (most critical) examples** in the
  entire dataset. Its S1 predictions should not be trusted on this evidence
  alone, and its validation split (14 rows) is too small for any metric
  from it to be reliable.
- **The recovery model's cross-validated ROC-AUC is 0.627, on only 31
  labelled cases** — only a little better than chance, and too small a
  sample to be confident in that number itself.
- **Fairness cannot be meaningfully judged on the severity validation
  set.** Slicing it by gender/age group/customer segment/state leaves most
  groups below the minimum size needed to compute a reliable metric at all
  (18 of the slices in the current fairness report are marked
  `insufficient_data`).
- **The anomaly model's fairness report flags a disparity on every group
  column it checked** (gender, age group, customer segment, state), with
  no determined cause. This has not been investigated further — it may
  reflect a real issue, or it may be an artifact of how synthetic fraud was
  generated relative to those fields. Treat it as an open question for a
  human reviewer, not a conclusion either way.
- **Duplicate-incident detection, entity extraction (NLP), next-action
  recommendation, and summarisation are not built.** Their request/response
  schemas exist, but there is no trainer, no inference service, and no API
  route for any of them.

See `model_registry/{severity,anomaly,recovery}/v1.0/model_card.md` and
`fairness_report.json` for the full detail behind each of these.
