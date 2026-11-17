# EFIMS AI — Full-Stack Integration Handover

Everything in this document was verified against the running service in this
repository, not written from memory or assumption. See `OPENAPI.json` in this
folder for the machine-readable spec (exported directly from the running app)
and `POSTMAN_COLLECTION.json` for ready-to-run requests built from the real
samples in `data/samples/`.

## What this service does

Three trained models, wrapped behind a FastAPI service:

| Capability | Endpoint | Model |
|---|---|---|
| Incident severity classification | `POST /api/v1/severity/predict` | LightGBM multiclass |
| Transaction anomaly scoring | `POST /api/v1/anomaly/score` | Isolation Forest |
| Fraud-loss recovery propensity | `POST /api/v1/recovery/predict` | LightGBM binary |

Plus `GET /api/v1/health` (no auth required) and `GET /` (service info).

**Every prediction is advisory.** No endpoint approves, blocks, or executes
anything. Every response includes a `disclaimer` field and a
`requires_human_review` boolean — your backend should surface both, not just
the prediction value.

**Not implemented** (schemas exist in source as scaffolding, but there is no
trainer, no inference service, and no live route): duplicate-incident
detection, entity extraction, next-action recommendation, summarisation.
Routes referencing them exist in `src/api/routes/` but are **not registered**
in `src/api/main.py` — do not call them.

## Authentication

Header: `X-API-Key: <value>`

- Missing or wrong key → `401` on every prediction endpoint.
- `GET /api/v1/health` and `GET /` do not require the key.
- The key is set via the `EFIMS_API_KEY` environment variable on the server
  you deploy to — ask whoever owns the deployment for the value; it is never
  in source control.
- If `EFIMS_API_KEY` is unset on the server, every prediction endpoint
  returns `503` (fail-closed, not fail-open).

## CORS

Configured via `EFIMS_CORS_ORIGINS` (comma-separated origins) on the server.
Defaults to **no cross-origin access at all**. If your frontend calls this
API directly from the browser (not recommended — see Architecture below),
your frontend's origin must be added to that list on the server.

## Architecture (recommended)

```
EFIMS frontend → EFIMS backend → EFIMS AI API → EFIMS backend → EFIMS frontend
```

Your backend should hold the `EFIMS_API_KEY`, call this service server-to-
server, and relay results to the frontend. The AI service does not manage
users, permissions, or case state — that stays entirely in your application.

## Example: severity prediction

Request:
```json
POST /api/v1/severity/predict
X-API-Key: <your key>
Content-Type: application/json

{
  "incident_id": "INC-0000099",
  "financial_exposure_ngn": 4800000.0,
  "customer_impact_count": 3,
  "ongoing_threat": true,
  "insider_involvement": false,
  "data_compromise": false,
  "regulatory_relevance": true,
  "source": "Customer Complaint",
  "channel": "Mobile Banking",
  "incident_type": "Account Takeover",
  "reported_by_type": "Customer",
  "narrative": "Customer called to report unauthorised transfers from their account."
}
```

Response (captured from a real run — see `data/samples/severity_response.json`
for the full shape):
```json
{
  "prediction_id": "...",
  "severity_prediction": "S2",
  "confidence_score": 0.96,
  "requires_human_review": false,
  "all_class_probabilities": {"S1": 0.02, "S2": 0.96, "S3": 0.01, "S4": 0.01},
  "top_factors": [...],
  "disclaimer": "This prediction is advisory only and requires human review before action is taken."
}
```

Full request/response examples for all three endpoints, captured from real
runs, are in `data/samples/`. Full field-by-field schemas are in
`OPENAPI.json` (also browsable live at `/docs` on a running instance).

## Error handling your backend should implement

| Status | Meaning | What to do |
|---|---|---|
| `422` | Request failed validation (e.g. negative amount) | Fix the payload — this is a client error, don't retry as-is |
| `401` | Missing/wrong `X-API-Key` | Configuration error on your side — check the key |
| `503` | That model isn't loaded, or `EFIMS_API_KEY` isn't set on the server | Not a request problem — alert/retry later, don't show the user a prediction |
| `200` | Success — **still check `requires_human_review`** | Route to a human reviewer when true |

## What we still need from you

To build the exact field-mapping between your incident/case records and
this service's request fields, we need:

1. Your incident/case schema (field names and types) for the fields that
   would map to `severity`'s and `recovery`'s inputs above.
2. How your backend expects to receive and store prediction results
   (webhook callback, synchronous response, polling, etc.).
3. The origin(s) your backend/frontend will call from, so CORS can be
   configured correctly if ever needed.

Until that's confirmed, treat the field names in the request examples above
as the AI service's side of the contract — your backend is responsible for
translating your own case/incident fields into them.

## Known limitations (read before integrating)

- All three models are trained on **synthetic data only** — not validated
  against real bank data.
- Severity has only 3 S1 (most critical) examples in the whole training set;
  don't treat S1 predictions as well-evidenced.
- Recovery's cross-validated ROC-AUC is 0.627 on only 31 labelled cases —
  barely better than chance, from a very small sample.
- The anomaly model's fairness report flags a disparity on every demographic
  group column checked, with no established cause — open question, not
  resolved either way.
- None of this has been through independent model validation, a security
  review, or institutional production approval. Technically functional and
  integration-ready; not approved for consequential production decisions.

See `model_registry/{severity,anomaly,recovery}/v1.0/model_card.md` for full
detail per model.
