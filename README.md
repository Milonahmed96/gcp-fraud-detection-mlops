# GCP MLOps Pipeline — Real-Time Fraud Detection on Vertex AI

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/Milonahmed96/gcp-fraud-detection-mlops/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![Cloud Run](https://img.shields.io/badge/deploy-Cloud%20Run-4285F4)](https://cloud.google.com/run)

A production-grade, end-to-end MLOps pipeline that detects fraudulent card transactions in real time on Google Cloud. Raw transactions land in **BigQuery**, engineered features are served at low latency from the **Vertex AI Feature Store**, two model variants (**XGBoost** and **LightGBM**) are trained and versioned on **Vertex AI** with experiment tracking, and the resulting traffic split is served from a **FastAPI** service on **Cloud Run**. Every prediction carries a **SHAP** explanation logged to Vertex AI Experiments for auditability — a hard requirement in regulated financial services. A **Cloud Scheduler** job runs a daily drift check that can trigger retraining, and **GitHub Actions** deploys to Cloud Run on every merge to `main`. The two model variants run as a live **A/B test**, compared not just on AUC and F1 but on a business cost metric that prices false negatives (missed fraud) against false positives (blocked genuine customers).

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Data & Features"]
        DS[("Transaction<br/>Data Source")]
        BQ[("BigQuery<br/>Offline Store + Audit Log")]
        FS[["Vertex AI<br/>Feature Store"]]
        DS --> BQ
        BQ -->|feature engineering| FS
    end

    subgraph Train["Training"]
        VT["Vertex AI Training<br/>XGBoost + LightGBM"]
        EXP["Vertex AI Experiments<br/>metrics + SHAP"]
        MR[["Vertex AI<br/>Model Registry"]]
        VT --> EXP
        VT --> MR
    end

    subgraph Serve["Serving"]
        CR["Cloud Run<br/>FastAPI + traffic split"]
        CL(["Client / Payment Gateway"])
        CR <--> CL
    end

    subgraph Monitor["Monitoring"]
        CS["Cloud Scheduler<br/>every 24h"]
        DM["Drift Monitor<br/>src/monitoring"]
        CS --> DM
        DM -->|drift detected| VT
        DM -->|reference stats| BQ
    end

    BQ --> VT
    FS -->|online lookup| CR
    MR -->|model A / model B| CR
    CR -->|predictions + SHAP| BQ
    CR -->|per-prediction logs| EXP
```

A rendered `draw.io` XML export lives in `infrastructure/` once Phase 9 completes.

---

## Tech stack

| Component | Technology | Purpose |
|---|---|---|
| Offline feature store | BigQuery | Historical features, training sets, immutable audit log of predictions |
| Online feature store | Vertex AI Feature Store (managed) | Low-latency feature lookup at inference time |
| Training | Vertex AI Custom Training | Runs XGBoost and LightGBM jobs on managed compute |
| Experiment tracking | Vertex AI Experiments | Params, metrics, SHAP artefacts per run |
| Model registry | Vertex AI Model Registry | Versioned, promotable model artefacts |
| Inference API | FastAPI + Docker | Typed request/response schema, low-latency handler |
| Serving platform | Cloud Run | Scale-to-zero HTTP serving with native traffic splitting |
| A/B routing | Cloud Run revision traffic split | Splits live traffic between the XGBoost and LightGBM revisions |
| Explainability | SHAP (TreeExplainer) | Per-prediction feature attributions, logged for audit |
| Drift monitoring | Cloud Scheduler + custom job | Daily PSI / KS check against the training reference distribution |
| CI/CD | GitHub Actions | Lint, pytest, build image, deploy to Cloud Run on merge to `main` |
| Models | XGBoost, LightGBM | The two A/B variants |
| Language | Python 3.11 | — |
| Packaging | uv | Fast, lockfile-backed dependency resolution |

---

## Repository structure

```
.
├── CLAUDE.md               # Agent instructions + engineering conventions
├── project_context.md      # Living project state, decisions log, session handoff
├── README.md               # You are here
├── pyproject.toml          # Project metadata + dependencies (uv)
├── .env.example            # Placeholder GCP config — copy to .env, never commit .env
├── .github/
│   └── workflows/          # GitHub Actions CI/CD (lint, test, build, deploy)
├── src/
│   ├── features/           # Feature engineering + BigQuery ingestion + Feature Store writes
│   ├── training/           # Vertex AI training jobs (XGBoost + LightGBM)
│   ├── evaluation/         # SHAP explainability + A/B test metrics
│   ├── inference/          # FastAPI app served on Cloud Run
│   └── monitoring/         # Drift detection + Cloud Scheduler integration
├── tests/                  # pytest unit + integration tests, mirrors src/
├── notebooks/              # EDA + experiment notebooks (never production code)
├── infrastructure/         # Terraform / gcloud scripts for GCP resource provisioning
└── data/
    └── sample/             # Small, anonymised sample data only — no real data, no credentials
```

---

## Quickstart

### Prerequisites

- A **Google Cloud account** with billing enabled (the free trial credit is sufficient for a full demo run)
- **[gcloud CLI](https://cloud.google.com/sdk/docs/install)**, authenticated: `gcloud auth login && gcloud auth application-default login`
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
- **Python 3.11**
- **Docker** (only needed to build the inference image locally)

### Setup

```bash
# 1. Clone
git clone https://github.com/Milonahmed96/gcp-fraud-detection-mlops.git
cd gcp-fraud-detection-mlops

# 2. Configure — copy the template and fill in your own GCP values
cp .env.example .env
$EDITOR .env

# 3. Point gcloud at the same project
gcloud config set project YOUR_GCP_PROJECT_ID

# 4. Install dependencies into a managed virtualenv
uv sync
```

`.env` is gitignored and must never be committed. Every GCP identifier is read from the environment via `python-dotenv` — nothing is hardcoded.

### Required environment variables

| Key | Example | Notes |
|---|---|---|
| `GCP_PROJECT_ID` | `fraud-detection-mlops` | Your project ID, not the display name |
| `GCP_REGION` | `europe-west2` | London — keeps data residency in the UK |
| `GCP_BUCKET_NAME` | `fraud-mlops-artifacts` | GCS bucket for model artefacts and staging |
| `VERTEX_AI_ENDPOINT` | `projects/.../endpoints/...` | Populated after the first deploy |
| `BIGQUERY_DATASET` | `fraud_features` | Offline store + prediction audit log |
| `FEATURE_STORE_ID` | `fraud_online_store` | Vertex AI Feature Store instance |
| `CLOUD_RUN_SERVICE_NAME` | `fraud-inference-api` | Target service for CI/CD deploys |

### Running locally

```bash
# Run the test suite (must pass before any merge)
uv run pytest

# Train both A/B variants locally against the committed sample — no GCP needed
uv run python -m src.training.train --backend local

# Or submit the same script as a Vertex AI Custom Training job (requires .env + gcloud auth)
uv run python -m src.training.train --backend vertex --source bigquery \
    --start-date 2024-01-01 --end-date 2024-03-01

# Regenerate the synthetic sample data
uv run python -m src.features.sample_data

# Serve the inference API locally on :8080 (needs the artefacts from the step above)
uv run uvicorn src.inference.app:app --reload --port 8080

# Run the drift check against the sample, without triggering a retraining job
uv run python -m src.monitoring.monitor --source sample --dry-run

# Build and run the container exactly as Cloud Run will
docker build -t fraud-inference-api .
docker run --rm -p 8080:8080 -e SERVING_VARIANT=xgboost fraud-inference-api
```

Dependencies are split so the serving image stays lean. `uv sync` installs only what inference needs; add extras for everything else:

```bash
uv sync                            # inference only — what the Docker image installs
uv sync --extra gcp --extra dev    # + BigQuery/Vertex SDKs + pytest/ruff (what CI uses)
```

### Scoring a transaction

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "transaction_id": "txn_001", "customer_id": "c_000",
  "timestamp": "2024-03-01T03:15:00", "amount": 4800.0,
  "merchant_id": "m_012", "merchant_category": "electronics",
  "country": "RO", "customer_home_country": "GB", "card_present": false }'
```

```json
{
  "transaction_id": "txn_001", "variant": "xgboost",
  "fraud_probability": 0.9945, "threshold": 0.5039, "is_flagged": true,
  "base_value": 1.7425, "latency_ms": 8.97, "new_customer": false,
  "top_features": [
    {"feature": "amount_vs_customer_mean",  "value": 46.41,  "shap_value":  3.074, "direction": "toward_fraud"},
    {"feature": "is_foreign",               "value": 1.0,    "shap_value":  1.607, "direction": "toward_fraud"},
    {"feature": "seconds_since_prev_txn",   "value": 85936,  "shap_value": -1.463, "direction": "toward_genuine"},
    {"feature": "customer_amount_mean_prior","value": 103.44,"shap_value": -1.335, "direction": "toward_genuine"},
    {"feature": "card_not_present",         "value": 1.0,    "shap_value":  1.307, "direction": "toward_fraud"}
  ]
}
```

The same request for a domestic, card-present, £42 afternoon purchase scores `0.0011` and is not flagged. Handler latency is **6–13 ms including the SHAP explanation** — measured locally and in the container.

Note the threshold is `0.5039`, not `0.5`. It is the cost-minimising threshold fitted on the validation split and read from `artifacts/metrics.json`; the service refuses to start rather than silently default it.

> **Note:** modules land phase by phase. Commands referencing `src/monitoring` become live from Phase 6 onward — see `project_context.md` for the current phase.

---

## GCP cost breakdown

Estimates for a **typical dev/demo run**: one training cycle per model variant, a Cloud Run endpoint serving light demo traffic, and a daily drift check, over one month in `europe-west2`.

| Service | Configuration | Estimated cost (USD/month) |
|---|---|---|
| Vertex AI Training | 2 jobs × ~20 min on `n1-standard-4` (~$0.22/hr) | **~$0.15** per full training cycle |
| Vertex AI Experiments | Metadata storage, a few hundred runs | **~$0.00** (metadata free; artefacts billed as GCS) |
| Cloud Run | Scale-to-zero, ~10k requests/mo, 1 vCPU / 512 MiB | **~$0.00–$2** (2M requests/mo are free-tier) |
| Vertex AI Feature Store (managed) | ~1 GB online storage + light read traffic | **~$1–$5** (online serving nodes dominate) |
| BigQuery | <1 GB storage, <10 GB scanned/mo | **~$0.00** (10 GB storage + 1 TB queries free/mo) |
| Cloud Scheduler | 1 job, daily | **~$0.00** (3 jobs free/mo) |
| Artifact Registry | ~1 GB of container images | **~$0.10** |
| Cloud Storage | ~1 GB of model artefacts | **~$0.02** |
| **Total** | | **≈ $2–$8 / month** |

**These are estimates**, taken from the [GCP pricing pages](https://cloud.google.com/pricing) and rounded generously. Actual cost varies with region, traffic, and how long the Feature Store online nodes stay provisioned — that is the single largest cost lever here. Tear the Feature Store down when not demoing:

```bash
gcloud ai feature-stores delete "$FEATURE_STORE_ID" --region="$GCP_REGION"
```

The whole project is designed to fit inside the GCP free trial credit.

---

## A/B testing

Both variants are trained on identical features and identical train/test splits, so the only meaningful difference between them is the learning algorithm. Cloud Run's native revision traffic splitting sends a configurable share of live requests to each — starting at 50/50 — and every prediction is written to BigQuery tagged with the serving variant, enabling honest offline comparison on real traffic.

| Variant | Model | Cloud Run revision |
|---|---|---|
| A | XGBoost | `fraud-inference-api-xgb` |
| B | LightGBM | `fraud-inference-api-lgbm` |

Metrics compared:

- **ROC-AUC** — ranking quality, robust to the extreme class imbalance typical of fraud data
- **PR-AUC** — the more honest headline metric when positives are <1% of rows
- **F1 / precision / recall at the operating threshold** — what the fraud ops team actually feels
- **Business cost metric** — the metric that decides the winner:

  ```
  cost = (false_negatives × mean_fraud_value) + (false_positives × cost_of_blocking_genuine_customer)
  ```

  A missed fraud costs the chargeback. A false positive costs a declined transaction and some goodwill. These are not symmetric, so accuracy-flavoured metrics alone pick the wrong model. The variant with the **lower expected cost per 1,000 transactions wins**, and the decision is reported with a bootstrap confidence interval rather than a bare point estimate.

- **p50 / p95 / p99 latency** — a model that wins on cost but blows the latency budget doesn't ship

### Current result (local run, synthetic sample)

Reproduce with `uv run python -m src.training.train --backend local`:

| Variant | ROC-AUC | PR-AUC | F1 | Cost / 1k txns | FP | FN |
|---|---|---|---|---|---|---|
| XGBoost | 0.754 | 0.413 | 0.444 | **984.69** | 6 | 14 |
| LightGBM | 0.729 | 0.401 | 0.452 | 1017.40 | 2 | 15 |

Two things worth noticing. **LightGBM wins on F1 but loses on cost** — precisely the disagreement the business metric exists to expose, since F1 treats a missed fraud and a blocked customer as equally bad. And the bootstrap interval for the cost difference is `[-143.96, +29.17]`, which **straddles zero**: on this test set the two variants are statistically indistinguishable. The pipeline therefore reports the result as *not significant* and keeps the incumbent rather than shipping a coin flip.

These numbers come from the synthetic sample, whose difficulty is calibrated by a deliberate stealth-fraud cohort (35% of fraud carries no distinguishing signal). That caps achievable recall, which is why ROC-AUC sits near 0.75 rather than 1.0 — an honest ceiling rather than a leaky one.

Results are published to an A/B dashboard (Phase 8).

---

## SHAP explainability

Regulated lenders must be able to explain adverse automated decisions. Every prediction returns, alongside its fraud probability, the top contributing features and their signed SHAP attributions.

- `shap.TreeExplainer` is used for both variants — exact for tree ensembles and fast enough to sit in the request path. Both A/B variants being tree ensembles is a genuine constraint on the variant choice, not a coincidence
- The explainer is built once at training time and shipped as a model artefact (`artifacts/explainer_<variant>.joblib`), so no explainer construction happens per request
- Per-prediction attributions are logged to **Vertex AI Experiments** and mirrored into **BigQuery**, giving a queryable audit trail: *why* was transaction `X` blocked on date `Y`?
- Global feature importance (mean absolute SHAP) is recomputed each training run and compared against the previous run via `importance_shift` — a large shift in what drives the model is itself a drift signal, and Phase 6's monitor watches it alongside the feature distributions

### Two things the implementation gets right

**Attributions live in log-odds space, not probability space.** SHAP's additivity guarantee is `base_value + Σ shap_values == raw margin` (the ensemble's pre-sigmoid output). Summing attributions and expecting a probability is a common, silent error. `Explanation` names the space it is in, and `verify_additivity` asserts the identity to a `1e-4` tolerance against the model's own raw margin.

**The shape of `shap_values` is not stable across shap versions or model types.** Some releases return a two-element list (one array per class) for LightGBM binary classifiers; some return `(n, n_features, 2)`; current ones return `(n, n_features)`. Taking the wrong element inverts the sign of every explanation and *nothing raises*. `normalise_shap_values` collapses all three shapes onto the positive class, and the test suite covers each.

### Example: a real explanation

For the highest-scoring true fraud in the test set (`p = 0.999`):

| Feature | Value | SHAP | Direction |
|---|---|---|---|
| `is_foreign` | 1.00 | **+2.682** | toward fraud |
| `amount_vs_customer_mean` | 4.16 | **+2.252** | toward fraud |
| `card_not_present` | 1.00 | **+1.539** | toward fraud |
| `amount_sum_24h` | 226.05 | −0.782 | toward genuine |
| `day_of_week` | 1.00 | −0.631 | toward genuine |

A foreign, card-not-present transaction at 4.16× that customer's own spending baseline. Note that exculpatory features are surfaced too — `top_contributions` ranks by *absolute* effect, because an auditor asking "why was this blocked?" needs the evidence that argued against the decision as well.

Globally, `amount_vs_customer_mean` is the strongest driver (mean |SHAP| 1.42), ahead of `amount_log` (1.08) and `amount_sum_24h` (1.08) — the model relies most on spending *relative to the customer's own baseline* rather than on raw transaction size, which is the behaviour a fraud analyst would want.

---

## Inference service

A FastAPI app on Cloud Run. **One revision per model variant**, selected by the `SERVING_VARIANT` environment variable; Cloud Run's native revision traffic splitting performs the A/B allocation, so there is no routing logic in application code and no shared mutable state between variants.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness probe. Returns 503, with the reason, if artefacts failed to load |
| `POST /predict` | Score one transaction; returns probability, decision, and SHAP top-5 |

The model, the SHAP explainer, and the trained threshold load once at startup. A startup failure is *recorded*, not raised — the process stays up so `/health` can tell an operator why it is unhealthy, rather than handing Cloud Run a crash loop.

### Defeating train/serve skew

This is the hard part of real-time ML, and it gets its own module (`src/inference/state.py`) and its own test file (`tests/inference/test_skew.py`).

Offline, features are engineered over a whole history with pandas rolling windows. Online, there is one transaction and a Feature Store lookup. These are two implementations of the same function, and if they disagree the model receives inputs it never trained on — silently, with offline metrics still looking healthy.

The naive design, storing precomputed window aggregates in the Feature Store, does not work: a trailing 24h window depends on *when you ask*, and the stored value was computed when the customer last transacted. It goes stale by exactly the inter-transaction gap, which is precisely when velocity matters. So the online store holds the customer's **recent event log**, and the windows are computed against the incoming transaction's own timestamp.

`test_skew.py` replays 150 sample transactions through both paths and asserts every one of the 13 features matches exactly, including dtype. Two real bugs surfaced this way:

- **Microsecond truncation.** `pd.Timedelta.total_seconds()` inherits `datetime.timedelta`'s microsecond resolution, so `seconds_since_prev_txn` came out as `220.472044` where training saw `220.472044709`. Fixed by dividing nanoseconds directly.
- **A half-open window.** pandas rolling uses `(t − w, t]`, so an event *exactly* one hour old is excluded. The obvious `<=` would have been wrong on exactly the high-velocity transactions the feature exists to catch.

The customer state lookup is also causal: `lookup(customer_id, as_of=event_time)` never returns the transaction being scored, nor anything after it — which protects against a duplicated or late-arriving write in production, not just historical replay.

### Container

Multi-stage build on `python:3.11-slim`, running as a non-root user. `libgomp1` is installed explicitly — both XGBoost and LightGBM link against OpenMP, and without it `import lightgbm` fails at startup with an opaque loader error.

Because every GCP call in this repo is lazy-imported, the BigQuery and Vertex AI SDKs live in an optional `gcp` extra and never enter the serving image, cutting it from **2.67 GB to 2.16 GB**. Cloud Run charges cold-start time against image size.

---

## Drift monitoring

Fraud labels arrive weeks late — a chargeback is not instant — so production AUC is not observable in time to act on. Drift detection is therefore **unsupervised**: it compares the distribution of features the model is *currently seeing* against the distribution it was *trained on*.

Two independent signals, answering different questions:

| Signal | Question | Gates retraining? |
|---|---|---|
| **PSI** per feature | Has the input distribution moved? | **Yes** — any feature ≥ 0.25 |
| **KS** two-sample | Same, but sensitive to shifts coarse bins hide | No — its p-value collapses with sample size |
| **SHAP `importance_shift`** | Has *what drives the model* changed? | No — reported for a human |

Drift is `any(feature significant)`, not a mean. One feature moving hard is exactly the fraud-ring signature; averaging it across thirteen stable features would hide it.

### Binning is where PSI implementations quietly break

`is_night` is a bool. `txn_count_1h` is almost always 1. Hand either to a quantile binner and you get duplicate edges, empty bins, and a `log(0)` that surfaces as `inf` or `nan` rather than an error — a monitor that silently never fires. So:

- Features with ≤ 10 distinct values (and all bools) are treated as **categorical**, not quantile-binned.
- Outer bin edges are **infinite**, so a value beyond the training range lands in a bin instead of vanishing and deflating the PSI.
- A category never seen in training is folded into the smallest reference bin — new behaviour is drift, not an error.
- Every proportion is floored at `EPSILON` before any logarithm.

The reference profile (`artifacts/reference_profile.json`) is captured from the **training split** — the distribution the model actually learned — not from test, which it never saw.

### Verified behaviour

```bash
uv run python -m src.monitoring.monitor --source sample --dry-run
```

Against its own training distribution, the monitor is quiet — which matters more than catching drift, because a monitor that pages someone nightly gets muted:

```
stable: 0/13 features drifted significantly; worst customer_amount_mean_prior psi=0.0113 (n=6000)
explanation drift (SHAP importance shift): 0.0111
```

Simulating a fraud ring (traffic goes foreign, card-not-present, 20× the customers' baselines, at 03:00):

```
DRIFT: 6/13 features drifted significantly; worst is_foreign psi=15.4015 (n=1500)
explanation drift (SHAP importance shift): 0.2165
  is_foreign psi=15.4015 (significant)
  hour_of_day psi=12.7174 (significant)
  card_not_present psi=11.9003 (significant)
  amount_vs_customer_mean psi=6.4754 (significant)
  ...
drift detected; retraining SKIPPED (--dry-run)
```

### Deployment shape

Cloud Scheduler fires an authenticated POST at a **separate Cloud Run service** (`Dockerfile.monitoring`), not at the inference API. Two concrete reasons: the inference image has no BigQuery SDK — that is what keeps it 500 MB smaller — and a minutes-long batch job must not share a request pool or scale-to-zero policy with a 10 ms latency-critical endpoint.

```
Cloud Scheduler ──(OIDC, 02:00 daily)──▶ Cloud Run: /drift-check
                                              │
                                              ├─ PSI vs reference_profile.json
                                              ├─ SHAP importance_shift
                                              └─ if drifted → Vertex AI retraining job
```

Authentication is an **OIDC token** minted for a service account, with the audience set to the bare service URL (Cloud Run rejects a token whose audience carries the request path). No API key lives in the job definition. Provisioning via `ensure_drift_check_job` is idempotent — it creates or updates in place, so a redeploy neither fails with `AlreadyExists` nor leaves a stale schedule behind.

A retraining submission failure is logged, never raised: a 5xx would make Cloud Scheduler retry, and a retry storm here means a thundering herd of Vertex AI training jobs.

---

## CI/CD

`.github/workflows/` defines the pipeline (Phase 7):

**On pull request into `develop`:**
1. `ruff check` + `ruff format --check` — lint and format gates
2. `uv run pytest` — full unit and integration suite
3. Docker build (build-only, no push) — proves the image still assembles

**On merge to `main`:**
1. Everything above
2. Build and push the image to Artifact Registry, tagged with the commit SHA
3. `gcloud run deploy` to Cloud Run in `europe-west2`
4. Smoke-test the new revision's `/health` endpoint before shifting traffic
5. Shift traffic per the configured A/B split; roll back automatically if the smoke test fails

Authentication uses **Workload Identity Federation** — GitHub Actions assumes a GCP service account via OIDC. No service-account JSON key is ever stored in the repository or in GitHub secrets.

---

## Project status

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository scaffold + documentation | ✅ Complete |
| 2 | Data ingestion + feature engineering | ✅ Complete |
| 3 | Model training on Vertex AI | ✅ Complete |
| 4 | SHAP explainability module | ✅ Complete |
| 5 | FastAPI inference service | ✅ Complete |
| 6 | Drift monitoring | ✅ Complete |
| 7 | GitHub Actions CI/CD | ⬜ Not started |
| 8 | A/B test dashboard | ⬜ Not started |
| 9 | Final polish + `v1.0.0` tag | ⬜ Not started |

See `project_context.md` for the live state and decisions log.

---

## Engineering conventions

- **GitFlow:** `main` (protected, production) ← `develop` (integration) ← `feature/*`
- **Conventional commits:** `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- **One logical change per commit.** No monolithic commits.
- **Tests accompany every `src/` module.** `pytest` must pass before any merge.
- **No credentials in git, ever.** All config flows through `.env` → `python-dotenv`.

Full detail in [CLAUDE.md](CLAUDE.md).

---

Part of Milon Ahmed's AI Engineer portfolio. See also: [links to other portfolio projects]
