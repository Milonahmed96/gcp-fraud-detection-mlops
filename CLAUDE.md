# CLAUDE.md — Agent Instructions

## Project
GCP MLOps Pipeline: Real-Time Fraud Detection on Vertex AI
Portfolio project for Milon Ahmed (AI Engineer, London). Targets London FinTech/banking AI Engineer roles.

## What this project does
End-to-end fraud detection ML pipeline on GCP:
- Feature engineering → BigQuery + Vertex AI Feature Store
- Model training → Vertex AI (XGBoost vs LightGBM A/B variants)
- Explainability → SHAP values logged to Vertex AI Experiments
- Inference → FastAPI on Cloud Run
- Drift monitoring → Cloud Scheduler triggers retraining pipeline
- CI/CD → GitHub Actions deploys to Cloud Run on merge to main

## Architecture decisions
- Two model variants trained (XGBoost, LightGBM) — traffic split in Cloud Run for A/B test
- SHAP returned per-prediction; global importance logged to Vertex AI Experiments
- BigQuery as the offline feature store and audit log
- Cloud Scheduler triggers the drift check on a second Cloud Run service every 24h
- **Feature Store is provisioned but NOT wired to serving.** Measured online lookup is
  ~153 ms median; the whole `/predict` handler answers in ~26 ms including SHAP. The
  serving path uses `InMemoryStateStore`. This is a deliberate call, drawn as a dashed
  edge on the architecture diagram.
- The Feature Store module targets the **deprecated legacy API** (`aiplatform.Featurestore`),
  sunset 2027-02-17. `gcloud ai feature-stores list|delete` no longer exist; use the
  Python SDK or REST.

## Cost discipline (read before provisioning anything)
- **The Feature Store node bills ~£0.90/hr whether or not anything reads it.** It is ~99% of
  this project's bill. Everything else is free-tier or scale-to-zero.
- Always create a budget alert **before** the first billable resource.
- Delete the Feature Store the moment a demo ends. Recreate: ~11 min provision + ~6 min ingest.
- Cloud Run scales to zero; BigQuery/Scheduler are free-tier; finished Vertex jobs cost nothing.

## Branching strategy
- main: production only, protected
- develop: integration branch
- feature/*: all new work starts here, merges to develop via CLI

## Commit convention
feat: new feature
fix: bug fix
chore: config, CI, dependencies
docs: documentation only
refactor: restructure without behaviour change
test: tests only

## Session continuity rules
- Always read CLAUDE.md and project_context.md at the start of every session
- Always read the current git log (last 10 commits) to understand where work stopped
- Never assume a file exists without checking with ls or cat
- Never skip tests — run pytest before every merge
- `gcloud` lives at /opt/homebrew/share/google-cloud-sdk/bin — export PATH first

## Hard-won lessons (all found by running against real GCP)
A green test suite is not working software. 604 tests, green CI and a v1.0.0 tag existed
before a single GCP resource did; the first live run found **eleven** defects.

- **BigQuery TIMESTAMP is microsecond-resolution.** Nanoseconds raise `ArrowInvalid`. Use
  `bigquery.truncate_timestamps()` on ANY frame headed for BigQuery — including the
  Feature Store, whose `ingest_from_df` stages through BigQuery.
- **Never let a missing column silently default.** `_slice` fell back to `np.zeros()` for
  `amount`, so every missed fraud cost £0 and the model learned to block nobody. ROC-AUC
  still read 0.76. Fail loudly instead.
- **A test written from the same wrong mental model as the code ratifies the bug.** Two
  tests here asserted the bugs (excluding `amount`; the broken auth mechanism).
- **Vertex `CustomTrainingJob(script_path=)` packages ONE file.** Use a container `CustomJob`
  with our own image; write artefacts to the `/gcs` FUSE mount so they survive.
- **Cloud Run rejects `--no-traffic` when creating a service.** Only applies from deploy #2.
- **Cloud Run ID token audience must be the BASE service URL**, not the tagged revision URL,
  even when calling the tagged URL. `print-identity-token --impersonate-service-account`
  does not work under Workload Identity Federation; use `google-github-actions/auth`
  with `token_format: id_token`.
- **Never print a token** into logs or transcripts. Probe with `-o /dev/null -w '%{http_code}'`.
- **Cross-building a 2.7 GB amd64 image on an ARM Mac takes >1 hr.** Use Cloud Build (~3 min).
- `python-dotenv` discovery walks up from the *calling module*, not the cwd, so a real `.env`
  breaks config-test isolation. CI never sees this.

## Key file locations
src/features/          — feature engineering + BigQuery ingestion
src/training/          — Vertex AI training jobs (XGBoost + LightGBM)
src/inference/         — FastAPI app (Cloud Run)
src/monitoring/        — drift detection + Cloud Scheduler integration
src/evaluation/        — SHAP explainability + A/B test metrics
notebooks/             — EDA + experiment notebooks (not production code)
infrastructure/        — Terraform or gcloud CLI scripts for GCP resource provisioning
.github/workflows/     — GitHub Actions CI/CD
tests/                 — pytest unit + integration tests
data/                  — local sample data only (never commit real credentials or large files)

## GCP project config
All GCP config lives in .env (never committed) and is loaded via python-dotenv.
A .env.example file with placeholder keys must always be kept up to date.

## Never do these things
- Never hardcode GCP project ID, credentials, or API keys
- Never commit .env files
- Never make a monolithic commit — one logical change per commit
- Never skip writing tests for src/ code
- Never merge to main directly — always go through develop
