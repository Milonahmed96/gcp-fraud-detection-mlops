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
- SHAP logged per-prediction to Vertex AI Experiments for auditability
- Feature Store used for online serving (low-latency feature lookup)
- BigQuery as the offline feature store and audit log
- Cloud Scheduler triggers a drift check Lambda every 24h

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
