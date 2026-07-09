# project_context.md — Living Project State

## Status
Phase: Phase 1 complete — repository scaffold + documentation merged to develop
Last completed: TASK 5 (feature/repo-scaffold merged into develop, pushed to origin)
Next task: Phase 2 — data ingestion + feature engineering (src/features/, BigQuery schema, Feature Store entity)

## Completed tasks
- [x] TASK 1 — CLAUDE.md written (agent instructions, branching, commit convention)
- [x] TASK 2 — project_context.md created (living state + decisions log)
- [x] TASK 3 — README.md full professional write (architecture, cost breakdown, A/B, SHAP, CI/CD)
- [x] TASK 4 — repository structure scaffolded (src/, tests/, notebooks/, infrastructure/, .github/workflows/, data/sample/) + pyproject.toml, .env.example, .gitignore
- [x] TASK 5 — feature/repo-scaffold committed, pushed, merged --no-ff into develop

## Decisions log
| Decision | Rationale |
|---|---|
| XGBoost vs LightGBM as A/B variants | Both are tree-based so confounders are minimised; candidate already has strong existing work with both on Rossmann project |
| Cloud Run over Vertex AI Prediction | Cloud Run gives more control over FastAPI request/response schema and is cheaper at low-moderate traffic |
| BigQuery as offline store | Already in GCP free tier; integrates natively with Vertex AI Feature Store |
| SHAP for explainability | Consistent with existing portfolio (P1 Rossmann) — deepens the story rather than introducing a new library |
| Traffic splitting in Cloud Run | Native GCP feature, zero extra infrastructure for A/B test |

## Environment
- Python 3.11
- GCP SDK: gcloud CLI (must be installed and authenticated before src/ code runs)
- Local dev: venv or uv
- Package manager: uv (preferred, consistent with candidate's other projects)

## Open questions / blockers
- GCP project ID: to be set in .env by user
- Cloud Run region: europe-west2 (London) preferred
- Vertex AI Feature Store: use managed (not optimised) for cost control on free-tier/trial

## Session handoff notes
Phase 1 finished. `develop` now carries the full scaffold and documentation; `feature/repo-scaffold`
is merged and can be deleted. Nothing has been merged to `main` yet — that waits for the first
deployable milestone (per working rule 6).

No dependencies are pinned yet: `pyproject.toml` has an empty `dependencies = []` list by design,
filled in module by module. `uv sync` therefore currently installs nothing but the project itself.
There are no tests yet, so `pytest` has nothing to run — the first real test arrives with Phase 2.

Stopped here, awaiting go-ahead for Phase 2. Phase 2 starts with:
`git checkout develop && git pull && git checkout -b feature/data-ingestion`
