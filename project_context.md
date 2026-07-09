# project_context.md — Living Project State

## Status
Phase: Phase 2 complete — data ingestion + feature engineering merged to develop
Last completed: feature/data-ingestion (src/features/ built, 111 tests passing)
Next task: Phase 3 — model training on Vertex AI (XGBoost + LightGBM jobs, experiment tracking)

## Completed tasks
- [x] TASK 1 — CLAUDE.md written (agent instructions, branching, commit convention)
- [x] TASK 2 — project_context.md created (living state + decisions log)
- [x] TASK 3 — README.md full professional write (architecture, cost breakdown, A/B, SHAP, CI/CD)
- [x] TASK 4 — repository structure scaffolded (src/, tests/, notebooks/, infrastructure/, .github/workflows/, data/sample/) + pyproject.toml, .env.example, .gitignore
- [x] TASK 5 — feature/repo-scaffold committed, pushed, merged --no-ff into develop
- [x] PHASE 2 — src/features/: config, schema, transforms, bigquery, feature_store, sample_data (+ 111 tests)

## Decisions log
| Decision | Rationale |
|---|---|
| XGBoost vs LightGBM as A/B variants | Both are tree-based so confounders are minimised; candidate already has strong existing work with both on Rossmann project |
| Cloud Run over Vertex AI Prediction | Cloud Run gives more control over FastAPI request/response schema and is cheaper at low-moderate traffic |
| BigQuery as offline store | Already in GCP free tier; integrates natively with Vertex AI Feature Store |
| SHAP for explainability | Consistent with existing portfolio (P1 Rossmann) — deepens the story rather than introducing a new library |
| Traffic splitting in Cloud Run | Native GCP feature, zero extra infrastructure for A/B test |
| schema.py is the single source of truth, free of GCP imports | BigQuery schema, Feature Store value types, and transform outputs all derive from `FEATURE_SPECS`. A contract test asserts they agree, so a rename cannot silently cause train/serve skew |
| All features are causal (expanding mean over `shift(1)`) | A customer's spending baseline must exclude the transaction being scored. Otherwise offline AUC is inflated and the online Feature Store cannot reproduce the value |
| Only customer-level features go in the online store | Row-local features (hour, is_foreign, amount_log) are computed from the request payload at serving time. Storing them online would be wasteful and wrong — they change per transaction |
| GCP clients are injected, never constructed implicitly | Makes the credential boundary explicit and lets every BigQuery/Vertex function be unit-tested against a fake with no network |
| Date bounds are bound query parameters, not f-string interpolation | Prevents SQL injection and forces callers through timestamp validation |
| Python pinned to 3.11 via `.python-version` | `requires-python = ">=3.11"` alone let uv resolve 3.13, which changes BigQuery SDK behaviour. Discovered via a test failure |

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
Phase 2 finished. `develop` carries `src/features/` end to end: `config.py` (env-driven, rejects
unfilled `.env.example` placeholders), `schema.py` (the shared contract), `transforms.py` (causal
feature engineering), `bigquery.py` (offline store + parameterised training-set query),
`feature_store.py` (Vertex AI online store), and `sample_data.py` (deterministic synthetic data).
111 tests pass; `ruff check` and `ruff format --check` are clean.

Nothing has been merged to `main` yet — that waits for the first deployable milestone (rule 6).

**No GCP resource has actually been provisioned.** Every cloud call is unit-tested against a fake
client; none has been executed against a live project. The first real `gcloud` interaction happens
in Phase 3. Before then the user must fill in `.env` and run `gcloud auth application-default login`.
`ensure_dataset` / `ensure_table` / `create_feature_store` are therefore *written but unexercised*
against the real API — expect small signature corrections on first live run.

Sample data lives at `data/sample/transactions_sample.csv` (1,200 rows, 1.92% fraud). Regenerate
with `uv run python -m src.features.sample_data`. A test fails if the generator changes and the CSV
is not regenerated.

Phase 3 starts with:
`git checkout develop && git pull && git checkout -b feature/model-training`
