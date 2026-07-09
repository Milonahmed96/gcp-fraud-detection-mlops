# project_context.md — Living Project State

## Status
Phase: Phase 3 complete — model training + A/B comparison merged to develop
Last completed: feature/model-training (src/training/ built, 240 tests passing)
Next task: Phase 4 — SHAP explainability module (src/evaluation/, logged to Vertex AI Experiments)

## Completed tasks
- [x] TASK 1 — CLAUDE.md written (agent instructions, branching, commit convention)
- [x] TASK 2 — project_context.md created (living state + decisions log)
- [x] TASK 3 — README.md full professional write (architecture, cost breakdown, A/B, SHAP, CI/CD)
- [x] TASK 4 — repository structure scaffolded (src/, tests/, notebooks/, infrastructure/, .github/workflows/, data/sample/) + pyproject.toml, .env.example, .gitignore
- [x] TASK 5 — feature/repo-scaffold committed, pushed, merged --no-ff into develop
- [x] PHASE 2 — src/features/: config, schema, transforms, bigquery, feature_store, sample_data (+ 111 tests)
- [x] PHASE 3 — src/training/: metrics, dataset, models, train, vertex, experiments (+ 129 tests)

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
| Training runs behind `--backend local\|vertex` | Same fitting code both ways; `vertex` ships `train.py` to a managed machine and runs it with `--backend local`. CI and pytest use `local` (free, no GCP); the portfolio demo uses `vertex` |
| Business cost, not AUC/F1, decides the A/B winner | A missed fraud costs its transaction amount; a blocked genuine customer costs a flat ~£5. F1 implicitly prices them equally. On the current sample LightGBM wins F1 and loses on cost |
| Winner reported with a bootstrap CI; ties keep the incumbent | The current cost delta is `[-143.96, +29.17]`, straddling zero. Shipping the point-estimate "winner" would be shipping a coin flip |
| Decision threshold fitted on validation, never test | Tuning the threshold on test leaks and flatters every variant. A test asserts refitting on test can only lower cost — if it doesn't, the threshold leaked |
| Temporal split, never random | Fraud is non-stationary. A random split lets the model see next month's fraud ring while training on this month's |
| Sample data has a 35% "stealth fraud" cohort with no injected signal | Without it both variants scored ROC-AUC 1.0 and the A/B test was degenerate. Stealth fraud sets an honest recall ceiling, as it does in production |
| Experiment logging catches all exceptions | A model that trained successfully but could not be logged is still a model. Tracking must never fail a training job |

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
Phase 3 finished. `develop` carries `src/training/`: `metrics.py` (business cost + bootstrap CI),
`dataset.py` (temporal split, local/BigQuery sources), `models.py` (the two variants, capacity
matched), `train.py` (orchestrator + CLI), `vertex.py` (Custom Training submission), and
`experiments.py` (Vertex AI Experiments logging). 240 tests pass; ruff is clean.

Current local A/B result on the sample: XGBoost cost/1k = 984.69, LightGBM = 1017.40, delta
`-32.71 [-143.96, +29.17]` — **not significant**, so the incumbent (XGBoost) is kept. LightGBM wins
F1 (0.452 vs 0.444) and loses on cost, which is the point of the business metric.

Nothing has been merged to `main` yet — that waits for the first deployable milestone (rule 6).

**Still no GCP resource has been provisioned.** Both `--backend vertex` and the BigQuery source are
unit-tested against fakes only. `submit_training_job`, `create_feature_store`, `ensure_dataset` and
`log_training_run` have never run against the live API — expect small signature corrections on the
first real invocation. Before that: fill in `.env` and run `gcloud auth application-default login`.

Sample data is now 6,000 rows at 2.13% fraud, with a 35% stealth-fraud cohort. Regenerate with
`uv run python -m src.features.sample_data`; a test fails if the generator changes and the CSV does
not. Model artefacts land in `artifacts/` (gitignored).

Phase 4 starts with:
`git checkout develop && git pull && git checkout -b feature/shap-explainability`
It will build `src/evaluation/`: `shap.TreeExplainer` over both variants (both are tree ensembles,
which is why TreeExplainer works for each), global importance, per-prediction attributions, and
attachment of SHAP artefacts to the Vertex AI Experiments runs created in Phase 3.
