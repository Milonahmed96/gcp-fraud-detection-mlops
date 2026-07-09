# project_context.md — Living Project State

## Status
Phase: Phase 4 complete — SHAP explainability merged to develop
Last completed: feature/shap-explainability (src/evaluation/ built, 318 tests passing)
Next task: Phase 5 — FastAPI inference service (src/inference/, Dockerised, A/B traffic split config)

## Completed tasks
- [x] TASK 1 — CLAUDE.md written (agent instructions, branching, commit convention)
- [x] TASK 2 — project_context.md created (living state + decisions log)
- [x] TASK 3 — README.md full professional write (architecture, cost breakdown, A/B, SHAP, CI/CD)
- [x] TASK 4 — repository structure scaffolded (src/, tests/, notebooks/, infrastructure/, .github/workflows/, data/sample/) + pyproject.toml, .env.example, .gitignore
- [x] TASK 5 — feature/repo-scaffold committed, pushed, merged --no-ff into develop
- [x] PHASE 2 — src/features/: config, schema, transforms, bigquery, feature_store, sample_data (+ 111 tests)
- [x] PHASE 3 — src/training/: metrics, dataset, models, train, vertex, experiments (+ 129 tests)
- [x] PHASE 4 — src/evaluation/: explainer (SHAP), experiments (Vertex + BigQuery audit log) (+ 78 tests)

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
| SHAP output is normalised through `normalise_shap_values` | `TreeExplainer` returns `(n,f)`, `(n,f,2)`, or a 2-element per-class list depending on shap version and model type. Taking the wrong class axis silently inverts every attribution's sign. shap itself warns that the LightGBM binary output "has changed" |
| Attributions are documented as log-odds, and `verify_additivity` asserts it | `base + sum(shap) == raw margin`, not probability. Summing attributions and expecting a probability is a silent, common error |
| Explainer built at training time, persisted beside the model | Constructing a `TreeExplainer` per request puts tree traversal on the hot path. A test asserts the reloaded explainer reproduces its reloaded model's probabilities |
| `top_contributions` ranks by absolute SHAP, not signed | An auditor asking "why was this blocked?" needs the exculpatory evidence too |
| Per-prediction attributions stored as a JSON string in BigQuery | Keeps the `prediction_log` schema stable as the feature set evolves; `JSON_VALUE` keeps it queryable |
| `importance_shift` = total variation distance between normalised importance profiles | Scale-invariant, symmetric, bounded in [0,1]. A shift in *what drives the model* is a drift signal even when AUC is flat — Phase 6 consumes this |

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
Phase 4 finished. `develop` carries `src/evaluation/`: `explainer.py` (`FraudExplainer`,
`normalise_shap_values`, `importance_shift`) and `experiments.py` (SHAP importance → Vertex AI
Experiments; per-prediction attributions → BigQuery `prediction_log`). `train.py` now persists an
explainer beside each model. 318 tests pass; ruff is clean.

A/B result unchanged from Phase 3: XGBoost cost/1k = 984.69, LightGBM = 1017.40, delta
`-32.71 [-143.96, +29.17]` — **not significant**, incumbent kept.

Global SHAP importance on the sample: `amount_vs_customer_mean` (1.42) > `amount_log` (1.08) ≈
`amount_sum_24h` (1.08) > `customer_amount_mean_prior` (1.03) > `card_not_present` (0.99). The model
leans on spend *relative to the customer's own baseline*, which is the desired behaviour.

Nothing has been merged to `main` yet — that waits for the first deployable milestone (rule 6).

**Still no GCP resource has been provisioned.** Every cloud call across Phases 2–4 is unit-tested
against fakes only. `submit_training_job`, `create_feature_store`, `ensure_dataset`,
`log_training_run`, `log_global_importance`, and `log_predictions_to_bigquery` have never run
against the live API — expect small signature corrections on first real invocation. In particular
`aiplatform.start_run(resume=True)` in `src/evaluation/experiments.py` assumes the run already
exists from `src/training/experiments.py`; that ordering is untested against the real SDK.

The `prediction_log` BigQuery table is written by `log_predictions_to_bigquery` but **has no schema
defined in `src/features/schema.py`** and no `ensure_table` call — `PREDICTIONS_TABLE` is only a name
constant. Phase 5 must add its schema (transaction_id, variant, fraud_probability, base_value,
top_features JSON, timestamp) before the audit log can actually be written.

Sample data: 6,000 rows, 2.13% fraud, 35% stealth-fraud cohort. Regenerate with
`uv run python -m src.features.sample_data`. Artefacts (models + explainers) land in `artifacts/`
(gitignored).

Phase 5 starts with:
`git checkout develop && git pull && git checkout -b feature/inference-service`
It will build `src/inference/`: a FastAPI app with typed Pydantic request/response schemas, model +
explainer loaded once at startup, `/health` and `/predict` (returning probability + SHAP top-k),
online feature lookup, a Dockerfile targeting Cloud Run, and the A/B traffic-split configuration.
