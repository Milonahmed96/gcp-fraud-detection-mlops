# project_context.md — Living Project State

## Status
Phase: Phase 5 complete — FastAPI inference service merged to develop
Last completed: feature/inference-service (src/inference/ built, 415 tests passing, container verified)
Next task: Phase 6 — drift monitoring (src/monitoring/, Cloud Scheduler integration)

## Completed tasks
- [x] TASK 1 — CLAUDE.md written (agent instructions, branching, commit convention)
- [x] TASK 2 — project_context.md created (living state + decisions log)
- [x] TASK 3 — README.md full professional write (architecture, cost breakdown, A/B, SHAP, CI/CD)
- [x] TASK 4 — repository structure scaffolded (src/, tests/, notebooks/, infrastructure/, .github/workflows/, data/sample/) + pyproject.toml, .env.example, .gitignore
- [x] TASK 5 — feature/repo-scaffold committed, pushed, merged --no-ff into develop
- [x] PHASE 2 — src/features/: config, schema, transforms, bigquery, feature_store, sample_data (+ 111 tests)
- [x] PHASE 3 — src/training/: metrics, dataset, models, train, vertex, experiments (+ 129 tests)
- [x] PHASE 4 — src/evaluation/: explainer (SHAP), experiments (Vertex + BigQuery audit log) (+ 78 tests)
- [x] PHASE 5 — src/inference/: state, features, schemas, registry, app; Dockerfile; prediction_log schema (+ 97 tests)

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
| Online store holds the customer's **recent event log**, not precomputed window aggregates | A trailing 24h window depends on when you ask. A stored aggregate is stale by exactly the inter-transaction gap — precisely when velocity matters. Windows are computed at request time against the incoming timestamp |
| `elapsed_seconds` divides nanoseconds instead of calling `Timedelta.total_seconds()` | The scalar method inherits `datetime.timedelta`'s microsecond resolution and truncated `seconds_since_prev_txn` to `220.472044` vs training's `220.472044709`. Real skew, caught by `tests/inference/test_skew.py` |
| Serving windows use a strict `<` (half-open `(t-w, t]`) | Mirrors pandas' `closed="right"` rolling. An event exactly `w` old is excluded. `<=` would be wrong on exactly the high-velocity transactions the feature exists to catch |
| `store.lookup(customer_id, as_of=event_time)` | State is only defined relative to the transaction being scored. Guards historical replay *and* a duplicated/late-arriving write in production |
| One Cloud Run revision per variant, chosen by `SERVING_VARIANT` | Cloud Run's native traffic splitting does the A/B allocation. No routing logic in app code, no shared mutable state between variants |
| Threshold read from `artifacts/metrics.json`, never defaulted to 0.5 | 0.5 discards the entire business-cost calibration. The service refuses to start rather than come up healthy and mis-calibrated |
| `create_app()` factory; state on `app.state`, not module globals | Two app instances (one per variant) must not clobber each other's loaded model. Found via test pollution |
| Startup failure is recorded, not raised | `/health` can then report *why* it is unhealthy. A container that exits on startup gives Cloud Run only a crash loop |
| GCP SDKs moved to an optional `gcp` extra | Every GCP call is lazy-imported, so the serving image never needs them. Cuts the Cloud Run image from 2.67 GB to 2.16 GB; Cloud Run bills cold-start time against image size |
| `to_naive_utc` converts offsets rather than dropping them | Discarding `+02:00` would shift the transaction two hours and silently change `is_night`, `hour_of_day`, and every velocity window |
| Pydantic `extra="forbid"` on the request schema | A typo'd field name is a bug, not something to silently ignore into a confidently wrong prediction |

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
Phase 5 finished. `develop` carries `src/inference/`: `state.py` (causal `CustomerState` +
`InMemoryStateStore`), `features.py` (`build_serving_features`), `schemas.py` (Pydantic contract),
`registry.py` (`load_bundle`, trained threshold), `app.py` (`create_app()`, `/health`, `/predict`),
plus a `Dockerfile` and `.dockerignore`. The Phase 4 gap is closed: `prediction_log` now has a real
schema and `ensure_prediction_log()`. 415 tests pass; ruff is clean.

**Verified for real, not just unit-tested.** The service was started with uvicorn and curled: a
foreign / card-not-present / 03:15 / £4800 transaction scored `0.9945` (flagged), a domestic
card-present £42 afternoon purchase scored `0.0011` (not flagged), latency 6–13 ms including SHAP.
The Docker image was built and run; the container returned identical predictions and `/health` was
ok. Image size 2.16 GB after moving the GCP SDKs to the `gcp` extra (was 2.67 GB).

Dependency layout changed: `uv sync` = inference only (what Docker installs);
`uv sync --extra gcp --extra dev` = everything (what CI and local dev need). **The test suite
requires the `gcp` extra** — `tests/features/test_bigquery.py` constructs real `SchemaField`s.
Phase 7's CI workflow must use `--extra gcp --extra dev`.

Nothing has been merged to `main` yet. Phase 5 is arguably the first deployable milestone, so a
`develop` → `main` merge is now reasonable — but Phase 7 (CI/CD) is what makes the deploy
reproducible, so waiting for it is defensible too. **Decision left to the user.**

**Still no GCP resource has been provisioned.** Every cloud call across Phases 2–5 is unit-tested
against fakes only. `submit_training_job`, `create_feature_store`, `ensure_dataset`,
`ensure_prediction_log`, `log_training_run`, `log_global_importance`, and
`log_predictions_to_bigquery` have never run against the live API. In particular
`aiplatform.start_run(resume=True)` assumes the run already exists from
`src/training/experiments.py`; that ordering is untested against the real SDK.

**Known gaps for later phases:**
- The inference service does **not yet write to `prediction_log`**. The schema and the writer both
  exist (`src/evaluation/experiments.py:log_predictions_to_bigquery`) but `app.py` never calls it —
  wiring it in needs a BigQuery client on the serving path and an async/background write so the
  request is not blocked. Phase 6 or 7.
- `InMemoryStateStore` reads the committed CSV. The real `Featurestore` reader satisfying
  `lookup(customer_id, as_of)` is not written yet.
- `MAX_RECENT_EVENTS = 100` caps the online event log. `CustomerState.truncated` records when the
  cap bit; nothing currently alerts on it.

Sample data: 6,000 rows, 2.13% fraud, 35% stealth-fraud cohort. Artefacts (models + explainers +
`metrics.json`) land in `artifacts/` (gitignored) — the Dockerfile copies them in at build time, so
`docker build` requires a prior training run.

Phase 6 starts with:
`git checkout develop && git pull && git checkout -b feature/drift-monitoring`
It will build `src/monitoring/`: PSI/KS feature-distribution drift against the training reference,
`importance_shift` on SHAP profiles as a second signal, a Cloud Scheduler-triggered entrypoint, and
a retraining trigger.
