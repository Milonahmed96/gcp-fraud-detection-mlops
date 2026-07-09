# project_context.md — Living Project State

## Status
Phase: Phase 8 complete — A/B dashboard merged to develop
Last completed: feature/ab-dashboard (src/evaluation/{report,dashboard}.py, /dashboard route, 604 tests)
Next task: Phase 9 — final polish (architecture diagram, cost breakdown refresh), then tag v1.0.0

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
- [x] PHASE 6 — src/monitoring/: drift, monitor, scheduler, app; Dockerfile.monitoring (+ 94 tests)
- [x] PHASE 7 — .github/workflows/{ci,deploy}.yml, infrastructure/setup_gcp.sh (+ 42 workflow tests)
- [x] PHASE 8 — src/evaluation/{report,dashboard}.py, GET /dashboard on the monitor (+ 49 tests)

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
| Drift detection is unsupervised (PSI on features), not AUC-based | Fraud labels arrive weeks late via chargebacks. Production AUC is not observable in time to act on |
| Drift = `any(feature significant)`, never a mean | One feature moving hard is the fraud-ring signature. Averaging across 13 stable features would hide it |
| Features with ≤10 distinct values (and all bools) are treated as categorical | Quantile-binning `is_night` or `txn_count_1h` (almost always 1) yields duplicate edges, empty bins, and `log(0)` → `inf`/`nan`. A monitor that silently never fires |
| Every PSI proportion floored at `EPSILON=1e-6`; outer bin edges are `±inf` | Prevents `log(0)`, and stops a value beyond the training range from vanishing and deflating the PSI |
| KS is reported but does not gate retraining | Its p-value shrinks with sample size, so a large daily batch flags drift that is real yet negligible |
| Reference profile is built from the **training** split | The distribution the model actually learned. Test is a distribution it never saw |
| Drift monitor is a **separate Cloud Run service + image** | The inference image has no BigQuery SDK (that's the 500 MB saving), and a minutes-long batch job must not share a request pool or scale-to-zero policy with a 10 ms endpoint |
| Cloud Scheduler → OIDC token, audience = bare service URL | Cloud Run rejects a token whose audience carries the request path. No API key in the job definition |
| `ensure_drift_check_job` creates-or-updates | A redeploy must not fail with `AlreadyExists`, nor leave a stale schedule behind |
| A retraining submission failure is logged, not raised | A 5xx makes Cloud Scheduler retry → a thundering herd of Vertex AI training jobs. The check already did its job by reporting drift |
| `scipy` declared explicitly despite arriving via scikit-learn | `drift.py` imports `ks_2samp` directly. Relying on a transitive dependency is a trap |
| `deploy.yml` calls `ci.yml` via `workflow_call` | `main` must never deploy code that has not passed the same gate a PR does |
| New revisions deploy with `--no-traffic`, smoke-tested on their `--tag` URL, then traffic shifts | The ordering *is* the rollback strategy. If the smoke test fails, the job stops and the previous revision keeps 100%. A rollback you never execute is the only kind that reliably works |
| Smoke test impersonates the deployer SA with `--audiences` to mint an ID token | A Workload Identity *federated* credential cannot mint an ID token. Bare `gcloud auth print-identity-token` would 403 against the private service on every deploy. Requires `serviceAccountTokenCreator` on itself |
| WIF provider carries `attribute-condition` pinning `assertion.repository` | Without it, *any* GitHub repository could mint tokens for the deployer service account |
| CI smoke-tests the running containers, not just `docker build` | Proving an image assembles is not proving it serves. The greps (`"status":"ok"`, `"is_flagged":true`) were verified against real output |
| `.github/workflows/*.yml` is parsed and asserted in `tests/workflows/` | A broken workflow is otherwise discovered on push to `main` — the worst possible moment |
| `concurrency: deploy-main`, `cancel-in-progress: false` | Two racing deploys would leave the traffic split indeterminate |
| CI provisions the schedule via `python -m src.monitoring.scheduler` | Not a hand-rolled `gcloud scheduler jobs create \|\| update`, so the idempotency the tests cover is the idempotency that runs |
| Dashboard headline is the **verdict**, not the winner | The interval straddles zero, so the hero reads "No significant difference". A skimming reader must not believe a coin flip was a result |
| The two SHAP panels share one scale | Per-panel normalisation drew LightGBM's 0.73 as long as XGBoost's 1.43 — a false cross-panel comparison the side-by-side layout invites. Found by rendering the page and looking at it |
| Ranking metrics and cost get separate charts | Different scales. Never a dual axis |
| Dashboard is inline SVG, no JS, no CDN | Renders from `file://`, inside a locked-down Cloud Run service, and as an email attachment. Nothing to break, nothing to fetch |
| Every bar directly labelled + a table view | The aqua series is below 3:1 contrast on the light surface, so colour alone never carries a value |
| `/dashboard` lives on the **monitor**, not the inference service | It is an operator surface and needs both variants' metrics; an inference revision only knows the variant it serves |
| Per-variant `importance_<variant>.json` is written at training time | The drift baseline (`reference_importance.json`) is incumbent-only by design; the dashboard compares both |

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
Phase 7 finished. `develop` carries `.github/workflows/ci.yml` (lint, tests, build **and run** both
containers), `.github/workflows/deploy.yml` (WIF auth, push to Artifact Registry, two no-traffic
revisions, smoke test, traffic split, monitor + scheduler), `infrastructure/setup_gcp.sh` (one-time
idempotent bootstrap, creates no keys), and `tests/workflows/` (42 tests parsing the YAML and
asserting the invariants). `src/monitoring/scheduler.py` gained a CLI so CI provisions the schedule
through tested code. 551 tests pass; ruff is clean.

**Verified before shipping:** `Dockerfile.monitoring` was built (2.67 GB, correctly carries the gcp
extra) and its container serves `/health` and `POST /drift-check`. Every `run:` block in both
workflows was extracted and passed through `bash -n`. The `jq -er` tagged-URL extraction was tested
against a realistic Cloud Run `describe` payload and confirmed to exit non-zero on a missing tag. The
CI smoke-test greps (`"status":"ok"`, `"is_flagged":true`) were verified against real server output.

**Two footguns fixed before they could fail a real deploy:**
1. `gcloud auth print-identity-token` does **not** work under Workload Identity Federation — a
   federated credential cannot mint an ID token. The smoke test now impersonates the deployer SA
   with `--audiences=<exact URL>`, and `setup_gcp.sh` grants it `serviceAccountTokenCreator` on
   itself. Without that binding the smoke test 403s and nothing ever deploys.
2. The original `--format="...extract(url)"` projection for the tagged revision URL was fragile;
   replaced with `jq -er`.

**README corrected, not just extended.** It previously claimed per-prediction attributions were
"mirrored into BigQuery". They are not — the writer exists but `app.py` never calls it. The claim
is now accurate and a **Known limitations** section was added.

**`develop` → `main` is DONE.** The GitHub Actions run on `main` is green: Lint, Test, and Build
images all pass, and `Deploy to Cloud Run` is correctly **skipped** because `vars.GCP_PROJECT_ID`
is unset. Configure the repo secrets/variables below to enable real deploys.

**No tag yet.** The merge commit message says "release: v1.0.0", which is premature — Phase 8
(dashboard) and Phase 9 (architecture diagram, final cost breakdown) are outstanding deliverables.
Tag `v1.0.0` at the end of Phase 9, not before. Do not rewrite the pushed `main` history to fix the
message.

**Two bugs the real CI run found that local testing did not:**
1. `cache-to: type=gha` fails on buildx's default `docker` driver
   ("Cache export is not supported for the docker driver"). Fixed by adding
   `docker/setup-buildx-action@v3` before any cached build, in both workflows. A test now asserts
   buildx is set up before any step using the GHA cache.
2. `ruff format --check` failed on `tests/workflows/test_workflows.py` — the pre-merge command run
   locally used `ruff check` but omitted `ruff format`. **Always run both.**

Required GitHub config (from `setup_gcp.sh` output):
- Secrets: `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`, `SCHEDULER_SERVICE_ACCOUNT`
- Variables: `GCP_PROJECT_ID`, `GCP_REGION`, `ARTIFACT_REPOSITORY`, `GCP_BUCKET_NAME`,
  `BIGQUERY_DATASET`, `FEATURE_STORE_ID`
- An `environment: production` must exist (deploy.yml targets it).

**Still no GCP resource has been provisioned.** Every cloud call across Phases 2–7 is unit-tested
against fakes only. `aiplatform.start_run(resume=True)` in `src/evaluation/experiments.py` assumes
the run already exists from `src/training/experiments.py`; untested against the real SDK.

**Known gaps, carried forward (also listed in README > Known limitations):**
- The inference service still does **not write to `prediction_log`**. So the Phase 8 A/B dashboard
  has **no production prediction data** and must read `artifacts/metrics.json`. The clean fix is
  structured logging to Cloud Logging + a BigQuery sink (no SDK on the serving path, no latency).
- `InMemoryStateStore` reads the committed CSV; the real Featurestore reader is unwritten.
- `MAX_RECENT_EVENTS = 100` caps the online event log; `truncated` is recorded but nothing alerts.
- Drift monitor compares the whole current batch; no windowing beyond `--start-date`/`--end-date`.

**Phase 8 done.** `src/evaluation/report.py` loads `metrics.json` + `importance_<variant>.json`;
`src/evaluation/dashboard.py` renders one self-contained HTML file (inline SVG, no JS, no CDN);
`GET /dashboard` on the monitoring service serves it. Training now writes per-variant importance.
Palette validated with a CVD checker (worst adjacent ΔE 73.6, both modes pass).

**Rendered and inspected in a real browser**, which caught two things tests would not have:
1. The two SHAP panels each normalised to their own max — LightGBM's 0.73 drew as long as
   XGBoost's 1.43. Now both share one scale, and a test asserts the bar-width ratio.
2. `LightGBM` wrapped under its swatch in the table (59px row vs 38px). Fixed with `nowrap`.

Phase 9 starts with:
`git checkout develop && git pull && git checkout -b feature/final-polish`
- Mermaid + draw.io architecture diagram (`infrastructure/`), reflecting the **two** Cloud Run
  services (inference + monitor), not one.
- Refresh the README cost breakdown — it still says "estimates from the pricing page" and has never
  been checked against the calculator. Either verify or keep the caveat explicit.
- Update the architecture Mermaid at the top of the README: it predates the monitoring service and
  the drift-check endpoint.
- Final `project_context.md` update, then `git tag v1.0.0` on `main`.
