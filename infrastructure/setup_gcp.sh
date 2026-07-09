#!/usr/bin/env bash
#
# One-time GCP bootstrap for the CI/CD pipeline.
#
# Creates the Artifact Registry repository, the two service accounts, and the
# Workload Identity Federation pool that lets GitHub Actions authenticate
# without a service-account key. Run once, by a human, from an authenticated
# shell. Everything after this is automated.
#
#   export GCP_PROJECT_ID=my-project GITHUB_REPO=Milonahmed96/gcp-fraud-detection-mlops
#   ./infrastructure/setup_gcp.sh
#
# Idempotent: safe to re-run. Every create is guarded by a describe.

set -euo pipefail

: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
: "${GITHUB_REPO:?set GITHUB_REPO, e.g. owner/repo}"

GCP_REGION="${GCP_REGION:-europe-west2}"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-fraud-detection}"
POOL="${POOL:-github-pool}"
PROVIDER="${PROVIDER:-github-provider}"

DEPLOYER_SA="github-deployer"
SCHEDULER_SA="drift-scheduler"
DEPLOYER_EMAIL="${DEPLOYER_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_EMAIL="${SCHEDULER_SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

echo "==> project ${GCP_PROJECT_ID}, region ${GCP_REGION}"
gcloud config set project "${GCP_PROJECT_ID}" --quiet

echo "==> enabling APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  cloudscheduler.googleapis.com \
  iamcredentials.googleapis.com \
  --quiet

echo "==> Artifact Registry"
gcloud artifacts repositories describe "${ARTIFACT_REPOSITORY}" --location="${GCP_REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${ARTIFACT_REPOSITORY}" \
      --repository-format=docker --location="${GCP_REGION}" \
      --description="Fraud detection service images" --quiet

echo "==> service accounts"
for SA in "${DEPLOYER_SA}" "${SCHEDULER_SA}"; do
  gcloud iam service-accounts describe "${SA}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 \
    || gcloud iam service-accounts create "${SA}" --quiet
done

echo "==> roles for the deployer"
# Least privilege: push images, deploy Cloud Run, submit Vertex jobs, manage the
# schedule. Notably NOT project owner/editor.
for ROLE in \
  roles/artifactregistry.writer \
  roles/run.admin \
  roles/aiplatform.user \
  roles/cloudscheduler.admin \
  roles/iam.serviceAccountUser \
  roles/bigquery.dataEditor \
  roles/bigquery.jobUser; do
  gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER_EMAIL}" --role="${ROLE}" --condition=None --quiet >/dev/null
done

echo "==> letting the deployer mint ID tokens for the smoke test"
# The smoke test calls the private Cloud Run revision, which needs an ID token
# whose audience is the revision URL. `google-github-actions/auth` obtains one
# via IAM Credentials `generateIdToken` on this service account.
#
# Note: `gcloud auth print-identity-token --impersonate-service-account` does
# NOT work here, whatever the docs imply. It fails under Workload Identity
# Federation, which the first real deploy demonstrated.
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_EMAIL}" \
  --member="serviceAccount:${DEPLOYER_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet >/dev/null

echo "==> roles for the scheduler"
# Scheduler only needs to invoke the private drift-check Cloud Run service.
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${SCHEDULER_EMAIL}" --role="roles/run.invoker" \
  --condition=None --quiet >/dev/null

echo "==> Workload Identity Federation pool"
gcloud iam workload-identity-pools describe "${POOL}" --location=global >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools create "${POOL}" \
      --location=global --display-name="GitHub Actions" --quiet

# The attribute-condition is the security boundary. Without it, ANY GitHub
# repository on the internet could mint tokens for this service account.
gcloud iam workload-identity-pools providers describe "${PROVIDER}" \
  --location=global --workload-identity-pool="${POOL}" >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" \
      --location=global \
      --workload-identity-pool="${POOL}" \
      --issuer-uri="https://token.actions.githubusercontent.com" \
      --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
      --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
      --quiet

PROJECT_NUMBER="$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')"
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"

echo "==> letting ${GITHUB_REPO} impersonate the deployer"
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

cat <<EOF

==> Done. Configure the GitHub repository with:

  Secrets (Settings > Secrets and variables > Actions > Secrets):
    WIF_PROVIDER              ${POOL_ID}/providers/${PROVIDER}
    WIF_SERVICE_ACCOUNT       ${DEPLOYER_EMAIL}
    SCHEDULER_SERVICE_ACCOUNT ${SCHEDULER_EMAIL}

  Variables (Settings > Secrets and variables > Actions > Variables):
    GCP_PROJECT_ID            ${GCP_PROJECT_ID}
    GCP_REGION                ${GCP_REGION}
    ARTIFACT_REPOSITORY       ${ARTIFACT_REPOSITORY}
    GCP_BUCKET_NAME           <your bucket>
    BIGQUERY_DATASET          <your dataset>
    FEATURE_STORE_ID          <your feature store>

No service-account key was created, and none is needed.
EOF
