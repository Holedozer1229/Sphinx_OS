#!/usr/bin/env bash
# ============================================================================
# sphinxskynet-bootstrap.sh — One-Command SphinxSkynet Deployment
# ============================================================================
# Usage:
#   chmod +x sphinxskynet-bootstrap.sh
#   ./sphinxskynet-bootstrap.sh
#
# This script will:
#   1. Build the Docker image for SphinxSkynet nodes
#   2. Deploy the Helm chart (creates namespace, installs/upgrades release)
#   3. Wait for all pods to become ready
#   4. Expose and print the Grafana dashboard URL
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="sphinxskynet"
RELEASE_NAME="sphinxskynet"
IMAGE_NAME="sphinxskynet/node"
IMAGE_TAG="latest"

echo "============================================"
echo "  SphinxSkynet Bootstrap"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Build Docker image
# ------------------------------------------------------------------
echo ""
echo "[1/4] Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "${SCRIPT_DIR}"
echo "      ✓ Docker image built successfully."

# ------------------------------------------------------------------
# Step 2: Deploy Helm chart
# ------------------------------------------------------------------
echo ""
echo "[2/4] Deploying Helm chart to namespace '${NAMESPACE}'"
kubectl create namespace "${NAMESPACE}" 2>/dev/null || true

helm upgrade --install "${RELEASE_NAME}" \
    "${SCRIPT_DIR}/helm/sphinxskynet" \
    --namespace "${NAMESPACE}" \
    --set image.repository="${IMAGE_NAME}" \
    --set image.tag="${IMAGE_TAG}" \
    --wait --timeout 10m
echo "      ✓ Helm chart deployed."

# ------------------------------------------------------------------
# Step 3: Wait for pods
# ------------------------------------------------------------------
echo ""
echo "[3/4] Waiting for pods to become ready..."
kubectl rollout status deployment/sphinxskynet-node \
    --namespace "${NAMESPACE}" --timeout=300s
kubectl rollout status deployment/sphinxskynet-grafana \
    --namespace "${NAMESPACE}" --timeout=300s
echo "      ✓ All pods are ready."

# ------------------------------------------------------------------
# Step 4: Expose Grafana dashboard URL
# ------------------------------------------------------------------
echo ""
echo "[4/4] Exposing Grafana dashboard..."
GRAFANA_URL=""
for i in $(seq 1 30); do
    GRAFANA_URL=$(kubectl get svc sphinxskynet-grafana \
        --namespace "${NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -n "${GRAFANA_URL}" ]; then
        break
    fi
    # Try hostname if IP not available
    GRAFANA_URL=$(kubectl get svc sphinxskynet-grafana \
        --namespace "${NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)
    if [ -n "${GRAFANA_URL}" ]; then
        break
    fi
    echo "      Waiting for LoadBalancer IP... (${i}/30)"
    sleep 10
done

echo ""
echo "============================================"
echo "  SphinxSkynet Deployment Complete!"
echo "============================================"
if [ -n "${GRAFANA_URL}" ]; then
    echo "  Grafana Dashboard: http://${GRAFANA_URL}:3000"
else
    echo "  Grafana Dashboard: (pending LoadBalancer IP)"
    echo "  Use: kubectl get svc sphinxskynet-grafana -n ${NAMESPACE}"
fi
echo "  Default credentials: admin / sphinxskynet"
echo "============================================"
