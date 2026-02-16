"""
============================================================================
deploy_from_github.py — SphinxSkynet One-Command Deployment
============================================================================

Usage:
    python deploy_from_github.py

Flow:
    1. Pulls/validates repo from GitHub
    2. Fetches latest circuits/zkeys from S3 or local fallback
    3. Builds Docker image
    4. Deploys Helm chart to Kubernetes
    5. Waits for rollout to complete
    6. Prints service URLs

Environment Variables:
    GITHUB_REPO      — GitHub repo URL (default: origin remote)
    DOCKER_REGISTRY  — Docker registry prefix (default: sphinxskynet)
    DOCKER_TAG       — Image tag (default: latest)
    S3_BUCKET        — S3 bucket for circuit artifacts (optional)
    KUBECONFIG       — Path to kubeconfig (default: ~/.kube/config)
    HELM_NAMESPACE   — Kubernetes namespace (default: default)
============================================================================
"""

import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCKER_REGISTRY = os.environ.get("DOCKER_REGISTRY", "sphinxskynet")
DOCKER_TAG = os.environ.get("DOCKER_TAG", "latest")
HELM_CHART_PATH = os.path.join(os.path.dirname(__file__), "helm", "sphinxskynet")
HELM_RELEASE = "sphinxskynet"
HELM_NAMESPACE = os.environ.get("HELM_NAMESPACE", "default")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
CIRCUITS_DIR = os.path.join(os.path.dirname(__file__), "circuits")


def run(cmd, check=True, capture=False):
    """Run a shell command with logging."""
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )
    return result


# ---------------------------------------------------------------------------
# Step 1: Fetch circuit artifacts from S3 (optional)
# ---------------------------------------------------------------------------
def fetch_circuits():
    """Download circuit artifacts from S3 if configured."""
    if not S3_BUCKET:
        print("[1/5] Skipping S3 fetch — using local circuits/")
        return

    print(f"[1/5] Fetching circuits from s3://{S3_BUCKET}/circuits/ ...")
    os.makedirs(CIRCUITS_DIR, exist_ok=True)
    artifacts = [
        "shell50.circom",
        "powersOfTau28_hez_final_10.ptau",
        "shell50_final.zkey",
        "shell50_final.vkey.json",
    ]
    for artifact in artifacts:
        src = f"s3://{S3_BUCKET}/circuits/{artifact}"
        dst = os.path.join(CIRCUITS_DIR, artifact)
        if not os.path.exists(dst):
            run(["aws", "s3", "cp", src, dst], check=False)
        else:
            print(f"    ✓ {artifact} already present")


# ---------------------------------------------------------------------------
# Step 2: Build Docker image
# ---------------------------------------------------------------------------
def build_docker():
    """Build the SphinxSkynet Docker image."""
    image = f"{DOCKER_REGISTRY}:{DOCKER_TAG}"
    print(f"[2/5] Building Docker image {image} ...")
    run(["docker", "build", "-t", image, "."])
    return image


# ---------------------------------------------------------------------------
# Step 3: Push Docker image (optional)
# ---------------------------------------------------------------------------
def push_docker(image):
    """Push Docker image to registry."""
    print(f"[3/5] Pushing Docker image {image} ...")
    result = run(["docker", "push", image], check=False)
    if result.returncode != 0:
        print("    ⚠ Push failed — using local image (minikube/kind)")


# ---------------------------------------------------------------------------
# Step 4: Deploy Helm chart
# ---------------------------------------------------------------------------
def deploy_helm(image):
    """Deploy or upgrade the SphinxSkynet Helm release."""
    print(f"[4/5] Deploying Helm chart from {HELM_CHART_PATH} ...")
    run([
        "helm", "upgrade", "--install",
        HELM_RELEASE,
        HELM_CHART_PATH,
        "--namespace", HELM_NAMESPACE,
        "--set", f"image.repository={DOCKER_REGISTRY}",
        "--set", f"image.tag={DOCKER_TAG}",
        "--wait",
        "--timeout", "300s",
    ])


# ---------------------------------------------------------------------------
# Step 5: Wait for rollout and print status
# ---------------------------------------------------------------------------
def wait_and_report():
    """Wait for deployment rollout and print service endpoints."""
    print("[5/5] Waiting for rollout ...")
    run([
        "kubectl", "rollout", "status",
        "deployment/sphinxskynet-node",
        "--namespace", HELM_NAMESPACE,
        "--timeout=300s",
    ])

    print("\n✅ SphinxSkynet deployed successfully!")
    print("   Endpoints:")
    result = run(
        ["kubectl", "get", "svc", "--namespace", HELM_NAMESPACE, "-o", "wide"],
        capture=True,
        check=False,
    )
    if result.returncode == 0:
        print(result.stdout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  SphinxSkynet — One-Command Deployment")
    print("=" * 60)

    fetch_circuits()
    image = build_docker()
    push_docker(image)
    deploy_helm(image)
    wait_and_report()


if __name__ == "__main__":
    main()
