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
import time
from pathlib import Path


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

# Timeout constants (in seconds)
COMMAND_CHECK_TIMEOUT = 5
CLUSTER_CHECK_TIMEOUT = 10
S3_DOWNLOAD_TIMEOUT = 60
DOCKER_BUILD_TIMEOUT = 1800  # 30 minutes
DOCKER_PUSH_TIMEOUT = 600    # 10 minutes
HELM_DEPLOY_TIMEOUT = 600    # 10 minutes
KUBECTL_ROLLOUT_TIMEOUT = 600  # 10 minutes
# Add buffer for subprocess wrapper
SUBPROCESS_TIMEOUT_BUFFER = 60


def run(cmd, check=True, capture=False, timeout=None):
    """Run a shell command with logging and improved error handling."""
    print(f"  → {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired as e:
        print(f"    ✗ Command timed out after {timeout}s: {' '.join(cmd)}")
        if check:
            raise RuntimeError(f"Command timed out: {' '.join(cmd)}") from e
        return subprocess.CompletedProcess(cmd, 1, "", str(e))
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"    Error: {e.stderr[:500]}")
        if check:
            raise
        return e
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}") from e
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


# ---------------------------------------------------------------------------
# Step 0: Validate prerequisites
# ---------------------------------------------------------------------------
def validate_prerequisites():
    """Validate that required tools and paths exist before deployment."""
    print("[0/5] Validating prerequisites ...")
    
    required_commands = ["docker", "helm", "kubectl"]
    missing = []
    
    for cmd in required_commands:
        try:
            result = run([cmd, "--version"], check=False, capture=True, timeout=COMMAND_CHECK_TIMEOUT)
            if result.returncode != 0:
                missing.append(cmd)
            else:
                print(f"    ✓ {cmd} is available")
        except Exception:
            missing.append(cmd)
    
    if missing:
        raise RuntimeError(
            f"Missing required commands: {', '.join(missing)}. "
            "Please install them before running deployment."
        )
    
    # Validate Helm chart path
    if not Path(HELM_CHART_PATH).exists():
        raise RuntimeError(
            f"Helm chart path does not exist: {HELM_CHART_PATH}. "
            "Please ensure the repository structure is correct."
        )
    
    # Validate kubectl cluster access
    try:
        result = run(["kubectl", "cluster-info"], check=False, capture=True, timeout=CLUSTER_CHECK_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(
                "Cannot access Kubernetes cluster. "
                "Please check your kubeconfig and cluster connectivity."
            )
        print("    ✓ Kubernetes cluster is accessible")
    except Exception as e:
        raise RuntimeError(f"Kubernetes cluster validation failed: {e}") from e
    
    print("    ✓ All prerequisites validated")


# ---------------------------------------------------------------------------
# Step 1: Fetch circuit artifacts from S3 (optional)
# ---------------------------------------------------------------------------
def fetch_circuits():
    """Download circuit artifacts from S3 if configured."""
    if not S3_BUCKET:
        print("[1/5] Skipping S3 fetch — using local circuits/")
        # Validate that local circuits exist
        if Path(CIRCUITS_DIR).exists():
            circuit_files = list(Path(CIRCUITS_DIR).glob("*"))
            if circuit_files:
                print(f"    ✓ Found {len(circuit_files)} local circuit file(s)")
            else:
                print("    ⚠ Warning: circuits directory is empty")
        return

    print(f"[1/5] Fetching circuits from s3://{S3_BUCKET}/circuits/ ...")
    os.makedirs(CIRCUITS_DIR, exist_ok=True)
    artifacts = [
        "shell50.circom",
        "powersOfTau28_hez_final_10.ptau",
        "shell50_final.zkey",
        "shell50_final.vkey.json",
    ]
    
    failed = []
    for artifact in artifacts:
        src = f"s3://{S3_BUCKET}/circuits/{artifact}"
        dst = os.path.join(CIRCUITS_DIR, artifact)
        if not os.path.exists(dst):
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = run(["aws", "s3", "cp", src, dst], check=False, timeout=S3_DOWNLOAD_TIMEOUT)
                    if result.returncode == 0:
                        print(f"    ✓ Downloaded {artifact}")
                        break
                    else:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"    ⚠ Retry {attempt + 1}/{max_retries} for {artifact} in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"    ✗ Failed to fetch {artifact} after {max_retries} attempts")
                            failed.append(artifact)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"    ⚠ Error fetching {artifact}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"    ✗ Failed to fetch {artifact}: {e}")
                        failed.append(artifact)
        else:
            print(f"    ✓ {artifact} already present")
    
    if failed:
        raise RuntimeError(
            f"Failed to fetch required circuit artifacts: {', '.join(failed)}. "
            "Deployment cannot proceed without these files."
        )


# ---------------------------------------------------------------------------
# Step 2: Build Docker image
# ---------------------------------------------------------------------------
def build_docker():
    """Build the SphinxSkynet Docker image."""
    image = f"{DOCKER_REGISTRY}:{DOCKER_TAG}"
    print(f"[2/5] Building Docker image {image} ...")
    
    # Verify Dockerfile exists
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    if not dockerfile_path.exists():
        raise RuntimeError(f"Dockerfile not found at {dockerfile_path}")
    
    try:
        # Build with timeout for large builds (30 minutes)
        run(["docker", "build", "-t", image, "."], timeout=DOCKER_BUILD_TIMEOUT)
        
        # Verify image was created
        result = run(["docker", "images", "-q", image], capture=True, check=False)
        if not result.stdout.strip():
            raise RuntimeError(f"Docker image {image} was not created successfully")
        
        print(f"    ✓ Docker image {image} built successfully")
        return image
    except Exception as e:
        raise RuntimeError(f"Docker build failed: {e}") from e


# ---------------------------------------------------------------------------
# Step 3: Push Docker image (optional)
# ---------------------------------------------------------------------------
def push_docker(image):
    """Push Docker image to registry."""
    print(f"[3/5] Pushing Docker image {image} ...")
    
    # Check if we're using a local registry
    # Check if registry starts with localhost or 127.0.0.1, or is the default local name
    registry_lower = DOCKER_REGISTRY.lower()
    is_local = (
        registry_lower in ["sphinxskynet", "localhost"] or
        registry_lower.startswith("localhost:") or
        registry_lower.startswith("127.0.0.1")
    )
    
    if is_local:
        print("    ℹ Using local registry — skipping push")
        return
    
    # Attempt push with timeout
    result = run(["docker", "push", image], check=False, timeout=DOCKER_PUSH_TIMEOUT)
    if result.returncode != 0:
        print("    ⚠ Push failed — using local image (minikube/kind)")
        print("    ℹ This is expected for local development environments")
    else:
        print(f"    ✓ Image pushed to registry")


# ---------------------------------------------------------------------------
# Step 4: Deploy Helm chart
# ---------------------------------------------------------------------------
def deploy_helm(image):
    """Deploy or upgrade the SphinxSkynet Helm release."""
    print(f"[4/5] Deploying Helm chart from {HELM_CHART_PATH} ...")
    
    try:
        # Create namespace if it doesn't exist
        run([
            "kubectl", "create", "namespace", HELM_NAMESPACE
        ], check=False, capture=True)
        
        # Deploy with timeout (10 minutes)
        run([
            "helm", "upgrade", "--install",
            HELM_RELEASE,
            HELM_CHART_PATH,
            "--namespace", HELM_NAMESPACE,
            "--set", f"image.repository={DOCKER_REGISTRY}",
            "--set", f"image.tag={DOCKER_TAG}",
            "--wait",
            "--timeout", f"{HELM_DEPLOY_TIMEOUT}s",
        ], timeout=HELM_DEPLOY_TIMEOUT + SUBPROCESS_TIMEOUT_BUFFER)
        
        print(f"    ✓ Helm release '{HELM_RELEASE}' deployed successfully")
    except Exception as e:
        # Try to get Helm release status for debugging
        print("    ✗ Helm deployment failed, checking release status...")
        run([
            "helm", "status", HELM_RELEASE, "--namespace", HELM_NAMESPACE
        ], check=False, capture=False)
        raise RuntimeError(f"Helm deployment failed: {e}") from e


# ---------------------------------------------------------------------------
# Step 5: Wait for rollout and print status
# ---------------------------------------------------------------------------
def wait_and_report():
    """Wait for deployment rollout and print service endpoints."""
    print("[5/5] Waiting for rollout ...")
    
    try:
        # Wait for rollout with timeout (10 minutes)
        run([
            "kubectl", "rollout", "status",
            "deployment/sphinxskynet-node",
            "--namespace", HELM_NAMESPACE,
            f"--timeout={KUBECTL_ROLLOUT_TIMEOUT}s",
        ], timeout=KUBECTL_ROLLOUT_TIMEOUT + SUBPROCESS_TIMEOUT_BUFFER)
        
        print(f"    ✓ Deployment rolled out successfully")
    except Exception as e:
        print("    ⚠ Rollout status check failed, checking pod status...")
        run([
            "kubectl", "get", "pods",
            "--namespace", HELM_NAMESPACE,
            "-l", "app=sphinxskynet"
        ], check=False)
        raise RuntimeError(f"Deployment rollout failed: {e}") from e

    print("\n✅ SphinxSkynet deployed successfully!")
    print("   Endpoints:")
    result = run(
        ["kubectl", "get", "svc", "--namespace", HELM_NAMESPACE, "-o", "wide"],
        capture=True,
        check=False,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("    ⚠ Could not retrieve service endpoints")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Main deployment orchestration with error handling."""
    print("=" * 60)
    print("  SphinxSkynet — One-Command Deployment")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        validate_prerequisites()
        fetch_circuits()
        image = build_docker()
        push_docker(image)
        deploy_helm(image)
        wait_and_report()
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Deployment completed successfully in {elapsed:.1f}s")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Deployment interrupted by user")
        return 130
    except RuntimeError as e:
        print(f"\n\n✗ Deployment failed: {e}")
        print("   Please check the error messages above and fix the issues")
        return 1
    except Exception as e:
        print(f"\n\n✗ Unexpected error during deployment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
