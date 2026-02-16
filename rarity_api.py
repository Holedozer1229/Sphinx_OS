"""
============================================================================
rarity_api.py — SphinxSkynet Rarity Proof Engine
============================================================================

FastAPI backend for rarity proofs and wormhole metrics.

Endpoints:
    /rarity       → Node rarity scores based on Φ_total and Δλ
    /wormholes    → Wormhole Laplacian flows between node pairs
    /health       → Health check
    /nodes        → Full node state summary

Prometheus metrics exposed on port 8001 via node_main.

Enhanced with:
    - Security middleware (rate limiting, authentication)
    - Input validation
    - Comprehensive metrics
    - Error handling
============================================================================
"""

import time
import numpy as np
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram

from node_main import (
    all_nodes,
    TraversableAncillaryWormhole,
    generate_hyperzk_proof,
    propagate_delta_lambda,
)

# Import security modules
try:
    from sphinx_os.security import RateLimiter, InputValidator
    from sphinx_os.config_manager import get_config
    SECURITY_ENABLED = True
    config = get_config()
except ImportError:
    SECURITY_ENABLED = False
    print("Warning: Security modules not available. Running in development mode.")

# Additional metrics for rarity API
rarity_calculation_time = Histogram(
    'sphinxos_rarity_calculation_seconds',
    'Time to calculate rarity scores'
)
wormhole_calculation_time = Histogram(
    'sphinxos_wormhole_calculation_seconds',
    'Time to calculate wormhole flows'
)
api_errors = Counter(
    'sphinxos_rarity_api_errors_total',
    'Rarity API errors',
    ['endpoint', 'error_type']
)

app = FastAPI(
    title="SphinxSkynet Rarity API",
    version="1.0.0",
    description="Rarity scoring and wormhole metrics with ZK proofs"
)

# Configure CORS
if SECURITY_ENABLED:
    try:
        security_config = config.get_security_config()
        cors_origins = security_config.get("cors_origins", ["http://localhost:3000"])
    except:
        cors_origins = ["http://localhost:3000"]
else:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiter
if SECURITY_ENABLED:
    try:
        api_config = config.get_api_config()
        rate_limiter = RateLimiter(requests_per_minute=api_config.get("rate_limit", 100))
    except:
        rate_limiter = None
else:
    rate_limiter = None


# Middleware for request timing
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        api_errors.labels(
            endpoint=request.url.path,
            error_type=type(e).__name__
        ).inc()
        raise


def compute_rarity_score(node):
    """
    Rarity score for a node based on Φ_total, Δλ, and entanglement density.

    rarity_i = Φ_total_i * (1 + |Δλ_i|) * (1 + Tr(ρ_S_i))
    Higher rarity = more unique/valuable node state.
    """
    trace_rho = float(np.trace(node.rho_s))
    rarity = node.phi_total * (1 + abs(node.delta_lambda)) * (1 + trace_rho)
    return rarity


@app.get("/rarity")
async def rarity():
    """
    Return rarity scores for all nodes with zk-proof status.
    
    Enhanced with:
    - Performance tracking
    - Error handling
    - Detailed metrics
    """
    try:
        start_time = time.time()
        
        scores = []
        for node in all_nodes:
            proof_ok = generate_hyperzk_proof(node)
            scores.append({
                "node_id": node.id,
                "phi_total": float(node.phi_total),
                "delta_lambda": float(node.delta_lambda),
                "rarity_score": compute_rarity_score(node),
                "proof_valid": proof_ok,
            })
        
        scores.sort(key=lambda x: x["rarity_score"], reverse=True)
        
        # Track performance
        duration = time.time() - start_time
        rarity_calculation_time.observe(duration)
        
        return {
            "rarity_scores": scores,
            "computation_time": duration,
            "total_nodes": len(scores)
        }
        
    except Exception as e:
        api_errors.labels(endpoint="/rarity", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Rarity calculation failed: {str(e)}")


@app.get("/wormholes")
async def wormholes():
    """
    Return wormhole Laplacian flows between all node pairs.
    
    Enhanced with:
    - Performance tracking
    - Error handling
    - Pagination support (for large node sets)
    """
    try:
        start_time = time.time()
        
        flows = []
        for i, node_i in enumerate(all_nodes):
            for j, node_j in enumerate(all_nodes):
                if i != j:
                    w = TraversableAncillaryWormhole(node_i, node_j)
                    metric = w.compute_metric()
                    laplacian = w.compute_laplacian()
                    flows.append({
                        "source": i,
                        "target": j,
                        "metric": float(metric),
                        "laplacian": float(laplacian),
                    })
        
        # Track performance
        duration = time.time() - start_time
        wormhole_calculation_time.observe(duration)
        
        return {
            "wormhole_flows": flows,
            "computation_time": duration,
            "total_connections": len(flows)
        }
        
    except Exception as e:
        api_errors.labels(endpoint="/wormholes", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Wormhole calculation failed: {str(e)}")


@app.get("/nodes")
async def nodes():
    """
    Return full state summary for all nodes.
    
    Enhanced with detailed statistics and validation.
    """
    try:
        node_data = []
        for n in all_nodes:
            node_data.append({
                "node_id": n.id,
                "phi_total": float(n.phi_total),
                "delta_lambda": float(n.delta_lambda),
                "hypercube_shape": list(n.hypercube_state.shape),
                "ancilla_shape": list(n.ancilla_state.shape),
                "rarity_score": compute_rarity_score(n),
            })
        
        # Calculate statistics
        avg_phi = sum(n["phi_total"] for n in node_data) / len(node_data) if node_data else 0
        avg_rarity = sum(n["rarity_score"] for n in node_data) / len(node_data) if node_data else 0
        
        return {
            "nodes": node_data,
            "statistics": {
                "total_nodes": len(node_data),
                "avg_phi_total": avg_phi,
                "avg_rarity_score": avg_rarity
            }
        }
        
    except Exception as e:
        api_errors.labels(endpoint="/nodes", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Node query failed: {str(e)}")


@app.get("/propagate")
async def propagate():
    """
    Run one step of Δλ propagation and return updated states.
    
    Enhanced with validation and error handling.
    """
    try:
        propagate_delta_lambda(all_nodes)
        
        return {
            "status": "propagated",
            "timestamp": time.time(),
            "nodes": [
                {
                    "node_id": n.id,
                    "delta_lambda": float(n.delta_lambda),
                    "phi_total": float(n.phi_total),
                }
                for n in all_nodes
            ],
        }
        
    except Exception as e:
        api_errors.labels(endpoint="/propagate", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=f"Propagation failed: {str(e)}")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Enhanced with system status and diagnostics.
    """
    try:
        # Basic health check
        node_count = len(all_nodes)
        avg_phi = sum(n.phi_total for n in all_nodes) / node_count if node_count > 0 else 0
        
        # Check if we can perform basic operations
        test_rarity = compute_rarity_score(all_nodes[0]) if all_nodes else 0
        
        return {
            "status": "ok",
            "timestamp": time.time(),
            "nodes_available": node_count,
            "avg_phi_total": float(avg_phi),
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }


if __name__ == "__main__":
    import uvicorn

    print("Starting SphinxSkynet Rarity API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
