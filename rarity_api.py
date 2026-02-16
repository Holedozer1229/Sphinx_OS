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
============================================================================
"""

import numpy as np
from fastapi import FastAPI

from node_main import (
    all_nodes,
    TraversableAncillaryWormhole,
    generate_hyperzk_proof,
    propagate_delta_lambda,
)

app = FastAPI(title="SphinxSkynet Rarity API", version="1.0.0")


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
def rarity():
    """Return rarity scores for all nodes with zk-proof status."""
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
    return {"rarity_scores": scores}


@app.get("/wormholes")
def wormholes():
    """Return wormhole Laplacian flows between all node pairs."""
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
    return {"wormhole_flows": flows}


@app.get("/nodes")
def nodes():
    """Return full state summary for all nodes."""
    return {
        "nodes": [
            {
                "node_id": n.id,
                "phi_total": float(n.phi_total),
                "delta_lambda": float(n.delta_lambda),
                "hypercube_shape": list(n.hypercube_state.shape),
                "ancilla_shape": list(n.ancilla_state.shape),
                "rarity_score": compute_rarity_score(n),
            }
            for n in all_nodes
        ]
    }


@app.get("/propagate")
def propagate():
    """Run one step of Δλ propagation and return updated states."""
    propagate_delta_lambda(all_nodes)
    return {
        "status": "propagated",
        "nodes": [
            {
                "node_id": n.id,
                "delta_lambda": float(n.delta_lambda),
                "phi_total": float(n.phi_total),
            }
            for n in all_nodes
        ],
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print("Starting SphinxSkynet Rarity API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
