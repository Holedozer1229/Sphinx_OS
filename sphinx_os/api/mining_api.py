"""
Mining API for SphinxSkynet Blockchain
FastAPI endpoints for mining operations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os
import math
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.blockchain.core import SphinxSkynetBlockchain
from sphinx_os.mining.miner import SphinxMiner
from sphinx_os.mining.merge_miner import MergeMiningCoordinator


# Pydantic models for requests
class StartMiningRequest(BaseModel):
    miner_address: str
    algorithm: str = "spectral"
    num_threads: int = 4


class MergeMiningRequest(BaseModel):
    chains: List[str]


class TransactionRequest(BaseModel):
    sender: str
    recipient: str
    amount: float
    fee: float = 0.001


# Initialize blockchain and miner (singleton)
blockchain = SphinxSkynetBlockchain()
miner: Optional[SphinxMiner] = None
merge_coordinator: Optional[MergeMiningCoordinator] = None


# Create FastAPI app
app = FastAPI(
    title="SphinxSkynet Mining API",
    description="API for SphinxSkynet blockchain mining operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SphinxSkynet Mining API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/mining/start")
async def start_mining(request: StartMiningRequest):
    """Start mining"""
    global miner
    
    if miner and miner.is_mining:
        raise HTTPException(status_code=400, detail="Mining already running")
    
    try:
        miner = SphinxMiner(
            blockchain=blockchain,
            miner_address=request.miner_address,
            algorithm=request.algorithm,
            num_threads=request.num_threads
        )
        
        miner.start_mining()
        
        return {
            "status": "started",
            "miner_address": request.miner_address,
            "algorithm": request.algorithm,
            "threads": request.num_threads
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mining/stop")
async def stop_mining():
    """Stop mining"""
    global miner
    
    if not miner or not miner.is_mining:
        raise HTTPException(status_code=400, detail="Mining not running")
    
    try:
        miner.stop_mining()
        return {"status": "stopped"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mining/status")
async def get_mining_status():
    """Get mining status"""
    if not miner:
        return {
            "is_mining": False,
            "miner_address": None
        }
    
    return miner.get_stats()


@app.get("/api/mining/hashrate")
async def get_hashrate():
    """Get current hashrate"""
    if not miner:
        return {"hashrate": 0}
    
    return {"hashrate": miner.get_hashrate()}


@app.get("/api/mining/rewards")
async def get_rewards():
    """Get mining rewards"""
    if not miner:
        return {"total_rewards": 0, "blocks_mined": 0}
    
    stats = miner.get_stats()
    return {
        "total_rewards": stats['total_rewards'],
        "blocks_mined": stats['blocks_mined'],
        "average_phi_score": stats['average_phi_score']
    }


@app.post("/api/mining/merge/enable")
async def enable_merge_mining(request: MergeMiningRequest):
    """Enable merge mining"""
    global merge_coordinator
    
    if not miner:
        raise HTTPException(status_code=400, detail="Mining not started")
    
    try:
        if not merge_coordinator:
            merge_coordinator = MergeMiningCoordinator(miner)
        
        for chain in request.chains:
            merge_coordinator.enable_chain(chain)
        
        return {
            "status": "enabled",
            "chains": merge_coordinator.enabled_chains
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/blocks")
async def get_recent_blocks(limit: int = 10):
    """Get recent blocks"""
    try:
        chain = blockchain.chain[-limit:]
        return {
            "blocks": [block.to_dict() for block in reversed(chain)],
            "count": len(chain)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/blocks/{block_hash}")
async def get_block_by_hash(block_hash: str):
    """Get specific block"""
    try:
        block = blockchain.get_block_by_hash(block_hash)
        
        if not block:
            raise HTTPException(status_code=404, detail="Block not found")
        
        return block.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transactions")
async def get_transaction_pool():
    """Get transaction pool"""
    try:
        return {
            "transactions": [tx.to_dict() for tx in blockchain.transaction_pool],
            "count": len(blockchain.transaction_pool)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transactions")
async def submit_transaction(request: TransactionRequest):
    """Submit new transaction"""
    try:
        # Create transaction (simplified - would need proper UTXO handling)
        from sphinx_os.blockchain.transaction import Transaction, TransactionInput, TransactionOutput
        
        # This is simplified - in production, need to:
        # 1. Look up sender's UTXOs
        # 2. Create proper inputs with signatures
        # 3. Validate sender has sufficient balance
        
        outputs = [TransactionOutput(address=request.recipient, amount=request.amount)]
        tx = Transaction(inputs=[], outputs=outputs, fee=request.fee)
        
        if blockchain.add_transaction(tx):
            return {
                "status": "accepted",
                "txid": tx.txid
            }
        else:
            raise HTTPException(status_code=400, detail="Transaction rejected")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chain/stats")
async def get_chain_stats():
    """Get blockchain statistics"""
    try:
        return blockchain.get_chain_stats()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/consciousness")
async def get_consciousness_metrics(block_index: Optional[int] = None):
    """
    Compute IIT (Integrated Information Theory) consciousness metrics.

    Returns von Neumann entropy, density matrix, eigenvalue spectrum,
    network adjacency matrix, Φ timeline, and consensus validation for
    the specified block (or latest block if none given).
    """
    try:
        # Determine seed from block index or chain length
        if block_index is not None:
            seed = block_index
        else:
            stats = blockchain.get_chain_stats()
            seed = stats.get("chain_length", 0)

        rng = np.random.default_rng(seed % (2 ** 31))

        # --- 9×9 Network adjacency matrix (raw, before zeroing diagonal) ---
        n_nodes = 9
        A_raw = rng.random((n_nodes, n_nodes))
        A_raw = (A_raw + A_raw.T) / 2        # symmetric

        # Display version: zero diagonal (no self-loops)
        A_display = np.round(A_raw.copy(), 1)
        np.fill_diagonal(A_display, 0)

        # --- 8×8 density matrix ρ = A_S / Tr(A_S) ---
        # Use raw matrix (non-zero diagonal) so Tr(A_S) ≠ 0
        dim = 8
        A_sub = A_raw[:dim, :dim].copy()
        trace_A = float(np.trace(A_sub))
        if trace_A <= 0:
            trace_A = 1.0
        rho = A_sub / trace_A

        # --- Eigenvalue spectrum ---
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[::-1]          # descending order

        # --- Von Neumann entropy Φ_S = -Σ λₖ log₂(λₖ) ---
        entropy_bits = float(
            -sum(lam * math.log2(lam) for lam in eigenvalues if lam > 1e-10)
        )

        # --- Normalised Φ ∈ [0, 1] ---
        max_entropy = math.log2(dim)
        phi = entropy_bits / max_entropy if max_entropy > 0 else 0.0

        # --- IIT bonus = e^Φ ---
        iit_bonus = math.exp(phi)

        # --- GWT broadcast score (simplified global workspace theory) ---
        gwt_score = float(0.5 * phi + 0.3)

        # --- Φ_total = α·Φ_IIT + β·GWT ---
        alpha, beta = 0.7, 0.3
        phi_total = alpha * entropy_bits + beta * gwt_score

        # --- Consensus: Φ_total > log₂(n_nodes) ---
        consensus_threshold = math.log2(n_nodes)
        consensus_valid = bool(phi_total > consensus_threshold)

        # --- Consciousness level ---
        if phi > 0.8:
            level = "🧠 COSMIC"
        elif phi > 0.6:
            level = "🌟 SELF AWARE"
        elif phi > 0.4:
            level = "✨ SENTIENT"
        elif phi > 0.2:
            level = "🔵 AWARE"
        else:
            level = "⚫ UNCONSCIOUS"

        # --- Φ timeline (5 samples up to current seed) ---
        phi_timeline = []
        for i in range(5):
            s = (seed - (4 - i)) % (2 ** 31)
            rng_t = np.random.default_rng(s)
            A_t = rng_t.random((dim, dim))
            A_t = (A_t + A_t.T) / 2
            tr_t = float(np.trace(A_t))
            rho_t = A_t / tr_t if tr_t > 0 else A_t
            eigs_t = np.linalg.eigvalsh(rho_t)
            ent_t = float(
                -sum(e * math.log2(e) for e in eigs_t if e > 1e-10)
            )
            phi_t = ent_t / max_entropy
            phi_timeline.append({"sample": i, "phi": round(phi_t, 3)})

        return {
            "block_index": seed,
            "phi": round(phi, 4),
            "entropy_bits": round(entropy_bits, 4),
            "iit_bonus": round(iit_bonus, 4),
            "phi_total": round(phi_total, 4),
            "gwt_score": round(gwt_score, 4),
            "alpha": alpha,
            "beta": beta,
            "consensus_valid": consensus_valid,
            "consensus_threshold": round(consensus_threshold, 4),
            "n_nodes": n_nodes,
            "consciousness_level": level,
            "lambda_max": round(float(np.max(eigenvalues)), 4),
            "eigenvalues": [round(float(e), 4) for e in eigenvalues],
            "adjacency_matrix": [[round(float(v), 1) for v in row] for row in A_display.tolist()],
            "density_matrix": [[round(float(v), 2) for v in row] for row in rho.tolist()],
            "phi_timeline": phi_timeline,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
