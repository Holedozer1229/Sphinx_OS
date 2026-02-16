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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
