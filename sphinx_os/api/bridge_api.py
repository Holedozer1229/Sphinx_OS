"""
Bridge API for SphinxSkynet Cross-Chain Bridge
FastAPI endpoints for bridge operations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.bridge.bridge import CrossChainBridge
from sphinx_os.bridge.relayer import BridgeRelayer


# Pydantic models for requests
class LockRequest(BaseModel):
    source_chain: str
    amount: float
    sender: str
    recipient: str


class MintRequest(BaseModel):
    tx_hash: str
    recipient: str
    signatures: List[str]


class BurnRequest(BaseModel):
    amount: float
    sender: str
    destination_chain: str
    recipient: str


class ReleaseRequest(BaseModel):
    tx_hash: str
    recipient: str
    signatures: List[str]


# Initialize bridge and relayer (singleton)
bridge = CrossChainBridge()
relayer = BridgeRelayer(bridge)
relayer.start()


# Create FastAPI app
app = FastAPI(
    title="SphinxSkynet Bridge API",
    description="API for SphinxSkynet cross-chain bridge operations",
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
        "name": "SphinxSkynet Bridge API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/api/bridge/lock")
async def lock_tokens(request: LockRequest):
    """Lock tokens for bridging"""
    try:
        tx_hash = bridge.lock_tokens(
            source_chain=request.source_chain,
            amount=request.amount,
            sender=request.sender,
            recipient=request.recipient
        )
        
        if not tx_hash:
            raise HTTPException(status_code=400, detail="Failed to lock tokens")
        
        # Queue for minting
        relayer.queue_mint(tx_hash)
        
        return {
            "status": "locked",
            "tx_hash": tx_hash,
            "source_chain": request.source_chain,
            "amount": request.amount,
            "fee": request.amount * bridge.BRIDGE_FEE
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bridge/mint")
async def mint_tokens(request: MintRequest):
    """Mint wrapped tokens"""
    try:
        success = bridge.mint_wrapped_tokens(
            tx_hash=request.tx_hash,
            recipient=request.recipient,
            signatures=request.signatures
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to mint tokens")
        
        return {
            "status": "minted",
            "tx_hash": request.tx_hash,
            "recipient": request.recipient
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bridge/burn")
async def burn_tokens(request: BurnRequest):
    """Burn wrapped tokens"""
    try:
        tx_hash = bridge.burn_wrapped_tokens(
            amount=request.amount,
            sender=request.sender,
            destination_chain=request.destination_chain,
            recipient=request.recipient
        )
        
        if not tx_hash:
            raise HTTPException(status_code=400, detail="Failed to burn tokens")
        
        # Queue for release
        relayer.queue_release(tx_hash)
        
        return {
            "status": "burned",
            "tx_hash": tx_hash,
            "destination_chain": request.destination_chain,
            "amount": request.amount,
            "fee": request.amount * bridge.BRIDGE_FEE
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bridge/release")
async def release_tokens(request: ReleaseRequest):
    """Release locked tokens"""
    try:
        success = bridge.release_tokens(
            tx_hash=request.tx_hash,
            recipient=request.recipient,
            signatures=request.signatures
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to release tokens")
        
        return {
            "status": "released",
            "tx_hash": request.tx_hash,
            "recipient": request.recipient
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bridge/status/{tx_hash}")
async def get_bridge_status(tx_hash: str):
    """Get bridge transaction status"""
    try:
        status = bridge.get_transaction_status(tx_hash)
        
        if not status:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bridge/supported-chains")
async def get_supported_chains():
    """Get supported chains"""
    try:
        return {
            "chains": bridge.get_supported_chains(),
            "count": len(bridge.SUPPORTED_CHAINS)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bridge/fees")
async def get_bridge_fees():
    """Get bridge fees"""
    return {
        "bridge_fee_percent": bridge.BRIDGE_FEE * 100,
        "example": {
            "amount": 100,
            "fee": 100 * bridge.BRIDGE_FEE,
            "net_amount": 100 * (1 - bridge.BRIDGE_FEE)
        }
    }


@app.get("/api/bridge/balance/{address}")
async def get_bridge_balance(address: str, chain: Optional[str] = None):
    """Get bridge balance for address"""
    try:
        if chain:
            # Get locked balance on specific chain
            locked = bridge.get_locked_balance(chain, address)
            return {
                "address": address,
                "chain": chain,
                "locked_balance": locked
            }
        else:
            # Get wrapped balance
            wrapped = bridge.get_wrapped_balance(address)
            return {
                "address": address,
                "wrapped_balance": wrapped
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bridge/stats")
async def get_bridge_stats():
    """Get bridge statistics"""
    try:
        bridge_stats = bridge.get_bridge_stats()
        relayer_stats = relayer.get_stats()
        
        return {
            "bridge": bridge_stats,
            "relayer": relayer_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bridge/health")
async def health_check():
    """Health check endpoint"""
    relayer_stats = relayer.get_stats()
    
    return {
        "status": "healthy" if relayer_stats['is_running'] else "degraded",
        "relayer_running": relayer_stats['is_running'],
        "pending_operations": relayer_stats['pending_mints'] + relayer_stats['pending_releases']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
