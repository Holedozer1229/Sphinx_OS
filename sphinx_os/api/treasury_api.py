"""
Treasury Dashboard API for SphinxSkynet

FastAPI endpoints for treasury management and bridge deployment
"""
from fastapi import APIRouter, HTTPException
from sphinx_os.treasury.self_funding import SelfFundingTreasury

router = APIRouter(prefix="/api/treasury", tags=["treasury"])

# Global treasury instance (in production, this should be a singleton or database-backed)
_treasury_instance = None


def get_treasury():
    """Get or create treasury instance"""
    global _treasury_instance
    if _treasury_instance is None:
        _treasury_instance = SelfFundingTreasury()
    return _treasury_instance


@router.get("/stats")
def get_treasury_stats():
    """
    Get current treasury statistics
    
    Returns:
        Treasury balance and deployment readiness
    """
    treasury = get_treasury()
    return treasury.get_treasury_stats()


@router.get("/deployments")
def get_deployment_status():
    """
    Get status of all bridge deployments
    
    Returns:
        Deployment status for each chain
    """
    treasury = get_treasury()
    stats = treasury.get_treasury_stats()
    return {
        "deployments": stats["deployments"],
        "treasury_balance": stats["balance_usd"]
    }


@router.post("/deploy/{chain}")
def manual_deploy(chain: str):
    """
    Manually trigger deployment if treasury has funds
    
    Args:
        chain: Target blockchain
    
    Returns:
        Deployment result
    """
    treasury = get_treasury()
    config = treasury.deployment_targets.get(chain)
    
    if not config:
        raise HTTPException(status_code=400, detail=f"Invalid chain: {chain}")
    
    if config["deployed"]:
        raise HTTPException(status_code=400, detail=f"{chain} bridge already deployed")
    
    if treasury.treasury_balance < config["threshold"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient treasury funds. Required: ${config['threshold']}, Current: ${treasury.treasury_balance}"
        )
    
    success = treasury.trigger_deployment(chain, config["threshold"])
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to deploy {chain} bridge")
    
    return {
        "success": success,
        "chain": chain,
        "remaining_balance": treasury.treasury_balance
    }


@router.post("/collect/nft_mint")
def collect_nft_mint_fee(amount: float):
    """
    Collect NFT mint fee (for testing/simulation)
    
    Args:
        amount: Fee amount in SPHINX
    
    Returns:
        Fee distribution
    """
    treasury = get_treasury()
    distribution = treasury.collect_nft_mint_fee(amount)
    return {
        "success": True,
        "distribution": distribution,
        "new_balance": treasury.treasury_balance
    }


@router.post("/collect/rarity_proof")
def collect_rarity_proof_fee(amount: float):
    """
    Collect rarity proof fee (for testing/simulation)
    
    Args:
        amount: Fee amount in SPHINX
    
    Returns:
        Fee distribution
    """
    treasury = get_treasury()
    distribution = treasury.collect_rarity_proof_fee(amount)
    return {
        "success": True,
        "distribution": distribution,
        "new_balance": treasury.treasury_balance
    }
