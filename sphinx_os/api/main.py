"""
FastAPI Main Application
SphinxSkynet Gasless Blockchain API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

from ..blockchain.standalone import StandaloneSphinxBlockchain
from ..wallet.builtin_wallet import BuiltInWallet, WalletManager
from ..mining.free_miner import FreeMiner, MiningTier, MiningPool
from ..revenue.fee_collector import FeeCollector
from ..revenue.subscriptions import SubscriptionManager, SubscriptionTier
from ..revenue.referrals import ReferralProgram


# ==================== FastAPI App Setup ====================

app = FastAPI(
    title="SphinxSkynet Blockchain API",
    description="Gasless blockchain with built-in wallet and free mining",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Global Instances ====================

# Initialize blockchain
blockchain = StandaloneSphinxBlockchain(db_path="data/sphinxskynet.db")

# Initialize wallet manager
wallet_manager = WalletManager(storage_path="data/wallets")

# Initialize mining pool
mining_pool = MiningPool()

# Initialize revenue systems
operator_address = os.getenv("OPERATOR_ADDRESS", "0xSPHINXOPERATOR")
fee_collector = FeeCollector(operator_address, db_path="data/revenue.db")
subscription_manager = SubscriptionManager(db_path="data/subscriptions.db")
referral_program = ReferralProgram(db_path="data/referrals.db")


# ==================== Pydantic Models ====================

class CreateWalletRequest(BaseModel):
    name: str = "default"


class ImportWalletRequest(BaseModel):
    name: str
    private_key: Optional[str] = None
    mnemonic: Optional[str] = None


class TransactionRequest(BaseModel):
    from_address: str
    to_address: str
    amount: float
    private_key: str


class StartMiningRequest(BaseModel):
    address: str
    tier: str = "free"


class UpgradeSubscriptionRequest(BaseModel):
    user_id: str
    tier: str
    payment_method_id: Optional[str] = None


class ReferralSignupRequest(BaseModel):
    user_id: str
    referral_code: Optional[str] = None


# ==================== Health Check ====================

@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "name": "SphinxSkynet Blockchain API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Gasless blockchain",
            "Built-in wallet",
            "Free mining",
            "Premium subscriptions"
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "blockchain": {
            "chain_length": len(blockchain.chain),
            "is_valid": blockchain.is_chain_valid()
        }
    }


# ==================== Blockchain Endpoints ====================

@app.get("/api/blockchain/info")
def get_blockchain_info():
    """Get blockchain information"""
    return blockchain.get_chain_info()


@app.get("/api/blockchain/chain")
def get_chain():
    """Get the entire blockchain"""
    return {
        "chain": [block.to_dict() for block in blockchain.chain],
        "length": len(blockchain.chain)
    }


@app.get("/api/blockchain/block/{index}")
def get_block(index: int):
    """Get a specific block by index"""
    if index < 0 or index >= len(blockchain.chain):
        raise HTTPException(status_code=404, detail="Block not found")
    
    return blockchain.chain[index].to_dict()


# ==================== Wallet Endpoints ====================

@app.post("/api/wallet/create")
def create_wallet(request: CreateWalletRequest):
    """
    Create a new wallet - 100% free!
    NO MetaMask, NO gas fees
    """
    wallet = wallet_manager.create_wallet(request.name)
    
    return {
        "success": True,
        "wallet": {
            "name": request.name,
            "address": wallet.address,
            "private_key": wallet.private_key,
            "mnemonic": wallet.mnemonic,
            "warning": "⚠️ Save your private key and mnemonic securely!"
        }
    }


@app.post("/api/wallet/import")
def import_wallet(request: ImportWalletRequest):
    """Import an existing wallet"""
    wallet = wallet_manager.import_wallet(
        request.name,
        private_key=request.private_key,
        mnemonic=request.mnemonic
    )
    
    return {
        "success": True,
        "wallet": {
            "name": request.name,
            "address": wallet.address
        }
    }


@app.get("/api/wallet/{address}/balance")
def get_balance(address: str):
    """Get wallet balance"""
    balance = blockchain.get_balance(address)
    
    return {
        "address": address,
        "balance": balance,
        "token": "SPHINX"
    }


@app.get("/api/wallet/{address}/transactions")
def get_transaction_history(address: str, limit: int = 100):
    """Get transaction history for an address"""
    transactions = blockchain.get_transaction_history(address, limit)
    
    return {
        "address": address,
        "transactions": transactions,
        "count": len(transactions)
    }


@app.get("/api/wallet/list")
def list_wallets():
    """List all wallets"""
    wallets = wallet_manager.list_wallets()
    
    return {
        "wallets": wallets,
        "count": len(wallets)
    }


# ==================== Transaction Endpoints ====================

@app.post("/api/transaction/send")
def send_transaction(request: TransactionRequest):
    """
    Send SPHINX tokens - fee paid in SPHINX (not ETH)
    Fee: 0.001 SPHINX per transaction
    """
    try:
        # Create transaction
        tx = blockchain.create_transaction(
            request.from_address,
            request.to_address,
            request.amount
        )
        
        # Sign transaction (in production, verify private key matches address)
        tx.sign_transaction(request.private_key)
        
        # Add to pending transactions
        blockchain.add_transaction(tx)
        
        # Collect transaction fee
        fee_collector.collect_transaction_fee(tx.tx_hash)
        
        return {
            "success": True,
            "transaction": {
                "hash": tx.tx_hash,
                "from": tx.from_address,
                "to": tx.to_address,
                "amount": tx.amount,
                "fee": tx.TRANSACTION_FEE,
                "status": "pending"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Mining Endpoints ====================

@app.post("/api/mining/start")
def start_mining(request: StartMiningRequest):
    """
    Start mining - NO gas costs!
    Free tier: 10 MH/s
    """
    try:
        # Parse tier
        tier = MiningTier(request.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mining tier")
    
    # Get or create miner
    miner = mining_pool.get_miner(request.address)
    if not miner:
        miner = mining_pool.add_miner(request.address, tier)
    
    # Start mining
    miner.start_mining()
    
    return {
        "success": True,
        "miner": {
            "address": request.address,
            "tier": tier.value,
            "hashrate": miner.config.hashrate,
            "status": "mining"
        }
    }


@app.post("/api/mining/stop")
def stop_mining(address: str):
    """Stop mining"""
    miner = mining_pool.get_miner(address)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    
    miner.stop_mining()
    
    return {
        "success": True,
        "message": "Mining stopped"
    }


@app.post("/api/mining/mine-block")
def mine_block(address: str):
    """
    Mine a block with pending transactions
    FREE - NO gas costs!
    """
    try:
        # Mine block
        block = blockchain.mine_pending_transactions(address)
        
        # Get miner and update stats
        miner = mining_pool.get_miner(address)
        if miner:
            # Simulate mining result
            result = {
                'success': True,
                'earned': blockchain.mining_reward,
                'block': block.index
            }
            miner.blocks_found += 1
            miner.total_earned += blockchain.mining_reward
        
        return {
            "success": True,
            "block": block.to_dict(),
            "reward": blockchain.mining_reward
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/mining/{address}/stats")
def get_mining_stats(address: str):
    """Get mining statistics"""
    miner = mining_pool.get_miner(address)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    
    stats = miner.get_stats()
    blockchain_stats = blockchain.get_mining_stats(address)
    
    return {
        **stats,
        **blockchain_stats
    }


@app.get("/api/mining/{address}/earnings")
def get_estimated_earnings(address: str, hours: int = 24):
    """Calculate estimated earnings"""
    miner = mining_pool.get_miner(address)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    
    return miner.calculate_estimated_earnings(hours)


@app.get("/api/mining/pool/stats")
def get_pool_stats():
    """Get mining pool statistics"""
    return mining_pool.get_pool_stats()


# ==================== Subscription Endpoints ====================

@app.post("/api/subscription/upgrade")
def upgrade_subscription(request: UpgradeSubscriptionRequest):
    """
    Upgrade to premium mining - $5/month for 100 MH/s
    """
    try:
        # Parse tier
        tier = SubscriptionTier(request.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subscription tier")
    
    # Create/upgrade subscription
    subscription = subscription_manager.upgrade_subscription(request.user_id, tier)
    
    # Update miner tier if exists
    miner = mining_pool.get_miner(request.user_id)
    if miner:
        mining_tier = MiningTier(request.tier)
        miner.upgrade_tier(mining_tier)
    
    # In production, process payment with Stripe here
    # For now, just record the subscription
    
    return {
        "success": True,
        "subscription": subscription
    }


@app.get("/api/subscription/{user_id}")
def get_subscription(user_id: str):
    """Get user's subscription details"""
    subscription = subscription_manager.get_subscription(user_id)
    
    if not subscription:
        return {
            "user_id": user_id,
            "tier": "free",
            "status": "none"
        }
    
    return subscription


@app.get("/api/subscription/tiers")
def get_subscription_tiers():
    """Get available subscription tiers"""
    from ..revenue.subscriptions import TIER_PRICING
    
    return {
        "tiers": [
            subscription_manager.get_tier_info(tier)
            for tier in SubscriptionTier
        ]
    }


@app.get("/api/subscription/stats")
def get_subscription_stats():
    """Get subscription statistics (admin only)"""
    return subscription_manager.get_subscription_stats()


# ==================== Referral Endpoints ====================

@app.post("/api/referral/signup")
def referral_signup(request: ReferralSignupRequest):
    """Sign up with referral code"""
    if request.referral_code:
        # Get referrer
        referrer_id = referral_program.get_referrer_by_code(request.referral_code)
        
        if referrer_id:
            # Track referral
            referral_program.track_referral(referrer_id, request.user_id)
            
            return {
                "success": True,
                "message": "Referral tracked successfully",
                "referrer_id": referrer_id
            }
    
    # Generate referral code for new user
    code = referral_program.generate_referral_code(request.user_id)
    
    return {
        "success": True,
        "referral_code": code
    }


@app.get("/api/referral/{user_id}/code")
def get_referral_code(user_id: str):
    """Get user's referral code"""
    code = referral_program.get_referral_code(user_id)
    
    return {
        "user_id": user_id,
        "referral_code": code,
        "commission_rate": referral_program.COMMISSION_RATE
    }


@app.get("/api/referral/{user_id}/stats")
def get_referral_stats(user_id: str):
    """Get referral statistics"""
    return referral_program.get_referral_stats(user_id)


@app.get("/api/referral/{user_id}/referrals")
def get_referrals(user_id: str):
    """Get list of referrals"""
    return {
        "referrals": referral_program.get_referrals(user_id)
    }


# ==================== Revenue/Admin Endpoints ====================

@app.get("/api/admin/revenue/today")
def get_today_revenue():
    """Get today's revenue (admin only)"""
    return fee_collector.get_daily_revenue()


@app.get("/api/admin/revenue/total")
def get_total_revenue():
    """Get total revenue (admin only)"""
    return fee_collector.get_total_revenue()


@app.get("/api/admin/revenue/history")
def get_revenue_history(days: int = 30):
    """Get revenue history (admin only)"""
    return {
        "history": fee_collector.get_revenue_history(days)
    }


@app.get("/api/admin/revenue/stats")
def get_revenue_stats():
    """Get comprehensive revenue statistics (admin only)"""
    fee_stats = fee_collector.get_revenue_stats()
    sub_stats = subscription_manager.get_subscription_stats()
    ref_stats = referral_program.get_program_stats()
    
    return {
        "revenue": fee_stats,
        "subscriptions": sub_stats,
        "referrals": ref_stats
    }


# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
