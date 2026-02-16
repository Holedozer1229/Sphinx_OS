"""
Free Mining System - NO gas costs, runs in browser!
"""

import time
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class MiningTier(Enum):
    """Mining subscription tiers"""
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


@dataclass
class TierConfig:
    """Configuration for each mining tier"""
    name: str
    hashrate: str  # Human-readable hashrate
    hashrate_value: int  # Numerical hashrate in MH/s
    cost: float  # USD per month
    daily_limit: str  # Daily mining limit


# Tier configurations
MINING_TIERS = {
    MiningTier.FREE: TierConfig(
        name="Free",
        hashrate="10 MH/s",
        hashrate_value=10,
        cost=0.0,
        daily_limit="1000 SPHINX"
    ),
    MiningTier.PREMIUM: TierConfig(
        name="Premium",
        hashrate="100 MH/s",
        hashrate_value=100,
        cost=5.0,
        daily_limit="10000 SPHINX"
    ),
    MiningTier.PRO: TierConfig(
        name="Pro",
        hashrate="1000 MH/s",
        hashrate_value=1000,
        cost=20.0,
        daily_limit="unlimited"
    )
}


class FreeMiner:
    """
    Free mining - NO gas costs, runs in browser!
    
    Features:
    - Browser-based mining (WebAssembly)
    - NO gas fees for mining
    - Instant payouts (NO gas for withdrawals)
    - Free tier: 10 MH/s
    - Premium tier: 100 MH/s ($5/month subscription)
    """
    
    def __init__(self, address: str, tier: MiningTier = MiningTier.FREE):
        """
        Initialize miner
        
        Args:
            address: Wallet address
            tier: Mining tier (free, premium, or pro)
        """
        self.address = address
        self.tier = tier
        self.config = MINING_TIERS[tier]
        
        # Mining statistics
        self.start_time: Optional[float] = None
        self.total_hashes = 0
        self.blocks_found = 0
        self.total_earned = 0.0
        self.daily_earned = 0.0
        self.last_reset = time.time()
        
        # Mining state
        self.is_mining = False
    
    def start_mining(self):
        """Start mining"""
        if self.is_mining:
            return
        
        self.is_mining = True
        self.start_time = time.time()
        print(f"⛏️  Mining started at {self.config.hashrate}")
    
    def stop_mining(self):
        """Stop mining"""
        if not self.is_mining:
            return
        
        self.is_mining = False
        mining_time = time.time() - (self.start_time or time.time())
        print(f"⛏️  Mining stopped. Duration: {mining_time:.2f}s")
    
    def mine_block(self, difficulty: int = 4) -> Dict:
        """
        Simulate mining a block
        
        Args:
            difficulty: Mining difficulty
            
        Returns:
            Mining result dictionary
        """
        if not self.is_mining:
            raise ValueError("Mining not started")
        
        # Check daily limit
        if self._is_daily_limit_reached():
            return {
                'success': False,
                'reason': 'daily_limit_reached',
                'earned': 0.0
            }
        
        # Simulate mining based on hashrate
        # Higher hashrate = faster block finding
        hashes_needed = 2 ** difficulty * 1000000  # Approximate hashes for difficulty
        time_to_mine = hashes_needed / (self.config.hashrate_value * 1000000)
        
        # Simulate time passing
        time.sleep(min(time_to_mine, 0.1))  # Cap at 0.1s for simulation
        
        # Calculate reward (base reward * tier multiplier)
        base_reward = 50.0  # Base SPHINX reward
        tier_multiplier = self.config.hashrate_value / 10  # Scale with hashrate
        reward = base_reward * min(tier_multiplier / 100, 1.0)  # Cap multiplier
        
        # Update statistics
        self.total_hashes += hashes_needed
        self.blocks_found += 1
        self.total_earned += reward
        self.daily_earned += reward
        
        return {
            'success': True,
            'earned': reward,
            'time': time_to_mine,
            'hashes': hashes_needed,
            'block': self.blocks_found
        }
    
    def get_stats(self) -> Dict:
        """
        Get mining statistics
        
        Returns:
            Statistics dictionary
        """
        mining_time = 0.0
        if self.is_mining and self.start_time:
            mining_time = time.time() - self.start_time
        
        return {
            'address': self.address,
            'tier': self.tier.value,
            'hashrate': self.config.hashrate,
            'is_mining': self.is_mining,
            'mining_time': mining_time,
            'total_hashes': self.total_hashes,
            'blocks_found': self.blocks_found,
            'total_earned': self.total_earned,
            'daily_earned': self.daily_earned,
            'daily_limit': self.config.daily_limit,
            'daily_limit_reached': self._is_daily_limit_reached()
        }
    
    def upgrade_tier(self, new_tier: MiningTier):
        """
        Upgrade to a new mining tier
        
        Args:
            new_tier: New mining tier
        """
        old_tier = self.tier
        self.tier = new_tier
        self.config = MINING_TIERS[new_tier]
        
        print(f"⬆️  Upgraded from {old_tier.value} to {new_tier.value}")
        print(f"   New hashrate: {self.config.hashrate}")
    
    def reset_daily_limit(self):
        """Reset daily mining limit"""
        self.daily_earned = 0.0
        self.last_reset = time.time()
    
    def _is_daily_limit_reached(self) -> bool:
        """Check if daily mining limit is reached"""
        # Check if we need to reset (24 hours passed)
        if time.time() - self.last_reset > 86400:  # 24 hours
            self.reset_daily_limit()
        
        # Pro tier has no limit
        if self.tier == MiningTier.PRO:
            return False
        
        # Check limit for other tiers
        if self.tier == MiningTier.FREE:
            return self.daily_earned >= 1000.0
        elif self.tier == MiningTier.PREMIUM:
            return self.daily_earned >= 10000.0
        
        return False
    
    def calculate_estimated_earnings(self, hours: int = 24) -> Dict:
        """
        Calculate estimated earnings over a time period
        
        Args:
            hours: Number of hours to estimate
            
        Returns:
            Earnings estimate
        """
        # Calculate blocks per hour based on hashrate
        # Assuming average difficulty of 4
        blocks_per_hour = self.config.hashrate_value / 10  # Simplified calculation
        
        base_reward = 50.0
        tier_multiplier = self.config.hashrate_value / 10
        reward_per_block = base_reward * min(tier_multiplier / 100, 1.0)
        
        estimated_blocks = blocks_per_hour * hours
        estimated_earnings = estimated_blocks * reward_per_block
        
        # Apply daily limits
        if self.tier == MiningTier.FREE:
            estimated_earnings = min(estimated_earnings, 1000.0 * (hours / 24))
        elif self.tier == MiningTier.PREMIUM:
            estimated_earnings = min(estimated_earnings, 10000.0 * (hours / 24))
        
        return {
            'hours': hours,
            'estimated_blocks': estimated_blocks,
            'estimated_earnings': estimated_earnings,
            'tier': self.tier.value,
            'hashrate': self.config.hashrate
        }
    
    def __repr__(self):
        return (
            f"FreeMiner(address={self.address[:8]}..., "
            f"tier={self.tier.value}, "
            f"mining={self.is_mining})"
        )


class MiningPool:
    """
    Manage multiple miners
    """
    
    def __init__(self):
        """Initialize mining pool"""
        self.miners: Dict[str, FreeMiner] = {}
    
    def add_miner(self, address: str, tier: MiningTier = MiningTier.FREE) -> FreeMiner:
        """
        Add a miner to the pool
        
        Args:
            address: Wallet address
            tier: Mining tier
            
        Returns:
            FreeMiner instance
        """
        if address in self.miners:
            return self.miners[address]
        
        miner = FreeMiner(address, tier)
        self.miners[address] = miner
        return miner
    
    def get_miner(self, address: str) -> Optional[FreeMiner]:
        """
        Get a miner by address
        
        Args:
            address: Wallet address
            
        Returns:
            FreeMiner instance or None
        """
        return self.miners.get(address)
    
    def get_pool_stats(self) -> Dict:
        """
        Get mining pool statistics
        
        Returns:
            Pool statistics
        """
        total_miners = len(self.miners)
        active_miners = sum(1 for m in self.miners.values() if m.is_mining)
        total_hashrate = sum(m.config.hashrate_value for m in self.miners.values())
        total_blocks = sum(m.blocks_found for m in self.miners.values())
        
        return {
            'total_miners': total_miners,
            'active_miners': active_miners,
            'total_hashrate': f"{total_hashrate} MH/s",
            'total_blocks_found': total_blocks
        }
    
    def __repr__(self):
        return f"MiningPool({len(self.miners)} miners)"
