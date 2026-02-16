"""
STX → BTC Yield Calculator

Implements the mathematical formulas for yield distribution including:
- Pool efficiency calculations
- Treasury split based on spectral integration score (Φ)
- NFT yield multipliers
- User payout calculations
"""

import math
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class YieldResult:
    """Result of yield calculation"""
    total_reward: float
    treasury_share: float
    user_payout: float
    nft_multiplier: float
    effective_payout: float


class YieldCalculator:
    """
    Calculate BTC yield distribution for STX delegation.
    
    Formula:
    R = α · (S / ΣS_i) · R_total
    
    Where:
    - S: STX delegated by user
    - R: BTC reward per cycle
    - Φ: Spectral integration score  
    - α: Pool efficiency (0.92-0.98)
    - τ: Treasury cut
    - U: User yield
    """
    
    # Constants
    MIN_POOL_EFFICIENCY = 0.92
    MAX_POOL_EFFICIENCY = 0.98
    BASE_TREASURY_RATE = 0.05
    MAX_TREASURY_RATE = 0.30
    PHI_TREASURY_FACTOR = 2000.0
    
    def __init__(self, pool_efficiency: float = 0.95):
        """
        Initialize yield calculator.
        
        Args:
            pool_efficiency: Pool efficiency factor (0.92-0.98)
        """
        if not (self.MIN_POOL_EFFICIENCY <= pool_efficiency <= self.MAX_POOL_EFFICIENCY):
            raise ValueError(
                f"Pool efficiency must be between {self.MIN_POOL_EFFICIENCY} "
                f"and {self.MAX_POOL_EFFICIENCY}"
            )
        self.pool_efficiency = pool_efficiency
    
    def calculate_base_reward(
        self,
        stx_delegated: float,
        total_stx_pool: float,
        total_btc_reward: float
    ) -> float:
        """
        Calculate base BTC reward before treasury split.
        
        R = α · (S / ΣS_i) · R_total
        
        Args:
            stx_delegated: Amount of STX user delegated
            total_stx_pool: Total STX in the pool
            total_btc_reward: Total BTC rewards for the cycle
            
        Returns:
            Base BTC reward for user
        """
        if total_stx_pool <= 0:
            return 0.0
        
        pool_share = stx_delegated / total_stx_pool
        base_reward = self.pool_efficiency * pool_share * total_btc_reward
        
        return base_reward
    
    def calculate_treasury_split(self, phi_score: float) -> float:
        """
        Calculate treasury split based on spectral integration score.
        
        R_T = R · min(0.30, 0.05 + Φ/2000)
        
        Properties:
        - High Φ increases protocol revenue
        - Treasury never exceeds 30%
        - Sybil resistant (Φ monotonic)
        
        Args:
            phi_score: Spectral integration score (Φ)
            
        Returns:
            Treasury split ratio (0.05 - 0.30)
        """
        phi_contribution = phi_score / self.PHI_TREASURY_FACTOR
        treasury_rate = self.BASE_TREASURY_RATE + phi_contribution
        
        return min(self.MAX_TREASURY_RATE, treasury_rate)
    
    def calculate_nft_multiplier(self, phi_score: float, has_nft: bool = False) -> float:
        """
        Calculate NFT yield multiplier.
        
        U' = U · (1 + log₂(1 + Φ/500))
        
        Args:
            phi_score: Spectral integration score (Φ)
            has_nft: Whether user holds rarity NFT
            
        Returns:
            Yield multiplier (1.0 if no NFT, >1.0 if has NFT)
        """
        if not has_nft:
            return 1.0
        
        multiplier = 1.0 + math.log2(1.0 + phi_score / 500.0)
        return multiplier
    
    def calculate_yield(
        self,
        stx_delegated: float,
        total_stx_pool: float,
        total_btc_reward: float,
        phi_score: float,
        has_nft: bool = False
    ) -> YieldResult:
        """
        Calculate complete yield distribution.
        
        Args:
            stx_delegated: User's STX delegation
            total_stx_pool: Total STX in pool
            total_btc_reward: Total BTC rewards for cycle
            phi_score: Spectral integration score
            has_nft: Whether user holds rarity NFT
            
        Returns:
            YieldResult with detailed breakdown
        """
        # Step 1: Calculate base reward
        base_reward = self.calculate_base_reward(
            stx_delegated, total_stx_pool, total_btc_reward
        )
        
        # Step 2: Calculate treasury split
        treasury_rate = self.calculate_treasury_split(phi_score)
        treasury_share = base_reward * treasury_rate
        
        # Step 3: Calculate user payout
        user_payout = base_reward - treasury_share
        
        # Step 4: Apply NFT multiplier
        nft_multiplier = self.calculate_nft_multiplier(phi_score, has_nft)
        effective_payout = user_payout * nft_multiplier
        
        return YieldResult(
            total_reward=base_reward,
            treasury_share=treasury_share,
            user_payout=user_payout,
            nft_multiplier=nft_multiplier,
            effective_payout=effective_payout
        )
    
    def calculate_batch_yields(
        self,
        delegations: Dict[str, float],
        total_btc_reward: float,
        phi_scores: Dict[str, float],
        nft_holders: Optional[set] = None
    ) -> Dict[str, YieldResult]:
        """
        Calculate yields for multiple users.
        
        Args:
            delegations: Dict mapping user_id to STX amount
            total_btc_reward: Total BTC rewards for cycle
            phi_scores: Dict mapping user_id to Φ score
            nft_holders: Set of user_ids who hold NFTs
            
        Returns:
            Dict mapping user_id to YieldResult
        """
        if nft_holders is None:
            nft_holders = set()
        
        total_stx = sum(delegations.values())
        results = {}
        
        for user_id, stx_amount in delegations.items():
            phi = phi_scores.get(user_id, 500.0)  # Default Φ
            has_nft = user_id in nft_holders
            
            result = self.calculate_yield(
                stx_amount, total_stx, total_btc_reward, phi, has_nft
            )
            results[user_id] = result
        
        return results
    
    def get_treasury_total(self, batch_results: Dict[str, YieldResult]) -> float:
        """
        Calculate total treasury revenue from batch results.
        
        Args:
            batch_results: Results from calculate_batch_yields
            
        Returns:
            Total BTC going to treasury
        """
        return sum(result.treasury_share for result in batch_results.values())


# Example usage and testing
if __name__ == "__main__":
    calculator = YieldCalculator(pool_efficiency=0.95)
    
    # Example: Single user calculation
    result = calculator.calculate_yield(
        stx_delegated=10000,
        total_stx_pool=50000,
        total_btc_reward=1.0,
        phi_score=650,
        has_nft=True
    )
    
    print("=" * 60)
    print("YIELD CALCULATION EXAMPLE")
    print("=" * 60)
    print(f"Total Reward:      {result.total_reward:.8f} BTC")
    print(f"Treasury Share:    {result.treasury_share:.8f} BTC")
    print(f"User Payout:       {result.user_payout:.8f} BTC")
    print(f"NFT Multiplier:    {result.nft_multiplier:.4f}x")
    print(f"Effective Payout:  {result.effective_payout:.8f} BTC")
    print("=" * 60)
    
    # Example: Batch calculation
    delegations = {
        "user1": 10000,
        "user2": 15000,
        "user3": 25000
    }
    phi_scores = {
        "user1": 650,
        "user2": 720,
        "user3": 580
    }
    nft_holders = {"user1", "user3"}
    
    batch_results = calculator.calculate_batch_yields(
        delegations, 1.0, phi_scores, nft_holders
    )
    
    print("\nBATCH CALCULATION")
    print("=" * 60)
    for user_id, result in batch_results.items():
        print(f"{user_id}:")
        print(f"  STX: {delegations[user_id]}")
        print(f"  Φ: {phi_scores.get(user_id, 500)}")
        print(f"  NFT: {'Yes' if user_id in nft_holders else 'No'}")
        print(f"  Payout: {result.effective_payout:.8f} BTC")
        print()
    
    treasury_total = calculator.get_treasury_total(batch_results)
    print(f"Total Treasury Revenue: {treasury_total:.8f} BTC")
    print("=" * 60)
