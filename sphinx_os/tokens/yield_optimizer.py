"""
Multi-Token Yield Optimizer

Integrates with existing STX→BTC yield calculator to support all tokens.
Optimizes yield strategies across multiple chains and protocols.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .token_registry import Token, TokenRegistry, ChainType


class YieldStrategy(Enum):
    """Yield generation strategies"""
    STAKING = "staking"
    LENDING = "lending"
    LIQUIDITY_MINING = "liquidity_mining"
    YIELD_FARMING = "yield_farming"
    POX_DELEGATION = "pox_delegation"  # Stacks-specific
    VAULT = "vault"  # Automated strategies


@dataclass
class YieldOpportunity:
    """A yield opportunity for a token"""
    token: Token
    strategy: YieldStrategy
    apr: float
    tvl: float
    risk_score: float  # 0-10, lower is safer
    protocol: str
    chain_type: ChainType
    min_deposit: float
    phi_boost: float  # Spectral integration score boost
    

@dataclass
class OptimizedYieldResult:
    """Result of yield optimization"""
    total_yield_usd: float
    opportunities: List[YieldOpportunity]
    total_apy: float  # Compound APY
    expected_return_30d: float
    expected_return_1y: float
    risk_adjusted_score: float  # Sharpe ratio approximation


class MultiTokenYieldOptimizer:
    """
    Optimize yield across all supported tokens and chains.
    
    Integrates:
    - SphinxSkynet Φ score for enhanced yields
    - Cross-chain opportunities
    - Risk-adjusted returns
    - Automated portfolio rebalancing
    """
    
    def __init__(self, token_registry: Optional[TokenRegistry] = None):
        """Initialize optimizer"""
        self.registry = token_registry or TokenRegistry()
        self.opportunities: List[YieldOpportunity] = []
        self._initialize_opportunities()
    
    def _initialize_opportunities(self):
        """Initialize yield opportunities for all tokens"""
        
        # Staking opportunities
        self.opportunities.extend([
            YieldOpportunity(
                token=self.registry.get_token("ETH", ChainType.ETHEREUM),
                strategy=YieldStrategy.STAKING,
                apr=3.5,
                tvl=50_000_000_000,
                risk_score=2.0,
                protocol="Lido",
                chain_type=ChainType.ETHEREUM,
                min_deposit=0.01,
                phi_boost=1.15
            ),
            YieldOpportunity(
                token=self.registry.get_token("MATIC", ChainType.POLYGON),
                strategy=YieldStrategy.STAKING,
                apr=5.2,
                tvl=8_000_000_000,
                risk_score=3.0,
                protocol="Polygon",
                chain_type=ChainType.POLYGON,
                min_deposit=1.0,
                phi_boost=1.12
            ),
            YieldOpportunity(
                token=self.registry.get_token("STX", ChainType.STACKS),
                strategy=YieldStrategy.POX_DELEGATION,
                apr=12.3,
                tvl=2_500_000_000,
                risk_score=4.5,
                protocol="SphinxOS PoX",
                chain_type=ChainType.STACKS,
                min_deposit=100.0,
                phi_boost=1.35  # Enhanced by spectral integration
            ),
        ])
        
        # Lending opportunities
        self.opportunities.extend([
            YieldOpportunity(
                token=self.registry.get_token("USDC", ChainType.ETHEREUM),
                strategy=YieldStrategy.LENDING,
                apr=4.5,
                tvl=25_000_000_000,
                risk_score=2.5,
                protocol="Aave V3",
                chain_type=ChainType.ETHEREUM,
                min_deposit=10.0,
                phi_boost=1.08
            ),
            YieldOpportunity(
                token=self.registry.get_token("DAI", ChainType.ETHEREUM),
                strategy=YieldStrategy.LENDING,
                apr=5.8,
                tvl=5_000_000_000,
                risk_score=2.8,
                protocol="Compound V3",
                chain_type=ChainType.ETHEREUM,
                min_deposit=10.0,
                phi_boost=1.09
            ),
        ])
        
        # Liquidity mining
        self.opportunities.extend([
            YieldOpportunity(
                token=self.registry.get_token("UNI", ChainType.ETHEREUM),
                strategy=YieldStrategy.LIQUIDITY_MINING,
                apr=8.5,
                tvl=4_500_000_000,
                risk_score=5.5,
                protocol="Uniswap V3",
                chain_type=ChainType.ETHEREUM,
                min_deposit=50.0,
                phi_boost=1.18
            ),
            YieldOpportunity(
                token=self.registry.get_token("CAKE", ChainType.BSC),
                strategy=YieldStrategy.YIELD_FARMING,
                apr=35.5,
                tvl=600_000_000,
                risk_score=7.0,
                protocol="PancakeSwap",
                chain_type=ChainType.BSC,
                min_deposit=10.0,
                phi_boost=1.25
            ),
            YieldOpportunity(
                token=self.registry.get_token("CRV", ChainType.ETHEREUM),
                strategy=YieldStrategy.YIELD_FARMING,
                apr=8.2,
                tvl=1_200_000_000,
                risk_score=4.5,
                protocol="Curve Finance",
                chain_type=ChainType.ETHEREUM,
                min_deposit=100.0,
                phi_boost=1.14
            ),
            YieldOpportunity(
                token=self.registry.get_token("GMX", ChainType.ARBITRUM),
                strategy=YieldStrategy.STAKING,
                apr=28.5,
                tvl=400_000_000,
                risk_score=6.5,
                protocol="GMX",
                chain_type=ChainType.ARBITRUM,
                min_deposit=0.1,
                phi_boost=1.22
            ),
        ])
    
    def optimize_portfolio(
        self,
        capital_usd: float,
        phi_score: float = 500.0,
        max_risk: float = 5.0,
        min_apr: float = 3.0
    ) -> OptimizedYieldResult:
        """
        Optimize portfolio allocation across available opportunities.
        
        Args:
            capital_usd: Total capital to allocate
            phi_score: Spectral integration score (200-1000)
            max_risk: Maximum acceptable risk score
            min_apr: Minimum APR to consider
        
        Returns:
            OptimizedYieldResult with optimal allocation
        """
        
        # Filter opportunities
        eligible = [
            opp for opp in self.opportunities
            if opp.risk_score <= max_risk and opp.apr >= min_apr
        ]
        
        # Apply Φ boost
        phi_multiplier = 1.0 + (phi_score - 500) / 2000.0  # 0.85 to 1.25
        
        # Score opportunities (risk-adjusted return with Φ boost)
        scored = []
        for opp in eligible:
            boosted_apr = opp.apr * opp.phi_boost * phi_multiplier
            risk_adjusted = boosted_apr / (1 + opp.risk_score)
            scored.append((opp, boosted_apr, risk_adjusted))
        
        # Sort by risk-adjusted return
        scored.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate capital (simplified - equal weight top 5)
        top_opportunities = scored[:5]
        allocation_per_opp = capital_usd / len(top_opportunities)
        
        selected = []
        total_yield = 0.0
        weighted_apr_sum = 0.0
        
        for opp, boosted_apr, _ in top_opportunities:
            selected.append(opp)
            opp_yield = allocation_per_opp * (boosted_apr / 100.0)
            total_yield += opp_yield
            weighted_apr_sum += boosted_apr
        
        avg_apy = weighted_apr_sum / len(selected) if selected else 0.0
        
        # Calculate compound APY
        compound_apy = (1 + avg_apy / 100.0 / 12) ** 12 - 1
        compound_apy *= 100
        
        # Expected returns
        expected_30d = capital_usd * (avg_apy / 100.0) / 12
        expected_1y = capital_usd * (compound_apy / 100.0)
        
        # Risk-adjusted score (Sharpe ratio approximation)
        avg_risk = sum(opp.risk_score for opp in selected) / len(selected) if selected else 0
        risk_adjusted_score = avg_apy / (1 + avg_risk) if avg_risk > 0 else avg_apy
        
        return OptimizedYieldResult(
            total_yield_usd=total_yield,
            opportunities=selected,
            total_apy=compound_apy,
            expected_return_30d=expected_30d,
            expected_return_1y=expected_1y,
            risk_adjusted_score=risk_adjusted_score
        )
    
    def calculate_cross_chain_yield(
        self,
        token_amounts: Dict[str, float],
        phi_score: float = 500.0
    ) -> Dict[str, float]:
        """
        Calculate yield for specific token amounts across chains.
        
        Args:
            token_amounts: Dict of {symbol: amount}
            phi_score: Spectral integration score
        
        Returns:
            Dict of {symbol: annual_yield_usd}
        """
        phi_multiplier = 1.0 + (phi_score - 500) / 2000.0
        yields = {}
        
        for symbol, amount in token_amounts.items():
            # Find best opportunity for this token
            token_opps = [
                opp for opp in self.opportunities
                if opp.token and opp.token.symbol == symbol
            ]
            
            if not token_opps:
                yields[symbol] = 0.0
                continue
            
            # Sort by boosted APR
            best_opp = max(
                token_opps,
                key=lambda o: o.apr * o.phi_boost * phi_multiplier
            )
            
            # Calculate yield (simplified - assumes 1:1 USD for stablecoins)
            # In production, would fetch real-time prices
            boosted_apr = best_opp.apr * best_opp.phi_boost * phi_multiplier
            annual_yield = amount * (boosted_apr / 100.0)
            yields[symbol] = annual_yield
        
        return yields
    
    def get_max_yield_strategy(
        self,
        capital_usd: float,
        phi_score: float = 800.0
    ) -> OptimizedYieldResult:
        """
        Get maximum yield strategy (higher risk acceptable).
        
        Optimized for maximum monetization.
        """
        return self.optimize_portfolio(
            capital_usd=capital_usd,
            phi_score=phi_score,
            max_risk=8.0,  # Accept higher risk
            min_apr=5.0     # Higher minimum APR
        )
    
    def get_conservative_strategy(
        self,
        capital_usd: float,
        phi_score: float = 500.0
    ) -> OptimizedYieldResult:
        """
        Get conservative yield strategy (lower risk).
        """
        return self.optimize_portfolio(
            capital_usd=capital_usd,
            phi_score=phi_score,
            max_risk=3.0,   # Lower risk
            min_apr=2.0     # Lower minimum APR
        )
    
    def simulate_compound_growth(
        self,
        initial_capital: float,
        phi_score: float,
        months: int = 12,
        monthly_contribution: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Simulate compound growth over time.
        
        Returns:
            List of (month, total_value) tuples
        """
        result = self.optimize_portfolio(initial_capital, phi_score)
        monthly_rate = result.total_apy / 100.0 / 12
        
        balance = initial_capital
        history = [(0, balance)]
        
        for month in range(1, months + 1):
            balance = balance * (1 + monthly_rate) + monthly_contribution
            history.append((month, balance))
        
        return history
    
    def summary(self, capital_usd: float = 10000.0, phi_score: float = 750.0) -> str:
        """Get summary of optimization results"""
        
        result = self.optimize_portfolio(capital_usd, phi_score)
        
        output = f"""
SphinxSkynet Multi-Token Yield Optimizer
{'='*60}
Capital: ${capital_usd:,.2f}
Φ Score: {phi_score:.2f}

Optimized Portfolio:
{'─'*60}
Total APY: {result.total_apy:.2f}%
Expected 30d Return: ${result.expected_return_30d:,.2f}
Expected 1y Return: ${result.expected_return_1y:,.2f}
Risk-Adjusted Score: {result.risk_adjusted_score:.2f}

Selected Opportunities:
"""
        
        for opp in result.opportunities:
            output += f"\n  • {opp.token.symbol} on {opp.chain_type.value}"
            output += f"\n    Protocol: {opp.protocol}"
            output += f"\n    Strategy: {opp.strategy.value}"
            output += f"\n    APR: {opp.apr:.2f}% (Φ-boosted: {opp.apr * opp.phi_boost:.2f}%)"
            output += f"\n    Risk: {opp.risk_score:.1f}/10"
            output += f"\n"
        
        # Compound growth simulation
        growth = self.simulate_compound_growth(capital_usd, phi_score, 12)
        final_value = growth[-1][1]
        
        output += f"\n{'─'*60}"
        output += f"\nProjected Value After 1 Year: ${final_value:,.2f}"
        output += f"\nTotal Gain: ${final_value - capital_usd:,.2f} ({(final_value/capital_usd - 1)*100:.2f}%)"
        
        return output


if __name__ == "__main__":
    # Demo
    optimizer = MultiTokenYieldOptimizer()
    
    # Conservative strategy
    print("CONSERVATIVE STRATEGY ($10,000)")
    print("="*60)
    conservative = optimizer.get_conservative_strategy(10000, 500)
    print(f"APY: {conservative.total_apy:.2f}%")
    print(f"Expected 1y: ${conservative.expected_return_1y:,.2f}")
    
    # Maximum yield strategy
    print("\n\nMAXIMUM YIELD STRATEGY ($10,000, Φ=800)")
    print("="*60)
    max_yield = optimizer.get_max_yield_strategy(10000, 800)
    print(f"APY: {max_yield.total_apy:.2f}%")
    print(f"Expected 1y: ${max_yield.expected_return_1y:,.2f}")
    
    # Full summary
    print("\n" + optimizer.summary(50000, 850))
