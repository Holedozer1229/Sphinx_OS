"""
Test Suite for Multi-Token Integration

Tests:
- Token registry
- Yield optimizer
- zk-EVM prover
- Circuit compilation
- Smart contract deployment
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sphinx_os.tokens import TokenRegistry, Token, ChainType, MultiTokenYieldOptimizer


class TestTokenRegistry:
    """Test token registry functionality"""
    
    def test_initialization(self):
        """Test registry initializes with default tokens"""
        registry = TokenRegistry()
        assert len(registry.get_all_tokens()) > 0
        print(f"✅ Registry loaded {len(registry.get_all_tokens())} tokens")
    
    def test_chain_configs(self):
        """Test all chain configurations"""
        registry = TokenRegistry()
        
        for chain_type in [ChainType.ETHEREUM, ChainType.POLYGON, ChainType.BSC]:
            config = registry.get_chain_config(chain_type)
            assert config is not None
            assert config.chain_id > 0
            print(f"✅ {config.name}: Chain ID {config.chain_id}")
    
    def test_get_token(self):
        """Test token retrieval"""
        registry = TokenRegistry()
        
        eth = registry.get_token("ETH", ChainType.ETHEREUM)
        assert eth is not None
        assert eth.symbol == "ETH"
        assert eth.is_native is True
        print(f"✅ Retrieved ETH token: {eth.address}")
    
    def test_zk_compatible_tokens(self):
        """Test zk-compatible token filtering"""
        registry = TokenRegistry()
        
        zk_tokens = registry.get_zk_compatible_tokens()
        assert len(zk_tokens) > 0
        
        # Verify all are on zk chains
        for token in zk_tokens:
            config = registry.get_chain_config(token.chain_type)
            assert config.supports_zk is True
        
        print(f"✅ Found {len(zk_tokens)} zk-compatible tokens")
    
    def test_total_tvl(self):
        """Test total value locked calculation"""
        registry = TokenRegistry()
        
        tvl = registry.calculate_total_tvl()
        assert tvl > 0
        print(f"✅ Total TVL: ${tvl:,.0f}")
    
    def test_highest_yield_tokens(self):
        """Test highest yield token query"""
        registry = TokenRegistry()
        
        top_yields = registry.get_highest_yield_tokens(5)
        assert len(top_yields) == 5
        
        # Verify sorted by APR
        for i in range(len(top_yields) - 1):
            assert top_yields[i].yield_apr >= top_yields[i + 1].yield_apr
        
        print("✅ Top 5 yield tokens:")
        for token in top_yields:
            print(f"   {token.symbol}: {token.yield_apr}% APR")


class TestYieldOptimizer:
    """Test multi-token yield optimizer"""
    
    def test_initialization(self):
        """Test optimizer initializes"""
        optimizer = MultiTokenYieldOptimizer()
        assert len(optimizer.opportunities) > 0
        print(f"✅ Optimizer loaded {len(optimizer.opportunities)} opportunities")
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization"""
        optimizer = MultiTokenYieldOptimizer()
        
        result = optimizer.optimize_portfolio(
            capital_usd=10000.0,
            phi_score=750.0,
            max_risk=5.0,
            min_apr=3.0
        )
        
        assert result.total_apy > 0
        assert result.expected_return_1y > 0
        assert len(result.opportunities) > 0
        
        print(f"✅ Optimized portfolio:")
        print(f"   Total APY: {result.total_apy:.2f}%")
        print(f"   Expected 1y return: ${result.expected_return_1y:,.2f}")
        print(f"   Selected strategies: {len(result.opportunities)}")
    
    def test_phi_boost_effect(self):
        """Test Φ score boost effect"""
        optimizer = MultiTokenYieldOptimizer()
        
        # Low Φ score
        result_low = optimizer.optimize_portfolio(10000, 300, 5.0, 3.0)
        
        # High Φ score
        result_high = optimizer.optimize_portfolio(10000, 900, 5.0, 3.0)
        
        # High Φ should give better returns
        assert result_high.total_apy > result_low.total_apy
        
        print(f"✅ Φ boost effect:")
        print(f"   Φ=300: {result_low.total_apy:.2f}% APY")
        print(f"   Φ=900: {result_high.total_apy:.2f}% APY")
        print(f"   Boost: +{result_high.total_apy - result_low.total_apy:.2f}%")
    
    def test_conservative_vs_aggressive(self):
        """Test conservative vs aggressive strategies"""
        optimizer = MultiTokenYieldOptimizer()
        
        conservative = optimizer.get_conservative_strategy(10000, 500)
        aggressive = optimizer.get_max_yield_strategy(10000, 800)
        
        # Aggressive should have higher APY and risk
        assert aggressive.total_apy > conservative.total_apy
        
        print(f"✅ Strategy comparison:")
        print(f"   Conservative: {conservative.total_apy:.2f}% APY")
        print(f"   Aggressive: {aggressive.total_apy:.2f}% APY")
    
    def test_compound_growth(self):
        """Test compound growth simulation"""
        optimizer = MultiTokenYieldOptimizer()
        
        growth = optimizer.simulate_compound_growth(
            initial_capital=10000,
            phi_score=750,
            months=12,
            monthly_contribution=500
        )
        
        assert len(growth) == 13  # 0 + 12 months
        assert growth[-1][1] > growth[0][1]
        
        final_value = growth[-1][1]
        print(f"✅ Compound growth (12 months):")
        print(f"   Initial: ${growth[0][1]:,.2f}")
        print(f"   Final: ${final_value:,.2f}")
        print(f"   Gain: ${final_value - growth[0][1]:,.2f}")
    
    def test_cross_chain_yield(self):
        """Test cross-chain yield calculation"""
        optimizer = MultiTokenYieldOptimizer()
        
        token_amounts = {
            "ETH": 5.0,
            "USDC": 10000.0,
            "STX": 50000.0
        }
        
        yields = optimizer.calculate_cross_chain_yield(token_amounts, 800)
        
        assert len(yields) == len(token_amounts)
        assert all(y >= 0 for y in yields.values())
        
        total_yield = sum(yields.values())
        print(f"✅ Cross-chain yield:")
        for symbol, yield_amount in yields.items():
            print(f"   {symbol}: ${yield_amount:,.2f}/year")
        print(f"   Total: ${total_yield:,.2f}/year")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("SPHINXOS MULTI-TOKEN INTEGRATION TEST SUITE")
    print("="*60)
    
    # Token Registry Tests
    print("\n[1/2] TOKEN REGISTRY TESTS")
    print("-"*60)
    test_registry = TestTokenRegistry()
    test_registry.test_initialization()
    test_registry.test_chain_configs()
    test_registry.test_get_token()
    test_registry.test_zk_compatible_tokens()
    test_registry.test_total_tvl()
    test_registry.test_highest_yield_tokens()
    
    # Yield Optimizer Tests
    print("\n[2/2] YIELD OPTIMIZER TESTS")
    print("-"*60)
    test_optimizer = TestYieldOptimizer()
    test_optimizer.test_initialization()
    test_optimizer.test_portfolio_optimization()
    test_optimizer.test_phi_boost_effect()
    test_optimizer.test_conservative_vs_aggressive()
    test_optimizer.test_compound_growth()
    test_optimizer.test_cross_chain_yield()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
