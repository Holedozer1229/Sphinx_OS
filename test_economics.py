#!/usr/bin/env python3
"""
Test script for SphinxOS economic modules
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sphinx_os.economics.yield_calculator import YieldCalculator
from sphinx_os.economics.simulator import EconomicSimulator

def test_yield_calculator():
    """Test yield calculation functions"""
    print("=" * 70)
    print("TESTING YIELD CALCULATOR")
    print("=" * 70)
    
    calculator = YieldCalculator(pool_efficiency=0.95)
    
    # Test 1: Single user
    result = calculator.calculate_yield(
        stx_delegated=10000,
        total_stx_pool=50000,
        total_btc_reward=1.0,
        phi_score=650,
        has_nft=True
    )
    
    print("\nTest 1: Single User with NFT")
    print(f"  Total Reward:     {result.total_reward:.8f} BTC")
    print(f"  Treasury Share:   {result.treasury_share:.8f} BTC")
    print(f"  User Payout:      {result.user_payout:.8f} BTC")
    print(f"  NFT Multiplier:   {result.nft_multiplier:.4f}x")
    print(f"  Effective Payout: {result.effective_payout:.8f} BTC")
    
    assert result.total_reward > 0, "Total reward should be positive"
    assert result.treasury_share < result.total_reward, "Treasury should be less than total"
    assert result.nft_multiplier > 1.0, "NFT multiplier should be > 1.0"
    
    print("\n✅ Yield calculator tests passed!")
    return True

def test_economic_simulator():
    """Test economic simulator"""
    print("\n" + "=" * 70)
    print("TESTING ECONOMIC SIMULATOR")
    print("=" * 70)
    
    from sphinx_os.economics.simulator import SimulationScenario
    
    simulator = EconomicSimulator()
    
    # Test scenario
    scenario = SimulationScenario(
        name="Test Scenario",
        num_users=1000,
        avg_stx_per_user=5000,
        phi_mean=600,
        phi_stddev=100,
        btc_price_usd=50000
    )
    
    result = simulator.simulate_scenario(scenario, verbose=True)
    
    assert result.annual_treasury_usd > 0, "Treasury revenue should be positive"
    assert result.annual_user_yield_usd > 0, "User yield should be positive"
    assert result.treasury_percentage < 30, "Treasury should be < 30%"
    
    print("\n✅ Economic simulator tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_yield_calculator()
        test_economic_simulator()
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
