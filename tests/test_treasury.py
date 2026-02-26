"""
Tests for Self-Funding Treasury System
"""
import pytest
from sphinx_os.treasury.self_funding import SelfFundingTreasury


def test_treasury_initialization():
    """Test treasury is initialized correctly"""
    treasury = SelfFundingTreasury()
    assert treasury.treasury_balance == 0
    assert "polygon" in treasury.deployment_targets
    assert treasury.deployment_targets["polygon"]["threshold"] == 50
    assert treasury.deployment_targets["polygon"]["deployed"] is False


def test_collect_nft_mint_fee():
    """Test NFT mint fee collection"""
    treasury = SelfFundingTreasury()
    
    result = treasury.collect_nft_mint_fee(0.1)
    
    # Check fee distribution (use pytest.approx for floating point comparison)
    assert result["treasury"] == pytest.approx(0.07)  # 70% of 0.1
    assert result["operator"] == pytest.approx(0.02)  # 20% of 0.1
    assert result["rewards"] == pytest.approx(0.01)   # 10% of 0.1
    
    # Check treasury balance updated
    assert treasury.treasury_balance == pytest.approx(0.07)


def test_collect_rarity_proof_fee():
    """Test rarity proof fee collection"""
    treasury = SelfFundingTreasury()
    
    result = treasury.collect_rarity_proof_fee(0.05)
    
    # Check fee distribution (use pytest.approx for floating point comparison)
    assert result["treasury"] == pytest.approx(0.04)  # 80% of 0.05
    assert result["operator"] == pytest.approx(0.0075)  # 15% of 0.05
    assert result["miners"] == pytest.approx(0.0025)   # 5% of 0.05
    
    # Check treasury balance updated
    assert treasury.treasury_balance == pytest.approx(0.04)


def test_treasury_accumulation():
    """Test treasury accumulates fees over multiple transactions"""
    treasury = SelfFundingTreasury()
    
    # Collect multiple fees
    for _ in range(10):
        treasury.collect_nft_mint_fee(0.1)
    
    # 10 * 0.1 * 0.7 = 0.7
    assert treasury.treasury_balance == pytest.approx(0.7)


def test_get_treasury_stats():
    """Test treasury stats retrieval"""
    treasury = SelfFundingTreasury()
    treasury.treasury_balance = 25.0
    
    stats = treasury.get_treasury_stats()
    
    assert stats["balance_skynt"] == 25.0
    assert stats["balance_usd"] == 25.0
    assert "deployments" in stats
    assert "polygon" in stats["deployments"]
    
    # Check polygon progress (25/50 = 50%)
    polygon_stats = stats["deployments"]["polygon"]
    assert polygon_stats["threshold"] == 50
    assert polygon_stats["progress"] == 50.0
    assert polygon_stats["ready"] is False
    assert polygon_stats["deployed"] is False


def test_deployment_readiness():
    """Test deployment readiness detection"""
    treasury = SelfFundingTreasury()
    
    # Add enough for avalanche (threshold: 30)
    treasury.treasury_balance = 30.0
    
    stats = treasury.get_treasury_stats()
    
    # Avalanche should be ready
    assert stats["deployments"]["avalanche"]["ready"] is True
    assert stats["deployments"]["avalanche"]["progress"] == 100.0
    
    # Others should not be ready
    assert stats["deployments"]["polygon"]["ready"] is False
    assert stats["deployments"]["ethereum"]["ready"] is False


def test_swap_to_native():
    """Test SKYNT to native token swap"""
    treasury = SelfFundingTreasury()
    
    # Test mock conversion
    native_amount = treasury.swap_to_native("polygon", 50.0)
    
    # Should return some amount (mock implementation)
    assert native_amount > 0


def test_deployment_targets_structure():
    """Test deployment targets have correct structure"""
    treasury = SelfFundingTreasury()
    
    required_chains = ["polygon", "avalanche", "bnb", "ethereum"]
    
    for chain in required_chains:
        assert chain in treasury.deployment_targets
        assert "threshold" in treasury.deployment_targets[chain]
        assert "deployed" in treasury.deployment_targets[chain]
        assert isinstance(treasury.deployment_targets[chain]["threshold"], (int, float))
        assert isinstance(treasury.deployment_targets[chain]["deployed"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
