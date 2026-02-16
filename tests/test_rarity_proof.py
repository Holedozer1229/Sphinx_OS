"""
Tests for Rarity Proof System
"""
import pytest
from sphinx_os.nft.rarity_proof import RarityProofSystem
from sphinx_os.treasury.self_funding import SelfFundingTreasury


def test_rarity_proof_initialization():
    """Test rarity proof system is initialized correctly"""
    rarity_system = RarityProofSystem()
    assert rarity_system.PROOF_FEE == 0.05
    assert rarity_system.treasury is not None
    assert isinstance(rarity_system.treasury, SelfFundingTreasury)


def test_rarity_proof_with_custom_treasury():
    """Test rarity proof system can use a custom treasury instance"""
    treasury = SelfFundingTreasury()
    rarity_system = RarityProofSystem(treasury=treasury)
    assert rarity_system.treasury is treasury


def test_generate_rarity_proof_success():
    """Test successful rarity proof generation"""
    treasury = SelfFundingTreasury()
    rarity_system = RarityProofSystem(treasury=treasury)
    
    nft_id = 12345
    user_address = "0x123456789abcdef"
    
    # Generate proof with sufficient balance
    result = rarity_system.generate_rarity_proof(nft_id, user_address, balance=1.0)
    
    # Check result structure
    assert result["nft_id"] == nft_id
    assert "rarity_score" in result
    assert "proof" in result
    assert result["fee_paid"] == 0.05
    assert "fee_distribution" in result
    
    # Check rarity score is in valid range
    assert 0.0 <= result["rarity_score"] <= 1.0
    
    # Check treasury received fee
    assert treasury.treasury_balance == pytest.approx(0.04)  # 80% of 0.05


def test_generate_rarity_proof_insufficient_balance():
    """Test rarity proof generation fails with insufficient balance"""
    rarity_system = RarityProofSystem()
    
    nft_id = 12345
    user_address = "0x123456789abcdef"
    
    # Try to generate proof with insufficient balance
    with pytest.raises(ValueError, match="Insufficient balance"):
        rarity_system.generate_rarity_proof(nft_id, user_address, balance=0.02)


def test_compute_rarity_score():
    """Test rarity score computation"""
    rarity_system = RarityProofSystem()
    
    # Test with different NFT IDs
    score1 = rarity_system._compute_rarity_score(12345)
    score2 = rarity_system._compute_rarity_score(67890)
    
    # Scores should be different for different IDs
    assert score1 != score2
    
    # Scores should be in valid range
    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0


def test_compute_rarity_score_deterministic():
    """Test rarity score is deterministic for same NFT ID"""
    rarity_system = RarityProofSystem()
    
    nft_id = 12345
    score1 = rarity_system._compute_rarity_score(nft_id)
    score2 = rarity_system._compute_rarity_score(nft_id)
    
    # Same ID should produce same score
    assert score1 == score2


def test_generate_zk_proof():
    """Test ZK proof generation"""
    rarity_system = RarityProofSystem()
    
    nft_id = 12345
    rarity_score = 0.75
    
    proof = rarity_system._generate_zk_proof(nft_id, rarity_score)
    
    # Check proof structure
    assert isinstance(proof, dict)
    assert "proof_hash" in proof or "verified" in proof


def test_multiple_proof_generations():
    """Test generating multiple proofs"""
    treasury = SelfFundingTreasury()
    rarity_system = RarityProofSystem(treasury=treasury)
    
    user_address = "0x123456789abcdef"
    
    # Generate 5 proofs
    for nft_id in range(1000, 1005):
        result = rarity_system.generate_rarity_proof(nft_id, user_address, balance=10.0)
        assert result["nft_id"] == nft_id
    
    # Check treasury accumulated fees: 5 * 0.05 * 0.8 = 0.2
    assert treasury.treasury_balance == pytest.approx(0.2)


def test_fee_distribution():
    """Test fee distribution is correct"""
    treasury = SelfFundingTreasury()
    rarity_system = RarityProofSystem(treasury=treasury)
    
    nft_id = 12345
    user_address = "0x123456789abcdef"
    
    result = rarity_system.generate_rarity_proof(nft_id, user_address, balance=1.0)
    
    distribution = result["fee_distribution"]
    assert distribution["treasury"] == pytest.approx(0.04)  # 80% of 0.05
    assert distribution["operator"] == pytest.approx(0.0075)  # 15% of 0.05
    assert distribution["miners"] == pytest.approx(0.0025)  # 5% of 0.05


def test_proof_contains_nft_info():
    """Test generated proof contains NFT information"""
    treasury = SelfFundingTreasury()
    rarity_system = RarityProofSystem(treasury=treasury)
    
    nft_id = 99999
    user_address = "0x123456789abcdef"
    
    result = rarity_system.generate_rarity_proof(nft_id, user_address, balance=1.0)
    
    proof = result["proof"]
    # Proof should contain NFT ID in some form
    assert "nft_id" in proof or str(nft_id) in str(proof)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
