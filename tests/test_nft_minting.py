"""
Tests for NFT Minting System
"""
import pytest
from sphinx_os.nft.minting import SphinxNFTMinter
from sphinx_os.treasury.self_funding import SelfFundingTreasury


def test_nft_minter_initialization():
    """Test NFT minter is initialized correctly"""
    minter = SphinxNFTMinter()
    assert minter.MINT_FEE == 0.1
    assert minter.treasury is not None
    assert isinstance(minter.treasury, SelfFundingTreasury)


def test_nft_minter_with_custom_treasury():
    """Test NFT minter can use a custom treasury instance"""
    treasury = SelfFundingTreasury()
    minter = SphinxNFTMinter(treasury=treasury)
    assert minter.treasury is treasury


def test_mint_nft_success():
    """Test successful NFT minting"""
    treasury = SelfFundingTreasury()
    minter = SphinxNFTMinter(treasury=treasury)
    
    user_address = "0x123456789abcdef"
    metadata = {
        "name": "Test NFT",
        "rarity": "rare",
        "attributes": {"power": 100}
    }
    
    # Mint NFT with sufficient balance
    result = minter.mint_nft(user_address, metadata, balance=1.0)
    
    # Check result structure
    assert "nft_id" in result
    assert result["fee_paid"] == 0.1
    assert "fee_distribution" in result
    assert result["metadata"] == metadata
    
    # Check treasury received fee
    assert treasury.treasury_balance == pytest.approx(0.07)  # 70% of 0.1


def test_mint_nft_insufficient_balance():
    """Test NFT minting fails with insufficient balance"""
    minter = SphinxNFTMinter()
    
    user_address = "0x123456789abcdef"
    metadata = {"name": "Test NFT"}
    
    # Try to mint with insufficient balance
    with pytest.raises(ValueError, match="Insufficient balance"):
        minter.mint_nft(user_address, metadata, balance=0.05)


def test_mint_nft_no_balance_provided():
    """Test NFT minting requires balance when wallet not available"""
    minter = SphinxNFTMinter()
    
    user_address = "0x123456789abcdef"
    metadata = {"name": "Test NFT"}
    
    # Try to mint without providing balance (wallet import will fail)
    with pytest.raises(ValueError):
        minter.mint_nft(user_address, metadata)


def test_multiple_nft_mints():
    """Test minting multiple NFTs"""
    treasury = SelfFundingTreasury()
    minter = SphinxNFTMinter(treasury=treasury)
    
    user_address = "0x123456789abcdef"
    
    # Mint 5 NFTs
    nft_ids = []
    for i in range(5):
        metadata = {"name": f"NFT {i}"}
        result = minter.mint_nft(user_address, metadata, balance=10.0)
        nft_ids.append(result["nft_id"])
    
    # Check all NFTs have unique IDs
    assert len(nft_ids) == len(set(nft_ids))
    
    # Check treasury accumulated fees: 5 * 0.1 * 0.7 = 0.35
    assert treasury.treasury_balance == pytest.approx(0.35)


def test_nft_id_increments():
    """Test NFT IDs increment correctly"""
    minter = SphinxNFTMinter()
    
    user_address = "0x123456789abcdef"
    metadata = {"name": "Test NFT"}
    
    result1 = minter.mint_nft(user_address, metadata, balance=1.0)
    result2 = minter.mint_nft(user_address, metadata, balance=1.0)
    
    # IDs should increment
    assert result2["nft_id"] == result1["nft_id"] + 1


def test_fee_distribution():
    """Test fee distribution is correct"""
    treasury = SelfFundingTreasury()
    minter = SphinxNFTMinter(treasury=treasury)
    
    user_address = "0x123456789abcdef"
    metadata = {"name": "Test NFT"}
    
    result = minter.mint_nft(user_address, metadata, balance=1.0)
    
    distribution = result["fee_distribution"]
    assert distribution["treasury"] == pytest.approx(0.07)
    assert distribution["operator"] == pytest.approx(0.02)
    assert distribution["rewards"] == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
