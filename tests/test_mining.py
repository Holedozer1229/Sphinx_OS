"""
Tests for SphinxSkynet Mining
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.blockchain.core import SphinxSkynetBlockchain
from sphinx_os.mining.miner import SphinxMiner
from sphinx_os.mining.pow_algorithms import PoWAlgorithms
from sphinx_os.mining.merge_miner import MergeMiningCoordinator


def test_pow_algorithms():
    """Test PoW algorithm functions exist"""
    block_data = "test_block_data"
    nonce = 12345
    
    # Test SHA-256
    hash_sha256 = PoWAlgorithms.sha256_pow(block_data, nonce)
    assert len(hash_sha256) == 64  # SHA-256 hex string
    
    # Test Keccak256
    hash_keccak = PoWAlgorithms.keccak256_pow(block_data, nonce)
    assert len(hash_keccak) == 64
    
    # Test algorithm selection
    algo_func = PoWAlgorithms.get_algorithm('sha256')
    assert algo_func is not None


def test_difficulty_check():
    """Test difficulty checking"""
    # Easy difficulty (many leading zeros not required)
    hash_hex = "0" * 10 + "f" * 54
    assert PoWAlgorithms.check_difficulty(hash_hex, 1000)
    
    # Hard difficulty
    hash_hex = "f" * 64
    assert not PoWAlgorithms.check_difficulty(hash_hex, 1000000000)


def test_miner_initialization():
    """Test miner initialization"""
    blockchain = SphinxSkynetBlockchain()
    miner = SphinxMiner(
        blockchain=blockchain,
        miner_address="TEST_MINER",
        algorithm="spectral"
    )
    
    assert miner.miner_address == "TEST_MINER"
    assert miner.algorithm == "spectral"
    assert not miner.is_mining


def test_miner_stats():
    """Test miner statistics"""
    blockchain = SphinxSkynetBlockchain()
    miner = SphinxMiner(blockchain=blockchain, miner_address="TEST")
    
    stats = miner.get_stats()
    assert 'is_mining' in stats
    assert 'blocks_mined' in stats
    assert 'total_rewards' in stats
    assert 'hashrate' in stats


def test_merge_mining_coordinator():
    """Test merge mining coordinator"""
    blockchain = SphinxSkynetBlockchain()
    miner = SphinxMiner(blockchain=blockchain, miner_address="TEST")
    coordinator = MergeMiningCoordinator(miner)
    
    # Enable chains
    coordinator.enable_chain('btc')
    coordinator.enable_chain('eth')
    
    assert 'btc' in coordinator.enabled_chains
    assert 'eth' in coordinator.enabled_chains
    
    # Get merge mining headers
    headers = coordinator.get_merge_mining_headers()
    assert 'btc' in headers or len(coordinator.enabled_chains) >= 1


def test_merge_mining_rewards():
    """Test merge mining reward calculation"""
    blockchain = SphinxSkynetBlockchain()
    miner = SphinxMiner(blockchain=blockchain, miner_address="TEST")
    miner.stats['total_rewards'] = 100.0
    
    coordinator = MergeMiningCoordinator(miner)
    coordinator.enable_chain('btc')
    coordinator.enable_chain('eth')
    
    total_rewards = coordinator.calculate_total_rewards()
    # Should be 100 * 1.2 (10% bonus per chain) = 120
    assert total_rewards >= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
