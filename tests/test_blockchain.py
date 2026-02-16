"""
Tests for SphinxSkynet Blockchain
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.blockchain.core import SphinxSkynetBlockchain
from sphinx_os.blockchain.block import Block
from sphinx_os.blockchain.transaction import Transaction, TransactionInput, TransactionOutput


def test_genesis_block_creation():
    """Test genesis block is created correctly"""
    blockchain = SphinxSkynetBlockchain()
    
    assert len(blockchain.chain) == 1
    assert blockchain.chain[0].index == 0
    assert blockchain.chain[0].previous_hash == "0" * 64
    assert blockchain.chain[0].phi_score == 1000.0


def test_block_creation():
    """Test creating a new block"""
    blockchain = SphinxSkynetBlockchain()
    
    block = blockchain.create_block(
        miner_address="TEST_MINER",
        phi_score=750.0,
        pow_algorithm="spectral"
    )
    
    assert block.index == 1
    assert block.miner == "TEST_MINER"
    assert block.phi_score == 750.0
    assert block.pow_algorithm == "spectral"
    assert len(block.transactions) >= 1  # At least coinbase
    assert block.transactions[0].is_coinbase()


def test_transaction_creation():
    """Test creating transactions"""
    outputs = [TransactionOutput(address="RECIPIENT", amount=100.0)]
    tx = Transaction(inputs=[], outputs=outputs, fee=0.001)
    
    assert tx.txid is not None
    assert len(tx.outputs) == 1
    assert tx.outputs[0].amount == 100.0
    assert tx.is_coinbase()


def test_coinbase_transaction():
    """Test coinbase transaction creation"""
    tx = Transaction.create_coinbase(
        miner_address="MINER",
        block_height=100,
        phi_boost=1.5
    )
    
    assert tx.is_coinbase()
    assert len(tx.inputs) == 0
    assert len(tx.outputs) == 1
    assert tx.outputs[0].amount == 50.0 * 1.5  # Base reward * phi boost


def test_utxo_balance():
    """Test UTXO balance calculation"""
    blockchain = SphinxSkynetBlockchain()
    
    # Genesis block gives 50 SPHINX to GENESIS_ADDRESS
    balance = blockchain.get_balance("GENESIS_ADDRESS")
    assert balance == 50.0


def test_chain_stats():
    """Test chain statistics"""
    blockchain = SphinxSkynetBlockchain()
    stats = blockchain.get_chain_stats()
    
    assert stats['chain_length'] == 1
    assert stats['total_transactions'] == 1
    assert stats['max_supply'] == 21_000_000


def test_merkle_tree():
    """Test Merkle tree creation"""
    from sphinx_os.utils.merkle_tree import MerkleTree
    
    txs = ["tx1", "tx2", "tx3", "tx4"]
    tree = MerkleTree(txs)
    
    root = tree.get_root()
    assert root is not None
    
    # Test proof generation and verification
    proof = tree.get_proof("tx1")
    assert proof is not None
    
    is_valid = MerkleTree.verify_proof("tx1", root, proof)
    assert is_valid


def test_difficulty_adjustment():
    """Test difficulty adjustment"""
    from sphinx_os.blockchain.consensus import ConsensusEngine
    
    consensus = ConsensusEngine()
    
    # Test no adjustment on non-adjustment block
    new_diff = consensus.calculate_next_difficulty(1000000, 100)
    assert new_diff == 1000000
    
    # Test adjustment on adjustment block (2016)
    new_diff = consensus.calculate_next_difficulty(1000000, 2016, 20160)
    assert new_diff > 0


def test_phi_boost_calculation():
    """Test Φ boost calculation"""
    from sphinx_os.blockchain.consensus import ConsensusEngine
    
    consensus = ConsensusEngine()
    
    # Min phi score (200) -> 1.0x boost
    boost = consensus.calculate_phi_boost(200.0)
    assert boost == 1.0
    
    # Max phi score (1000) -> 2.0x boost
    boost = consensus.calculate_phi_boost(1000.0)
    assert boost == 2.0
    
    # Mid phi score (600) -> 1.5x boost
    boost = consensus.calculate_phi_boost(600.0)
    assert abs(boost - 1.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
