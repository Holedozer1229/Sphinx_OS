"""
Tests for SphinxSkynet Cross-Chain Bridge
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.bridge.bridge import CrossChainBridge, BridgeStatus
from sphinx_os.bridge.validator import BridgeValidator, ZKProofVerifier


def test_bridge_initialization():
    """Test bridge initialization"""
    bridge = CrossChainBridge()
    
    assert len(bridge.guardians) == 9
    assert bridge.REQUIRED_SIGNATURES == 5
    assert bridge.BRIDGE_FEE == 0.001


def test_supported_chains():
    """Test supported chains"""
    bridge = CrossChainBridge()
    
    chains = bridge.get_supported_chains()
    assert len(chains) == 7
    
    chain_ids = [c['chain_id'] for c in chains]
    assert 'btc' in chain_ids
    assert 'eth' in chain_ids
    assert 'etc' in chain_ids


def test_lock_tokens():
    """Test locking tokens"""
    bridge = CrossChainBridge()
    
    tx_hash = bridge.lock_tokens(
        source_chain='eth',
        amount=100.0,
        sender='SENDER_ETH',
        recipient='RECIPIENT_SPHINX'
    )
    
    assert tx_hash is not None
    assert len(tx_hash) == 64  # SHA-256 hex
    
    # Check status
    status = bridge.get_transaction_status(tx_hash)
    assert status is not None
    assert status['status'] == BridgeStatus.LOCKED.value


def test_mint_tokens():
    """Test minting wrapped tokens"""
    bridge = CrossChainBridge()
    
    # Lock first
    tx_hash = bridge.lock_tokens(
        source_chain='eth',
        amount=100.0,
        sender='SENDER',
        recipient='RECIPIENT'
    )
    
    # Mint with guardian signatures
    signatures = [f"GUARDIAN_{i}" for i in range(1, 6)]
    success = bridge.mint_wrapped_tokens(
        tx_hash=tx_hash,
        recipient='RECIPIENT',
        signatures=signatures
    )
    
    assert success
    
    # Check wrapped balance
    balance = bridge.get_wrapped_balance('RECIPIENT')
    assert balance > 0


def test_burn_and_release():
    """Test burn and release flow"""
    bridge = CrossChainBridge()
    
    # Setup: Lock and mint first
    lock_hash = bridge.lock_tokens('eth', 100.0, 'SENDER', 'USER')
    signatures = [f"GUARDIAN_{i}" for i in range(1, 6)]
    bridge.mint_wrapped_tokens(lock_hash, 'USER', signatures)
    
    # Burn tokens
    burn_hash = bridge.burn_wrapped_tokens(
        amount=50.0,
        sender='USER',
        destination_chain='btc',
        recipient='BTC_ADDRESS'
    )
    
    assert burn_hash is not None
    
    # Release with guardian signatures
    success = bridge.release_tokens(
        tx_hash=burn_hash,
        recipient='BTC_ADDRESS',
        signatures=signatures
    )
    
    assert success


def test_bridge_fees():
    """Test bridge fee calculation"""
    bridge = CrossChainBridge()
    
    amount = 1000.0
    fee = amount * bridge.BRIDGE_FEE
    net_amount = amount - fee
    
    assert fee == 1.0  # 0.1% of 1000
    assert net_amount == 999.0


def test_validator():
    """Test bridge validator"""
    validator = BridgeValidator(required_signatures=5, total_guardians=9)
    
    assert len(validator.guardians) == 9
    assert validator.required_signatures == 5


def test_zk_proof_verifier():
    """Test ZK proof verifier"""
    verifier = ZKProofVerifier()
    
    tx_data = {'amount': 100, 'sender': 'A', 'recipient': 'B'}
    proof = verifier.generate_proof(tx_data)
    
    assert proof is not None
    
    # Verify proof
    is_valid = verifier.verify_proof(proof, tx_data)
    assert is_valid
    
    # Invalid proof should fail
    is_valid = verifier.verify_proof("invalid_proof", tx_data)
    assert not is_valid


def test_invalid_chain():
    """Test invalid chain rejection"""
    bridge = CrossChainBridge()
    
    tx_hash = bridge.lock_tokens(
        source_chain='invalid_chain',
        amount=100.0,
        sender='SENDER',
        recipient='RECIPIENT'
    )
    
    assert tx_hash is None


def test_insufficient_signatures():
    """Test insufficient signature rejection"""
    bridge = CrossChainBridge()
    
    tx_hash = bridge.lock_tokens('eth', 100.0, 'SENDER', 'RECIPIENT')
    
    # Try to mint with only 3 signatures (need 5)
    signatures = [f"GUARDIAN_{i}" for i in range(1, 4)]
    success = bridge.mint_wrapped_tokens(tx_hash, 'RECIPIENT', signatures)
    
    assert not success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
