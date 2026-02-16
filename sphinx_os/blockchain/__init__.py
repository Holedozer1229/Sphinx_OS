"""
SphinxSkynet Blockchain Module

Production-ready blockchain implementation with:
- Multiple PoW algorithms (Spectral, SHA-256, Ethash, Keccak256)
- Merge mining support
- Φ-boosted consensus
- UTXO transaction model
"""

from .block import Block
from .transaction import Transaction, TransactionInput, TransactionOutput
from .core import SphinxSkynetBlockchain
from .consensus import ConsensusEngine
from .chain_manager import ChainManager

__all__ = [
    'Block',
    'Transaction',
    'TransactionInput',
    'TransactionOutput',
    'SphinxSkynetBlockchain',
    'ConsensusEngine',
    'ChainManager'
]
