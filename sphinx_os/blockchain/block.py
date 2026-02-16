"""
Block class for SphinxSkynet Blockchain
"""

import hashlib
import time
import json
from typing import List, Dict, Optional
from .transaction import Transaction
from ..utils.merkle_tree import MerkleTree


class Block:
    """
    Block for SphinxSkynet Blockchain
    Supports multiple PoW algorithms and merge mining
    """
    
    def __init__(
        self,
        index: int,
        transactions: List[Transaction],
        previous_hash: str,
        difficulty: int,
        miner: str,
        phi_score: float = 500.0,
        pow_algorithm: str = "spectral",
        merge_mining_headers: Optional[Dict[str, str]] = None
    ):
        """
        Create a new block
        
        Args:
            index: Block height
            transactions: List of transactions
            previous_hash: Hash of previous block
            difficulty: Mining difficulty
            miner: Miner address
            phi_score: Φ consciousness score (200-1000)
            pow_algorithm: PoW algorithm used (spectral, sha256, ethash, keccak256)
            merge_mining_headers: Headers from auxiliary chains (BTC, ETH, ETC)
        """
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.difficulty = difficulty
        self.miner = miner
        self.phi_score = max(200.0, min(1000.0, phi_score))
        self.pow_algorithm = pow_algorithm
        self.merge_mining_headers = merge_mining_headers or {}
        
        self.timestamp = int(time.time())
        self.nonce = 0
        
        # Calculate Merkle root
        tx_hashes = [tx.txid for tx in transactions]
        merkle_tree = MerkleTree(tx_hashes)
        self.merkle_root = merkle_tree.get_root()
        
        # Block hash (set after mining)
        self.hash = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'miner': self.miner,
            'phi_score': self.phi_score,
            'pow_algorithm': self.pow_algorithm,
            'merge_mining_headers': self.merge_mining_headers
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'merkle_root': self.merkle_root,
            'difficulty': self.difficulty,
            'miner': self.miner,
            'phi_score': self.phi_score,
            'pow_algorithm': self.pow_algorithm,
            'merge_mining_headers': self.merge_mining_headers,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        """Create block from dictionary"""
        transactions = [Transaction.from_dict(tx) for tx in data['transactions']]
        
        block = cls(
            index=data['index'],
            transactions=transactions,
            previous_hash=data['previous_hash'],
            difficulty=data['difficulty'],
            miner=data['miner'],
            phi_score=data.get('phi_score', 500.0),
            pow_algorithm=data.get('pow_algorithm', 'spectral'),
            merge_mining_headers=data.get('merge_mining_headers', {})
        )
        
        block.timestamp = data['timestamp']
        block.nonce = data['nonce']
        block.merkle_root = data['merkle_root']
        block.hash = data['hash']
        
        return block
    
    def get_size_bytes(self) -> int:
        """Get block size in bytes"""
        block_json = json.dumps(self.to_dict())
        return len(block_json.encode('utf-8'))
    
    def is_valid_size(self, max_size: int = 2 * 1024 * 1024) -> bool:
        """Check if block size is within limit (default 2MB)"""
        return self.get_size_bytes() <= max_size
    
    @staticmethod
    def create_genesis_block() -> 'Block':
        """Create the genesis block (first block in chain)"""
        genesis_tx = Transaction.create_coinbase(
            miner_address="GENESIS_ADDRESS",
            block_height=0,
            phi_boost=1.0
        )
        
        block = Block(
            index=0,
            transactions=[genesis_tx],
            previous_hash="0" * 64,
            difficulty=1000000,
            miner="GENESIS",
            phi_score=1000.0,
            pow_algorithm="spectral"
        )
        
        # Mine genesis block with nonce 0
        block.nonce = 0
        block.hash = block.calculate_hash()
        
        return block
