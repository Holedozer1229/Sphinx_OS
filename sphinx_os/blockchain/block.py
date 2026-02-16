"""
Block class for SphinxSkynet Blockchain
Pure PoW consensus - NO gas fees!
"""

import time
import hashlib
import json
from typing import List, Optional
from .transaction import Transaction


class Block:
    """
    Block in the SphinxSkynet blockchain
    Uses Proof of Work (PoW) consensus - NO gas required!
    """
    
    def __init__(
        self,
        index: int,
        transactions: List[Transaction],
        timestamp: Optional[float] = None,
        previous_hash: str = "0",
        nonce: int = 0
    ):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate block hash using SHA-256"""
        block_data = {
            'index': self.index,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """
        Mine block using Proof of Work
        FREE mining - NO gas costs!
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        print(f"Block mined! Hash: {self.hash}")
    
    def has_valid_transactions(self) -> bool:
        """Validate all transactions in the block"""
        for tx in self.transactions:
            if not tx.is_valid():
                return False
        return True
    
    def to_dict(self) -> dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """Create block from dictionary"""
        transactions = [Transaction.from_dict(tx) for tx in data['transactions']]
        block = cls(
            index=data['index'],
            transactions=transactions,
            timestamp=data['timestamp'],
            previous_hash=data['previous_hash'],
            nonce=data['nonce']
        )
        block.hash = data['hash']
        return block
    
    def __repr__(self):
        return f"Block(#{self.index}, {len(self.transactions)} txs, hash={self.hash[:8]}...)"
