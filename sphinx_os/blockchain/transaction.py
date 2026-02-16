"""
Transaction class for SphinxSkynet Blockchain
NO gas fees - fees paid in SPHINX tokens
"""

import time
import hashlib
import json
from typing import Optional


class Transaction:
    """
    Transaction in SPHINX tokens - NO ETH gas needed!
    Fee: 0.001 SPHINX per transaction
    """
    
    TRANSACTION_FEE = 0.001  # SPHINX tokens
    
    def __init__(
        self,
        from_address: str,
        to_address: str,
        amount: float,
        timestamp: Optional[float] = None,
        signature: Optional[str] = None
    ):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
        self.timestamp = timestamp or time.time()
        self.signature = signature
        self.tx_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate transaction hash"""
        tx_data = {
            'from': self.from_address,
            'to': self.to_address,
            'amount': self.amount,
            'timestamp': self.timestamp
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def sign_transaction(self, private_key: str):
        """Sign transaction with private key"""
        # Simple signing - in production, use proper ECDSA
        signing_data = f"{self.tx_hash}{private_key}"
        self.signature = hashlib.sha256(signing_data.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """Validate transaction"""
        # Check if from_address is set (not mining reward)
        if self.from_address is None:
            return True
        
        # Check if transaction is signed
        if not self.signature:
            return False
        
        # Check amount is positive
        if self.amount <= 0:
            return False
        
        return True
    
    def to_dict(self) -> dict:
        """Convert transaction to dictionary"""
        return {
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'tx_hash': self.tx_hash
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create transaction from dictionary"""
        tx = cls(
            from_address=data['from_address'],
            to_address=data['to_address'],
            amount=data['amount'],
            timestamp=data['timestamp'],
            signature=data.get('signature')
        )
        tx.tx_hash = data.get('tx_hash', tx.calculate_hash())
        return tx
    
    def __repr__(self):
        return f"Transaction({self.from_address[:8]}... -> {self.to_address[:8]}..., {self.amount} SPHINX)"
