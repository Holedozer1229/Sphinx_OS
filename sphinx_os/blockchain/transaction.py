"""
Transaction class for SphinxSkynet Blockchain
UTXO model implementation
"""

import hashlib
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TransactionInput:
    """Input reference to previous transaction output"""
    prev_txid: str
    output_index: int
    signature: str
    public_key: str = ""


@dataclass
class TransactionOutput:
    """Output sending amount to an address"""
    address: str
    amount: float


class Transaction:
    """
    UTXO-based transaction for SphinxSkynet Blockchain
    """
    
    def __init__(
        self,
        inputs: List[TransactionInput],
        outputs: List[TransactionOutput],
        fee: float = 0.001,
        phi_boost: float = 1.0
    ):
        """
        Create a new transaction
        
        Args:
            inputs: List of transaction inputs
            outputs: List of transaction outputs
            fee: Transaction fee
            phi_boost: Φ boost multiplier (1.0-2.0)
        """
        self.inputs = inputs
        self.outputs = outputs
        self.fee = fee
        self.phi_boost = max(1.0, min(2.0, phi_boost))
        self.timestamp = int(time.time())
        self.txid = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate transaction hash"""
        tx_data = {
            'inputs': [asdict(inp) for inp in self.inputs],
            'outputs': [asdict(out) for out in self.outputs],
            'fee': self.fee,
            'timestamp': self.timestamp,
            'phi_boost': self.phi_boost
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary"""
        return {
            'txid': self.txid,
            'inputs': [asdict(inp) for inp in self.inputs],
            'outputs': [asdict(out) for out in self.outputs],
            'fee': self.fee,
            'timestamp': self.timestamp,
            'phi_boost': self.phi_boost
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """Create transaction from dictionary"""
        inputs = [TransactionInput(**inp) for inp in data['inputs']]
        outputs = [TransactionOutput(**out) for out in data['outputs']]
        
        tx = cls(
            inputs=inputs,
            outputs=outputs,
            fee=data.get('fee', 0.001),
            phi_boost=data.get('phi_boost', 1.0)
        )
        tx.timestamp = data['timestamp']
        tx.txid = data['txid']
        return tx
    
    def get_total_input(self) -> float:
        """Calculate total input amount (requires UTXO lookup)"""
        # This would normally look up UTXO values
        # For now, return calculated from outputs + fee
        return self.get_total_output() + self.fee
    
    def get_total_output(self) -> float:
        """Calculate total output amount"""
        return sum(out.amount for out in self.outputs)
    
    def is_coinbase(self) -> bool:
        """Check if this is a coinbase (mining reward) transaction"""
        return len(self.inputs) == 0
    
    def verify(self, utxo_set: Optional[Dict] = None) -> bool:
        """
        Verify transaction validity
        
        Args:
            utxo_set: Available UTXOs for validation
            
        Returns:
            True if transaction is valid
        """
        # Coinbase transactions are always valid (created by miners)
        if self.is_coinbase():
            return True
        
        # Check inputs exist
        if not self.inputs:
            return False
        
        # Check outputs exist and are positive
        if not self.outputs:
            return False
        
        for output in self.outputs:
            if output.amount <= 0:
                return False
        
        # If UTXO set provided, verify inputs are unspent
        if utxo_set:
            for inp in self.inputs:
                utxo_key = f"{inp.prev_txid}:{inp.output_index}"
                if utxo_key not in utxo_set:
                    return False
        
        # Verify input amount >= output amount + fee
        # (simplified - would need actual UTXO lookup)
        total_out = self.get_total_output()
        if total_out <= 0:
            return False
        
        return True
    
    @staticmethod
    def create_coinbase(miner_address: str, block_height: int, phi_boost: float = 1.0) -> 'Transaction':
        """
        Create coinbase transaction (mining reward)
        
        Args:
            miner_address: Address to receive reward
            block_height: Current block height
            phi_boost: Φ boost multiplier
            
        Returns:
            Coinbase transaction
        """
        # Calculate block reward with halving
        base_reward = 50.0  # Initial reward
        halvings = block_height // 210000
        block_reward = base_reward / (2 ** halvings)
        
        # Apply Φ boost (1.0x to 2.0x)
        boosted_reward = block_reward * phi_boost
        
        # Create coinbase transaction (no inputs)
        outputs = [TransactionOutput(address=miner_address, amount=boosted_reward)]
        
        return Transaction(
            inputs=[],
            outputs=outputs,
            fee=0.0,
            phi_boost=phi_boost
        )
