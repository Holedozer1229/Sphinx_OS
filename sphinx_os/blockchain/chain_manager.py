"""
Chain manager for SphinxSkynet Blockchain
Handles chain validation and reorganization
"""

from typing import List, Optional, Dict
from .block import Block
from .transaction import Transaction


class ChainManager:
    """
    Manages blockchain validation and reorganization
    """
    
    def __init__(self):
        self.orphan_blocks: List[Block] = []
    
    def validate_block(
        self,
        block: Block,
        previous_block: Optional[Block] = None
    ) -> bool:
        """
        Validate a block
        
        Args:
            block: Block to validate
            previous_block: Previous block in chain
            
        Returns:
            True if block is valid
        """
        # Check block size
        if not block.is_valid_size():
            return False
        
        # Validate previous hash
        if previous_block:
            if block.previous_hash != previous_block.hash:
                return False
            
            # Validate index is sequential
            if block.index != previous_block.index + 1:
                return False
        
        # Validate transactions
        for tx in block.transactions:
            if not tx.verify():
                return False
        
        # Validate at least one transaction (coinbase)
        if not block.transactions:
            return False
        
        # Validate first transaction is coinbase
        if not block.transactions[0].is_coinbase():
            return False
        
        # Validate only first transaction is coinbase
        for tx in block.transactions[1:]:
            if tx.is_coinbase():
                return False
        
        return True
    
    def validate_chain(self, chain: List[Block]) -> bool:
        """
        Validate entire blockchain
        
        Args:
            chain: List of blocks
            
        Returns:
            True if chain is valid
        """
        if not chain:
            return False
        
        # Validate genesis block
        if chain[0].index != 0:
            return False
        
        # Validate each block
        for i in range(1, len(chain)):
            if not self.validate_block(chain[i], chain[i-1]):
                return False
        
        return True
    
    def find_fork_point(
        self,
        main_chain: List[Block],
        new_chain: List[Block]
    ) -> int:
        """
        Find the fork point between two chains
        
        Args:
            main_chain: Main blockchain
            new_chain: New competing chain
            
        Returns:
            Index of fork point
        """
        fork_point = 0
        
        for i in range(min(len(main_chain), len(new_chain))):
            if main_chain[i].hash != new_chain[i].hash:
                break
            fork_point = i
        
        return fork_point
    
    def should_reorganize(
        self,
        main_chain: List[Block],
        new_chain: List[Block]
    ) -> bool:
        """
        Determine if chain should reorganize to new chain
        Uses longest chain rule
        
        Args:
            main_chain: Current blockchain
            new_chain: New competing chain
            
        Returns:
            True if should reorganize
        """
        # Validate new chain
        if not self.validate_chain(new_chain):
            return False
        
        # Longest chain wins
        return len(new_chain) > len(main_chain)
    
    def get_utxo_set(self, chain: List[Block]) -> Dict[str, Dict]:
        """
        Build UTXO set from blockchain
        
        Args:
            chain: Blockchain
            
        Returns:
            Dictionary of unspent transaction outputs
        """
        utxo_set = {}
        
        for block in chain:
            for tx in block.transactions:
                # Remove spent inputs
                for inp in tx.inputs:
                    utxo_key = f"{inp.prev_txid}:{inp.output_index}"
                    utxo_set.pop(utxo_key, None)
                
                # Add new outputs
                for idx, out in enumerate(tx.outputs):
                    utxo_key = f"{tx.txid}:{idx}"
                    utxo_set[utxo_key] = {
                        'txid': tx.txid,
                        'index': idx,
                        'address': out.address,
                        'amount': out.amount
                    }
        
        return utxo_set
    
    def get_balance(self, address: str, utxo_set: Dict[str, Dict]) -> float:
        """
        Get balance for an address from UTXO set
        
        Args:
            address: Address to check
            utxo_set: UTXO set
            
        Returns:
            Balance
        """
        balance = 0.0
        
        for utxo_key, utxo in utxo_set.items():
            if utxo['address'] == address:
                balance += utxo['amount']
        
        return balance
