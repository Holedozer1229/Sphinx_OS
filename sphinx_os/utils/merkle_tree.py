"""
Merkle Tree Implementation for SphinxSkynet Blockchain
"""

import hashlib
from typing import List, Optional


class MerkleTree:
    """Merkle tree for transaction verification"""
    
    def __init__(self, transactions: List[str]):
        """
        Initialize Merkle tree from transaction hashes
        
        Args:
            transactions: List of transaction hashes
        """
        self.transactions = transactions
        self.tree = []
        self._leaf_index: dict = {}
        self.root = self._build_tree()
    
    def _hash(self, data: str) -> str:
        """SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _build_tree(self) -> str:
        """Build Merkle tree and return root hash"""
        if not self.transactions:
            return self._hash("empty")
        
        # Start with transaction hashes as leaves
        current_level = [self._hash(tx) for tx in self.transactions]
        self.tree.append(current_level[:])
        self._leaf_index = {h: i for i, h in enumerate(current_level)}
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Pair up hashes and combine them
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Concatenate and hash
                combined = self._hash(left + right)
                next_level.append(combined)
            
            self.tree.append(next_level[:])
            current_level = next_level
        
        return current_level[0]
    
    def get_root(self) -> str:
        """Get Merkle root hash"""
        return self.root
    
    def get_proof(self, tx_hash: str) -> Optional[List[tuple]]:
        """
        Get Merkle proof for a transaction
        
        Args:
            tx_hash: Transaction hash to prove
            
        Returns:
            List of (hash, position) tuples for proof, or None if not found
        """
        # Find transaction in leaves
        tx_hash_computed = self._hash(tx_hash)
        
        if tx_hash_computed not in self._leaf_index:
            return None
        
        proof = []
        index = self._leaf_index[tx_hash_computed]
        
        # Build proof path from leaf to root
        for level in self.tree[:-1]:
            sibling_index = index ^ 1  # XOR to get sibling (flip last bit)
            
            if sibling_index < len(level):
                sibling_hash = level[sibling_index]
                position = 'right' if index % 2 == 0 else 'left'
                proof.append((sibling_hash, position))
            
            index //= 2
        
        return proof
    
    @staticmethod
    def verify_proof(tx_hash: str, merkle_root: str, proof: List[tuple]) -> bool:
        """
        Verify a Merkle proof
        
        Args:
            tx_hash: Transaction hash
            merkle_root: Expected Merkle root
            proof: Merkle proof from get_proof()
            
        Returns:
            True if proof is valid
        """
        current_hash = hashlib.sha256(tx_hash.encode()).hexdigest()
        
        for sibling_hash, position in proof:
            if position == 'left':
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == merkle_root
