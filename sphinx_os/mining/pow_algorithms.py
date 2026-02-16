"""
PoW Algorithms for SphinxSkynet Blockchain
"""

import hashlib
import struct
from typing import Tuple


class PoWAlgorithms:
    """
    Multiple Proof-of-Work algorithms
    """
    
    @staticmethod
    def sha256_pow(block_data: str, nonce: int) -> str:
        """
        Bitcoin-compatible SHA-256 PoW
        
        Args:
            block_data: Block data to hash
            nonce: Nonce value
            
        Returns:
            Hash result
        """
        data = f"{block_data}{nonce}".encode()
        return hashlib.sha256(hashlib.sha256(data).digest()).hexdigest()
    
    @staticmethod
    def keccak256_pow(block_data: str, nonce: int) -> str:
        """
        ETC-compatible Keccak256 PoW
        
        Args:
            block_data: Block data to hash
            nonce: Nonce value
            
        Returns:
            Hash result
        """
        data = f"{block_data}{nonce}".encode()
        return hashlib.sha3_256(data).hexdigest()
    
    @staticmethod
    def spectral_pow(block_data: str, nonce: int) -> str:
        """
        Spectral hash PoW using Riemann zeta properties
        
        Args:
            block_data: Block data to hash
            nonce: Nonce value
            
        Returns:
            Hash result
        """
        from sphinx_wallet.backend.spectral_hash import SpectralHash
        
        spectral = SpectralHash()
        data = f"{block_data}{nonce}".encode()
        
        # Compute spectral signature
        return spectral.compute_spectral_signature(data)
    
    @staticmethod
    def ethash_simplified_pow(block_data: str, nonce: int) -> str:
        """
        Simplified Ethash-style PoW (not full DAG-based)
        For production, use full Ethash implementation
        
        Args:
            block_data: Block data to hash
            nonce: Nonce value
            
        Returns:
            Hash result
        """
        # Simplified version - multiple rounds of Keccak
        data = f"{block_data}{nonce}".encode()
        
        hash_result = data
        for _ in range(64):  # Multiple rounds
            hash_result = hashlib.sha3_256(hash_result).digest()
        
        return hash_result.hex()
    
    @staticmethod
    def get_algorithm(name: str):
        """
        Get PoW algorithm function by name
        
        Args:
            name: Algorithm name
            
        Returns:
            Algorithm function
        """
        algorithms = {
            'sha256': PoWAlgorithms.sha256_pow,
            'keccak256': PoWAlgorithms.keccak256_pow,
            'spectral': PoWAlgorithms.spectral_pow,
            'ethash': PoWAlgorithms.ethash_simplified_pow
        }
        
        return algorithms.get(name, PoWAlgorithms.spectral_pow)
    
    @staticmethod
    def check_difficulty(hash_hex: str, difficulty: int) -> bool:
        """
        Check if hash meets difficulty target
        
        Args:
            hash_hex: Hash to check
            difficulty: Required difficulty
            
        Returns:
            True if hash meets difficulty
        """
        # Convert hash to integer
        hash_int = int(hash_hex, 16)
        
        # Calculate target (2^(256-difficulty_bits))
        target = 2 ** (256 - difficulty.bit_length())
        
        return hash_int < target
