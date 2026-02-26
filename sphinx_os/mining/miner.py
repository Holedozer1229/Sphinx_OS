"""
SphinxMiner - Main mining engine for SphinxSkynet Blockchain
"""

import time
import threading
from typing import Optional, Dict, Callable
from ..blockchain.core import SphinxSkynetBlockchain
from ..blockchain.block import Block
from .pow_algorithms import PoWAlgorithms
from .spectral_pow import SpectralPoW


class SphinxMiner:
    """
    Multi-algorithm mining engine
    
    Features:
    - Spectral PoW
    - SHA-256
    - Ethash
    - Keccak256
    - Multi-threaded mining
    - Φ-boosted rewards
    """
    
    def __init__(
        self,
        blockchain: SphinxSkynetBlockchain,
        miner_address: str,
        algorithm: str = "spectral",
        num_threads: int = 4
    ):
        """
        Initialize miner
        
        Args:
            blockchain: Blockchain instance
            miner_address: Address to receive rewards
            algorithm: PoW algorithm to use
            num_threads: Number of mining threads
        """
        self.blockchain = blockchain
        self.miner_address = miner_address
        self.algorithm = algorithm
        self.num_threads = num_threads
        
        self.is_mining = False
        self.mining_thread = None
        self.current_block: Optional[Block] = None
        
        # Mining statistics
        self.stats = {
            'blocks_mined': 0,
            'total_hashrate': 0,
            'average_phi_score': 500.0,
            'total_rewards': 0.0,
            'start_time': 0,
            'last_block_time': 0
        }
        
        # Algorithm selection
        self.pow_algorithm = PoWAlgorithms.get_algorithm(algorithm)
        self.spectral_pow = SpectralPoW() if algorithm == "spectral" else None
    
    def start_mining(self, merge_mining_headers: Optional[Dict[str, str]] = None):
        """
        Start mining process
        
        Args:
            merge_mining_headers: Optional merge mining headers
        """
        if self.is_mining:
            return
        
        self.is_mining = True
        self.stats['start_time'] = time.time()
        
        # Start mining thread
        self.mining_thread = threading.Thread(
            target=self._mine_loop,
            args=(merge_mining_headers,),
            daemon=True
        )
        self.mining_thread.start()
    
    def stop_mining(self):
        """Stop mining process"""
        self.is_mining = False
        if self.mining_thread:
            self.mining_thread.join(timeout=5)
    
    def _mine_loop(self, merge_mining_headers: Optional[Dict[str, str]] = None):
        """Main mining loop"""
        while self.is_mining:
            try:
                # Mine a block
                block = self.mine_block(merge_mining_headers)
                
                if block:
                    # Add to blockchain
                    if self.blockchain.add_block(block):
                        self.stats['blocks_mined'] += 1
                        self.stats['last_block_time'] = time.time()
                        
                        # Update rewards
                        for tx in block.transactions:
                            if tx.is_coinbase():
                                self.stats['total_rewards'] += tx.get_total_output()
                        
                        # Update average Φ score
                        old_avg = self.stats['average_phi_score']
                        count = self.stats['blocks_mined']
                        new_avg = (old_avg * (count - 1) + block.phi_score) / count
                        self.stats['average_phi_score'] = new_avg
                        
                        print(f"✅ Block #{block.index} mined! Hash: {block.hash[:16]}...")
                        print(f"   Φ Score: {block.phi_score:.2f}, Reward: {self.stats['total_rewards']:.2f} SKYNT")
                    else:
                        print("❌ Failed to add mined block to chain")
                
                # Small delay to prevent CPU spinning
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Mining error: {e}")
                time.sleep(1)
    
    def mine_block(
        self,
        merge_mining_headers: Optional[Dict[str, str]] = None,
        max_attempts: int = 1000000
    ) -> Optional[Block]:
        """
        Mine a single block
        
        Args:
            merge_mining_headers: Optional merge mining headers
            max_attempts: Maximum mining attempts
            
        Returns:
            Mined block or None
        """
        # Calculate Φ score for this mining attempt
        phi_score = 500.0  # Default
        if self.spectral_pow:
            seed_data = f"{self.miner_address}{time.time()}".encode()
            phi_score = self.spectral_pow.compute_phi_score(seed_data)
        
        # Create block template
        block = self.blockchain.create_block(
            miner_address=self.miner_address,
            phi_score=phi_score,
            pow_algorithm=self.algorithm,
            merge_mining_headers=merge_mining_headers
        )
        
        self.current_block = block
        
        # Mine using selected algorithm
        if self.algorithm == "spectral" and self.spectral_pow:
            return self._mine_spectral(block, max_attempts)
        else:
            return self._mine_standard(block, max_attempts)
    
    def _mine_spectral(self, block: Block, max_attempts: int) -> Optional[Block]:
        """
        Mine using spectral PoW
        
        Args:
            block: Block to mine
            max_attempts: Maximum attempts
            
        Returns:
            Mined block or None
        """
        block_data = block.calculate_hash()
        
        nonce, hash_result, phi_score = self.spectral_pow.mine_with_phi(
            block_data,
            block.difficulty,
            max_attempts
        )
        
        if nonce is not None:
            block.nonce = nonce
            block.hash = hash_result
            block.phi_score = phi_score
            return block
        
        return None
    
    def _mine_standard(self, block: Block, max_attempts: int) -> Optional[Block]:
        """
        Mine using standard PoW algorithms
        
        Args:
            block: Block to mine
            max_attempts: Maximum attempts
            
        Returns:
            Mined block or None
        """
        start_time = time.time()
        
        for nonce in range(max_attempts):
            block.nonce = nonce
            block_hash = block.calculate_hash()
            
            # Check if hash meets difficulty
            if PoWAlgorithms.check_difficulty(block_hash, block.difficulty):
                block.hash = block_hash
                
                # Calculate hashrate
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.stats['total_hashrate'] = nonce / elapsed
                
                return block
            
            # Update hashrate periodically
            if nonce % 10000 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.stats['total_hashrate'] = nonce / elapsed
        
        return None
    
    def get_hashrate(self) -> float:
        """Get current hashrate in hashes/second"""
        return self.stats['total_hashrate']
    
    def get_stats(self) -> Dict:
        """Get mining statistics"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] > 0 else 0
        
        return {
            'is_mining': self.is_mining,
            'algorithm': self.algorithm,
            'blocks_mined': self.stats['blocks_mined'],
            'total_rewards': self.stats['total_rewards'],
            'hashrate': self.stats['total_hashrate'],
            'average_phi_score': self.stats['average_phi_score'],
            'uptime_seconds': uptime,
            'miner_address': self.miner_address,
            'current_block_height': self.current_block.index if self.current_block else 0
        }
