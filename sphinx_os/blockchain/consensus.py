"""
Consensus engine for SphinxSkynet Blockchain
Hybrid PoW + Proof-of-Consciousness (Φ)
"""

from typing import Optional
from ..utils.difficulty import DifficultyAdjuster


class ConsensusEngine:
    """
    Hybrid consensus combining PoW with Φ scoring
    """
    
    def __init__(self):
        self.difficulty_adjuster = DifficultyAdjuster()
    
    def validate_pow(self, block_hash: str, difficulty: int) -> bool:
        """
        Validate proof of work
        
        Args:
            block_hash: Block hash to validate
            difficulty: Required difficulty
            
        Returns:
            True if PoW is valid
        """
        # Check if hash meets difficulty target
        # Difficulty = number of leading zeros required
        hash_int = int(block_hash, 16)
        target = 2 ** (256 - difficulty.bit_length())
        
        return hash_int < target
    
    def validate_phi_score(self, phi_score: float) -> bool:
        """
        Validate Φ consciousness score
        
        Args:
            phi_score: Φ score to validate
            
        Returns:
            True if valid (200-1000 range)
        """
        return 200.0 <= phi_score <= 1000.0
    
    def calculate_phi_boost(self, phi_score: float) -> float:
        """
        Calculate mining reward boost from Φ score
        
        Args:
            phi_score: Φ score (200-1000)
            
        Returns:
            Boost multiplier (1.0-2.0)
        """
        # Linear mapping: 200 -> 1.0x, 1000 -> 2.0x
        normalized = (phi_score - 200) / 800
        boost = 1.0 + normalized
        
        return max(1.0, min(2.0, boost))
    
    def validate_block_consensus(
        self,
        block_hash: str,
        difficulty: int,
        phi_score: float,
        pow_algorithm: str
    ) -> bool:
        """
        Validate block meets consensus requirements
        
        Args:
            block_hash: Block hash
            difficulty: Block difficulty
            phi_score: Φ score
            pow_algorithm: PoW algorithm used
            
        Returns:
            True if block is valid
        """
        # Validate PoW
        if not self.validate_pow(block_hash, difficulty):
            return False
        
        # Validate Φ score
        if not self.validate_phi_score(phi_score):
            return False
        
        # Validate algorithm
        valid_algorithms = ['spectral', 'sha256', 'ethash', 'keccak256']
        if pow_algorithm not in valid_algorithms:
            return False
        
        return True
    
    def calculate_next_difficulty(
        self,
        current_difficulty: int,
        block_height: int,
        last_interval_time: Optional[int] = None
    ) -> int:
        """
        Calculate difficulty for next block
        
        Args:
            current_difficulty: Current difficulty
            block_height: Current block height
            last_interval_time: Actual time for last 2016 blocks
            
        Returns:
            New difficulty
        """
        # Check if adjustment needed
        if not self.difficulty_adjuster.should_adjust(block_height):
            return current_difficulty
        
        # If no timing info, keep current difficulty
        if last_interval_time is None:
            return current_difficulty
        
        expected_time = self.difficulty_adjuster.get_expected_time_for_interval()
        
        return self.difficulty_adjuster.calculate_next_difficulty(
            current_difficulty,
            last_interval_time,
            expected_time
        )
