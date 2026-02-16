"""
Difficulty adjustment for SphinxSkynet Blockchain
"""

from typing import List


class DifficultyAdjuster:
    """
    Bitcoin-style difficulty adjustment
    Adjusts every 2016 blocks to maintain target block time
    """
    
    TARGET_BLOCK_TIME = 10  # seconds
    ADJUSTMENT_INTERVAL = 2016  # blocks
    MAX_ADJUSTMENT_FACTOR = 4  # Maximum adjustment per period
    
    @staticmethod
    def calculate_next_difficulty(
        current_difficulty: int,
        actual_time: int,
        expected_time: int
    ) -> int:
        """
        Calculate next difficulty based on actual vs expected time
        
        Args:
            current_difficulty: Current difficulty target
            actual_time: Actual time taken for last interval
            expected_time: Expected time for interval
            
        Returns:
            New difficulty target
        """
        # Prevent extreme adjustments
        min_time = expected_time // DifficultyAdjuster.MAX_ADJUSTMENT_FACTOR
        max_time = expected_time * DifficultyAdjuster.MAX_ADJUSTMENT_FACTOR
        
        actual_time = max(min_time, min(max_time, actual_time))
        
        # Adjust difficulty proportionally
        new_difficulty = int(current_difficulty * expected_time / actual_time)
        
        # Ensure minimum difficulty
        return max(1000, new_difficulty)
    
    @staticmethod
    def should_adjust(block_height: int) -> bool:
        """Check if difficulty should be adjusted at this block height"""
        return block_height > 0 and block_height % DifficultyAdjuster.ADJUSTMENT_INTERVAL == 0
    
    @staticmethod
    def get_expected_time_for_interval() -> int:
        """Get expected time for adjustment interval"""
        return DifficultyAdjuster.TARGET_BLOCK_TIME * DifficultyAdjuster.ADJUSTMENT_INTERVAL
    
    @staticmethod
    def apply_phi_adjustment(difficulty: int, phi_score: float) -> int:
        """
        Apply Φ-based difficulty adjustment
        Higher Φ = slightly lower difficulty (reward quality)
        
        Args:
            difficulty: Base difficulty
            phi_score: Φ score (200-1000)
            
        Returns:
            Adjusted difficulty
        """
        # Φ reduces difficulty by 0-10% based on score
        phi_normalized = (phi_score - 200) / 800  # 0-1 range
        phi_reduction = phi_normalized * 0.1  # 0-10% reduction
        
        adjusted = int(difficulty * (1 - phi_reduction))
        return max(1000, adjusted)
