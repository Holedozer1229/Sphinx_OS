"""
AutoMiner - Automated mining for SphinxSkynet Blockchain
Starts mining automatically with optimal settings
"""

import time
import sys
from typing import Optional
from ..blockchain.core import SphinxSkynetBlockchain
from .miner import SphinxMiner
from .merge_miner import MergeMiningCoordinator


class AutoMiner:
    """
    Automated mining with:
    - Automatic start on node launch
    - Algorithm auto-selection based on profitability
    - Merge mining coordination
    - Φ score optimization
    - Automatic payout
    """
    
    def __init__(
        self,
        blockchain: SphinxSkynetBlockchain,
        miner_address: str,
        auto_start: bool = True,
        merge_mining: bool = False,
        algorithm: Optional[str] = None
    ):
        """
        Initialize AutoMiner
        
        Args:
            blockchain: Blockchain instance
            miner_address: Payout address
            auto_start: Start mining automatically
            merge_mining: Enable merge mining
            algorithm: PoW algorithm (None for auto-select)
        """
        self.blockchain = blockchain
        self.miner_address = miner_address
        self.merge_mining_enabled = merge_mining
        
        # Auto-select algorithm if not specified
        if algorithm is None:
            algorithm = self._select_best_algorithm()
        
        self.algorithm = algorithm
        
        # Create miner
        self.miner = SphinxMiner(
            blockchain=blockchain,
            miner_address=miner_address,
            algorithm=algorithm,
            num_threads=4
        )
        
        # Create merge mining coordinator if enabled
        self.merge_coordinator = None
        if merge_mining:
            self.merge_coordinator = MergeMiningCoordinator(self.miner)
            # Enable default chains
            self.merge_coordinator.enable_chain('btc')
            self.merge_coordinator.enable_chain('eth')
            self.merge_coordinator.enable_chain('etc')
        
        # Auto-start if configured
        if auto_start:
            self.start()
    
    def _select_best_algorithm(self) -> str:
        """
        Auto-select most profitable algorithm
        
        Returns:
            Algorithm name
        """
        # For now, default to spectral (highest Φ potential)
        # In production, would analyze network hashrate and difficulty
        # for each algorithm
        return "spectral"
    
    def start(self):
        """Start automated mining"""
        print("=" * 70)
        print("🚀 SPHINXSKYNET AUTOMINER STARTING")
        print("=" * 70)
        print(f"Miner Address: {self.miner_address}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Merge Mining: {'Enabled' if self.merge_mining_enabled else 'Disabled'}")
        print("=" * 70)
        print()
        
        if self.merge_mining_enabled and self.merge_coordinator:
            self.merge_coordinator.start_merge_mining()
        else:
            self.miner.start_mining()
        
        print("✅ Mining started successfully!")
        print("Press Ctrl+C to stop...")
        print()
    
    def stop(self):
        """Stop automated mining"""
        print("\n🛑 Stopping mining...")
        
        if self.merge_mining_enabled and self.merge_coordinator:
            self.merge_coordinator.stop_merge_mining()
        else:
            self.miner.stop_mining()
        
        # Print final statistics
        self._print_final_stats()
        
        print("✅ Mining stopped successfully!")
    
    def _print_final_stats(self):
        """Print final mining statistics"""
        stats = self.miner.get_stats()
        
        print("\n" + "=" * 70)
        print("📊 FINAL MINING STATISTICS")
        print("=" * 70)
        print(f"Blocks Mined: {stats['blocks_mined']}")
        print(f"Total Rewards: {stats['total_rewards']:.2f} SKYNT")
        print(f"Average Φ Score: {stats['average_phi_score']:.2f}")
        print(f"Final Hashrate: {stats['hashrate']:.2f} H/s")
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        
        if self.merge_mining_enabled and self.merge_coordinator:
            merge_stats = self.merge_coordinator.get_stats()
            print(f"\n🔗 Merge Mining:")
            print(f"   BTC Blocks: {merge_stats['btc_blocks']}")
            print(f"   ETH Blocks: {merge_stats['eth_blocks']}")
            print(f"   ETC Blocks: {merge_stats['etc_blocks']}")
            print(f"   Total with Bonus: {merge_stats['total_rewards']:.2f} SKYNT")
        
        print("=" * 70)
    
    def run_forever(self):
        """Run mining indefinitely until interrupted"""
        try:
            while True:
                time.sleep(1)
                
                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    self._print_status()
        
        except KeyboardInterrupt:
            self.stop()
            sys.exit(0)
    
    def _print_status(self):
        """Print current mining status"""
        stats = self.miner.get_stats()
        chain_stats = self.blockchain.get_chain_stats()
        
        print(f"⛏️  Mining | Blocks: {stats['blocks_mined']} | "
              f"Rewards: {stats['total_rewards']:.2f} SKYNT | "
              f"Hashrate: {stats['hashrate']:.2f} H/s | "
              f"Chain Length: {chain_stats['chain_length']}")
    
    def get_status(self) -> dict:
        """Get current mining status"""
        miner_stats = self.miner.get_stats()
        chain_stats = self.blockchain.get_chain_stats()
        
        status = {
            'miner': miner_stats,
            'blockchain': chain_stats,
            'merge_mining': None
        }
        
        if self.merge_mining_enabled and self.merge_coordinator:
            status['merge_mining'] = self.merge_coordinator.get_stats()
        
        return status


# Entry point for standalone mining
if __name__ == "__main__":
    import sys
    
    # Initialize blockchain
    blockchain = SphinxSkynetBlockchain()
    
    # Get miner address from command line or use default
    miner_address = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT_MINER_ADDRESS"
    
    # Enable merge mining if specified
    merge_mining = "--merge" in sys.argv or "-m" in sys.argv
    
    # Create and start autominer
    autominer = AutoMiner(
        blockchain=blockchain,
        miner_address=miner_address,
        auto_start=True,
        merge_mining=merge_mining
    )
    
    # Run forever
    autominer.run_forever()
