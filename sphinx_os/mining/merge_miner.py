"""
Merge Mining Coordinator for SphinxSkynet Blockchain
Coordinates mining across multiple chains (BTC, ETH, ETC)
"""

from typing import Dict, Optional, List
from .miner import SphinxMiner


class MergeMiningCoordinator:
    """
    Coordinates merge mining across multiple chains
    
    Supported chains:
    - BTC (Bitcoin)
    - ETH (Ethereum pre-merge)
    - ETC (Ethereum Classic)
    """
    
    def __init__(self, primary_miner: SphinxMiner):
        """
        Initialize merge mining coordinator
        
        Args:
            primary_miner: Primary SphinxSkynet miner
        """
        self.primary_miner = primary_miner
        self.enabled_chains: List[str] = []
        self.auxiliary_headers: Dict[str, str] = {}
        
        # Reward distribution (70% main, 30% aux chains)
        self.main_chain_share = 0.70
        self.aux_chain_share = 0.30
        
        # Merge mining stats
        self.stats = {
            'btc_blocks_found': 0,
            'eth_blocks_found': 0,
            'etc_blocks_found': 0,
            'skynt_blocks_found': 0,
            'total_aux_rewards': 0.0
        }
    
    def enable_chain(self, chain: str):
        """
        Enable merge mining for a chain
        
        Args:
            chain: Chain name (btc, eth, etc)
        """
        chain = chain.lower()
        supported = ['btc', 'eth', 'etc', 'skynt']
        
        if chain not in supported:
            raise ValueError(f"Unsupported chain: {chain}. Supported: {supported}")
        
        if chain not in self.enabled_chains:
            self.enabled_chains.append(chain)
            print(f"✅ Merge mining enabled for {chain.upper()}")
    
    def disable_chain(self, chain: str):
        """
        Disable merge mining for a chain
        
        Args:
            chain: Chain name
        """
        chain = chain.lower()
        if chain in self.enabled_chains:
            self.enabled_chains.remove(chain)
            self.auxiliary_headers.pop(chain, None)
            print(f"❌ Merge mining disabled for {chain.upper()}")
    
    def update_auxiliary_header(self, chain: str, header_hash: str):
        """
        Update auxiliary chain header
        
        Args:
            chain: Chain name
            header_hash: Header hash from auxiliary chain
        """
        if chain in self.enabled_chains:
            self.auxiliary_headers[chain] = header_hash
    
    def get_merge_mining_headers(self) -> Dict[str, str]:
        """
        Get current auxiliary chain headers
        
        Returns:
            Dictionary of chain headers
        """
        return self.auxiliary_headers.copy()
    
    def start_merge_mining(self):
        """Start merge mining with enabled auxiliary chains"""
        if not self.enabled_chains:
            print("⚠️  No auxiliary chains enabled")
            self.primary_miner.start_mining()
            return
        
        print(f"🔗 Starting merge mining with: {', '.join(c.upper() for c in self.enabled_chains)}")
        
        # Generate placeholder auxiliary headers
        # In production, these would be fetched from actual chains
        for chain in self.enabled_chains:
            self.auxiliary_headers[chain] = f"0x{'0' * 60}{chain[:4]}"
        
        # Start mining with merge mining headers
        self.primary_miner.start_mining(
            merge_mining_headers=self.auxiliary_headers
        )
    
    def stop_merge_mining(self):
        """Stop merge mining"""
        self.primary_miner.stop_mining()
    
    def submit_auxiliary_pow(
        self,
        chain: str,
        block_hash: str,
        nonce: int
    ) -> bool:
        """
        Submit PoW to auxiliary chain
        
        Args:
            chain: Auxiliary chain name
            block_hash: Block hash
            nonce: Nonce found
            
        Returns:
            True if accepted
        """
        # In production, this would submit to actual chain
        # For now, just track statistics
        
        if chain not in self.enabled_chains:
            return False
        
        chain_key = f"{chain}_blocks_found"
        if chain_key in self.stats:
            self.stats[chain_key] += 1
        
        print(f"📤 Submitted PoW to {chain.upper()} auxiliary chain")
        return True
    
    def calculate_total_rewards(self) -> float:
        """
        Calculate total rewards including merge mining bonus
        
        Returns:
            Total rewards
        """
        # Get primary chain rewards
        primary_rewards = self.primary_miner.stats['total_rewards']
        
        # Calculate aux chain bonus (10% per chain)
        num_aux_chains = len(self.enabled_chains)
        bonus_multiplier = 1.0 + (0.1 * num_aux_chains)
        
        return primary_rewards * bonus_multiplier
    
    def get_stats(self) -> Dict:
        """Get merge mining statistics"""
        return {
            'enabled_chains': self.enabled_chains,
            'btc_blocks': self.stats['btc_blocks_found'],
            'eth_blocks': self.stats['eth_blocks_found'],
            'etc_blocks': self.stats['etc_blocks_found'],
            'skynt_blocks': self.stats['skynt_blocks_found'],
            'total_rewards': self.calculate_total_rewards(),
            'reward_distribution': {
                'main_chain': f"{self.main_chain_share * 100}%",
                'aux_chains': f"{self.aux_chain_share * 100}%"
            }
        }
