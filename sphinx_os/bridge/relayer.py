"""
Bridge Relayer for SphinxSkynet Cross-Chain Bridge
Monitors and relays transactions between chains
"""

import time
import threading
from typing import Dict, Optional, List
from .bridge import CrossChainBridge, BridgeStatus


class BridgeRelayer:
    """
    Monitors and relays bridge transactions
    """
    
    def __init__(self, bridge: CrossChainBridge):
        """
        Initialize relayer
        
        Args:
            bridge: Bridge instance
        """
        self.bridge = bridge
        self.is_running = False
        self.relayer_thread = None
        
        # Pending transactions to relay
        self.pending_mints: List[str] = []
        self.pending_releases: List[str] = []
        
        # Statistics
        self.stats = {
            'relayed_count': 0,
            'failed_count': 0,
            'start_time': 0
        }
    
    def start(self):
        """Start relayer service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        self.relayer_thread = threading.Thread(
            target=self._relay_loop,
            daemon=True
        )
        self.relayer_thread.start()
        
        print("✅ Bridge relayer started")
    
    def stop(self):
        """Stop relayer service"""
        self.is_running = False
        if self.relayer_thread:
            self.relayer_thread.join(timeout=5)
        
        print("🛑 Bridge relayer stopped")
    
    def _relay_loop(self):
        """Main relay loop"""
        while self.is_running:
            try:
                # Process pending mints
                self._process_pending_mints()
                
                # Process pending releases
                self._process_pending_releases()
                
                # Sleep before next iteration
                time.sleep(10)
            
            except Exception as e:
                print(f"Relay error: {e}")
                time.sleep(5)
    
    def _process_pending_mints(self):
        """Process pending mint transactions"""
        for tx_hash in self.pending_mints[:]:
            try:
                tx_status = self.bridge.get_transaction_status(tx_hash)
                
                if not tx_status:
                    self.pending_mints.remove(tx_hash)
                    continue
                
                if tx_status['status'] == BridgeStatus.LOCKED.value:
                    # Generate guardian signatures (simulated)
                    signatures = self._get_guardian_signatures(tx_hash, 5)
                    
                    # Attempt mint
                    success = self.bridge.mint_wrapped_tokens(
                        tx_hash=tx_hash,
                        recipient=tx_status['recipient'],
                        signatures=signatures
                    )
                    
                    if success:
                        print(f"✅ Minted tokens for tx {tx_hash[:16]}...")
                        self.pending_mints.remove(tx_hash)
                        self.stats['relayed_count'] += 1
                    else:
                        self.stats['failed_count'] += 1
            
            except Exception as e:
                print(f"Error processing mint {tx_hash[:16]}...: {e}")
    
    def _process_pending_releases(self):
        """Process pending release transactions"""
        for tx_hash in self.pending_releases[:]:
            try:
                tx_status = self.bridge.get_transaction_status(tx_hash)
                
                if not tx_status:
                    self.pending_releases.remove(tx_hash)
                    continue
                
                if tx_status['status'] == BridgeStatus.BURNED.value:
                    # Generate guardian signatures (simulated)
                    signatures = self._get_guardian_signatures(tx_hash, 5)
                    
                    # Attempt release
                    success = self.bridge.release_tokens(
                        tx_hash=tx_hash,
                        recipient=tx_status['recipient'],
                        signatures=signatures
                    )
                    
                    if success:
                        print(f"✅ Released tokens for tx {tx_hash[:16]}...")
                        self.pending_releases.remove(tx_hash)
                        self.stats['relayed_count'] += 1
                    else:
                        self.stats['failed_count'] += 1
            
            except Exception as e:
                print(f"Error processing release {tx_hash[:16]}...: {e}")
    
    def _get_guardian_signatures(self, tx_hash: str, count: int) -> List[str]:
        """
        Get guardian signatures for transaction
        In production, this would collect actual signatures from guardians
        
        Args:
            tx_hash: Transaction hash
            count: Number of signatures needed
            
        Returns:
            List of guardian signatures
        """
        # Simulate getting signatures from guardians
        return [f"GUARDIAN_{i}" for i in range(1, count + 1)]
    
    def queue_mint(self, tx_hash: str):
        """Queue a transaction for minting"""
        if tx_hash not in self.pending_mints:
            self.pending_mints.append(tx_hash)
            print(f"📥 Queued mint for tx {tx_hash[:16]}...")
    
    def queue_release(self, tx_hash: str):
        """Queue a transaction for release"""
        if tx_hash not in self.pending_releases:
            self.pending_releases.append(tx_hash)
            print(f"📥 Queued release for tx {tx_hash[:16]}...")
    
    def get_stats(self) -> Dict:
        """Get relayer statistics"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] > 0 else 0
        
        return {
            'is_running': self.is_running,
            'relayed_count': self.stats['relayed_count'],
            'failed_count': self.stats['failed_count'],
            'pending_mints': len(self.pending_mints),
            'pending_releases': len(self.pending_releases),
            'uptime_seconds': uptime
        }
