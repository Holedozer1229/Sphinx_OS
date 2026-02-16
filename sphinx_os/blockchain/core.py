"""
SphinxSkynet Blockchain Core
Main blockchain implementation
"""

import time
import json
from typing import List, Dict, Optional
from .block import Block
from .transaction import Transaction, TransactionInput, TransactionOutput
from .consensus import ConsensusEngine
from .chain_manager import ChainManager


class SphinxSkynetBlockchain:
    """
    Production-ready blockchain implementation
    
    Features:
    - Multiple PoW algorithms
    - UTXO transaction model
    - Φ-boosted consensus
    - Difficulty adjustment
    - Chain reorganization
    """
    
    # Constants
    BLOCK_TIME_TARGET = 10  # seconds
    MAX_BLOCK_SIZE = 2 * 1024 * 1024  # 2 MB
    MAX_SUPPLY = 21_000_000  # 21 million SPHINX
    HALVING_INTERVAL = 210_000  # blocks
    
    def __init__(self):
        """Initialize blockchain"""
        self.chain: List[Block] = []
        self.transaction_pool: List[Transaction] = []
        self.consensus = ConsensusEngine()
        self.chain_manager = ChainManager()
        
        # Create genesis block
        genesis = Block.create_genesis_block()
        self.chain.append(genesis)
        
        # Statistics
        self.stats = {
            'total_blocks': 1,
            'total_transactions': 1,
            'total_mined': 50.0,
            'network_hashrate': 0,
            'current_difficulty': genesis.difficulty
        }
    
    def get_latest_block(self) -> Block:
        """Get the latest block in chain"""
        return self.chain[-1]
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Block]:
        """Get block by hash"""
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None
    
    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get block by height/index"""
        if 0 <= height < len(self.chain):
            return self.chain[height]
        return None
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add transaction to pool
        
        Args:
            transaction: Transaction to add
            
        Returns:
            True if added successfully
        """
        # Get UTXO set for validation
        utxo_set = self.chain_manager.get_utxo_set(self.chain)
        
        # Validate transaction
        if not transaction.verify(utxo_set):
            return False
        
        # Check for double-spend in pool
        for tx in self.transaction_pool:
            for inp1 in transaction.inputs:
                for inp2 in tx.inputs:
                    if (inp1.prev_txid == inp2.prev_txid and 
                        inp1.output_index == inp2.output_index):
                        return False
        
        self.transaction_pool.append(transaction)
        return True
    
    def create_block(
        self,
        miner_address: str,
        phi_score: float = 500.0,
        pow_algorithm: str = "spectral",
        merge_mining_headers: Optional[Dict[str, str]] = None,
        max_transactions: int = 1000
    ) -> Block:
        """
        Create a new block (before mining)
        
        Args:
            miner_address: Miner's address for reward
            phi_score: Φ consciousness score
            pow_algorithm: PoW algorithm to use
            merge_mining_headers: Auxiliary chain headers
            max_transactions: Max transactions per block
            
        Returns:
            New unmined block
        """
        latest_block = self.get_latest_block()
        
        # Calculate difficulty
        difficulty = self.consensus.calculate_next_difficulty(
            current_difficulty=latest_block.difficulty,
            block_height=latest_block.index + 1
        )
        
        # Select transactions from pool
        selected_transactions = self.transaction_pool[:max_transactions]
        
        # Create coinbase transaction
        phi_boost = self.consensus.calculate_phi_boost(phi_score)
        coinbase = Transaction.create_coinbase(
            miner_address=miner_address,
            block_height=latest_block.index + 1,
            phi_boost=phi_boost
        )
        
        # Add merge mining bonus
        if merge_mining_headers:
            bonus_multiplier = 1.0 + (0.1 * len(merge_mining_headers))
            for output in coinbase.outputs:
                output.amount *= bonus_multiplier
        
        # Combine coinbase with transactions
        all_transactions = [coinbase] + selected_transactions
        
        # Create block
        block = Block(
            index=latest_block.index + 1,
            transactions=all_transactions,
            previous_hash=latest_block.hash,
            difficulty=difficulty,
            miner=miner_address,
            phi_score=phi_score,
            pow_algorithm=pow_algorithm,
            merge_mining_headers=merge_mining_headers
        )
        
        return block
    
    def add_block(self, block: Block) -> bool:
        """
        Add mined block to chain
        
        Args:
            block: Mined block to add
            
        Returns:
            True if added successfully
        """
        latest_block = self.get_latest_block()
        
        # Validate block
        if not self.chain_manager.validate_block(block, latest_block):
            return False
        
        # Validate consensus
        if not self.consensus.validate_block_consensus(
            block.hash,
            block.difficulty,
            block.phi_score,
            block.pow_algorithm
        ):
            return False
        
        # Add to chain
        self.chain.append(block)
        
        # Remove mined transactions from pool
        mined_txids = {tx.txid for tx in block.transactions}
        self.transaction_pool = [
            tx for tx in self.transaction_pool
            if tx.txid not in mined_txids
        ]
        
        # Update stats
        self.stats['total_blocks'] += 1
        self.stats['total_transactions'] += len(block.transactions)
        self.stats['current_difficulty'] = block.difficulty
        
        # Update total mined
        for tx in block.transactions:
            if tx.is_coinbase():
                self.stats['total_mined'] += tx.get_total_output()
        
        return True
    
    def get_balance(self, address: str) -> float:
        """
        Get balance for an address
        
        Args:
            address: Address to check
            
        Returns:
            Balance in SPHINX
        """
        utxo_set = self.chain_manager.get_utxo_set(self.chain)
        return self.chain_manager.get_balance(address, utxo_set)
    
    def get_transaction(self, txid: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        # Search in chain
        for block in self.chain:
            for tx in block.transactions:
                if tx.txid == txid:
                    return tx
        
        # Search in pool
        for tx in self.transaction_pool:
            if tx.txid == txid:
                return tx
        
        return None
    
    def get_chain_stats(self) -> Dict:
        """Get blockchain statistics"""
        latest_block = self.get_latest_block()
        
        return {
            'chain_length': len(self.chain),
            'total_transactions': self.stats['total_transactions'],
            'total_supply': self.stats['total_mined'],
            'max_supply': self.MAX_SUPPLY,
            'current_difficulty': self.stats['current_difficulty'],
            'latest_block_hash': latest_block.hash,
            'latest_block_height': latest_block.index,
            'transactions_in_pool': len(self.transaction_pool),
            'target_block_time': self.BLOCK_TIME_TARGET
        }
    
    def save_to_file(self, filename: str):
        """Save blockchain to file"""
        chain_data = [block.to_dict() for block in self.chain]
        
        with open(filename, 'w') as f:
            json.dump({
                'chain': chain_data,
                'stats': self.stats
            }, f, indent=2)
    
    def load_from_file(self, filename: str) -> bool:
        """Load blockchain from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load blocks
            chain = [Block.from_dict(block_data) for block_data in data['chain']]
            
            # Validate chain
            if not self.chain_manager.validate_chain(chain):
                return False
            
            self.chain = chain
            self.stats = data.get('stats', self.stats)
            
            return True
        except Exception as e:
            print(f"Error loading blockchain: {e}")
            return False
