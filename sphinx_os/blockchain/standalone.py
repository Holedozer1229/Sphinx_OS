"""
Standalone SphinxSkynet Blockchain
100% gasless, NO Ethereum dependencies, NO external blockchain costs!
"""

import time
import sqlite3
import json
from typing import List, Optional, Dict
from pathlib import Path

from .block import Block
from .transaction import Transaction


class StandaloneSphinxBlockchain:
    """
    Standalone blockchain with ZERO external dependencies.
    NO Ethereum, NO gas fees, 100% free to operate.
    
    Features:
    - Pure PoW consensus (no gas)
    - Internal token (SPHINX) with NO bridging initially
    - Transaction fees paid in SPHINX (not ETH)
    - Free mining for all users
    - Built-in wallet system (no MetaMask needed initially)
    - SQLite database (free, no external DB needed)
    - P2P networking (no centralized infrastructure)
    """
    
    MINING_REWARD = 50.0  # SPHINX tokens per block
    DIFFICULTY = 4  # Mining difficulty (number of leading zeros)
    
    def __init__(self, db_path: str = "sphinxskynet.db"):
        """
        Initialize the standalone blockchain
        
        Args:
            db_path: Path to SQLite database (default: sphinxskynet.db)
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.mining_reward = self.MINING_REWARD
        self.difficulty = self.DIFFICULTY
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Load existing chain or create genesis block
        self._load_chain()
        if len(self.chain) == 0:
            self._create_genesis_block()
    
    def _init_database(self):
        """Initialize SQLite database for blockchain storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                block_index INTEGER PRIMARY KEY,
                data TEXT NOT NULL,
                timestamp REAL NOT NULL,
                hash TEXT NOT NULL
            )
        ''')
        
        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                tx_hash TEXT PRIMARY KEY,
                from_address TEXT,
                to_address TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp REAL NOT NULL,
                block_index INTEGER,
                signature TEXT
            )
        ''')
        
        # Create balances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balances (
                address TEXT PRIMARY KEY,
                balance REAL DEFAULT 0.0
            )
        ''')
        
        # Create mining_stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mining_stats (
                address TEXT PRIMARY KEY,
                blocks_mined INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                last_mined REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_chain(self):
        """Load blockchain from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT data FROM blocks ORDER BY block_index')
        rows = cursor.fetchall()
        
        for row in rows:
            block_data = json.loads(row[0])
            block = Block.from_dict(block_data)
            self.chain.append(block)
        
        conn.close()
    
    def _save_block(self, block: Block):
        """Save block to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        block_data = json.dumps(block.to_dict())
        cursor.execute(
            'INSERT INTO blocks (block_index, data, timestamp, hash) VALUES (?, ?, ?, ?)',
            (block.index, block_data, block.timestamp, block.hash)
        )
        
        # Save transactions
        for tx in block.transactions:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions 
                (tx_hash, from_address, to_address, amount, timestamp, block_index, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx.tx_hash,
                tx.from_address,
                tx.to_address,
                tx.amount,
                tx.timestamp,
                block.index,
                tx.signature
            ))
        
        conn.commit()
        conn.close()
    
    def _create_genesis_block(self):
        """Create the genesis block (first block in the chain)"""
        genesis_tx = Transaction(
            from_address=None,
            to_address="GENESIS",
            amount=0,
            timestamp=time.time()
        )
        
        genesis_block = Block(
            index=0,
            transactions=[genesis_tx],
            timestamp=time.time(),
            previous_hash="0"
        )
        
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        self._save_block(genesis_block)
        
        print("✅ Genesis block created!")
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain"""
        return self.chain[-1] if self.chain else None
    
    def create_transaction(self, from_address: str, to_address: str, amount: float) -> Transaction:
        """
        Create a new transaction
        Fee: 0.001 SPHINX per transaction
        """
        # Check balance
        balance = self.get_balance(from_address)
        total_needed = amount + Transaction.TRANSACTION_FEE
        
        if balance < total_needed:
            raise ValueError(f"Insufficient balance. Have {balance}, need {total_needed}")
        
        # Create transaction
        tx = Transaction(from_address, to_address, amount)
        return tx
    
    def add_transaction(self, transaction: Transaction):
        """Add a transaction to pending transactions pool"""
        if not transaction.is_valid():
            raise ValueError("Cannot add invalid transaction to chain")
        
        # Check balance for non-mining transactions
        if transaction.from_address:
            balance = self.get_balance(transaction.from_address)
            total_needed = transaction.amount + Transaction.TRANSACTION_FEE
            
            if balance < total_needed:
                raise ValueError(f"Insufficient balance for transaction")
        
        self.pending_transactions.append(transaction)
        print(f"Transaction added to pending pool: {transaction}")
    
    def mine_pending_transactions(self, mining_address: str):
        """
        Mine pending transactions and create a new block
        FREE mining - NO gas costs!
        
        Args:
            mining_address: Address to receive mining reward
        """
        # Create mining reward transaction
        reward_tx = Transaction(
            from_address=None,
            to_address=mining_address,
            amount=self.mining_reward
        )
        
        # Add pending transactions to block
        block_transactions = [reward_tx] + self.pending_transactions
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            transactions=block_transactions,
            previous_hash=self.get_latest_block().hash
        )
        
        # Mine the block (PoW)
        print(f"⛏️  Mining block {new_block.index}...")
        new_block.mine_block(self.difficulty)
        
        # Add block to chain
        self.chain.append(new_block)
        self._save_block(new_block)
        
        # Update balances
        self._update_balances(new_block)
        
        # Update mining stats
        self._update_mining_stats(mining_address, self.mining_reward)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        print(f"✅ Block {new_block.index} mined successfully!")
        return new_block
    
    def _update_balances(self, block: Block):
        """Update account balances based on block transactions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tx in block.transactions:
            # Deduct from sender (if not mining reward)
            if tx.from_address:
                cursor.execute(
                    'UPDATE balances SET balance = balance - ? WHERE address = ?',
                    (tx.amount + Transaction.TRANSACTION_FEE, tx.from_address)
                )
                
                # Collect transaction fee (goes to mining address)
                # This is handled separately in revenue collection
            
            # Add to receiver
            cursor.execute(
                'INSERT INTO balances (address, balance) VALUES (?, ?) '
                'ON CONFLICT(address) DO UPDATE SET balance = balance + ?',
                (tx.to_address, tx.amount, tx.amount)
            )
        
        conn.commit()
        conn.close()
    
    def _update_mining_stats(self, address: str, reward: float):
        """Update mining statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mining_stats (address, blocks_mined, total_reward, last_mined)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                blocks_mined = blocks_mined + 1,
                total_reward = total_reward + ?,
                last_mined = ?
        ''', (address, reward, time.time(), reward, time.time()))
        
        conn.commit()
        conn.close()
    
    def get_balance(self, address: str) -> float:
        """Get balance of an address"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT balance FROM balances WHERE address = ?', (address,))
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else 0.0
    
    def get_mining_stats(self, address: str) -> Dict:
        """Get mining statistics for an address"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT blocks_mined, total_reward, last_mined FROM mining_stats WHERE address = ?',
            (address,)
        )
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                'blocks_mined': row[0],
                'total_reward': row[1],
                'last_mined': row[2]
            }
        return {
            'blocks_mined': 0,
            'total_reward': 0.0,
            'last_mined': None
        }
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # Check if all transactions are valid
            if not current_block.has_valid_transactions():
                return False
        
        return True
    
    def get_chain_info(self) -> Dict:
        """Get blockchain information"""
        return {
            'chain_length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'mining_reward': self.mining_reward,
            'difficulty': self.difficulty,
            'latest_block': self.get_latest_block().to_dict() if self.chain else None,
            'is_valid': self.is_chain_valid()
        }
    
    def get_transaction_history(self, address: str, limit: int = 100) -> List[Dict]:
        """Get transaction history for an address"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT tx_hash, from_address, to_address, amount, timestamp, block_index
            FROM transactions
            WHERE from_address = ? OR to_address = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (address, address, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'tx_hash': row[0],
                'from': row[1],
                'to': row[2],
                'amount': row[3],
                'timestamp': row[4],
                'block': row[5]
            }
            for row in rows
        ]
    
    def __repr__(self):
        return f"StandaloneSphinxBlockchain(blocks={len(self.chain)}, pending={len(self.pending_transactions)})"
