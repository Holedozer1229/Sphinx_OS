"""
SphinxOS Secure Admin Wallet - Backend
MetaMask-like wallet with secure credential management
"""

import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import sqlite3


class SecureWallet:
    """
    Secure wallet manager with encryption and authentication.
    
    Features:
    - Password-based encryption (PBKDF2)
    - Secure session management
    - Multi-account support
    - Transaction history
    """
    
    def __init__(self, db_path: str = "sphinx_wallet/wallet.db"):
        """Initialize secure wallet"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Wallets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                address TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                key_salt TEXT NOT NULL,
                chain TEXT DEFAULT 'Stacks',
                balance REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_id INTEGER NOT NULL,
                tx_hash TEXT,
                from_address TEXT NOT NULL,
                to_address TEXT NOT NULL,
                amount REAL NOT NULL,
                token TEXT DEFAULT 'STX',
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wallet_id) REFERENCES wallets(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password using PBKDF2"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # PBKDF2 with 100,000 iterations
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return pwd_hash.hex(), salt
    
    def create_user(self, username: str, password: str) -> Dict:
        """Create new user account"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return {"success": False, "error": "Username already exists"}
            
            # Hash password
            pwd_hash, salt = self.hash_password(password)
            
            # Create user
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pwd_hash, salt)
            )
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "username": username
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def authenticate(self, username: str, password: str) -> Dict:
        """Authenticate user and create session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get user
            cursor.execute(
                "SELECT id, password_hash, salt FROM users WHERE username = ?",
                (username,)
            )
            user = cursor.fetchone()
            
            if not user:
                return {"success": False, "error": "Invalid credentials"}
            
            user_id, stored_hash, salt = user
            
            # Verify password
            pwd_hash, _ = self.hash_password(password, salt)
            
            if pwd_hash != stored_hash:
                return {"success": False, "error": "Invalid credentials"}
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            cursor.execute(
                "INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)",
                (user_id, session_token, expires_at)
            )
            
            conn.commit()
            
            return {
                "success": True,
                "session_token": session_token,
                "user_id": user_id,
                "username": username
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def create_wallet(self, user_id: int, name: str, password: str) -> Dict:
        """Create new wallet for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Generate wallet keypair
            private_key = secrets.token_hex(32)
            address = "ST" + secrets.token_hex(20).upper()[:38]
            
            # Encrypt private key
            key_salt = secrets.token_hex(32)
            encrypted_key = self._encrypt_private_key(private_key, password, key_salt)
            
            cursor.execute(
                """
                INSERT INTO wallets (user_id, name, address, encrypted_key, key_salt, chain)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, name, address, encrypted_key, key_salt, 'Stacks')
            )
            
            wallet_id = cursor.lastrowid
            conn.commit()
            
            return {
                "success": True,
                "wallet_id": wallet_id,
                "address": address,
                "name": name
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()
    
    def _encrypt_private_key(self, private_key: str, password: str, salt: str) -> str:
        """Encrypt private key using password"""
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        encrypted = bytes(a ^ b for a, b in zip(private_key.encode(), key * (len(private_key) // len(key) + 1)))
        return encrypted.hex()
    
    def get_wallets(self, user_id: int) -> list:
        """Get all wallets for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, name, address, chain, balance, created_at 
                FROM wallets WHERE user_id = ?
                """,
                (user_id,)
            )
            
            wallets = []
            for row in cursor.fetchall():
                wallets.append({
                    "id": row[0],
                    "name": row[1],
                    "address": row[2],
                    "chain": row[3],
                    "balance": row[4],
                    "created_at": row[5]
                })
            
            return wallets
        
        finally:
            conn.close()


# CLI for wallet management
if __name__ == "__main__":
    import sys
    
    wallet = SecureWallet()
    
    print("=" * 70)
    print("SPHINXOS SECURE WALLET - ADMIN SETUP")
    print("=" * 70)
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        print("Creating admin user...")
        username = input("Enter admin username: ")
        password = input("Enter admin password: ")
        confirm = input("Confirm password: ")
        
        if password != confirm:
            print("❌ Passwords don't match!")
            sys.exit(1)
        
        result = wallet.create_user(username, password)
        
        if result["success"]:
            print(f"✅ Admin user '{username}' created successfully!")
            print(f"User ID: {result['user_id']}")
            
            # Create default wallet
            wallet_result = wallet.create_wallet(result['user_id'], "Main Wallet", password)
            if wallet_result["success"]:
                print(f"✅ Default wallet created: {wallet_result['address']}")
        else:
            print(f"❌ Error: {result['error']}")
    else:
        print("Usage: python wallet_backend.py create-admin")
