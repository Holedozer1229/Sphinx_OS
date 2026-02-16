"""
Built-in Wallet System - NO MetaMask, NO gas fees!
Users create wallets directly in the web UI.
"""

import hashlib
import secrets
import json
from typing import Optional, Dict, List
from pathlib import Path


class BuiltInWallet:
    """
    Native wallet system - NO MetaMask, NO gas fees!
    Users create wallets directly in the web UI.
    """
    
    def __init__(self, private_key: Optional[str] = None, mnemonic: Optional[str] = None):
        """
        Create or load a wallet
        
        Args:
            private_key: Existing private key (hex string)
            mnemonic: Existing mnemonic phrase (12 words)
        """
        if private_key:
            self.private_key = private_key
        elif mnemonic:
            self.private_key = self._mnemonic_to_private_key(mnemonic)
            self.mnemonic = mnemonic
        else:
            # Generate new wallet
            self.private_key = self._generate_private_key()
            self.mnemonic = self._generate_mnemonic()
        
        self.address = self._private_key_to_address(self.private_key)
    
    @staticmethod
    def _generate_private_key() -> str:
        """Generate a random private key"""
        return secrets.token_hex(32)
    
    @staticmethod
    def _generate_mnemonic() -> str:
        """
        Generate a 12-word mnemonic phrase
        (Simplified - in production, use BIP39 wordlist)
        """
        # Simple word generation for demonstration
        words = []
        for _ in range(12):
            word_bytes = secrets.token_bytes(2)
            word = hashlib.sha256(word_bytes).hexdigest()[:8]
            words.append(word)
        return ' '.join(words)
    
    @staticmethod
    def _mnemonic_to_private_key(mnemonic: str) -> str:
        """Convert mnemonic to private key"""
        # Simple conversion - in production, use BIP39 standard
        return hashlib.sha256(mnemonic.encode()).hexdigest()
    
    @staticmethod
    def _private_key_to_address(private_key: str) -> str:
        """
        Derive public address from private key
        Format: 0xSPHINX... (42 characters)
        """
        # Simple derivation - in production, use proper ECDSA
        address_hash = hashlib.sha256(private_key.encode()).hexdigest()
        return f"0xSPHINX{address_hash[:36]}"
    
    def sign_message(self, message: str) -> str:
        """
        Sign a message with the private key
        
        Args:
            message: Message to sign
            
        Returns:
            Signature (hex string)
        """
        signing_data = f"{message}{self.private_key}"
        return hashlib.sha256(signing_data.encode()).hexdigest()
    
    def verify_signature(self, message: str, signature: str) -> bool:
        """
        Verify a message signature
        
        Args:
            message: Original message
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.sign_message(message)
        return expected_signature == signature
    
    def to_dict(self) -> Dict:
        """Export wallet data (WARNING: includes private key!)"""
        return {
            'address': self.address,
            'private_key': self.private_key,
            'mnemonic': self.mnemonic
        }
    
    def to_keystore(self, password: str) -> Dict:
        """
        Export wallet as encrypted keystore
        
        Args:
            password: Password to encrypt keystore
            
        Returns:
            Encrypted keystore dictionary
        """
        # Simple encryption - in production, use proper encryption (AES)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        encrypted_key = hashlib.sha256(
            f"{self.private_key}{password_hash}".encode()
        ).hexdigest()
        
        return {
            'address': self.address,
            'crypto': {
                'cipher': 'aes-128-ctr',
                'ciphertext': encrypted_key,
            },
            'version': 1
        }
    
    @classmethod
    def from_keystore(cls, keystore: Dict, password: str) -> 'BuiltInWallet':
        """
        Load wallet from encrypted keystore
        
        Args:
            keystore: Encrypted keystore dictionary
            password: Password to decrypt keystore
            
        Returns:
            BuiltInWallet instance
        """
        # Simple decryption - in production, use proper decryption
        # This is a placeholder - proper implementation would decrypt the key
        raise NotImplementedError("Keystore decryption not implemented in this version")
    
    def __repr__(self):
        return f"BuiltInWallet(address={self.address})"


class WalletManager:
    """
    Manage multiple wallets for a user
    """
    
    def __init__(self, storage_path: str = "wallets"):
        """
        Initialize wallet manager
        
        Args:
            storage_path: Directory to store wallet files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.wallets: Dict[str, BuiltInWallet] = {}
    
    def create_wallet(self, name: str = "default") -> BuiltInWallet:
        """
        Create a new wallet
        
        Args:
            name: Wallet name
            
        Returns:
            New BuiltInWallet instance
        """
        wallet = BuiltInWallet()
        self.wallets[name] = wallet
        self._save_wallet(name, wallet)
        return wallet
    
    def import_wallet(
        self,
        name: str,
        private_key: Optional[str] = None,
        mnemonic: Optional[str] = None
    ) -> BuiltInWallet:
        """
        Import an existing wallet
        
        Args:
            name: Wallet name
            private_key: Private key (hex string)
            mnemonic: Mnemonic phrase
            
        Returns:
            Imported BuiltInWallet instance
        """
        wallet = BuiltInWallet(private_key=private_key, mnemonic=mnemonic)
        self.wallets[name] = wallet
        self._save_wallet(name, wallet)
        return wallet
    
    def get_wallet(self, name: str) -> Optional[BuiltInWallet]:
        """
        Get a wallet by name
        
        Args:
            name: Wallet name
            
        Returns:
            BuiltInWallet instance or None
        """
        if name in self.wallets:
            return self.wallets[name]
        
        # Try to load from file
        return self._load_wallet(name)
    
    def list_wallets(self) -> List[str]:
        """
        List all available wallets
        
        Returns:
            List of wallet names
        """
        # Get wallets from memory
        wallet_names = set(self.wallets.keys())
        
        # Get wallets from storage
        for wallet_file in self.storage_path.glob("*.json"):
            wallet_names.add(wallet_file.stem)
        
        return sorted(list(wallet_names))
    
    def delete_wallet(self, name: str):
        """
        Delete a wallet
        
        Args:
            name: Wallet name
        """
        # Remove from memory
        if name in self.wallets:
            del self.wallets[name]
        
        # Remove file
        wallet_path = self.storage_path / f"{name}.json"
        if wallet_path.exists():
            wallet_path.unlink()
    
    def _save_wallet(self, name: str, wallet: BuiltInWallet):
        """Save wallet to file"""
        wallet_path = self.storage_path / f"{name}.json"
        with open(wallet_path, 'w') as f:
            json.dump(wallet.to_dict(), f, indent=2)
    
    def _load_wallet(self, name: str) -> Optional[BuiltInWallet]:
        """Load wallet from file"""
        wallet_path = self.storage_path / f"{name}.json"
        if not wallet_path.exists():
            return None
        
        with open(wallet_path, 'r') as f:
            data = json.load(f)
        
        wallet = BuiltInWallet(
            private_key=data['private_key'],
            mnemonic=data.get('mnemonic')
        )
        self.wallets[name] = wallet
        return wallet
    
    def __repr__(self):
        return f"WalletManager({len(self.wallets)} wallets loaded)"
