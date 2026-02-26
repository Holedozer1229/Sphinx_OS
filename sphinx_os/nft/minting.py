"""
NFT Minting System with Self-Funding Fees for SphinxSkynet
"""


class SphinxNFTMinter:
    """
    NFT minting system with self-funding fees
    """
    
    MINT_FEE = 0.1  # SKYNT per mint
    
    def __init__(self, treasury=None):
        from sphinx_os.treasury.self_funding import SelfFundingTreasury
        self.treasury = treasury if treasury else SelfFundingTreasury()
        self._nft_counter = 1000  # Start from 1000
    
    def mint_nft(self, user_address: str, metadata: dict, balance: float = None):
        """
        Mint NFT and collect fee
        
        Args:
            user_address: User's wallet address
            metadata: NFT metadata (image, rarity, attributes)
            balance: User's balance (if None, will check wallet)
        
        Returns:
            NFT token ID and transaction details
        """
        # Check user has enough SKYNT for fee
        if balance is None:
            # Try to import wallet, but if not available, raise error
            try:
                from sphinx_os.wallet.builtin_wallet import BuiltInWallet
                wallet = BuiltInWallet()
                balance = wallet.get_balance(user_address)
            except (ImportError, AttributeError):
                raise ValueError("Balance must be provided or wallet must be available")
        
        if balance < self.MINT_FEE:
            raise ValueError(f"Insufficient balance. Need {self.MINT_FEE} SKYNT")
        
        # Collect minting fee
        fee_distribution = self.treasury.collect_nft_mint_fee(self.MINT_FEE)
        
        # Deduct fee from user (if wallet available)
        try:
            from sphinx_os.wallet.builtin_wallet import BuiltInWallet
            wallet = BuiltInWallet()
            wallet.deduct_balance(user_address, self.MINT_FEE)
        except (ImportError, AttributeError):
            # Wallet not available, fee already collected in treasury
            pass
        
        # Create NFT
        nft_id = self._create_nft(user_address, metadata)
        
        return {
            "nft_id": nft_id,
            "fee_paid": self.MINT_FEE,
            "fee_distribution": fee_distribution,
            "metadata": metadata
        }
    
    def _create_nft(self, owner: str, metadata: dict):
        """Create NFT on blockchain"""
        # Implementation for NFT creation
        # For now, return a sequential NFT ID
        nft_id = self._nft_counter
        self._nft_counter += 1
        return nft_id
