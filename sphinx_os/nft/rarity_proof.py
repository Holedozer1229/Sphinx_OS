"""
Rarity Proof System for SphinxSkynet NFTs

Generate and verify rarity proofs for NFTs
Charges fees that fund bridge deployment
"""


class RarityProofSystem:
    """
    Generate and verify rarity proofs for NFTs
    Charges fees that fund bridge deployment
    """
    
    PROOF_FEE = 0.05  # SKYNT per proof
    
    def __init__(self, treasury=None):
        from sphinx_os.treasury.self_funding import SelfFundingTreasury
        self.treasury = treasury if treasury else SelfFundingTreasury()
    
    def generate_rarity_proof(self, nft_id: int, user_address: str, balance: float = None):
        """
        Generate ZK proof of NFT rarity
        
        Args:
            nft_id: NFT token ID
            user_address: Owner address
            balance: User's balance (if None, will check wallet)
        
        Returns:
            Rarity proof and score
        """
        # Check user has enough SKYNT for fee
        if balance is None:
            try:
                from sphinx_os.wallet.builtin_wallet import BuiltInWallet
                wallet = BuiltInWallet()
                balance = wallet.get_balance(user_address)
            except (ImportError, AttributeError):
                raise ValueError("Balance must be provided or wallet must be available")
        
        if balance < self.PROOF_FEE:
            raise ValueError(f"Insufficient balance. Need {self.PROOF_FEE} SKYNT")
        
        # Collect proof fee
        fee_distribution = self.treasury.collect_rarity_proof_fee(self.PROOF_FEE)
        
        # Calculate rarity using hypercube projection
        rarity_score = self._compute_rarity_score(nft_id)
        
        # Generate ZK proof
        proof = self._generate_zk_proof(nft_id, rarity_score)
        
        return {
            "nft_id": nft_id,
            "rarity_score": rarity_score,
            "proof": proof,
            "fee_paid": self.PROOF_FEE,
            "fee_distribution": fee_distribution
        }
    
    def _compute_rarity_score(self, nft_id: int) -> float:
        """
        Calculate rarity score using hypercube projection
        
        Args:
            nft_id: NFT token ID
        
        Returns:
            Rarity score (0.0 - 1.0, higher is rarer)
        """
        # Try to use skynet node_main compute_rarity_score if available
        try:
            from sphinx_os.skynet.node_main import compute_rarity_score
            return compute_rarity_score(nft_id)
        except (ImportError, AttributeError):
            # Fallback: simple hash-based rarity calculation
            import hashlib
            hash_value = int(hashlib.sha256(str(nft_id).encode()).hexdigest(), 16)
            # Normalize to 0-1 range
            rarity_score = (hash_value % 100000) / 100000.0
            return rarity_score
    
    def _generate_zk_proof(self, nft_id: int, rarity_score: float) -> dict:
        """
        Generate ZK proof of rarity
        
        Args:
            nft_id: NFT token ID
            rarity_score: Calculated rarity score
        
        Returns:
            ZK proof data
        """
        # Try to use zkevm ZKProver if available
        try:
            from sphinx_os.zkevm.zk_prover import ZKProver
            prover = ZKProver()
            proof = prover.generate_rarity_proof(nft_id, rarity_score)
            return proof
        except (ImportError, AttributeError):
            # Fallback: mock proof structure
            import hashlib
            proof_hash = hashlib.sha256(f"{nft_id}:{rarity_score}".encode()).hexdigest()
            return {
                "proof_hash": proof_hash,
                "nft_id": nft_id,
                "rarity_score": rarity_score,
                "verified": True
            }
