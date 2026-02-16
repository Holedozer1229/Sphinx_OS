"""
Tests for SphinxSkynet Gasless Blockchain
"""

import pytest
import time
from pathlib import Path
import tempfile
import os

from sphinx_os.blockchain.standalone import StandaloneSphinxBlockchain
from sphinx_os.blockchain.transaction import Transaction
from sphinx_os.wallet.builtin_wallet import BuiltInWallet, WalletManager
from sphinx_os.mining.free_miner import FreeMiner, MiningTier
from sphinx_os.revenue.fee_collector import FeeCollector
from sphinx_os.revenue.subscriptions import SubscriptionManager, SubscriptionTier
from sphinx_os.revenue.referrals import ReferralProgram


class TestBlockchain:
    """Test standalone blockchain functionality"""
    
    def setup_method(self):
        """Set up test blockchain"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_blockchain.db")
        self.blockchain = StandaloneSphinxBlockchain(self.db_path)
    
    def test_genesis_block_created(self):
        """Test that genesis block is created"""
        assert len(self.blockchain.chain) == 1
        assert self.blockchain.chain[0].index == 0
        assert self.blockchain.chain[0].previous_hash == "0"
    
    def test_blockchain_valid(self):
        """Test blockchain validation"""
        assert self.blockchain.is_chain_valid()
    
    def test_create_transaction(self):
        """Test creating a transaction"""
        # First, need to mine a block to have balance
        wallet = BuiltInWallet()
        self.blockchain.mine_pending_transactions(wallet.address)
        
        # Now create transaction
        tx = self.blockchain.create_transaction(
            wallet.address,
            "0xSPHINXRECEIVER",
            10.0
        )
        
        assert tx.from_address == wallet.address
        assert tx.to_address == "0xSPHINXRECEIVER"
        assert tx.amount == 10.0
    
    def test_mine_block(self):
        """Test mining a block"""
        wallet = BuiltInWallet()
        
        # Mine a block
        block = self.blockchain.mine_pending_transactions(wallet.address)
        
        assert block.index == 1
        assert len(self.blockchain.chain) == 2
        
        # Check mining reward
        balance = self.blockchain.get_balance(wallet.address)
        assert balance == self.blockchain.mining_reward
    
    def test_get_balance(self):
        """Test getting wallet balance"""
        wallet = BuiltInWallet()
        
        # Initially should be 0
        balance = self.blockchain.get_balance(wallet.address)
        assert balance == 0.0
        
        # After mining, should have reward
        self.blockchain.mine_pending_transactions(wallet.address)
        balance = self.blockchain.get_balance(wallet.address)
        assert balance == self.blockchain.mining_reward


class TestWallet:
    """Test wallet functionality"""
    
    def test_create_wallet(self):
        """Test creating a wallet"""
        wallet = BuiltInWallet()
        
        assert wallet.address is not None
        assert wallet.private_key is not None
        assert wallet.mnemonic is not None
        assert wallet.address.startswith("0xSPHINX")
    
    def test_sign_message(self):
        """Test message signing"""
        wallet = BuiltInWallet()
        message = "test message"
        
        signature = wallet.sign_message(message)
        assert signature is not None
        assert len(signature) == 64  # SHA-256 hash length in hex
    
    def test_verify_signature(self):
        """Test signature verification"""
        wallet = BuiltInWallet()
        message = "test message"
        
        signature = wallet.sign_message(message)
        assert wallet.verify_signature(message, signature)
    
    def test_wallet_manager(self):
        """Test wallet manager"""
        temp_dir = tempfile.mkdtemp()
        manager = WalletManager(temp_dir)
        
        # Create wallet
        wallet = manager.create_wallet("test_wallet")
        assert wallet is not None
        
        # Get wallet
        retrieved = manager.get_wallet("test_wallet")
        assert retrieved.address == wallet.address
        
        # List wallets
        wallets = manager.list_wallets()
        assert "test_wallet" in wallets


class TestMining:
    """Test mining functionality"""
    
    def test_create_miner(self):
        """Test creating a miner"""
        wallet = BuiltInWallet()
        miner = FreeMiner(wallet.address, MiningTier.FREE)
        
        assert miner.address == wallet.address
        assert miner.tier == MiningTier.FREE
        assert miner.config.hashrate == "10 MH/s"
    
    def test_start_stop_mining(self):
        """Test starting and stopping mining"""
        wallet = BuiltInWallet()
        miner = FreeMiner(wallet.address)
        
        assert not miner.is_mining
        
        miner.start_mining()
        assert miner.is_mining
        
        miner.stop_mining()
        assert not miner.is_mining
    
    def test_mining_tiers(self):
        """Test different mining tiers"""
        wallet = BuiltInWallet()
        
        # Free tier
        free_miner = FreeMiner(wallet.address, MiningTier.FREE)
        assert free_miner.config.hashrate_value == 10
        assert free_miner.config.cost == 0.0
        
        # Premium tier
        premium_miner = FreeMiner(wallet.address, MiningTier.PREMIUM)
        assert premium_miner.config.hashrate_value == 100
        assert premium_miner.config.cost == 5.0
        
        # Pro tier
        pro_miner = FreeMiner(wallet.address, MiningTier.PRO)
        assert pro_miner.config.hashrate_value == 1000
        assert pro_miner.config.cost == 20.0
    
    def test_upgrade_tier(self):
        """Test upgrading mining tier"""
        wallet = BuiltInWallet()
        miner = FreeMiner(wallet.address, MiningTier.FREE)
        
        assert miner.tier == MiningTier.FREE
        
        miner.upgrade_tier(MiningTier.PREMIUM)
        assert miner.tier == MiningTier.PREMIUM
        assert miner.config.hashrate_value == 100


class TestRevenue:
    """Test revenue/monetization functionality"""
    
    def test_fee_collector(self):
        """Test fee collection"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "revenue.db")
        
        collector = FeeCollector("0xOPERATOR", db_path)
        
        # Collect transaction fee
        collector.collect_transaction_fee("tx_hash_123")
        
        # Get today's revenue
        revenue = collector.get_daily_revenue()
        assert revenue['transaction_fees'] == 0.001
    
    def test_subscription_manager(self):
        """Test subscription management"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "subscriptions.db")
        
        manager = SubscriptionManager(db_path)
        
        # Create subscription
        sub = manager.create_subscription("user123", SubscriptionTier.PREMIUM)
        
        assert sub['tier'] == 'premium'
        assert sub['status'] == 'active'
        
        # Get subscription
        retrieved = manager.get_subscription("user123")
        assert retrieved['tier'] == 'premium'
    
    def test_referral_program(self):
        """Test referral program"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "referrals.db")
        
        program = ReferralProgram(db_path)
        
        # Generate referral code
        code = program.generate_referral_code("user123")
        assert code is not None
        assert len(code) == 8
        
        # Track referral
        program.track_referral("user123", "user456")
        
        # Get stats
        stats = program.get_referral_stats("user123")
        assert stats['total_referrals'] == 1
        
        # Distribute commission
        commission = program.distribute_commission("user456", 100.0)
        assert commission == 5.0  # 5% of 100


class TestIntegration:
    """Integration tests"""
    
    def test_complete_flow(self):
        """Test complete flow: create wallet, mine, transact"""
        temp_dir = tempfile.mkdtemp()
        
        # Create blockchain
        blockchain = StandaloneSphinxBlockchain(
            os.path.join(temp_dir, "blockchain.db")
        )
        
        # Create wallets
        wallet1 = BuiltInWallet()
        wallet2 = BuiltInWallet()
        
        # Mine block to get initial balance
        blockchain.mine_pending_transactions(wallet1.address)
        
        balance1 = blockchain.get_balance(wallet1.address)
        assert balance1 == 50.0  # Mining reward
        
        # Create and add transaction
        tx = blockchain.create_transaction(
            wallet1.address,
            wallet2.address,
            10.0
        )
        tx.sign_transaction(wallet1.private_key)
        blockchain.add_transaction(tx)
        
        # Mine another block
        blockchain.mine_pending_transactions(wallet1.address)
        
        # Check balances
        balance1 = blockchain.get_balance(wallet1.address)
        balance2 = blockchain.get_balance(wallet2.address)
        
        # wallet1 should have: 50 (first mining) + 50 (second mining) - 10 (sent) - 0.001 (fee)
        assert balance1 > 89.0  # Approximately 89.999
        
        # wallet2 should have: 10 (received)
        assert balance2 == 10.0
        
        # Blockchain should be valid
        assert blockchain.is_chain_valid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
