#!/usr/bin/env python3
"""
SphinxSkynet Gasless Blockchain Demo
Demonstrates the complete gasless blockchain system
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sphinx_os.blockchain.standalone import StandaloneSphinxBlockchain
from sphinx_os.wallet.builtin_wallet import BuiltInWallet
from sphinx_os.mining.free_miner import FreeMiner, MiningTier
from sphinx_os.revenue.fee_collector import FeeCollector
from sphinx_os.revenue.subscriptions import SubscriptionManager, SubscriptionTier
from sphinx_os.revenue.referrals import ReferralProgram


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_blockchain():
    """Demonstrate blockchain functionality"""
    print_section("1. STANDALONE BLOCKCHAIN (NO GAS FEES!)")
    
    # Create blockchain
    blockchain = StandaloneSphinxBlockchain("demo_blockchain.db")
    print(f"✅ Blockchain created with {len(blockchain.chain)} block(s)")
    print(f"   Genesis block hash: {blockchain.chain[0].hash[:16]}...")
    print(f"   Mining difficulty: {blockchain.difficulty}")
    print(f"   Mining reward: {blockchain.mining_reward} SPHINX")


def demo_wallet():
    """Demonstrate wallet functionality"""
    print_section("2. BUILT-IN WALLET (NO METAMASK!)")
    
    # Create wallet
    wallet = BuiltInWallet()
    print(f"✅ Wallet created:")
    print(f"   Address: {wallet.address}")
    print(f"   Private Key: {wallet.private_key[:16]}...")
    print(f"   Mnemonic: {' '.join(wallet.mnemonic.split()[:3])}... (12 words)")


def demo_mining():
    """Demonstrate mining functionality"""
    print_section("3. FREE MINING SYSTEM")
    
    wallet = BuiltInWallet()
    
    # Create miners with different tiers
    print("🔨 Mining Tiers:")
    
    free_miner = FreeMiner(wallet.address, MiningTier.FREE)
    print(f"   Free: {free_miner.config.hashrate} - ${free_miner.config.cost}/month")
    
    premium_miner = FreeMiner(wallet.address, MiningTier.PREMIUM)
    print(f"   Premium: {premium_miner.config.hashrate} - ${premium_miner.config.cost}/month")
    
    pro_miner = FreeMiner(wallet.address, MiningTier.PRO)
    print(f"   Pro: {pro_miner.config.hashrate} - ${pro_miner.config.cost}/month")
    
    # Estimate earnings
    print("\n💰 Estimated Earnings (24 hours):")
    for tier_name, miner in [("Free", free_miner), ("Premium", premium_miner), ("Pro", pro_miner)]:
        estimate = miner.calculate_estimated_earnings(24)
        print(f"   {tier_name}: {estimate['estimated_earnings']:.2f} SPHINX/day")


def demo_transaction():
    """Demonstrate transaction flow"""
    print_section("4. GASLESS TRANSACTIONS")
    
    # Create blockchain and wallets
    blockchain = StandaloneSphinxBlockchain("demo_blockchain.db")
    wallet1 = BuiltInWallet()
    wallet2 = BuiltInWallet()
    
    print("👛 Wallets created:")
    print(f"   Wallet 1: {wallet1.address[:20]}...")
    print(f"   Wallet 2: {wallet2.address[:20]}...")
    
    # Mine block to get initial balance
    print("\n⛏️  Mining block to get SPHINX tokens...")
    blockchain.mine_pending_transactions(wallet1.address)
    
    balance1 = blockchain.get_balance(wallet1.address)
    print(f"✅ Wallet 1 balance: {balance1} SPHINX")
    
    # Create transaction
    print(f"\n💸 Sending 10 SPHINX from Wallet 1 to Wallet 2...")
    tx = blockchain.create_transaction(wallet1.address, wallet2.address, 10.0)
    tx.sign_transaction(wallet1.private_key)
    blockchain.add_transaction(tx)
    
    print(f"   Transaction fee: {tx.TRANSACTION_FEE} SPHINX (NOT ETH!)")
    
    # Mine another block
    print("\n⛏️  Mining block with transaction...")
    blockchain.mine_pending_transactions(wallet1.address)
    
    # Check balances
    balance1 = blockchain.get_balance(wallet1.address)
    balance2 = blockchain.get_balance(wallet2.address)
    
    print(f"\n✅ Final balances:")
    print(f"   Wallet 1: {balance1:.3f} SPHINX")
    print(f"   Wallet 2: {balance2:.3f} SPHINX")


def demo_revenue():
    """Demonstrate revenue/monetization features"""
    print_section("5. REVENUE & MONETIZATION")
    
    # Fee collector
    print("💰 Fee Collection:")
    collector = FeeCollector("0xOPERATOR", "demo_revenue.db")
    
    # Simulate collecting fees
    for i in range(10):
        collector.collect_transaction_fee(f"tx_hash_{i}")
    
    revenue = collector.get_daily_revenue()
    print(f"   Today's transaction fees: {revenue['transaction_fees']} SPHINX")
    print(f"   Total revenue: {revenue['total_revenue']:.4f} SPHINX")
    
    # Subscriptions
    print("\n📊 Subscription System:")
    sub_manager = SubscriptionManager("demo_subscriptions.db")
    
    # Create subscriptions
    sub_manager.create_subscription("user1", SubscriptionTier.PREMIUM)
    sub_manager.create_subscription("user2", SubscriptionTier.PRO)
    
    stats = sub_manager.get_subscription_stats()
    print(f"   Active subscriptions: {stats['active_subscriptions']}")
    print(f"   Premium users: {stats['premium_users']}")
    print(f"   Pro users: {stats['pro_users']}")
    print(f"   Monthly recurring revenue: ${stats['monthly_revenue']:.2f}")
    
    # Referral program
    print("\n🎁 Referral Program:")
    referral = ReferralProgram("demo_referrals.db")
    
    # Create referral
    code = referral.generate_referral_code("user1")
    referral.track_referral("user1", "user2")
    referral.distribute_commission("user2", 100.0)
    
    ref_stats = referral.get_referral_stats("user1")
    print(f"   Referral code: {code}")
    print(f"   Total referrals: {ref_stats['total_referrals']}")
    print(f"   Commission earned: {ref_stats['total_commission']:.2f} SPHINX")
    print(f"   Commission rate: {ref_stats['commission_rate']*100}%")


def demo_complete_flow():
    """Demonstrate complete flow"""
    print_section("6. COMPLETE FLOW DEMO")
    
    print("🚀 Complete User Journey:")
    print("   1. User creates wallet (FREE)")
    print("   2. User starts mining (FREE)")
    print("   3. User mines blocks and earns SPHINX")
    print("   4. User sends transactions (0.001 SPHINX fee)")
    print("   5. User upgrades to Premium ($5/month)")
    print("   6. User refers friends (5% commission)")
    print("   7. User earns passive income!")
    
    print("\n💡 Revenue Projections:")
    print("\n   Week 1:")
    print("     • 100 free miners → $10/day in tx fees")
    print("     • 5 premium users → $25/month")
    print("     • Total: ~$70-100")
    
    print("\n   Month 1:")
    print("     • 1,000 free miners → $100/day in tx fees")
    print("     • 50 premium users → $250/month")
    print("     • 10 hosted nodes → $100/month")
    print("     • Total: ~$3,000-3,500")
    
    print("\n   Month 3:")
    print("     • 10,000 free miners → $500/day in tx fees")
    print("     • 200 premium users → $1,000/month")
    print("     • 50 hosted nodes → $500/month")
    print("     • Total: ~$15,000-20,000/month")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  SPHINXSKYNET GASLESS BLOCKCHAIN DEMO")
    print("  100% FREE • NO GAS FEES • START EARNING TODAY!")
    print("=" * 60)
    
    try:
        demo_blockchain()
        demo_wallet()
        demo_mining()
        demo_transaction()
        demo_revenue()
        demo_complete_flow()
        
        print("\n" + "=" * 60)
        print("  ✅ DEMO COMPLETE!")
        print("=" * 60)
        print("\n🚀 Ready to deploy?")
        print("   1. Run: uvicorn sphinx_os.api.main:app --reload")
        print("   2. Visit: http://localhost:8000/docs")
        print("   3. Start earning!")
        print("\n📚 Documentation: GASLESS_BLOCKCHAIN.md")
        print("=" * 60 + "\n")
        
    finally:
        # Cleanup demo files
        import os
        for db_file in ["demo_blockchain.db", "demo_revenue.db", 
                       "demo_subscriptions.db", "demo_referrals.db"]:
            if os.path.exists(db_file):
                os.remove(db_file)


if __name__ == "__main__":
    main()
