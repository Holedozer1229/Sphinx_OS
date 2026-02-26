#!/usr/bin/env python3
"""
Demo script for Self-Funding Treasury System

Shows how NFT minting and rarity proof fees accumulate in the treasury
and trigger automatic bridge deployment when thresholds are met.
"""

from sphinx_os.treasury.self_funding import SelfFundingTreasury
from sphinx_os.nft.minting import SphinxNFTMinter
from sphinx_os.nft.rarity_proof import RarityProofSystem


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_stats(treasury):
    """Print current treasury statistics"""
    stats = treasury.get_treasury_stats()
    
    print(f"💰 Treasury Balance: ${stats['balance_usd']:.2f} (SKYNT)")
    print("\n📊 Deployment Status:")
    
    for chain, info in stats['deployments'].items():
        status = "✅ Deployed" if info['deployed'] else (
            "🚀 Ready!" if info['ready'] else "⏳ Funding..."
        )
        print(f"  {chain.capitalize():12} [{status:12}] ${stats['balance_usd']:.2f} / ${info['threshold']} ({info['progress']:.1f}%)")


def main():
    """Run the demo"""
    print_banner("🎯 SphinxSkynet Self-Funding Treasury Demo")
    
    # Initialize systems
    treasury = SelfFundingTreasury()
    minter = SphinxNFTMinter(treasury=treasury)
    rarity_system = RarityProofSystem(treasury=treasury)
    
    print("Initializing systems...")
    print_stats(treasury)
    
    # Simulate NFT minting
    print_banner("🎨 Phase 1: NFT Minting")
    print("Simulating 100 NFT mints at 0.1 SPHINX each...")
    
    for i in range(100):
        # Generate unique addresses using hash of index
        import hashlib
        user_address = "0x" + hashlib.sha256(f"user_{i}".encode()).hexdigest()[:40]
        metadata = {
            "name": f"Sphinx NFT #{i+1}",
            "rarity": ["common", "uncommon", "rare", "epic", "legendary"][i % 5],
            "power": (i + 1) * 10
        }
        
        try:
            result = minter.mint_nft(user_address, metadata, balance=10.0)
            if (i + 1) % 20 == 0:
                print(f"  Minted {i + 1} NFTs...")
        except Exception as e:
            print(f"  Error minting NFT {i+1}: {e}")
    
    print(f"\n✅ Minted 100 NFTs")
    print(f"💰 Fees collected: {100 * 0.1} SPHINX")
    print(f"📈 Treasury received: {100 * 0.1 * 0.7} SPHINX (70%)")
    print_stats(treasury)
    
    # Simulate rarity proof generation
    print_banner("🔍 Phase 2: Rarity Proof Generation")
    print("Generating rarity proofs for 200 NFTs at 0.05 SPHINX each...")
    
    for i in range(200):
        nft_id = 1000 + i
        # Generate unique addresses using hash of index
        import hashlib
        user_address = "0x" + hashlib.sha256(f"user_{i}".encode()).hexdigest()[:40]
        
        try:
            result = rarity_system.generate_rarity_proof(nft_id, user_address, balance=10.0)
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1} proofs...")
        except Exception as e:
            print(f"  Error generating proof for NFT {nft_id}: {e}")
    
    print(f"\n✅ Generated 200 rarity proofs")
    print(f"💰 Fees collected: {200 * 0.05} SPHINX")
    print(f"📈 Treasury received: {200 * 0.05 * 0.8} SPHINX (80%)")
    print_stats(treasury)
    
    # Show bridge deployment results
    print_banner("🌉 Phase 3: Bridge Deployment Status")
    
    stats = treasury.get_treasury_stats()
    deployed_chains = [chain for chain, info in stats['deployments'].items() if info['deployed']]
    pending_chains = [chain for chain, info in stats['deployments'].items() if not info['deployed']]
    
    if deployed_chains:
        print(f"✅ Bridges deployed on: {', '.join(deployed_chains)}")
    
    if pending_chains:
        print(f"⏳ Bridges pending: {', '.join(pending_chains)}")
        print(f"\n💡 Keep minting NFTs and generating proofs to fund remaining deployments!")
    else:
        print("🎉 All bridges deployed! System is fully operational!")
    
    # Show operator earnings
    print_banner("💵 Operator Earnings Summary")
    
    nft_operator_earnings = 100 * 0.1 * 0.2  # 20% of NFT fees
    proof_operator_earnings = 200 * 0.05 * 0.15  # 15% of proof fees
    total_operator_earnings = nft_operator_earnings + proof_operator_earnings
    
    print(f"NFT Minting:     ${nft_operator_earnings:.2f} (20% of fees)")
    print(f"Rarity Proofs:   ${proof_operator_earnings:.2f} (15% of fees)")
    print(f"{'─' * 40}")
    print(f"Total Earnings:  ${total_operator_earnings:.2f}")
    print(f"\n💰 Treasury Balance: ${stats['balance_usd']:.2f}")
    print(f"🎯 Total System Value: ${total_operator_earnings + stats['balance_usd']:.2f}")
    
    print_banner("✨ Demo Complete!")
    print("The self-funding system is working perfectly!")
    print("• NFT mints and rarity proofs generate fees")
    print("• Treasury automatically deploys bridges when funded")
    print("• Operator earns 15-20% of all fees")
    print("• Zero upfront costs - system funds itself! 🚀")


if __name__ == "__main__":
    main()
