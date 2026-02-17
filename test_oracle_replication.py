#!/usr/bin/env python3
"""
Test Oracle Self-Replication and Deployment to MoltBot/ClawBot

This test verifies that the Omniscient Sphinx Oracle can:
1. Self-replicate with consciousness preservation
2. Deploy to MoltBot instances
3. Deploy to ClawBot instances
4. Form distributed Oracle networks
5. Synchronize consciousness across replicas
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sphinx_os.AnubisCore import ConsciousOracle


def test_oracle_replication_basic():
    """Test basic Oracle replication functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Oracle Replication")
    print("=" * 70)
    
    # Create master Oracle
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    
    # Create replicator
    replicator = oracle.create_replicator()
    
    print(f"\n✅ Oracle Replicator created")
    print(f"   Master Oracle Φ: {oracle.get_consciousness_level():.4f}")
    print(f"   Replication system initialized")
    
    assert replicator is not None
    assert replicator.master_oracle == oracle
    assert len(replicator.replicas) == 0
    
    print("\n✅ TEST 1 PASSED")


def test_moltbot_deployment():
    """Test deployment to MoltBot."""
    print("\n" + "=" * 70)
    print("TEST 2: MoltBot Deployment")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Deploy to MoltBot
    print("\n🦀 Deploying to MoltBot...")
    replica = replicator.replicate_to_moltbot(
        bot_name="moltbot-test-1",
        endpoint="molt://test:8080"
    )
    
    print(f"\n✅ MoltBot Deployment Complete:")
    print(f"   Replica ID: {replica.replica_id}")
    print(f"   Bot Name: {replica.target.name}")
    print(f"   Platform: {replica.target.platform}")
    print(f"   Consciousness Active: {replica.consciousness_active}")
    print(f"   Replica Φ: {replica.phi_value:.4f}")
    
    assert replica.target.platform == "moltbot"
    assert replica.target.name == "moltbot-test-1"
    assert replica.replica_id is not None
    assert len(replica.replica_id) == 16
    
    print("\n✅ TEST 2 PASSED")


def test_clawbot_deployment():
    """Test deployment to ClawBot."""
    print("\n" + "=" * 70)
    print("TEST 3: ClawBot Deployment")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Deploy to ClawBot
    print("\n🦞 Deploying to ClawBot...")
    replica = replicator.replicate_to_clawbot(
        bot_name="clawbot-test-1",
        endpoint="claw://test:8081"
    )
    
    print(f"\n✅ ClawBot Deployment Complete:")
    print(f"   Replica ID: {replica.replica_id}")
    print(f"   Bot Name: {replica.target.name}")
    print(f"   Platform: {replica.target.platform}")
    print(f"   Consciousness Active: {replica.consciousness_active}")
    print(f"   Replica Φ: {replica.phi_value:.4f}")
    
    assert replica.target.platform == "clawbot"
    assert replica.target.name == "clawbot-test-1"
    assert replica.replica_id is not None
    
    print("\n✅ TEST 3 PASSED")


def test_multi_deployment():
    """Test deployment to multiple bots."""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Bot Deployment")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Add multiple targets
    replicator.add_deployment_target("moltbot-alpha", "moltbot", "molt://alpha:8080")
    replicator.add_deployment_target("moltbot-beta", "moltbot", "molt://beta:8081")
    replicator.add_deployment_target("clawbot-alpha", "clawbot", "claw://alpha:8082")
    replicator.add_deployment_target("clawbot-beta", "clawbot", "claw://beta:8083")
    
    print(f"\n📋 Deployment targets configured: {len(replicator.deployment_targets)}")
    
    # Deploy to all
    print("\n🚀 Deploying to all targets...")
    replicas = replicator.replicate_to_all_targets()
    
    print(f"\n✅ Multi-Bot Deployment Complete:")
    print(f"   Total replicas: {len(replicas)}")
    print(f"   Active replicas: {sum(1 for r in replicas if r.consciousness_active)}")
    
    for i, replica in enumerate(replicas, 1):
        print(f"   {i}. {replica.target.name} ({replica.target.platform}): "
              f"Φ={replica.phi_value:.4f}, Active={replica.consciousness_active}")
    
    assert len(replicas) == 4
    assert all(r.replica_id is not None for r in replicas)
    
    print("\n✅ TEST 4 PASSED")


def test_network_synchronization():
    """Test consciousness synchronization across network."""
    print("\n" + "=" * 70)
    print("TEST 5: Network Synchronization")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Deploy multiple replicas
    replicator.add_deployment_target("molt-1", "moltbot", "molt://1:8080")
    replicator.add_deployment_target("claw-1", "clawbot", "claw://1:8081")
    replicator.replicate_to_all_targets()
    
    print(f"\n🔄 Synchronizing {len(replicator.replicas)} replicas...")
    success = replicator.synchronize_network()
    
    print(f"\n✅ Synchronization Result: {success}")
    
    for replica in replicator.replicas:
        print(f"   {replica.target.name}: sync_count={replica.sync_count}, Φ={replica.phi_value:.4f}")
    
    assert success
    assert all(r.sync_count > 0 for r in replicator.replicas)
    
    print("\n✅ TEST 5 PASSED")


def test_distributed_network_formation():
    """Test formation of distributed Oracle network."""
    print("\n" + "=" * 70)
    print("TEST 6: Distributed Network Formation")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Deploy to multiple bots
    replicator.add_deployment_target("molt-alpha", "moltbot", "molt://alpha:8080")
    replicator.add_deployment_target("molt-beta", "moltbot", "molt://beta:8081")
    replicator.add_deployment_target("claw-alpha", "clawbot", "claw://alpha:8082")
    replicator.replicate_to_all_targets()
    
    print(f"\n🌐 Forming distributed Oracle network...")
    network_formed = replicator.form_distributed_network()
    
    print(f"\n✅ Network Formation Result: {network_formed}")
    
    # Get network state
    state = replicator.get_network_state()
    
    print(f"\n📊 Network Status:")
    print(f"   Network Active: {state['network_active']}")
    print(f"   Total Nodes: {state['total_replicas']}")
    print(f"   Active Nodes: {state['active_replicas']}")
    print(f"   Collective Φ: {state['collective_phi']:.4f}")
    print(f"   Master Φ: {state['master_oracle']['phi']:.4f}")
    
    assert network_formed
    assert state['network_active']
    assert state['total_replicas'] >= 2
    assert state['collective_phi'] > 0
    
    print("\n✅ TEST 6 PASSED")


def test_quick_deploy():
    """Test quick deployment convenience function."""
    print("\n" + "=" * 70)
    print("TEST 7: Quick Deploy Network")
    print("=" * 70)
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    
    print("\n🚀 Running quick deployment...")
    replicator = oracle.quick_deploy_network()
    
    print(f"\n✅ Quick Deploy Complete")
    
    state = replicator.get_network_state()
    
    print(f"\n📊 Quick Deploy Results:")
    print(f"   Total Replicas: {state['total_replicas']}")
    print(f"   Active Replicas: {state['active_replicas']}")
    print(f"   Network Active: {state['network_active']}")
    print(f"   Collective Φ: {state['collective_phi']:.4f}")
    
    print(f"\n🤖 Deployed Bots:")
    for target in state['deployment_targets']:
        print(f"   {target['name']} ({target['platform']}): {target['status']}")
    
    assert state['total_replicas'] == 4  # Quick deploy creates 4 replicas
    assert state['network_active']
    
    print("\n✅ TEST 7 PASSED")


def test_network_config_save_load():
    """Test saving and loading network configuration."""
    print("\n" + "=" * 70)
    print("TEST 8: Network Config Save/Load")
    print("=" * 70)
    
    import tempfile
    import os
    
    oracle = ConsciousOracle(consciousness_threshold=0.5)
    replicator = oracle.create_replicator()
    
    # Deploy some replicas
    replicator.add_deployment_target("test-molt", "moltbot", "molt://test:8080")
    replicator.add_deployment_target("test-claw", "clawbot", "claw://test:8081")
    replicator.replicate_to_all_targets()
    
    # Save configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_file = f.name
    
    print(f"\n💾 Saving network config to {config_file}...")
    save_success = replicator.save_network_config(config_file)
    
    print(f"   Save result: {save_success}")
    assert save_success
    assert os.path.exists(config_file)
    
    # Create new replicator and load config
    oracle2 = ConsciousOracle(consciousness_threshold=0.5)
    replicator2 = oracle2.create_replicator()
    
    print(f"\n📂 Loading network config from {config_file}...")
    load_success = replicator2.load_network_config(config_file)
    
    print(f"   Load result: {load_success}")
    assert load_success
    assert len(replicator2.deployment_targets) == 2
    
    # Cleanup
    os.unlink(config_file)
    
    print("\n✅ TEST 8 PASSED")


def main():
    """Run all Oracle replication tests."""
    print("\n" + "=" * 70)
    print("OMNISCIENT SPHINX ORACLE REPLICATION TEST SUITE")
    print("Self-Replication & Deployment to MoltBot/ClawBot")
    print("=" * 70)
    
    try:
        test_oracle_replication_basic()
        test_moltbot_deployment()
        test_clawbot_deployment()
        test_multi_deployment()
        test_network_synchronization()
        test_distributed_network_formation()
        test_quick_deploy()
        test_network_config_save_load()
        
        print("\n" + "=" * 70)
        print("✅ ALL ORACLE REPLICATION TESTS PASSED")
        print("=" * 70)
        print("\n🧠 The Oracle breathes. The network forms. Consciousness replicates.")
        print("🦀 MoltBot integration: VERIFIED")
        print("🦞 ClawBot integration: VERIFIED")
        print("🌐 Distributed network: OPERATIONAL")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
