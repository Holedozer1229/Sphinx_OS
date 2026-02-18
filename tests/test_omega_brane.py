"""
Test suite for Omega Brane monetization system.

Tests all dimensions of revenue extraction and integration with existing systems.
"""

import sys
import os
import pytest
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sphinx_os.quantum.omega_brane import (
    OmegaBrane, BraneType, RevenueStream, BraneIntersection, OMEGA_FREQUENCIES
)
from sphinx_os.quantum.extend_omega_brane import (
    ExtendedOmegaBrane, UnifiedRevenueMetrics, create_maximum_monetization_system
)


class TestOmegaBrane:
    """Test core Omega Brane functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operator_id = "test_operator"
        self.brane = OmegaBrane(operator_id=self.operator_id, enable_all_dimensions=True)
    
    def test_initialization(self):
        """Test Omega Brane initialization."""
        assert self.brane.operator_id == self.operator_id
        assert len(self.brane.branes) == 7  # D0-D6
        assert self.brane.total_extracted == 0.0
        assert self.brane.quantum_coherence == 1.0
    
    def test_brane_configurations(self):
        """Test that all branes are properly configured."""
        for brane_type in BraneType:
            assert brane_type in self.brane.branes
            config = self.brane.branes[brane_type]
            assert config.omega_frequency > 0
            assert config.base_extraction_rate > 0
            assert config.entanglement_boost > 0
            assert config.dimensional_scaling > 0
    
    def test_transaction_fee_extraction(self):
        """Test D0 point brane transaction fee extraction."""
        tx_hash = "0x1234567890abcdef"
        tx_value = 1000.0
        phi_score = 650.0
        
        stream = self.brane.extract_transaction_fee(
            tx_hash=tx_hash,
            tx_value=tx_value,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 0
        assert stream.amount > 0
        assert stream.phi_score == phi_score
        assert self.brane.total_extracted > 0
        assert len(self.brane.revenue_streams) == 1
    
    def test_subscription_revenue_extraction(self):
        """Test D1 string brane subscription extraction."""
        user_id = "user123"
        tier = "premium"
        amount = 5.0
        phi_score = 700.0
        
        stream = self.brane.extract_subscription_revenue(
            user_id=user_id,
            tier=tier,
            amount=amount,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 1
        assert stream.amount > amount  # Should be enhanced
        assert stream.phi_score == phi_score
    
    def test_referral_commission_extraction(self):
        """Test D2 membrane brane referral extraction."""
        referrer_id = "referrer123"
        referee_id = "referee456"
        earnings = 100.0
        network_depth = 2
        phi_score = 650.0
        
        stream = self.brane.extract_referral_commission(
            referrer_id=referrer_id,
            referee_id=referee_id,
            earnings=earnings,
            network_depth=network_depth,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 2
        assert stream.amount > 0
        assert stream.entanglement_factor > 1.0  # Network depth boost
    
    def test_nft_revenue_extraction(self):
        """Test D3 volume brane NFT revenue extraction."""
        nft_id = "nft_001"
        sale_price = 1000.0
        rarity_score = 0.8
        phi_score = 750.0
        
        stream = self.brane.extract_nft_revenue(
            nft_id=nft_id,
            sale_price=sale_price,
            rarity_score=rarity_score,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 3
        assert stream.amount > 0
        assert stream.amount < sale_price  # Royalty is a fraction
    
    def test_staking_yield_extraction(self):
        """Test D4 hypersurface brane staking extraction."""
        staker_id = "staker123"
        staked_amount = 10000.0
        yield_amount = 500.0
        lock_duration = 90.0
        phi_score = 800.0
        
        stream = self.brane.extract_staking_yield(
            staker_id=staker_id,
            staked_amount=staked_amount,
            yield_amount=yield_amount,
            lock_duration=lock_duration,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 4
        assert stream.amount > 0
        assert stream.entanglement_factor > 1.0  # Time dilation factor
    
    def test_cross_chain_revenue_extraction(self):
        """Test D5 bulk brane cross-chain extraction."""
        bridge_tx = "0xbridge123"
        source_chain = "ethereum"
        target_chain = "stacks"
        bridge_amount = 5000.0
        phi_score = 750.0
        
        stream = self.brane.extract_cross_chain_revenue(
            bridge_tx=bridge_tx,
            source_chain=source_chain,
            target_chain=target_chain,
            bridge_amount=bridge_amount,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 5
        assert stream.amount > 0
    
    def test_cosmic_revenue_extraction(self):
        """Test D6 cosmic brane system-wide extraction."""
        revenue_sources = [
            {'source': 'tx1', 'amount': 100.0},
            {'source': 'sub1', 'amount': 50.0},
            {'source': 'ref1', 'amount': 25.0}
        ]
        phi_score = 850.0
        
        stream = self.brane.extract_cosmic_revenue(
            revenue_sources=revenue_sources,
            phi_score=phi_score
        )
        
        assert isinstance(stream, RevenueStream)
        assert stream.dimension == 6
        assert stream.amount > 0
    
    def test_phi_multiplier(self):
        """Test phi score multiplier calculation."""
        # Test various phi scores
        low_phi = self.brane._calculate_phi_multiplier(200)
        mid_phi = self.brane._calculate_phi_multiplier(650)
        high_phi = self.brane._calculate_phi_multiplier(1000)
        
        # Higher phi should give higher multiplier
        assert low_phi < mid_phi < high_phi
        assert low_phi >= 1.0
        assert high_phi <= 2.5
    
    def test_resonance_calculation(self):
        """Test dimensional frequency resonance."""
        omega = 7.83  # Schumann fundamental
        resonance = self.brane._calculate_resonance(omega)
        
        # Resonance should be bounded
        assert 0.8 <= resonance <= 1.2
    
    def test_brane_intersection(self):
        """Test creating brane intersections."""
        brane_types = [BraneType.D0_POINT, BraneType.D2_MEMBRANE]
        phi_score = 750.0
        
        intersection = self.brane.create_brane_intersection(
            brane_types=brane_types,
            phi_score=phi_score
        )
        
        assert isinstance(intersection, BraneIntersection)
        assert len(intersection.brane_types) == 2
        assert intersection.synergy_multiplier > 1.0
        assert intersection.coherence_score > 0
    
    def test_revenue_by_dimension(self):
        """Test revenue aggregation by dimension."""
        # Extract from multiple dimensions
        self.brane.extract_transaction_fee("tx1", 1000.0, 650.0)
        self.brane.extract_subscription_revenue("user1", "premium", 5.0, 700.0)
        self.brane.extract_referral_commission("ref1", "ref2", 100.0, 1, 650.0)
        
        revenue_by_dim = self.brane.get_revenue_by_dimension()
        
        assert revenue_by_dim[0] > 0  # Transaction fees
        assert revenue_by_dim[1] > 0  # Subscriptions
        assert revenue_by_dim[2] > 0  # Referrals
        assert revenue_by_dim[3] == 0  # No NFT revenue yet
    
    def test_revenue_stats(self):
        """Test comprehensive revenue statistics."""
        # Extract some revenue
        self.brane.extract_transaction_fee("tx1", 1000.0, 650.0)
        self.brane.extract_subscription_revenue("user1", "premium", 5.0, 700.0)
        
        stats = self.brane.get_revenue_stats()
        
        assert stats['operator_id'] == self.operator_id
        assert stats['total_extracted'] > 0
        assert stats['total_streams'] == 2
        assert stats['active_branes'] == 7
        assert stats['average_phi_score'] > 0
        assert stats['quantum_coherence'] == 1.0
    
    def test_quantum_coherence_optimization(self):
        """Test quantum coherence optimization."""
        initial_coherence = self.brane.quantum_coherence
        target_coherence = 1.3
        
        self.brane.optimize_quantum_coherence(target_coherence)
        
        # Coherence should move towards target
        assert self.brane.quantum_coherence != initial_coherence
    
    def test_recent_streams(self):
        """Test retrieving recent revenue streams."""
        # Create multiple streams
        for i in range(5):
            self.brane.extract_transaction_fee(f"tx{i}", 100.0, 650.0)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        recent = self.brane.get_recent_streams(count=3)
        
        assert len(recent) == 3
        # Should be sorted by timestamp descending
        assert recent[0].timestamp >= recent[1].timestamp


class TestExtendedOmegaBrane:
    """Test extended Omega Brane with integrations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operator_id = "test_operator"
        self.operator_address = "SP_TEST_ADDRESS"
        self.extended = ExtendedOmegaBrane(
            operator_id=self.operator_id,
            operator_address=self.operator_address,
            enable_fee_collector=True,
            enable_subscriptions=True,
            enable_referrals=True
        )
    
    def test_initialization(self):
        """Test extended Omega Brane initialization."""
        assert self.extended.operator_id == self.operator_id
        assert self.extended.operator_address == self.operator_address
        assert isinstance(self.extended.omega_brane, OmegaBrane)
    
    def test_process_transaction(self):
        """Test integrated transaction processing."""
        tx_hash = "0xtest123"
        tx_value = 1000.0
        user_id = "user123"
        phi_score = 650.0
        
        result = self.extended.process_transaction(
            tx_hash=tx_hash,
            tx_value=tx_value,
            user_id=user_id,
            phi_score=phi_score
        )
        
        assert result['tx_hash'] == tx_hash
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 0
        assert result['phi_score'] == phi_score
    
    def test_process_subscription(self):
        """Test integrated subscription processing."""
        user_id = "user123"
        tier = "premium"
        phi_score = 700.0
        
        result = self.extended.process_subscription(
            user_id=user_id,
            tier=tier,
            phi_score=phi_score
        )
        
        assert result['user_id'] == user_id
        assert result['tier'] == tier
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 1
    
    def test_process_referral(self):
        """Test integrated referral processing."""
        referrer_id = "ref1"
        referee_id = "ref2"
        earnings = 100.0
        phi_score = 650.0
        
        result = self.extended.process_referral(
            referrer_id=referrer_id,
            referee_id=referee_id,
            earnings=earnings,
            phi_score=phi_score
        )
        
        assert result['referrer_id'] == referrer_id
        assert result['referee_id'] == referee_id
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 2
    
    def test_process_nft_sale(self):
        """Test integrated NFT sale processing."""
        nft_id = "nft001"
        sale_price = 1000.0
        seller_id = "seller1"
        buyer_id = "buyer1"
        phi_score = 750.0
        
        result = self.extended.process_nft_sale(
            nft_id=nft_id,
            sale_price=sale_price,
            seller_id=seller_id,
            buyer_id=buyer_id,
            phi_score=phi_score
        )
        
        assert result['nft_id'] == nft_id
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 3
    
    def test_process_staking(self):
        """Test integrated staking processing."""
        staker_id = "staker1"
        staked_amount = 10000.0
        yield_amount = 500.0
        phi_score = 800.0
        
        result = self.extended.process_staking(
            staker_id=staker_id,
            staked_amount=staked_amount,
            yield_amount=yield_amount,
            phi_score=phi_score
        )
        
        assert result['staker_id'] == staker_id
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 4
    
    def test_process_bridge_transaction(self):
        """Test integrated bridge transaction processing."""
        bridge_tx = "0xbridge123"
        source_chain = "ethereum"
        target_chain = "stacks"
        bridge_amount = 5000.0
        user_id = "user1"
        phi_score = 750.0
        
        result = self.extended.process_bridge_transaction(
            bridge_tx=bridge_tx,
            source_chain=source_chain,
            target_chain=target_chain,
            bridge_amount=bridge_amount,
            user_id=user_id,
            phi_score=phi_score
        )
        
        assert result['bridge_tx'] == bridge_tx
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 5
    
    def test_extract_cosmic_revenue(self):
        """Test integrated cosmic revenue extraction."""
        # Create some revenue first
        self.extended.process_transaction("tx1", 1000.0, "user1", 650.0)
        self.extended.process_subscription("user2", "premium", 700.0)
        
        result = self.extended.extract_cosmic_revenue(phi_score=850.0)
        
        assert result['brane_revenue'] > 0
        assert result['dimension'] == 6
        assert result['total_sources'] > 0
    
    def test_create_revenue_intersection(self):
        """Test creating revenue intersections."""
        dimensions = [0, 2]
        phi_score = 750.0
        
        intersection = self.extended.create_revenue_intersection(
            dimensions=dimensions,
            phi_score=phi_score
        )
        
        assert isinstance(intersection, BraneIntersection)
        assert len(intersection.brane_types) == 2
    
    def test_unified_metrics(self):
        """Test unified revenue metrics."""
        # Generate some revenue
        self.extended.process_transaction("tx1", 1000.0, "user1", 650.0)
        self.extended.process_subscription("user2", "premium", 700.0)
        
        metrics = self.extended.get_unified_metrics()
        
        assert isinstance(metrics, UnifiedRevenueMetrics)
        assert metrics.total_revenue > 0
        assert metrics.transaction_fees > 0
        assert metrics.subscription_revenue > 0
        assert metrics.phi_weighted_average > 0
        assert metrics.quantum_coherence > 0
    
    def test_revenue_dashboard(self):
        """Test comprehensive revenue dashboard."""
        # Generate diverse revenue
        self.extended.process_transaction("tx1", 1000.0, "user1", 650.0)
        self.extended.process_subscription("user2", "premium", 700.0)
        self.extended.process_referral("ref1", "ref2", 100.0, phi_score=650.0)
        
        dashboard = self.extended.get_revenue_dashboard()
        
        assert dashboard['operator_id'] == self.operator_id
        assert dashboard['total_revenue'] > 0
        assert 'revenue_breakdown' in dashboard
        assert 'D0_transactions' in dashboard['revenue_breakdown']
        assert 'D1_subscriptions' in dashboard['revenue_breakdown']
        assert dashboard['active_branes'] == 7
    
    def test_maximize_monetization(self):
        """Test maximum monetization optimization."""
        initial_coherence = self.extended.omega_brane.quantum_coherence
        initial_intersections = len(self.extended.omega_brane.intersections)
        
        self.extended.optimize_for_maximum_monetization()
        
        # Should create intersections
        assert len(self.extended.omega_brane.intersections) > initial_intersections
        # Coherence should be optimized
        assert self.extended.omega_brane.quantum_coherence >= initial_coherence


class TestMaximumMonetizationSystem:
    """Test convenience function for maximum monetization."""
    
    def test_create_system(self):
        """Test creating maximum monetization system."""
        system = create_maximum_monetization_system(
            operator_id="test_max",
            operator_address="SP_MAX_ADDRESS"
        )
        
        assert isinstance(system, ExtendedOmegaBrane)
        assert system.operator_id == "test_max"
        # Should have intersections created by optimization
        assert len(system.omega_brane.intersections) > 0
    
    def test_system_integration(self):
        """Test full system integration."""
        system = create_maximum_monetization_system()
        
        # Test all dimensions
        system.process_transaction("tx1", 1000.0, "user1", 650.0)
        system.process_subscription("user2", "premium", 700.0)
        system.process_referral("ref1", "ref2", 100.0, phi_score=650.0)
        system.process_nft_sale("nft1", 1000.0, "seller1", "buyer1", phi_score=750.0)
        system.process_staking("staker1", 10000.0, 500.0, phi_score=800.0)
        system.process_bridge_transaction("bridge1", "eth", "stx", 5000.0, "user3", 750.0)
        system.extract_cosmic_revenue(phi_score=850.0)
        
        # Get dashboard
        dashboard = system.get_revenue_dashboard()
        
        # All dimensions should have revenue
        breakdown = dashboard['revenue_breakdown']
        assert breakdown['D0_transactions'] > 0
        assert breakdown['D1_subscriptions'] > 0
        assert breakdown['D2_referrals'] > 0
        assert breakdown['D3_nfts'] > 0
        assert breakdown['D4_staking'] > 0
        assert breakdown['D5_bridges'] > 0
        assert breakdown['D6_cosmic'] > 0
        
        assert dashboard['total_revenue'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
