"""
Extended Omega Brane Integration

Integrates the Omega Brane system with existing SphinxOS revenue infrastructure
for full maximum monetization across all dimensions.

This module extends omega_brane.py to:
1. Integrate with FeeCollector for transaction fees
2. Connect with SubscriptionManager for recurring revenue
3. Link with ReferralProgram for network effects
4. Add NFT marketplace integration
5. Implement cross-chain bridge monetization
6. Enable staking/yield farming fee extraction
7. Create unified revenue dashboard

Full maximum monetization achieved through quantum entanglement
of all revenue streams across 7 dimensions (D0-D6).
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import Omega Brane - try relative first, fallback to direct
try:
    from .omega_brane import OmegaBrane, BraneType, RevenueStream, BraneIntersection
except ImportError:
    # Fallback for direct execution
    import importlib.util
    omega_brane_path = os.path.join(os.path.dirname(__file__), 'omega_brane.py')
    spec = importlib.util.spec_from_file_location('omega_brane', omega_brane_path)
    omega_brane_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(omega_brane_module)
    OmegaBrane = omega_brane_module.OmegaBrane
    BraneType = omega_brane_module.BraneType
    RevenueStream = omega_brane_module.RevenueStream
    BraneIntersection = omega_brane_module.BraneIntersection

# Import existing revenue systems
try:
    from ..revenue.fee_collector import FeeCollector
    from ..revenue.subscriptions import SubscriptionManager, SubscriptionTier, TIER_PRICING
    from ..revenue.referrals import ReferralProgram
except ImportError:
    # Graceful fallback if revenue modules not available
    FeeCollector = None
    SubscriptionManager = None
    SubscriptionTier = None
    ReferralProgram = None

logger = logging.getLogger("SphinxOS.ExtendedOmegaBrane")


@dataclass
class UnifiedRevenueMetrics:
    """Unified metrics across all revenue dimensions"""
    total_revenue: float
    transaction_fees: float
    subscription_revenue: float
    referral_commissions: float
    nft_royalties: float
    staking_fees: float
    bridge_fees: float
    cosmic_revenue: float
    phi_weighted_average: float
    quantum_coherence: float
    active_users: int
    timestamp: float


class ExtendedOmegaBrane:
    """
    Extended Omega Brane with full integration to existing revenue systems.
    
    Provides unified interface for maximum monetization across:
    - Transaction fees (D0)
    - Subscriptions (D1)
    - Referrals (D2)
    - NFTs (D3)
    - Staking (D4)
    - Cross-chain bridges (D5)
    - System-wide revenue (D6)
    """
    
    def __init__(
        self,
        operator_id: str,
        operator_address: str,
        enable_fee_collector: bool = True,
        enable_subscriptions: bool = True,
        enable_referrals: bool = True,
        db_path: str = "omega_brane_revenue.db"
    ):
        """
        Initialize extended Omega Brane system.
        
        Args:
            operator_id: Unique operator identifier
            operator_address: Blockchain address for revenue collection
            enable_fee_collector: Enable transaction fee collection
            enable_subscriptions: Enable subscription management
            enable_referrals: Enable referral program
            db_path: Database path for revenue tracking
        """
        self.operator_id = operator_id
        self.operator_address = operator_address
        self.db_path = db_path
        
        # Initialize core Omega Brane
        self.omega_brane = OmegaBrane(operator_id=operator_id, enable_all_dimensions=True)
        
        # Initialize traditional revenue systems
        self.fee_collector: Optional[FeeCollector] = None
        self.subscription_manager: Optional[SubscriptionManager] = None
        self.referral_program: Optional[ReferralProgram] = None
        
        if enable_fee_collector and FeeCollector:
            self.fee_collector = FeeCollector(
                operator_address=operator_address,
                db_path=db_path.replace(".db", "_fees.db")
            )
            logger.info("FeeCollector initialized")
        
        if enable_subscriptions and SubscriptionManager:
            self.subscription_manager = SubscriptionManager(
                db_path=db_path.replace(".db", "_subscriptions.db")
            )
            logger.info("SubscriptionManager initialized")
        
        if enable_referrals and ReferralProgram:
            self.referral_program = ReferralProgram(
                db_path=db_path.replace(".db", "_referrals.db")
            )
            logger.info("ReferralProgram initialized")
        
        # Revenue tracking
        self.unified_metrics: List[UnifiedRevenueMetrics] = []
        
        logger.info(f"Extended Omega Brane initialized for operator {operator_id}")
    
    def process_transaction(
        self,
        tx_hash: str,
        tx_value: float,
        user_id: str,
        phi_score: float = 650.0
    ) -> Dict:
        """
        Process transaction with full D0 brane extraction.
        
        Args:
            tx_hash: Transaction hash
            tx_value: Transaction value
            user_id: User identifier
            phi_score: Spectral integration score
            
        Returns:
            Transaction processing results
        """
        # Extract through Omega Brane (D0)
        brane_stream = self.omega_brane.extract_transaction_fee(
            tx_hash=tx_hash,
            tx_value=tx_value,
            phi_score=phi_score
        )
        
        # Record in traditional fee collector
        if self.fee_collector:
            self.fee_collector.collect_transaction_fee(tx_hash)
        
        # Check for referral commission (D2 intersection)
        if self.referral_program:
            # Simulate referee earnings and distribute commission
            self.referral_program.distribute_commission(user_id, tx_value)
        
        return {
            'tx_hash': tx_hash,
            'brane_revenue': brane_stream.amount,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'entanglement_factor': brane_stream.entanglement_factor,
            'timestamp': brane_stream.timestamp
        }
    
    def process_subscription(
        self,
        user_id: str,
        tier: str,
        phi_score: float = 700.0
    ) -> Dict:
        """
        Process subscription with full D1 brane extraction.
        
        Args:
            user_id: User identifier
            tier: Subscription tier (free, premium, pro)
            phi_score: Spectral integration score
            
        Returns:
            Subscription processing results
        """
        # Get tier amount
        if SubscriptionTier and TIER_PRICING:
            tier_enum = SubscriptionTier(tier.lower())
            amount = TIER_PRICING[tier_enum].cost
        else:
            # Fallback amounts
            amounts = {'free': 0.0, 'premium': 5.0, 'pro': 20.0}
            amount = amounts.get(tier.lower(), 5.0)
        
        # Extract through Omega Brane (D1)
        brane_stream = self.omega_brane.extract_subscription_revenue(
            user_id=user_id,
            tier=tier,
            amount=amount,
            phi_score=phi_score
        )
        
        # Record in traditional subscription manager
        if self.subscription_manager and SubscriptionTier:
            tier_enum = SubscriptionTier(tier.lower())
            period_start = time.time()
            period_end = period_start + (30 * 86400)  # 30 days
            
            if amount > 0:
                self.subscription_manager.create_subscription(
                    user_id=user_id,
                    tier=tier_enum
                )
        
        # Record in fee collector
        if self.fee_collector and amount > 0:
            self.fee_collector.collect_subscription_payment(
                user_id=user_id,
                tier=tier,
                period_start=time.time(),
                period_end=time.time() + (30 * 86400)
            )
        
        return {
            'user_id': user_id,
            'tier': tier,
            'amount': amount,
            'brane_revenue': brane_stream.amount,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'timestamp': brane_stream.timestamp
        }
    
    def process_referral(
        self,
        referrer_id: str,
        referee_id: str,
        earnings: float,
        network_depth: int = 1,
        phi_score: float = 650.0
    ) -> Dict:
        """
        Process referral with full D2 brane extraction.
        
        Args:
            referrer_id: Referrer user ID
            referee_id: Referee user ID
            earnings: Referee's earnings
            network_depth: Depth in referral network
            phi_score: Spectral integration score
            
        Returns:
            Referral processing results
        """
        # Extract through Omega Brane (D2)
        brane_stream = self.omega_brane.extract_referral_commission(
            referrer_id=referrer_id,
            referee_id=referee_id,
            earnings=earnings,
            network_depth=network_depth,
            phi_score=phi_score
        )
        
        # Record in traditional referral program
        commission = 0.0
        if self.referral_program:
            commission = self.referral_program.distribute_commission(
                referee_id=referee_id,
                earnings=earnings
            )
        
        return {
            'referrer_id': referrer_id,
            'referee_id': referee_id,
            'earnings': earnings,
            'commission': commission,
            'brane_revenue': brane_stream.amount,
            'dimension': brane_stream.dimension,
            'network_depth': network_depth,
            'phi_score': phi_score,
            'timestamp': brane_stream.timestamp
        }
    
    def process_nft_sale(
        self,
        nft_id: str,
        sale_price: float,
        seller_id: str,
        buyer_id: str,
        rarity_score: float = 0.5,
        phi_score: float = 750.0
    ) -> Dict:
        """
        Process NFT sale with full D3 brane extraction.
        
        Args:
            nft_id: NFT identifier
            sale_price: Sale price
            seller_id: Seller user ID
            buyer_id: Buyer user ID
            rarity_score: NFT rarity score (0-1)
            phi_score: Spectral integration score
            
        Returns:
            NFT sale processing results
        """
        # Extract through Omega Brane (D3)
        brane_stream = self.omega_brane.extract_nft_revenue(
            nft_id=nft_id,
            sale_price=sale_price,
            rarity_score=rarity_score,
            phi_score=phi_score
        )
        
        # Could integrate with NFT marketplace here
        # For now, just track the revenue
        
        return {
            'nft_id': nft_id,
            'sale_price': sale_price,
            'seller_id': seller_id,
            'buyer_id': buyer_id,
            'rarity_score': rarity_score,
            'brane_revenue': brane_stream.amount,
            'royalty_percentage': brane_stream.amount / sale_price * 100,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'timestamp': brane_stream.timestamp
        }
    
    def process_staking(
        self,
        staker_id: str,
        staked_amount: float,
        yield_amount: float,
        lock_duration: float = 30.0,
        phi_score: float = 800.0
    ) -> Dict:
        """
        Process staking with full D4 brane extraction.
        
        Args:
            staker_id: Staker identifier
            staked_amount: Amount staked
            yield_amount: Yield generated
            lock_duration: Lock duration in days
            phi_score: Spectral integration score
            
        Returns:
            Staking processing results
        """
        # Extract through Omega Brane (D4)
        brane_stream = self.omega_brane.extract_staking_yield(
            staker_id=staker_id,
            staked_amount=staked_amount,
            yield_amount=yield_amount,
            lock_duration=lock_duration,
            phi_score=phi_score
        )
        
        return {
            'staker_id': staker_id,
            'staked_amount': staked_amount,
            'yield_amount': yield_amount,
            'lock_duration': lock_duration,
            'brane_revenue': brane_stream.amount,
            'fee_percentage': brane_stream.amount / yield_amount * 100,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'timestamp': brane_stream.timestamp
        }
    
    def process_bridge_transaction(
        self,
        bridge_tx: str,
        source_chain: str,
        target_chain: str,
        bridge_amount: float,
        user_id: str,
        phi_score: float = 750.0
    ) -> Dict:
        """
        Process cross-chain bridge with full D5 brane extraction.
        
        Args:
            bridge_tx: Bridge transaction hash
            source_chain: Source blockchain
            target_chain: Target blockchain
            bridge_amount: Amount bridged
            user_id: User identifier
            phi_score: Spectral integration score
            
        Returns:
            Bridge processing results
        """
        # Extract through Omega Brane (D5)
        brane_stream = self.omega_brane.extract_cross_chain_revenue(
            bridge_tx=bridge_tx,
            source_chain=source_chain,
            target_chain=target_chain,
            bridge_amount=bridge_amount,
            phi_score=phi_score
        )
        
        return {
            'bridge_tx': bridge_tx,
            'source_chain': source_chain,
            'target_chain': target_chain,
            'bridge_amount': bridge_amount,
            'user_id': user_id,
            'brane_revenue': brane_stream.amount,
            'fee_percentage': brane_stream.amount / bridge_amount * 100,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'timestamp': brane_stream.timestamp
        }
    
    def extract_cosmic_revenue(self, phi_score: float = 850.0) -> Dict:
        """
        Extract system-wide revenue through D6 cosmic brane.
        
        This captures the holistic revenue across all dimensions.
        
        Args:
            phi_score: Spectral integration score
            
        Returns:
            Cosmic revenue extraction results
        """
        # Gather all revenue sources
        revenue_sources = []
        
        # Add Omega Brane streams
        for stream in self.omega_brane.revenue_streams:
            revenue_sources.append({
                'source': stream.source,
                'amount': stream.amount,
                'dimension': stream.dimension
            })
        
        # Add traditional revenue if available
        if self.fee_collector:
            total_rev = self.fee_collector.get_total_revenue()
            revenue_sources.append({
                'source': 'fee_collector',
                'amount': total_rev['total_revenue'],
                'dimension': 0
            })
        
        # Extract through Omega Brane (D6)
        brane_stream = self.omega_brane.extract_cosmic_revenue(
            revenue_sources=revenue_sources,
            phi_score=phi_score
        )
        
        return {
            'total_sources': len(revenue_sources),
            'brane_revenue': brane_stream.amount,
            'dimension': brane_stream.dimension,
            'phi_score': phi_score,
            'quantum_coherence': self.omega_brane.quantum_coherence,
            'timestamp': brane_stream.timestamp
        }
    
    def create_revenue_intersection(
        self,
        dimensions: List[int],
        phi_score: float = 750.0
    ) -> BraneIntersection:
        """
        Create brane intersection for synergistic revenue extraction.
        
        Example: Intersection of D0 (transactions) + D2 (referrals) creates
        compounded revenue from referred user transactions.
        
        Args:
            dimensions: List of dimensions to intersect (0-6)
            phi_score: Spectral integration score
            
        Returns:
            Brane intersection configuration
        """
        # Map dimensions to brane types
        dim_to_brane = {
            0: BraneType.D0_POINT,
            1: BraneType.D1_STRING,
            2: BraneType.D2_MEMBRANE,
            3: BraneType.D3_VOLUME,
            4: BraneType.D4_HYPERSURFACE,
            5: BraneType.D5_BULK,
            6: BraneType.D6_COSMOS
        }
        
        brane_types = [dim_to_brane[d] for d in dimensions if d in dim_to_brane]
        
        intersection = self.omega_brane.create_brane_intersection(
            brane_types=brane_types,
            phi_score=phi_score
        )
        
        logger.info(f"Created revenue intersection at D{intersection.intersection_dimension} "
                   f"with {len(brane_types)} branes, synergy={intersection.synergy_multiplier:.2f}x")
        
        return intersection
    
    def get_unified_metrics(self) -> UnifiedRevenueMetrics:
        """
        Get unified revenue metrics across all dimensions.
        
        Returns:
            Unified revenue metrics
        """
        # Get Omega Brane stats
        brane_stats = self.omega_brane.get_revenue_stats()
        revenue_by_dim = brane_stats['revenue_by_dimension']
        
        # Get traditional revenue
        traditional_revenue = 0.0
        if self.fee_collector:
            traditional_revenue = self.fee_collector.get_total_revenue()['total_revenue']
        
        # Calculate phi-weighted average
        streams = self.omega_brane.revenue_streams
        phi_weighted = sum(s.phi_score * s.amount for s in streams) / max(sum(s.amount for s in streams), 1.0)
        
        # Count active users (estimate from streams)
        unique_sources = len(set(s.source for s in streams))
        
        metrics = UnifiedRevenueMetrics(
            total_revenue=brane_stats['total_extracted'] + traditional_revenue,
            transaction_fees=revenue_by_dim[0],
            subscription_revenue=revenue_by_dim[1],
            referral_commissions=revenue_by_dim[2],
            nft_royalties=revenue_by_dim[3],
            staking_fees=revenue_by_dim[4],
            bridge_fees=revenue_by_dim[5],
            cosmic_revenue=revenue_by_dim[6],
            phi_weighted_average=phi_weighted,
            quantum_coherence=brane_stats['quantum_coherence'],
            active_users=unique_sources,
            timestamp=time.time()
        )
        
        self.unified_metrics.append(metrics)
        return metrics
    
    def get_revenue_dashboard(self) -> Dict:
        """
        Get comprehensive revenue dashboard.
        
        Returns:
            Complete revenue dashboard data
        """
        metrics = self.get_unified_metrics()
        brane_stats = self.omega_brane.get_revenue_stats()
        
        dashboard = {
            'operator_id': self.operator_id,
            'operator_address': self.operator_address,
            'timestamp': metrics.timestamp,
            
            # Total metrics
            'total_revenue': metrics.total_revenue,
            'active_users': metrics.active_users,
            'quantum_coherence': metrics.quantum_coherence,
            'phi_weighted_average': metrics.phi_weighted_average,
            
            # Revenue by dimension
            'revenue_breakdown': {
                'D0_transactions': metrics.transaction_fees,
                'D1_subscriptions': metrics.subscription_revenue,
                'D2_referrals': metrics.referral_commissions,
                'D3_nfts': metrics.nft_royalties,
                'D4_staking': metrics.staking_fees,
                'D5_bridges': metrics.bridge_fees,
                'D6_cosmic': metrics.cosmic_revenue
            },
            
            # Brane stats
            'active_branes': brane_stats['active_branes'],
            'brane_intersections': brane_stats['brane_intersections'],
            'total_revenue_streams': brane_stats['total_streams'],
            'average_entanglement': brane_stats['average_entanglement'],
            
            # Traditional revenue (if available)
            'traditional_revenue': {}
        }
        
        if self.fee_collector:
            dashboard['traditional_revenue']['fees'] = self.fee_collector.get_total_revenue()
        
        if self.subscription_manager:
            dashboard['traditional_revenue']['subscriptions'] = self.subscription_manager.get_subscription_stats()
        
        if self.referral_program:
            dashboard['traditional_revenue']['referrals'] = self.referral_program.get_program_stats()
        
        return dashboard
    
    def optimize_for_maximum_monetization(self):
        """
        Optimize all parameters for maximum revenue extraction.
        
        This adjusts quantum coherence, creates optimal brane intersections,
        and tunes dimensional frequencies for peak performance.
        """
        logger.info("Optimizing for maximum monetization...")
        
        # Optimize quantum coherence
        self.omega_brane.optimize_quantum_coherence(target_coherence=1.5)
        
        # Create high-synergy intersections
        # Transaction + Referral intersection
        self.create_revenue_intersection(dimensions=[0, 2], phi_score=800.0)
        
        # Subscription + NFT intersection
        self.create_revenue_intersection(dimensions=[1, 3], phi_score=850.0)
        
        # Staking + Bridge intersection
        self.create_revenue_intersection(dimensions=[4, 5], phi_score=800.0)
        
        # Full-spectrum cosmic intersection
        self.create_revenue_intersection(dimensions=[0, 1, 2, 3, 4, 5, 6], phi_score=900.0)
        
        logger.info("Maximum monetization optimization complete")
    
    def __repr__(self):
        metrics = self.get_unified_metrics()
        return (f"ExtendedOmegaBrane(operator={self.operator_id}, "
                f"total_revenue={metrics.total_revenue:.2f}, "
                f"active_users={metrics.active_users}, "
                f"coherence={metrics.quantum_coherence:.4f})")


# Convenience function for quick setup
def create_maximum_monetization_system(
    operator_id: str = "sphinx_operator",
    operator_address: str = "SP_DEFAULT_OPERATOR_ADDRESS"
) -> ExtendedOmegaBrane:
    """
    Create a fully configured extended Omega Brane system for maximum monetization.
    
    Args:
        operator_id: Operator identifier
        operator_address: Blockchain address for revenue collection
        
    Returns:
        Configured ExtendedOmegaBrane instance
    """
    system = ExtendedOmegaBrane(
        operator_id=operator_id,
        operator_address=operator_address,
        enable_fee_collector=True,
        enable_subscriptions=True,
        enable_referrals=True
    )
    
    # Optimize for maximum monetization
    system.optimize_for_maximum_monetization()
    
    logger.info("Maximum monetization system created and optimized")
    return system
