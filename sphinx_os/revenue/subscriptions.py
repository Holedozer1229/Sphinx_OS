"""
Subscription Management - Premium mining tiers
"""

import sqlite3
import time
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass


class SubscriptionTier(Enum):
    """Subscription tiers"""
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


@dataclass
class TierPricing:
    """Pricing configuration for each tier"""
    name: str
    cost: float  # USD per month
    hashrate: str
    daily_limit: str
    features: List[str]


# Tier pricing configurations
TIER_PRICING = {
    SubscriptionTier.FREE: TierPricing(
        name="Free",
        cost=0.0,
        hashrate="10 MH/s",
        daily_limit="1000 SPHINX",
        features=[
            "Browser mining",
            "Instant payouts",
            "Basic support"
        ]
    ),
    SubscriptionTier.PREMIUM: TierPricing(
        name="Premium",
        cost=5.0,
        hashrate="100 MH/s",
        daily_limit="10000 SPHINX",
        features=[
            "10x faster mining",
            "Priority payouts",
            "Email support",
            "Mining analytics"
        ]
    ),
    SubscriptionTier.PRO: TierPricing(
        name="Pro",
        cost=20.0,
        hashrate="1000 MH/s",
        daily_limit="unlimited",
        features=[
            "100x faster mining",
            "Unlimited mining",
            "Priority support",
            "Advanced analytics",
            "API access",
            "Custom branding"
        ]
    )
}


class SubscriptionManager:
    """
    Manage user subscriptions for premium mining
    """
    
    def __init__(self, db_path: str = "subscriptions.db"):
        """
        Initialize subscription manager
        
        Args:
            db_path: Path to subscriptions database
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize subscriptions database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Subscriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                status TEXT NOT NULL,
                start_date REAL NOT NULL,
                end_date REAL NOT NULL,
                stripe_subscription_id TEXT,
                stripe_customer_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        ''')
        
        # Payment history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                status TEXT NOT NULL,
                stripe_payment_id TEXT,
                timestamp REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None
    ) -> Dict:
        """
        Create a new subscription
        
        Args:
            user_id: User identifier
            tier: Subscription tier
            stripe_subscription_id: Stripe subscription ID (if paid)
            stripe_customer_id: Stripe customer ID (if paid)
            
        Returns:
            Subscription details
        """
        now = time.time()
        end_date = now + (30 * 86400)  # 30 days from now
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO subscriptions
            (user_id, tier, status, start_date, end_date,
             stripe_subscription_id, stripe_customer_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            tier.value,
            'active',
            now,
            end_date,
            stripe_subscription_id,
            stripe_customer_id,
            now,
            now
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'user_id': user_id,
            'tier': tier.value,
            'status': 'active',
            'start_date': now,
            'end_date': end_date
        }
    
    def get_subscription(self, user_id: str) -> Optional[Dict]:
        """
        Get user's subscription
        
        Args:
            user_id: User identifier
            
        Returns:
            Subscription details or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT tier, status, start_date, end_date,
                   stripe_subscription_id, stripe_customer_id
            FROM subscriptions
            WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Check if subscription is expired
        if row[3] < time.time() and row[1] == 'active':
            self.cancel_subscription(user_id, reason='expired')
            return None
        
        return {
            'user_id': user_id,
            'tier': row[0],
            'status': row[1],
            'start_date': row[2],
            'end_date': row[3],
            'stripe_subscription_id': row[4],
            'stripe_customer_id': row[5]
        }
    
    def upgrade_subscription(self, user_id: str, new_tier: SubscriptionTier) -> Dict:
        """
        Upgrade user's subscription
        
        Args:
            user_id: User identifier
            new_tier: New subscription tier
            
        Returns:
            Updated subscription details
        """
        current_sub = self.get_subscription(user_id)
        
        if not current_sub:
            # Create new subscription
            return self.create_subscription(user_id, new_tier)
        
        # Update existing subscription
        now = time.time()
        end_date = now + (30 * 86400)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE subscriptions
            SET tier = ?, start_date = ?, end_date = ?, updated_at = ?
            WHERE user_id = ?
        ''', (new_tier.value, now, end_date, now, user_id))
        
        conn.commit()
        conn.close()
        
        return self.get_subscription(user_id)
    
    def cancel_subscription(self, user_id: str, reason: str = 'user_cancelled'):
        """
        Cancel user's subscription
        
        Args:
            user_id: User identifier
            reason: Cancellation reason
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE subscriptions
            SET status = ?, updated_at = ?
            WHERE user_id = ?
        ''', ('cancelled', time.time(), user_id))
        
        conn.commit()
        conn.close()
    
    def record_payment(
        self,
        user_id: str,
        amount: float,
        status: str = 'completed',
        stripe_payment_id: Optional[str] = None
    ):
        """
        Record a payment
        
        Args:
            user_id: User identifier
            amount: Payment amount
            status: Payment status
            stripe_payment_id: Stripe payment ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO payment_history
            (user_id, amount, status, stripe_payment_id, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, amount, status, stripe_payment_id, time.time()))
        
        conn.commit()
        conn.close()
    
    def get_active_subscriptions(self) -> List[Dict]:
        """
        Get all active subscriptions
        
        Returns:
            List of active subscriptions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, tier, start_date, end_date
            FROM subscriptions
            WHERE status = 'active' AND end_date > ?
        ''', (time.time(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'user_id': row[0],
                'tier': row[1],
                'start_date': row[2],
                'end_date': row[3]
            }
            for row in rows
        ]
    
    def get_subscription_stats(self) -> Dict:
        """
        Get subscription statistics
        
        Returns:
            Subscription statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count active subscriptions by tier
        cursor.execute('''
            SELECT tier, COUNT(*) as count
            FROM subscriptions
            WHERE status = 'active' AND end_date > ?
            GROUP BY tier
        ''', (time.time(),))
        
        tier_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Calculate MRR (Monthly Recurring Revenue)
        mrr = (
            tier_counts.get('premium', 0) * TIER_PRICING[SubscriptionTier.PREMIUM].cost +
            tier_counts.get('pro', 0) * TIER_PRICING[SubscriptionTier.PRO].cost
        )
        
        # Total active subscriptions
        total_active = sum(tier_counts.values())
        
        conn.close()
        
        return {
            'active_subscriptions': total_active,
            'free_users': tier_counts.get('free', 0),
            'premium_users': tier_counts.get('premium', 0),
            'pro_users': tier_counts.get('pro', 0),
            'monthly_revenue': mrr,
            'tier_counts': tier_counts
        }
    
    def get_tier_info(self, tier: SubscriptionTier) -> Dict:
        """
        Get pricing information for a tier
        
        Args:
            tier: Subscription tier
            
        Returns:
            Tier information
        """
        config = TIER_PRICING[tier]
        return {
            'tier': tier.value,
            'name': config.name,
            'cost': config.cost,
            'hashrate': config.hashrate,
            'daily_limit': config.daily_limit,
            'features': config.features
        }
    
    def __repr__(self):
        stats = self.get_subscription_stats()
        return f"SubscriptionManager({stats['active_subscriptions']} active)"
