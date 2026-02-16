"""
Fee Collector - Collect transaction fees (revenue stream!)
"""

import sqlite3
import time
from typing import Dict, List
from pathlib import Path


class FeeCollector:
    """
    Collect transaction fees - YOUR revenue stream!
    
    Fee Structure:
    - Transaction fee: 0.001 SPHINX per tx
    - Withdrawal fee: 0.01 SPHINX per withdrawal
    - Premium mining: $5.00 USD per month
    - Node hosting: $10.00 USD per month
    """
    
    FEE_STRUCTURE = {
        "transaction_fee": 0.001,  # SPHINX per tx
        "withdrawal_fee": 0.01,    # SPHINX per withdrawal
        "premium_mining": 5.00,    # USD per month
        "node_hosting": 10.00      # USD per month
    }
    
    def __init__(self, operator_address: str, db_path: str = "revenue.db"):
        """
        Initialize fee collector
        
        Args:
            operator_address: Address to collect fees to
            db_path: Path to revenue database
        """
        self.operator_address = operator_address
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize revenue tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Transaction fees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transaction_fees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash TEXT NOT NULL,
                fee_amount REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        
        # Withdrawal fees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS withdrawal_fees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT NOT NULL,
                fee_amount REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        
        # Subscription revenue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscription_revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL
            )
        ''')
        
        # Node hosting revenue table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hosting_revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL
            )
        ''')
        
        # Total revenue summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS revenue_summary (
                date TEXT PRIMARY KEY,
                transaction_fees REAL DEFAULT 0.0,
                withdrawal_fees REAL DEFAULT 0.0,
                subscription_revenue REAL DEFAULT 0.0,
                hosting_revenue REAL DEFAULT 0.0,
                total_revenue REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_transaction_fee(self, tx_hash: str):
        """
        Collect fee from a transaction
        
        Args:
            tx_hash: Transaction hash
        """
        fee_amount = self.FEE_STRUCTURE["transaction_fee"]
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO transaction_fees (tx_hash, fee_amount, timestamp) VALUES (?, ?, ?)',
            (tx_hash, fee_amount, timestamp)
        )
        
        conn.commit()
        conn.close()
        
        # Update daily summary in a separate connection
        self._update_daily_summary('transaction_fees', fee_amount)
    
    def collect_withdrawal_fee(self, address: str):
        """
        Collect fee from a withdrawal
        
        Args:
            address: Withdrawing address
        """
        fee_amount = self.FEE_STRUCTURE["withdrawal_fee"]
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO withdrawal_fees (address, fee_amount, timestamp) VALUES (?, ?, ?)',
            (address, fee_amount, timestamp)
        )
        
        conn.commit()
        conn.close()
        
        # Update daily summary in a separate connection
        self._update_daily_summary('withdrawal_fees', fee_amount)
    
    def collect_subscription_payment(
        self,
        user_id: str,
        tier: str,
        period_start: float,
        period_end: float
    ):
        """
        Record subscription payment
        
        Args:
            user_id: User identifier
            tier: Subscription tier
            period_start: Subscription period start timestamp
            period_end: Subscription period end timestamp
        """
        amount = self.FEE_STRUCTURE["premium_mining"]
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO subscription_revenue 
            (user_id, tier, amount, timestamp, period_start, period_end)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, tier, amount, timestamp, period_start, period_end))
        
        conn.commit()
        conn.close()
        
        # Update daily summary in a separate connection
        self._update_daily_summary('subscription_revenue', amount)
    
    def collect_hosting_payment(
        self,
        user_id: str,
        period_start: float,
        period_end: float
    ):
        """
        Record node hosting payment
        
        Args:
            user_id: User identifier
            period_start: Hosting period start timestamp
            period_end: Hosting period end timestamp
        """
        amount = self.FEE_STRUCTURE["node_hosting"]
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO hosting_revenue 
            (user_id, amount, timestamp, period_start, period_end)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, amount, timestamp, period_start, period_end))
        
        conn.commit()
        conn.close()
        
        # Update daily summary in a separate connection
        self._update_daily_summary('hosting_revenue', amount)
    
    def _update_daily_summary(self, category: str, amount: float):
        """Update daily revenue summary"""
        from datetime import date
        today = str(date.today())
        
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        
        # Use proper SQL parameter substitution to avoid injection
        cursor.execute('''
            INSERT INTO revenue_summary (date, transaction_fees, withdrawal_fees, 
                                        subscription_revenue, hosting_revenue, total_revenue)
            VALUES (?, 0, 0, 0, 0, 0)
            ON CONFLICT(date) DO NOTHING
        ''', (today,))
        
        # Update the specific category
        if category == 'transaction_fees':
            cursor.execute('''
                UPDATE revenue_summary
                SET transaction_fees = transaction_fees + ?,
                    total_revenue = total_revenue + ?
                WHERE date = ?
            ''', (amount, amount, today))
        elif category == 'withdrawal_fees':
            cursor.execute('''
                UPDATE revenue_summary
                SET withdrawal_fees = withdrawal_fees + ?,
                    total_revenue = total_revenue + ?
                WHERE date = ?
            ''', (amount, amount, today))
        elif category == 'subscription_revenue':
            cursor.execute('''
                UPDATE revenue_summary
                SET subscription_revenue = subscription_revenue + ?,
                    total_revenue = total_revenue + ?
                WHERE date = ?
            ''', (amount, amount, today))
        elif category == 'hosting_revenue':
            cursor.execute('''
                UPDATE revenue_summary
                SET hosting_revenue = hosting_revenue + ?,
                    total_revenue = total_revenue + ?
                WHERE date = ?
            ''', (amount, amount, today))
        
        conn.commit()
        conn.close()
    
    def get_daily_revenue(self, date: str = None) -> Dict:
        """
        Get revenue for a specific date
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Revenue breakdown dictionary
        """
        if date is None:
            from datetime import date as dt
            date = str(dt.today())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT transaction_fees, withdrawal_fees, 
                   subscription_revenue, hosting_revenue, total_revenue
            FROM revenue_summary
            WHERE date = ?
        ''', (date,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'date': date,
                'transaction_fees': row[0],
                'withdrawal_fees': row[1],
                'subscription_revenue': row[2],
                'hosting_revenue': row[3],
                'total_revenue': row[4]
            }
        
        return {
            'date': date,
            'transaction_fees': 0.0,
            'withdrawal_fees': 0.0,
            'subscription_revenue': 0.0,
            'hosting_revenue': 0.0,
            'total_revenue': 0.0
        }
    
    def get_total_revenue(self) -> Dict:
        """
        Get all-time total revenue
        
        Returns:
            Total revenue breakdown
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                SUM(transaction_fees),
                SUM(withdrawal_fees),
                SUM(subscription_revenue),
                SUM(hosting_revenue),
                SUM(total_revenue)
            FROM revenue_summary
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'transaction_fees': row[0] or 0.0,
            'withdrawal_fees': row[1] or 0.0,
            'subscription_revenue': row[2] or 0.0,
            'hosting_revenue': row[3] or 0.0,
            'total_revenue': row[4] or 0.0
        }
    
    def get_revenue_history(self, days: int = 30) -> List[Dict]:
        """
        Get revenue history
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of daily revenue records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, transaction_fees, withdrawal_fees,
                   subscription_revenue, hosting_revenue, total_revenue
            FROM revenue_summary
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': row[0],
                'transaction_fees': row[1],
                'withdrawal_fees': row[2],
                'subscription_revenue': row[3],
                'hosting_revenue': row[4],
                'total_revenue': row[5]
            }
            for row in rows
        ]
    
    def get_revenue_stats(self) -> Dict:
        """
        Get comprehensive revenue statistics
        
        Returns:
            Revenue statistics
        """
        total = self.get_total_revenue()
        today = self.get_daily_revenue()
        
        # Calculate growth (simplified - comparing to yesterday)
        from datetime import date, timedelta
        yesterday = str(date.today() - timedelta(days=1))
        yesterday_revenue = self.get_daily_revenue(yesterday)
        
        growth = 0.0
        if yesterday_revenue['total_revenue'] > 0:
            growth = (
                (today['total_revenue'] - yesterday_revenue['total_revenue']) /
                yesterday_revenue['total_revenue'] * 100
            )
        
        return {
            'today': today,
            'total': total,
            'growth_percent': growth,
            'operator_address': self.operator_address
        }
    
    def __repr__(self):
        return f"FeeCollector(operator={self.operator_address})"
