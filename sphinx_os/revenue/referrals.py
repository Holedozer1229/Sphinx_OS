"""
Referral Program - Users get 5% of their referrals' earnings
"""

import sqlite3
import time
from typing import Dict, Optional, List


class ReferralProgram:
    """
    Users get 5% of their referrals' earnings
    YOU get 95%
    Creates viral growth!
    """
    
    COMMISSION_RATE = 0.05  # 5% to referrer
    
    def __init__(self, db_path: str = "referrals.db"):
        """
        Initialize referral program
        
        Args:
            db_path: Path to referrals database
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize referrals database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Referrals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS referrals (
                referee_id TEXT PRIMARY KEY,
                referrer_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                total_earned REAL DEFAULT 0.0,
                commission_paid REAL DEFAULT 0.0
            )
        ''')
        
        # Referral codes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS referral_codes (
                user_id TEXT PRIMARY KEY,
                referral_code TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                total_referrals INTEGER DEFAULT 0,
                total_commission REAL DEFAULT 0.0
            )
        ''')
        
        # Commission history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS commission_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                referrer_id TEXT NOT NULL,
                referee_id TEXT NOT NULL,
                amount REAL NOT NULL,
                earnings_source REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_referral_code(self, user_id: str) -> str:
        """
        Generate a referral code for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Referral code
        """
        import hashlib
        
        # Generate unique referral code
        code_data = f"{user_id}{time.time()}"
        referral_code = hashlib.sha256(code_data.encode()).hexdigest()[:8].upper()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO referral_codes
            (user_id, referral_code, created_at)
            VALUES (?, ?, ?)
        ''', (user_id, referral_code, time.time()))
        
        conn.commit()
        conn.close()
        
        return referral_code
    
    def get_referral_code(self, user_id: str) -> Optional[str]:
        """
        Get user's referral code
        
        Args:
            user_id: User identifier
            
        Returns:
            Referral code or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT referral_code FROM referral_codes WHERE user_id = ?',
            (user_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row[0]
        
        # Generate code if doesn't exist
        return self.generate_referral_code(user_id)
    
    def get_referrer_by_code(self, referral_code: str) -> Optional[str]:
        """
        Get referrer ID by referral code
        
        Args:
            referral_code: Referral code
            
        Returns:
            Referrer user ID or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT user_id FROM referral_codes WHERE referral_code = ?',
            (referral_code,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None
    
    def track_referral(self, referrer_id: str, referee_id: str):
        """
        Track a new referral
        
        Args:
            referrer_id: Referrer user ID
            referee_id: Referee (new user) ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add referral
        cursor.execute('''
            INSERT OR REPLACE INTO referrals
            (referee_id, referrer_id, created_at)
            VALUES (?, ?, ?)
        ''', (referee_id, referrer_id, time.time()))
        
        # Update referrer's total referrals
        cursor.execute('''
            UPDATE referral_codes
            SET total_referrals = total_referrals + 1
            WHERE user_id = ?
        ''', (referrer_id,))
        
        conn.commit()
        conn.close()
    
    def distribute_commission(self, referee_id: str, earnings: float):
        """
        Distribute commission to referrer based on referee's earnings
        
        Args:
            referee_id: Referee user ID
            earnings: Earnings amount
            
        Returns:
            Commission amount distributed
        """
        # Get referrer
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT referrer_id FROM referrals WHERE referee_id = ?',
            (referee_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return 0.0
        
        referrer_id = row[0]
        
        # Calculate commission
        commission = earnings * self.COMMISSION_RATE
        
        # Update referral record
        cursor.execute('''
            UPDATE referrals
            SET total_earned = total_earned + ?,
                commission_paid = commission_paid + ?
            WHERE referee_id = ?
        ''', (earnings, commission, referee_id))
        
        # Update referrer's total commission
        cursor.execute('''
            UPDATE referral_codes
            SET total_commission = total_commission + ?
            WHERE user_id = ?
        ''', (commission, referrer_id))
        
        # Record commission history
        cursor.execute('''
            INSERT INTO commission_history
            (referrer_id, referee_id, amount, earnings_source, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (referrer_id, referee_id, commission, earnings, time.time()))
        
        conn.commit()
        conn.close()
        
        return commission
    
    def get_referral_stats(self, user_id: str) -> Dict:
        """
        Get referral statistics for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Referral statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get referral code stats
        cursor.execute('''
            SELECT referral_code, total_referrals, total_commission
            FROM referral_codes
            WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {
                'referral_code': None,
                'total_referrals': 0,
                'total_commission': 0.0,
                'active_referrals': 0
            }
        
        referral_code, total_referrals, total_commission = row
        
        # Count active referrals (referred users who are still active)
        cursor.execute('''
            SELECT COUNT(*) FROM referrals
            WHERE referrer_id = ? AND total_earned > 0
        ''', (user_id,))
        
        active_referrals = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'referral_code': referral_code,
            'total_referrals': total_referrals,
            'total_commission': total_commission,
            'active_referrals': active_referrals,
            'commission_rate': self.COMMISSION_RATE
        }
    
    def get_referrals(self, user_id: str) -> List[Dict]:
        """
        Get list of referrals for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of referral details
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT referee_id, created_at, total_earned, commission_paid
            FROM referrals
            WHERE referrer_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'referee_id': row[0],
                'joined_at': row[1],
                'total_earned': row[2],
                'commission_paid': row[3]
            }
            for row in rows
        ]
    
    def get_commission_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        Get commission payment history
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            
        Returns:
            List of commission payments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT referee_id, amount, earnings_source, timestamp
            FROM commission_history
            WHERE referrer_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'referee_id': row[0],
                'commission': row[1],
                'from_earnings': row[2],
                'timestamp': row[3]
            }
            for row in rows
        ]
    
    def get_program_stats(self) -> Dict:
        """
        Get overall referral program statistics
        
        Returns:
            Program statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total referrals
        cursor.execute('SELECT COUNT(*) FROM referrals')
        total_referrals = cursor.fetchone()[0]
        
        # Total commission paid
        cursor.execute('SELECT SUM(commission_paid) FROM referrals')
        total_commission = cursor.fetchone()[0] or 0.0
        
        # Active referrers
        cursor.execute('SELECT COUNT(*) FROM referral_codes WHERE total_referrals > 0')
        active_referrers = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_referrals': total_referrals,
            'total_commission_paid': total_commission,
            'active_referrers': active_referrers,
            'commission_rate': self.COMMISSION_RATE
        }
    
    def __repr__(self):
        stats = self.get_program_stats()
        return f"ReferralProgram({stats['total_referrals']} referrals, {stats['active_referrers']} referrers)"
