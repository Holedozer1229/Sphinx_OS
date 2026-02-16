"""
Economics Module for SphinxOS

This module handles STX → BTC yield routing, treasury splits, and NFT multipliers.
"""

__version__ = "1.0.0"
__all__ = ["YieldCalculator", "EconomicSimulator"]

try:
    from .yield_calculator import YieldCalculator
    from .simulator import EconomicSimulator
except ImportError:
    # Handle case when imports fail
    YieldCalculator = None
    EconomicSimulator = None
