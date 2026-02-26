"""
SphinxSkynet Mining Module

Multi-algorithm mining with:
- Spectral PoW
- SHA-256
- Ethash
- Keccak256
- Merge mining support
- Φ-boosted rewards
- Quantum Gravity IIT v8 kernel
"""

from .miner import SphinxMiner
from .pow_algorithms import PoWAlgorithms
from .spectral_pow import SpectralPoW
from .merge_miner import MergeMiningCoordinator
from .auto_miner import AutoMiner
from .quantum_gravity_miner_iit_v8 import QuantumGravityMinerIITv8, MineResultV8

__all__ = [
    'SphinxMiner',
    'PoWAlgorithms',
    'SpectralPoW',
    'MergeMiningCoordinator',
    'AutoMiner',
    'QuantumGravityMinerIITv8',
    'MineResultV8',
]
