"""
Quantum Gravity module for SphinxOS.

This module implements the NPTC (Non-Periodic Thermodynamic Control) framework
for quantum gravity proofs and unification with hyper-relativity.
"""

from .nptc_framework import NPTCFramework, NPTCInvariant
from .quantum_gravity_proof import QuantumGravityProof
from .hyper_relativity import HyperRelativityUnification
from .toe_integration import NPTCEnhancedTOE, create_nptc_enhanced_toe
from .weyl_nodes import (
    TaAsLikeHamiltonian,
    WeylNode,
    BerryCurvatureField,
    ChernNumberCalculator,
    BerryPhaseCalculator,
    PairedWeylNodeAnalysis,
)

__all__ = [
    'NPTCFramework',
    'NPTCInvariant',
    'QuantumGravityProof',
    'HyperRelativityUnification',
    'NPTCEnhancedTOE',
    'create_nptc_enhanced_toe',
    'TaAsLikeHamiltonian',
    'WeylNode',
    'BerryCurvatureField',
    'ChernNumberCalculator',
    'BerryPhaseCalculator',
    'PairedWeylNodeAnalysis',
]
