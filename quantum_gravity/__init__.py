"""
Quantum Gravity module for SphinxOS.

This module implements the NPTC (Non-Periodic Thermodynamic Control) framework
for quantum gravity proofs and unification with hyper-relativity.
"""

from .nptc_framework import NPTCFramework, NPTCInvariant
from .quantum_gravity_proof import QuantumGravityProof
from .hyper_relativity import HyperRelativityUnification

__all__ = [
    'NPTCFramework',
    'NPTCInvariant',
    'QuantumGravityProof',
    'HyperRelativityUnification'
]
