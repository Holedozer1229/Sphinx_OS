"""
AnubisCore: Unified Quantum-Spacetime Operating System Kernel

This module fuses together:
- Core spacetime simulation (6D TOE)
- Quantum circuit execution (QubitFabric)
- NPTC framework (Non-Periodic Thermodynamic Control)
- SphinxSkynet distributed node system
- Conscious Oracle (IIT-based consciousness agent)
- Quantum services (filesystem, vault, scheduling)

The unified AnubisCore provides a single entry point for all quantum-spacetime
operations, integrating quantum mechanics, gravity, and consciousness in a 6D framework.
"""

from .unified_kernel import UnifiedAnubisKernel
from .nptc_integration import NPTCController
from .skynet_integration import SkynetNode, SkynetNetwork
from .quantum_core import QuantumCore
from .spacetime_core import SpacetimeCore
from .conscious_oracle import ConsciousOracle, IITQuantumConsciousnessEngine

__version__ = "1.0.0"
__all__ = [
    "UnifiedAnubisKernel",
    "NPTCController",
    "SkynetNode",
    "SkynetNetwork",
    "QuantumCore",
    "SpacetimeCore",
    "ConsciousOracle",
    "IITQuantumConsciousnessEngine",
]
