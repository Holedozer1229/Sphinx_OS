# sphinx_os/__init__.py
"""
SphinxOS: A Unified Quantum-Spacetime Operating System Kernel

This package integrates a 6D Theory of Everything (TOE) simulation with a universal quantum
circuit simulator, supporting arbitrary quantum circuits and entanglement testing via Bell state
simulation and CHSH inequality verification. Enhanced with Rydberg gates at wormhole nodes for
advanced quantum interactions, SphinxOS is designed for researchers and developers exploring
quantum-spacetime interactions. Note that commercial use requires a separate license; see the
LICENSE file for details.
"""
import logging

# Configure logging for the package
logging.getLogger("SphinxOS").addHandler(logging.NullHandler())

# Import main components to expose at the package level
from .main import SphinxOS
from .core.anubis_core import AnubisCore
from .core.unified_result import UnifiedResult

# Package metadata
__version__ = "0.2.1"  # Updated to reflect latest enhancements
__author__ = "Travis D. Jones"
__email__ = "holedozer@iCloud.com"
__license__ = "SphinxOS Commercial License"
__all__ = ["SphinxOS", "AnubisCore", "UnifiedResult"]
