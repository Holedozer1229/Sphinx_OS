"""
SphinxSkynet zk-EVM Module

Zero-knowledge EVM integration for:
- Smart contract verification
- Transaction privacy
- Cross-chain bridges
- Token transfers
"""

from .zk_prover import ZKProver, ProofType
from .evm_transpiler import EVMToCircomTranspiler
from .circuit_builder import CircuitBuilder

__all__ = ['ZKProver', 'ProofType', 'EVMToCircomTranspiler', 'CircuitBuilder']
