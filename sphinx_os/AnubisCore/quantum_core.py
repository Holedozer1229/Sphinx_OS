"""
QuantumCore - Unified quantum computing subsystem for AnubisCore

Integrates:
- QubitFabric (quantum circuit simulation)
- ErrorNexus (error correction and decoherence)
- Quantum volume metrics
- Entanglement caching
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("SphinxOS.AnubisCore.QuantumCore")


class QuantumCore:
    """
    Unified quantum computing core for AnubisCore.
    
    Manages all quantum circuit operations, qubit state, and error correction.
    """
    
    def __init__(self, num_qubits: int = 64):
        """
        Initialize QuantumCore.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.state = None
        self.circuit_history = []
        
        logger.info(f"QuantumCore initialized with {num_qubits} qubits")
        
        # Try to import existing QubitFabric
        try:
            from ..quantum.qubit_fabric import QubitFabric
            self.qubit_fabric = QubitFabric(num_qubits)
            logger.info("✅ QubitFabric integrated")
        except ImportError as e:
            logger.warning(f"Could not import QubitFabric: {e}")
            self.qubit_fabric = None
        
        # Try to import ErrorNexus
        try:
            from ..quantum.error_nexus import ErrorNexus
            self.error_nexus = ErrorNexus()
            logger.info("✅ ErrorNexus integrated")
        except ImportError as e:
            logger.warning(f"Could not import ErrorNexus: {e}")
            self.error_nexus = None
    
    def execute_circuit(self, circuit: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: List of quantum gate operations
            
        Returns:
            Quantum execution results
        """
        logger.debug(f"Executing circuit with {len(circuit)} gates")
        
        if self.qubit_fabric is not None:
            # Use real QubitFabric
            self.qubit_fabric.reset()
            results = self.qubit_fabric.run(circuit)
            self.state = self.qubit_fabric.state
            self.circuit_history.append(circuit)
            
            return {
                "measurements": results.measurements if hasattr(results, 'measurements') else {},
                "state": self.state,
                "circuit_depth": len(circuit),
                "num_qubits": self.num_qubits
            }
        else:
            # Fallback: simulate simple quantum state
            logger.warning("Using fallback quantum simulation")
            self.state = np.zeros(2**self.num_qubits, dtype=complex)
            self.state[0] = 1.0  # |0...0⟩ state
            
            return {
                "measurements": {},
                "state": self.state,
                "circuit_depth": len(circuit),
                "num_qubits": self.num_qubits,
                "fallback": True
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current quantum state."""
        return {
            "num_qubits": self.num_qubits,
            "state": self.state,
            "circuit_history_length": len(self.circuit_history),
            "has_qubit_fabric": self.qubit_fabric is not None,
            "has_error_nexus": self.error_nexus is not None
        }
    
    def reset(self):
        """Reset quantum state to |0...0⟩."""
        if self.qubit_fabric is not None:
            self.qubit_fabric.reset()
        else:
            self.state = np.zeros(2**self.num_qubits, dtype=complex)
            self.state[0] = 1.0
        
        logger.debug("Quantum state reset")
