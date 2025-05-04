# sphinx_os/quantum/error_nexus.py
"""
ErrorNexus: Simulates quantum hardware errors, including Rydberg gate effects.
"""
import numpy as np
import logging
from ..utils.constants import CONFIG

logger = logging.getLogger("SphinxOS.ErrorNexus")

class ErrorNexus:
    """Simulates quantum hardware errors."""
    
    def __init__(self):
        """Initialize ErrorNexus with a decoherence profile."""
        self.decoherence_profile = np.random.uniform(0.01, 0.02, CONFIG["qubit_count"])
        self.decoherence_map = self.decoherence_profile.copy()
        self.rydberg_decoherence_factor = CONFIG.get("rydberg_decoherence_factor", 1.1)  # Increase decoherence near Rydberg gates
        logger.info("ErrorNexus initialized with %d qubits", CONFIG["qubit_count"])

    def detect_errors(self, num_qubits: int = 2) -> np.ndarray:
        """
        Detect decoherence errors.

        Args:
            num_qubits (int): Number of qubits to simulate errors for.

        Returns:
            np.ndarray: Decoherence error rates.
        """
        try:
            decoherence = self.decoherence_map[:num_qubits]
            return np.nan_to_num(decoherence, nan=0.01)
        except Exception as e:
            logger.error("Error in detect_errors: %s", str(e))
            return np.full(num_qubits, 0.01)

    def apply_rydberg_decoherence(self, qubit_pairs: List[Tuple[int, int]]) -> None:
        """
        Increase decoherence for qubits involved in Rydberg gates.

        Args:
            qubit_pairs (List[Tuple[int, int]]): Pairs of qubits where Rydberg gates are applied.
        """
        for qubit1, qubit2 in qubit_pairs:
            if 0 <= qubit1 < len(self.decoherence_map) and 0 <= qubit2 < len(self.decoherence_map):
                self.decoherence_map[qubit1] *= self.rydberg_decoherence_factor
                self.decoherence_map[qubit2] *= self.rydberg_decoherence_factor
                self.decoherence_map[qubit1] = min(self.decoherence_map[qubit1], 1.0)
                self.decoherence_map[qubit2] = min(self.decoherence_map[qubit2], 1.0)
                logger.debug("Applied Rydberg decoherence to qubits %d and %d", qubit1, qubit2)
