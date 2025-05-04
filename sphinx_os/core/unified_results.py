# sphinx_os/core/unified_result.py
"""
UnifiedResult: Result object combining quantum and spacetime outcomes.
"""
import numpy as np
from typing import Dict, Any

class UnifiedResult:
    """Result object combining quantum and spacetime outcomes."""
    
    def __init__(self, quantum_results: Any, spacetime_results: Dict[str, list]):
        """
        Initialize UnifiedResult.

        Args:
            quantum_results (Any): Results from quantum circuit execution.
            spacetime_results (Dict[str, list]): Results from spacetime simulation, including entanglement history.
        """
        self.quantum = quantum_results
        self.spacetime = spacetime_results
        self.fidelity = self._calculate_unified_fidelity()

    def _calculate_unified_fidelity(self) -> float:
        """Compute unified fidelity metric.

        Returns:
            float: Combined fidelity value.
        """
        q_fidelity = getattr(self.quantum, 'temporal_fidelity', 1.0)
        entanglement_history = self.spacetime.get('entanglement_history', [1.0])
        s_fidelity = np.mean(entanglement_history[-10:]) if entanglement_history else 1.0
        # Ensure numeric values
        q_fidelity = float(q_fidelity) if isinstance(q_fidelity, (int, float)) else 1.0
        s_fidelity = float(s_fidelity) if isinstance(s_fidelity, (int, float)) else 1.0
        return 0.7 * q_fidelity + 0.3 * s_fidelity
