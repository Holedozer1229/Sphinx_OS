# tests/test_anubis_core.py
"""
Unit tests for AnubisCore.
"""
import unittest
import numpy as np
from sphinx_os.core.anubis_core import AnubisCore
from sphinx_os.utils.constants import CONFIG

class TestAnubisCore(unittest.TestCase):
    """Test suite for AnubisCore."""
    
    def setUp(self):
        """Set up the test environment."""
        self.core = AnubisCore(grid_size=(2, 2, 2, 2, 2, 2), num_qubits=64)

    def test_initialization(self):
        """Test AnubisCore initialization with 64 qubits."""
        self.assertEqual(self.core.grid_size, (2, 2, 2, 2, 2, 2))
        self.assertEqual(self.core.num_qubits, 64)
        self.assertIsNotNone(self.core.toe)
        self.assertIsNotNone(self.core.qubit_fabric)
        self.assertGreater(len(self.core.toe.get_wormhole_nodes()), 0)

    def test_execute_with_rydberg(self):
        """Test the execute method with a Rydberg gate on a subset of qubits."""
        circuit = [
            {"gate": "H", "target": 0},
            {"gate": "CNOT", "target": 1, "control": 0},
            {"gate": "CZ", "target": 1, "control": 0, "type": "rydberg"}
        ]
        result = self.core.execute(circuit)
        self.assertIsNotNone(result)
        self.assertGreater(len(self.core.entanglement_history), 0)
        entanglement_map = self.core.qubit_fabric.entanglement_map
        self.assertGreater(entanglement_map[0, 1], 0)
        for i in range(2, 64):
            for j in range(64):
                if i != j:
                    self.assertEqual(entanglement_map[i, j], 0)

    def test_evolve_spacetime(self):
        """Test spacetime evolution with Rydberg effects."""
        self.core._evolve_spacetime()
        self.assertFalse(np.any(np.isnan(self.core.metric)))
        self.assertFalse(np.any(np.isnan(self.core.higgs_field)))
        self.assertFalse(np.any(np.isnan(self.core.toe.rydberg_effect)))
