# tests/test_quantum_circuit.py
"""
Unit tests for QuantumCircuitSimulator (integrated into QubitFabric).
"""
import unittest
import numpy as np
from sphinx_os.quantum.qubit_fabric import QubitFabric
from sphinx_os.utils.constants import CONFIG

class TestQuantumCircuitSimulator(unittest.TestCase):
    """Test suite for QuantumCircuitSimulator (now integrated into QubitFabric)."""
    
    def setUp(self):
        """Set up the test environment."""
        self.qubit_fabric = QubitFabric(num_qubits=64)

    def test_initialization(self):
        """Test QubitFabric initialization with 64 qubits and TVLE lattice."""
        self.assertEqual(self.qubit_fabric.num_qubits, 64)
        self.assertEqual(self.qubit_fabric.grid_size, (5, 5, 5, 5, 3, 3))
        self.assertEqual(self.qubit_fabric.total_points, 5625)
        self.assertEqual(self.qubit_fabric.quantum_state.state.size, 5625)
        self.assertEqual(self.qubit_fabric.entanglement_map.shape, (64, 64))
        self.assertEqual(self.qubit_fabric.qubit_positions.shape, (64, 6))
        self.assertAlmostEqual(np.linalg.norm(self.qubit_fabric.quantum_state.state), 1.0)
        state = self.qubit_fabric.get_state()
        self.assertEqual(state.shape, (5625,))
        self.assertAlmostEqual(np.linalg.norm(state), 1.0)

    def test_bell_state(self):
        """Test Bell state preparation on qubits 0 and 1."""
        circuit = [
            {"gate": "H", "target": 0},
            {"gate": "CNOT", "target": 1, "control": 0}
        ]
        result = self.qubit_fabric.run(circuit, shots=1024)
        counts = result.results
        self.assertIn("00", counts)
        self.assertIn("11", counts)
        self.assertAlmostEqual(counts["00"] / 1024, 0.5, delta=0.1)
        self.assertAlmostEqual(counts["11"] / 1024, 0.5, delta=0.1)
        initial_state = self.qubit_fabric.get_state()
        self.qubit_fabric.reset()
        reset_state = self.qubit_fabric.get_state()
        self.assertFalse(np.array_equal(initial_state, reset_state))
        self.assertAlmostEqual(np.linalg.norm(reset_state), 1.0)

    def test_rydberg_cz_gate(self):
        """Test circuit with a Rydberg CZ gate on qubits 0 and 1."""
        circuit = [
            {"gate": "H", "target": 0},
            {"gate": "H", "target": 1},
            {"gate": "CZ", "target": 1, "control": 0, "type": "rydberg"}
        ]
        grid_size = (2, 2, 2, 2, 2, 2)
        wormhole_nodes = np.zeros((*grid_size, 6))
        wormhole_nodes[0, 0, 0, 0, 0, 0] = np.zeros(6)
        wormhole_nodes[1, 1, 1, 1, 1, 1] = np.ones(6) * 1e-7
        self.qubit_fabric.apply_rydberg_gates(wormhole_nodes)
        result = self.qubit_fabric.run(circuit, shots=1024)
        counts = result.results
        self.assertIn("00", counts)
        self.assertIn("11", counts)
        self.assertGreater(self.qubit_fabric.entanglement_map[0, 1], 0)
