# tests/test_main.py
"""
Unit tests for SphinxOS main class.
"""
import unittest
from sphinx_os.main import SphinxOS
from sphinx_os.utils.constants import CONFIG

class TestSphinxOS(unittest.TestCase):
    """Test suite for SphinxOS."""
    
    def setUp(self):
        """Set up the test environment."""
        self.sphinx_os = SphinxOS()

    def test_initialization(self):
        """Test SphinxOS initialization with 64 qubits."""
        self.assertEqual(self.sphinx_os.grid_size, (2, 2, 2, 2, 2, 2))
        self.assertEqual(self.sphinx_os.num_qubits, 64)
        self.assertIsNotNone(self.sphinx_os.anubis_core)
        state = self.sphinx_os.qubit_fabric.get_state()
        self.assertEqual(state.shape, (5625,))
        self.assertAlmostEqual(np.linalg.norm(state), 1.0)

    def test_emulate_on_hardware_with_rydberg(self):
        """Test hardware emulation with CHSH test including a Rydberg CZ gate."""
        result = self.sphinx_os.emulate_on_hardware()
        self.assertIn("counts", result)
        self.assertIn("fidelity", result)
        self.assertIn("S", result)
        self.assertGreater(abs(result["S"]), 2)
        entanglement_map = self.sphinx_os.qubit_fabric.entanglement_map
        self.assertGreater(entanglement_map[0, 1], 0)
        for i in range(2, 64):
            for j in range(64):
                if i != j:
                    self.assertEqual(entanglement_map[i, j], 0)
        initial_state = self.sphinx_os.qubit_fabric.get_state()
        self.sphinx_os.qubit_fabric.reset()
        reset_state = self.sphinx_os.qubit_fabric.get_state()
        self.assertFalse(np.array_equal(initial_state, reset_state))

    def test_run(self):
        """Test full simulation run with a simple circuit on a subset of qubits."""
        quantum_program = [
            {"gate": "H", "target": 0},
            {"gate": "CNOT", "target": 1, "control": 0}
        ]
        self.sphinx_os.run(quantum_program)
        self.assertGreater(len(self.sphinx_os.entanglement_history), 0)
