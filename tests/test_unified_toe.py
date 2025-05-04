# tests/test_unified_toe.py
import unittest
import numpy as np
from sphinx_os.quantum.unified_toe import Unified6DTOE

class TestUnified6DTOE(unittest.TestCase):
    def test_initialization(self):
        toe = Unified6DTOE()
        self.assertEqual(toe.grid_size, (5, 5, 5, 5, 3, 3))
        self.assertTrue(hasattr(toe, 'spin_network'))
        self.assertTrue(hasattr(toe, 'lattice'))
        self.assertEqual(toe.quantum_state.shape, toe.grid_size)

    def test_quantum_walk(self):
        toe = Unified6DTOE()
        toe.quantum_walk(0)
        self.assertEqual(len(toe.history), 1)
        self.assertGreater(len(toe.entanglement_history), 0)

if __name__ == '__main__':
    unittest.main()
