# tests/test_anubis_core.py
import unittest
from sphinx_os.core.anubis_core import AnubisCore

class TestAnubisCore(unittest.TestCase):
    def test_initialization(self):
        core = AnubisCore(grid_size=(5, 5, 5, 5, 3, 3))
        self.assertEqual(core.grid_size, (5, 5, 5, 5, 3, 3))
        self.assertTrue(hasattr(core, 'error_nexus'))
        self.assertEqual(len(core.entanglement_history), 0)

    def test_sync_entanglement(self):
        core = AnubisCore(grid_size=(5, 5, 5, 5, 3, 3))
        quantum_result = type('QuantumResult', (), {'temporal_fidelity': 0.95})()
        metadata = {"entanglement_history": [0.5]}
        core._sync_entanglement(quantum_result, metadata)
        self.assertEqual(core.entanglement_history, [0.5])

if __name__ == '__main__':
    unittest.main()
