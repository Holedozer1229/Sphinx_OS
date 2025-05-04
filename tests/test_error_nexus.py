# tests/test_error_nexus.py
import unittest
import numpy as np
from sphinx_os.quantum.error_nexus import ErrorNexus

class TestErrorNexus(unittest.TestCase):
    def test_detect_errors(self):
        nexus = ErrorNexus()
        errors = nexus.detect_errors(num_qubits=2)
        self.assertEqual(len(errors), 2)
        self.assertTrue(np.all((errors >= 0.01) & (errors <= 0.02)))

if __name__ == '__main__':
    unittest.main()
