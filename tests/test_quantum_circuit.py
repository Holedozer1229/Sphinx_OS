# tests/test_quantum_circuit.py
import unittest
import numpy as np
from sphinx_os.quantum.quantum_circuit import QuantumCircuitSimulator

class TestQuantumCircuitSimulator(unittest.TestCase):
    def test_bell_state(self):
        sim = QuantumCircuitSimulator(num_qubits=2)
        circuit = [
            {'gate': 'H', 'target': 0},
            {'gate': 'CNOT', 'target': 1, 'control': 0}
        ]
        counts = sim.run_circuit(circuit, shots=1000)
        self.assertIn('00', counts)
        self.assertIn('11', counts)
        self.assertTrue(counts['00'] + counts['11'] > 900)  # Expect ~50% each

    def test_t_gate(self):
        sim = QuantumCircuitSimulator(num_qubits=1)
        circuit = [{'gate': 'T', 'target': 0}]
        sim.run_circuit(circuit, shots=1)
        expected_phase = np.exp(1j * np.pi / 4)
        self.assertAlmostEqual(sim.state[1] / sim.state[0], expected_phase, places=5)

if __name__ == '__main__':
    unittest.main()
