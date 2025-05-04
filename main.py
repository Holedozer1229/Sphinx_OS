# sphinx_os/main.py
"""
SphinxOS: Main class integrating TOE and quantum simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from .core.anubis_core import AnubisCore
from .quantum.unified_toe import Unified6DTOE
from .quantum.quantum_circuit import QuantumCircuitSimulator
from .utils.constants import CONFIG
from .utils.helpers import compute_entanglement_entropy

logger = logging.getLogger("SphinxOS")

class SphinxOS:
    """Main class for SphinxOS simulation."""
    def __init__(self):
        """Initialize SphinxOS."""
        logging.basicConfig(
            filename='sphinx_os.log',
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.grid_size = CONFIG["grid_size"]
        self.anubis_core = AnubisCore(self.grid_size)
        self.toe = Unified6DTOE()
        self.qubit_states = {}
        self.entanglement_history = []

    def emulate_on_hardware(self, circuit: List[Dict[str, any]] = None) -> Dict:
        """
        Simulate a quantum circuit, defaulting to a Bell state with CHSH test.

        Args:
            circuit (List[Dict[str, any]], optional): List of gate operations.
                Each dict contains 'gate' ('H', 'T', 'CNOT'), 'target', and optional 'control'.
                If None, runs a Bell state circuit with CHSH test.

        Returns:
            Dict: Results including counts, fidelity, and CHSH parameter (if applicable).
        """
        num_qubits = 2
        shots = CONFIG["shots"]
        simulator = QuantumCircuitSimulator(num_qubits)

        # Default Bell state circuit if none provided
        if circuit is None:
            circuit = [
                {'gate': 'H', 'target': 0},
                {'gate': 'CNOT', 'target': 1, 'control': 0}
            ]
            run_chsh = True
        else:
            run_chsh = False

        # Run the circuit
        counts = simulator.run_circuit(circuit, shots)

        # CHSH Test for Bell state
        S = None
        if run_chsh:
            correlations = []
            measurement_bases = [
                (np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]])),  # Z, Z
                (np.array([[1, 0], [0, -1]]), (np.array([[1, 0], [0, -1]]) + np.array([[0, 1], [1, 0]])) / np.sqrt(2)),  # Z, (Z+X)/sqrt(2)
                (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])),  # X, Z
                (np.array([[0, 1], [1, 0]]), (np.array([[1, 0], [0, -1]]) - np.array([[0, 1], [1, 0]])) / np.sqrt(2))  # X, (Z-X)/sqrt(2)
            ]
            for A, B in measurement_bases:
                temp_sim = QuantumCircuitSimulator(num_qubits)
                temp_sim.run_circuit(circuit, shots=1)  # Run to set state
                U_A = np.linalg.eigh(A)[1]
                U_B = np.linalg.eigh(B)[1]
                U = np.kron(U_A, U_B)
                rotated_state = U.conj().T @ temp_sim.state
                probs = np.abs(rotated_state)**2
                probs /= np.sum(probs) + 1e-15
                outcomes = np.random.choice(4, size=shots, p=probs)
                counts_ij = {'00': 0, '01': 0, '10': 0, '11': 0}
                for outcome in outcomes:
                    counts_ij[f'{outcome//2}{outcome%2}'] += 1
                E = (counts_ij['00']/shots + counts_ij['11']/shots -
                     counts_ij['01']/shots - counts_ij['10']/shots)
                correlations.append(E)

            # Apply fidelity factor using TOE fields
            decoherence = self.anubis_core.error_nexus.detect_errors(num_qubits)
            entanglement_entropy = compute_entanglement_entropy(self.toe.electron_field, self.grid_size)
            ricci_scalar = self.toe.ricci_scalar
            fidelity_factor = max(0.1, min(1.0, (1 - 0.1 * np.mean(np.abs(ricci_scalar)) -
                                                 0.05 * np.mean(decoherence)) * (1 + 0.1 * entanglement_entropy)))
            correlations = [c * fidelity_factor for c in correlations]
            S = correlations[0] + correlations[1] + correlations[2] - correlations[3]

        # Update qubit states
        self.qubit_states[(0, 0, 0, 0, 0, 0)] = np.array([counts.get('00', 0), counts.get('01', 0)], dtype=np.float64) / shots
        self.qubit_states[(0, 0, 0, 0, 0, 1)] = np.array([counts.get('10', 0), counts.get('11', 0)], dtype=np.float64) / shots

        # Sync entanglement
        self.anubis_core._sync_entanglement(
            type('QuantumResult', (), {'temporal_fidelity': fidelity_factor if run_chsh else 1.0})(),
            {"entanglement_history": [entanglement_entropy if run_chsh else 0.0]}
        )

        result = {"counts": counts, "fidelity": fidelity_factor if run_chsh else 1.0}
        if run_chsh:
            result["S"] = S
        return result

    def run(self):
        """Run the full simulation."""
        print("Starting SphinxOS 6D TOE Simulation with Universal Quantum Computing...")
        for i in range(CONFIG["max_iterations"]):
            try:
                self.toe.quantum_walk(i)
                if i % 10 == 0:
                    self.toe.visualize(i)
                    self.toe.visualize_quantum_flux(i)
                    if CONFIG["log_tensors"]:
                        np.savetxt(f"metric_iter{i}.txt", self.toe.metric[0, 0, 0, 0, 0, 0], fmt='%.6e')
                    # Run CHSH test periodically
                    result = self.emulate_on_hardware()
                    logger.info(f"Iteration {i}: CHSH |S| = {abs(result['S']):.3f}, "
                                f"{'Violates' if abs(result['S']) > 2 else 'Does not violate'} CHSH inequality")
                    self.visualize_chsh(result)
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                break
        self.toe.visualize(CONFIG["max_iterations"])
        self.toe.visualize_quantum_flux(CONFIG["max_iterations"])
        print("Simulation complete.")

    def visualize_chsh(self, result: Dict):
        """Visualize CHSH test results."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(result["counts"].keys(), result["counts"].values(), color="green")
        ax.set_xlabel("Quantum State")
        ax.set_ylabel("Counts")
        title = f"Bell State Measurement\nCHSH |S| = {abs(result['S']):.3f}, "
        title += "Violates" if abs(result['S']) > 2 else "Does not violate"
        title += " CHSH inequality"
        ax.set_title(title)
        plt.savefig('chsh_results.png')
        plt.close()
