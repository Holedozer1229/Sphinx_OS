# sphinx_os/quantum/quantum_circuit.py
"""
QuantumCircuitSimulator: Simulates arbitrary quantum circuits using a universal gate set.
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("SphinxOS.QuantumCircuitSimulator")

class QuantumCircuitSimulator:
    """Simulates quantum circuits with H, CNOT, and T gates for universality."""
    def __init__(self, num_qubits: int):
        """
        Initialize the quantum circuit simulator.

        Args:
            num_qubits (int): Number of qubits.
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Initialize to |0...0⟩
        self.gates = {
            'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
            'CNOT': np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2)
        }

    def apply_gate(self, gate: str, target: int, control: Optional[int] = None) -> None:
        """
        Apply a quantum gate to the state.

        Args:
            gate (str): Gate name ('H', 'T', 'CNOT').
            target (int): Target qubit index.
            control (Optional[int]): Control qubit index for CNOT.
        """
        try:
            if gate not in self.gates:
                raise ValueError(f"Unsupported gate: {gate}")
            if target >= self.num_qubits or (control is not None and control >= self.num_qubits):
                raise ValueError("Invalid qubit index")

            if gate == 'CNOT':
                if control is None:
                    raise ValueError("CNOT requires a control qubit")
                # Construct full CNOT matrix
                U = self._construct_controlled_gate(control, target)
            else:
                # Single-qubit gate
                U = self.gates[gate]
                for i in range(self.num_qubits):
                    if i != target:
                        U = np.kron(U, np.eye(2))
                    elif i == target:
                        U = np.kron(np.eye(2**i), U) if i > 0 else U
                    else:
                        U = np.kron(U, np.eye(2))

            self.state = U @ self.state
            norm = np.linalg.norm(self.state)
            if norm > 0:
                self.state /= norm
            else:
                logger.warning("Zero norm after gate application")
        except Exception as e:
            logger.error(f"Error applying gate {gate}: {e}")
            raise

    def _construct_controlled_gate(self, control: int, target: int) -> np.ndarray:
        """
        Construct the full CNOT matrix for the system.

        Args:
            control (int): Control qubit index.
            target (int): Target qubit index.

        Returns:
            np.ndarray: Full CNOT matrix.
        """
        # Ensure control and target are distinct
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        # Initialize identity matrix for the system
        dim = 2**self.num_qubits
        U = np.eye(dim, dtype=np.complex128)
        
        # Construct CNOT for control and target qubits
        cnot = self.gates['CNOT']
        # Permute qubits to apply CNOT between control and target
        qubit_order = list(range(self.num_qubits))
        min_idx = min(control, target)
        max_idx = max(control, target)
        
        # Create permutation to bring control and target to positions 0 and 1
        new_order = qubit_order.copy()
        new_order[0] = min_idx
        new_order[1] = max_idx
        for i in range(2, self.num_qubits):
            if i <= min_idx:
                new_order[i] = i - 1
            elif i <= max_idx:
                new_order[i] = i - 2
            else:
                new_order[i] = i
        
        # Compute permutation matrix
        perm = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> j) & 1 for j in range(self.num_qubits)]
            new_bits = [0] * self.num_qubits
            for j, k in enumerate(new_order):
                new_bits[j] = bits[k]
            new_i = sum(b << j for j, b in enumerate(new_bits))
            perm[new_i, i] = 1
        
        # Apply CNOT on first two qubits
        cnot_full = self.gates['CNOT']
        for _ in range(self.num_qubits - 2):
            cnot_full = np.kron(cnot_full, np.eye(2))
        
        # Apply permutation, CNOT, and inverse permutation
        U = perm.T @ cnot_full @ perm
        return U

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """
        Simulate measurement of all qubits.

        Args:
            shots (int): Number of measurement shots.

        Returns:
            Dict[str, int]: Measurement counts.
        """
        try:
            probs = np.abs(self.state)**2
            probs /= np.sum(probs) + 1e-15
            outcomes = np.random.choice(2**self.num_qubits, size=shots, p=probs)
            counts = {}
            for i in range(2**self.num_qubits):
                binary = format(i, f'0{self.num_qubits}b')
                counts[binary] = np.sum(outcomes == i)
            return counts
        except Exception as e:
            logger.error(f"Error in measurement: {e}")
            raise

    def run_circuit(self, circuit: List[Dict[str, any]], shots: int = 1024) -> Dict[str, int]:
        """
        Run a quantum circuit.

        Args:
            circuit (List[Dict[str, any]]): List of gate operations, each with 'gate', 'target', and optional 'control'.
            shots (int): Number of measurement shots.

        Returns:
            Dict[str, int]: Measurement counts.
        """
        try:
            for op in circuit:
                gate = op.get('gate')
                target = op.get('target')
                control = op.get('control')
                self.apply_gate(gate, target, control)
            return self.measure(shots)
        except Exception as e:
            logger.error(f"Error running circuit: {e}")
            raise
