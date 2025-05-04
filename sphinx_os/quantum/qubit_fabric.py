# sphinx_os/quantum/qubit_fabric.py
"""
QubitFabric: Quantum computing fabric for executing circuits, including Rydberg gates at wormhole nodes.
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Dict, Optional, Tuple
from ..utils.constants import CONFIG
from .quantum_volume import QuantumVolume
import logging

logger = logging.getLogger("SphinxOS.QubitFabric")

class QuantumResult:
    """Quantum result object."""
    def __init__(self, results: Dict[str, int], temporal_fidelity: float = 1.0):
        """
        Initialize QuantumResult.

        Args:
            results (Dict[str, int]): Circuit measurement outcomes.
            temporal_fidelity (float): Temporal fidelity of the execution.
        """
        self.results = results
        self.temporal_fidelity = temporal_fidelity

class QubitFabric:
    """Quantum computing fabric for executing circuits, including Rydberg gates."""
    
    def __init__(self, num_qubits: int):
        """
        Initialize the QubitFabric.

        Args:
            num_qubits (int): Number of qubits.
        """
        self.num_qubits = num_qubits
        self.metric_tensor = self._initialize_quantum_geometry(num_qubits)
        self.entanglement_map = np.zeros((num_qubits, num_qubits))
        self.quantum_volume = QuantumVolume()
        self.gates = {
            'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
            'CNOT': np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2),
            'CZ': np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=np.complex128).reshape(2, 2, 2, 2)
        }
        self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Initialize to |0...0⟩
        # Rydberg gate parameters
        self.rydberg_blockade_radius = CONFIG.get("rydberg_blockade_radius", 1e-6)  # in meters
        self.rydberg_interaction_strength = CONFIG.get("rydberg_interaction_strength", 1e6)  # in Hz
        self.qubit_positions = np.random.rand(self.num_qubits, 6) * 1e-6  # Random positions in 6D (scaled to μm)
        logger.info("QubitFabric initialized with %d qubits", num_qubits)

    def _initialize_quantum_geometry(self, n: int) -> np.ndarray:
        """
        Initialize the quantum geometry metric.

        Args:
            n (int): Number of qubits.

        Returns:
            np.ndarray: Distance metric tensor.
        """
        base_grid = np.random.rand(n, 6)
        return pairwise_distances(base_grid, metric='cosine')

    def run(self, circuit: List[Dict[str, any]], shots: int = CONFIG["shots"]) -> QuantumResult:
        """
        Execute a quantum circuit.

        Args:
            circuit (List[Dict[str, any]]): List of gate operations.
            shots (int): Number of measurement shots.

        Returns:
            QuantumResult: Results of the circuit execution.
        """
        logger.debug("Running quantum circuit with %d operations", len(circuit))
        # Reset state
        self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # |0...0⟩
        self.entanglement_map = np.zeros((self.num_qubits, self.num_qubits))

        for operation in circuit:
            gate = operation.get('gate')
            target = operation.get('target')
            control = operation.get('control')
            self._apply_gate(gate, target, control)

        # Measure and compute entanglement
        counts = self._measure(shots)
        self._compute_entanglement(counts)
        return QuantumResult(counts, temporal_fidelity=1.0)

    def apply_rydberg_gates(self, wormhole_nodes: np.ndarray) -> List[Tuple[int, int]]:
        """
        Apply Rydberg gates (CZ) at wormhole nodes using all 6 dimensions for distance calculations.

        Args:
            wormhole_nodes (np.ndarray): Array of wormhole node coordinates with shape (*grid_size, 6).

        Returns:
            List[Tuple[int, int]]: List of qubit pairs where Rydberg gates were applied.
        """
        logger.debug("Applying Rydberg gates at wormhole nodes")
        # Reshape wormhole_nodes to (num_points, 6) for distance calculation
        grid_shape = wormhole_nodes.shape[:-1]  # (*grid_size,)
        num_points = np.prod(grid_shape)
        node_coords = wormhole_nodes.reshape(num_points, 6)  # Shape: (num_points, 6)

        # Compute distances between qubits and all wormhole nodes in 6D space
        # qubit_positions: (num_qubits, 6), node_coords: (num_points, 6)
        # Broadcasting to compute distances for all pairs
        distances = np.linalg.norm(
            self.qubit_positions[:, np.newaxis, :] - node_coords[np.newaxis, :, :],
            axis=2
        )  # Shape: (num_qubits, num_points)

        # Find qubits within blockade radius of any wormhole node
        within_radius = distances < self.rydberg_blockade_radius  # Shape: (num_qubits, num_points)
        node_qubit_pairs = []

        # For each node, find qubits within radius and pair them
        for node_idx in range(num_points):
            eligible_qubits = np.where(within_radius[:, node_idx])[0]
            if len(eligible_qubits) >= 2:
                # Select the two closest qubits
                distances_to_node = distances[eligible_qubits, node_idx]
                closest_qubits = eligible_qubits[np.argsort(distances_to_node)[:2]]
                node_qubit_pairs.append((closest_qubits[0], closest_qubits[1]))
                # Apply CZ gate
                self._apply_gate('CZ', closest_qubits[0], closest_qubits[1])
                # Update entanglement map with enhanced entanglement due to Rydberg interaction
                self.entanglement_map[closest_qubits[0], closest_qubits[1]] += self.rydberg_interaction_strength * CONFIG.get("rydberg_coupling", 1e-3)
                self.entanglement_map[closest_qubits[1], closest_qubits[0]] = self.entanglement_map[closest_qubits[0], closest_qubits[1]]
                logger.debug("Applied Rydberg CZ gate between qubits %d and %d", closest_qubits[0], closest_qubits[1])

        return node_qubit_pairs

    def _apply_gate(self, gate: str, target: int, control: Optional[int] = None) -> None:
        """
        Apply a quantum gate to the state.

        Args:
            gate (str): Gate name ('H', 'T', 'CNOT', 'CZ').
            target (int): Target qubit index.
            control (Optional[int]): Control qubit index for CNOT or CZ.
        """
        if gate not in self.gates:
            logger.error("Unsupported gate: %s", gate)
            raise ValueError(f"Unsupported gate: {gate}")
        gate_matrix = self.gates[gate]
        if gate in ['H', 'T']:
            # Single-qubit gate
            U = np.eye(1, dtype=np.complex128)
            for i in range(self.num_qubits):
                if i == target:
                    U = np.kron(U, gate_matrix)
                else:
                    U = np.kron(U, np.eye(2))
            self.state = U @ self.state
        elif gate in ['CNOT', 'CZ']:
            if control is None:
                raise ValueError(f"{gate} gate requires a control qubit")
            # Ensure control and target are ordered correctly
            control_pos, target_pos = min(control, target), max(control, target)
            U = np.eye(1, dtype=np.complex128)
            gate_tensor = gate_matrix
            for i in range(self.num_qubits):
                if i == control_pos:
                    U = np.kron(U, np.eye(2))
                elif i == target_pos:
                    U = np.kron(U, gate_tensor.reshape(4, 4))
                else:
                    U = np.kron(U, np.eye(2))
            self.state = U @ self.state
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm
        else:
            logger.warning("Zero norm detected after gate application")
            self.state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            self.state[0] = 1.0  # Reset to |0...0⟩

    def _measure(self, shots: int) -> Dict[str, int]:
        """
        Measure the quantum state.

        Args:
            shots (int): Number of measurement shots.

        Returns:
            Dict[str, int]: Measurement counts.
        """
        probs = np.abs(self.state)**2
        probs /= np.sum(probs) + 1e-15
        outcomes = np.random.choice(2**self.num_qubits, size=shots, p=probs)
        counts = {'{:0{}b}'.format(i, self.num_qubits): 0 for i in range(2**self.num_qubits)}
        for outcome in outcomes:
            bitstring = '{:0{}b}'.format(outcome, self.num_qubits)
            counts[bitstring] += 1
        return counts

    def _compute_entanglement(self, counts: Dict[str, int]) -> None:
        """
        Compute entanglement map based on measurement outcomes, including Rydberg effects.

        Args:
            counts (Dict[str, int]): Measurement counts.
        """
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                # Base correlation from standard gates
                corr = 0
                for bitstring, count in counts.items():
                    if bitstring[i] == bitstring[j]:
                        corr += count
                    else:
                        corr -= count
                base_corr = corr / sum(counts.values())
                # Add Rydberg contribution (already updated in entanglement_map)
                self.entanglement_map[i, j] += base_corr
                self.entanglement_map[j, i] = self.entanglement_map[i, j]
        # Normalize entanglement map
        max_val = np.max(np.abs(self.entanglement_map))
        if max_val > 0:
            self.entanglement_map /= max_val
