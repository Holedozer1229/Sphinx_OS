# sphinx_os/quantum/qubit_fabric.py
"""
QubitFabric: Quantum computing fabric for executing circuits, using Temporal Vector Lattice Entanglement (TVLE).
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Dict, Optional, Tuple
from ..utils.constants import CONFIG, G, c, hbar, e, epsilon_0, m_n, v_higgs, l_p, kappa, INV_LAMBDA_SQ, TEMPORAL_CONSTANT, SECP256k1_N, SEARCH_START, SEARCH_END
from ..utils.helpers import compute_entanglement_entropy
from .quantum_volume import QuantumVolume
import logging
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import hashlib
import base58
from scipy.integrate import solve_ivp

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

class Hamiltonian:
    """Defines the Hamiltonian for the 6D TOE simulation with nonlinear scalar field."""
    def __init__(self, grid_size, dx, V, wormhole_state, logger):
        self.grid_size = grid_size
        self.total_points = np.prod(grid_size)
        self.dx = dx
        self.V = V  # Initial gravitational potential
        self.wormhole_state = wormhole_state
        self.logger = logger
        self.indices = np.arange(self.total_points).reshape(self.grid_size)
        self.center = np.array([s // 2 for s in grid_size], dtype=np.float64)
        self.weights = np.array(CONFIG["anisotropic_weights"])
        self.k = 1e-3 / self.dx
        self.omega = 2 * np.pi / (CONFIG["dt"] * 100)
        self.alpha = CONFIG["scalar_coupling"]
        self.time = 0.0

    def compute_scalar_field(self, t: float) -> np.ndarray:
        """
        Compute the nonlinear scalar field phi(r, t) at time t.

        Args:
            t (float): Current simulation time.

        Returns:
            np.ndarray: Scalar field phi(r, t) over the lattice.
        """
        phi = np.zeros(self.total_points, dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            r_vec = (np.array(idx, dtype=np.float64) - self.center) * self.dx
            r_6d = np.sqrt(np.sum(self.weights * r_vec**2)) + 1e-15
            kr = self.k * r_6d
            wt = self.omega * t
            phi[i] = -r_6d**2 * np.cos(kr - wt) + 2 * r_6d * np.sin(kr - wt) + 2 * np.cos(kr - wt)
        return phi

    def __call__(self, t, y, state_history, temporal_entanglement):
        """
        Compute the time derivative of the quantum state.

        Args:
            t (float): Current time
            y (np.ndarray): Current quantum state
            state_history (list): History of quantum states for CTC feedback
            temporal_entanglement (np.ndarray): Temporal entanglement vector

        Returns:
            np.ndarray: Derivative of the quantum state
        """
        self.time = t
        y_grid = y.reshape(self.grid_size)
        laplacian = np.zeros_like(y_grid, dtype=np.complex128)
        entanglement_term = np.zeros_like(y_grid, dtype=np.complex128)
        for axis in range(6):
            laplacian += (np.roll(y_grid, 1, axis=axis) + 
                          np.roll(y_grid, -1, axis=axis) - 2 * y_grid) / (self.dx**2)
            shift_plus = np.roll(y_grid, 1, axis=axis)
            shift_minus = np.roll(y_grid, -1, axis=axis)
            coupling = CONFIG["entanglement_coupling"] * (1 + np.sin(t))
            entanglement_term += coupling * (shift_plus - y_grid) * np.conj(shift_minus - y_grid)
        laplacian = laplacian.flatten()
        entanglement_term = entanglement_term.flatten()
        kinetic = -hbar**2 / (2 * m_n) * CONFIG["hopping_strength"] * laplacian
        phi = self.compute_scalar_field(t)
        V_scalar = self.V * (1 + 2.0 * np.sin(t)) + self.alpha * phi
        potential = V_scalar * y
        entanglement = entanglement_term
        H_psi = kinetic + potential + entanglement
        H_psi = -1j * H_psi / hbar
        phase_factor = np.exp(1j * 2 * t)
        wormhole_term = CONFIG["wormhole_coupling"] * phase_factor * (self.wormhole_state.conj().dot(y)) * self.wormhole_state
        ctc_term = np.zeros_like(y, dtype=np.complex128)
        if len(state_history) > 0:
            past_state = state_history[-1]
            phase_diff = np.angle(y) - np.angle(past_state)
            demon_sorting = TEMPORAL_CONSTANT * np.tanh(phase_diff)
            ctc_term = CONFIG["ctc_feedback_factor"] * np.exp(1j * demon_sorting) * np.abs(y)
        total_deriv = H_psi + wormhole_term + ctc_term
        total_deriv = np.clip(total_deriv, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        return total_deriv

class KeyExtractor:
    """Extracts Bitcoin private keys from the quantum state."""
    @staticmethod
    def extract(state, target_address, total_points, key_prediction_history):
        """
        Extract a private key from the quantum state.

        Args:
            state (QuantumState): The quantum state object
            target_address (str): Target Bitcoin address
            total_points (int): Total number of lattice points
            key_prediction_history (list): History of predicted keys

        Returns:
            tuple: (int, bool, str) - (key integer, success flag, WIF key if successful)
        """
        state_magnitude = state.get_magnitude()
        state_phase = state.get_phase()
        state_6d = state.reshape_to_6d()
        demon_observation = np.sum(state_6d, axis=(0, 1, 2, 3, 4))
        demon_observation = demon_observation.flatten()
        demon_factor = np.tile(demon_observation, total_points // 3)[:total_points]
        scalar_wave = CONFIG["j4_coupling"] * np.sin(state_phase)
        combined = state_magnitude + 0.5 * (state_phase / np.pi) + 0.1 * demon_factor + 0.1 * scalar_wave
        indices = np.argsort(combined)
        key_bits = np.zeros_like(combined, dtype=int)
        key_bits[indices[total_points // 2:]] = 1
        key_int = 0
        for bit in key_bits[:256]:
            key_int = (key_int << 1) | bit
        if key_int == 0:
            key_bits = np.zeros(256, dtype=int)
            key_bits[np.random.choice(256, 128, replace=False)] = 1
            key_int = 0
            for bit in key_bits:
                key_int = (key_int << 1) | bit
        key_int = max(SEARCH_START, min(SEARCH_END, key_int))
        success, wif = KeyExtractor.validate_key(key_int, target_address)
        if success:
            key_prediction_history.append(key_int)
        return key_int, success, wif

    @staticmethod
    def validate_key(key_int: int, target_address: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the private key corresponds to the target Bitcoin address.
        
        Uses cryptography library instead of ecdsa to avoid Minerva timing attack vulnerability.

        Args:
            key_int (int): Integer representation of the private key
            target_address (str): Target Bitcoin address

        Returns:
            Tuple[bool, Optional[str]]: (success flag, WIF key if successful)
        """
        try:
            key_bytes = key_int.to_bytes(32, byteorder='big')
            
            # Use cryptography library (resistant to side-channel attacks)
            private_key = ec.derive_private_key(
                int.from_bytes(key_bytes, byteorder='big'),
                ec.SECP256K1(),
                default_backend()
            )
            
            # Get public key
            public_key_obj = private_key.public_key()
            public_numbers = public_key_obj.public_numbers()
            
            # Convert to uncompressed format (0x04 + x + y)
            x_bytes = public_numbers.x.to_bytes(32, byteorder='big')
            y_bytes = public_numbers.y.to_bytes(32, byteorder='big')
            public_key = b'\x04' + x_bytes + y_bytes
            
            # Generate Bitcoin address
            sha256_hash = hashlib.sha256(public_key).digest()
            ripemd160_hash = hashlib.new('ripemd160')
            ripemd160_hash.update(sha256_hash)
            hashed_public_key = ripemd160_hash.digest()
            versioned = b'\x00' + hashed_public_key
            checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
            address_bytes = versioned + checksum
            computed_address = base58.b58encode(address_bytes).decode('utf-8')
            success = (computed_address == target_address)
            if success:
                extended_key = b'\x80' + key_bytes + b'\x01'
                checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
                wif = base58.b58encode(extended_key + checksum).decode('utf-8')
            else:
                wif = None
        except Exception as e:
            logger.error(f"Key validation failed: {e}")
            success = False
            wif = None
        return success, wif

class QuantumState:
    """Handles the quantum state and its evolution in the 6D grid."""
    def __init__(self, grid_size, logger):
        self.grid_size = grid_size
        self.total_points = np.prod(grid_size)
        self.logger = logger
        self.indices = np.arange(self.total_points).reshape(self.grid_size)
        self.center = np.array([s // 2 for s in grid_size], dtype=np.float64)
        self.weights = np.array(CONFIG["anisotropic_weights"])
        phases = np.random.uniform(0, 2 * np.pi, self.total_points)
        self.state = np.exp(1j * phases) / np.sqrt(self.total_points)
        self.temporal_entanglement = np.zeros(self.total_points, dtype=np.complex128)
        self.state_history = []
        self.time = 0.0
        self.k = 1e-3 / CONFIG["dx"]
        self.omega = 2 * np.pi / (CONFIG["dt"] * 100)
        self.beta = 1e-3
        self.gamma = 1e-3
        self.delta = 1e-6
        self.Lambda_0 = INV_LAMBDA_SQ**0.5
        self.r_6d = np.zeros(self.total_points, dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            r_vec = (np.array(idx, dtype=np.float64) - self.center) * CONFIG["dx"]
            self.r_6d[i] = np.sqrt(np.sum(self.weights * r_vec**2)) + 1e-15

    def compute_scalar_field(self, t: float) -> np.ndarray:
        """
        Compute the nonlinear scalar field phi(r, t) at time t.

        Args:
            t (float): Current simulation time.

        Returns:
            np.ndarray: Scalar field phi(r, t) over the lattice.
        """
        phi = np.zeros(self.total_points, dtype=np.complex128)
        for i in range(self.total_points):
            r = self.r_6d[i]
            kr = self.k * r
            wt = self.omega * t
            phi[i] = -r**2 * np.cos(kr - wt) + 2 * r * np.sin(kr - wt) + 2 * np.cos(kr - wt)
        return phi

    def evolve(self, dt, rtol, atol, hamiltonian):
        """
        Evolve the quantum state using the Schrödinger equation.

        Args:
            dt (float): Time step
            rtol (float): Relative tolerance for ODE solver
            atol (float): Absolute tolerance for ODE solver
            hamiltonian (callable): Hamiltonian function for evolution
        """
        self.time += dt
        phi = self.compute_scalar_field(self.time)
        self.state *= np.exp(1j * self.beta * phi)
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm
        state_flat = self.state.copy()
        sol = solve_ivp(
            lambda t, y: hamiltonian(t, y, self.state_history, self.temporal_entanglement),
            [0, dt],
            state_flat,
            method='RK45',
            rtol=rtol,
            atol=atol
        )
        if not sol.success:
            self.logger.error("Quantum state evolution failed")
            raise RuntimeError("ODE solver failed")
        self.state = sol.y[:, -1]
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm
        else:
            self.logger.warning("Quantum state norm is zero; resetting")
            phases = np.random.uniform(0, 2 * np.pi, self.total_points)
            self.state = np.exp(1j * phases) / np.sqrt(self.total_points)
        self.state_history.append(self.state.copy())
        if len(self.state_history) > 1:
            self.state_history = self.state_history[-1:]
        self.temporal_entanglement = self.state.conj() * CONFIG["entanglement_factor"]

    def get_magnitude(self):
        return np.abs(self.state)

    def get_phase(self):
        return np.angle(self.state)

    def reshape_to_6d(self):
        return self.state.reshape(self.grid_size)

    def compute_nonlinear_lambda(self) -> float:
        """
        Compute the nonlinear Lambda term.

        Returns:
            float: Modified Lambda value.
        """
        phi = self.compute_scalar_field(self.time)
        phi_energy_density = np.sum(phi**2) * (CONFIG["dx"]**6)
        return self.Lambda_0 * (1 + self.delta * phi_energy_density)

    def compute_entanglement_entropy(self) -> float:
        """
        Compute the entanglement entropy of the state, modified by the scalar field.

        Returns:
            float: Entanglement entropy.
        """
        return compute_entanglement_entropy(self.state, self.grid_size)

class QubitFabric:
    """Quantum computing fabric for executing circuits, using TVLE state representation."""
    
    def __init__(self, num_qubits: int):
        """
        Initialize the QubitFabric with TVLE state representation.

        Args:
            num_qubits (int): Number of qubits (64 as per requirement).
        """
        self.num_qubits = num_qubits
        self.grid_size = CONFIG["grid_size"]
        self.total_points = np.prod(self.grid_size)
        self.indices = np.arange(self.total_points).reshape(self.grid_size)
        self.quantum_state = QuantumState(self.grid_size, logger)
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
        self.rydberg_blockade_radius = CONFIG.get("rydberg_blockade_radius", 1e-6)
        self.rydberg_interaction_strength = CONFIG.get("rydberg_interaction_strength", 1e6)
        self.qubit_positions = np.random.rand(self.num_qubits, 6) * 1e-6
        self.dx = CONFIG["dx"]
        self.dt = CONFIG["dt"]
        self.key_prediction_history = []
        self.target_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        self.center = np.array([s // 2 for s in self.grid_size], dtype=np.float64)
        self.weights = np.array(CONFIG["anisotropic_weights"])
        self.r_6d_qubits = np.zeros(self.num_qubits, dtype=np.float64)
        for i in range(self.num_qubits):
            r_vec = self.qubit_positions[i] * self.dx
            self.r_6d_qubits[i] = np.sqrt(np.sum(self.weights * r_vec**2)) + 1e-15
        self.V = -G * m_n / (self.r_6d_qubits**4) * INV_LAMBDA_SQ
        points_per_qubit = self.total_points // self.num_qubits
        remainder = self.total_points % self.num_qubits
        V_expanded = np.zeros(self.total_points, dtype=np.complex128)
        start = 0
        for qubit in range(self.num_qubits):
            num_points = points_per_qubit + (1 if qubit < remainder else 0)
            V_expanded[start:start + num_points] = self.V[qubit]
            start += num_points
        self.V = V_expanded
        center_z = self.grid_size[2] // 2
        center_w1 = self.grid_size[4] // 2
        sigma = 1.0
        self.wormhole_state = np.zeros(self.total_points, dtype=np.complex128)
        pubkey_bits = [0, 1] * 128
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            z, w1 = idx[2], idx[4]
            r_6d = np.linalg.norm(np.array(idx, dtype=np.float64) * self.dx) + 1e-15
            bit = pubkey_bits[i % 256]
            self.wormhole_state[i] = np.exp(-r_6d**2 / (2 * sigma**2)) * (1 + 2 * (z - center_z) * (w1 - center_w1)) * bit
        self.wormhole_state /= np.linalg.norm(self.wormhole_state) + 1e-15
        self.hamiltonian = Hamiltonian(self.grid_size, self.dx, self.V, self.wormhole_state, logger)
        self.qubit_to_lattice = self._map_qubits_to_lattice()
        self.metric_tensor = self._initialize_quantum_geometry(self.num_qubits)
        logger.info("QubitFabric initialized with %d qubits on TVLE lattice (%d points)", num_qubits, self.total_points)

    def _map_qubits_to_lattice(self) -> Dict[int, List[int]]:
        """
        Map each qubit to a subset of lattice points.

        Returns:
            Dict[int, List[int]]: Mapping of qubit index to list of lattice point indices.
        """
        points_per_qubit = self.total_points // self.num_qubits
        remainder = self.total_points % self.num_qubits
        mapping = {}
        start = 0
        for qubit in range(self.num_qubits):
            num_points = points_per_qubit + (1 if qubit < remainder else 0)
            mapping[qubit] = list(range(start, start + num_points))
            start += num_points
        return mapping

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

    def _lattice_to_qubit_state(self, qubit_idx: int) -> np.ndarray:
        """
        Extract the effective 2D state vector for a qubit from the lattice state.

        Args:
            qubit_idx (int): Index of the qubit.

        Returns:
            np.ndarray: 2D state vector [alpha, beta] for the qubit.
        """
        lattice_indices = self.qubit_to_lattice[qubit_idx]
        if not lattice_indices:
            return np.array([1.0, 0.0], dtype=np.complex128)
        amplitude = np.sum(self.quantum_state.state[lattice_indices])
        norm = np.abs(amplitude) + 1e-15
        state = np.array([amplitude / norm, np.sqrt(1 - (np.abs(amplitude)**2 / norm**2))], dtype=np.complex128)
        return state / (np.linalg.norm(state) + 1e-15)

    def _apply_qubit_state_to_lattice(self, qubit_idx: int, qubit_state: np.ndarray) -> None:
        """
        Apply a 2D qubit state back to the lattice points associated with the qubit.

        Args:
            qubit_idx (int): Index of the qubit.
            qubit_state (np.ndarray): 2D state vector [alpha, beta] for the qubit.
        """
        lattice_indices = self.qubit_to_lattice[qubit_idx]
        if not lattice_indices:
            return
        num_points = len(lattice_indices)
        amplitude_per_point = qubit_state[0] / np.sqrt(num_points)
        for idx in lattice_indices:
            self.quantum_state.state[idx] = amplitude_per_point
        norm = np.linalg.norm(self.quantum_state.state)
        if norm >ऀ

System: 0:
            self.logger.warning("Zero norm in TVLE state application")
            phases = np.random.uniform(0, 2 * np.pi, self.total_points)
            self.quantum_state.state = np.exp(1j * phases) / np.sqrt(self.total_points)

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
        target_state = self._lattice_to_qubit_state(target)
        if gate in ['H', 'T']:
            U = self.gates[gate]
            new_target_state = U @ target_state
            self._apply_qubit_state_to_lattice(target, new_target_state)
        elif gate in ['CNOT', 'CZ']:
            if control is None:
                raise ValueError(f"{gate} gate requires a control qubit")
            control_state = self._lattice_to_qubit_state(control)
            state_2q = np.kron(control_state, target_state)
            U = self.gates[gate].reshape(4, 4)
            new_state_2q = U @ state_2q
            new_control_state = new_state_2q[:2] + new_state_2q[2:]
            new_target_state = np.array([new_state_2q[0] + new_state_2q[2], new_state_2q[1] + new_state_2q[3]])
            new_control_state /= np.linalg.norm(new_control_state) + 1e-15
            new_target_state /= np.linalg.norm(new_target_state) + 1e-15
            self._apply_qubit_state_to_lattice(control, new_control_state)
            self._apply_qubit_state_to_lattice(target, new_target_state)

    def reset(self) -> None:
        """
        Reset the quantum state to initial superposition.

        Returns:
            None
        """
        logger.debug("Resetting quantum state")
        phases = np.random.uniform(0, 2 * np.pi, self.total_points)
        self.quantum_state.state = np.exp(1j * phases) / np.sqrt(self.total_points)
        self.entanglement_map = np.zeros((self.num_qubits, self.num_qubits))
        self.quantum_state.state_history = []
        self.quantum_state.temporal_entanglement = np.zeros(self.total_points, dtype=np.complex128)
        self.quantum_state.time = 0.0

    def get_state(self) -> np.ndarray:
        """
        Get the current quantum state.

        Returns:
            np.ndarray: The current lattice state.
        """
        return self.quantum_state.state.copy()

    def run(self, circuit: List[Dict[str, any]], shots: int = CONFIG["shots"]) -> QuantumResult:
        """
        Execute a quantum circuit using TVLE state representation.

        Args:
            circuit (List[Dict[str, any]]): List of gate operations.
            shots (int): Number of measurement shots.

        Returns:
            QuantumResult: Results of the circuit execution.
        """
        logger.debug("Running quantum circuit with %d operations", len(circuit))
        self.reset()

        # Update gravitational potential with nonlinear Lambda and entanglement entropy
        Lambda = self.quantum_state.compute_nonlinear_lambda()
        S = self.quantum_state.compute_entanglement_entropy()
        self.V = -G * m_n / (self.r_6d_qubits**4) / (Lambda**2) * (1 + self.quantum_state.gamma * S)
        points_per_qubit = self.total_points // self.num_qubits
        remainder = self.total_points % self.num_qubits
        V_expanded = np.zeros(self.total_points, dtype=np.complex128)
        start = 0
        for qubit in range(self.num_qubits):
            num_points = points_per_qubit + (1 if qubit < remainder else 0)
            V_expanded[start:start + num_points] = self.V[qubit]
            start += num_points
        self.hamiltonian.V = V_expanded

        for operation in circuit:
            gate = operation.get('gate')
            target = operation.get('target')
            control = operation.get('control')
            self._apply_gate(gate, target, control)
            self.quantum_state.evolve(self.dt, CONFIG["rtol"], CONFIG["atol"], self.hamiltonian)
            # Update gravitational potential after each evolution
            Lambda = self.quantum_state.compute_nonlinear_lambda()
            S = self.quantum_state.compute_entanglement_entropy()
            self.V = -G * m_n / (self.r_6d_qubits**4) / (Lambda**2) * (1 + self.quantum_state.gamma * S)
            V_expanded = np.zeros(self.total_points, dtype=np.complex128)
            start = 0
            for qubit in range(self.num_qubits):
                num_points = points_per_qubit + (1 if qubit < remainder else 0)
                V_expanded[start:start + num_points] = self.V[qubit]
                start += num_points
            self.hamiltonian.V = V_expanded

        counts = self._measure(shots)
        self._compute_entanglement(counts)
        key_int, success, wif = KeyExtractor.extract(
            self.quantum_state, self.target_address, self.total_points, self.key_prediction_history
        )
        if success:
            logger.info(f"Predicted Bitcoin private key: WIF={wif}, int={hex(key_int)}")
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
        grid_shape = wormhole_nodes.shape[:-1]
        num_points = np.prod(grid_shape)
        node_coords = wormhole_nodes.reshape(num_points, 6)
        weights = np.array(CONFIG["anisotropic_weights"])
        delta = self.qubit_positions[:, np.newaxis, :] - node_coords[np.newaxis, :, :]
        delta_weighted = delta * weights
        distances = np.linalg.norm(delta_weighted, axis=2)
        within_radius = distances < self.rydberg_blockade_radius
        node_qubit_pairs = []
        for node_idx in range(num_points):
            eligible_qubits = np.where(within_radius[:, node_idx])[0]
            if len(eligible_qubits) >= 2:
                distances_to_node = distances[eligible_qubits, node_idx]
                closest_qubits = eligible_qubits[np.argsort(distances_to_node)[:2]]
                control, target = closest_qubits
                self._apply_gate('CZ', target, control)
                self.entanglement_map[control, target] += self.rydberg_interaction_strength * CONFIG.get("rydberg_coupling", 1e-3)
                self.entanglement_map[target, control] = self.entanglement_map[control, target]
                node_qubit_pairs.append((control, target))
                logger.debug("Applied Rydberg CZ gate between qubits %d and %d", control, target)
            self.quantum_state.evolve(self.dt, CONFIG["rtol"], CONFIG["atol"], self.hamiltonian)
            Lambda = self.quantum_state.compute_nonlinear_lambda()
            S = self.quantum_state.compute_entanglement_entropy()
            points_per_qubit = self.total_points // self.num_qubits
            remainder = self.total_points % self.num_qubits
            self.V = -G * m_n / (self.r_6d_qubits**4) / (Lambda**2) * (1 + self.quantum_state.gamma * S)
            V_expanded = np.zeros(self.total_points, dtype=np.complex128)
            start = 0
            for qubit in range(self.num_qubits):
                num_points = points_per_qubit + (1 if qubit < remainder else 0)
                V_expanded[start:start + num_points] = self.V[qubit]
                start += num_points
            self.hamiltonian.V = V_expanded
        return node_qubit_pairs

    def _measure(self, shots: int) -> Dict[str, int]:
        """
        Measure the quantum state on the lattice, projecting to qubit measurements.

        Args:
            shots (int): Number of measurement shots.

        Returns:
            Dict[str, int]: Measurement counts for all 64 qubits.
        """
        bitstrings = []
        for _ in range(shots):
            bitstring = ""
            for qubit in range(self.num_qubits):
                qubit_state = self._lattice_to_qubit_state(qubit)
                probs = np.abs(qubit_state)**2
                probs /= np.sum(probs) + 1e-15
                outcome = np.random.choice([0, 1], p=probs)
                bitstring += str(outcome)
            bitstrings.append(bitstring)
        counts = {}
        for bitstring in bitstrings:
            counts[bitstring] = counts.get(bitstring, 0) + 1
        for i in range(2**self.num_qubits):
            bitstring = '{:0{}b}'.format(i, self.num_qubits)
            counts[bitstring] = counts.get(bitstring, 0)
        return counts

    def _compute_entanglement(self, counts: Dict[str, int]) -> None:
        """
        Compute entanglement map based on measurement outcomes, including Rydberg effects.

        Args:
            counts (Dict[str, int]): Measurement counts.
        """
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                corr = 0
                for bitstring, count in counts.items():
                    if bitstring[i] == bitstring[j]:
                        corr += count
                    else:
                        corr -= count
                base_corr = corr / sum(counts.values())
                self.entanglement_map[i, j] += base_corr
                self.entanglement_map[j, i] = self.entanglement_map[i, j]
        max_val = np.max(np.abs(self.entanglement_map))
        if max_val > 0:
            self.entanglement_map /= max_val
