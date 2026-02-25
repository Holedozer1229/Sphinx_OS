"""
ANUBIS™ Fiber-Optic Structured-Light Continuous-Variable Quantum Computing

Implements the patent-pending architecture:
    ANUBIS™: A Fiber-Optic Structured-Light Continuous-Variable Quantum Computing
    System Powered by SphinxOS™ Adaptive Entanglement Control and SKYNT™
    Scalar Phase Networking

Components:
- SqueezedLightSource       : OPO-based squeezed quadrature states at 1550 nm
- STOVModeImprinter         : Spatiotemporal optical vortex OAM mode imprinting
- FiberLoopMultiplexer      : Time-multiplexed fiber delay loop
- ClusterStateEntangler     : Cylindrical time × OAM cluster-state lattice
- RareEarthMemory           : Coherent rare-earth doped fiber optical memory
- HomodyneDetector          : Homodyne quadrature measurement arrays
- SphinxOSAdaptiveController: Adaptive entanglement weight optimization (SphinxOS™)
- SKYNTPhaseNetwork         : Scalar phase synchronization network (SKYNT™)
- ANUBISFiberCVProcessor    : Top-level integrated CV quantum processor
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("SphinxOS.AnubisCore.FiberCV")


# ---------------------------------------------------------------------------
# Squeezed Light Source
# ---------------------------------------------------------------------------

class SqueezedLightSource:
    """
    Optical parametric oscillator (OPO) squeezed-light source.

    Generates continuous-variable squeezed vacuum states in the quadrature
    basis at telecom wavelength (1550 nm).

    The squeezed quadrature variance satisfies:
        Var(X_squeezed) = exp(-2r) / 2
        Var(P_anti-squeezed) = exp(+2r) / 2

    where r is the squeezing parameter related to squeezing level in dB:
        S_dB = -10 log10(exp(-2r)) = 20r / ln(10)
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        squeezing_db: float = 12.0,
        num_modes: int = 2048,
    ):
        """
        Initialize squeezed light source.

        Args:
            wavelength_nm: Carrier wavelength in nanometres (default 1550 nm)
            squeezing_db : Squeezing level in dB (12–15 dB achievable)
            num_modes    : Number of temporal pulse modes to generate
        """
        self.wavelength_nm = wavelength_nm
        self.squeezing_db = squeezing_db
        self.num_modes = num_modes

        # Squeezing parameter r from dB level
        self.r = squeezing_db * np.log(10) / 20.0
        self.var_squeezed = np.exp(-2 * self.r) / 2.0
        self.var_antisqueezed = np.exp(+2 * self.r) / 2.0

        logger.info(
            f"SqueezedLightSource: λ={wavelength_nm} nm, "
            f"S={squeezing_db} dB, r={self.r:.4f}, "
            f"Var(X)={self.var_squeezed:.6f}"
        )

    def generate_pulse_train(self) -> np.ndarray:
        """
        Generate a train of squeezed quadrature pairs (X, P).

        Returns:
            Array of shape (num_modes, 2) — columns are X and P quadratures.
        """
        X = np.random.normal(0.0, np.sqrt(self.var_squeezed), self.num_modes)
        P = np.random.normal(0.0, np.sqrt(self.var_antisqueezed), self.num_modes)
        pulses = np.column_stack([X, P])
        logger.debug(f"Generated {self.num_modes} squeezed pulses")
        return pulses

    def get_state(self) -> Dict[str, Any]:
        """Return current source parameters."""
        return {
            "wavelength_nm": self.wavelength_nm,
            "squeezing_db": self.squeezing_db,
            "squeezing_r": self.r,
            "var_squeezed": self.var_squeezed,
            "var_antisqueezed": self.var_antisqueezed,
            "num_modes": self.num_modes,
        }


# ---------------------------------------------------------------------------
# STOV Mode Imprinter
# ---------------------------------------------------------------------------

class STOVModeImprinter:
    """
    Spatiotemporal optical vortex (STOV) structured-light mode imprinter.

    Implements the OAM unitary:
        U_OAM = exp(i ℓ L_z)

    generating helical phase profiles:
        ψ(r, φ, t) = A(r, t) · exp(i ℓ φ)

    Realised via silicon-nitride photonics / metasurface spiral phase plates
    and multi-plane light conversion elements.
    """

    def __init__(self, oam_modes: Optional[List[int]] = None):
        """
        Initialize mode imprinter.

        Args:
            oam_modes: List of OAM indices ℓ to support.
                       Default: [-2, -1, 0, 1, 2] (5 modes).
        """
        if oam_modes is None:
            oam_modes = list(range(-2, 3))  # ℓ ∈ {-2,-1,0,1,2}
        self.oam_modes = oam_modes
        self.num_oam = len(oam_modes)

        logger.info(
            f"STOVModeImprinter: {self.num_oam} OAM modes, "
            f"ℓ ∈ {{{oam_modes[0]},...,{oam_modes[-1]}}}"
        )

    def imprint(self, pulses: np.ndarray) -> np.ndarray:
        """
        Imprint OAM modes onto a temporal pulse train.

        Each temporal mode t is assigned a cyclic OAM index.  The imprinting
        adds a phase shift exp(i ℓ φ_t) to the complex amplitude derived from
        the (X, P) quadratures.

        Args:
            pulses: Array (T, 2) of (X, P) quadrature pairs.

        Returns:
            Array (T, num_oam, 2) — (X, P) for each (time, OAM) mode cell.
        """
        T = pulses.shape[0]
        oam_tensor = np.zeros((T, self.num_oam, 2))

        for t_idx in range(T):
            x, p = pulses[t_idx]
            amplitude = x + 1j * p
            for ell_idx, ell in enumerate(self.oam_modes):
                phi_t = 2 * np.pi * t_idx / T  # Azimuthal angle proxy
                phase = np.exp(1j * ell * phi_t)
                rotated = amplitude * phase
                oam_tensor[t_idx, ell_idx, 0] = rotated.real  # X
                oam_tensor[t_idx, ell_idx, 1] = rotated.imag  # P

        logger.debug(f"Imprinted OAM modes onto {T} temporal pulses → shape {oam_tensor.shape}")
        return oam_tensor

    def get_state(self) -> Dict[str, Any]:
        """Return imprinter configuration."""
        return {
            "oam_modes": self.oam_modes,
            "num_oam_modes": self.num_oam,
        }


# ---------------------------------------------------------------------------
# Fiber Loop Multiplexer
# ---------------------------------------------------------------------------

class FiberLoopMultiplexer:
    """
    Fiber delay loop for time-domain multiplexing.

    Maps the pulse train into a finite buffer of T_buffer temporal slots,
    modelling propagation through a low-loss single-mode fiber at 1550 nm.

    Fiber loss < 0.2 dB/km is applied per round-trip.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        loop_length_km: float = 1.0,
        fiber_loss_db_per_km: float = 0.18,
    ):
        """
        Initialize fiber loop multiplexer.

        Args:
            buffer_size          : Number of temporal time slots in the loop.
            loop_length_km       : Physical fiber loop length in km.
            fiber_loss_db_per_km : Fiber attenuation in dB/km (≤0.2 dB/km).
        """
        self.buffer_size = buffer_size
        self.loop_length_km = loop_length_km
        self.fiber_loss_db_per_km = fiber_loss_db_per_km

        # Round-trip power transmission coefficient
        loss_db = fiber_loss_db_per_km * loop_length_km
        self.transmission = 10 ** (-loss_db / 10.0)

        self._buffer: Optional[np.ndarray] = None

        logger.info(
            f"FiberLoopMultiplexer: {buffer_size} slots, "
            f"L={loop_length_km} km, T={self.transmission:.6f}"
        )

    def load(self, oam_tensor: np.ndarray) -> np.ndarray:
        """
        Load OAM-imprinted modes into the fiber loop buffer.

        Truncates or zero-pads the time axis to match buffer_size.

        Args:
            oam_tensor: Array (T, num_oam, 2).

        Returns:
            Buffered array (buffer_size, num_oam, 2) after loss.
        """
        T, num_oam, _ = oam_tensor.shape

        if T >= self.buffer_size:
            buffered = oam_tensor[: self.buffer_size].copy()
        else:
            pad = np.zeros((self.buffer_size - T, num_oam, 2))
            buffered = np.concatenate([oam_tensor, pad], axis=0)

        # Apply fiber propagation loss
        buffered *= np.sqrt(self.transmission)

        self._buffer = buffered
        logger.debug(f"Fiber loop loaded: shape={buffered.shape}, T={self.transmission:.6f}")
        return buffered

    def get_buffer(self) -> Optional[np.ndarray]:
        """Return the current buffer contents."""
        return self._buffer

    def get_state(self) -> Dict[str, Any]:
        """Return multiplexer parameters."""
        return {
            "buffer_size": self.buffer_size,
            "loop_length_km": self.loop_length_km,
            "fiber_loss_db_per_km": self.fiber_loss_db_per_km,
            "transmission": self.transmission,
            "buffer_loaded": self._buffer is not None,
        }


# ---------------------------------------------------------------------------
# Cluster State Entangler
# ---------------------------------------------------------------------------

class ClusterStateEntangler:
    """
    Generates a cylindrical time × OAM continuous-variable cluster lattice.

    The effective Hamiltonian is:
        H = Σ_{t,ℓ} κ X_{t,ℓ} X_{t+1,ℓ}  (temporal edges)
              + γ X_{t,ℓ} X_{t,ℓ+1}         (OAM edges)

    Entanglement gates are beam-splitter-realised CZ operations:
        U_CZ = exp(i X_i X_j)

    The result is a cylindrical graph state on the (T × num_oam) lattice
    with periodic OAM boundary conditions.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        gamma: float = 0.5,
    ):
        """
        Initialize cluster state entangler.

        Args:
            kappa: Temporal entanglement coupling strength.
            gamma: OAM-mode entanglement coupling strength.
        """
        self.kappa = kappa
        self.gamma = gamma

        # Entanglement weight matrix (updated adaptively by SphinxOS)
        self._weights: Optional[np.ndarray] = None

        logger.info(f"ClusterStateEntangler: κ={kappa}, γ={gamma}")

    def entangle(self, buffered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CZ-type entanglement across the cylindrical lattice.

        For each (t, ℓ) site the X-quadrature displacement is coupled to
        its neighbours in a Gaussian-boson-sampling style:
            X_{t,ℓ} → X_{t,ℓ} + κ X_{t+1,ℓ} + γ (X_{t,ℓ+1} + X_{t,ℓ-1})
            P_{t,ℓ} → P_{t,ℓ} + κ P_{t+1,ℓ} + γ (P_{t,ℓ+1} + P_{t,ℓ-1})

        Args:
            buffered: Array (T, num_oam, 2) from the fiber loop.

        Returns:
            cluster_state : Array (T, num_oam, 2) — entangled cluster state.
            adjacency     : Binary adjacency matrix (T*num_oam, T*num_oam).
        """
        T, num_oam, _ = buffered.shape
        cluster = buffered.copy()

        # Build entanglement adjacency (for record-keeping / SphinxOS feedback)
        N = T * num_oam
        adjacency = np.zeros((N, N))

        for t in range(T):
            for ell in range(num_oam):
                idx = t * num_oam + ell
                t_next = (t + 1) % T
                ell_next = (ell + 1) % num_oam
                ell_prev = (ell - 1) % num_oam

                # Temporal coupling (κ)
                w_t = self.kappa if self._weights is None else float(
                    self._weights[idx, t_next * num_oam + ell]
                )
                cluster[t, ell, 0] += w_t * buffered[t_next, ell, 0]
                cluster[t, ell, 1] += w_t * buffered[t_next, ell, 1]
                adjacency[idx, t_next * num_oam + ell] = w_t

                # OAM coupling (γ)
                w_oam = self.gamma if self._weights is None else float(
                    self._weights[idx, t * num_oam + ell_next]
                )
                cluster[t, ell, 0] += w_oam * (
                    buffered[t, ell_next, 0] + buffered[t, ell_prev, 0]
                )
                cluster[t, ell, 1] += w_oam * (
                    buffered[t, ell_next, 1] + buffered[t, ell_prev, 1]
                )
                adjacency[idx, t * num_oam + ell_next] = w_oam
                adjacency[idx, t * num_oam + ell_prev] = w_oam

        logger.debug(f"Cluster state generated: shape={cluster.shape}")
        return cluster, adjacency

    def update_weights(self, new_weights: np.ndarray):
        """
        Update entanglement weight matrix (called by SphinxOSAdaptiveController).

        Args:
            new_weights: Square array (T*num_oam, T*num_oam) of coupling weights.
        """
        self._weights = new_weights
        logger.debug("Entanglement weights updated")

    def get_state(self) -> Dict[str, Any]:
        """Return entangler configuration."""
        return {
            "kappa": self.kappa,
            "gamma": self.gamma,
            "weights_set": self._weights is not None,
        }


# ---------------------------------------------------------------------------
# Rare-Earth Optical Memory
# ---------------------------------------------------------------------------

class RareEarthMemory:
    """
    Rare-earth doped fiber optical memory segment.

    Models coherent coupling between photonic field modes (a) and a
    rare-earth spin ensemble (b) via the beam-splitter Hamiltonian:
        H_mem = g (a b† + a† b)

    Used for synchronisation buffering and inter-node delay compensation.
    The stored fidelity decays with dephasing time T2.
    """

    def __init__(
        self,
        coupling_g: float = 1.0,
        dephasing_t2_us: float = 1000.0,
        num_memory_modes: int = 8,
    ):
        """
        Initialize rare-earth optical memory.

        Args:
            coupling_g       : Photon-spin coupling constant g (arb. units).
            dephasing_t2_us  : Spin dephasing time T2 in microseconds.
            num_memory_modes : Number of independent memory cells.
        """
        self.coupling_g = coupling_g
        self.dephasing_t2_us = dephasing_t2_us
        self.num_memory_modes = num_memory_modes

        # Internal memory state: each cell stores a (X, P) pair
        self._stored = np.zeros((num_memory_modes, 2))

        logger.info(
            f"RareEarthMemory: g={coupling_g}, T2={dephasing_t2_us} µs, "
            f"{num_memory_modes} cells"
        )

    def write(self, cluster_state: np.ndarray, time_index: int = 0) -> None:
        """
        Write a slice of the cluster state into memory.

        Stores the first num_memory_modes (X, P) cells from the given
        temporal slice of the cluster state.

        Args:
            cluster_state: Array (T, num_oam, 2).
            time_index   : Temporal index to write from.
        """
        modes = cluster_state[time_index]  # (num_oam, 2)
        n = min(self.num_memory_modes, modes.shape[0])
        # Coherent beam-splitter coupling: stored ← cos(g·π/4)·stored + sin(g·π/4)·input
        theta = self.coupling_g * np.pi / 4.0
        self._stored[:n] = (
            np.cos(theta) * self._stored[:n] + np.sin(theta) * modes[:n]
        )
        logger.debug(f"Memory write at t={time_index}: {n} modes stored")

    def read(self) -> np.ndarray:
        """
        Read out stored field modes (non-destructive, with dephasing).

        Returns:
            Array (num_memory_modes, 2) of stored (X, P) pairs.
        """
        # Apply dephasing noise proportional to 1/T2
        sigma_dephase = 1.0 / (self.dephasing_t2_us + 1e-12)
        noise = np.random.normal(0.0, sigma_dephase, self._stored.shape)
        return self._stored + noise

    def get_state(self) -> Dict[str, Any]:
        """Return memory parameters and current stored values."""
        return {
            "coupling_g": self.coupling_g,
            "dephasing_t2_us": self.dephasing_t2_us,
            "num_memory_modes": self.num_memory_modes,
            "stored_norm": float(np.linalg.norm(self._stored)),
        }


# ---------------------------------------------------------------------------
# Homodyne Detector
# ---------------------------------------------------------------------------

class HomodyneDetector:
    """
    Homodyne detection array for continuous-variable quadrature measurement.

    Each detector projects a mode onto an arbitrary quadrature angle θ:
        m = X cos(θ) + P sin(θ)

    Shot-noise-limited detection with efficiency η ≤ 1.
    """

    def __init__(
        self,
        detection_efficiency: float = 0.95,
    ):
        """
        Initialize homodyne detector array.

        Args:
            detection_efficiency: Quantum efficiency η ∈ (0, 1].
        """
        if not (0 < detection_efficiency <= 1.0):
            raise ValueError("detection_efficiency must be in (0, 1]")
        self.eta = detection_efficiency

        logger.info(f"HomodyneDetector: η={detection_efficiency}")

    def measure(
        self,
        cluster_state: np.ndarray,
        angles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Measure all (t, ℓ) modes along specified quadrature angles.

        Args:
            cluster_state: Array (T, num_oam, 2) — (X, P) per mode.
            angles       : Array (T, num_oam) of local oscillator angles θ.
                           If None, X-quadrature (θ=0) is measured.

        Returns:
            Array (T, num_oam) of homodyne measurement outcomes.
        """
        T, num_oam, _ = cluster_state.shape

        if angles is None:
            angles = np.zeros((T, num_oam))

        X = cluster_state[..., 0]
        P = cluster_state[..., 1]

        # Ideal quadrature projection
        ideal = X * np.cos(angles) + P * np.sin(angles)

        # Loss from finite efficiency: scale + vacuum noise
        shot_noise_var = (1.0 - self.eta) / (2.0 * self.eta + 1e-12)
        noise = np.random.normal(0.0, np.sqrt(shot_noise_var), ideal.shape)

        outcomes = np.sqrt(self.eta) * ideal + noise
        logger.debug(f"Homodyne measurement: shape={outcomes.shape}")
        return outcomes

    def get_state(self) -> Dict[str, Any]:
        """Return detector parameters."""
        return {"detection_efficiency": self.eta}


# ---------------------------------------------------------------------------
# SphinxOS™ Adaptive Entanglement Controller
# ---------------------------------------------------------------------------

class SphinxOSAdaptiveController:
    """
    SphinxOS™ classical adaptive entanglement optimisation layer.

    Implements the gradient ascent update rule:
        g_{ij} → g_{ij} + η · ∂Φ/∂g_{ij}

    where Φ is a subsystem information metric (proxy for entanglement
    entropy) computed from homodyne measurement outcomes.

    The updated weights are fed back via electro-optic modulators /
    variable beam splitters to the ClusterStateEntangler.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        target_entropy: float = 1.0,
    ):
        """
        Initialize SphinxOS adaptive controller.

        Args:
            learning_rate  : Gradient step size η.
            target_entropy : Desired subsystem entropy level Φ*.
        """
        self.eta = learning_rate
        self.target_entropy = target_entropy
        self.phi_history: List[float] = []

        logger.info(
            f"SphinxOSAdaptiveController: η={learning_rate}, Φ*={target_entropy}"
        )

    def compute_phi(self, measurements: np.ndarray) -> float:
        """
        Compute subsystem information metric Φ from measurement outcomes.

        Φ is approximated as the normalised Shannon entropy of the
        empirical amplitude distribution of homodyne outcomes.

        Args:
            measurements: Array (T, num_oam) of homodyne outcomes.

        Returns:
            Scalar Φ ∈ [0, 1].
        """
        flat = measurements.ravel()
        # Bin into histogram to estimate entropy
        counts, _ = np.histogram(flat, bins=32)
        probs = counts / (counts.sum() + 1e-12)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        # Normalise to [0, 1]
        phi = entropy / np.log(len(probs) + 1e-12 + 1)
        self.phi_history.append(float(phi))
        return float(phi)

    def update_weights(
        self,
        current_weights: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """
        Compute updated entanglement weight matrix via gradient ascent on Φ.

        Args:
            current_weights: Current weight matrix (N, N).
            measurements   : Array (T, num_oam) of homodyne outcomes.

        Returns:
            Updated weight matrix (N, N).
        """
        phi = self.compute_phi(measurements)
        # Gradient proxy: perturb each weight, observe Φ shift
        # For efficiency, use a rank-1 outer-product approximation
        N = current_weights.shape[0]
        flat = measurements.ravel()

        # Approximate ∂Φ/∂g_{ij} ≈ correlation between mode activities
        # (simplified Hebbian-style rule)
        if flat.size >= N:
            activations = flat[:N]
        else:
            activations = np.pad(flat, (0, N - flat.size))

        grad = np.outer(activations, activations)
        # Normalise gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm

        new_weights = current_weights + self.eta * (self.target_entropy - phi) * grad
        # Clip to non-negative couplings
        new_weights = np.clip(new_weights, 0.0, None)

        logger.debug(f"SphinxOS update: Φ={phi:.4f}, Φ*={self.target_entropy:.4f}")
        return new_weights

    def get_state(self) -> Dict[str, Any]:
        """Return controller state."""
        return {
            "learning_rate": self.eta,
            "target_entropy": self.target_entropy,
            "phi_history_length": len(self.phi_history),
            "last_phi": self.phi_history[-1] if self.phi_history else None,
        }


# ---------------------------------------------------------------------------
# SKYNT™ Scalar Phase Synchronisation Network
# ---------------------------------------------------------------------------

class SKYNTPhaseNetwork:
    """
    SKYNT™ scalar phase synchronisation layer.

    Maintains deterministic optical carrier phase coherence across
    distributed ANUBIS nodes via a classical scalar phase reference.

    Each node holds a phase offset φ_i; synchronisation drives all offsets
    toward the global mean through a proportional feedback loop:
        φ_i(k+1) = φ_i(k) - α (φ_i(k) - φ_mean(k))

    Provides:
    - Deterministic optical carrier synchronisation
    - Phase-coherent clock distribution
    - Multi-node alignment
    """

    def __init__(
        self,
        num_nodes: int = 4,
        sync_gain: float = 0.5,
    ):
        """
        Initialize SKYNT phase network.

        Args:
            num_nodes : Number of ANUBIS nodes to synchronise.
            sync_gain : Proportional feedback gain α ∈ (0, 1].
        """
        if not (0 < sync_gain <= 1.0):
            raise ValueError("sync_gain must be in (0, 1]")
        self.num_nodes = num_nodes
        self.sync_gain = sync_gain

        # Initial random phase offsets (radians)
        self.phase_offsets = np.random.uniform(-np.pi, np.pi, num_nodes)
        self.coherence_history: List[float] = []

        logger.info(f"SKYNTPhaseNetwork: {num_nodes} nodes, α={sync_gain}")

    def synchronise(self, steps: int = 10) -> Dict[str, Any]:
        """
        Run synchronisation iterations and return final phase coherence.

        Args:
            steps: Number of feedback iterations.

        Returns:
            Dict with final phases, coherence, and convergence history.
        """
        coherence_trace = []

        for _ in range(steps):
            phi_mean = np.mean(self.phase_offsets)
            self.phase_offsets -= self.sync_gain * (self.phase_offsets - phi_mean)
            coherence = self._compute_coherence()
            coherence_trace.append(coherence)
            self.coherence_history.append(coherence)

        result = {
            "phase_offsets": self.phase_offsets.tolist(),
            "mean_phase": float(np.mean(self.phase_offsets)),
            "phase_variance": float(np.var(self.phase_offsets)),
            "coherence": float(coherence_trace[-1]),
            "convergence_trace": coherence_trace,
        }
        logger.debug(
            f"SKYNT sync ({steps} steps): coherence={result['coherence']:.6f}"
        )
        return result

    def _compute_coherence(self) -> float:
        """
        Compute scalar phase coherence C = |<exp(i φ)>|.

        C = 1 means perfect phase locking; C = 0 means random phases.
        """
        phasors = np.exp(1j * self.phase_offsets)
        return float(np.abs(np.mean(phasors)))

    def get_phase_reference(self, node_id: int) -> float:
        """
        Return the current phase reference for a given node.

        Args:
            node_id: Node index in [0, num_nodes).

        Returns:
            Phase offset φ_node in radians.
        """
        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"node_id {node_id} out of range [0, {self.num_nodes})")
        return float(self.phase_offsets[node_id])

    def get_state(self) -> Dict[str, Any]:
        """Return network state."""
        return {
            "num_nodes": self.num_nodes,
            "sync_gain": self.sync_gain,
            "phase_offsets": self.phase_offsets.tolist(),
            "coherence": self._compute_coherence(),
            "coherence_history_length": len(self.coherence_history),
        }


# ---------------------------------------------------------------------------
# ANUBIS™ Fiber-Optic CV Quantum Processor  (top-level)
# ---------------------------------------------------------------------------

class ANUBISFiberCVProcessor:
    """
    ANUBIS™ Fiber-Optic Structured-Light Continuous-Variable Quantum Processor.

    Integrates the full pipeline described in the patent specification:

        Laser → OPO (SqueezedLightSource)
              → STOV Chip (STOVModeImprinter)
              → Fiber Loop (FiberLoopMultiplexer)
              → Entanglement Network (ClusterStateEntangler)
              → Rare-Earth Memory (RareEarthMemory)
              → Homodyne Array (HomodyneDetector)
              → SphinxOS™ Feedback (SphinxOSAdaptiveController)
              ↕
            SKYNT™ (SKYNTPhaseNetwork)

    The architecture realises a scalable measurement-based quantum computation
    on a cylindrical time × OAM cluster lattice with adaptive entanglement
    topology optimisation.

    Patent parameters (Table 1):
        Wavelength  : 1550 nm
        Squeezing   : 12–15 dB
        OAM modes   : 5–9
        Logical modes: 2048
        Fiber loss  : < 0.2 dB/km
    """

    def __init__(
        self,
        wavelength_nm: float = 1550.0,
        squeezing_db: float = 12.0,
        oam_modes: Optional[List[int]] = None,
        num_logical_modes: int = 2048,
        fiber_loss_db_per_km: float = 0.18,
        loop_length_km: float = 1.0,
        kappa: float = 1.0,
        gamma: float = 0.5,
        memory_coupling_g: float = 1.0,
        memory_t2_us: float = 1000.0,
        detection_efficiency: float = 0.95,
        learning_rate: float = 0.01,
        target_entropy: float = 1.0,
        num_skynt_nodes: int = 4,
        skynt_sync_gain: float = 0.5,
    ):
        """
        Initialize ANUBIS™ fiber-optic CV quantum processor.

        Args:
            wavelength_nm         : Carrier wavelength (nm).
            squeezing_db          : OPO squeezing level (dB).
            oam_modes             : List of OAM indices ℓ.
            num_logical_modes     : Number of temporal pulse modes (T).
            fiber_loss_db_per_km  : Fiber attenuation (dB/km).
            loop_length_km        : Fiber delay loop length (km).
            kappa                 : Temporal entanglement coupling.
            gamma                 : OAM entanglement coupling.
            memory_coupling_g     : Rare-earth coupling constant.
            memory_t2_us          : Memory dephasing time (µs).
            detection_efficiency  : Homodyne detector efficiency η.
            learning_rate         : SphinxOS™ gradient step η.
            target_entropy        : SphinxOS™ target entropy Φ*.
            num_skynt_nodes       : Number of SKYNT™ distributed nodes.
            skynt_sync_gain       : SKYNT™ phase feedback gain α.
        """
        if oam_modes is None:
            oam_modes = list(range(-2, 3))

        logger.info("Initialising ANUBIS™ Fiber-Optic CV Quantum Processor...")

        # --- Subsystem initialisation ---
        self.source = SqueezedLightSource(
            wavelength_nm=wavelength_nm,
            squeezing_db=squeezing_db,
            num_modes=num_logical_modes,
        )
        self.imprinter = STOVModeImprinter(oam_modes=oam_modes)
        self.fiber_loop = FiberLoopMultiplexer(
            buffer_size=num_logical_modes,
            loop_length_km=loop_length_km,
            fiber_loss_db_per_km=fiber_loss_db_per_km,
        )
        self.entangler = ClusterStateEntangler(kappa=kappa, gamma=gamma)
        self.memory = RareEarthMemory(
            coupling_g=memory_coupling_g,
            dephasing_t2_us=memory_t2_us,
            num_memory_modes=len(oam_modes),
        )
        self.detector = HomodyneDetector(detection_efficiency=detection_efficiency)
        self.controller = SphinxOSAdaptiveController(
            learning_rate=learning_rate,
            target_entropy=target_entropy,
        )
        self.skynt = SKYNTPhaseNetwork(
            num_nodes=num_skynt_nodes,
            sync_gain=skynt_sync_gain,
        )

        # Initialise entanglement weight matrix
        T = num_logical_modes
        L = len(oam_modes)
        N = T * L
        self._weights = np.eye(N) * kappa  # diagonal = self-coupling (initial)

        logger.info(
            f"ANUBIS™ processor ready: T={T}, L={L}, N={N} modes, "
            f"λ={wavelength_nm} nm, S={squeezing_db} dB"
        )

    def run(
        self,
        measurement_angles: Optional[np.ndarray] = None,
        num_sync_steps: int = 10,
        adaptive_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Execute one full measurement-based quantum computation cycle.

        Pipeline:
            1. Generate squeezed pulse train (OPO)
            2. Imprint OAM modes (STOV chip)
            3. Load into fiber loop (time multiplexing)
            4. Apply cluster-state entanglement
            5. Write to rare-earth memory
            6. Perform homodyne measurements
            7. SphinxOS™ adaptive entanglement update
            8. SKYNT™ phase synchronisation

        Args:
            measurement_angles: Optional (T, num_oam) angles for homodyne.
                                 Defaults to X-quadrature (θ=0).
            num_sync_steps    : SKYNT™ feedback iterations.
            adaptive_steps    : SphinxOS™ weight update rounds.

        Returns:
            Dict with measurement outcomes, SphinxOS Φ metric, SKYNT
            coherence, and subsystem states.
        """
        # 1. Generate squeezed pulses
        pulses = self.source.generate_pulse_train()

        # 2. Imprint OAM modes
        oam_tensor = self.imprinter.imprint(pulses)

        # 3. Load fiber loop
        buffered = self.fiber_loop.load(oam_tensor)

        # 4. Cluster-state entanglement (with current weights)
        self.entangler.update_weights(self._weights)
        cluster_state, adjacency = self.entangler.entangle(buffered)

        # 5. Write to rare-earth memory
        self.memory.write(cluster_state, time_index=0)

        # 6. Homodyne measurement
        outcomes = self.detector.measure(cluster_state, angles=measurement_angles)

        # 7. SphinxOS™ adaptive weight update
        for _ in range(adaptive_steps):
            self._weights = self.controller.update_weights(self._weights, outcomes)
            self.entangler.update_weights(self._weights)

        # 8. SKYNT™ phase synchronisation
        skynt_result = self.skynt.synchronise(steps=num_sync_steps)

        results = {
            "measurement_outcomes": outcomes,
            "cluster_state_shape": list(cluster_state.shape),
            "adjacency_nnz": int(np.count_nonzero(adjacency)),
            "sphinxos": self.controller.get_state(),
            "skynt": skynt_result,
            "memory": self.memory.get_state(),
            "source": self.source.get_state(),
            "processor_state": self.get_state(),
        }

        logger.info(
            f"ANUBIS™ cycle complete: Φ={self.controller.get_state()['last_phi']:.4f}, "
            f"SKYNT coherence={skynt_result['coherence']:.6f}"
        )
        return results

    def get_state(self) -> Dict[str, Any]:
        """Return complete processor state."""
        return {
            "source": self.source.get_state(),
            "imprinter": self.imprinter.get_state(),
            "fiber_loop": self.fiber_loop.get_state(),
            "entangler": self.entangler.get_state(),
            "memory": self.memory.get_state(),
            "detector": self.detector.get_state(),
            "sphinxos_controller": self.controller.get_state(),
            "skynt_network": self.skynt.get_state(),
        }
