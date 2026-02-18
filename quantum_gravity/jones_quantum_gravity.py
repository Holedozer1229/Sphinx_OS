"""
Jones Quantum Gravity Resolution Implementation

This module implements the mathematical framework described in:
"Jones Quantum Gravity Resolution: Modular Hamiltonian, Deterministic Page Curve, 
and Emergent Islands" by Travis Jones (2026)

Key Components:
1. 27-dimensional octonionic operator space (exceptional Jordan algebra J_3(O))
2. Modular Hamiltonian construction with operators C, T, U, F
3. Entanglement islands as rank-reduction projections
4. Deterministic Page curve with modular nuclearity bounds
5. Geodesic flow in operator space
6. Spectral gap κ calculations

References:
- D. Page, "Average entropy of a subsystem," Phys. Rev. Lett. 71, 1291 (1993)
- S. Hawking, "Particle creation by black holes," Commun. Math. Phys. 43, 199 (1975)
- H. Araki, "Mathematical Theory of Quantum Fields," Oxford University Press, 1999
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from scipy.linalg import logm, expm, eigh
from scipy.integrate import solve_ivp, cumulative_trapezoid, trapezoid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger("SphinxOS.JonesQuantumGravity")


@dataclass
class ModularSpectrum:
    """Spectrum of the modular Hamiltonian."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kappa: float  # Spectral gap (minimum eigenvalue)
    
    def __post_init__(self):
        """Validate spectrum properties."""
        assert np.all(self.eigenvalues >= 0), "Modular Hamiltonian must have non-negative spectrum"


@dataclass
class EntanglementIsland:
    """Represents an entanglement island in operator space."""
    location: np.ndarray  # Position in operator space where Δ(k)=1
    rank_reduction: int   # Amount of rank reduction
    projection: np.ndarray  # Island projection operator P_island
    entropy_contribution: float  # Contribution to Page curve


class ExceptionalJordanAlgebra:
    """
    Exceptional Jordan algebra J_3(O) - 27-dimensional operator space.
    
    Represents 3×3 Hermitian matrices over the octonions O.
    In the real representation, this gives a 27-dimensional space.
    
    The algebra structure encodes non-associative and non-commutative
    properties essential for quantum gravity emergence.
    """
    
    def __init__(self, dimension: int = 27):
        """
        Initialize the exceptional Jordan algebra.
        
        Args:
            dimension: Dimension of the operator space (default 27 for J_3(O))
        """
        if dimension != 27:
            logger.warning(f"Non-standard dimension {dimension}. Standard J_3(O) has dimension 27.")
        
        self.dimension = dimension
        self.block_size = 9  # Each 3×3 octonionic block is 9-dimensional in real representation
        
        # Initialize structure constants (simplified model)
        self._init_structure_constants()
        
        logger.info(f"Initialized exceptional Jordan algebra J_3(O) with dimension {dimension}")
    
    def _init_structure_constants(self):
        """Initialize structure constants for the Jordan product."""
        # For full implementation, would need complete octonionic structure constants
        # Using simplified version that captures essential properties
        self.structure_constants = np.random.randn(self.dimension, self.dimension, self.dimension)
        # Symmetrize to enforce commutativity: a·b = b·a
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                for k in range(self.dimension):
                    avg = (self.structure_constants[i,j,k] + self.structure_constants[j,i,k]) / 2
                    self.structure_constants[i,j,k] = avg
                    self.structure_constants[j,i,k] = avg
    
    def jordan_product(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute Jordan product: A · B = (AB + BA) / 2
        
        Args:
            A, B: Operator matrices
            
        Returns:
            Jordan product A · B
        """
        return (A @ B + B @ A) / 2
    
    def create_hermitian_element(self) -> np.ndarray:
        """
        Create a random Hermitian element of J_3(O).
        
        Returns:
            27×27 Hermitian matrix
        """
        # Generate random Hermitian matrix
        H = np.random.randn(self.dimension, self.dimension)
        H = (H + H.T) / 2  # Symmetrize
        return H


class ContractionOperator:
    """
    Contraction operator D_p.
    
    Implements the contraction in octonionic operator space,
    reducing degrees of freedom analogously to gravitational collapse.
    """
    
    def __init__(self, dimension: int = 27, contraction_strength: float = 1.0):
        """
        Initialize contraction operator.
        
        Args:
            dimension: Dimension of operator space
            contraction_strength: Strength of contraction (default 1.0)
        """
        self.dimension = dimension
        self.strength = contraction_strength
        self._build_operator()
    
    def _build_operator(self):
        """Build the contraction operator matrix."""
        # Contraction operator reduces high-frequency modes
        # Using a diagonal operator with decreasing eigenvalues
        eigenvalues = np.exp(-np.arange(self.dimension) * 0.1 * self.strength)
        self.operator = np.diag(eigenvalues)
        
        logger.debug(f"Contraction operator built with eigenvalue range: [{eigenvalues[-1]:.4f}, {eigenvalues[0]:.4f}]")
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply contraction to operator state.
        
        Args:
            state: Operator state vector or matrix
            
        Returns:
            Contracted state
        """
        if state.ndim == 1:
            return self.operator @ state
        else:
            return self.operator @ state @ self.operator.T


class TrialityOperator:
    """
    Triality operator T.
    
    Implements triality rotations in the octonionic structure,
    cycling through the three fundamental representations.
    """
    
    def __init__(self, dimension: int = 27):
        """
        Initialize triality operator.
        
        Args:
            dimension: Dimension of operator space (should be 27 for J_3(O))
        """
        self.dimension = dimension
        self.block_size = dimension // 3
        self._build_operator()
    
    def _build_operator(self):
        """Build the triality rotation operator."""
        # Triality cycles the three 9×9 blocks: (D, E, F) → (E, F, D)
        T = np.zeros((self.dimension, self.dimension))
        
        # Permutation matrix for cyclic rotation of blocks
        bs = self.block_size
        # Block 1 (0:bs) → Block 2 (bs:2*bs)
        T[bs:2*bs, 0:bs] = np.eye(bs)
        # Block 2 (bs:2*bs) → Block 3 (2*bs:3*bs)
        T[2*bs:3*bs, bs:2*bs] = np.eye(bs)
        # Block 3 (2*bs:3*bs) → Block 1 (0:bs)
        T[0:bs, 2*bs:3*bs] = np.eye(bs)
        
        self.operator = T
        logger.debug(f"Triality operator built for {self.dimension}D space with block size {bs}")
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply triality rotation.
        
        Args:
            state: Operator state vector or matrix
            
        Returns:
            Rotated state
        """
        if state.ndim == 1:
            return self.operator @ state
        else:
            return self.operator @ state @ self.operator.T


class CTCRotationOperator:
    """
    CTC (Closed Timelike Curve) rotation operator U.
    
    Implements rotations associated with retrocausal structure,
    encoding temporal non-locality in the operator algebra.
    """
    
    def __init__(self, dimension: int = 27, rotation_angle: float = np.pi/6):
        """
        Initialize CTC rotation operator.
        
        Args:
            dimension: Dimension of operator space
            rotation_angle: Rotation angle in operator space
        """
        self.dimension = dimension
        self.angle = rotation_angle
        self._build_operator()
    
    def _build_operator(self):
        """Build the CTC rotation operator."""
        # Use a rotation in a principal 2-plane, extended to full space
        # This creates a unitary operator with complex phase structure
        
        # Generate random orthogonal basis
        Q, _ = np.linalg.qr(np.random.randn(self.dimension, self.dimension))
        
        # Create rotation in first two dimensions
        R = np.eye(self.dimension)
        c, s = np.cos(self.angle), np.sin(self.angle)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        
        # Conjugate by Q to get rotation in random 2-plane
        self.operator = Q @ R @ Q.T
        
        logger.debug(f"CTC rotation operator built with angle {self.angle:.4f} rad")
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply CTC rotation.
        
        Args:
            state: Operator state vector or matrix
            
        Returns:
            Rotated state
        """
        if state.ndim == 1:
            return self.operator @ state
        else:
            return self.operator @ state @ self.operator.T


class FreezingOperator:
    """
    Freezing operator F.
    
    Implements freezing of degrees of freedom at quantum-classical boundary,
    corresponding to decoherence and gravitational interaction.
    """
    
    def __init__(self, dimension: int = 27, freeze_threshold: float = 0.1):
        """
        Initialize freezing operator.
        
        Args:
            dimension: Dimension of operator space
            freeze_threshold: Threshold below which modes are frozen
        """
        self.dimension = dimension
        self.threshold = freeze_threshold
        self._build_operator()
    
    def _build_operator(self):
        """Build the freezing operator."""
        # Freezing operator suppresses low-energy modes
        # Using a smooth cutoff function
        eigenvalues = 1.0 / (1.0 + np.exp(-(np.arange(self.dimension) / self.dimension - 0.5) / self.threshold))
        self.operator = np.diag(eigenvalues)
        
        logger.debug(f"Freezing operator built with threshold {self.threshold}")
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply freezing to operator state.
        
        Args:
            state: Operator state vector or matrix
            
        Returns:
            Frozen state
        """
        if state.ndim == 1:
            return self.operator @ state
        else:
            return self.operator @ state @ self.operator.T


class ModularHamiltonian:
    """
    Modular Hamiltonian K = -ln(Δ) where Δ = C·T·U·F.
    
    The modular operator Δ is composed of:
    - C: Contraction operator
    - T: Triality operator
    - U: CTC rotation operator
    - F: Freezing operator
    
    Enforcing Δ(k)=1 induces rank-reducing projections corresponding to
    entanglement islands.
    """
    
    def __init__(self, 
                 dimension: int = 27,
                 contraction_strength: float = 1.0,
                 rotation_angle: float = np.pi/6,
                 freeze_threshold: float = 0.1):
        """
        Initialize modular Hamiltonian.
        
        Args:
            dimension: Dimension of operator space (27 for J_3(O))
            contraction_strength: Strength of contraction operator
            rotation_angle: CTC rotation angle
            freeze_threshold: Freezing threshold
        """
        self.dimension = dimension
        
        # Initialize component operators
        self.C = ContractionOperator(dimension, contraction_strength)
        self.T = TrialityOperator(dimension)
        self.U = CTCRotationOperator(dimension, rotation_angle)
        self.F = FreezingOperator(dimension, freeze_threshold)
        
        # Build modular operator Δ = C·T·U·F
        self._build_modular_operator()
        
        # Compute modular Hamiltonian K = -ln(Δ)
        self._compute_modular_hamiltonian()
        
        logger.info(f"Modular Hamiltonian constructed for {dimension}D operator space")
    
    def _build_modular_operator(self):
        """Build the modular operator Δ = C·T·U·F."""
        # Compose all operators
        # To ensure positive definiteness, we use Δ = Δ†·Δ where Δ† = F†·U†·T†·C†
        
        # Build forward composition
        Delta_forward = self.C.operator @ self.T.operator @ self.U.operator @ self.F.operator
        
        # Ensure positive definiteness by computing Δ†·Δ
        Delta = Delta_forward.T @ Delta_forward
        
        # Add small regularization for numerical stability
        Delta = Delta + 1e-8 * np.eye(self.dimension)
        
        # Verify positive definiteness
        eigenvals = np.linalg.eigvalsh(Delta)
        if np.any(eigenvals <= 0):
            logger.warning(f"Modular operator has non-positive eigenvalues. Min eigenvalue: {eigenvals.min():.4e}")
            # Add stronger regularization
            Delta = Delta + (abs(eigenvals.min()) + 1e-6) * np.eye(self.dimension)
            eigenvals = np.linalg.eigvalsh(Delta)
        
        self.Delta = Delta
        logger.debug(f"Modular operator Δ built, eigenvalue range: [{eigenvals.min():.4e}, {eigenvals.max():.4e}]")
    
    def _compute_modular_hamiltonian(self):
        """Compute modular Hamiltonian K = -ln(Δ)."""
        # Compute matrix logarithm
        # For Hermitian positive definite matrices, use eigendecomposition
        eigenvals, eigenvecs = eigh(self.Delta)
        
        # K = -ln(Δ) = -V ln(Λ) V^T
        log_eigenvals = -np.log(eigenvals)
        self.K = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T
        
        # Store spectrum
        self.spectrum = ModularSpectrum(
            eigenvalues=log_eigenvals,
            eigenvectors=eigenvecs,
            kappa=log_eigenvals.min()
        )
        
        logger.info(f"Modular Hamiltonian computed with spectral gap κ = {self.spectrum.kappa:.6f}")
    
    def get_spectral_gap(self) -> float:
        """
        Get the spectral gap κ = min eig(K).
        
        Returns:
            Spectral gap κ
        """
        return self.spectrum.kappa
    
    def compute_block_spectral_gaps(self, block_size: int = 3) -> np.ndarray:
        """
        Compute spectral gaps for each block of the operator space.
        
        This creates a heatmap showing where islands form (zero-gap regions).
        
        Args:
            block_size: Size of blocks to analyze
            
        Returns:
            Matrix of spectral gaps for each block
        """
        n_blocks = self.dimension // block_size
        gaps = np.zeros((n_blocks, n_blocks))
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                # Extract block
                i_start, i_end = i * block_size, (i + 1) * block_size
                j_start, j_end = j * block_size, (j + 1) * block_size
                block = self.K[i_start:i_end, j_start:j_end]
                
                # Compute minimum eigenvalue of block
                block_eigs = np.linalg.eigvalsh(block)
                gaps[i, j] = block_eigs.min()
        
        return gaps
    
    def find_islands(self, tolerance: float = 0.1) -> List[EntanglementIsland]:
        """
        Find entanglement islands where Δ(k) ≈ 1 (i.e., K(k) ≈ 0).
        
        These are rank-reduction projections in operator space.
        
        Args:
            tolerance: Tolerance for identifying islands
            
        Returns:
            List of entanglement islands
        """
        islands = []
        
        # Find eigenvalues near zero (where Δ ≈ 1)
        near_zero_mask = np.abs(self.spectrum.eigenvalues) < tolerance
        
        if np.any(near_zero_mask):
            # For each near-zero eigenvalue, create an island
            for idx in np.where(near_zero_mask)[0]:
                location = self.spectrum.eigenvectors[:, idx]
                
                # Projection operator onto this eigenspace
                projection = np.outer(location, location)
                
                # Rank reduction is 1 for each island
                rank_reduction = 1
                
                # Entropy contribution (placeholder - would need full calculation)
                entropy = np.log(self.dimension) / len(np.where(near_zero_mask)[0])
                
                island = EntanglementIsland(
                    location=location,
                    rank_reduction=rank_reduction,
                    projection=projection,
                    entropy_contribution=entropy
                )
                islands.append(island)
                
            logger.info(f"Found {len(islands)} entanglement islands")
        else:
            logger.info("No entanglement islands found within tolerance")
        
        return islands


class DeterministicPageCurve:
    """
    Deterministic Page curve from ergotropy-based entropy.
    
    Implements:
    - S(x) = ∫₀ˣ K(x') dx' (ergotropy-based entropy)
    - Modular nuclearity bounds: S(x) ≤ ln(dim H_R) ≤ N^(1/2)
    - Island saturation points
    """
    
    def __init__(self, modular_hamiltonian: ModularHamiltonian):
        """
        Initialize Page curve calculator.
        
        Args:
            modular_hamiltonian: Modular Hamiltonian instance
        """
        self.K = modular_hamiltonian
        self.dimension = modular_hamiltonian.dimension
    
    def modular_density(self, x: float) -> float:
        """
        Continuous modular Hamiltonian density K(x).
        
        Args:
            x: Position in operator space
            
        Returns:
            K(x) value
        """
        # Use a smooth interpolation of the eigenvalue distribution
        idx = int(x * (self.dimension - 1))
        idx = np.clip(idx, 0, self.dimension - 1)
        return self.K.spectrum.eigenvalues[idx]
    
    def entropy(self, x: float) -> float:
        """
        Compute ergotropy-based entropy S(x) = ∫₀ˣ K(x') dx'.
        
        Args:
            x: Upper integration limit (0 ≤ x ≤ 1)
            
        Returns:
            Entropy S(x)
        """
        # Integrate modular density from 0 to x
        x_vals = np.linspace(0, x, 100)
        K_vals = np.array([self.modular_density(xi) for xi in x_vals])
        
        # Trapezoidal integration
        S = trapezoid(K_vals, x_vals)
        return S
    
    def compute_page_curve(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Page curve S(x) for x ∈ [0, 1].
        
        Args:
            n_points: Number of points to compute
            
        Returns:
            (x_values, S_values) tuple
        """
        x_vals = np.linspace(0, 1, n_points)
        S_vals = np.array([self.entropy(x) for x in x_vals])
        
        logger.info(f"Page curve computed: S ranges from {S_vals.min():.4f} to {S_vals.max():.4f}")
        
        return x_vals, S_vals
    
    def nuclearity_bound(self, d_R: int = None) -> float:
        """
        Compute modular nuclearity bound.
        
        S(x) ≤ ln(d_R) ≤ N^(1/2)
        
        Args:
            d_R: Dimension of reduced Hilbert space (default: self.dimension)
            
        Returns:
            Upper bound on entropy
        """
        if d_R is None:
            d_R = self.dimension
        
        bound = np.log(d_R)
        return bound
    
    def verify_nuclearity(self, tolerance: float = 0.01) -> Dict:
        """
        Verify that Page curve satisfies nuclearity bounds.
        
        Args:
            tolerance: Tolerance for bound verification
            
        Returns:
            Verification results
        """
        x_vals, S_vals = self.compute_page_curve()
        bound = self.nuclearity_bound()
        
        max_entropy = S_vals.max()
        satisfies_bound = max_entropy <= bound + tolerance
        
        result = {
            'max_entropy': max_entropy,
            'nuclearity_bound': bound,
            'satisfies_bound': satisfies_bound,
            'margin': bound - max_entropy
        }
        
        logger.info(f"Nuclearity verification: S_max = {max_entropy:.4f}, bound = {bound:.4f}, satisfied = {satisfies_bound}")
        
        return result


class EntanglementMetric:
    """
    Entanglement metric induced by the Page curve.
    
    Computes:
    - g_ij(x) = ∂²S(x)/∂x^i∂x^j
    - Christoffel symbols Γ^i_jk
    - Geodesic equations
    """
    
    def __init__(self, page_curve: DeterministicPageCurve):
        """
        Initialize entanglement metric.
        
        Args:
            page_curve: DeterministicPageCurve instance
        """
        self.page_curve = page_curve
        self.dimension = page_curve.dimension
    
    def metric_tensor(self, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute metric tensor g_ij = ∂²S/∂x^i∂x^j at point x.
        
        Args:
            x: Point in operator space (vector)
            epsilon: Finite difference step
            
        Returns:
            Metric tensor g_ij
        """
        n = len(x)
        g = np.zeros((n, n))
        
        # Compute second derivatives using finite differences
        for i in range(n):
            for j in range(n):
                # Central difference for second derivative
                x_ipp = x.copy()
                x_ipp[i] += epsilon
                x_ipp[j] += epsilon
                
                x_ipm = x.copy()
                x_ipm[i] += epsilon
                x_ipm[j] -= epsilon
                
                x_imp = x.copy()
                x_imp[i] -= epsilon
                x_imp[j] += epsilon
                
                x_imm = x.copy()
                x_imm[i] -= epsilon
                x_imm[j] -= epsilon
                
                # Convert to scalar position for entropy calculation
                def to_scalar(vec):
                    return np.linalg.norm(vec) / np.linalg.norm(np.ones_like(vec))
                
                S_pp = self.page_curve.entropy(to_scalar(x_ipp))
                S_pm = self.page_curve.entropy(to_scalar(x_ipm))
                S_mp = self.page_curve.entropy(to_scalar(x_imp))
                S_mm = self.page_curve.entropy(to_scalar(x_imm))
                
                g[i, j] = (S_pp - S_pm - S_mp + S_mm) / (4 * epsilon**2)
        
        # Symmetrize
        g = (g + g.T) / 2
        
        return g
    
    def christoffel_symbols(self, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^k_ij at point x.
        
        Args:
            x: Point in operator space
            epsilon: Finite difference step
            
        Returns:
            Christoffel symbols (3D array)
        """
        n = len(x)
        Gamma = np.zeros((n, n, n))
        
        # Compute metric and its inverse
        g = self.metric_tensor(x, epsilon)
        g_inv = np.linalg.pinv(g)  # Use pseudo-inverse for stability
        
        # Compute metric derivatives
        dg = np.zeros((n, n, n))
        for k in range(n):
            x_plus = x.copy()
            x_plus[k] += epsilon
            x_minus = x.copy()
            x_minus[k] -= epsilon
            
            g_plus = self.metric_tensor(x_plus, epsilon)
            g_minus = self.metric_tensor(x_minus, epsilon)
            
            dg[:, :, k] = (g_plus - g_minus) / (2 * epsilon)
        
        # Compute Christoffel symbols: Γ^k_ij = (1/2) g^kl (∂g_il/∂x^j + ∂g_jl/∂x^i - ∂g_ij/∂x^l)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        Gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[i, l, j] + dg[j, l, i] - dg[i, j, l]
                        )
        
        return Gamma
    
    def geodesic_equation(self, t: float, state: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
        """
        Geodesic equation: d²x^i/dt² + Γ^i_jk dx^j/dt dx^k/dt = 0
        
        Args:
            t: Time parameter
            state: State vector [x, dx/dt] (concatenated)
            Gamma: Christoffel symbols
            
        Returns:
            Derivative [dx/dt, d²x/dt²]
        """
        n = len(state) // 2
        x = state[:n]
        v = state[n:]
        
        # Compute acceleration
        a = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    a[i] -= Gamma[i, j, k] * v[j] * v[k]
        
        return np.concatenate([v, a])
    
    def compute_geodesic(self, 
                        x0: np.ndarray, 
                        v0: np.ndarray,
                        t_span: Tuple[float, float],
                        n_points: int = 100) -> Dict:
        """
        Compute geodesic trajectory starting from x0 with velocity v0.
        
        Args:
            x0: Initial position
            v0: Initial velocity
            t_span: Time span (t_start, t_end)
            n_points: Number of points to compute
            
        Returns:
            Dictionary with trajectory data
        """
        # Compute Christoffel symbols at initial point
        Gamma = self.christoffel_symbols(x0)
        
        # Initial state
        state0 = np.concatenate([x0, v0])
        
        # Solve geodesic equation
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        def rhs(t, state):
            return self.geodesic_equation(t, state, Gamma)
        
        sol = solve_ivp(rhs, t_span, state0, t_eval=t_eval, method='RK45')
        
        n = len(x0)
        trajectory = sol.y[:n, :].T  # Positions
        velocities = sol.y[n:, :].T  # Velocities
        
        result = {
            't': sol.t,
            'trajectory': trajectory,
            'velocities': velocities,
            'success': sol.success
        }
        
        logger.info(f"Geodesic computed: {len(sol.t)} points, success = {sol.success}")
        
        return result
    
    def project_to_3d(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Project high-dimensional geodesic to 3D for visualization.
        
        Uses PCA to find the principal 3D projection.
        
        Args:
            trajectory: N × D trajectory array
            
        Returns:
            N × 3 projected trajectory
        """
        if trajectory.shape[1] <= 3:
            return trajectory
        
        try:
            # Try to use sklearn if available
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            trajectory_3d = pca.fit_transform(trajectory)
            variance_explained = pca.explained_variance_ratio_.sum()
            logger.info(f"3D projection explains {variance_explained:.2%} of variance")
            return trajectory_3d
        except ImportError:
            # Fallback to manual PCA implementation
            logger.info("sklearn not available, using manual PCA")
            
            # Center the data
            mean = trajectory.mean(axis=0)
            centered = trajectory - mean
            
            # Compute covariance matrix
            cov = np.cov(centered.T)
            
            # Get top 3 eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvals)[::-1]
            eigenvecs = eigenvecs[:, idx]
            
            # Project onto top 3 components
            trajectory_3d = centered @ eigenvecs[:, :3]
            
            variance_explained = eigenvals[idx[:3]].sum() / eigenvals.sum()
            logger.info(f"3D projection explains {variance_explained:.2%} of variance")
            
            return trajectory_3d


class JonesQuantumGravityResolution:
    """
    Main class for Jones Quantum Gravity Resolution framework.
    
    Integrates all components:
    - Modular Hamiltonian construction
    - Entanglement islands
    - Page curve
    - Geodesic flow
    - Visualizations
    """
    
    def __init__(self,
                 dimension: int = 27,
                 contraction_strength: float = 1.0,
                 rotation_angle: float = np.pi/6,
                 freeze_threshold: float = 0.1):
        """
        Initialize Jones Quantum Gravity Resolution framework.
        
        Args:
            dimension: Dimension of operator space (27 for J_3(O))
            contraction_strength: Contraction operator strength
            rotation_angle: CTC rotation angle
            freeze_threshold: Freezing operator threshold
        """
        logger.info("=" * 60)
        logger.info("Jones Quantum Gravity Resolution Initialization")
        logger.info("=" * 60)
        
        # Initialize Jordan algebra
        self.jordan_algebra = ExceptionalJordanAlgebra(dimension)
        
        # Build modular Hamiltonian
        self.modular_hamiltonian = ModularHamiltonian(
            dimension=dimension,
            contraction_strength=contraction_strength,
            rotation_angle=rotation_angle,
            freeze_threshold=freeze_threshold
        )
        
        # Initialize Page curve
        self.page_curve = DeterministicPageCurve(self.modular_hamiltonian)
        
        # Initialize entanglement metric
        self.metric = EntanglementMetric(self.page_curve)
        
        logger.info("Initialization complete")
    
    def analyze_spectral_structure(self) -> Dict:
        """
        Analyze the spectral structure of the modular Hamiltonian.
        
        Returns:
            Analysis results
        """
        logger.info("\nAnalyzing spectral structure...")
        
        kappa = self.modular_hamiltonian.get_spectral_gap()
        eigenvalues = self.modular_hamiltonian.spectrum.eigenvalues
        
        result = {
            'spectral_gap_kappa': kappa,
            'eigenvalue_range': (eigenvalues.min(), eigenvalues.max()),
            'eigenvalue_mean': eigenvalues.mean(),
            'eigenvalue_std': eigenvalues.std(),
            'dimension': self.jordan_algebra.dimension
        }
        
        logger.info(f"  Spectral gap κ = {kappa:.6f}")
        logger.info(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        
        return result
    
    def find_entanglement_islands(self, tolerance: float = 0.1) -> List[EntanglementIsland]:
        """
        Find entanglement islands in operator space.
        
        Args:
            tolerance: Tolerance for island identification
            
        Returns:
            List of islands
        """
        logger.info("\nFinding entanglement islands...")
        islands = self.modular_hamiltonian.find_islands(tolerance)
        
        logger.info(f"  Found {len(islands)} islands")
        for i, island in enumerate(islands):
            logger.info(f"    Island {i+1}: entropy contribution = {island.entropy_contribution:.4f}")
        
        return islands
    
    def compute_page_curve(self, n_points: int = 100) -> Dict:
        """
        Compute the deterministic Page curve.
        
        Args:
            n_points: Number of points
            
        Returns:
            Page curve data and verification results
        """
        logger.info("\nComputing Page curve...")
        
        x_vals, S_vals = self.page_curve.compute_page_curve(n_points)
        verification = self.page_curve.verify_nuclearity()
        
        result = {
            'x': x_vals,
            'S': S_vals,
            'verification': verification,
            'max_entropy': S_vals.max(),
            'saturation_point': x_vals[np.argmax(S_vals)]
        }
        
        logger.info(f"  Max entropy: {S_vals.max():.4f}")
        logger.info(f"  Saturation at x = {result['saturation_point']:.4f}")
        logger.info(f"  Nuclearity bound satisfied: {verification['satisfies_bound']}")
        
        return result
    
    def compute_geodesic_flow(self, 
                             x0: np.ndarray = None,
                             v0: np.ndarray = None,
                             t_span: Tuple[float, float] = (0, 1),
                             n_points: int = 50) -> Dict:
        """
        Compute geodesic flow in operator space.
        
        Args:
            x0: Initial position (default: random)
            v0: Initial velocity (default: random)
            t_span: Time span
            n_points: Number of points
            
        Returns:
            Geodesic data
        """
        logger.info("\nComputing geodesic flow...")
        
        # Default initial conditions
        if x0 is None:
            # Use first 3 dimensions for simplicity
            x0 = np.array([0.5, 0.5, 0.5])
        if v0 is None:
            v0 = np.array([0.1, 0.1, 0.1])
        
        geodesic = self.metric.compute_geodesic(x0, v0, t_span, n_points)
        
        # Project to 3D if needed
        if geodesic['trajectory'].shape[1] > 3:
            trajectory_3d = self.metric.project_to_3d(geodesic['trajectory'])
        else:
            trajectory_3d = geodesic['trajectory']
        
        result = {
            'geodesic': geodesic,
            'trajectory_3d': trajectory_3d,
            'success': geodesic['success']
        }
        
        logger.info(f"  Geodesic computation: {'successful' if geodesic['success'] else 'failed'}")
        
        return result
    
    def generate_visualizations(self, output_dir: str = None) -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Args:
            output_dir: Output directory for plots (default: current directory)
            
        Returns:
            Dictionary of generated plot filenames
        """
        logger.info("\nGenerating visualizations...")
        
        if output_dir is None:
            output_dir = "."
        
        plots = {}
        
        # 1. Spectral gap heatmap
        logger.info("  Creating spectral gap heatmap...")
        block_gaps = self.modular_hamiltonian.compute_block_spectral_gaps(block_size=3)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(block_gaps, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Spectral gap κ')
        plt.title('Modular Hamiltonian Spectral Gap Heatmap\n(Islands appear as zero-gap regions)')
        plt.xlabel('Block index j')
        plt.ylabel('Block index i')
        filename = f"{output_dir}/spectral_gap_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plots['heatmap'] = filename
        logger.info(f"    Saved: {filename}")
        
        # 2. Page curve
        logger.info("  Creating Page curve plot...")
        page_data = self.compute_page_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(page_data['x'], page_data['S'], 'b-', linewidth=2, label='S(x)')
        plt.axhline(y=page_data['verification']['nuclearity_bound'], 
                   color='r', linestyle='--', label='Nuclearity bound')
        plt.xlabel('x (position in operator space)')
        plt.ylabel('S(x) (ergotropy-based entropy)')
        plt.title('Deterministic Page Curve with Island Saturation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = f"{output_dir}/page_curve.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plots['page_curve'] = filename
        logger.info(f"    Saved: {filename}")
        
        # 3. Geodesic trajectory (3D)
        logger.info("  Creating geodesic trajectory plot...")
        geodesic_data = self.compute_geodesic_flow()
        
        if geodesic_data['success']:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            traj = geodesic_data['trajectory_3d']
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, label='Geodesic')
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=100, marker='o', label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=100, marker='s', label='End')
            
            ax.set_xlabel(r'$x^1$')
            ax.set_ylabel(r'$x^2$')
            ax.set_zlabel(r'$x^3$')
            ax.set_title('Geodesic Trajectory in Operator Space (3D Projection)')
            ax.legend()
            
            filename = f"{output_dir}/geodesic_trajectory_3d.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            plots['geodesic'] = filename
            logger.info(f"    Saved: {filename}")
        
        logger.info(f"Generated {len(plots)} visualizations")
        
        return plots
    
    def generate_full_analysis(self) -> Dict:
        """
        Generate complete analysis of the quantum gravity framework.
        
        Returns:
            Complete analysis results
        """
        logger.info("\n" + "=" * 60)
        logger.info("JONES QUANTUM GRAVITY RESOLUTION - FULL ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            'spectral_analysis': self.analyze_spectral_structure(),
            'islands': self.find_entanglement_islands(),
            'page_curve': self.compute_page_curve(),
            'geodesic_flow': self.compute_geodesic_flow(),
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nKey Results:")
        logger.info(f"  - Spectral gap κ: {results['spectral_analysis']['spectral_gap_kappa']:.6f}")
        logger.info(f"  - Entanglement islands: {len(results['islands'])}")
        logger.info(f"  - Max entropy: {results['page_curve']['max_entropy']:.4f}")
        logger.info(f"  - Nuclearity satisfied: {results['page_curve']['verification']['satisfies_bound']}")
        logger.info(f"  - Geodesic computation: {'successful' if results['geodesic_flow']['success'] else 'failed'}")
        
        return results
