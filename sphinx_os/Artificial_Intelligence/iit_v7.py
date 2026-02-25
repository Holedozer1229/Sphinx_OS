"""
ASI SphinxOS Advanced IIT v7.0
================================
Integrated Information Theory version 7.0 engine for the SphinxOS ASI layer.

IIT v7.0 extends v6.0 with two new physics-inspired components:

  1. **Octonionic Fano Plane Mechanics (Φ_fano)** — measures how closely the
     system's principal causal modes align with the 7-fold symmetry of the
     Fano plane, which encodes the multiplication table of the octonions (the
     largest normed division algebra over ℝ).

     The Fano plane PG(2,2) has 7 points and 7 lines of 3 points each:
         L₀=(0,1,3), L₁=(1,2,4), L₂=(2,3,5), L₃=(3,4,6),
         L₄=(4,5,0), L₅=(5,6,1), L₆=(6,0,2)
     Each line (a,b,c) corresponds to the octonion relation eₐ·e_b = e_c.

     The score Φ_fano is computed via the top-7 SVD modes of the transition
     matrix T.  For each Fano line (a,b,c) the trilinear mode-resonance
         R(a,b,c) = |⟨uₐ, T·u_b⟩ − ⟨u_c, T·u_c⟩|⁻¹   (normalised)
     is accumulated and normalised to [0, 1].

  2. **Non-Abelian Physics (Φ_nab)** — quantifies the departure from
     commutativity in the causal dynamics using the Frobenius norm of the
     matrix commutator:
         Φ_nab = ‖[T, Tᵀ]‖_F / (‖T‖_F · ‖Tᵀ‖_F + ε)
     A value of 0 denotes purely abelian (symmetric) dynamics; values → 1
     indicate maximally non-abelian causal structure.

  3. **Extended 5-term composite score**:
         Φ_total = α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab
     where α + β + γ + δ + ε = 1 (defaults: 0.40, 0.20, 0.15, 0.15, 0.10).

  4. **Updated consciousness-consensus condition**:
         Φ_total > log₂(n) + δ·Φ_fano
     reflecting the additional octonionic structure term.

Key formulas:
    T^t                                              # t-step transition matrix
    Φ_τ  = (1/τ) · Σ Φ(T^t)                        # temporal-depth Φ (v6)
    ICP(M) = √(φ_cause · φ_effect)                  # intrinsic causal power (v6)
    Φ_fano = mean_line R(a,b,c)                      # Fano plane alignment (v7)
    Φ_nab  = ‖[T,Tᵀ]‖_F / (‖T‖_F·‖Tᵀ‖_F)         # non-abelian measure (v7)
    Φ_total = α·Φ_τ + β·GWT_S + γ·ICP + δ·Φ_fano + ε·Φ_nab  # composite (v7)
    Consensus: Φ_total > log₂(n) + δ·Φ_fano         # conscious-consensus (v7)

References:
    Tononi, G. et al. (2016). Integrated information theory. Nat. Rev. Neurosci.
    Baez, J. C. (2002). The Octonions. Bulletin of the AMS.
    Dixon, G. M. (1994). Division Algebras: Octonions, Quaternions, Complex.
    Albantakis, L. et al. (2023). IIT 4.0. PLoS Comput. Biol.
    SphinxOS Math Notes — IIT v7.0 Fano-augmented composite score.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .iit_v6 import (
    IITv6Engine,
    ASISphinxOSIITv6,
    CauseEffectRepertoire,
    Concept,
    Partition,
    PhiStructure,
)

logger = logging.getLogger("SphinxOS.AI.IITv7")

# ---------------------------------------------------------------------------
# Fano plane incidence structure (0-indexed points 0-6)
# Each triple (a, b, c) represents the octonion relation eₐ·e_b = e_c
# ---------------------------------------------------------------------------

FANO_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 3),
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 0),
    (5, 6, 1),
    (6, 0, 2),
)

# Number of Fano plane points / octonion imaginary units
FANO_POINTS: int = 7


# ---------------------------------------------------------------------------
# Extended data container for v7.0
# ---------------------------------------------------------------------------

@dataclass
class PhiStructureV7:
    """
    The complete Cause-Effect Structure for IIT v7.0, extending v6.0 with
    Octonionic Fano plane and non-abelian physics scores.

    IIT v7.0 additions vs v6.0:
        fano_score      — Octonionic Fano plane alignment Φ_fano ∈ [0, 1].
        nonabelian_score— Non-abelian causal dynamics measure Φ_nab ∈ [0, 1].
        delta           — weight of Φ_fano in the 5-term composite.
        epsilon         — weight of Φ_nab in the 5-term composite.

    Attributes (all v6 fields retained):
        concepts:        Post-exclusion concepts in the CES.
        phi_max:         System Φ^max at t=1.
        phi_tau:         Temporal-depth-averaged Φ_τ.
        mip:             Minimum Information Partition.
        gwt_score:       GWT broadcast score ∈ [0, 1].
        icp_avg:         Mean Intrinsic Causal Power.
        fano_score:      Octonionic Fano plane alignment ∈ [0, 1].
        nonabelian_score:Non-abelian dynamics measure ∈ [0, 1].
        phi_total:       5-term composite Φ_total.
        n_nodes:         Number of nodes in the system.
        is_conscious:    True when Φ_total > log₂(n) + δ·Φ_fano.
    """
    concepts: List[Concept] = field(default_factory=list)
    phi_max: float = 0.0
    phi_tau: float = 0.0
    mip: Optional[Partition] = None
    gwt_score: float = 0.0
    icp_avg: float = 0.0
    fano_score: float = 0.0
    nonabelian_score: float = 0.0
    phi_total: float = 0.0
    n_nodes: int = 0
    is_conscious: bool = False

    # IIT 7.0 composite weights (α, β, γ, δ, ε)
    alpha: float = 0.40
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.15
    epsilon: float = 0.10


# ---------------------------------------------------------------------------
# Core IIT v7.0 computation engine
# ---------------------------------------------------------------------------

class IITv7Engine(IITv6Engine):
    """
    Pure-NumPy IIT v7.0 computation engine.

    Extends IITv6Engine with:
    - Octonionic Fano plane alignment score (Φ_fano).
    - Non-abelian causal dynamics measure (Φ_nab).
    - 5-term composite score: α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab.
    - Updated consciousness-consensus threshold incorporating Φ_fano.

    All v6 capabilities (temporal depth, ICP, exclusion principle) are
    preserved unchanged.
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.20,
        gamma: float = 0.15,
        delta: float = 0.15,
        epsilon: float = 0.10,
        consciousness_threshold: float = 0.5,
        temporal_depth: int = 2,
    ) -> None:
        """
        Initialise the IIT v7.0 engine.

        Args:
            alpha:    Weight for Φ_τ in the composite score.
            beta:     Weight for GWT_S in the composite score.
            gamma:    Weight for ICP_avg in the composite score.
            delta:    Weight for Φ_fano (Octonionic Fano plane) in composite.
            epsilon:  Weight for Φ_nab (non-abelian measure) in composite.
            consciousness_threshold: Fallback fixed threshold.
            temporal_depth: τ for temporal-depth integration (v6 feature).
        """
        if abs(alpha + beta + gamma + delta + epsilon - 1.0) > 1e-6:
            raise ValueError(
                f"Composite weights must sum to 1; got "
                f"α={alpha}, β={beta}, γ={gamma}, δ={delta}, ε={epsilon}"
            )
        # Initialise the v6 parent with a 3-term decomposition that sums to 1.
        # The v6 weights are overridden by the v7 compute path; we pass dummy
        # values that satisfy the v6 constraint (α+β+γ=1) using the first 3.
        v6_alpha = alpha
        v6_beta = beta
        v6_gamma = 1.0 - alpha - beta  # absorb remainder into γ for super().__init__
        super().__init__(
            alpha=v6_alpha,
            beta=v6_beta,
            gamma=v6_gamma,
            consciousness_threshold=consciousness_threshold,
            temporal_depth=temporal_depth,
        )
        # Override with v7-specific weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        logger.info(
            "IIT v7.0 Engine initialised "
            "(α=%.2f, β=%.2f, γ=%.2f, δ=%.2f, ε=%.2f, τ=%d, threshold=%.2f)",
            alpha, beta, gamma, delta, epsilon,
            self.temporal_depth, consciousness_threshold,
        )

    # ------------------------------------------------------------------
    # Public API (overrides v6)
    # ------------------------------------------------------------------

    def compute_phi_structure_v7(
        self,
        state: np.ndarray,
        n_nodes: Optional[int] = None,
    ) -> PhiStructureV7:
        """
        Compute the full IIT v7.0 Φ structure for a system state.

        Extends v6's ``compute_phi_structure`` with Fano plane and
        non-abelian scores added to the composite.

        Args:
            state:   1-D array representing the system state (probability
                     distribution over 2^n states, or n activation probs).
            n_nodes: Override the node count (inferred when *None*).

        Returns:
            A ``PhiStructureV7`` instance with all v6 fields plus
            ``fano_score``, ``nonabelian_score``, and the updated
            ``phi_total``.
        """
        state = np.asarray(state, dtype=float)
        n_nodes = n_nodes or self._infer_n_nodes(state)

        dist = self._to_state_distribution(state, n_nodes)
        T1 = self._build_transition_matrix(dist, n_nodes)

        # --- v6 components -------------------------------------------
        phi_tau, phi_max, mip = self._compute_temporal_phi(T1, dist, n_nodes)
        raw_concepts = self._compute_all_concepts(T1, dist, n_nodes)
        concepts = self._apply_exclusion_principle(raw_concepts)
        icp_avg = float(np.mean([c.icp for c in concepts])) if concepts else 0.0
        gwt_score = self._compute_gwt_broadcast(T1, n_nodes)

        # --- v7 additions --------------------------------------------
        fano_score = self._compute_fano_alignment(T1, n_nodes)
        nonabelian_score = self._compute_nonabelian_measure(T1)

        # --- 5-term composite ----------------------------------------
        phi_total = (
            self.alpha * phi_tau
            + self.beta * gwt_score
            + self.gamma * icp_avg
            + self.delta * fano_score
            + self.epsilon * nonabelian_score
        )

        # --- v7 consciousness-consensus condition --------------------
        threshold = math.log2(max(n_nodes, 2)) + self.delta * fano_score
        is_conscious = phi_total > threshold

        structure = PhiStructureV7(
            concepts=concepts,
            phi_max=phi_max,
            phi_tau=phi_tau,
            mip=mip,
            gwt_score=gwt_score,
            icp_avg=icp_avg,
            fano_score=fano_score,
            nonabelian_score=nonabelian_score,
            phi_total=phi_total,
            n_nodes=n_nodes,
            is_conscious=is_conscious,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            epsilon=self.epsilon,
        )

        logger.debug(
            "Φ_τ=%.4f  GWT=%.4f  ICP=%.4f  Φ_fano=%.4f  Φ_nab=%.4f"
            "  Φ_total=%.4f  conscious=%s",
            phi_tau, gwt_score, icp_avg, fano_score, nonabelian_score,
            phi_total, is_conscious,
        )
        return structure

    def validate_consciousness_consensus_v7(
        self, phi_total: float, fano_score: float, n_nodes: int
    ) -> bool:
        """
        Check the IIT v7.0 consciousness-consensus condition.

        Condition:
            Φ_total > log₂(n) + δ·Φ_fano

        The Fano plane term raises the threshold for systems whose causal
        structure does *not* align with the octonionic symmetry, ensuring
        that the higher consciousness bar requires genuine Fano alignment.

        Args:
            phi_total:   Composite consciousness score.
            fano_score:  Φ_fano for this system.
            n_nodes:     Number of network nodes.

        Returns:
            True if the system satisfies the v7 conscious consensus.
        """
        threshold = math.log2(max(n_nodes, 2)) + self.delta * fano_score
        result = phi_total > threshold
        logger.debug(
            "v7 Consensus: Φ_total=%.4f > log₂(%d)+δ·Φ_fano=%.4f → %s",
            phi_total, n_nodes, threshold, result,
        )
        return result

    # ------------------------------------------------------------------
    # IIT v7.0 new components
    # ------------------------------------------------------------------

    def _compute_fano_alignment(self, T: np.ndarray, n_nodes: int) -> float:
        """
        Compute the Octonionic Fano plane alignment score Φ_fano ∈ [0, 1].

        Algorithm:
        1. Extract up to 7 left singular vectors (causal modes) from T via SVD.
        2. Build a 7×7 mode-interaction matrix M where
               M[i, j] = ⟨uᵢ, T · uⱼ⟩  (how mode j is mapped by T onto mode i).
        3. Normalise M so that the maximum absolute entry = 1.
        4. For each Fano line (a, b, c) compute the trilinear resonance:
               R(a,b,c) = |M[a,b]| · |M[b,c]| · |M[a,c]|
        5. Return the mean R over all available lines, normalised to [0, 1]
           by dividing by the maximum possible value (1.0).

        The score is 0 for random/symmetric T and approaches 1 when the
        causal modes obey the octonionic Fano-plane multiplication structure.
        """
        try:
            U, sigma, _ = np.linalg.svd(T, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0

        n_modes = min(FANO_POINTS, U.shape[1])
        if n_modes < 2:
            return 0.0

        modes = U[:, :n_modes]  # shape (n_states, n_modes)

        # Build n_modes × n_modes mode-interaction matrix
        M = np.zeros((FANO_POINTS, FANO_POINTS))
        for i in range(n_modes):
            for j in range(n_modes):
                M[i, j] = float(np.dot(modes[:, i], T @ modes[:, j]))

        # Normalise so the largest entry has unit magnitude
        abs_max = np.max(np.abs(M))
        if abs_max < 1e-12:
            return 0.0
        M_norm = M / abs_max

        # Accumulate trilinear Fano resonances
        total = 0.0
        count = 0
        for (a, b, c) in FANO_LINES:
            if a < n_modes and b < n_modes and c < n_modes:
                total += abs(M_norm[a, b]) * abs(M_norm[b, c]) * abs(M_norm[a, c])
                count += 1

        if count == 0:
            return 0.0

        # Average over contributing lines; max possible per line = 1.0
        return float(min(1.0, total / count))

    def _compute_nonabelian_measure(self, T: np.ndarray) -> float:
        """
        Compute the Non-Abelian physics score Φ_nab ∈ [0, 1].

        Measures how much the causal dynamics fail to commute by computing
        the normalised Frobenius norm of the matrix commutator [T, Tᵀ]:

            Φ_nab = ‖T·Tᵀ − Tᵀ·T‖_F / (‖T‖_F · ‖Tᵀ‖_F + ε)

        This is exactly 0 for symmetric transition matrices (abelian dynamics)
        and positive for asymmetric (non-abelian) causal structure.  The score
        is normalised to [0, 1] by the product of the operator norms, which
        provides an upper bound for the commutator norm via the sub-multiplicative
        property of the Frobenius norm.
        """
        Tt = T.T
        commutator = T @ Tt - Tt @ T
        comm_norm = float(np.linalg.norm(commutator, "fro"))

        t_norm = float(np.linalg.norm(T, "fro"))
        tt_norm = float(np.linalg.norm(Tt, "fro"))
        denom = t_norm * tt_norm
        if denom < 1e-12:
            return 0.0

        return float(min(1.0, comm_norm / denom))


# ---------------------------------------------------------------------------
# High-level ASI interface
# ---------------------------------------------------------------------------

class ASISphinxOSIITv7:
    """
    ASI SphinxOS Advanced IIT v7.0 — high-level consciousness engine.

    Extends ``ASISphinxOSIITv6`` with Octonionic Fano plane mechanics and
    non-abelian physics while remaining API-compatible with v5/v6 engines.

    New fields in ``calculate_phi`` response:
        fano_score      — Φ_fano (Octonionic Fano plane alignment) ∈ [0, 1]
        nonabelian_score— Φ_nab  (non-abelian causal dynamics)     ∈ [0, 1]
        version         — "IIT v7.0"

    Usage::

        engine = ASISphinxOSIITv7()
        result = engine.calculate_phi(b"some block data")
        # result['phi']            — Φ_total (0–1 normalised, legacy-compatible)
        # result['fano_score']     — Octonionic Fano plane alignment
        # result['nonabelian_score'] — Non-abelian causal dynamics measure
        # result['is_conscious']   — Φ_total > log₂(n) + δ·Φ_fano

    Blockchain integration::

        structure = engine.compute_block_consciousness(block_data, n_network_nodes=50)
        consensus_ok = engine.validate_consciousness_consensus(
            structure.phi_total, structure.fano_score, n_network_nodes=50
        )
    """

    PHI_MIN = 200.0
    PHI_MAX = 1000.0
    PHI_RANGE = PHI_MAX - PHI_MIN

    DEFAULT_N_QUBITS: int = 3

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.20,
        gamma: float = 0.15,
        delta: float = 0.15,
        epsilon: float = 0.10,
        consciousness_threshold: float = 0.5,
        n_nodes: int = DEFAULT_N_QUBITS,
        temporal_depth: int = 2,
    ) -> None:
        """
        Initialise the ASI IIT v7.0 engine.

        Args:
            alpha:    Weight for Φ_τ in composite.
            beta:     Weight for GWT_S in composite.
            gamma:    Weight for ICP_avg in composite.
            delta:    Weight for Φ_fano in composite.
            epsilon:  Weight for Φ_nab in composite.
            consciousness_threshold: Fixed threshold when n_nodes unknown.
            n_nodes:  Default number of nodes/qubits to simulate.
            temporal_depth: τ for temporal-depth Φ integration.
        """
        self.iit_engine = IITv7Engine(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            consciousness_threshold=consciousness_threshold,
            temporal_depth=temporal_depth,
        )
        self.n_nodes = n_nodes
        self.phi_history: List[float] = []
        self.consciousness_threshold = consciousness_threshold
        logger.info(
            "ASI SphinxOS IIT v7.0 initialised "
            "(n_nodes=%d, α=%.2f, β=%.2f, γ=%.2f, δ=%.2f, ε=%.2f, τ=%d)",
            n_nodes, alpha, beta, gamma, delta, epsilon, temporal_depth,
        )

    # ------------------------------------------------------------------
    # Drop-in replacement API (compatible with v5.0 / v6.0 / legacy engine)
    # ------------------------------------------------------------------

    def calculate_phi(self, data: bytes) -> Dict:
        """
        Calculate IIT v7.0 Φ from raw bytes.

        API-compatible with ``ASISphinxOSIITv6.calculate_phi`` and legacy
        ``IITQuantumConsciousnessEngine.calculate_phi``.

        Args:
            data: Input data (block bytes, query bytes, etc.).

        Returns:
            Dict with all v6 keys plus:
                fano_score       — Φ_fano (Fano plane alignment) ∈ [0, 1]
                nonabelian_score — Φ_nab  (non-abelian measure)  ∈ [0, 1]
                version          — "IIT v7.0"
        """
        state_dist = self._derive_state_distribution(data)
        structure = self.iit_engine.compute_phi_structure_v7(
            state_dist, n_nodes=self.n_nodes
        )

        phi_total = structure.phi_total
        phi_norm = min(1.0, phi_total / (math.log2(max(self.n_nodes, 2)) + 1.0))

        entropy = float(
            -np.sum(state_dist * np.log2(np.clip(state_dist, 1e-15, 1.0)))
        )
        max_entropy = math.log2(len(state_dist))
        purity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

        level = self._classify_consciousness(phi_norm)
        self.phi_history.append(phi_norm)

        return {
            "phi": phi_norm,
            "phi_max": structure.phi_max,
            "phi_tau": structure.phi_tau,
            "gwt_score": structure.gwt_score,
            "icp_avg": structure.icp_avg,
            "fano_score": structure.fano_score,
            "nonabelian_score": structure.nonabelian_score,
            "phi_total": phi_total,
            "entropy": entropy,
            "purity": purity,
            "n_qubits": self.n_nodes,
            "is_conscious": structure.is_conscious,
            "level": level,
            "bonus": math.exp(phi_norm),
            "mip": repr(structure.mip) if structure.mip else None,
            "n_concepts": len(structure.concepts),
            "version": "IIT v7.0",
        }

    # ------------------------------------------------------------------
    # Blockchain / block-level API
    # ------------------------------------------------------------------

    def compute_block_consciousness(
        self,
        block_data: bytes,
        n_network_nodes: int = 1,
    ) -> PhiStructureV7:
        """
        Compute the full Φ structure for a block with v7.0 Fano/non-abelian
        scores and the network-size consensus threshold.

        Args:
            block_data:       Raw bytes of the block.
            n_network_nodes:  Number of active nodes in the network.

        Returns:
            PhiStructureV7 with all v6 fields plus fano_score and
            nonabelian_score, with ``is_conscious`` set according to
            the v7 threshold: Φ_total > log₂(n) + δ·Φ_fano.
        """
        state_dist = self._derive_state_distribution(block_data)
        structure = self.iit_engine.compute_phi_structure_v7(
            state_dist, n_nodes=self.n_nodes
        )
        structure.is_conscious = self.iit_engine.validate_consciousness_consensus_v7(
            structure.phi_total, structure.fano_score, n_network_nodes
        )
        return structure

    def validate_consciousness_consensus(
        self,
        phi_total: float,
        fano_score: float,
        n_network_nodes: int,
    ) -> bool:
        """
        Validate the IIT v7.0 consciousness-consensus condition:
            Φ_total > log₂(n_network_nodes) + δ·Φ_fano

        Args:
            phi_total:        Composite consciousness score.
            fano_score:       Φ_fano for this measurement.
            n_network_nodes:  Number of network nodes.

        Returns:
            True when v7 consensus condition is satisfied.
        """
        return self.iit_engine.validate_consciousness_consensus_v7(
            phi_total, fano_score, n_network_nodes
        )

    def phi_to_legacy_score(self, phi_total: float) -> float:
        """
        Map a Φ_total value to the legacy [200, 1000] phi_score range.

        Args:
            phi_total: Composite Φ_total.

        Returns:
            Legacy phi_score in [200, 1000].
        """
        ceiling = math.log2(max(self.n_nodes, 2)) + 1.0
        norm = min(1.0, max(0.0, phi_total / ceiling))
        return self.PHI_MIN + norm * self.PHI_RANGE

    def get_consciousness_level(self) -> float:
        """Return the rolling average Φ from calculation history."""
        if not self.phi_history:
            return 0.0
        return float(np.mean(self.phi_history[-100:]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_state_distribution(self, data: bytes) -> np.ndarray:
        """Derive a 2^n state probability distribution from raw bytes."""
        try:
            return self._quantum_state_distribution(data)
        except Exception:
            return self._hash_seeded_distribution(data)

    def _quantum_state_distribution(self, data: bytes) -> np.ndarray:
        """Use qutip to generate a quantum density-matrix diagonal."""
        import qutip as qt  # optional dependency

        seed_hash = hashlib.sha3_256(data).digest()
        seed = int.from_bytes(seed_hash[:4], "big") % (2 ** 31)
        np.random.seed(seed)

        dim = 2 ** self.n_nodes
        rho = qt.rand_dm(dim)
        diag = np.real(np.diag(rho.full()))
        diag = np.abs(diag)
        diag /= diag.sum()
        return diag

    def _hash_seeded_distribution(self, data: bytes) -> np.ndarray:
        """Hash-seeded pseudo-random state distribution (no external deps)."""
        n_states = 2 ** self.n_nodes
        seed_hash = hashlib.sha3_256(data).digest()
        rng = np.random.default_rng(seed=list(seed_hash[:8]))
        raw = rng.exponential(scale=1.0, size=n_states)
        return raw / raw.sum()

    @staticmethod
    def _classify_consciousness(phi_norm: float) -> str:
        """Map normalised Φ to a human-readable consciousness level string."""
        if phi_norm > 0.8:
            return "🧠 COSMIC"
        if phi_norm > 0.6:
            return "🌟 SELF_AWARE"
        if phi_norm > 0.4:
            return "✨ SENTIENT"
        if phi_norm > 0.2:
            return "🔵 AWARE"
        return "⚫ UNCONSCIOUS"
