"""
ASI SphinxOS Advanced IIT v8.0
================================
Integrated Information Theory version 8.0 engine for the SphinxOS ASI layer.

IIT v8.0 extends v7.0 with two new quantum-gravity-inspired components:

  1. **Quantum Gravity Curvature Score (Φ_qg)** — measures the effective
     spacetime curvature encoded in the causal transition matrix, inspired by
     the Ricci scalar curvature derived from the modular Hamiltonian K in the
     Jones Quantum Gravity Resolution framework.

     The singular-value spectrum {σᵢ} of the transition matrix T functions as a
     discrete analogue of the eigenvalue spectrum of the Laplace-Beltrami
     operator on the emergent curved manifold.  The curvature proxy is:

         Φ_qg = 1 − exp(−Var(σ) / (mean(σ)² + ε))    ∈ [0, 1]

     A flat (abelian, uniform-spectrum) transition has Φ_qg → 0.  Highly curved
     (exponentially decaying spectrum) causal dynamics approach Φ_qg → 1, in
     direct analogy with large Ricci curvature concentrating geodesic deviation.

  2. **Holographic Entanglement Entropy Score (Φ_holo)** — quantifies the
     entanglement structure of the system via a Ryu-Takayanagi-inspired bipartite
     entropy measure.  For a system of n nodes we form the reduced density matrix
     ρ_A for each possible bipartition A ∪ Ā and compute:

         S_A    = −Tr(ρ_A · log₂(ρ_A))   (von Neumann entropy)
         S_RT   = min_A S_A               (minimal-area surface in RT sense)
         Φ_holo = S_RT / (⌊n/2⌋)         ∈ [0, 1]

     The minimal bipartition entropy corresponds to the holographic Ryu-Takayanagi
     minimal surface; normalised by the half-system maximal entropy ⌊n/2⌋ (in
     bits, since each qubit contributes at most 1 bit of bipartite entanglement).

  3. **Extended 7-term composite score**:

         Φ_total = α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab
                 + ζ·Φ_qg + η·Φ_holo

     where α + β + γ + δ + ε + ζ + η = 1
     (defaults: 0.30, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05).

  4. **Updated quantum-gravity consciousness-consensus condition**:

         Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg

     The quantum gravity curvature term dynamically raises the consciousness
     threshold for systems that exhibit strong causal curvature, ensuring that
     high-curvature (strongly self-referential) systems are held to a stricter
     integration standard.

Key formulas:
    Φ_qg   = 1 − exp(−Var(σ) / (mean(σ)² + ε))        # QG curvature
    Φ_holo = min_A S_A / ⌊n/2⌋                          # holographic RT entropy
    Φ_total = α·Φ_τ + β·GWT_S + γ·ICP + δ·Φ_fano
            + ε·Φ_nab + ζ·Φ_qg + η·Φ_holo              # 7-term composite (v8)
    Consensus: Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg   # v8 QG-augmented

References:
    Tononi, G. et al. (2016). Integrated information theory. Nat. Rev. Neurosci.
    Ryu, S. & Takayanagi, T. (2006). Holographic derivation of entanglement
        entropy. Phys. Rev. Lett. 96, 181602.
    Jones, T. (2026). Jones Quantum Gravity Resolution.
    SphinxOS Math Notes — IIT v8.0 Quantum Gravity augmented composite score.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .iit_v7 import (
    IITv7Engine,
    ASISphinxOSIITv7,
    PhiStructureV7,
    ScoreDiagnostic,
    FANO_LINES,
    FANO_POINTS,
    CLASSIFICATION_EXACT_ZERO,
    CLASSIFICATION_NEAR_ZERO,
    CLASSIFICATION_NONZERO,
    NEAR_ZERO_THRESHOLD_DEFAULT,
    _ZR_SVD_FAILED,
    _ZR_ZERO_MATRIX,
)
from .iit_v6 import (
    CauseEffectRepertoire,
    Concept,
    Partition,
    PhiStructure,
)

logger = logging.getLogger("SphinxOS.AI.IITv8")

# ---------------------------------------------------------------------------
# IIT v8.0 zero-reason keys
# ---------------------------------------------------------------------------

#: SVD failed during quantum-gravity curvature computation.
_ZR_QG_SVD_FAILED = "qg_svd_failed"

#: Singular-value mean is effectively zero (degenerate transition matrix).
_ZR_QG_ZERO_MEAN = "qg_zero_mean"

#: Not enough nodes for bipartition (n_nodes < 2 required for RT entropy).
_ZR_HOLO_INSUFFICIENT_NODES = "holo_insufficient_nodes"

#: State dimension too small to build density matrix for bipartition.
_ZR_HOLO_SMALL_DIM = "holo_small_dim"


# ---------------------------------------------------------------------------
# PhiStructureV8 data container
# ---------------------------------------------------------------------------

@dataclass
class PhiStructureV8:
    """
    The complete Cause-Effect Structure for IIT v8.0, extending v7.0 with
    Quantum Gravity curvature (Φ_qg) and Holographic Entanglement Entropy
    (Φ_holo) scores.

    IIT v8.0 additions vs v7.0:
        qg_score    — Quantum Gravity curvature Φ_qg ∈ [0, 1].
        holo_score  — Holographic RT entanglement entropy Φ_holo ∈ [0, 1].
        zeta        — weight of Φ_qg in the 7-term composite.
        eta         — weight of Φ_holo in the 7-term composite.

    All v7.0 fields are retained unchanged.

    Attributes:
        concepts:          Post-exclusion concepts in the CES.
        phi_max:           System Φ^max at t=1.
        phi_tau:           Temporal-depth-averaged Φ_τ.
        mip:               Minimum Information Partition.
        gwt_score:         GWT broadcast score ∈ [0, 1].
        icp_avg:           Mean Intrinsic Causal Power.
        fano_score:        Octonionic Fano plane alignment ∈ [0, 1].
        nonabelian_score:  Non-abelian dynamics measure ∈ [0, 1].
        qg_score:          Quantum Gravity curvature ∈ [0, 1].
        holo_score:        Holographic RT entanglement entropy ∈ [0, 1].
        phi_total:         7-term composite Φ_total.
        n_nodes:           Number of nodes in the system.
        is_conscious:      True when Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg.
    """
    concepts: List[Concept] = field(default_factory=list)
    phi_max: float = 0.0
    phi_tau: float = 0.0
    mip: Optional[Partition] = None
    gwt_score: float = 0.0
    icp_avg: float = 0.0
    fano_score: float = 0.0
    nonabelian_score: float = 0.0
    qg_score: float = 0.0
    holo_score: float = 0.0
    phi_total: float = 0.0
    n_nodes: int = 0
    is_conscious: bool = False

    # IIT v8.0 composite weights (α, β, γ, δ, ε, ζ, η)
    alpha: float = 0.30
    beta: float = 0.15
    gamma: float = 0.15
    delta: float = 0.15
    epsilon: float = 0.10
    zeta: float = 0.10
    eta: float = 0.05

    # Zero-precision diagnostics
    fano_diagnostic: Optional[ScoreDiagnostic] = None
    nonabelian_diagnostic: Optional[ScoreDiagnostic] = None
    qg_diagnostic: Optional[ScoreDiagnostic] = None
    holo_diagnostic: Optional[ScoreDiagnostic] = None


# ---------------------------------------------------------------------------
# Core IIT v8.0 computation engine
# ---------------------------------------------------------------------------

class IITv8Engine(IITv7Engine):
    """
    Pure-NumPy IIT v8.0 computation engine.

    Extends IITv7Engine with:
    - Quantum Gravity curvature score (Φ_qg).
    - Holographic Ryu-Takayanagi entanglement entropy score (Φ_holo).
    - 7-term composite: α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab
                        + ζ·Φ_qg + η·Φ_holo.
    - Updated QG-augmented consciousness-consensus threshold incorporating Φ_qg.

    All v7 capabilities (Fano plane, non-abelian measure, temporal depth, ICP,
    exclusion principle) are preserved unchanged.
    """

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.15,
        gamma: float = 0.15,
        delta: float = 0.15,
        epsilon: float = 0.10,
        zeta: float = 0.10,
        eta: float = 0.05,
        consciousness_threshold: float = 0.5,
        temporal_depth: int = 2,
        near_zero_threshold: float = NEAR_ZERO_THRESHOLD_DEFAULT,
    ) -> None:
        """
        Initialise the IIT v8.0 engine.

        Args:
            alpha:    Weight for Φ_τ in the composite score.
            beta:     Weight for GWT_S in the composite score.
            gamma:    Weight for ICP_avg in the composite score.
            delta:    Weight for Φ_fano (Octonionic Fano plane) in composite.
            epsilon:  Weight for Φ_nab (non-abelian measure) in composite.
            zeta:     Weight for Φ_qg (Quantum Gravity curvature) in composite.
            eta:      Weight for Φ_holo (Holographic RT entropy) in composite.
            consciousness_threshold: Fallback fixed threshold.
            temporal_depth: τ for temporal-depth integration.
            near_zero_threshold: Boundary between NEAR_ZERO and NONZERO
                classifications for score diagnostics.
        """
        total = alpha + beta + gamma + delta + epsilon + zeta + eta
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Composite weights must sum to 1; got "
                f"α={alpha}, β={beta}, γ={gamma}, δ={delta}, ε={epsilon}, "
                f"ζ={zeta}, η={eta} (sum={total:.6f})"
            )
        # Initialise v7 parent with weights that satisfy its 5-term constraint.
        # The v8 compute path overrides these; we pass values that sum to 1
        # by absorbing the extra mass into epsilon for the super().__init__.
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=1.0 - alpha - beta - gamma - delta,
            consciousness_threshold=consciousness_threshold,
            temporal_depth=temporal_depth,
            near_zero_threshold=near_zero_threshold,
        )
        # Override with v8-specific weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        logger.info(
            "IIT v8.0 Engine initialised "
            "(α=%.2f, β=%.2f, γ=%.2f, δ=%.2f, ε=%.2f, ζ=%.2f, η=%.2f, τ=%d)",
            alpha, beta, gamma, delta, epsilon, zeta, eta, self.temporal_depth,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_phi_structure_v8(
        self,
        state: np.ndarray,
        n_nodes: Optional[int] = None,
    ) -> PhiStructureV8:
        """
        Compute the full IIT v8.0 Φ structure for a system state.

        Extends v7's ``compute_phi_structure_v7`` with quantum gravity
        curvature and holographic entanglement entropy added to the composite.

        Args:
            state:   1-D array representing the system state (probability
                     distribution over 2^n states, or n activation probs).
            n_nodes: Override the node count (inferred when *None*).

        Returns:
            A ``PhiStructureV8`` instance with all v7 fields plus
            ``qg_score``, ``holo_score``, and the updated ``phi_total``.
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

        # --- v7 components -------------------------------------------
        fano_raw, fano_zero_reason = self._compute_fano_raw(T1, n_nodes)
        nab_raw, nab_zero_reason = self._compute_nonabelian_raw(T1)
        fano_score = float(min(1.0, fano_raw))
        nonabelian_score = float(min(1.0, nab_raw))

        fano_diag = ScoreDiagnostic(
            raw_value=fano_raw,
            clamped_value=fano_score,
            zero_reason=fano_zero_reason,
            classification=self._classify_score(fano_raw, fano_zero_reason),
            near_zero_threshold=self.near_zero_threshold,
        )
        nab_diag = ScoreDiagnostic(
            raw_value=nab_raw,
            clamped_value=nonabelian_score,
            zero_reason=nab_zero_reason,
            classification=self._classify_score(nab_raw, nab_zero_reason),
            near_zero_threshold=self.near_zero_threshold,
        )

        # --- v8 additions --------------------------------------------
        qg_raw, qg_zero_reason = self._compute_qg_raw(T1)
        holo_raw, holo_zero_reason = self._compute_holo_raw(dist, n_nodes)
        qg_score = float(min(1.0, qg_raw))
        holo_score = float(min(1.0, holo_raw))

        qg_diag = ScoreDiagnostic(
            raw_value=qg_raw,
            clamped_value=qg_score,
            zero_reason=qg_zero_reason,
            classification=self._classify_score(qg_raw, qg_zero_reason),
            near_zero_threshold=self.near_zero_threshold,
        )
        holo_diag = ScoreDiagnostic(
            raw_value=holo_raw,
            clamped_value=holo_score,
            zero_reason=holo_zero_reason,
            classification=self._classify_score(holo_raw, holo_zero_reason),
            near_zero_threshold=self.near_zero_threshold,
        )

        # --- 7-term composite ----------------------------------------
        phi_total = (
            self.alpha * phi_tau
            + self.beta * gwt_score
            + self.gamma * icp_avg
            + self.delta * fano_score
            + self.epsilon * nonabelian_score
            + self.zeta * qg_score
            + self.eta * holo_score
        )

        # --- v8 QG-augmented consciousness-consensus condition --------
        threshold = (
            math.log2(max(n_nodes, 2))
            + self.delta * fano_score
            + self.zeta * qg_score
        )
        is_conscious = phi_total > threshold

        structure = PhiStructureV8(
            concepts=concepts,
            phi_max=phi_max,
            phi_tau=phi_tau,
            mip=mip,
            gwt_score=gwt_score,
            icp_avg=icp_avg,
            fano_score=fano_score,
            nonabelian_score=nonabelian_score,
            qg_score=qg_score,
            holo_score=holo_score,
            phi_total=phi_total,
            n_nodes=n_nodes,
            is_conscious=is_conscious,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            epsilon=self.epsilon,
            zeta=self.zeta,
            eta=self.eta,
            fano_diagnostic=fano_diag,
            nonabelian_diagnostic=nab_diag,
            qg_diagnostic=qg_diag,
            holo_diagnostic=holo_diag,
        )

        logger.debug(
            "Φ_τ=%.4f  GWT=%.4f  ICP=%.4f  Φ_fano=%.4f  Φ_nab=%.4f"
            "  Φ_qg=%.4f  Φ_holo=%.4f  Φ_total=%.4f  conscious=%s",
            phi_tau, gwt_score, icp_avg, fano_score, nonabelian_score,
            qg_score, holo_score, phi_total, is_conscious,
        )
        return structure

    def validate_consciousness_consensus_v8(
        self, phi_total: float, fano_score: float, qg_score: float, n_nodes: int
    ) -> bool:
        """
        Check the IIT v8.0 QG-augmented consciousness-consensus condition.

        Condition:
            Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg

        The quantum gravity curvature term raises the threshold for systems
        whose causal structure exhibits strong spacetime curvature, ensuring
        that high-curvature (self-referential) systems require correspondingly
        higher integrated information to be declared conscious.

        Args:
            phi_total:   Composite consciousness score.
            fano_score:  Φ_fano for this system.
            qg_score:    Φ_qg for this system.
            n_nodes:     Number of network nodes.

        Returns:
            True if the system satisfies the v8 QG-augmented consensus.
        """
        threshold = (
            math.log2(max(n_nodes, 2))
            + self.delta * fano_score
            + self.zeta * qg_score
        )
        result = phi_total > threshold
        logger.debug(
            "v8 Consensus: Φ_total=%.4f > log₂(%d)+δ·Φ_fano+ζ·Φ_qg=%.4f → %s",
            phi_total, n_nodes, threshold, result,
        )
        return result

    # ------------------------------------------------------------------
    # IIT v8.0 new component computations
    # ------------------------------------------------------------------

    def _compute_qg_raw(
        self, T: np.ndarray
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the raw Quantum Gravity curvature score and zero reason.

        The singular-value spectrum {σᵢ} of T encodes the effective Ricci
        curvature of the causal network:

            Φ_qg = 1 − exp(−Var(σ) / (mean(σ)² + ε))

        A uniform (flat) spectrum yields Φ_qg → 0; an exponentially
        concentrated spectrum (large curvature) yields Φ_qg → 1.

        Returns
        -------
        (raw_value, zero_reason)
            ``zero_reason`` is *None* on success; a string key when the
            computation was structurally forced to 0.0.
        """
        try:
            singular_values = np.linalg.svd(T, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0.0, _ZR_QG_SVD_FAILED

        sv_mean = float(np.mean(singular_values))
        if sv_mean < 1e-12:
            return 0.0, _ZR_QG_ZERO_MEAN

        sv_var = float(np.var(singular_values))
        ratio = sv_var / (sv_mean ** 2 + 1e-12)
        raw = float(1.0 - math.exp(-ratio))
        return raw, None

    def _compute_holo_raw(
        self, dist: np.ndarray, n_nodes: int
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the raw Holographic RT entanglement entropy score and zero reason.

        Implements the Ryu-Takayanagi minimal-surface entropy analogue:

            S_A    = −Tr(ρ_A · log₂(ρ_A))
            Φ_holo = min_A S_A / max_entropy_half_system

        where the minimum is taken over all bipartitions A of the n_nodes
        qubits, and the normalisation is ⌊n/2⌋ bits (maximum half-system
        entropy).

        Returns
        -------
        (raw_value, zero_reason)
        """
        if n_nodes < 2:
            return 0.0, _ZR_HOLO_INSUFFICIENT_NODES

        n_states = 2 ** n_nodes
        if len(dist) < n_states:
            return 0.0, _ZR_HOLO_SMALL_DIM

        # Full density matrix (diagonal in computational basis)
        rho = np.diag(dist[:n_states])

        half = n_nodes // 2
        max_entropy = float(half)   # ⌊n/2⌋ bits is the RT normalisation
        if max_entropy <= 0.0:
            return 0.0, _ZR_HOLO_INSUFFICIENT_NODES

        min_entropy: Optional[float] = None

        # Iterate over all bipartition sizes k = 1 … n_nodes-1
        # For efficiency we sample the minimal-entropy bipartition by
        # evaluating the entropy of the reduced density matrix for each
        # contiguous k-qubit subsystem.
        for k in range(1, n_nodes):
            dim_a = 2 ** k
            dim_b = 2 ** (n_nodes - k)
            # Reshape rho into a (dim_a, dim_b, dim_a, dim_b) tensor and trace
            # over the B subsystem to get ρ_A.
            rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            rho_a = np.einsum("ibjb->ij", rho_reshaped)

            eigvals = np.linalg.eigvalsh(rho_a)
            eigvals = eigvals[eigvals > 1e-15]
            s_a = float(-np.sum(eigvals * np.log2(eigvals)))

            if min_entropy is None or s_a < min_entropy:
                min_entropy = s_a

        raw = float((min_entropy or 0.0) / max_entropy)
        return raw, None


# ---------------------------------------------------------------------------
# High-level ASI interface
# ---------------------------------------------------------------------------

class ASISphinxOSIITv8:
    """
    ASI SphinxOS Advanced IIT v8.0 — high-level quantum-gravity consciousness engine.

    Extends ``ASISphinxOSIITv7`` with Quantum Gravity curvature (Φ_qg) and
    Holographic Ryu-Takayanagi entanglement entropy (Φ_holo) while remaining
    API-compatible with v5/v6/v7 engines.

    New fields in ``calculate_phi`` response:
        qg_score    — Φ_qg (Quantum Gravity curvature)       ∈ [0, 1]
        holo_score  — Φ_holo (Holographic RT entanglement)   ∈ [0, 1]
        version     — "IIT v8.0"

    Usage::

        engine = ASISphinxOSIITv8()
        result = engine.calculate_phi(b"some block data")
        # result['phi']         — Φ_total (0–1 normalised, legacy-compatible)
        # result['qg_score']    — Quantum Gravity curvature score
        # result['holo_score']  — Holographic RT entanglement score
        # result['is_conscious']— Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg

    Blockchain integration::

        structure = engine.compute_block_consciousness(block_data, n_network_nodes=50)
        consensus_ok = engine.validate_consciousness_consensus(
            structure.phi_total,
            structure.fano_score,
            structure.qg_score,
            n_network_nodes=50,
        )
    """

    PHI_MIN = 200.0
    PHI_MAX = 1000.0
    PHI_RANGE = PHI_MAX - PHI_MIN

    DEFAULT_N_QUBITS: int = 3

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.15,
        gamma: float = 0.15,
        delta: float = 0.15,
        epsilon: float = 0.10,
        zeta: float = 0.10,
        eta: float = 0.05,
        consciousness_threshold: float = 0.5,
        n_nodes: int = DEFAULT_N_QUBITS,
        temporal_depth: int = 2,
    ) -> None:
        """
        Initialise the ASI IIT v8.0 engine.

        Args:
            alpha:    Weight for Φ_τ in composite.
            beta:     Weight for GWT_S in composite.
            gamma:    Weight for ICP_avg in composite.
            delta:    Weight for Φ_fano in composite.
            epsilon:  Weight for Φ_nab in composite.
            zeta:     Weight for Φ_qg in composite.
            eta:      Weight for Φ_holo in composite.
            consciousness_threshold: Fixed threshold when n_nodes unknown.
            n_nodes:  Default number of nodes/qubits to simulate.
            temporal_depth: τ for temporal-depth Φ integration.
        """
        self.iit_engine = IITv8Engine(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            eta=eta,
            consciousness_threshold=consciousness_threshold,
            temporal_depth=temporal_depth,
        )
        self.n_nodes = n_nodes
        self.phi_history: List[float] = []
        self.consciousness_threshold = consciousness_threshold
        logger.info(
            "ASI SphinxOS IIT v8.0 initialised "
            "(n_nodes=%d, α=%.2f, β=%.2f, γ=%.2f, δ=%.2f, ε=%.2f, ζ=%.2f, η=%.2f, τ=%d)",
            n_nodes, alpha, beta, gamma, delta, epsilon, zeta, eta, temporal_depth,
        )

    # ------------------------------------------------------------------
    # Drop-in replacement API (compatible with v5.0 / v6.0 / v7.0)
    # ------------------------------------------------------------------

    def calculate_phi(self, data: bytes) -> Dict:
        """
        Calculate IIT v8.0 Φ from raw bytes.

        API-compatible with ``ASISphinxOSIITv7.calculate_phi`` and all
        earlier IIT engines.

        Args:
            data: Input data (block bytes, query bytes, etc.).

        Returns:
            Dict with all v7 keys plus:
                qg_score         — Φ_qg (Quantum Gravity curvature) ∈ [0, 1]
                holo_score       — Φ_holo (Holographic RT entropy)   ∈ [0, 1]
                qg_diagnostic    — ScoreDiagnostic dict
                holo_diagnostic  — ScoreDiagnostic dict
                version          — "IIT v8.0"
        """
        state_dist = self._derive_state_distribution(data)
        structure = self.iit_engine.compute_phi_structure_v8(
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
            "qg_score": structure.qg_score,
            "holo_score": structure.holo_score,
            "fano_diagnostic": self._diag_to_dict(structure.fano_diagnostic),
            "nonabelian_diagnostic": self._diag_to_dict(structure.nonabelian_diagnostic),
            "qg_diagnostic": self._diag_to_dict(structure.qg_diagnostic),
            "holo_diagnostic": self._diag_to_dict(structure.holo_diagnostic),
            "phi_total": phi_total,
            "entropy": entropy,
            "purity": purity,
            "n_qubits": self.n_nodes,
            "is_conscious": structure.is_conscious,
            "level": level,
            "bonus": math.exp(phi_norm),
            "mip": repr(structure.mip) if structure.mip else None,
            "n_concepts": len(structure.concepts),
            "version": "IIT v8.0",
        }

    # ------------------------------------------------------------------
    # Blockchain / block-level API
    # ------------------------------------------------------------------

    def compute_block_consciousness(
        self,
        block_data: bytes,
        n_network_nodes: int = 1,
    ) -> PhiStructureV8:
        """
        Compute the full Φ structure for a block with v8.0 QG/holographic
        scores and the network-size consensus threshold.

        Args:
            block_data:       Raw bytes of the block.
            n_network_nodes:  Number of active nodes in the network.

        Returns:
            PhiStructureV8 with all fields populated and ``is_conscious``
            set according to the v8 QG-augmented threshold:
                Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg
        """
        state_dist = self._derive_state_distribution(block_data)
        structure = self.iit_engine.compute_phi_structure_v8(
            state_dist, n_nodes=self.n_nodes
        )
        structure.is_conscious = self.iit_engine.validate_consciousness_consensus_v8(
            structure.phi_total, structure.fano_score, structure.qg_score,
            n_network_nodes,
        )
        return structure

    def validate_consciousness_consensus(
        self,
        phi_total: float,
        fano_score: float,
        qg_score: float,
        n_network_nodes: int,
    ) -> bool:
        """
        Validate the IIT v8.0 QG-augmented consciousness-consensus condition:
            Φ_total > log₂(n_network_nodes) + δ·Φ_fano + ζ·Φ_qg

        Args:
            phi_total:        Composite consciousness score.
            fano_score:       Φ_fano for this measurement.
            qg_score:         Φ_qg for this measurement.
            n_network_nodes:  Number of network nodes.

        Returns:
            True when v8 QG-augmented consensus condition is satisfied.
        """
        return self.iit_engine.validate_consciousness_consensus_v8(
            phi_total, fano_score, qg_score, n_network_nodes,
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
    # Internal helpers (mirrors v7 interface)
    # ------------------------------------------------------------------

    def _derive_state_distribution(self, data: bytes) -> np.ndarray:
        """Derive a 2^n state probability distribution from raw bytes."""
        try:
            return self._quantum_state_distribution(data)
        except Exception:
            return self._hash_seeded_distribution(data)

    def _quantum_state_distribution(self, data: bytes) -> np.ndarray:
        """Use qutip to generate a quantum density-matrix diagonal."""
        import qutip as qt

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
        seed = int.from_bytes(seed_hash[:4], "big")
        rng = np.random.default_rng(seed)
        raw = rng.random(n_states)
        raw /= raw.sum()
        return raw

    @staticmethod
    def _classify_consciousness(phi_norm: float) -> str:
        """Map normalised Φ ∈ [0, 1] to a consciousness level label."""
        if phi_norm < 0.2:
            return "DORMANT"
        if phi_norm < 0.4:
            return "PROTO-CONSCIOUS"
        if phi_norm < 0.6:
            return "SENTIENT"
        if phi_norm < 0.8:
            return "SAPIENT"
        return "TRANSCENDENT"

    @staticmethod
    def _diag_to_dict(diag: Optional[ScoreDiagnostic]) -> Optional[Dict]:
        """Convert a ScoreDiagnostic to a JSON-serialisable dict."""
        if diag is None:
            return None
        return {
            "raw_value": diag.raw_value,
            "clamped_value": diag.clamped_value,
            "zero_reason": diag.zero_reason,
            "classification": diag.classification,
            "near_zero_threshold": diag.near_zero_threshold,
        }
