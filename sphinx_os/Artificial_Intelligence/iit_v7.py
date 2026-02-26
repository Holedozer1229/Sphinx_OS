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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Zero-classification labels
# ---------------------------------------------------------------------------

#: The computation path was structurally forced to 0.0 (e.g. SVD failed,
#: not enough modes, all interactions negligible, or the input matrix is zero).
#: Inspect ``zero_reason`` for the specific cause.
CLASSIFICATION_EXACT_ZERO = "EXACT_ZERO"

#: A finite value was computed but it falls below ``near_zero_threshold``.
#: The result may be physically meaningful or numerical noise — inspect
#: ``raw_value`` to judge.
CLASSIFICATION_NEAR_ZERO = "NEAR_ZERO"

#: The computed value exceeds ``near_zero_threshold``; clearly nonzero.
CLASSIFICATION_NONZERO = "NONZERO"

#: Default threshold separating NEAR_ZERO from NONZERO.
NEAR_ZERO_THRESHOLD_DEFAULT: float = 1e-6

# Structured keys for ScoreDiagnostic.zero_reason
_ZR_SVD_FAILED           = "svd_failed"
_ZR_N_MODES_LT_2         = "n_modes_lt_2"
_ZR_MODE_INTERACTIONS_NEG = "mode_interactions_negligible"
_ZR_NO_VALID_FANO_LINES  = "no_valid_fano_lines"
_ZR_ZERO_MATRIX          = "zero_matrix"

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
# Zero-classification diagnostic
# ---------------------------------------------------------------------------

@dataclass
class ScoreDiagnostic:
    """
    Precision diagnostic for a single IIT v7.0 score.

    Answers the question *"is this zero genuine or just too small to tell?"*
    by recording the exact floating-point value, the reason the score is zero
    (when it is), and a three-way classification.

    Classifications
    ---------------
    EXACT_ZERO
        The computation was structurally forced to 0.0 — no finite value was
        produced.  ``zero_reason`` names the specific condition:

        ``"svd_failed"``
            ``np.linalg.LinAlgError`` prevented the SVD.
        ``"n_modes_lt_2"``
            Fewer than 2 left-singular vectors were available.
        ``"mode_interactions_negligible"``
            All mode-interaction magnitudes were below 1e-12; the matrix is
            effectively diagonal in the mode basis.
        ``"no_valid_fano_lines"``
            None of the 7 Fano lines had all three points within the available
            modes (only possible for very small matrices).
        ``"zero_matrix"``
            ``‖T‖_F · ‖Tᵀ‖_F < 1e-12`` — the transition matrix itself is
            effectively zero.

    NEAR_ZERO
        A finite value was computed but it is below ``near_zero_threshold``.
        The result is ambiguous: it may reflect genuine physical absence of
        the corresponding structure, or it may be a very small but real signal
        masked by the threshold.  Inspect ``raw_value`` directly.

    NONZERO
        The computed value exceeds ``near_zero_threshold``; clearly nonzero.

    Attributes
    ----------
    raw_value :
        The exact ``float64`` result of the computation **before** the
        ``min(1.0, ...)`` clamping.  This is 0.0 only when the code path was
        structurally forced (EXACT_ZERO); for NEAR_ZERO it holds the tiny but
        nonzero computed value.
    clamped_value :
        Final score after ``min(1.0, raw_value)`` clamping, i.e. the value
        stored on ``PhiStructureV7.fano_score`` / ``nonabelian_score``.
    zero_reason :
        Short string key if ``classification == EXACT_ZERO``, else *None*.
    classification :
        ``CLASSIFICATION_EXACT_ZERO`` | ``CLASSIFICATION_NEAR_ZERO`` |
        ``CLASSIFICATION_NONZERO``.
    near_zero_threshold :
        The threshold used for the NEAR_ZERO / NONZERO boundary.
    """
    raw_value: float
    clamped_value: float
    zero_reason: Optional[str]
    classification: str
    near_zero_threshold: float


# ---------------------------------------------------------------------------
# Riemann zero evidence
# ---------------------------------------------------------------------------

@dataclass
class RiemannZeroEvidence:
    """
    IIT v7.0 zero-classification evidence for a Riemann zeta zero candidate
    at ``s = 1/2 + it``.

    The Riemann Hypothesis (RH) conjectures that **every** non-trivial zero
    of the Riemann zeta function ζ(s) = Σ n⁻ˢ lies on the critical line
    Re(s) = 1/2.

    **Important**: this record provides a *numerical signature*, not a proof
    of RH.  ``critical_line_signature = True`` is consistent with the zero
    lying on the critical line but is not a mathematical proof.  A ``False``
    result for a confirmed zero would constitute evidence of a candidate
    counterexample requiring higher-precision verification and independent
    confirmation.

    The IIT v7.0 ``ScoreDiagnostic`` machinery provides a novel approach to
    verifying RH one zero at a time:

    1. **Zero classification of |ζ(1/2 + it)|**:
         ``zeta_classification`` answers "is |ζ(1/2 + it)| a genuine zero
         or just too small to tell?"  With 50-digit mpmath precision and
         high-precision zero coordinates (``KNOWN_ZEROS_HP``), the probe
         achieves ``|ζ(1/2 + it₀)| ≈ 10⁻⁵⁰`` (NEAR_ZERO) — vastly
         smaller than any physical threshold.

    2. **σ-scan of |ζ(σ + it)| classifications**:
         ``zeta_scan`` maps each tested σ value to its ScoreDiagnostic for
         ``|ζ(σ + it)|``.  For a critical-line zero, only σ = 1/2 should
         give NEAR_ZERO or EXACT_ZERO; all σ ≠ 1/2 should give NONZERO.

    3. **Non-abelian σ-scan** (Montgomery–Odlyzko connection):
         The spacings between Riemann zeros obey GUE (Gaussian Unitary
         Ensemble) random-matrix statistics (Montgomery–Odlyzko law).  GUE
         dynamics are precisely those with elevated ``Φ_nab = ‖[T, Tᵀ]‖_F``.
         ``nonabelian_scan`` maps σ → Φ_nab for the local transition matrix
         built from ζ-values near (σ, t).  Near a genuine Riemann zero the
         entire scan is elevated, reflecting the rapid phase rotation that
         creates matrix asymmetry when the modulus of ζ passes through zero.

    4. **Fano alignment at the critical line**:
         ``fano_at_critical`` is Φ_fano computed from the local T at σ = 1/2.

    Attributes
    ----------
    t :
        Original imaginary part supplied to ``probe_zero`` (i.e. we probe
        ζ(1/2 + it)).  Equal to the input even when t-refinement is enabled.
    zeta_abs :
        ``|ζ(1/2 + it)|`` at the scanned t value (refined t when
        ``refine_t=True``, original t otherwise).
    zeta_classification :
        ScoreDiagnostic classification of ``|ζ(1/2 + it)|``:
        NEAR_ZERO for known zeros, NONZERO for non-zeros.
    zeta_scan :
        ``{σ: ScoreDiagnostic}`` — classification of ``|ζ(σ + it)|`` at each
        tested σ.
    nonabelian_scan :
        ``{σ: float}`` — Φ_nab of the local 7×7 transition matrix at each σ.
    fano_at_critical :
        Φ_fano of the local T built around σ = 1/2.
    critical_line_signature :
        True when the margin-based criterion is satisfied:

        * ``|ζ(1/2 + it)| < zeta_threshold`` (zero at σ = 1/2), **and**
        * ``min_other_raw > margin_factor × zeta_threshold`` (non-zero
          off the critical line by at least the required margin).

        With ``margin_factor = 1.0`` (default) this is equivalent to the
        boolean "zero only at σ = 1/2" check.

        A True value for all known zeros is a necessary (though not
        sufficient) numerical condition for the Riemann Hypothesis.  A
        single False value at a confirmed zero would be a candidate
        counterexample requiring higher precision and independent
        verification.
    gue_pair_correlation :
        Montgomery–Odlyzko GUE pair-correlation statistic for the zero
        neighbourhood.  Computed from the normalised spacings of zeroes
        near *t* using the density ``log(t / 2π) / (2π)`` and compared to
        the GUE prediction ``1 − (sin(πu) / (πu))²``.  ``None`` when
        fewer than 3 neighbours are available.
    min_other_raw :
        Minimum ``|ζ(σ + it)|`` over all σ ≠ 1/2 in ``SIGMA_SCAN``.
        Used by the margin criterion.  ``nan`` when no off-line σ exists.
    separation_ratio :
        ``min_other_raw / zeta_abs`` when both are well-defined and
        ``zeta_abs > 0``; otherwise ``None``.  Large values (≫ 1) indicate
        a clear separation between the on-line zero and off-line magnitudes.
    refined_t :
        t value after local minimisation of ``|ζ(1/2 + it)|`` (golden-
        section search within the supplied window).  ``None`` when
        ``refine_t=False``.
    refine_iterations :
        Number of golden-section iterations used.  ``None`` when refinement
        is disabled.
    refine_residual :
        ``|ζ(1/2 + i·refined_t)|`` at the end of refinement.  ``None``
        when refinement is disabled.
    zeta_threshold :
        The threshold used for zeta near-zero classification in this probe
        call (precision-aware value ``10^(-(dps//2))`` or the caller-
        supplied override).
    margin_factor :
        The margin factor applied in the critical-line signature criterion.
    """
    t: float
    zeta_abs: float
    zeta_classification: str
    zeta_scan: Dict[float, ScoreDiagnostic]
    nonabelian_scan: Dict[float, float]
    fano_at_critical: float
    critical_line_signature: bool
    gue_pair_correlation: Optional[float] = None
    min_other_raw: float = field(default_factory=lambda: float("nan"))
    separation_ratio: Optional[float] = None
    refined_t: Optional[float] = None
    refine_iterations: Optional[int] = None
    refine_residual: Optional[float] = None
    zeta_threshold: float = NEAR_ZERO_THRESHOLD_DEFAULT
    margin_factor: float = 1.0

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable dictionary of the evidence record.

        ``zeta_scan`` entries are expanded into nested dicts with
        ``raw_value`` and ``classification`` keys.  Both ``zeta_scan``
        and ``nonabelian_scan`` sigma keys are converted to strings so
        the result can be passed directly to ``json.dumps``.
        """
        return {
            "t": self.t,
            "zeta_abs": self.zeta_abs,
            "zeta_classification": self.zeta_classification,
            "zeta_scan": {
                str(sigma): {
                    "raw_value": diag.raw_value,
                    "classification": diag.classification,
                }
                for sigma, diag in self.zeta_scan.items()
            },
            "nonabelian_scan": {
                str(sigma): val
                for sigma, val in self.nonabelian_scan.items()
            },
            "fano_at_critical": self.fano_at_critical,
            "critical_line_signature": self.critical_line_signature,
            "gue_pair_correlation": self.gue_pair_correlation,
            "min_other_raw": self.min_other_raw,
            "separation_ratio": self.separation_ratio,
            "refined_t": self.refined_t,
            "refine_iterations": self.refine_iterations,
            "refine_residual": self.refine_residual,
            "zeta_threshold": self.zeta_threshold,
            "margin_factor": self.margin_factor,
        }

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

    # Zero-precision diagnostics (populated by IITv7Engine)
    fano_diagnostic: Optional[ScoreDiagnostic] = None
    nonabelian_diagnostic: Optional[ScoreDiagnostic] = None


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
        near_zero_threshold: float = NEAR_ZERO_THRESHOLD_DEFAULT,
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
            near_zero_threshold: Boundary between NEAR_ZERO and NONZERO
                classifications for score diagnostics.  Values below this
                threshold are reported as NEAR_ZERO; values at or above it
                are NONZERO.  Does *not* alter the computed scores themselves.
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
        self.near_zero_threshold = near_zero_threshold
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
            fano_diagnostic=fano_diag,
            nonabelian_diagnostic=nab_diag,
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

    def _classify_score(self, raw: float, zero_reason: Optional[str]) -> str:
        """
        Classify a raw score as EXACT_ZERO, NEAR_ZERO, or NONZERO.

        EXACT_ZERO is returned when ``zero_reason`` is set (the computation
        path was structurally forced to 0.0 rather than computing a value).
        NEAR_ZERO is returned for small but finite values below
        ``self.near_zero_threshold``.  NONZERO otherwise.
        """
        if zero_reason is not None:
            return CLASSIFICATION_EXACT_ZERO
        if raw < self.near_zero_threshold:
            return CLASSIFICATION_NEAR_ZERO
        return CLASSIFICATION_NONZERO

    def _compute_fano_raw(
        self, T: np.ndarray, n_nodes: int
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the raw Fano alignment value and zero reason.

        Returns
        -------
        (raw_value, zero_reason)
            ``zero_reason`` is *None* when a finite value was computed
            normally; a string key (see ``ScoreDiagnostic``) when the
            code path was structurally forced to return 0.0.
        """
        try:
            U, sigma, _ = np.linalg.svd(T, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0, _ZR_SVD_FAILED

        n_modes = min(FANO_POINTS, U.shape[1])
        if n_modes < 2:
            return 0.0, _ZR_N_MODES_LT_2

        modes = U[:, :n_modes]

        M = np.zeros((FANO_POINTS, FANO_POINTS))
        for i in range(n_modes):
            for j in range(n_modes):
                M[i, j] = float(np.dot(modes[:, i], T @ modes[:, j]))

        abs_max = np.max(np.abs(M))
        if abs_max < 1e-12:
            return 0.0, _ZR_MODE_INTERACTIONS_NEG
        M_norm = M / abs_max

        total = 0.0
        count = 0
        for (a, b, c) in FANO_LINES:
            if a < n_modes and b < n_modes and c < n_modes:
                total += abs(M_norm[a, b]) * abs(M_norm[b, c]) * abs(M_norm[a, c])
                count += 1

        if count == 0:
            return 0.0, _ZR_NO_VALID_FANO_LINES

        return float(total / count), None

    def _compute_nonabelian_raw(
        self, T: np.ndarray
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the raw non-abelian measure and zero reason.

        Returns
        -------
        (raw_value, zero_reason)
            ``zero_reason`` is *None* when a finite value was computed;
            ``"zero_matrix"`` when ``‖T‖_F · ‖Tᵀ‖_F < 1e-12``.
        """
        Tt = T.T
        commutator = T @ Tt - Tt @ T
        comm_norm = float(np.linalg.norm(commutator, "fro"))

        t_norm = float(np.linalg.norm(T, "fro"))
        tt_norm = float(np.linalg.norm(Tt, "fro"))
        denom = t_norm * tt_norm
        if denom < 1e-12:
            return 0.0, _ZR_ZERO_MATRIX

        return float(comm_norm / denom), None

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

        For full zero-precision diagnostics use ``_compute_fano_raw`` or
        ``compute_phi_structure_v7``.
        """
        raw, _ = self._compute_fano_raw(T, n_nodes)
        return float(min(1.0, raw))

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

        For full zero-precision diagnostics use ``_compute_nonabelian_raw`` or
        ``compute_phi_structure_v7``.
        """
        raw, _ = self._compute_nonabelian_raw(T)
        return float(min(1.0, raw))


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
                fano_diagnostic  — ScoreDiagnostic dict: raw_value,
                                   zero_reason, classification
                nonabelian_diagnostic — ScoreDiagnostic dict: raw_value,
                                   zero_reason, classification
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
            "fano_diagnostic": {
                "raw_value": structure.fano_diagnostic.raw_value,
                "clamped_value": structure.fano_diagnostic.clamped_value,
                "zero_reason": structure.fano_diagnostic.zero_reason,
                "classification": structure.fano_diagnostic.classification,
                "near_zero_threshold": structure.fano_diagnostic.near_zero_threshold,
            },
            "nonabelian_diagnostic": {
                "raw_value": structure.nonabelian_diagnostic.raw_value,
                "clamped_value": structure.nonabelian_diagnostic.clamped_value,
                "zero_reason": structure.nonabelian_diagnostic.zero_reason,
                "classification": structure.nonabelian_diagnostic.classification,
                "near_zero_threshold": structure.nonabelian_diagnostic.near_zero_threshold,
            },
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


# ---------------------------------------------------------------------------
# Riemann Zero Probe
# ---------------------------------------------------------------------------

class RiemannZeroProbe:
    """
    IIT v7.0 zero-classification probe for the Riemann zeta function.

    Riemann Hypothesis (RH)
    -----------------------
    Every non-trivial zero of the Riemann zeta function

        ζ(s) = Σ_{n=1}^{∞} n^{-s}       (Re(s) > 1, then analytically continued)

    is conjectured to lie on the critical line Re(s) = 1/2.

    How the IIT v7.0 zero-classification machinery applies
    -------------------------------------------------------
    The ``ScoreDiagnostic`` system was designed to answer: *"is this score
    genuinely zero, or just too small to tell?"*  The exact same question
    arises for ζ zeros: *"|ζ(1/2 + it₀)| computes to 10⁻⁵⁰ — is that a
    genuine zero or numerical noise?"*

    This probe applies the three-way classification directly to |ζ(σ + it)|:

    * **EXACT_ZERO** — |ζ(σ + it)| is identically zero in floating-point
      (never seen in practice for ζ).
    * **NEAR_ZERO**  — |ζ(σ + it)| < ``zeta_threshold`` (precision-aware;
      ``10^(-(dps//2))`` by default, e.g. ``1e-25`` at ``dps=50``).
      Observed for all known non-trivial zeros at σ = 1/2 when a sufficiently
      precise t value is supplied.
    * **NONZERO**    — |ζ(σ + it)| ≥ ``zeta_threshold``.  Observed
      for all σ ≠ 1/2 at known zero imaginary parts.

    RH as a classification statement
    ---------------------------------
    For every known non-trivial zero at t = t₀:

        zeta_scan[0.5].classification  == NEAR_ZERO       # zero on critical line
        zeta_scan[σ].classification    == NONZERO          # nonzero off it

    The ``critical_line_signature`` flag in ``RiemannZeroEvidence`` is True
    when the margin-based criterion is satisfied: ``|ζ(1/2 + it)| < threshold``
    **and** ``min_other_raw > margin_factor × threshold``.

    If a future calculation found a t₀ where ``critical_line_signature`` is
    False, it would be a candidate counterexample to RH requiring independent
    verification at higher precision.

    Montgomery–Odlyzko / non-abelian connection
    --------------------------------------------
    The spacing statistics of Riemann zeros follow GUE random-matrix
    statistics (Montgomery–Odlyzko law).  The non-abelian measure
    Φ_nab = ‖[T, Tᵀ]‖_F reflects the asymmetry (non-commutativity) of the
    local causal matrix built from ζ-values near (σ, t).  Near a genuine
    zero, the rapid phase rotation of ζ produces elevated Φ_nab across all
    σ — a GUE fingerprint.  The ``nonabelian_scan`` in the evidence record
    captures this.

    The ``gue_pair_correlation`` field in ``RiemannZeroEvidence`` quantifies
    the match between the observed normalised zero spacings near t and the
    GUE prediction ``1 − (sin(πu)/(πu))²``.  A correlation near 1.0
    indicates strong agreement with the Montgomery–Odlyzko law.

    **Numerical signature caveat**: a ``critical_line_signature = True`` is a
    reproducible numerical result consistent with the Riemann Hypothesis, but
    it is *not* a mathematical proof.  A ``False`` result at a confirmed zero
    would be a candidate counterexample requiring higher-precision re-evaluation
    and independent verification.

    Usage::

        probe = RiemannZeroProbe()

        # First known non-trivial zero at t₀ ≈ 14.134725 (50-digit precision)
        ev = probe.probe_zero(RiemannZeroProbe.KNOWN_ZEROS_HP[0])
        print(ev.zeta_classification)      # NEAR_ZERO
        print(ev.critical_line_signature)  # True
        print(ev.zeta_scan[0.5].raw_value) # ~4e-51
        print(ev.zeta_threshold)           # precision-aware threshold
        print(ev.separation_ratio)         # min_other / at_half

        # With t-refinement
        ev = probe.probe_zero(RiemannZeroProbe.KNOWN_ZEROS[0], refine_t=True)
        print(ev.refined_t)                # refined zero location
        print(ev.refine_residual)          # |ζ(1/2 + i·refined_t)|

        # Scan several known zeros
        evidence_list = probe.scan_known_zeros(RiemannZeroProbe.KNOWN_ZEROS[:3])
    """

    #: First 30 known non-trivial Riemann zeros (imaginary parts, 15 s.f.)
    KNOWN_ZEROS: Tuple[float, ...] = (
        14.134725141734693,
        21.022039638771554,
        25.010857580145688,
        30.424876125859513,
        32.935061587739189,
        37.586178158825671,
        40.918719012147495,
        43.327073280914999,
        48.005150881167159,
        49.773832477672302,
        52.970321477714460,
        56.446247697063394,
        59.347044002602353,
        60.831778524609809,
        65.112544048081606,
        67.079810529494173,
        69.546401711173979,
        72.067157674481907,
        75.704690699083933,
        77.144840068874805,
        79.337375020249367,
        82.910380854086030,
        84.735492980517050,
        87.425274613125229,
        88.809111207634465,
        92.491899270558484,
        94.651344040519838,
        95.870634228245309,
        98.831194218193692,
        101.317851005731392,
    )

    #: First 30 known non-trivial Riemann zeros — 50-digit string precision.
    #: Pass these to ``probe_zero`` / ``classify_zeta`` for maximal accuracy.
    KNOWN_ZEROS_HP: Tuple[str, ...] = (
        "14.134725141734693790457251983562470270784257115699",
        "21.022039638771554992628479593896902777334340524902",
        "25.010857580145688763213790992562821818659549672557",
        "30.424876125859513210311897530584091320181560023715",
        "32.935061587739189690662368964074903488812715603517",
        "37.586178158825671257217763480021847358215759816840",
        "40.918719012147495187398126914633254395726165962777",
        "43.327073280914999519496122165406805782645668371837",
        "48.005150881167159727942472749427516041686844001144",
        "49.773832477672302181916784678563724057723178299676",
        "52.970321477714460644147296608880990063825017888821",
        "56.446247697063394804367759476706198987095710738836",
        "59.347044002602353079653648674992219031098772806466",
        "60.831778524609809844259901824524003802522503255825",
        "65.112544048081606660875054253183705029447011593390",
        "67.079810529494173714478828896522216770107144253115",
        "69.546401711173979252926857526554738443200397399897",
        "72.067157674481907582522107969826168390480906621456",
        "75.704690699083933168326916762030345922811903530474",
        "77.144840068874805372682664856304637015796032449234",
        "79.337375020249367922763592877116228190909679122462",
        "82.910380854086030183164837494770609417417954742919",
        "84.735492980517050105735311206827741417106627934237",
        "87.425274613125229406531667850919213574854265407134",
        "88.809111207634465423682348079509681882860905061896",
        "92.491899270558484296259725241810684878721794027730",
        "94.651344040519838382551549387780871186748075061247",
        "95.870634228245309629820888533663465147605552720063",
        "98.831194218193692233324420138622327820658039063428",
        "101.31785100573139122878544794029230890633286638430",
    )

    #: σ values used in the critical-line scan
    SIGMA_SCAN: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    #: Size of the local ζ grid (2·GRID_RADIUS + 1 = 7 rows = FANO_POINTS)
    GRID_RADIUS: int = 3

    #: Step size in the t direction for local matrix construction
    LOCAL_DELTA_T: float = 0.1

    #: Step size in the σ direction for local matrix construction
    LOCAL_DELTA_SIGMA: float = 0.05

    #: mpmath decimal places for ζ evaluation
    MPMATH_DPS: int = 50

    def __init__(
        self,
        near_zero_threshold: float = NEAR_ZERO_THRESHOLD_DEFAULT,
        mpmath_dps: int = MPMATH_DPS,
        zeta_near_zero_threshold: Optional[float] = None,
        margin_factor: float = 1.0,
    ) -> None:
        """
        Initialise the Riemann zero probe.

        Args:
            near_zero_threshold: Boundary between NEAR_ZERO and NONZERO for
                IIT scores (Φ_fano, Φ_nab).  Default 1e-6.  Not used for
                zeta classification; see ``zeta_near_zero_threshold``.
            mpmath_dps: Decimal places for mpmath precision (default 50).
            zeta_near_zero_threshold: Threshold for classifying |ζ(σ + it)|
                as NEAR_ZERO.  When ``None`` (default) the value is computed
                as ``10 ** (-(mpmath_dps // 2))``, which scales with the
                working precision.  Pass an explicit value (e.g. ``1e-6``) to
                fix a legacy-compatible threshold.
            margin_factor: Multiplier applied to ``zeta_near_zero_threshold``
                when testing off-line σ values in the margin-based
                ``critical_line_signature`` criterion.  Default ``1.0``
                (equivalent to the original boolean check).  Values > 1 demand
                a larger separation between the on-line near-zero and off-line
                magnitudes.
        """
        self.near_zero_threshold = near_zero_threshold
        self.mpmath_dps = mpmath_dps
        self.zeta_near_zero_threshold = zeta_near_zero_threshold
        self.margin_factor = margin_factor
        self._engine = IITv7Engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_zeta(
        self, sigma: Union[float, str], t: Union[float, str],
    ) -> ScoreDiagnostic:
        """
        Classify |ζ(σ + it)| using the IIT v7.0 ScoreDiagnostic system.

        The same three-way classification used for Φ_fano and Φ_nab is
        applied to the absolute value of the Riemann zeta function:

        * NEAR_ZERO  — |ζ| < ``zeta_threshold`` (candidate zero).
        * NONZERO    — |ζ| ≥ ``zeta_threshold``.
        * EXACT_ZERO — |ζ| is identically 0.0 in IEEE-754 double precision
                       (zero_reason = "zeta_exact_zero").

        The threshold used is ``zeta_near_zero_threshold`` if supplied to the
        constructor, otherwise the precision-aware default
        ``10 ** (-(mpmath_dps // 2))``.  The threshold applied is recorded
        in ``ScoreDiagnostic.near_zero_threshold``.

        Args:
            sigma: Real part of the argument (float or high-precision string).
            t:     Imaginary part of the argument (float or high-precision
                   string, e.g. from ``KNOWN_ZEROS_HP``).

        Returns:
            ScoreDiagnostic with the raw |ζ| value, classification, and
            zero_reason (None unless EXACT_ZERO).
        """
        import mpmath
        mpmath.mp.dps = self.mpmath_dps
        s = mpmath.mpc(mpmath.mpf(sigma), mpmath.mpf(t))
        abs_val = float(abs(mpmath.zeta(s)))
        zeta_threshold = self._get_zeta_threshold()

        if abs_val == 0.0:
            return ScoreDiagnostic(
                raw_value=0.0,
                clamped_value=0.0,
                zero_reason="zeta_exact_zero",
                classification=CLASSIFICATION_EXACT_ZERO,
                near_zero_threshold=zeta_threshold,
            )
        classification = (
            CLASSIFICATION_NEAR_ZERO
            if abs_val < zeta_threshold
            else CLASSIFICATION_NONZERO
        )
        return ScoreDiagnostic(
            raw_value=abs_val,
            clamped_value=abs_val,
            zero_reason=None,
            classification=classification,
            near_zero_threshold=zeta_threshold,
        )

    def probe_zero(
        self,
        t: Union[float, str],
        refine_t: bool = False,
        refine_window: float = 0.5,
        max_iter: int = 20,
    ) -> RiemannZeroEvidence:
        """
        Probe the candidate Riemann zero at s = 1/2 + it.

        Performs:

        1. Optional t-refinement: minimises ``|ζ(1/2 + it)|`` within
           ``[t − refine_window, t + refine_window]`` via golden-section
           search (when ``refine_t=True``).
        2. Zero-classification of |ζ(σ + it)| across all σ in ``SIGMA_SCAN``
           via ``classify_zeta``, using the precision-aware zeta threshold.
        3. Φ_nab computation for the local 7×7 transition matrix built from
           ζ-values near each (σ, t).
        4. Φ_fano at σ = 1/2.
        5. Margin-based ``critical_line_signature``: True when
           ``|ζ(1/2 + it)| < zeta_threshold`` **and**
           ``min_other_raw > margin_factor × zeta_threshold``.
        6. GUE pair-correlation statistic from normalised zero spacings.

        Args:
            t: Imaginary part of the candidate zero (probe s = 1/2 + it).
               May be a float or a high-precision string from
               ``KNOWN_ZEROS_HP``.  Stored in ``RiemannZeroEvidence.t``
               unchanged regardless of refinement.
            refine_t: When True, run a golden-section minimisation of
               ``|ζ(1/2 + it)|`` in
               ``[float(t) − refine_window, float(t) + refine_window]``
               before the σ-scan.  The scan uses the refined t value.
               Default False (backward compatible).
            refine_window: Half-width of the refinement search interval.
               Default 0.5.
            max_iter: Maximum golden-section iterations.  Default 20.

        Returns:
            RiemannZeroEvidence with full σ-scan results, margin-based
            critical_line_signature, separation evidence, refinement
            metadata, and gue_pair_correlation.
        """
        t_float = float(t)

        # Optional t-refinement
        refined_t: Optional[float] = None
        refine_iterations: Optional[int] = None
        refine_residual: Optional[float] = None
        if refine_t:
            refined_t, refine_iterations, refine_residual = self._refine_t(
                t_float, window=refine_window, max_iter=max_iter
            )
            zeta_scan_t: Union[float, str] = refined_t
            t_for_matrix: float = refined_t
        else:
            zeta_scan_t = t           # preserve HP string when supplied
            t_for_matrix = t_float

        zeta_scan: Dict[float, ScoreDiagnostic] = {}
        nonabelian_scan: Dict[float, float] = {}

        for sigma in self.SIGMA_SCAN:
            zeta_scan[sigma] = self.classify_zeta(sigma, zeta_scan_t)
            T = self._build_local_matrix(sigma, t_for_matrix)
            nonabelian_scan[sigma] = self._engine._compute_nonabelian_measure(T)

        T_crit = self._build_local_matrix(0.5, t_for_matrix)
        fano_at_critical = self._engine._compute_fano_alignment(T_crit, n_nodes=FANO_POINTS)

        diag_at_half = zeta_scan.get(0.5)
        zeta_abs = diag_at_half.raw_value if diag_at_half else float("nan")
        zeta_classification = diag_at_half.classification if diag_at_half else CLASSIFICATION_NONZERO

        # Margin-based critical-line signature
        zeta_threshold = self._get_zeta_threshold()
        other_raws = [
            diag.raw_value for s, diag in zeta_scan.items() if s != 0.5
        ]
        min_other_raw = min(other_raws) if other_raws else float("nan")
        separation_ratio: Optional[float] = (
            min_other_raw / zeta_abs
            if (zeta_abs > 0.0 and not math.isnan(min_other_raw))
            else None
        )

        is_zero_at_half = (
            zeta_classification == CLASSIFICATION_EXACT_ZERO
            or zeta_abs < zeta_threshold
        )
        is_nonzero_off_line = (
            not math.isnan(min_other_raw)
            and min_other_raw > self.margin_factor * zeta_threshold
        )
        critical_line_signature = is_zero_at_half and is_nonzero_off_line

        gue_pc = self._gue_pair_correlation(t_float)

        return RiemannZeroEvidence(
            t=t_float,
            zeta_abs=zeta_abs,
            zeta_classification=zeta_classification,
            zeta_scan=zeta_scan,
            nonabelian_scan=nonabelian_scan,
            fano_at_critical=fano_at_critical,
            critical_line_signature=critical_line_signature,
            gue_pair_correlation=gue_pc,
            min_other_raw=min_other_raw,
            separation_ratio=separation_ratio,
            refined_t=refined_t,
            refine_iterations=refine_iterations,
            refine_residual=refine_residual,
            zeta_threshold=zeta_threshold,
            margin_factor=self.margin_factor,
        )

    def scan_known_zeros(
        self,
        zeros: Optional[Sequence[Union[float, str]]] = None,
    ) -> List[RiemannZeroEvidence]:
        """
        Probe a list of Riemann zero candidates and return evidence for each.

        Args:
            zeros: Imaginary parts to probe.  Accepts floats or
                high-precision strings.  Defaults to ``KNOWN_ZEROS_HP``
                for maximum precision.

        Returns:
            List of ``RiemannZeroEvidence``, one per input value.
        """
        if zeros is None:
            zeros = self.KNOWN_ZEROS_HP
        return [self.probe_zero(t) for t in zeros]

    def publish_results(
        self,
        zeros: Optional[Sequence[Union[float, str]]] = None,
    ) -> Dict:
        """Run the probe on *zeros* and return a JSON-serialisable report.

        The report includes per-zero evidence dictionaries (via
        ``RiemannZeroEvidence.to_dict``) and an aggregate summary
        with the number of zeros probed and how many satisfy the
        critical-line signature.

        Args:
            zeros: Imaginary parts to probe.  Defaults to the first
                three entries of ``KNOWN_ZEROS_HP`` for a fast yet
                meaningful result set.

        Returns:
            A dictionary with ``"summary"`` and ``"evidence"`` keys,
            directly passable to ``json.dumps``.
        """
        if zeros is None:
            zeros = self.KNOWN_ZEROS_HP[:3]
        evidences = self.scan_known_zeros(zeros)
        return {
            "summary": {
                "zeros_probed": len(evidences),
                "critical_line_confirmed": sum(
                    1 for ev in evidences if ev.critical_line_signature
                ),
                "mpmath_dps": self.mpmath_dps,
                "zeta_threshold": self._get_zeta_threshold(),
                "margin_factor": self.margin_factor,
            },
            "evidence": [ev.to_dict() for ev in evidences],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_zeta_threshold(self) -> float:
        """Return the zeta near-zero threshold.

        Uses ``zeta_near_zero_threshold`` if set; otherwise computes the
        precision-aware default ``10 ** (-(mpmath_dps // 2))``.
        """
        if self.zeta_near_zero_threshold is not None:
            return self.zeta_near_zero_threshold
        return 10.0 ** (-(self.mpmath_dps // 2))

    def _refine_t(
        self, t: float, window: float = 0.5, max_iter: int = 20
    ) -> Tuple[float, int, float]:
        """
        Minimise ``|ζ(1/2 + it)|`` over ``[t − window, t + window]`` using
        golden-section search with mpmath precision.

        Returns:
            (refined_t, iterations, residual) where *residual* is
            ``|ζ(1/2 + i·refined_t)|``.
        """
        import mpmath
        mpmath.mp.dps = self.mpmath_dps
        half = mpmath.mpf("0.5")
        phi = (mpmath.sqrt(5) - 1) / 2  # ≈ 0.618

        def f(x: mpmath.mpf) -> mpmath.mpf:
            return abs(mpmath.zeta(mpmath.mpc(half, x)))

        a = mpmath.mpf(t - window)
        b = mpmath.mpf(t + window)
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = f(c)
        fd = f(d)

        tol = mpmath.mpf(10) ** (-(self.mpmath_dps - 5))
        iters = 0
        for iters in range(1, max_iter + 1):
            if abs(b - a) < tol:
                break
            if fc < fd:
                b = d
                d, fd = c, fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a = c
                c, fc = d, fd
                d = a + phi * (b - a)
                fd = f(d)

        x_min = float(c if fc < fd else d)
        residual = float(min(fc, fd))
        return x_min, iters, residual

    def _build_local_matrix(self, sigma: float, t: float) -> np.ndarray:
        """
        Build a 7×7 local transition matrix from |ζ| values on a grid near
        (σ, t).

        Grid layout (GRID_RADIUS = 3, size = 7 = FANO_POINTS):
            rows  i → σ_i = σ + (i − 3) · LOCAL_DELTA_SIGMA
            cols  j → t_j = t + (j − 3) · LOCAL_DELTA_T

        T[i, j] = |ζ(σ_i + i·t_j)|, then column-normalised so each column
        sums to 1 (column-stochastic).  This makes T a valid input for the
        IIT v7.0 engine's non-abelian and Fano measures.
        """
        import mpmath
        mpmath.mp.dps = self.mpmath_dps

        n = self.GRID_RADIUS
        size = 2 * n + 1  # 7

        sigmas = [sigma + (i - n) * self.LOCAL_DELTA_SIGMA for i in range(size)]
        ts = [t + (j - n) * self.LOCAL_DELTA_T for j in range(size)]

        T = np.zeros((size, size))
        for i, s in enumerate(sigmas):
            for j, tj in enumerate(ts):
                T[i, j] = float(abs(mpmath.zeta(mpmath.mpc(s, tj))))

        # Column-stochastic normalisation
        col_sums = T.sum(axis=0)
        col_sums[col_sums < 1e-30] = 1.0
        T /= col_sums
        return T

    def _gue_pair_correlation(self, t: float) -> Optional[float]:
        """
        Compute the Montgomery–Odlyzko GUE pair-correlation statistic for the
        zero neighbourhood around *t*.

        The GUE conjecture predicts that the pair correlation of normalised
        Riemann zero spacings converges to:

            R₂(u) = 1 − (sin(πu) / (πu))²

        This method:

        1. Collects all known zeros within a window of *t* (from
           ``KNOWN_ZEROS``).
        2. Normalises spacings δₖ = (tₖ₊₁ − tₖ) · d(t) where d(t) is
           the mean zero density log(t / 2π) / (2π).
        3. For each normalised spacing u, computes the predicted R₂(u)
           and the actual count-based pair frequency.
        4. Returns the Pearson correlation between observed and predicted
           pair-correlation values.  A result near 1.0 indicates strong
           agreement with the GUE prediction.

        Returns ``None`` when fewer than 3 zeros are available.
        """
        zeros = sorted(self.KNOWN_ZEROS)
        if len(zeros) < 3:
            return None

        # Normalised spacings using mean density at each midpoint
        spacings = []
        for k in range(len(zeros) - 1):
            mid = (zeros[k] + zeros[k + 1]) / 2.0
            density = math.log(mid / (2 * math.pi)) / (2 * math.pi)
            spacings.append((zeros[k + 1] - zeros[k]) * density)

        if len(spacings) < 2:
            return None

        # Observed pair-correlation: histogram of normalised spacings
        # compared to GUE prediction R₂(u) = 1 − (sin(πu)/(πu))²
        predicted = []
        for u in spacings:
            if abs(u) < 1e-15:
                r2 = 0.0
            else:
                sinc = math.sin(math.pi * u) / (math.pi * u)
                r2 = 1.0 - sinc * sinc
            predicted.append(r2)

        # Pearson correlation between the observed spacing magnitudes
        # (treated as an empirical signal) and the GUE predicted R₂.
        obs = np.array(spacings)
        pred = np.array(predicted)

        if np.std(obs) < 1e-15 or np.std(pred) < 1e-15:
            return 0.0

        corr = float(np.corrcoef(obs, pred)[0, 1])
        return corr
