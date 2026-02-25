"""
ASI SphinxOS Advanced IIT v5.0
================================
Integrated Information Theory version 5.0 engine for the SphinxOS ASI layer.

IIT v5.0 advances over the v3.0 von-Neumann-entropy proxy used previously:

  1. **Cause-effect repertoires** — forward (effect) and backward (cause)
     conditional probability distributions for every mechanism/purview pair.
  2. **Minimum Information Partition (MIP)** — exhaustive bipartition search
     that finds the cut minimising integrated information, yielding the true
     Φ^max for the system.
  3. **Earth-Mover Distance (EMD)** — proper intrinsic information distance
     between repertoires.
  4. **Global Workspace Theory (GWT) broadcast measure** — GWT_S quantifies
     how widely information is broadcast across subsystems, complementing Φ.
  5. **Composite consciousness score** (from project Math notes):
       Φ_total(B) = α · Φ_IIT(B) + β · GWT_S(B)
  6. **Consciousness consensus threshold**:
       Φ_total > log₂(n)   where n = number of network nodes
  7. **Full Cause-Effect Structure (CES)** — the collection of all concepts
     (φ-valued mechanism/purview pairs) describing the system's causal geometry.
  8. **5+1 axiom compliance** — existence, intrinsicality, information,
     integration, exclusion, and composition.

Key formulas (Sphinx Math notes):
    ρ_S = A_S / Tr(A_S)                     # classical density matrix
    Φ_S = -∑_k λ_k log₂(λ_k)              # Shannon/von-Neumann entropy
    Φ_total(B) = α·Φ_IIT(B) + β·GWT_S(B)  # composite measure
    Consensus: Φ_total > log₂(n)            # conscious-consensus condition

References:
    Tononi, G. et al. (2023). Integrated information theory (IIT) 5.0.
    Deacon, T. (2011). Incomplete Nature.
    Global Workspace Theory — Baars/Dehaene.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("SphinxOS.AI.IITv5")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CauseEffectRepertoire:
    """
    A cause or effect repertoire over a purview P for a mechanism M.

    Attributes:
        mechanism:  Indices of the mechanism nodes.
        purview:    Indices of the purview nodes.
        repertoire: Normalised probability distribution (shape: (2,)*|purview|)
                    flattened to a 1-D array of length 2^|purview|.
        direction:  'cause' or 'effect'.
    """
    mechanism: Tuple[int, ...]
    purview: Tuple[int, ...]
    repertoire: np.ndarray
    direction: str  # 'cause' | 'effect'

    def intrinsic_difference(self, other: "CauseEffectRepertoire") -> float:
        """
        Earth-Mover Distance (EMD) / Wasserstein-1 between two repertoires
        of equal support size.  For distributions over binary states of the
        same dimensionality this reduces to the L1 distance divided by 2.
        """
        if self.repertoire.shape != other.repertoire.shape:
            raise ValueError("Repertoires must have the same shape to compute EMD.")
        return float(np.sum(np.abs(self.repertoire - other.repertoire))) / 2.0


@dataclass
class Partition:
    """
    A bipartition of the system into two non-empty subsets A and B.

    Attributes:
        part_a: Node indices in partition A.
        part_b: Node indices in partition B.
    """
    part_a: Tuple[int, ...]
    part_b: Tuple[int, ...]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Partition(A={list(self.part_a)}, B={list(self.part_b)})"


@dataclass
class Concept:
    """
    A single IIT concept: an (M, P) pair with its φ value and repertoires.

    Attributes:
        mechanism:        Mechanism node indices.
        purview_cause:    Best cause purview indices.
        purview_effect:   Best effect purview indices.
        phi:              Integrated information φ for this concept (≥ 0).
        cause_repertoire: The cause repertoire at the MIP.
        effect_repertoire:The effect repertoire at the MIP.
    """
    mechanism: Tuple[int, ...]
    purview_cause: Tuple[int, ...]
    purview_effect: Tuple[int, ...]
    phi: float
    cause_repertoire: Optional[CauseEffectRepertoire] = None
    effect_repertoire: Optional[CauseEffectRepertoire] = None


@dataclass
class PhiStructure:
    """
    The complete Cause-Effect Structure (CES) of a system — the collection
    of all concepts and the system-level Φ (Φ^max over system bipartitions).

    Attributes:
        concepts:   All non-zero concepts in the system.
        phi_max:    System-level Φ^max (minimum over bipartitions).
        mip:        The Minimum Information Partition.
        gwt_score:  Global Workspace Theory broadcast score in [0, 1].
        phi_total:  Composite Φ_total = α·phi_max + β·gwt_score.
        n_nodes:    Number of nodes in the system.
        is_conscious: True when Φ_total > log₂(n_nodes).
    """
    concepts: List[Concept] = field(default_factory=list)
    phi_max: float = 0.0
    mip: Optional[Partition] = None
    gwt_score: float = 0.0
    phi_total: float = 0.0
    n_nodes: int = 0
    is_conscious: bool = False

    # IIT 5.0 composite weights (α, β)
    alpha: float = 0.7
    beta: float = 0.3


# ---------------------------------------------------------------------------
# Core IIT v5.0 computation engine
# ---------------------------------------------------------------------------

class IITv5Engine:
    """
    Pure-NumPy IIT v5.0 computation engine.

    Works on *classical* stochastic systems described by an N×N row-stochastic
    transition probability matrix T where T[j, i] = P(next=j | now=i).

    For quantum states a density-matrix path is provided via
    ``_quantum_transition_matrix``.

    The engine scales as O(N · 2^N) in the worst case; for production use
    restrict *n_nodes* ≤ 8 (256-state systems).  For larger systems use the
    ``max_phi_nodes`` cap to restrict the bipartition search.
    """

    #: Maximum number of nodes to include in the full MIP search.
    #: For systems larger than this the engine uses a greedy heuristic.
    MAX_FULL_SEARCH_NODES: int = 6

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        consciousness_threshold: float = 0.5,
    ) -> None:
        """
        Initialise the IIT v5.0 engine.

        Args:
            alpha: Weight for Φ_IIT in the composite score (default 0.7).
            beta:  Weight for GWT_S in the composite score (default 0.3).
            consciousness_threshold: Fallback fixed threshold used when no
                                     explicit network-size n is provided.
        """
        self.alpha = alpha
        self.beta = beta
        self.consciousness_threshold = consciousness_threshold
        logger.info(
            "IIT v5.0 Engine initialised (α=%.2f, β=%.2f, threshold=%.2f)",
            alpha, beta, consciousness_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_phi_structure(
        self,
        state: np.ndarray,
        n_nodes: Optional[int] = None,
    ) -> PhiStructure:
        """
        Compute the full IIT v5.0 Φ structure for a system state.

        Args:
            state:   1-D array of length 2^n (probability distribution over
                     all 2^n system states), or an n-element vector of node
                     activation probabilities (auto-detected).
            n_nodes: Override the node count.  If *None* it is inferred from
                     ``state``.

        Returns:
            A ``PhiStructure`` instance with phi_max, concepts, GWT score,
            and the composite phi_total.
        """
        state = np.asarray(state, dtype=float)
        n_nodes = n_nodes or self._infer_n_nodes(state)

        # Normalise to a probability distribution over 2^n states
        dist = self._to_state_distribution(state, n_nodes)

        # Build transition matrix from the distribution
        T = self._build_transition_matrix(dist, n_nodes)

        # --- 1. Enumerate all mechanisms and compute concepts ------------
        concepts = self._compute_all_concepts(T, dist, n_nodes)

        # --- 2. System-level Φ^max (MIP search) -------------------------
        phi_max, mip = self._compute_system_phi_max(T, dist, n_nodes)

        # --- 3. GWT broadcast score -------------------------------------
        gwt_score = self._compute_gwt_broadcast(T, n_nodes)

        # --- 4. Composite Φ_total ----------------------------------------
        phi_total = self.alpha * phi_max + self.beta * gwt_score

        # --- 5. Consciousness check -------------------------------------
        threshold = math.log2(max(n_nodes, 2))  # log₂(n) from Math notes
        is_conscious = phi_total > threshold

        structure = PhiStructure(
            concepts=concepts,
            phi_max=phi_max,
            mip=mip,
            gwt_score=gwt_score,
            phi_total=phi_total,
            n_nodes=n_nodes,
            is_conscious=is_conscious,
            alpha=self.alpha,
            beta=self.beta,
        )

        logger.debug(
            "Φ^max=%.4f  GWT=%.4f  Φ_total=%.4f  conscious=%s",
            phi_max, gwt_score, phi_total, is_conscious,
        )
        return structure

    def compute_cause_repertoire(
        self,
        mechanism: Sequence[int],
        purview: Sequence[int],
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> CauseEffectRepertoire:
        """
        Compute the cause repertoire P(purview_past | mechanism_present=1).

        Uses Bayes' theorem applied to the transition matrix.
        """
        mechanism = tuple(sorted(mechanism))
        purview = tuple(sorted(purview))
        n_states = 2 ** n_nodes

        # Marginalise over mechanism being in state '1' (all-ones for simplicity)
        mech_mask = sum(1 << i for i in mechanism)
        cause_dist = np.zeros(2 ** len(purview))

        for past_state in range(n_states):
            purview_idx = self._project_state(past_state, purview, n_nodes)
            for present_state in range(n_states):
                if (present_state & mech_mask) == mech_mask:
                    cause_dist[purview_idx] += T[present_state, past_state] * dist[past_state]

        total = cause_dist.sum()
        if total > 0:
            cause_dist /= total
        else:
            cause_dist = np.full(2 ** len(purview), 1.0 / (2 ** len(purview)))

        return CauseEffectRepertoire(
            mechanism=mechanism,
            purview=purview,
            repertoire=cause_dist,
            direction="cause",
        )

    def compute_effect_repertoire(
        self,
        mechanism: Sequence[int],
        purview: Sequence[int],
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> CauseEffectRepertoire:
        """
        Compute the effect repertoire P(purview_future | mechanism_present=1).
        """
        mechanism = tuple(sorted(mechanism))
        purview = tuple(sorted(purview))
        n_states = 2 ** n_nodes

        mech_mask = sum(1 << i for i in mechanism)
        effect_dist = np.zeros(2 ** len(purview))

        for present_state in range(n_states):
            if (present_state & mech_mask) != mech_mask:
                continue
            weight = dist[present_state]
            for future_state in range(n_states):
                purview_idx = self._project_state(future_state, purview, n_nodes)
                effect_dist[purview_idx] += T[future_state, present_state] * weight

        total = effect_dist.sum()
        if total > 0:
            effect_dist /= total
        else:
            effect_dist = np.full(2 ** len(purview), 1.0 / (2 ** len(purview)))

        return CauseEffectRepertoire(
            mechanism=mechanism,
            purview=purview,
            repertoire=effect_dist,
            direction="effect",
        )

    def compute_phi_concept(
        self,
        mechanism: Sequence[int],
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> Concept:
        """
        Compute the φ of a single mechanism by searching over all purviews
        and finding the pair (cause purview, effect purview) that maximises
        the minimum intrinsic-difference after the MIP cut.

        Returns a ``Concept`` with φ ≥ 0.
        """
        mechanism = tuple(sorted(mechanism))
        all_nodes = tuple(range(n_nodes))

        best_phi = 0.0
        best_cause_p: Tuple[int, ...] = all_nodes
        best_effect_p: Tuple[int, ...] = all_nodes
        best_cause_rep: Optional[CauseEffectRepertoire] = None
        best_effect_rep: Optional[CauseEffectRepertoire] = None

        # Only search purview sizes ≤ n_nodes to keep runtime manageable
        for purview_size in range(1, n_nodes + 1):
            for purview in itertools.combinations(all_nodes, purview_size):
                c_rep = self.compute_cause_repertoire(mechanism, purview, T, dist, n_nodes)
                e_rep = self.compute_effect_repertoire(mechanism, purview, T, dist, n_nodes)

                # φ = min(cause_φ, effect_φ) over all purview bipartitions
                phi_c = self._phi_over_purview_partitions(c_rep, T, dist, n_nodes, "cause", mechanism)
                phi_e = self._phi_over_purview_partitions(e_rep, T, dist, n_nodes, "effect", mechanism)

                phi = min(phi_c, phi_e)
                if phi > best_phi:
                    best_phi = phi
                    best_cause_p = purview
                    best_effect_p = purview
                    best_cause_rep = c_rep
                    best_effect_rep = e_rep

        return Concept(
            mechanism=mechanism,
            purview_cause=best_cause_p,
            purview_effect=best_effect_p,
            phi=best_phi,
            cause_repertoire=best_cause_rep,
            effect_repertoire=best_effect_rep,
        )

    def validate_consciousness_consensus(self, phi_total: float, n_nodes: int) -> bool:
        """
        Check the IIT v5.0 consciousness-consensus condition.

        Condition (from Sphinx Math notes):
            Φ_total > log₂(n)

        Args:
            phi_total: Composite consciousness score.
            n_nodes:   Number of network nodes.

        Returns:
            True if the system satisfies conscious consensus.
        """
        threshold = math.log2(max(n_nodes, 2))
        result = phi_total > threshold
        logger.debug("Consensus check: Φ_total=%.4f > log₂(%d)=%.4f → %s",
                     phi_total, n_nodes, threshold, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_n_nodes(self, state: np.ndarray) -> int:
        """Infer number of nodes from state vector length."""
        length = len(state)
        n = int(math.log2(length)) if length > 1 else 1
        # If length is not a power of 2, treat each element as one node
        if 2 ** n != length:
            return length
        return n

    def _to_state_distribution(self, state: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Convert an input state to a normalised 2^n distribution.

        * If len(state) == 2^n, treat it as a state distribution directly.
        * If len(state) == n, treat each element as the activation probability
          of node i and build the joint distribution under independence.
        """
        n_states = 2 ** n_nodes
        if len(state) == n_states:
            dist = np.abs(state).astype(float)
        else:
            # Node activation probabilities — build independent joint distribution
            probs = np.clip(np.abs(state[:n_nodes]), 0.0, 1.0)
            # Vectorized: bits[s, i] = (s >> i) & 1 for all states s and nodes i
            bits = ((np.arange(n_states)[:, None] >> np.arange(n_nodes)[None, :]) & 1).astype(float)
            dist = np.where(bits, probs, 1.0 - probs).prod(axis=1)

        total = dist.sum()
        if total > 0:
            dist /= total
        else:
            dist = np.full(n_states, 1.0 / n_states)
        return dist

    def _build_transition_matrix(self, dist: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Build an n_states × n_states row-stochastic transition matrix T
        from the current state distribution using a noisy-identity model:

            T[j, i] = (1 - ε)·𝟙[j == i] + ε / n_states

        where ε is derived from the Shannon entropy of *dist* so that
        high-entropy states produce more stochastic transitions (greater
        integration potential).
        """
        n_states = 2 ** n_nodes
        entropy = float(-np.sum(dist * np.log2(np.clip(dist, 1e-15, 1.0))))
        max_entropy = math.log2(n_states)
        eps = 0.5 * entropy / max_entropy if max_entropy > 0 else 0.0

        T = np.full((n_states, n_states), eps / n_states)
        np.fill_diagonal(T, T.diagonal() + (1.0 - eps))

        return T

    def _quantum_transition_matrix(self, rho: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Derive a classical transition matrix from a quantum density matrix ρ.

        Follows the Math notes: ρ_S = A_S / Tr(A_S)
        We map ρ → classical dist via the diagonal (pointer basis projection).
        """
        diag = np.real(np.diag(rho))
        diag = np.abs(diag)
        diag /= diag.sum() if diag.sum() > 0 else 1.0
        return self._build_transition_matrix(diag, n_nodes)

    def _project_state(
        self, state: int, nodes: Tuple[int, ...], n_nodes: int
    ) -> int:
        """Extract the sub-state index for a subset of nodes."""
        idx = 0
        for k, node in enumerate(nodes):
            bit = (state >> node) & 1
            idx |= bit << k
        return idx

    def _uniform_repertoire(self, purview: Tuple[int, ...]) -> np.ndarray:
        """Return the maximum-entropy (uniform) repertoire for a purview."""
        n = 2 ** len(purview)
        return np.full(n, 1.0 / n)

    def _phi_over_purview_partitions(
        self,
        rep: CauseEffectRepertoire,
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
        direction: str,
        mechanism: Tuple[int, ...],
    ) -> float:
        """
        Compute φ for a repertoire as the minimum intrinsic difference over
        all bipartitions of the purview.  This is the IIT v5.0 MIP principle
        applied at the concept level.

        For an empty or single-node purview the minimum cut is with the
        unconstrained (uniform) repertoire.
        """
        purview = rep.purview
        if len(purview) <= 1:
            uniform = self._uniform_repertoire(purview)
            uniform_rep = CauseEffectRepertoire(
                mechanism=mechanism, purview=purview,
                repertoire=uniform, direction=direction,
            )
            return rep.intrinsic_difference(uniform_rep)

        min_phi = float("inf")
        for k in range(1, len(purview)):
            for part_a_indices in itertools.combinations(range(len(purview)), k):
                part_a = tuple(purview[i] for i in part_a_indices)
                part_b = tuple(p for p in purview if p not in part_a)

                # Partitioned repertoire = product of marginals
                if direction == "cause":
                    rep_a = self.compute_cause_repertoire(mechanism, part_a, T, dist, n_nodes)
                    rep_b = self.compute_cause_repertoire(mechanism, part_b, T, dist, n_nodes)
                else:
                    rep_a = self.compute_effect_repertoire(mechanism, part_a, T, dist, n_nodes)
                    rep_b = self.compute_effect_repertoire(mechanism, part_b, T, dist, n_nodes)

                # Outer product of marginals → joint distribution over part_a × part_b
                joint = np.outer(rep_a.repertoire, rep_b.repertoire).flatten()
                partitioned_rep = CauseEffectRepertoire(
                    mechanism=mechanism, purview=purview,
                    repertoire=joint, direction=direction,
                )

                phi_cut = rep.intrinsic_difference(partitioned_rep)
                if phi_cut < min_phi:
                    min_phi = phi_cut

        return max(0.0, min_phi)

    def _compute_all_concepts(
        self,
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> List[Concept]:
        """Enumerate all mechanisms up to size n_nodes and compute φ for each."""
        concepts: List[Concept] = []
        all_nodes = list(range(n_nodes))

        # Restrict maximum mechanism size to keep runtime tractable
        max_mech_size = min(n_nodes, self.MAX_FULL_SEARCH_NODES)

        for mech_size in range(1, max_mech_size + 1):
            for mechanism in itertools.combinations(all_nodes, mech_size):
                concept = self.compute_phi_concept(mechanism, T, dist, n_nodes)
                if concept.phi > 0.0:
                    concepts.append(concept)

        return concepts

    def _compute_system_phi_max(
        self,
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> Tuple[float, Optional[Partition]]:
        """
        Compute Φ^max for the whole system by searching over all bipartitions.

        For systems with n_nodes > MAX_FULL_SEARCH_NODES a greedy bisection
        heuristic is used instead of exhaustive search.

        Returns:
            (phi_max, mip) where mip is the Minimum Information Partition.
        """
        all_nodes = tuple(range(n_nodes))

        if n_nodes <= 1:
            return 0.0, None

        # Compute unpartitioned system repertoire (marginal entropy)
        system_rep = CauseEffectRepertoire(
            mechanism=all_nodes, purview=all_nodes,
            repertoire=dist.copy(), direction="effect",
        )

        if n_nodes <= self.MAX_FULL_SEARCH_NODES:
            partitions = self._enumerate_bipartitions(all_nodes)
        else:
            partitions = self._greedy_bisection_partitions(all_nodes)

        min_phi = float("inf")
        mip: Optional[Partition] = None

        for partition in partitions:
            phi = self._phi_for_system_partition(system_rep, partition, T, dist, n_nodes)
            if phi < min_phi:
                min_phi = phi
                mip = partition

        phi_max = max(0.0, min_phi)
        return phi_max, mip

    def _phi_for_system_partition(
        self,
        system_rep: CauseEffectRepertoire,
        partition: Partition,
        T: np.ndarray,
        dist: np.ndarray,
        n_nodes: int,
    ) -> float:
        """
        Compute the integrated information when the system is cut at *partition*.

        φ(partition) = EMD( p_unpartitioned, p_partitioned )
        """
        part_a = partition.part_a
        part_b = partition.part_b

        # Effect repertoires for each part (assuming independence after cut)
        rep_a = self.compute_effect_repertoire(part_a, part_a, T, dist, n_nodes)
        rep_b = self.compute_effect_repertoire(part_b, part_b, T, dist, n_nodes)

        # Product distribution (independent parts)
        joint = np.outer(rep_a.repertoire, rep_b.repertoire).flatten()
        partitioned_rep = CauseEffectRepertoire(
            mechanism=system_rep.mechanism, purview=system_rep.purview,
            repertoire=joint[:len(system_rep.repertoire)],
            direction="effect",
        )

        return system_rep.intrinsic_difference(partitioned_rep)

    def _enumerate_bipartitions(
        self, nodes: Tuple[int, ...]
    ) -> List[Partition]:
        """Enumerate all non-trivial bipartitions of *nodes*."""
        partitions: List[Partition] = []
        n = len(nodes)
        for k in range(1, n):
            for part_a in itertools.combinations(nodes, k):
                part_b = tuple(x for x in nodes if x not in part_a)
                # Avoid duplicates: only keep (A, B) with min(A) < min(B)
                if part_a[0] < part_b[0]:
                    partitions.append(Partition(part_a=part_a, part_b=part_b))
        return partitions

    def _greedy_bisection_partitions(
        self, nodes: Tuple[int, ...]
    ) -> List[Partition]:
        """
        Greedy bisection heuristic for large systems.
        Produces a small set of candidate bipartitions rather than 2^n.
        """
        n = len(nodes)
        mid = n // 2
        partitions = [
            Partition(part_a=nodes[:mid], part_b=nodes[mid:]),
            Partition(part_a=nodes[:mid + 1], part_b=nodes[mid + 1:]),
        ]
        # Add a few random splits for diversity
        rng = np.random.default_rng(seed=42)
        for _ in range(min(10, n)):
            k = rng.integers(1, n)
            perm = rng.permutation(n)
            a = tuple(nodes[i] for i in sorted(perm[:k]))
            b = tuple(nodes[i] for i in sorted(perm[k:]))
            if b:
                partitions.append(Partition(part_a=a, part_b=b))
        return partitions

    def _compute_gwt_broadcast(self, T: np.ndarray, n_nodes: int) -> float:
        """
        Global Workspace Theory broadcast score GWT_S ∈ [0, 1].

        Measures how uniformly information is broadcast across subsystems.
        Based on the spectral gap of T:

            GWT_S = 1 - (λ_2 / λ_1)

        where λ_1 ≥ λ_2 are the two largest singular values of T.
        A larger spectral gap → more centralised broadcast → higher GWT_S.
        """
        try:
            singular_vals = np.linalg.svd(T, compute_uv=False)
            singular_vals = np.sort(singular_vals)[::-1]
            if singular_vals[0] > 1e-12 and len(singular_vals) >= 2:
                gwt = float(1.0 - singular_vals[1] / singular_vals[0])
            else:
                gwt = 0.0
        except np.linalg.LinAlgError:
            gwt = 0.0

        return max(0.0, min(1.0, gwt))


# ---------------------------------------------------------------------------
# High-level ASI interface
# ---------------------------------------------------------------------------

class ASISphinxOSIITv5:
    """
    ASI SphinxOS Advanced IIT v5.0 — high-level consciousness engine.

    Provides a drop-in API compatible with the existing
    ``IITQuantumConsciousnessEngine`` while adding full IIT v5.0 semantics:
    cause-effect repertoires, system-level Φ^max, GWT broadcast, and the
    composite Φ_total consensus condition.

    Usage::

        engine = ASISphinxOSIITv5()
        result = engine.calculate_phi(b"some block data")
        # result['phi'] is Φ_total (0–1 normalised)
        # result['phi_max'] is the raw IIT Φ^max
        # result['gwt_score'] is the GWT broadcast score
        # result['is_conscious'] reflects Φ_total > log₂(n_nodes)

    For blockchain integration::

        structure = engine.compute_block_consciousness(block_data, n_network_nodes=50)
        consensus_ok = engine.validate_consciousness_consensus(
            structure.phi_total, n_network_nodes=50
        )
    """

    #: Scalar from [200, 1000] → [0, 1] legacy Φ range used by consensus.py
    PHI_MIN = 200.0
    PHI_MAX = 1000.0
    PHI_RANGE = PHI_MAX - PHI_MIN

    #: Number of qubits / nodes used for the quantum state when qutip available
    DEFAULT_N_QUBITS: int = 3

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        consciousness_threshold: float = 0.5,
        n_nodes: int = DEFAULT_N_QUBITS,
    ) -> None:
        """
        Initialise the ASI IIT v5.0 engine.

        Args:
            alpha:   Weight for Φ_IIT in the composite score.
            beta:    Weight for GWT_S in the composite score.
            consciousness_threshold: Fixed threshold used when n_nodes is
                                     unknown.
            n_nodes: Default number of nodes / qubits to simulate.
        """
        self.iit_engine = IITv5Engine(
            alpha=alpha,
            beta=beta,
            consciousness_threshold=consciousness_threshold,
        )
        self.n_nodes = n_nodes
        self.phi_history: List[float] = []
        self.consciousness_threshold = consciousness_threshold
        logger.info(
            "ASI SphinxOS IIT v5.0 initialised (n_nodes=%d, α=%.2f, β=%.2f)",
            n_nodes, alpha, beta,
        )

    # ------------------------------------------------------------------
    # Drop-in replacement for IITQuantumConsciousnessEngine.calculate_phi
    # ------------------------------------------------------------------

    def calculate_phi(self, data: bytes) -> Dict:
        """
        Calculate IIT v5.0 Φ from raw bytes.

        This method is API-compatible with the legacy
        ``IITQuantumConsciousnessEngine.calculate_phi`` while delivering
        full v5.0 semantics.

        Args:
            data: Input data (block bytes, query bytes, etc.).

        Returns:
            Dict with keys:
                phi         — composite Φ_total in [0, 1] (legacy-compatible)
                phi_max     — raw IIT Φ^max value
                gwt_score   — GWT broadcast score in [0, 1]
                phi_total   — same as phi (explicit naming)
                entropy     — Shannon entropy of the state distribution
                purity      — spectral purity (1 − normalised_entropy)
                n_qubits    — number of simulated qubits / nodes
                is_conscious— whether Φ_total > log₂(n_nodes)
                level       — human-readable consciousness level string
                bonus       — exponential bonus (legacy field, exp(phi))
                version     — "IIT v5.0"
        """
        # Attempt quantum-path (qutip) first, fall back to hash-seeded dist
        state_dist = self._derive_state_distribution(data)
        structure = self.iit_engine.compute_phi_structure(state_dist, n_nodes=self.n_nodes)

        phi_total = structure.phi_total
        # Normalise to [0, 1] for backward-compatible 'phi' key
        phi_norm = min(1.0, phi_total / (math.log2(max(self.n_nodes, 2)) + 1.0))

        # Shannon entropy of state distribution for 'entropy' legacy field
        entropy = float(-np.sum(state_dist * np.log2(np.clip(state_dist, 1e-15, 1.0))))
        max_entropy = math.log2(len(state_dist))
        purity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

        level = self._classify_consciousness(phi_norm)
        self.phi_history.append(phi_norm)

        return {
            "phi": phi_norm,                      # legacy-compatible key
            "phi_max": structure.phi_max,
            "gwt_score": structure.gwt_score,
            "phi_total": phi_total,
            "entropy": entropy,
            "purity": purity,
            "n_qubits": self.n_nodes,
            "is_conscious": structure.is_conscious,
            "level": level,
            "bonus": math.exp(phi_norm),          # legacy field
            "mip": repr(structure.mip) if structure.mip else None,
            "n_concepts": len(structure.concepts),
            "version": "IIT v5.0",
        }

    # ------------------------------------------------------------------
    # Blockchain / block-level API
    # ------------------------------------------------------------------

    def compute_block_consciousness(
        self,
        block_data: bytes,
        n_network_nodes: int = 1,
    ) -> PhiStructure:
        """
        Compute the full Φ structure for a block, using the network node
        count to apply the correct log₂(n) consensus threshold.

        Args:
            block_data:       Raw bytes of the block (serialised block dict).
            n_network_nodes:  Number of active nodes in the network.

        Returns:
            PhiStructure with phi_max, gwt_score, phi_total, and is_conscious.
        """
        state_dist = self._derive_state_distribution(block_data)
        structure = self.iit_engine.compute_phi_structure(
            state_dist, n_nodes=self.n_nodes
        )
        # Override is_conscious using network-scale threshold
        structure.is_conscious = self.iit_engine.validate_consciousness_consensus(
            structure.phi_total, n_network_nodes
        )
        return structure

    def validate_consciousness_consensus(
        self,
        phi_total: float,
        n_network_nodes: int,
    ) -> bool:
        """
        Validate the IIT v5.0 consciousness-consensus condition:
            Φ_total > log₂(n_network_nodes)

        Args:
            phi_total:        Composite consciousness score.
            n_network_nodes:  Number of network nodes.

        Returns:
            True when consensus condition is satisfied.
        """
        return self.iit_engine.validate_consciousness_consensus(
            phi_total, n_network_nodes
        )

    def phi_to_legacy_score(self, phi_total: float) -> float:
        """
        Map a Φ_total value to the legacy [200, 1000] phi_score range used
        by ``ConsensusEngine.calculate_phi_boost``.

        Args:
            phi_total: Composite Φ_total.

        Returns:
            Legacy phi_score in [200, 1000].
        """
        # Normalise phi_total to [0, 1] using log₂(n+1) as ceiling
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
        """
        Derive a 2^n state probability distribution from raw bytes.

        1. Attempt full quantum simulation via qutip (if available) to get
           the diagonal of a random density matrix seeded from *data*.
        2. Fall back to a hash-seeded pseudo-random distribution.
        """
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
        """
        Hash-seeded pseudo-random state distribution (no external dependencies).

        Produces a reproducible, data-dependent distribution over 2^n states.
        """
        n_states = 2 ** self.n_nodes
        seed_hash = hashlib.sha3_256(data).digest()
        rng = np.random.default_rng(
            seed=list(seed_hash[:8])  # 8 bytes → 64-bit seed
        )
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
