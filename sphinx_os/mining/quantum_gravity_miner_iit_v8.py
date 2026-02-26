"""
Quantum Gravity Miner IIT v8 Kernel
=====================================
Mining kernel that fuses three simultaneous validity gates:

1. **Spectral difficulty gate** — the spectral hash of
   ``block_data ‖ nonce`` must be numerically smaller than the difficulty
   target (standard PoW gate).

2. **IIT v8.0 consciousness gate** — the IIT v8.0 composite score
   Φ_total of the same input must satisfy:

       Φ_total > log₂(n) + δ·Φ_fano + ζ·Φ_qg

   ensuring that only "consciously integrated" blocks are valid.

3. **Quantum gravity curvature gate** — the Quantum Gravity curvature
   score Φ_qg of the candidate must be ≥ ``qg_threshold`` (default 0.1),
   guaranteeing that the causal structure of every accepted block exhibits
   at least a minimum level of emergent spacetime curvature.

A candidate block is **valid** only when *all three* conditions hold:

    spectral_hash(data)  <  difficulty_target
    Φ_total(data)        >  log₂(n) + δ·Φ_fano + ζ·Φ_qg
    Φ_qg(data)           ≥  qg_threshold

This is the v8 successor to :class:`~sphinx_os.mining.spectral_iit_pow.SpectralIITPow`
(the ``spectral_iit_pow`` v1 kernel) and inherits the dual-gate philosophy while
adding the quantum gravity curvature gate and the holographic entanglement bonus.

Usage example::

    from sphinx_os.mining.quantum_gravity_miner_iit_v8 import QuantumGravityMinerIITv8

    kernel = QuantumGravityMinerIITv8()
    result = kernel.mine(block_data="...", difficulty=50_000)
    if result.nonce is not None:
        print(f"Mined! nonce={result.nonce} hash={result.block_hash[:16]}")
        print(f"  Φ_total={result.phi_total:.4f}  Φ_qg={result.qg_score:.4f}")
        print(f"  Φ_holo={result.holo_score:.4f}  phi_score={result.phi_score:.2f}")
"""

import hashlib
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sphinx_wallet.backend.spectral_hash import SpectralHash
from ..Artificial_Intelligence.iit_v8 import (
    ASISphinxOSIITv8,
    IITv8Engine,
    PhiStructureV8,
)

logger = logging.getLogger("SphinxOS.Mining.QGMinerIITv8")


# ---------------------------------------------------------------------------
# Mining result container
# ---------------------------------------------------------------------------

@dataclass
class MineResultV8:
    """
    Result from a single :meth:`QuantumGravityMinerIITv8.mine` call.

    Attributes:
        nonce:       Winning nonce, or *None* when no valid nonce was found.
        block_hash:  64-char hex spectral hash, or *None*.
        phi_total:   IIT v8.0 composite Φ_total for the winning candidate.
        qg_score:    Quantum Gravity curvature score Φ_qg ∈ [0, 1].
        holo_score:  Holographic entanglement entropy score Φ_holo ∈ [0, 1].
        fano_score:  Octonionic Fano plane alignment Φ_fano ∈ [0, 1].
        phi_score:   Legacy phi_score in [200, 1000] for block storage.
        attempts:    Number of nonces tested before finding a valid one.
    """
    nonce: Optional[int]
    block_hash: Optional[str]
    phi_total: float
    qg_score: float
    holo_score: float
    fano_score: float
    phi_score: float
    attempts: int


# ---------------------------------------------------------------------------
# QuantumGravityMinerIITv8 kernel
# ---------------------------------------------------------------------------

class QuantumGravityMinerIITv8:
    """
    Quantum Gravity Miner IIT v8 Kernel.

    Validates blocks against three simultaneous requirements:

    * **Spectral difficulty**: the spectral hash of ``block_data ‖ nonce``
      must be numerically smaller than the difficulty target.
    * **IIT v8.0 consciousness threshold**: Φ_total must exceed
      ``log₂(n) + δ·Φ_fano + ζ·Φ_qg`` (the v8 QG-augmented threshold).
    * **Quantum gravity curvature gate**: Φ_qg must be ≥ ``qg_threshold``.

    The kernel integrates the :class:`~sphinx_os.Artificial_Intelligence.iit_v8.ASISphinxOSIITv8`
    consciousness engine with the existing
    :class:`~sphinx_wallet.backend.spectral_hash.SpectralHash` hasher so that
    the same data bytes flow through both computations simultaneously.

    Attributes:
        spectral:      SpectralHash instance for spectral difficulty gate.
        iit:           ASISphinxOSIITv8 instance for the consciousness gate.
        qg_threshold:  Minimum Φ_qg required for block acceptance ∈ [0, 1].
    """

    #: Default Quantum Gravity curvature threshold.
    DEFAULT_QG_THRESHOLD: float = 0.10

    def __init__(
        self,
        *,
        qg_threshold: float = DEFAULT_QG_THRESHOLD,
        n_nodes: int = 3,
        alpha: float = 0.30,
        beta: float = 0.15,
        gamma: float = 0.15,
        delta: float = 0.15,
        epsilon: float = 0.10,
        zeta: float = 0.10,
        eta: float = 0.05,
        temporal_depth: int = 2,
    ) -> None:
        """
        Initialise the Quantum Gravity Miner IIT v8 kernel.

        Args:
            qg_threshold:  Minimum Φ_qg required for a valid block.
            n_nodes:       Number of IIT qubit nodes.
            alpha:         Weight for Φ_τ in composite.
            beta:          Weight for GWT_S in composite.
            gamma:         Weight for ICP_avg in composite.
            delta:         Weight for Φ_fano in composite.
            epsilon:       Weight for Φ_nab in composite.
            zeta:          Weight for Φ_qg in composite.
            eta:           Weight for Φ_holo in composite.
            temporal_depth: τ for temporal-depth Φ integration.
        """
        self.spectral = SpectralHash()
        self.iit = ASISphinxOSIITv8(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            eta=eta,
            n_nodes=n_nodes,
            temporal_depth=temporal_depth,
        )
        self.qg_threshold = max(0.0, min(1.0, qg_threshold))
        logger.info(
            "QuantumGravityMinerIITv8 initialised "
            "(n_nodes=%d, qg_threshold=%.3f)",
            n_nodes, self.qg_threshold,
        )

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    def compute_hash(self, data: bytes) -> str:
        """
        Return the spectral hash (64-char hex) for *data*.

        Args:
            data: Raw bytes.

        Returns:
            64-character lowercase hex string.
        """
        return self.spectral.compute_spectral_signature(data)

    @staticmethod
    def meets_difficulty(hash_hex: str, difficulty: int) -> bool:
        """
        Return ``True`` if *hash_hex* meets *difficulty*.

        Uses the same target convention as the rest of the SphinxSkynet
        codebase: ``target = 2^(256 − bit_length(difficulty))``.

        Args:
            hash_hex:   64-char hex string.
            difficulty: Difficulty integer.

        Returns:
            ``True`` if the hash is below the target.
        """
        hash_int = int(hash_hex, 16)
        target = 2 ** (256 - difficulty.bit_length())
        return hash_int < target

    # ------------------------------------------------------------------
    # IIT v8 consciousness computation
    # ------------------------------------------------------------------

    def compute_phi_structure(self, data: bytes) -> PhiStructureV8:
        """
        Compute the full IIT v8.0 Φ structure for *data*.

        Args:
            data: Raw bytes (block_data ‖ nonce).

        Returns:
            :class:`~sphinx_os.Artificial_Intelligence.iit_v8.PhiStructureV8`.
        """
        return self.iit.compute_block_consciousness(data)

    def compute_phi_score(self, data: bytes) -> float:
        """
        Return the legacy [200, 1000] phi_score for *data*.

        Convenience shim for callers that interact with
        :class:`~sphinx_os.blockchain.block.Block`.

        Args:
            data: Raw bytes.

        Returns:
            phi_score ∈ [200.0, 1000.0].
        """
        result = self.iit.calculate_phi(data)
        return self.iit.phi_to_legacy_score(result["phi_total"])

    # ------------------------------------------------------------------
    # Three-gate validity check
    # ------------------------------------------------------------------

    def is_valid_block(
        self,
        data: bytes,
        difficulty: int,
        n_network_nodes: int = 1,
    ) -> Tuple[bool, PhiStructureV8, str]:
        """
        Check whether *data* satisfies all three validity gates.

        Args:
            data:             Raw bytes (block_data ‖ nonce).
            difficulty:       PoW difficulty target.
            n_network_nodes:  Network node count for consensus threshold.

        Returns:
            ``(valid, structure, gate_failed)`` where:

            * *valid*       — ``True`` when all gates pass.
            * *structure*   — :class:`PhiStructureV8` from the IIT computation.
            * *gate_failed* — ``""`` on success; ``"difficulty"``,
              ``"consciousness"``, or ``"qg_curvature"`` naming the first
              failed gate.
        """
        hash_hex = self.compute_hash(data)
        if not self.meets_difficulty(hash_hex, difficulty):
            # Return a minimal empty structure to avoid the expensive IIT call
            return False, PhiStructureV8(), "difficulty"

        structure = self.compute_phi_structure(data)

        # Consciousness gate
        if not self.iit.validate_consciousness_consensus(
            structure.phi_total,
            structure.fano_score,
            structure.qg_score,
            n_network_nodes,
        ):
            return False, structure, "consciousness"

        # Quantum gravity curvature gate
        if structure.qg_score < self.qg_threshold:
            return False, structure, "qg_curvature"

        return True, structure, ""

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine(
        self,
        block_data: str,
        difficulty: int,
        n_network_nodes: int = 1,
        max_attempts: int = 1_000_000,
    ) -> MineResultV8:
        """
        Iterate over nonces until all three validity gates are satisfied.

        For each candidate nonce the data string
        ``block_data + str(nonce)`` is encoded to bytes and tested against:

        1. ``spectral_hash(data) < difficulty_target``
        2. ``Φ_total(data) > log₂(n) + δ·Φ_fano + ζ·Φ_qg``
        3. ``Φ_qg(data) ≥ qg_threshold``

        Args:
            block_data:       Serialised block header data.
            difficulty:       PoW difficulty target.
            n_network_nodes:  Network node count for the consciousness gate.
            max_attempts:     Stop after this many nonce iterations.

        Returns:
            :class:`MineResultV8` — all fields are populated even on failure
            (``nonce`` and ``block_hash`` will be *None* when no valid nonce
            was found within *max_attempts*).
        """
        for nonce in range(max_attempts):
            data = f"{block_data}{nonce}".encode()
            valid, structure, _ = self.is_valid_block(data, difficulty, n_network_nodes)

            if valid:
                hash_hex = self.compute_hash(data)
                phi_score = self.iit.phi_to_legacy_score(structure.phi_total)
                logger.debug(
                    "Block found at nonce=%d hash=%s Φ_total=%.4f Φ_qg=%.4f",
                    nonce, hash_hex[:16], structure.phi_total, structure.qg_score,
                )
                return MineResultV8(
                    nonce=nonce,
                    block_hash=hash_hex,
                    phi_total=structure.phi_total,
                    qg_score=structure.qg_score,
                    holo_score=structure.holo_score,
                    fano_score=structure.fano_score,
                    phi_score=phi_score,
                    attempts=nonce + 1,
                )

        logger.debug(
            "No valid block found after %d attempts (difficulty=%d)",
            max_attempts, difficulty,
        )
        return MineResultV8(
            nonce=None,
            block_hash=None,
            phi_total=0.0,
            qg_score=0.0,
            holo_score=0.0,
            fano_score=0.0,
            phi_score=200.0,
            attempts=max_attempts,
        )

    def mine_with_stats(
        self,
        block_data: str,
        difficulty: int,
        n_network_nodes: int = 1,
        max_attempts: int = 1_000_000,
    ) -> Tuple[MineResultV8, dict]:
        """
        Like :meth:`mine` but also returns gate-rejection statistics.

        The statistics dict has the following keys:

        * ``total_attempts``      — total nonces tested.
        * ``difficulty_rejected`` — nonces that failed the spectral gate.
        * ``consciousness_rejected`` — nonces that failed the consciousness gate.
        * ``qg_curvature_rejected``  — nonces that failed the QG curvature gate.
        * ``accepted``            — 1 if a valid nonce was found, else 0.

        Args:
            block_data:       Serialised block header data.
            difficulty:       PoW difficulty target.
            n_network_nodes:  Network node count.
            max_attempts:     Nonce search limit.

        Returns:
            ``(MineResultV8, stats_dict)``
        """
        stats: dict = {
            "total_attempts": 0,
            "difficulty_rejected": 0,
            "consciousness_rejected": 0,
            "qg_curvature_rejected": 0,
            "accepted": 0,
        }

        for nonce in range(max_attempts):
            stats["total_attempts"] += 1
            data = f"{block_data}{nonce}".encode()
            valid, structure, gate_failed = self.is_valid_block(
                data, difficulty, n_network_nodes
            )

            if gate_failed == "difficulty":
                stats["difficulty_rejected"] += 1
                continue
            if gate_failed == "consciousness":
                stats["consciousness_rejected"] += 1
                continue
            if gate_failed == "qg_curvature":
                stats["qg_curvature_rejected"] += 1
                continue

            # All gates passed
            hash_hex = self.compute_hash(data)
            phi_score = self.iit.phi_to_legacy_score(structure.phi_total)
            stats["accepted"] = 1
            result = MineResultV8(
                nonce=nonce,
                block_hash=hash_hex,
                phi_total=structure.phi_total,
                qg_score=structure.qg_score,
                holo_score=structure.holo_score,
                fano_score=structure.fano_score,
                phi_score=phi_score,
                attempts=nonce + 1,
            )
            return result, stats

        result = MineResultV8(
            nonce=None,
            block_hash=None,
            phi_total=0.0,
            qg_score=0.0,
            holo_score=0.0,
            fano_score=0.0,
            phi_score=200.0,
            attempts=max_attempts,
        )
        return result, stats
