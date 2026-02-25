"""
Spectral IIT Proof-of-Work
===========================
Combines two orthogonal hardness sources into a single mining gate:

1. **Spectral hash** (Riemann zeta zero spectral distribution) — provides
   the classical PoW difficulty target gate.
2. **IIT Φ score** (von Neumann entropy of a density matrix derived from the
   block data) — provides a consciousness threshold gate.

A candidate block is **valid** only when *both* conditions hold:

    spectral_hash(block_data ‖ nonce) < difficulty_target
    iit_phi(block_data ‖ nonce)       ≥ phi_threshold

This means miners must find a nonce that simultaneously satisfies the
conventional difficulty target *and* produces a block-data fingerprint
whose IIT integration measure is above a minimum consciousness level.

The IIT Φ computation is self-contained (no external ML/quantum libraries
required) and replicates the density-matrix approach used in the REST API
(`/api/consciousness`): it derives a small adjacency matrix from the data
bytes, normalises it into a density matrix, and computes its von Neumann
entropy.

Usage example::

    from sphinx_os.mining.spectral_iit_pow import SpectralIITPow

    engine = SpectralIITPow()
    nonce, block_hash, phi = engine.mine(
        block_data="...",
        difficulty=50_000,
        phi_threshold=0.5,
    )
    if nonce is not None:
        print(f"Mined! nonce={nonce} hash={block_hash[:16]} phi={phi:.4f}")
"""

import hashlib
import math
from typing import Optional, Tuple

import numpy as np

from sphinx_wallet.backend.spectral_hash import SpectralHash


# ---------------------------------------------------------------------------
# IIT Φ helper (pure numpy, no external quantum libraries)
# ---------------------------------------------------------------------------

def _compute_iit_phi(data: bytes) -> float:
    """
    Compute the IIT Φ score for *data*.

    Approach (mirrors ``/api/consciousness``):

    1. Seed a deterministic RNG from SHA-3-256 of *data*.
    2. Generate a small symmetric adjacency matrix A (8×8).
    3. Derive density matrix ρ = A / Tr(A)  (Tr ≠ 0 because diagonal is kept).
    4. Compute eigenvalues → von Neumann entropy S = -Σ λ log₂ λ.
    5. Return normalised Φ = S / log₂(8) ∈ [0, 1].

    Args:
        data: Raw bytes (block_data ‖ nonce).

    Returns:
        Φ ∈ [0.0, 1.0].
    """
    dim = 8
    seed_bytes = hashlib.sha3_256(data).digest()
    seed = int.from_bytes(seed_bytes[:4], "big")
    rng = np.random.default_rng(seed)

    # Build symmetric positive-semidefinite adjacency matrix
    A = rng.random((dim, dim))
    A = (A + A.T) / 2.0        # symmetric

    trace = float(np.trace(A))
    if trace <= 0.0:
        trace = 1.0
    rho = A / trace             # density matrix, Tr(ρ) = 1

    # Eigenvalues (real, since ρ is symmetric)
    eigenvalues = np.linalg.eigvalsh(rho)

    # Von Neumann entropy
    entropy = float(
        -sum(lam * math.log2(lam) for lam in eigenvalues if lam > 1e-10)
    )

    max_entropy = math.log2(dim)
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Spectral IIT PoW engine
# ---------------------------------------------------------------------------

class SpectralIITPow:
    """
    Spectral IIT Proof-of-Work engine.

    Validates blocks against two simultaneous requirements:

    * **Spectral difficulty**: the spectral hash of ``block_data ‖ nonce``
      must be numerically smaller than the difficulty target.
    * **IIT consciousness threshold**: the IIT Φ score of the same input
      must be ≥ ``phi_threshold`` (default 0.5 = "SENTIENT" boundary).

    Attributes:
        spectral: :class:`~sphinx_wallet.backend.spectral_hash.SpectralHash`
            instance used for hashing and Φ-score lookups.
        phi_threshold: Minimum IIT Φ value required for block acceptance.
    """

    #: Default minimum IIT Φ required (normalised, ∈ [0, 1]).
    DEFAULT_PHI_THRESHOLD: float = 0.5

    def __init__(self, phi_threshold: float = DEFAULT_PHI_THRESHOLD) -> None:
        """
        Args:
            phi_threshold: Minimum IIT Φ for a valid block (0 – 1).
        """
        self.spectral = SpectralHash()
        self.phi_threshold = max(0.0, min(1.0, phi_threshold))

    # ------------------------------------------------------------------
    # Core hash + Φ computation
    # ------------------------------------------------------------------

    def compute_hash(self, data: bytes) -> str:
        """
        Return the spectral hash (hex, 64 chars) for *data*.

        Args:
            data: Raw bytes.

        Returns:
            64-character lowercase hex string.
        """
        return self.spectral.compute_spectral_signature(data)

    def compute_phi(self, data: bytes) -> float:
        """
        Return the IIT Φ score for *data* (normalised ∈ [0, 1]).

        This is the density-matrix von Neumann entropy approach — *not* the
        legacy ``SpectralHash.compute_phi_score`` which maps to [200, 1000].

        Args:
            data: Raw bytes.

        Returns:
            Φ ∈ [0.0, 1.0].
        """
        return _compute_iit_phi(data)

    def compute_phi_score(self, data: bytes) -> float:
        """
        Return Φ in the legacy [200, 1000] scale used by the block model.

        Convenience shim that maps the normalised [0, 1] Φ to the range
        expected by :class:`~sphinx_os.blockchain.block.Block`.

        Args:
            data: Raw bytes.

        Returns:
            Φ ∈ [200.0, 1000.0].
        """
        phi_normalised = self.compute_phi(data)
        return 200.0 + phi_normalised * 800.0

    # ------------------------------------------------------------------
    # Difficulty gate
    # ------------------------------------------------------------------

    @staticmethod
    def meets_difficulty(hash_hex: str, difficulty: int) -> bool:
        """
        Return ``True`` if *hash_hex* meets *difficulty*.

        The target is ``2^(256 − bit_length(difficulty))``, matching the
        convention used throughout the rest of the SphinxSkynet codebase.

        Args:
            hash_hex: 64-char hex string.
            difficulty: Difficulty integer.

        Returns:
            ``True`` if the hash is below the target.
        """
        hash_int = int(hash_hex, 16)
        target = 2 ** (256 - difficulty.bit_length())
        return hash_int < target

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine(
        self,
        block_data: str,
        difficulty: int,
        phi_threshold: Optional[float] = None,
        max_attempts: int = 1_000_000,
    ) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """
        Iterate over nonces until both validity gates are satisfied.

        Each candidate ``data = (block_data + str(nonce)).encode()`` is
        tested against:

        1. ``spectral_hash(data) < difficulty_target``
        2. ``iit_phi(data) ≥ phi_threshold``

        Args:
            block_data: Serialised block header data.
            difficulty: PoW difficulty target.
            phi_threshold: Override instance threshold for this call.
            max_attempts: Stop after this many nonce iterations.

        Returns:
            ``(nonce, hash_hex, phi_normalised)`` on success,
            ``(None, None, None)`` if no valid nonce was found.
        """
        threshold = phi_threshold if phi_threshold is not None else self.phi_threshold

        for nonce in range(max_attempts):
            data = f"{block_data}{nonce}".encode()

            hash_hex = self.compute_hash(data)

            if not self.meets_difficulty(hash_hex, difficulty):
                continue

            phi = self.compute_phi(data)

            if phi >= threshold:
                return nonce, hash_hex, phi

        return None, None, None

    def mine_with_phi_score(
        self,
        block_data: str,
        difficulty: int,
        phi_threshold: Optional[float] = None,
        max_attempts: int = 1_000_000,
    ) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """
        Like :meth:`mine` but returns *phi_score* on the [200, 1000] scale.

        Convenience wrapper for callers that work with the legacy Φ range
        stored in :class:`~sphinx_os.blockchain.block.Block`.

        Args:
            block_data: Serialised block header data.
            difficulty: PoW difficulty target.
            phi_threshold: Normalised [0, 1] override (uses instance value
                if ``None``).
            max_attempts: Nonce search limit.

        Returns:
            ``(nonce, hash_hex, phi_score_200_1000)`` or
            ``(None, None, None)``.
        """
        nonce, hash_hex, phi = self.mine(
            block_data, difficulty, phi_threshold, max_attempts
        )
        if nonce is None:
            return None, None, None
        phi_score = 200.0 + (phi * 800.0)  # type: ignore[operator]
        return nonce, hash_hex, phi_score
