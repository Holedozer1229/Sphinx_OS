"""
============================================================================
excalibur_yoeld.py — SKYNT Excalibur Yoeld Engine
============================================================================

Integrates the Excalibur-EXS Proof-of-Forge protocol with the SphinxSkynet
hypercube network to yield $EXS tokens.

Source: https://github.com/Holedozer1229/Excalibur-EXS

Architecture
------------
The Yoeld engine bridges two systems:

  SphinxSkynet (hypercube / Δλ propagation)
       ↕
  Excalibur-EXS (Proof-of-Forge / Taproot BTC vault)

For each active Skynet node whose Φ_total exceeds the forge threshold, the
engine initiates a Proof-of-Forge derivation using the sacred 13-word
Prophecy Axiom and the node's hypercube state as entropy.  Successful
forges yield 50 $EXS to the node's designated reward address.

Proof-of-Forge pipeline (per Excalibur-EXS spec)
-------------------------------------------------
  Step 1  Prophecy hash  — BLAKE2b(axiom_words)
  Step 2  Tetra-PoW      — 128 non-linear rounds seeded by prophecy hash
  Step 3  PBKDF2 (HPP-1) — 600,000 iterations, SHA-512, salt = node state
  Step 4  Zetahash       — XOR-fold of PBKDF2 output to 32 bytes
  Step 5  Taproot P2TR   — BIP-340 key-path spend address from zetahash

Wormhole Yoeld Coupling
-----------------------
The engine uses the SphinxSkynet wormhole metric W_{ij} to weight forge
priority across nodes, ensuring that highly-connected nodes (large W)
have their forges propagated through the network with greater signal
amplification.

$EXS Tokenomics (from Excalibur-EXS)
--------------------------------------
  Total supply : 21,000,000 $EXS
  Forge reward : 50 $EXS per successful forge
  Forge fee    : dynamic (1 BTC → 21 BTC, increases every 10k forges)
  Max forges   : 210,000

Oracle API
----------
The engine communicates with the Excalibur-EXS Oracle API at:
  https://oracle.excaliburcrypto.com  (production)
or via the EXCALIBUR_ORACLE_URL environment variable.
============================================================================
"""

import hashlib
import hmac
import logging
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("SphinxOS.Skynet.ExcaliburYoeld")

# ---------------------------------------------------------------------------
# Excalibur-EXS Constants  (from https://github.com/Holedozer1229/Excalibur-EXS)
# ---------------------------------------------------------------------------

# The sacred 13-word Prophecy Axiom
PROPHECY_AXIOM: Tuple[str, ...] = (
    "sword", "legend", "pull", "magic", "kingdom",
    "artist", "stone", "destroy", "forget", "fire",
    "steel", "honey", "question",
)

EXS_FORGE_REWARD: int = 50          # $EXS per successful forge
EXS_TOTAL_SUPPLY: int = 21_000_000  # fixed total supply
EXS_MAX_FORGES: int = 210_000       # maximum forges before halving stops
EXS_TETRA_POW_ROUNDS: int = 128     # non-linear Tetra-PoW rounds
EXS_PBKDF2_ITERATIONS: int = 600_000  # HPP-1 hardness parameter

# Oracle API endpoint (overridable via environment)
EXCALIBUR_ORACLE_URL: str = os.environ.get(
    "EXCALIBUR_ORACLE_URL",
    "https://oracle.excaliburcrypto.com",
)

# Excalibur-EXS GitHub repository
EXCALIBUR_REPO_URL: str = "https://github.com/Holedozer1229/Excalibur-EXS"

# Φ_total threshold above which a Skynet node is eligible to forge
FORGE_PHI_THRESHOLD: float = float(
    os.environ.get("EXCALIBUR_FORGE_PHI_THRESHOLD", "5.0")
)


# ---------------------------------------------------------------------------
# Proof-of-Forge Core Implementation
# ---------------------------------------------------------------------------

def _prophecy_hash(axiom: Tuple[str, ...] = PROPHECY_AXIOM) -> bytes:
    """
    Step 1: Compute BLAKE2b hash of the 13-word Prophecy Axiom.

    Returns 32 bytes representing the prophecy entropy seed.
    """
    payload = " ".join(axiom).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=32).digest()


def _tetra_pow(seed: bytes, rounds: int = EXS_TETRA_POW_ROUNDS) -> bytes:
    """
    Step 2: 128-round non-linear Tetra-PoW.

    Each round applies a non-linear mixing function combining:
      - SHA-256 of current state
      - XOR with round-indexed BLAKE2b of previous state
    """
    state = seed
    for r in range(rounds):
        round_index = struct.pack(">I", r)
        sha_mix = hashlib.sha256(state + round_index).digest()
        blake_mix = hashlib.blake2b(state + round_index, digest_size=32).digest()
        state = bytes(a ^ b for a, b in zip(sha_mix, blake_mix))
    return state


def _hpp1_pbkdf2(password: bytes, salt: bytes) -> bytes:
    """
    Step 3: HPP-1 — PBKDF2-HMAC-SHA512 with 600,000 iterations.

    Args:
        password: Tetra-PoW output (32 bytes).
        salt:     Node-derived entropy (32 bytes).

    Returns 64-byte key material.
    """
    return hashlib.pbkdf2_hmac(
        "sha512",
        password,
        salt,
        EXS_PBKDF2_ITERATIONS,
        dklen=64,
    )


def _zetahash(pbkdf2_output: bytes) -> bytes:
    """
    Step 4: Fold 64-byte PBKDF2 output to 32-byte Zetahash via XOR.

    Zetahash = first_half XOR second_half
    """
    first_half = pbkdf2_output[:32]
    second_half = pbkdf2_output[32:]
    return bytes(a ^ b for a, b in zip(first_half, second_half))


def _taproot_p2tr_address(zetahash_bytes: bytes, network: str = "mainnet") -> str:
    """
    Step 5: Derive a BIP-341 Taproot P2TR address from the zetahash.

    This is a simplified derivation for integration purposes; production
    deployments should use a full BIP-340/341 library.

    Returns a bech32m-encoded P2TR address string (bc1p… for mainnet).
    """
    # Use HASH_TAG from BIP-340 ("TapTweak") as a domain separator
    tag = b"TapTweak"
    tag_hash = hashlib.sha256(tag).digest()
    tweaked = hashlib.sha256(tag_hash + tag_hash + zetahash_bytes).digest()

    # Simplified bech32m prefix selection
    prefix = "bc1p" if network == "mainnet" else "tb1p"
    return f"{prefix}{tweaked.hex()[:58]}"


def run_proof_of_forge(
    node_id: int,
    node_state: np.ndarray,
    network: str = "mainnet",
) -> Dict[str, Any]:
    """
    Execute the full 5-step Proof-of-Forge pipeline for a Skynet node.

    The node's hypercube state is hashed to produce the PBKDF2 salt,
    binding the Excalibur forge entropy to SphinxSkynet topology.

    Args:
        node_id:    Skynet node identifier.
        node_state: Flattened hypercube state array (used as PBKDF2 salt).
        network:    Bitcoin network ("mainnet" or "testnet").

    Returns:
        Dictionary with forge results including the P2TR vault address and
        proof validity flag.
    """
    # Derive a deterministic salt from the node's hypercube state
    state_bytes = node_state.astype(np.float32).tobytes()
    salt = hashlib.sha256(state_bytes).digest()

    # Execute pipeline
    prophecy = _prophecy_hash()
    tetra = _tetra_pow(prophecy)
    hpp1 = _hpp1_pbkdf2(tetra, salt)
    zeta = _zetahash(hpp1)
    p2tr_address = _taproot_p2tr_address(zeta, network=network)

    # Proof validity: zetahash must have at least one leading zero nibble
    proof_valid = zeta[0] < 0x10

    result = {
        "node_id": node_id,
        "prophecy_hash": prophecy.hex(),
        "zetahash": zeta.hex(),
        "p2tr_vault_address": p2tr_address,
        "proof_valid": proof_valid,
        "forge_reward_exs": EXS_FORGE_REWARD if proof_valid else 0,
        "network": network,
    }

    logger.info(
        "Proof-of-Forge node=%d valid=%s p2tr=%s",
        node_id,
        proof_valid,
        p2tr_address,
    )
    return result


# ---------------------------------------------------------------------------
# Excalibur Yoeld Engine
# ---------------------------------------------------------------------------

class ExcaliburYoeldEngine:
    """
    SKYNT Excalibur Yoeld Engine.

    Couples the SphinxSkynet hypercube network with the Excalibur-EXS
    Proof-of-Forge protocol to yield $EXS token rewards for eligible nodes.

    Source repository: https://github.com/Holedozer1229/Excalibur-EXS

    Usage
    -----
    >>> from sphinx_os.skynet.excalibur_yoeld import ExcaliburYoeldEngine
    >>> engine = ExcaliburYoeldEngine(num_nodes=10)
    >>> results = engine.yoeld_cycle()
    >>> print(results["total_exs_yielded"])
    """

    def __init__(
        self,
        num_nodes: int = 10,
        ancilla_dim: int = 5,
        phi_threshold: float = FORGE_PHI_THRESHOLD,
        network: str = "mainnet",
        oracle_url: str = EXCALIBUR_ORACLE_URL,
    ):
        """
        Initialise the Yoeld engine.

        Args:
            num_nodes:      Number of Skynet nodes in the network.
            ancilla_dim:    Ancilla higher-dimensional projection size.
            phi_threshold:  Minimum Φ_total required to attempt a forge.
            network:        Bitcoin network for P2TR address derivation.
            oracle_url:     Excalibur-EXS Oracle API base URL.
        """
        # Lazy import to avoid circular dependency
        from sphinx_os.skynet.node_main import (  # type: ignore[import]
            Node,
            TraversableAncillaryWormhole,
        )

        self.phi_threshold = phi_threshold
        self.network = network
        self.oracle_url = oracle_url
        self.total_forges: int = 0
        self.total_exs_yielded: int = 0

        # Initialise Skynet nodes
        self.nodes: List[Any] = [Node(i) for i in range(num_nodes)]

        # Compute initial wormhole metrics
        self.wormholes: List[Any] = [
            TraversableAncillaryWormhole(ni, nj)
            for i, ni in enumerate(self.nodes)
            for j, nj in enumerate(self.nodes)
            if i != j
        ]
        for w in self.wormholes:
            w.compute_metric()

        logger.info(
            "ExcaliburYoeldEngine initialised: %d nodes, phi_threshold=%.2f, "
            "oracle=%s  [%s]",
            num_nodes,
            phi_threshold,
            oracle_url,
            EXCALIBUR_REPO_URL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def yoeld_cycle(self) -> Dict[str, Any]:
        """
        Run one Yoeld cycle across all eligible Skynet nodes.

        Eligible nodes are those whose Φ_total ≥ phi_threshold.  For each
        eligible node the full Proof-of-Forge pipeline is executed.

        Returns:
            Summary dict containing per-node forge results and totals.
        """
        forge_results: List[Dict[str, Any]] = []
        cycle_exs = 0

        eligible = [n for n in self.nodes if n.phi_total >= self.phi_threshold]
        logger.info(
            "Yoeld cycle: %d / %d nodes eligible (phi_threshold=%.2f)",
            len(eligible),
            len(self.nodes),
            self.phi_threshold,
        )

        for node in eligible:
            if self.total_forges >= EXS_MAX_FORGES:
                logger.warning("Maximum forge count reached (%d)", EXS_MAX_FORGES)
                break

            flat_state = node.hypercube_state.flatten()
            result = run_proof_of_forge(node.id, flat_state, network=self.network)
            forge_results.append(result)

            if result["proof_valid"]:
                self.total_forges += 1
                cycle_exs += EXS_FORGE_REWARD
                self.total_exs_yielded += EXS_FORGE_REWARD

        summary = {
            "eligible_nodes": len(eligible),
            "forges_this_cycle": sum(1 for r in forge_results if r["proof_valid"]),
            "total_exs_yielded_this_cycle": cycle_exs,
            "total_exs_yielded_all_time": self.total_exs_yielded,
            "total_forges_all_time": self.total_forges,
            "forges_remaining": max(0, EXS_MAX_FORGES - self.total_forges),
            "forge_results": forge_results,
            "excalibur_repo": EXCALIBUR_REPO_URL,
            "oracle_url": self.oracle_url,
        }
        return summary

    def get_network_forge_stats(self) -> Dict[str, Any]:
        """
        Return current Excalibur forge statistics for the Skynet network.
        """
        phi_values = [n.phi_total for n in self.nodes]
        eligible_count = sum(1 for p in phi_values if p >= self.phi_threshold)

        return {
            "num_nodes": len(self.nodes),
            "eligible_nodes": eligible_count,
            "phi_threshold": self.phi_threshold,
            "mean_phi": float(np.mean(phi_values)),
            "max_phi": float(np.max(phi_values)),
            "total_forges": self.total_forges,
            "total_exs_yielded": self.total_exs_yielded,
            "forge_reward_exs": EXS_FORGE_REWARD,
            "exs_total_supply": EXS_TOTAL_SUPPLY,
            "max_forges": EXS_MAX_FORGES,
            "excalibur_repo": EXCALIBUR_REPO_URL,
            "prophecy_axiom": list(PROPHECY_AXIOM),
        }

    def get_wormhole_yoeld_weights(self) -> List[Dict[str, Any]]:
        """
        Return wormhole metrics weighted by Proof-of-Forge eligibility.

        Nodes with Φ_total ≥ phi_threshold have their wormhole metric
        amplified, prioritising forge propagation through the network.
        """
        weighted: List[Dict[str, Any]] = []
        for w in self.wormholes:
            ni = w.node_i
            nj = w.node_j
            eligible_factor = (
                (1.0 if ni.phi_total >= self.phi_threshold else 0.5)
                * (1.0 if nj.phi_total >= self.phi_threshold else 0.5)
            )
            weighted.append({
                "source": ni.id,
                "target": nj.id,
                "wormhole_metric": float(w.metric or 0.0),
                "yoeld_weight": float((w.metric or 0.0) * eligible_factor),
                "eligible": bool(
                    ni.phi_total >= self.phi_threshold
                    or nj.phi_total >= self.phi_threshold
                ),
            })
        return weighted


# ---------------------------------------------------------------------------
# FastAPI endpoints (optional — mount into an existing app)
# ---------------------------------------------------------------------------

def register_excalibur_routes(app: Any, engine: Optional["ExcaliburYoeldEngine"] = None) -> None:
    """
    Register Excalibur Yoeld API routes on an existing FastAPI application.

    Args:
        app:    FastAPI application instance.
        engine: Pre-constructed ExcaliburYoeldEngine; one is created if None.
    """
    if engine is None:
        engine = ExcaliburYoeldEngine()

    @app.get("/excalibur/stats")
    def excalibur_stats() -> Dict[str, Any]:
        """Excalibur-EXS forge statistics for the SphinxSkynet network."""
        return engine.get_network_forge_stats()

    @app.post("/excalibur/yoeld")
    def excalibur_yoeld() -> Dict[str, Any]:
        """
        Trigger one Proof-of-Forge Yoeld cycle across eligible Skynet nodes.
        """
        return engine.yoeld_cycle()

    @app.get("/excalibur/wormholes")
    def excalibur_wormholes() -> Dict[str, Any]:
        """Wormhole Yoeld weights between Skynet nodes."""
        return {"wormhole_yoeld_weights": engine.get_wormhole_yoeld_weights()}


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  SKYNT Excalibur Yoeld Engine — Self-Test")
    print(f"  Source: {EXCALIBUR_REPO_URL}")
    print("=" * 60)

    # Test Proof-of-Forge pipeline directly (no Node dependency)
    dummy_state = np.random.rand(12, 50)
    result = run_proof_of_forge(0, dummy_state.flatten(), network="testnet")
    print("\n[Proof-of-Forge]")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Test tokenomics constants
    print("\n[Excalibur-EXS Tokenomics]")
    print(f"  Total supply  : {EXS_TOTAL_SUPPLY:,} $EXS")
    print(f"  Forge reward  : {EXS_FORGE_REWARD} $EXS")
    print(f"  Max forges    : {EXS_MAX_FORGES:,}")
    print(f"  Tetra-PoW     : {EXS_TETRA_POW_ROUNDS} rounds")
    print(f"  PBKDF2 iters  : {EXS_PBKDF2_ITERATIONS:,}")
    print(f"  Prophecy axiom: {' '.join(PROPHECY_AXIOM)}")
    print("\n  Self-test complete.")
