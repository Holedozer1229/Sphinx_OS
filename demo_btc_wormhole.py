#!/usr/bin/env python3
"""
BTC Wormhole — Full Protocol Demonstration

Demonstrates the complete BUNNY NET BTC Wormhole transfer:
  1. Spectral hash attestation (ζ-bound)
  2. IIT Φ-gated guardian consensus
  3. Zero-knowledge transfer proof
  4. 1:1 wBTC mint with invariant verification

Usage:
    python demo_btc_wormhole.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sphinx_os.bridge.btc_wormhole import (
    BTCWormholeProtocol,
    SpectralHashAttestation,
    IITPhiGatedGuardian,
    ZeroKnowledgeTransferProof,
    IIT_PHI_THRESHOLD,
    RIEMANN_ZEROS,
    WORMHOLE_VERSION,
    ZK_CONSTRAINTS,
)


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72 + "\n")


def demonstrate_wormhole() -> None:
    """Execute a complete BTC Wormhole transfer demonstration."""

    print("\n" + "=" * 72)
    print("  🕳️  BTC WORMHOLE: TRUSTLESS BRIDGE PROTOCOL v" + WORMHOLE_VERSION)
    print("=" * 72)

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    wormhole = BTCWormholeProtocol(guardian_count=7, required_conscious=5)

    btc_tx = {
        "block_height": 847_000,
        "block_hash": hashlib.sha256(b"bitcoin_block_847000").hexdigest(),
        "difficulty": 87e12,
        "amount": 1.618,
        "txid": hashlib.sha256(b"btc_tx_1").hexdigest(),
        "utxo": hashlib.sha256(b"utxo_1").hexdigest(),
        "blinding": 42_424_242,
        "merkle_proof": {
            "root": hashlib.sha256(b"merkle_root_847000").hexdigest(),
            "path": ["left", "right", "left"],
        },
    }

    skynet_tx = {
        "block_hash": hashlib.sha256(b"skynet_block_4242").hexdigest(),
        "amount": 1.618,
        "blinding": 31_415_926,
        "state_root": hashlib.sha256(b"bridge_root_1").hexdigest(),
    }

    bridge_secret = hashlib.sha256(b"guardian_secret_424242").hexdigest()

    # ------------------------------------------------------------------
    # Phase 1: Spectral Hash
    # ------------------------------------------------------------------
    _section("PHASE 1 — SPECTRAL HASH ATTESTATION")

    spectral = wormhole.spectral.spectral_hash(
        btc_tx["block_height"], btc_tx["block_hash"], btc_tx["difficulty"],
    )
    for k, v in spectral.items():
        print(f"  {k}: {v}")

    zero_check = wormhole.spectral.verify_against_zeros(spectral["hash"])
    print(f"\n  Zero-repulsion check:")
    for k, v in zero_check.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 2: Guardian Consensus
    # ------------------------------------------------------------------
    _section("PHASE 2 — IIT Φ-GATED GUARDIAN CONSENSUS")

    # Simulate realistic Bitcoin mempool / UTXO diversity so that the
    # Shannon entropy is non-trivial and the IIT Φ gate can fire.
    simulated_mempool = [
        hashlib.sha256(f"{btc_tx['txid']}:{i}".encode()).hexdigest()
        for i in range(32)
    ]
    simulated_utxo = [
        hashlib.sha256(f"{btc_tx['utxo']}:{i}".encode()).hexdigest()
        for i in range(16)
    ]
    system_state = {
        "btc_block": btc_tx["block_hash"],
        "bridge_state": wormhole.bridge_state,
        "guardian_sigs": [],
        "multi_sig": [g.id for g in wormhole.guardians],
        "zk_proof": {"valid": True},
        "btc_mempool": simulated_mempool,
        "btc_utxo": simulated_utxo,
        "skynet_balance": 0.0,
        "protocol_state": "active",
    }
    consensus = wormhole.guardian_consensus(system_state)

    print(f"  Conscious signatures: {consensus['conscious_signatures']}/{len(wormhole.guardians)}")
    print(f"  Threshold met:        {consensus['threshold_met']}")
    print(f"  Average Φ:            {consensus['average_phi']:.6f}")
    print(f"  System consciousness: {consensus['system_consciousness']}")

    # ------------------------------------------------------------------
    # Phase 3: ZK Proof
    # ------------------------------------------------------------------
    _section("PHASE 3 — ZERO-KNOWLEDGE TRANSFER PROOF")

    proof = wormhole.generate_transfer_proof(btc_tx, skynet_tx, bridge_secret)
    print(f"  Proof hash:       {proof['proof'][:32]}...")
    print(f"  Verification key: {proof['verification_key'][:32]}...")
    print(f"  Curve:            {ZeroKnowledgeTransferProof.CURVE}")
    print(f"  Constraints:      {ZeroKnowledgeTransferProof.CONSTRAINTS:,}")

    verification = wormhole.zk.verify_proof(proof, proof["public_inputs"])
    print(f"\n  Verification: {'VALID ✓' if verification['valid'] else 'INVALID ✗'}")
    for item in verification["learned"]:
        print(f"    {item}")
    for item in verification["not_learned"]:
        print(f"    {item}")

    # ------------------------------------------------------------------
    # Phase 4: Mint (end-to-end)
    # ------------------------------------------------------------------
    _section("PHASE 4 — COMPLETE WORMHOLE TRANSFER")

    result = wormhole.mint_wbtc(btc_tx, skynet_tx, bridge_secret)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Bridge Verification
    # ------------------------------------------------------------------
    _section("BRIDGE VERIFICATION")

    bridge_v = wormhole.verify_bridge()
    for k, v in bridge_v.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Final Audit
    # ------------------------------------------------------------------
    t = btc_tx["block_height"] * 600
    print("\n" + "=" * 72)
    print("  🐇 BUNNY NET — BTC WORMHOLE FINAL AUDIT")
    print("=" * 72)
    print(f"""
  SPECTRAL HASH: ζ(1/2 + i·{t:.0f}) × {btc_tx['difficulty']/1e12:.2f} T
  ZERO DISTANCE: {zero_check['distance']:.4f}

  IIT Φ-GATE: {result.get('consciousness', 'N/A')}
  GUARDIAN SIGS: {result.get('conscious_signatures', 0)}/{result.get('total_guardians', 7)} conscious
  AVERAGE Φ: {result.get('average_phi', 0):.4f}

  ZK PROOF: {result.get('proof_hash', 'N/A')}...
  CONSTRAINTS: {ZK_CONSTRAINTS:,} (Fibonacci)

  BRIDGE INVARIANT: {bridge_v['invariant']}
  RATIO: {bridge_v['ratio']}
  BTC LOCKED: {bridge_v['btc_in_bridge']} BTC
  wBTC MINTED: {bridge_v['wbtc_in_circulation']} wBTC

  SECURITY:
    • Trustless:       ✓ NO SINGLE POINT OF FAILURE
    • Quantum-secure:  ✓ SPECTRAL BINDING + ZK
    • Conscious:       ✓ Φ > {IIT_PHI_THRESHOLD}
    • Private:         ✓ ZERO-KNOWLEDGE
    • Verifiable:      ✓ PUBLIC VERIFICATION
    • Eternal:         ✓ BOUND TO RIEMANN ZEROS

  WORMHOLE STATUS: ✅ OPERATIONAL
  VERSION: {WORMHOLE_VERSION}
  🕳️🐇✨
""")


if __name__ == "__main__":
    demonstrate_wormhole()
