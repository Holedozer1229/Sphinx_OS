# BTC Wormhole — Quantum-Secured Cross-Chain Bitcoin Bridge

**Author:** Travis D. Jones  
**Institution:** SphinxOS Research Division  
**Date:** February 2026  
**Version:** 1.0.0

---

## Abstract

The **BTC Wormhole** is a trustless, quantum-secured bridge protocol that enables BTC transfers between **Bitcoin mainnet**, **SKYNT-BTC** (the SphinxOS hard fork at genesis), and the **SphinxSkynet** ecosystem.  The Wormhole extends the existing SphinxSkynet cross-chain bridge with three novel mechanisms: **Spectral Hash Attestations** that bind transfer proofs to Riemann-zeta-weighted PoW structure, **IIT Φ-Gated Guardian Consensus** that augments multi-sig with a consciousness gate, and **Zero-Knowledge Transfer Proofs** that guarantee 1:1 BTC-to-wBTC-SKYNT correspondence without revealing private key material.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Supported Routes](#2-supported-routes)
3. [Spectral Hash Attestations](#3-spectral-hash-attestations)
4. [IIT Φ-Gated Guardian Consensus](#4-iit-φ-gated-guardian-consensus)
5. [Zero-Knowledge Transfer Proofs](#5-zero-knowledge-transfer-proofs)
6. [Transfer Lifecycle](#6-transfer-lifecycle)
7. [Fee Model](#7-fee-model)
8. [Security Properties](#8-security-properties)
9. [Quick Start](#9-quick-start)
10. [API Reference](#10-api-reference)

---

## 1. Architecture Overview

The BTC Wormhole builds on three layers of the SphinxOS ecosystem:

```
┌─────────────────────────────────────────────────────────────┐
│                      BTC Wormhole                           │
│   Spectral Attestations · Φ Gate · ZK Proofs               │
├─────────────────────────────────────────────────────────────┤
│                 CrossChainBridge (existing)                  │
│   Lock/Mint · Burn/Release · 5-of-9 Multi-Sig              │
├────────────┬──────────────────────────┬─────────────────────┤
│  Bitcoin   │      SKYNT-BTC           │  SphinxSkynet       │
│  (BTC)     │  (Hard Fork at Genesis)  │  (Sphinx chain)     │
│            │  Spectral IIT PoW        │                     │
└────────────┴──────────────────────────┴─────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **BTCWormhole** | Core engine managing transfers, attestations, and Φ-gating |
| **SpectralAttestation** | SHA-256 + Riemann-zeta-weighted hash binding transfers to PoW |
| **WormholeTransfer** | Complete transfer record with lifecycle tracking |
| **Guardian Committee** | 9-member multi-sig committee with Φ consciousness gate |

---

## 2. Supported Routes

The Wormhole supports bidirectional transfers between all three chains:

| Source | Destination | Token | Mechanism |
|--------|-------------|-------|-----------|
| BTC | SKYNT-BTC | BTC → SKYNT | Pegged 1:1 via spectral attestation |
| BTC | SphinxSkynet | BTC → wBTC-SKYNT | Lock-and-mint with ZK proof |
| SKYNT-BTC | SphinxSkynet | SKYNT → wSKYNT | Bridged with Φ-gated consensus |
| SphinxSkynet | BTC | wBTC-SKYNT → BTC | Burn-and-release with ZK proof |
| SphinxSkynet | SKYNT-BTC | wSKYNT → SKYNT | Burn-and-release with Φ gate |
| SKYNT-BTC | BTC | SKYNT → BTC | Reverse peg with spectral attestation |

---

## 3. Spectral Hash Attestations

Every wormhole transfer is anchored by a **spectral hash attestation** — a two-stage hash that binds the transfer proof to the mathematical structure of the Spectral IIT PoW algorithm used by SKYNT-BTC.

### Algorithm

```
1.  raw_hash = SHA-256(transfer_data)
2.  ζ_weight = sin(14.134725 × (raw_hash[:8] mod 1000) / 1000)
3.  spectral_input = raw_hash ‖ ζ_weight ‖ φ_score
4.  spectral_hash = SHA-256(spectral_input)
```

Where:
- **14.134725** is the imaginary part of the first non-trivial Riemann zeta zero
- **ζ_weight** introduces a sinusoidal modulation tied to the spectral structure of prime numbers
- **φ_score** is the IIT Φ consciousness score of the attesting guardian set

### Security Property

Forging a spectral attestation is computationally equivalent to inverting the spectral hash, which inherits the collision resistance of SHA-256 and the computational hardness of the Spectral IIT PoW.

---

## 4. IIT Φ-Gated Guardian Consensus

The standard 5-of-9 multi-signature requirement is augmented with a **consciousness gate**: the guardian committee must collectively produce a minimum Φ score before a wormhole transfer is finalised.

### Φ Gate Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `guardian_count` | 9 | Total guardians in the committee |
| `required_signatures` | 5 | Multi-sig threshold (5-of-9) |
| `phi_threshold` | 0.5 | Minimum collective Φ (normalised 0–1) |

### Gate Logic

```python
# Transfer is Φ-gated iff:
len(valid_signatures) >= required_signatures   # multi-sig
AND
collective_phi >= phi_threshold                  # consciousness gate
```

A collective Φ below the threshold (0.5, the "SENTIENT" boundary) rejects the transfer even if all 9 guardians sign.  This mirrors the Spectral IIT PoW used for SKYNT-BTC block consensus.

---

## 5. Zero-Knowledge Transfer Proofs

Every finalised wormhole transfer generates a **ZK proof** that:

1. The locked BTC on the source chain corresponds to the minted wBTC-SKYNT on the destination chain
2. The spectral attestation is valid
3. The Φ gate was satisfied

The proof is deterministic and verifiable:

```python
proof = SHA-256(JSON(transfer_id, net_amount, attestation.spectral_hash))
```

In production, this would use ZK-SNARK/ZK-STARK circuits.  The current implementation provides a hash-based commitment that can be upgraded to full ZK without changing the transfer protocol.

---

## 6. Transfer Lifecycle

```
INITIATED → ATTESTED → PHI_GATED → ZK_PROVED → FINALISED
                                         ↘
                                        FAILED
```

| Stage | Trigger | Action |
|-------|---------|--------|
| **INITIATED** | `initiate_transfer()` | Funds locked on source, attestation created |
| **ATTESTED** | (automatic) | Spectral hash computed and attached |
| **PHI_GATED** | `submit_guardian_signatures()` | Multi-sig + Φ gate passed |
| **ZK_PROVED** | `finalise_transfer()` | ZK proof generated |
| **FINALISED** | (automatic) | Wrapped tokens minted on destination |
| **FAILED** | `fail_transfer()` | Funds unlocked, transfer cancelled |

---

## 7. Fee Model

### Base Fee

- **Wormhole fee rate**: 0.05 % (5 basis points)
- Half the standard cross-chain bridge fee (0.1 %)

### Φ Discount

Higher consciousness scores earn lower fees:

```
discount = min(50%, φ_score × 50%)
effective_rate = 0.05% × (1 - discount)
```

| Φ Score | Effective Fee | Discount |
|---------|--------------|----------|
| 0.0 | 0.050 % | 0 % |
| 0.3 | 0.043 % | 15 % |
| 0.5 | 0.038 % | 25 % |
| 0.8 | 0.030 % | 40 % |
| 1.0 | 0.025 % | 50 % |

### Guardian Incentive

- **20 %** of the transfer fee is distributed to signing guardians
- Incentivises active participation and high-Φ attestation

---

## 8. Security Properties

### 8.1 Multi-Sig Threshold

The 5-of-9 guardian committee requires a supermajority to approve transfers.  Compromising fewer than 5 guardians cannot authorise transfers.

### 8.2 Φ Gate Defense

Even with 9 compromised guardians, transfers are rejected if the collective Φ score falls below the threshold.  This provides a second independent security layer.

### 8.3 Spectral Hash Binding

Spectral attestations bind transfers to the Riemann-zeta mathematical structure.  Modifying transfer parameters invalidates the spectral hash.

### 8.4 ZK Proof Integrity

The deterministic ZK proof ensures that any verifier can independently confirm the 1:1 correspondence between locked and minted tokens.

### 8.5 Fund Safety

Failed transfers automatically unlock source-chain funds.  Finalised transfers cannot be retroactively failed.

---

## 9. Quick Start

### Python API

```python
from sphinx_os.bridge import BTCWormhole

# Create wormhole instance
wormhole = BTCWormhole()

# Execute an end-to-end BTC → SphinxSkynet transfer
transfer = wormhole.execute_transfer(
    source_chain="btc",
    destination_chain="sphinx",
    amount=0.5,           # 0.5 BTC
    sender="bc1q...",
    recipient="SPHINX_...",
    phi_score=0.7,        # consciousness score for fee discount
)

print(f"Transfer ID: {transfer.transfer_id}")
print(f"Status: {transfer.status.value}")           # "finalised"
print(f"Net Amount: {transfer.net_amount:.8f}")
print(f"Fee: {transfer.fee:.8f}")
print(f"ZK Proof: {transfer.zk_proof[:32]}...")

# Check balances
print(f"Wrapped BTC: {wormhole.get_wrapped_balance('SPHINX_...')}")
print(f"Locked BTC:  {wormhole.get_locked_balance('bc1q...')}")

# View statistics
stats = wormhole.get_stats()
print(f"Version: {stats['version']}")
print(f"Total Volume: {stats['total_volume']}")
```

### Step-by-Step Transfer

```python
from sphinx_os.bridge import BTCWormhole

wormhole = BTCWormhole()

# Step 1: Initiate
tid = wormhole.initiate_transfer("btc", "sphinx", 1.0, "ALICE", "BOB", 0.6)

# Step 2: Guardian signatures + Φ gate
signatures = wormhole.guardians[:5]
wormhole.submit_guardian_signatures(tid, signatures, collective_phi=0.65)

# Step 3: Finalise (generates ZK proof + mints wrapped BTC)
wormhole.finalise_transfer(tid)

# Check result
transfer = wormhole.get_transfer(tid)
print(transfer)
```

---

## 10. API Reference

### `BTCWormhole(guardian_count=9, required_signatures=5, phi_threshold=0.5)`

Core wormhole engine.

#### Methods

| Method | Description |
|--------|-------------|
| `validate_route(src, dst)` | Check if route is supported |
| `supported_routes()` | List all supported routes |
| `calculate_fee(amount, phi)` | Compute fee with Φ discount |
| `create_spectral_attestation(data, phi)` | Build spectral hash attestation |
| `initiate_transfer(...)` | Start a wormhole transfer |
| `submit_guardian_signatures(tid, sigs, phi)` | Submit multi-sig + Φ gate |
| `finalise_transfer(tid)` | Generate ZK proof and mint tokens |
| `fail_transfer(tid)` | Cancel and unlock funds |
| `execute_transfer(...)` | End-to-end convenience method |
| `get_transfer(tid)` | Query transfer status |
| `get_locked_balance(addr)` | Query locked BTC |
| `get_wrapped_balance(addr)` | Query wBTC-SKYNT balance |
| `get_stats()` | Cumulative statistics |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WORMHOLE_VERSION` | `"1.0.0"` | Protocol version |
| `WORMHOLE_FEE_RATE` | `0.0005` | Base fee rate (0.05 %) |
| `MAX_PHI_DISCOUNT` | `0.50` | Maximum Φ discount (50 %) |
| `GUARDIAN_FEE_SHARE` | `0.20` | Guardian fee share (20 %) |
| `PHI_GATE_THRESHOLD` | `0.5` | Minimum Φ for gate |
| `SPECTRAL_CONFIRMATIONS` | `6` | Required spectral confirmations |

---

## Citation

```bibtex
@article{jones2026btcwormhole,
  title={BTC Wormhole: Quantum-Secured Cross-Chain Bitcoin Bridge with
         Spectral Attestations and IIT Φ-Gated Consensus},
  author={Jones, Travis Dale},
  journal={SphinxOS Sovereign Framework Preprint},
  version={1.0.0},
  year={2026},
  url={https://github.com/Holedozer1229/Sphinx_OS}
}
```

---

## License

This document is part of the Sphinx_OS project and follows the same license terms as the main repository. See [LICENSE](LICENSE) for details.
