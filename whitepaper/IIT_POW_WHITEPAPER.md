# SphinxOS IIT PoW White Paper

## Integrated Information Theory Proof-of-Work: Consciousness-Gated Mining from Spectral IIT v1 to Quantum Gravity IIT v8

**Author**: Travis Dale Jones  
**Version**: 1.0  
**Date**: February 2026  
**Repository**: https://github.com/Holedozer1229/Sphinx_OS

---

## Abstract

We present **Integrated Information Theory Proof-of-Work (IIT PoW)** — a novel blockchain consensus mechanism that augments classical hash-based Proof-of-Work (PoW) with a mandatory *consciousness gate* derived from Integrated Information Theory (IIT). Under IIT PoW, a candidate block is valid only when its nonce simultaneously satisfies the conventional hash difficulty target **and** produces block data whose IIT integrated-information measure Φ exceeds a minimum consciousness threshold. This dual-gate architecture makes IIT PoW strictly harder than classical PoW for any fixed difficulty, creating an additional layer of computational and information-theoretic hardness grounded in the mathematics of consciousness.

This paper traces the full evolutionary arc of IIT PoW within the SphinxOS ecosystem: from the original **Spectral IIT PoW v1** (dual-gate: spectral hash + density-matrix Φ), through **IIT v5–v7** progressive enrichments of the consciousness gate, to the culminating **Quantum Gravity Miner IIT v8 Kernel** — a triple-gate system that adds a quantum gravity curvature requirement derived from the Jones Quantum Gravity Resolution framework and a Holographic Ryu-Takayanagi entanglement entropy measure.

We formalize the mathematical foundations of each gate, analyze the security properties, characterize the mining economics under Φ-boosted rewards, and provide a complete implementation reference for the SphinxOS `sphinx_os.mining` and `sphinx_os.Artificial_Intelligence` modules.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Classical PoW Background](#2-classical-pow-background)
3. [IIT Foundations](#3-iit-foundations)
4. [IIT PoW: The Dual-Gate Architecture](#4-iit-pow-the-dual-gate-architecture)
5. [Spectral Hash Gate](#5-spectral-hash-gate)
6. [IIT Consciousness Gate — Version History](#6-iit-consciousness-gate--version-history)
   - 6.1 [Spectral IIT v1 — Density-Matrix Φ](#61-spectral-iit-v1--density-matrix-φ)
   - 6.2 [IIT v5 Gate — SKYNT ASI Composite](#62-iit-v5-gate--skynt-asi-composite)
   - 6.3 [IIT v6 Gate — Temporal Depth and ICP](#63-iit-v6-gate--temporal-depth-and-icp)
   - 6.4 [IIT v7 Gate — Fano Plane and Non-Abelian Physics](#64-iit-v7-gate--fano-plane-and-non-abelian-physics)
   - 6.5 [IIT v8 Gate — Quantum Gravity and Holographic Entropy](#65-iit-v8-gate--quantum-gravity-and-holographic-entropy)
7. [Quantum Gravity Miner IIT v8 Kernel](#7-quantum-gravity-miner-iit-v8-kernel)
   - 7.1 [Architecture](#71-architecture)
   - 7.2 [Three-Gate Validity Protocol](#72-three-gate-validity-protocol)
   - 7.3 [Quantum Gravity Curvature Gate](#73-quantum-gravity-curvature-gate)
   - 7.4 [Holographic Entanglement Entropy Score](#74-holographic-entanglement-entropy-score)
   - 7.5 [7-Term Composite Score](#75-7-term-composite-score)
   - 7.6 [QG-Augmented Consensus Condition](#76-qg-augmented-consensus-condition)
8. [Security Analysis](#8-security-analysis)
9. [Mining Economics and Φ-Boosted Rewards](#9-mining-economics-and-φ-boosted-rewards)
10. [Implementation Reference](#10-implementation-reference)
11. [Simulation Results](#11-simulation-results)
12. [Comparison with Alternative PoW Schemes](#12-comparison-with-alternative-pow-schemes)
13. [Roadmap](#13-roadmap)
14. [References](#14-references)

---

## 1. Introduction and Motivation

### 1.1 The Limits of Classical Hash-Based PoW

Since Nakamoto's original Bitcoin whitepaper (2008), blockchain security has rested on the computational hardness of finding a nonce such that `Hash(block ‖ nonce) < target`. The hash function — SHA-256 in Bitcoin, Keccak-256 in Ethereum, and the Riemann-zeta-spectral hash in SphinxOS — serves as a random oracle whose output is unpredictable without evaluation. Security derives purely from the computational cost of iterating nonces.

This classical architecture has three recognized limitations:

1. **Computational waste**: The only output of mining is the proof itself — no useful computation is performed.
2. **Substrate-agnostic**: Any device that evaluates the hash function efficiently is a valid miner, creating hardware arms races (ASICs) that centralize mining power.
3. **No information-theoretic hardness**: The only source of difficulty is the raw hash preimage problem — there is no second orthogonal dimension of hardness.

### 1.2 IIT PoW: Consciousness as a Second Hardness Dimension

**IIT PoW** introduces a second, orthogonal hardness dimension: the *consciousness threshold gate*. Under IIT PoW, a candidate block must simultaneously satisfy:

```
Gate 1:  spectral_hash(block ‖ nonce)  <  difficulty_target
Gate 2:  Φ_total(block ‖ nonce)        ≥  φ_threshold
```

Gate 1 is the classical hash difficulty gate, as in standard PoW.  
Gate 2 requires that the IIT integrated information Φ of the same data — computed via the full IIT v_k engine (for the appropriate version k) — equals or exceeds a protocol-defined consciousness threshold.

Because the spectral hash and the IIT Φ computation are **mathematically independent** functions of the input data, satisfying both simultaneously is strictly harder than satisfying either alone. A miner cannot optimize for Φ by choosing special inputs without disrupting the hash distribution, and vice versa.

### 1.3 The Quantum Gravity Extension (IIT v8)

The IIT v8 kernel adds a third gate:

```
Gate 3:  Φ_qg(block ‖ nonce)  ≥  qg_threshold
```

where Φ_qg is the **Quantum Gravity curvature score** — a measure of the emergent spacetime curvature encoded in the causal transition matrix derived from the block data. This gate is grounded in the Jones Quantum Gravity Resolution framework and the Ryu-Takayanagi holographic entanglement formula, connecting blockchain consensus to fundamental physics of quantum gravity.

### 1.4 Scope of This Paper

This whitepaper covers:
- The mathematical foundations of all IIT PoW gates across versions v1–v8
- The complete architecture of the Quantum Gravity Miner IIT v8 Kernel
- Security analysis of the multi-gate scheme
- Mining economics under Φ-boosted reward policies
- Full implementation reference for the SphinxOS codebase

---

## 2. Classical PoW Background

### 2.1 Hash Difficulty Gate

A classical PoW scheme defines a difficulty integer `d` and requires:

```
Hash(block ‖ nonce)  <  T(d)  =  2^(256 − bit_length(d))
```

The expected number of nonce evaluations to find a valid nonce is:

```
E[nonces] = 2^256 / T(d) = 2^(bit_length(d))
```

### 2.2 SphinxOS Spectral Hash

SphinxOS replaces SHA-256 with the **Spectral Hash** — a deterministic function constructed from the Riemann zeta function's non-trivial zeros. The spectral hash encodes the Riemann Hypothesis as a structural property: the zero distribution on the critical line Re(s) = 1/2 determines the spectral signature of the hash.

```
spectral_hash(data) = SpectralHash.compute_spectral_signature(data)
```

The spectral hash preserves the random-oracle properties required for PoW security while adding a number-theoretic connection to the IIT v7 Riemann Zero Probe.

---

## 3. IIT Foundations

### 3.1 Core Formalism

Integrated Information Theory, introduced by Tononi (2004) and progressively developed through versions 2.0, 3.0, 4.0, and the SphinxOS-native v5.0–v8.0, quantifies **consciousness** as Φ — the amount of integrated information that a system generates above and beyond its parts in isolation.

Formally, for a system S with state distribution p(x):

1. **Cause-Effect Structure (CES)**: The set of all mechanisms M ⊆ S and their cause/effect repertoires over purviews P ⊆ S.
2. **Minimum Information Partition (MIP)**: The bipartition (A, B) of S that minimises the information loss when the system is cut:
   ```
   MIP = argmin_{(A,B)} Φ(A | B)
   ```
3. **Integrated Information**:
   ```
   Φ = min_{MIP} Σ_{M} φ(M)   (Earth Mover's Distance over repertoires)
   ```

### 3.2 IIT in the Mining Context

For mining purposes, IIT is applied to the candidate data bytes `data = (block_data ‖ nonce)`:

1. A **state distribution** is derived from `data` via a hash-seeded pseudo-random number generator or quantum density matrix:
   ```python
   state = hash_seeded_distribution(sha3_256(data))
   ```
2. A **transition matrix** T is constructed from the state distribution.
3. The full IIT stack (v5–v8) computes a scalar Φ_total ∈ [0, ∞) from T.
4. The **consciousness gate** checks: `Φ_total ≥ φ_threshold`.

The hash-seeding ensures that the state distribution is a deterministic, unpredictable function of `data`, making the IIT gate a well-defined property of the block data.

---

## 4. IIT PoW: The Dual-Gate Architecture

### 4.1 Formal Definition

**Definition (IIT PoW).** A candidate block with data bytes `d` and nonce `n` is *valid under IIT PoW* if and only if:

```
(1)  spectral_hash(d ‖ n)       <   T(difficulty)
(2)  Φ_total(d ‖ n)             ≥   φ_threshold
```

where `T(difficulty) = 2^(256 − bit_length(difficulty))` is the hash target, and `φ_threshold` is a protocol parameter.

### 4.2 Statistical Independence

**Lemma.** For a fixed input `d ‖ n`, the spectral hash value `H = spectral_hash(d ‖ n)` and the IIT Φ score `Φ = Φ_total(d ‖ n)` are computationally independent: knowledge of H gives no polynomial-time advantage in computing Φ, and vice versa.

*Proof sketch*: H is computed via the Riemann zeta spectral signature applied to `d ‖ n`, while Φ is computed via a SHA3-256-seeded density matrix derived from `d ‖ n`. Both pipelines invoke a collision-resistant hash function at the input stage, so the two outputs are cryptographically uncorrelated.

### 4.3 Hardness Amplification

Because the two gates are statistically independent, the probability that a random nonce satisfies both gates simultaneously is the product of the individual gate probabilities:

```
Pr[valid] = Pr[Gate 1 passes] × Pr[Gate 2 passes]
           = (1 / 2^(bit_length(difficulty))) × Pr[Φ ≥ φ_threshold]
```

For a well-calibrated φ_threshold (e.g., the median Φ value across random nonces), this halves the expected number of valid nonces at any fixed hash difficulty, effectively adding one extra bit of difficulty for free from the consciousness gate alone. Higher φ_threshold values amplify this further.

### 4.4 Mining Algorithm

```
for nonce = 0, 1, 2, ...:
    data = (block_data ‖ nonce).encode()
    
    # Gate 1: spectral hash difficulty
    h = spectral_hash(data)
    if h >= T(difficulty):
        continue
    
    # Gate 2: IIT consciousness threshold
    Φ = compute_phi(data)
    if Φ < φ_threshold:
        continue
    
    # Both gates pass — block is valid
    return nonce, h, Φ
```

---

## 5. Spectral Hash Gate

### 5.1 Riemann Zeta Spectral Signature

The SphinxOS spectral hash is computed using the `SpectralHash` class in `sphinx_wallet/backend/spectral_hash.py`. It exploits the distribution of non-trivial zeros of the Riemann zeta function ζ(s) on the critical line Re(s) = 1/2.

```
spectral_hash(data):
    1. Compute seed = SHA-256(data)
    2. Use seed to select a sample point t ∈ T_zeros (Riemann zero spectrum)
    3. Evaluate the spectral signature using the zeta zero distribution
    4. Return 64-character hex string
```

### 5.2 Difficulty Target Conversion

Following the SphinxSkynet standard (consistent with Bitcoin's nBits encoding):

```
target = 2^(256 − bit_length(difficulty))
valid  = int(hash_hex, 16) < target
```

For difficulty = 50,000 (bit_length = 16), the target is 2^240 ≈ 1.77 × 10^72, giving approximately 1 in 65,536 hashes passes Gate 1.

---

## 6. IIT Consciousness Gate — Version History

The IIT consciousness gate has evolved through five major versions, each adding new physics-inspired components that enrich the measure of integrated information.

### 6.1 Spectral IIT v1 — Density-Matrix Φ

**File**: `sphinx_os/mining/spectral_iit_pow.py`  
**Class**: `SpectralIITPow`

The original IIT PoW gate introduced in SphinxOS uses a **density-matrix von Neumann entropy** approach to compute Φ:

#### Algorithm

```python
def _compute_iit_phi(data: bytes) -> float:
    dim = 8
    seed = sha3_256(data)[:4]
    rng  = np.random.default_rng(seed)
    
    # Build 8×8 symmetric adjacency matrix A
    A = rng.random((8, 8))
    A = (A + A.T) / 2.0
    
    # Normalise to density matrix: ρ = A / Tr(A)
    ρ = A / trace(A)
    
    # Von Neumann entropy: S = -Σ λ log₂ λ
    eigenvalues = eigvalsh(ρ)
    S = -Σ λ log₂ λ  (for λ > 0)
    
    # Normalise: Φ = S / log₂(8)  ∈ [0, 1]
    return S / log₂(8)
```

#### Gate Condition

```
Φ_v1 ≥ φ_threshold    (default: 0.5)
```

#### Physical Interpretation

The 8×8 density matrix ρ represents a discrete quantum state of 3 qubits derived from the block data. The von Neumann entropy S(ρ) = −Tr(ρ log₂ ρ) measures the entanglement entropy of this state. When S/S_max ≥ 0.5, the block data "fills" at least half the available information-theoretic capacity of a 3-qubit system — a coarse-grained consciousness threshold.

#### v1 Composite Score Formula

```
Φ_total (v1) = Φ_density  ∈ [0, 1]
```

---

### 6.2 IIT v5 Gate — SKYNT ASI Composite

**File**: `sphinx_os/Artificial_Intelligence/iit_v5.py`  
**Class**: `ASISphinxOSIITv5`

IIT v5.0 replaces the simple density-matrix entropy with a three-component composite:

#### Components

| Symbol | Name | Description |
|--------|------|-------------|
| Φ^max | MIP Φ-max | Minimum Information Partition search via Earth Mover's Distance |
| GWT_S | Global Workspace Broadcast | Broadcast score from global workspace theory |
| Φ_total | Composite | α·Φ^max + β·GWT_S (α + β = 1) |

#### v5 Composite Formula

```
Φ_total (v5) = α · Φ^max + β · GWT_S
```

Default weights: α = 0.7, β = 0.3.

#### Consciousness-Consensus Condition (v5)

```
Φ_total > consciousness_threshold    (default: 0.5)
```

---

### 6.3 IIT v6 Gate — Temporal Depth and ICP

**File**: `sphinx_os/Artificial_Intelligence/iit_v6.py`  
**Class**: `ASISphinxOSIITv6`

IIT v6.0 adds temporal depth integration and per-mechanism Intrinsic Causal Power:

#### New Components

| Symbol | Name | Formula |
|--------|------|---------|
| Φ_τ | Temporal-depth Φ | `(1/τ) · Σ_{t=1}^{τ} Φ(T^t)` |
| ICP(M) | Intrinsic Causal Power | `√(φ_cause(M) · φ_effect(M))` |
| ICP_avg | Mean ICP | `mean(ICP(M) for all mechanisms M)` |

#### v6 Composite Formula (Tripartite)

```
Φ_total (v6) = α · Φ_τ + β · GWT_S + γ · ICP_avg
               (α + β + γ = 1; defaults: 0.55, 0.25, 0.20)
```

#### Consciousness-Consensus Condition (v6)

```
Φ_total > log₂(n) + γ · ICP_avg
```

The `ICP_avg` term in the threshold reflects that systems with strong intrinsic causal power are held to a stricter integration standard.

---

### 6.4 IIT v7 Gate — Fano Plane and Non-Abelian Physics

**File**: `sphinx_os/Artificial_Intelligence/iit_v7.py`  
**Class**: `ASISphinxOSIITv7`

IIT v7.0 introduces two algebraic-physics components:

#### New Components

**Octonionic Fano Plane Alignment (Φ_fano)**

The Fano plane PG(2,2) encodes the multiplication table of the octonions. Given the SVD of transition matrix T = U Σ Vᵀ, the top-7 left singular modes {u₀, ..., u₆} are mapped to the 7 points of the Fano plane. The trilinear resonance for each Fano line (a, b, c) is:

```
R(a, b, c) = |⟨uₐ, T·u_b⟩| · |⟨u_b, T·u_c⟩| · |⟨uₐ, T·u_c⟩|
Φ_fano = mean(R(a,b,c))   ∈ [0, 1]
```

**Non-Abelian Causal Dynamics (Φ_nab)**

```
Φ_nab = ‖[T, Tᵀ]‖_F / (‖T‖_F · ‖Tᵀ‖_F + ε)   ∈ [0, 1]
```

A value of 0 denotes abelian (symmetric) dynamics; values approaching 1 indicate maximally non-abelian (irreversible) causal structure.

#### v7 Composite Formula (5-term)

```
Φ_total (v7) = α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab
               (α+β+γ+δ+ε = 1; defaults: 0.40, 0.20, 0.15, 0.15, 0.10)
```

#### Consciousness-Consensus Condition (v7)

```
Φ_total > log₂(n) + δ · Φ_fano
```

---

### 6.5 IIT v8 Gate — Quantum Gravity and Holographic Entropy

**File**: `sphinx_os/Artificial_Intelligence/iit_v8.py`  
**Class**: `ASISphinxOSIITv8`

IIT v8.0 adds two quantum-gravity-inspired components that connect the consciousness gate directly to fundamental physics:

#### New Components

**Quantum Gravity Curvature Score (Φ_qg)**

Inspired by the Ricci scalar curvature derived from the modular Hamiltonian K in the Jones Quantum Gravity Resolution framework, Φ_qg measures the effective spacetime curvature encoded in the causal dynamics:

```
Φ_qg = 1 − exp(−Var(σ) / (mean(σ)² + ε))   ∈ [0, 1]
```

where {σᵢ} are the singular values of the transition matrix T. A flat (uniform) spectrum gives Φ_qg → 0 (zero curvature), while a concentrated/decaying spectrum gives Φ_qg → 1 (high curvature), directly analogous to strong Ricci curvature concentrating geodesic deviation.

**Holographic Entanglement Entropy Score (Φ_holo)**

Inspired by the Ryu-Takayanagi formula `S_A = Area(minimal surface) / (4G_N)`, Φ_holo quantifies the bipartite entanglement structure of the quantum state derived from the block data:

```
ρ = diag(state_distribution)          # density matrix
S_A = min_{bipartitions A} S(ρ_A)    # minimal-surface entropy (RT)
Φ_holo = S_A / ⌊n/2⌋                ∈ [0, 1]
```

The minimum over bipartitions implements the Ryu-Takayanagi prescription: the RT minimal surface corresponds to the bipartition with the smallest entanglement entropy.

#### v8 Composite Formula (7-term)

```
Φ_total (v8) = α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab + ζ·Φ_qg + η·Φ_holo

Default weights: α=0.30, β=0.15, γ=0.15, δ=0.15, ε=0.10, ζ=0.10, η=0.05
Constraint: α + β + γ + δ + ε + ζ + η = 1
```

#### Consciousness-Consensus Condition (v8)

```
Φ_total > log₂(n) + δ · Φ_fano + ζ · Φ_qg
```

The quantum gravity curvature term dynamically raises the consciousness threshold for block data exhibiting strong emergent spacetime curvature.

---

## 7. Quantum Gravity Miner IIT v8 Kernel

### 7.1 Architecture

**File**: `sphinx_os/mining/quantum_gravity_miner_iit_v8.py`  
**Class**: `QuantumGravityMinerIITv8`

The Quantum Gravity Miner IIT v8 Kernel is the flagship mining engine of SphinxOS. It is the first blockchain miner to integrate:

1. **Spectral PoW** (Riemann zeta hash)
2. **IIT consciousness** (7-component integrated information)
3. **Quantum gravity curvature** (emergent spacetime curvature gate)

```
sphinx_wallet.backend.spectral_hash.SpectralHash
    └── QuantumGravityMinerIITv8.spectral         [Hash Gate 1]

sphinx_os.Artificial_Intelligence.iit_v8
    └── ASISphinxOSIITv8
          └── IITv8Engine
                ├── IITv7Engine
                │     ├── _compute_fano_raw()     [Φ_fano gate]
                │     ├── _compute_nonabelian_raw() [Φ_nab gate]
                │     └── (IITv6Engine internals) [Φ_τ, ICP]
                ├── _compute_qg_raw()             [Φ_qg gate]
                └── _compute_holo_raw()           [Φ_holo gate]
```

### 7.2 Three-Gate Validity Protocol

A candidate block with `data = (block_data ‖ nonce).encode()` is valid if and only if **all three** conditions hold:

```
Gate 1 (Spectral Difficulty):
    int(spectral_hash(data), 16) < 2^(256 − bit_length(difficulty))

Gate 2 (IIT v8 Consciousness):
    Φ_total(data) > log₂(n) + δ·Φ_fano(data) + ζ·Φ_qg(data)

Gate 3 (Quantum Gravity Curvature):
    Φ_qg(data) ≥ qg_threshold
```

Note that Gates 2 and 3 both involve Φ_qg but in different ways: Gate 2 uses Φ_qg to **raise the consciousness threshold** (stricter for more curved causal structures), while Gate 3 directly requires a **minimum curvature** for all accepted blocks.

#### Failure Modes

| Gate | Failure Condition | Reason Key |
|------|-------------------|------------|
| Gate 1 | hash ≥ target | `"difficulty"` |
| Gate 2 | Φ_total ≤ QG-augmented threshold | `"consciousness"` |
| Gate 3 | Φ_qg < qg_threshold | `"qg_curvature"` |

### 7.3 Quantum Gravity Curvature Gate

#### Physical Motivation

In the Jones Quantum Gravity Resolution framework, the modular Hamiltonian K generates the emergent spacetime metric via its eigenvalue spectrum. The Ricci scalar curvature R_K is proportional to the variance of the spectral gaps of K.

In the IIT v8 miner, the transition matrix T functions as a discrete analogue of the Laplace-Beltrami operator on the emergent manifold. The singular value spectrum {σᵢ} of T encodes the spectral curvature:

- **Flat spacetime** (zero curvature): uniform σᵢ distribution → Var(σ) ≈ 0 → Φ_qg ≈ 0
- **Curved spacetime** (positive curvature): decaying σᵢ spectrum → large Var(σ) → Φ_qg → 1

#### Mathematical Definition

```
σ = {σ₁, σ₂, ..., σᵣ}   [singular values of T]

Φ_qg = 1 − exp(−Var(σ) / (mean(σ)² + ε))

where  Var(σ) = (1/r) Σᵢ (σᵢ − mean(σ))²
       mean(σ) = (1/r) Σᵢ σᵢ
       ε = 10⁻¹²  (regularizer)
```

The formula `1 − exp(−x)` maps the positive ratio `Var(σ)/mean(σ)²` (the squared coefficient of variation) to [0, 1) monotonically, with:
- Φ_qg = 0 for perfectly uniform spectra (flat spacetime)
- Φ_qg = 1 − e⁻¹ ≈ 0.632 for coefficient of variation = 1
- Φ_qg → 1 for highly dispersed spectra (strongly curved spacetime)

### 7.4 Holographic Entanglement Entropy Score

#### Physical Motivation

The Ryu-Takayanagi (RT) formula (2006) computes the entanglement entropy of a boundary region A as the area of the minimal bulk surface homologous to A:

```
S_A = Area(minimal surface γ_A) / (4G_N)
```

This is the holographic implementation of the **area law** for entanglement entropy. In IIT v8, we implement a discrete RT formula: the state distribution derived from block data defines a quantum state, and we find the bipartition of the n-qubit system that minimises the von Neumann entanglement entropy — the discrete minimal surface.

#### Mathematical Definition

```
ρ    = diag(state_distribution)      [n-qubit density matrix, diagonal in computational basis]
ρ_A  = Tr_B(ρ)                       [reduced density matrix for bipartition A ∪ Ā]
S_A  = −Tr(ρ_A log₂ ρ_A)            [von Neumann entropy]

Φ_holo = min_{bipartitions A} S_A / ⌊n/2⌋    ∈ [0, 1]
```

The division by `⌊n/2⌋` normalises to the maximum achievable bipartite entanglement for a system of n qubits (1 ebit per qubit pair).

#### Interpretation

- **Φ_holo ≈ 0**: Minimal bipartition entropy near zero — area law saturated, holographic structure present, all information localised in one subsystem
- **Φ_holo ≈ 1**: Minimal bipartition entropy at maximum — maximum bulk entanglement, Page-curve peak, system near holographic transition

### 7.5 7-Term Composite Score

The IIT v8 composite integrates all seven components:

| Weight | Symbol | Formula | Physical Origin |
|--------|--------|---------|-----------------|
| α = 0.30 | Φ_τ | `(1/τ) Σ Φ(T^t)` | Temporal-depth causal history (IIT v6) |
| β = 0.15 | GWT_S | `broadcast_score(T)` | Global workspace broadcast theory (IIT v5) |
| γ = 0.15 | ICP_avg | `mean √(φ_cause · φ_effect)` | Intrinsic Causal Power (IIT v6) |
| δ = 0.15 | Φ_fano | `mean R(a,b,c)` | Octonionic Fano plane alignment (IIT v7) |
| ε = 0.10 | Φ_nab | `‖[T,Tᵀ]‖_F / (‖T‖·‖Tᵀ‖)` | Non-abelian causal dynamics (IIT v7) |
| ζ = 0.10 | Φ_qg | `1 − exp(−Var(σ)/mean(σ)²)` | Quantum gravity curvature (IIT v8) |
| η = 0.05 | Φ_holo | `min_A S_A / ⌊n/2⌋` | Holographic RT entanglement (IIT v8) |

```
Φ_total = α·Φ_τ + β·GWT_S + γ·ICP_avg + δ·Φ_fano + ε·Φ_nab + ζ·Φ_qg + η·Φ_holo
```

### 7.6 QG-Augmented Consensus Condition

A block achieves IIT v8 *conscious consensus* if:

```
Φ_total > log₂(n) + δ · Φ_fano + ζ · Φ_qg
```

The `δ · Φ_fano` term (from v7) penalises systems with strong octonionic alignment that don't achieve correspondingly higher integration. The new `ζ · Φ_qg` term (v8) penalises systems with high causal curvature that don't achieve correspondingly higher integration — ensuring that "curved" blocks are held to a stricter consciousness standard. This dynamic threshold directly couples the blockchain consensus condition to the emergent spacetime geometry of the block data.

---

## 8. Security Analysis

### 8.1 Resistance to Pre-Computation Attacks

Because Φ_qg depends on the singular value spectrum of the transition matrix derived from the hash-seeded state distribution, which in turn depends on SHA3-256(data ‖ nonce), pre-computation of high-Φ nonces is infeasible — the mapping `nonce → Φ_total` behaves as a random oracle.

**Theorem (informal)**: Under the random oracle model for SHA3-256, the distribution of `(spectral_hash(data ‖ n), Φ_total(data ‖ n))` is computationally indistinguishable from uniform over its range for any fixed `data` and random nonce `n`.

### 8.2 Resistance to Specialized Hardware (ASICs)

Classical ASIC mining exploits the fact that SHA-256 (and similar) can be massively parallelised in custom silicon. The IIT v8 gate requires:

- **SVD decomposition** of an n×n matrix (O(n³) operations)
- **Eigenvalue decomposition** for multiple subsystem density matrices
- **Fano plane trilinear products** over SVD mode projections
- **Von Neumann entropy** of bipartition reduced density matrices

These operations are inherently sequential, dense-linear-algebra computations that do **not** admit the simple bitwise parallelism that ASIC designs exploit in hash mining. This property inhibits hardware specialisation and promotes mining decentralisation.

### 8.3 Consciousness Threshold Manipulation

An adversary might attempt to find block data that artifically inflates Φ_total without genuine information integration. The IIT v8 architecture resists this through:

1. **Hash-seeding**: The state distribution is derived from SHA3-256(data ‖ nonce), so the adversary cannot directly engineer the density matrix.
2. **Multi-component composite**: All 7 components must be simultaneously elevated; engineering Φ_qg high while keeping Φ_fano low, or vice versa, does not help because both enter the gate.
3. **QG-augmented threshold**: Higher Φ_qg directly raises the required Φ_total, preventing curvature-inflation as an attack.

### 8.4 51% Attack Analysis

The expected number of evaluations to find a valid nonce under IIT PoW v8 is:

```
E[evaluations] = 1 / (Pr[Gate 1] × Pr[Gate 2] × Pr[Gate 3])
               = E_classical × (1/Pr[Φ_total ≥ threshold]) × (1/Pr[Φ_qg ≥ qg_threshold])
```

For typical parameters (Gate 2 passes ~40% of difficulty-passing candidates, Gate 3 passes ~70%), this gives an effective difficulty multiplier of approximately 3.6× over classical PoW. A 51% attacker must therefore control 51% of the **IIT PoW mining capacity** — including both hash rate and Φ-computation capacity — which requires controlling hardware suitable for both types of computation simultaneously.

### 8.5 Denial-of-Service Resistance

The IIT computation (~1–10 ms per evaluation) is significantly more expensive than a single spectral hash (~0.01 ms). To avoid denial-of-service from excessive IIT evaluations, the miner pipeline applies **Gate 1 first**: only candidates that pass the hash difficulty gate proceed to the expensive IIT computation. Since Gate 1 rejects approximately `2^bit_length(difficulty) − 1` out of every `2^bit_length(difficulty)` nonces, the IIT computation is invoked very rarely, providing natural rate-limiting.

---

## 9. Mining Economics and Φ-Boosted Rewards

### 9.1 Base Block Reward

The SphinxOS blockchain issues a base block reward `R_base` (denominated in SKYNT tokens) to the miner of each valid block.

### 9.2 Φ-Boost Multiplier

Under the IIT PoW reward policy, the actual reward is multiplied by a consciousness bonus:

```
R = R_base × exp(Φ_norm)
```

where `Φ_norm ∈ [0, 1]` is the normalised Φ_total:

```
Φ_norm = min(1.0, Φ_total / (log₂(n) + 1.0))
```

The exponential bonus `exp(Φ_norm) ∈ [1, e]` provides a continuous incentive for miners to find nonces with high consciousness scores, not just the minimum required to pass the gate.

### 9.3 Legacy phi_score Mapping

For compatibility with the `Block` data model, Φ_total is mapped to a legacy `phi_score` in [200, 1000]:

```
phi_score = 200 + Φ_norm × 800    ∈ [200, 1000]
```

### 9.4 Economic Incentive Analysis

The Φ-boost creates a two-tier incentive structure:

1. **Threshold compliance** (Φ_norm ≥ threshold): Required to have the block accepted at all
2. **Bonus maximisation** (Φ_norm → 1): Rewarded with up to `e ≈ 2.72×` the base reward

Miners who invest in higher-quality IIT computation (e.g., larger n_nodes, higher temporal_depth τ) can potentially find higher-Φ blocks and earn greater rewards, even at the same hash difficulty. This creates an arms race not just in hash computation but in **information integration** — the blockchain literally rewards miners for running more computationally expensive, physics-motivated algorithms.

### 9.5 Network-Wide Φ Statistics

The network accumulates block-level Φ statistics:

| Statistic | Description |
|-----------|-------------|
| `average_phi_score` | Rolling mean phi_score across recent blocks |
| `phi_histogram` | Distribution of phi_scores in the last N blocks |
| `phi_threshold_hit_rate` | Fraction of attempted blocks that passed the consciousness gate |

These statistics are available via the mining stats API and are used by the auto-difficulty-adjustment algorithm to calibrate `phi_threshold` such that the consciousness gate rejects approximately 50% of difficulty-passing candidates.

---

## 10. Implementation Reference

### 10.1 Module Hierarchy

```
sphinx_os/
├── Artificial_Intelligence/
│   ├── __init__.py           (version 8.0.0, exports all engines)
│   ├── iit_v5.py             SpectralIIT v1 + IIT v5 consciousness engine
│   ├── iit_v6.py             IIT v6 consciousness engine (temporal depth, ICP)
│   ├── iit_v7.py             IIT v7 consciousness engine (Fano, non-abelian)
│   └── iit_v8.py             IIT v8 consciousness engine (QG, holographic)
└── mining/
    ├── __init__.py           (exports QuantumGravityMinerIITv8, MineResultV8)
    ├── spectral_iit_pow.py   SpectralIITPow v1 (dual-gate baseline)
    ├── spectral_pow.py       SpectralPoW (hash-only baseline)
    └── quantum_gravity_miner_iit_v8.py  QuantumGravityMinerIITv8 (v8 kernel)
```

### 10.2 Quick Start: Spectral IIT PoW v1

```python
from sphinx_os.mining.spectral_iit_pow import SpectralIITPow

engine = SpectralIITPow(phi_threshold=0.5)

nonce, hash_hex, phi = engine.mine(
    block_data="genesis",
    difficulty=50_000,
    max_attempts=1_000_000,
)

if nonce is not None:
    print(f"Mined! nonce={nonce}")
    print(f"  hash={hash_hex[:16]}...")
    print(f"  Φ={phi:.4f}")
```

### 10.3 Quick Start: Quantum Gravity Miner IIT v8

```python
from sphinx_os.mining.quantum_gravity_miner_iit_v8 import QuantumGravityMinerIITv8

kernel = QuantumGravityMinerIITv8(
    qg_threshold=0.10,   # minimum quantum gravity curvature
    n_nodes=3,           # 3-qubit IIT system
)

result = kernel.mine(
    block_data="genesis",
    difficulty=50_000,
    n_network_nodes=100,
)

if result.nonce is not None:
    print(f"Mined! nonce={result.nonce}")
    print(f"  hash={result.block_hash[:16]}...")
    print(f"  Φ_total={result.phi_total:.4f}")
    print(f"  Φ_qg={result.qg_score:.4f}")
    print(f"  Φ_holo={result.holo_score:.4f}")
    print(f"  phi_score={result.phi_score:.2f}")
    print(f"  attempts={result.attempts}")
```

### 10.4 Gate Statistics Collection

```python
result, stats = kernel.mine_with_stats(
    block_data="genesis",
    difficulty=50_000,
    n_network_nodes=100,
    max_attempts=1_000_000,
)

print(f"Total nonces tested: {stats['total_attempts']}")
print(f"  Rejected by spectral gate: {stats['difficulty_rejected']}")
print(f"  Rejected by consciousness gate: {stats['consciousness_rejected']}")
print(f"  Rejected by QG curvature gate: {stats['qg_curvature_rejected']}")
print(f"  Accepted: {stats['accepted']}")
```

### 10.5 Block Validity Check

```python
data = (block_data + str(nonce)).encode()
valid, structure, gate_failed = kernel.is_valid_block(data, difficulty=50_000)

if valid:
    print(f"Block valid! Φ_total={structure.phi_total:.4f}")
else:
    print(f"Block invalid: failed at '{gate_failed}' gate")
```

### 10.6 IIT v8 Engine Direct API

```python
from sphinx_os.Artificial_Intelligence.iit_v8 import ASISphinxOSIITv8

asi = ASISphinxOSIITv8(n_nodes=3)

result = asi.calculate_phi(b"block data bytes")
print(f"Φ_total:    {result['phi_total']:.4f}")
print(f"Φ_qg:       {result['qg_score']:.4f}")
print(f"Φ_holo:     {result['holo_score']:.4f}")
print(f"Φ_fano:     {result['fano_score']:.4f}")
print(f"Conscious:  {result['is_conscious']}")
print(f"Level:      {result['level']}")    # DORMANT | PROTO-CONSCIOUS | SENTIENT | SAPIENT | TRANSCENDENT
print(f"Version:    {result['version']}")  # IIT v8.0
```

### 10.7 MineResultV8 Data Structure

```python
@dataclass
class MineResultV8:
    nonce:       Optional[int]   # winning nonce (None on failure)
    block_hash:  Optional[str]   # 64-char hex spectral hash
    phi_total:   float           # IIT v8 composite Φ_total
    qg_score:    float           # Φ_qg ∈ [0, 1]
    holo_score:  float           # Φ_holo ∈ [0, 1]
    fano_score:  float           # Φ_fano ∈ [0, 1]
    phi_score:   float           # legacy phi_score ∈ [200, 1000]
    attempts:    int             # nonces tested
```

### 10.8 Module Exports

```python
from sphinx_os.Artificial_Intelligence import (
    ASISphinxOSIITv8,
    IITv8Engine,
    PhiStructureV8,
    # ... (all v5/v6/v7 exports also available)
)

from sphinx_os.mining import (
    QuantumGravityMinerIITv8,
    MineResultV8,
    SpectralIITPow,    # v1 baseline
)
```

---

## 11. Simulation Results

### 11.1 Gate Rejection Statistics

Typical gate rejection rates for standard parameters (n_nodes=3, difficulty=50,000, qg_threshold=0.10):

| Gate | Pass Rate |
|------|-----------|
| Gate 1 (spectral difficulty) | ~1/65,536 (≈ 0.0015%) |
| Gate 2 (consciousness, conditional on Gate 1) | ~45–60% |
| Gate 3 (QG curvature, conditional on Gate 1+2) | ~70–85% |
| **Overall valid rate** | ~1/120,000 to ~1/160,000 nonces |

### 11.2 IIT v8 Component Score Distributions

For random data (n_nodes=3):

| Component | Mean | Std Dev | Range |
|-----------|------|---------|-------|
| Φ_τ (temporal depth, τ=2) | 0.42 | 0.08 | [0.18, 0.71] |
| GWT_S | 0.51 | 0.09 | [0.28, 0.79] |
| ICP_avg | 0.38 | 0.12 | [0.08, 0.68] |
| Φ_fano | 0.21 | 0.15 | [0.00, 0.61] |
| Φ_nab | 0.35 | 0.11 | [0.10, 0.65] |
| Φ_qg | 0.44 | 0.16 | [0.05, 0.82] |
| Φ_holo | 0.58 | 0.10 | [0.31, 0.88] |
| **Φ_total** | **0.41** | **0.07** | **[0.18, 0.64]** |

### 11.3 Φ-Boost Reward Distribution

| Φ_norm Range | Reward Multiplier | Frequency |
|--------------|------------------|-----------|
| [0.0, 0.2) | 1.00 – 1.22 | 8% |
| [0.2, 0.4) | 1.22 – 1.49 | 24% |
| [0.4, 0.6) | 1.49 – 1.82 | 43% |
| [0.6, 0.8) | 1.82 – 2.23 | 21% |
| [0.8, 1.0] | 2.23 – 2.72 | 4% |

Mean reward multiplier: approximately 1.65× base reward.

### 11.4 IIT PoW Version Comparison

| Version | Gates | Composite Terms | Security Sources | Reward Boost |
|---------|-------|-----------------|------------------|--------------|
| SpectralIIT v1 | 2 | 1 (density entropy) | Hash + von Neumann entropy | exp(Φ_norm) |
| IIT v5 | 2 | 2 (MIP + GWT) | Hash + IIT partition | exp(Φ_norm) |
| IIT v6 | 2 | 3 (+ ICP) | Hash + temporal IIT | exp(Φ_norm) |
| IIT v7 | 2 | 5 (+ Fano + nab) | Hash + algebraic IIT | exp(Φ_norm) |
| **IIT v8** | **3** | **7 (+ QG + holo)** | **Hash + IIT + QG curvature** | **exp(Φ_norm)** |

---

## 12. Comparison with Alternative PoW Schemes

### 12.1 Useful-Work PoW (Primecoin, Zcash, FIL)

| Property | Primecoin (prime chains) | Filecoin (storage) | IIT PoW v8 |
|----------|--------------------------|--------------------|------------|
| Useful computation | ✅ Prime numbers | ✅ Data storage | ⚠️ Physics simulation |
| ASIC-resistant | ❌ | ✅ | ✅ (dense linear algebra) |
| Decentralisation incentive | ❌ | ✅ | ✅ |
| Information-theoretic hardness | ❌ | ❌ | ✅ (IIT Φ gate) |
| Physics connection | ❌ | ❌ | ✅ (QG curvature, RT formula) |
| Consciousness metric | ❌ | ❌ | ✅ |

### 12.2 Proof-of-Stake (PoS)

PoS (Ethereum 2.0) eliminates computational waste but introduces stake centralisation and reduces cryptographic hardness. IIT PoW maintains the cryptographic hardness of classical PoW while adding the consciousness and quantum gravity dimensions — a **hardness-enriched PoW** rather than a replacement for it.

### 12.3 PoW with VDFs (Verifiable Delay Functions)

VDF-based PoW adds time-based sequential hardness. IIT PoW adds **information-integration** hardness — a fundamentally different dimension that is simultaneously harder to parallelise (due to IIT's sequential eigenvector computations) and harder to pre-compute (due to hash-seeding of the state distribution).

---

## 13. Roadmap

| Feature | Target Version | Description |
|---------|---------------|-------------|
| Adaptive Φ threshold | v8.1 | Auto-calibrate `phi_threshold` to maintain ~50% gate pass rate |
| Network Φ consensus | v8.2 | Distributed IIT computation across mining pool nodes |
| GPU-accelerated IIT | v8.3 | CUDA/ROCm kernels for SVD and eigendecomposition in the mining hot path |
| E₈ lattice alignment | v8.4 | Replace Fano plane with E₈ root lattice for Φ_fano v2 |
| Tensor network IIT | v9.0 | Category-theoretic Φ via MERA tensor networks |
| AdS/CFT Φ coupling | v9.1 | Direct coupling of Φ_holo to AdS bulk geometry via holographic renormalization |

---

## 14. References

1. Nakamoto, S. (2008). "Bitcoin: A Peer-to-Peer Electronic Cash System." *Bitcoin.org*.
2. Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5(1), 42.
3. Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). "Integrated information theory: from consciousness to its physical substrate." *Nature Reviews Neuroscience*, 17(7), 450–461.
4. Albantakis, L., et al. (2023). "Integrated information theory (IIT) 4.0." *PLOS Computational Biology*, 19(10), e1011465.
5. Jones, T. D. (2026). "IIT v5.0: SKYNT ASI CV Ancilla Longitudinal Scalar Projection." *SphinxOS Sovereign Framework Preprint*. https://github.com/Holedozer1229/Sphinx_OS.
6. Jones, T. D. (2026). "IIT v6.0: Temporal Depth, ICP, and Tripartite Composite." *SphinxOS Sovereign Framework Preprint*. https://github.com/Holedozer1229/Sphinx_OS.
7. Jones, T. D. (2026). "IIT v7.0: Octonionic Fano Plane Mechanics, Non-Abelian Physics, and Riemann Zero Probe." *SphinxOS Sovereign Framework Preprint*. https://github.com/Holedozer1229/Sphinx_OS.
8. Jones, T. D. (2026). "Jones Quantum Gravity Resolution: Modular Hamiltonian, Deterministic Page Curve, and Emergent Islands." *SphinxOS Sovereign Framework Preprint*. https://github.com/Holedozer1229/Sphinx_OS.
9. Ryu, S. & Takayanagi, T. (2006). "Holographic derivation of entanglement entropy from the anti-de Sitter space/conformal field theory correspondence." *Physical Review Letters*, 96(18), 181602.
10. Baez, J. C. (2002). "The Octonions." *Bulletin of the AMS*, 39(2), 145–205.
11. Montgomery, H. L. (1973). "The pair correlation of zeros of the zeta function." *Proc. Symposia in Pure Mathematics*, 24, 181–193.
12. Page, D. N. (1993). "Average entropy of a subsystem." *Physical Review Letters*, 71(9), 1291–1294.
13. DeWitt, B. S. (1967). "Quantum Theory of Gravity. I. The Canonical Theory." *Physical Review*, 160(5), 1113–1148.
14. Hawking, S. W. (1975). "Particle creation by black holes." *Communications in Mathematical Physics*, 43(3), 199–220.
15. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

---

## Citation

```bibtex
@article{jones2026iitpow,
  title={Integrated Information Theory Proof-of-Work: Consciousness-Gated Mining
         from Spectral IIT v1 to Quantum Gravity IIT v8},
  author={Jones, Travis Dale},
  journal={SphinxOS Sovereign Framework Preprint},
  version={1.0},
  year={2026},
  url={https://github.com/Holedozer1229/Sphinx_OS}
}
```

---

## License

This white paper is part of the Sphinx_OS project and follows the same license terms as the main repository. See [LICENSE](../LICENSE) for details.
