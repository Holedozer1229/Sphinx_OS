# SphinxOS Whitepaper & Documentation

This directory contains comprehensive documentation for SphinxOS, including the NPTC framework whitepaper and economic system specifications.

## Contents

- **nptc_whitepaper.tex** - LaTeX source file for the NPTC whitepaper
- **nptc_whitepaper.pdf** - Compiled PDF whitepaper (1.9 MB, 13 pages)
- **TOKENOMICS_WHITEPAPER.md** - Economic system and tokenomics documentation
- **generate_images.py** - Python script to generate framework diagrams
- **images/** - Directory containing all figures used in the whitepaper

## Generated Images

The whitepaper includes six key diagrams illustrating the NPTC framework:

1. **icosahedron.png** - Au₁₃ icosahedral cluster structure (12 surface + 1 central vertex)
2. **fano_plane.png** - Fano plane showing seven imaginary octonions
3. **fibonacci_timing.png** - Non-periodic control timing based on Fibonacci sequence
4. **spectral_gap.png** - Eigenvalue spectrum of the icosahedral discrete Laplacian L₁₃
5. **xi_invariant.png** - NPTC invariant Ξ unifying three scales
6. **cross_chain.png** - Cross-chain verification network with Fano topology

## Building the PDF

To regenerate the PDF from source:

```bash
# Generate images (requires matplotlib and numpy)
python3 generate_images.py

# Compile LaTeX to PDF (requires pdflatex)
pdflatex nptc_whitepaper.tex
pdflatex nptc_whitepaper.tex  # Run twice for references
```

## Repository Links

All references to the Sphinx_OS repository have been updated to point to:
**https://github.com/Holedozer1229/Sphinx_OS**

## Key Features

### NPTC Framework
The whitepaper presents:

- **NPTC Framework**: Non-periodic thermodynamic control using Fibonacci timing
- **Experimental Platform**: Au₁₃-DmT-Ac aerogel in optomechanical cavity
- **Applications**: 
  - Cross-chain zk-EVM proof mining
  - Spectral Bitcoin miner
  - Megaminx proof-of-solve protocol
- **Six Predictions**: Three confirmed experimentally, three awaiting cosmological/gravitational tests

### Economic System
SphinxOS implements a self-funding economic organism with:

- **PoX Pool Automation**: Automated STX → BTC yield farming via Proof of Transfer
- **Yield Distribution**: Mathematical formulas for treasury split and user payouts
- **NFT Multipliers**: Rarity-based yield boosts (2x+)
- **Formal Security**: Cryptographic proofs for unforgeability and safety
- **Economic Simulator**: Revenue modeling and flywheel effect analysis

See [../ECONOMICS.md](../ECONOMICS.md) for complete economic system documentation.

## Abstract

We introduce Non-Periodic Thermodynamic Control (NPTC), a new class of feedback systems that operate at the critical interface where quantum coherence meets classical dissipation. NPTC abandons periodic sampling in favor of deterministic non-repeating Fibonacci timing, and replaces state-space stabilization with the preservation of a geometric invariant Ξ. The framework exhibits seven quantized Fano-plane eigenfrequencies and yields a non-associative Berry phase—the first laboratory signature of octonionic holonomy.

## Citation

```bibtex
@article{jones2026nptc,
  title={Non-Periodic Thermodynamic Control: A Universal Framework for Stabilizing Systems at the Quantum–Classical Boundary},
  author={Jones, Travis},
  journal={Sovereign Framework Preprint},
  year={2026},
  url={https://github.com/Holedozer1229/Sphinx_OS}
}
```

## License

This work is part of the Sphinx_OS project and follows the same license terms as the main repository.

---

## Economic System Overview

### 1. PoX Pool Automation

SphinxOS includes a Clarity smart contract for automated STX delegation to Proof of Transfer (PoX) pools:

**Features:**
- Non-custodial STX delegation
- DAO-governed pool rotation
- User-initiated revocation
- On-chain audit trail

**Contract Location:** `../contracts/pox-automation.clar`

### 2. STX → BTC Yield Math

The yield distribution follows mathematical formulas ensuring fairness and protocol sustainability:

#### Base Reward Formula
```
R = α · (S / ΣSᵢ) · R_total
```

Where:
- **S**: STX delegated by user
- **R**: BTC reward per cycle
- **Φ**: Spectral integration score (200-1000)
- **α**: Pool efficiency (0.92-0.98)

#### Treasury Split
```
R_T = R · min(0.30, 0.05 + Φ/2000)
```

Properties:
- Base rate: 5%
- Maximum rate: 30%
- Higher Φ scores increase treasury share
- Sybil resistant (monotonic in Φ)

#### NFT Yield Multiplier
```
U' = U · (1 + log₂(1 + Φ/500))
```

Users holding rarity NFTs receive boosted yields (typically 2x+).

### 3. Formal Security Proofs

The system includes four mathematical theorems proving security:

**Theorem 1 — Spectral Unforgeability**
- Cannot forge Φ scores without computing Riemann ζ zeros
- PSPACE-complete computational hardness
- Hash commitments bind entropy to blocks

**Theorem 2 — Cross-Chain Replay Resistance**
- Tetraroot proofs include chain_id binding
- Merkle structure prevents replay attacks
- Cryptographically enforced isolation

**Theorem 3 — Economic Capture Resistance**
- DAO cannot steal treasury funds
- Immutable economic rules
- No transfer authority in governance

**Theorem 4 — PoX Delegation Safety**
- User STX cannot be stolen
- Non-custodial architecture
- Revocable at any time

See [../docs/security/formal_proofs.md](../docs/security/formal_proofs.md) for complete proofs.

### 4. One-Click Installer

Install SphinxOS with a single command:

```bash
curl -sSL https://install.sphinxos.ai | bash
```

Or build from source:

```bash
git clone https://github.com/Holedozer1229/Sphinx_OS.git
cd Sphinx_OS
pip install -r requirements.txt
python3 -m sphinx_os.economics.simulator
```

The installer:
- Verifies checksums
- Installs binaries
- Initializes keys
- Launches local app
- Auto-connects to chains

**Binary Packaging:**
- macOS: `.app` bundle
- Linux: Static binary
- Windows: `.exe` installer
- iOS: Via Pyto/TestFlight
- Web: WASM (Pyodide)

### 5. Economic Simulator

Run revenue projections and scenario analysis:

```python
from sphinx_os.economics.simulator import EconomicSimulator, SimulationScenario

simulator = EconomicSimulator()

# Conservative scenario
scenario = SimulationScenario(
    name="Conservative",
    num_users=5000,
    avg_stx_per_user=10000,
    phi_mean=650,
    phi_stddev=100,
    btc_price_usd=45000
)

result = simulator.simulate_scenario(scenario, verbose=True)
print(f"Annual Treasury: ${result.annual_treasury_usd:,.2f}")
print(f"Annual User Yield: ${result.annual_user_yield_usd:,.2f}")
```

**Example Output (Conservative):**
- Treasury Revenue: ~$420,000/year
- User Yield: ~$2,800,000/year
- Average User ROI: ~11.2%

**Flywheel Effect:**
```
Higher Φ → Higher NFT value → More STX delegated →
More BTC rewards → Higher treasury → More development → Higher Φ
```

This self-reinforcing cycle creates exponential growth over time.

### Revenue Projections

| Scenario | Users | Φ Mean | Treasury/Year | User Yield/Year |
|----------|-------|--------|---------------|-----------------|
| Conservative | 5K | 650 | $420K | $2.8M |
| Moderate | 15K | 700 | $1.45M | $9.8M |
| Aggressive | 50K | 750 | $5.6M | $37M |
| Maximum | 100K | 800 | $13.2M | $87M |

---

## Next Steps

The economic system roadmap includes:

1. 📜 Full whitepaper PDF update
2. 🧠 AI-governed treasury heuristics
3. 🔒 Hardware wallet integration (Ledger/Xverse)
4. 🌐 WASM web-only version
5. 🚀 Mainnet launch checklist

For implementation details, see:
- [ECONOMICS.md](../ECONOMICS.md) - Complete economic system guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment and infrastructure
- [MAINNET_CHECKLIST.md](../MAINNET_CHECKLIST.md) - Launch preparation
