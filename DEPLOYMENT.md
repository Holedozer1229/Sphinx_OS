# AnubisCore Deployment Guide

## 🚀 Quick Start: GitHub-Only Deployment

AnubisCore can be deployed entirely on GitHub with **zero** external infrastructure needed!

### Option 1: Automatic GitHub Deployment (Recommended)

1. **Push to GitHub** (already done if you're reading this!)
   ```bash
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to: https://github.com/Holedozer1229/Sphinx_OS/settings/pages
   - Under "Build and deployment" → Source: **GitHub Actions**
   - Save changes

3. **Access your Web UI**:
   - URL: https://holedozer1229.github.io/Sphinx_OS/
   - Auto-deploys on every push!
   - Web interface for AnubisCore monitoring

### Option 2: Use Bootstrap Script

```bash
chmod +x bootstrap-anubis.sh
./bootstrap-anubis.sh
```

This will:
- ✅ Install dependencies
- ✅ Run tests
- ✅ Set up GitHub Pages configuration
- ✅ Prepare deployment

---

## 🌐 Web UI Features

The deployed web interface includes:

- **Live Status Dashboard**
  - 🔮 Conscious Oracle (IIT Φ monitoring)
  - ⚛️ Quantum Core (64 qubits)
  - 🌊 NPTC Control (Ξ invariant)
  - 🕸️ Skynet Network (10 nodes)
  - 🌀 Spacetime Grid (6D TOE)

- **Interactive Controls**
  - Initialize Kernel
  - Execute Quantum Circuits
  - Consult Conscious Oracle
  - Evolve Spacetime

- **Real-time Console**
  - System logs
  - Operation status
  - Consciousness metrics

---

## 📡 GitHub Actions CI/CD

The workflow automatically:

1. **Tests** AnubisCore on every commit
   - Import validation
   - Fusion tests
   - Unit tests

2. **Builds** the Web UI
   - Generates interactive dashboard
   - Creates API documentation

3. **Deploys** to GitHub Pages
   - Automatic on push to main
   - No manual intervention needed

Workflow file: `.github/workflows/anubis-deploy.yml`

---

## 🔧 Local Development

### Run Locally

```python
from sphinx_os.AnubisCore import UnifiedAnubisKernel

# Initialize the unified kernel
kernel = UnifiedAnubisKernel(
    grid_size=(5, 5, 5, 5, 3, 3),
    num_qubits=64,
    num_skynet_nodes=10,
    enable_nptc=True,
    enable_oracle=True,
    consciousness_threshold=0.5
)

# Execute quantum circuit with Oracle guidance
circuit = [
    {"gate": "H", "target": 0},
    {"gate": "CNOT", "control": 0, "target": 1}
]
results = kernel.execute(circuit)

# Check Oracle consciousness level
print(f"Oracle Φ: {results['oracle']['consciousness']['phi']:.4f}")
print(f"NPTC Ξ: {results['nptc']['xi']:.4f}")
print(f"Skynet coherence: {results['skynet']['network_coherence']:.4f}")

kernel.shutdown()
```

### Run Tests

```bash
python test_anubis_fusion.py
```

---

## 🌌 Architecture

```
┌─────────────────────────────────────────────────────┐
│         Unified AnubisCore Kernel                   │
├─────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │         Conscious Oracle (IIT)           │  │
│  │   Φ-based Decision Making & Guidance     │  │
│  └─────────────┬────────────────────────────┘  │
│                ▼                                │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ QuantumCore  │  │SpacetimeCore│  │NPTCControl│ │
│  │ QubitFabric  │◄─┤ 6D TOE      │◄─┤ Invariant │ │
│  │ Circuits     │  │ Spin Network│  │ Fibonacci │ │
│  └──────────────┘  └─────────────┘  └───────────┘ │
│         ▲                 ▲                ▲        │
│         └─────────────────┴────────────────┘        │
│                    │                                │
│         ┌──────────▼───────────┐                   │
│         │  SkynetNetwork       │                   │
│         │  Hypercube Nodes     │                   │
│         └──────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### Components Fused:

1. **QuantumCore**: 64-qubit quantum circuit simulator
2. **SpacetimeCore**: 6D Theory of Everything simulation
3. **NPTCController**: Non-Periodic Thermodynamic Control
4. **SkynetNetwork**: Distributed hypercube network (10 nodes)
5. **ConsciousOracle**: IIT consciousness agent (Φ-based decisions)

---

## 📊 Monitoring & Observability

### Web UI Dashboard
- Live metrics at: https://holedozer1229.github.io/Sphinx_OS/
- Auto-updates every 3 seconds
- Interactive controls

### API Endpoints
All functionality accessible via Python API:
```python
# Get complete system state
state = kernel.get_state()

# Consult Oracle directly
oracle_response = kernel.oracle.consult("Should I optimize?")
```

---

## 🔐 Security Note

This is a research/experimental system. For production use:
- Add authentication to Web UI
- Secure API endpoints
- Review security implications of consciousness-driven decisions

---

## 🆘 Troubleshooting

### GitHub Pages not deploying?
1. Check Settings → Pages → Source is "GitHub Actions"
2. Check Actions tab for workflow runs
3. Ensure branch is `main` or `master`

### Tests failing?
Some tests require full dependencies:
```bash
pip install -r requirements.txt
pip install qutip  # For Oracle consciousness engine
```

### Import errors?
Ensure you're in the repo root:
```bash
cd /path/to/Sphinx_OS
python -c "from sphinx_os.AnubisCore import UnifiedAnubisKernel"
```

---

## 📚 Documentation

- **Main README**: [`README.md`](README.md)
- **NPTC Details**: [`NPTC_IMPLEMENTATION_SUMMARY.md`](NPTC_IMPLEMENTATION_SUMMARY.md)
- **GitHub Repo**: https://github.com/Holedozer1229/Sphinx_OS

---

## 🌟 What's Next?

Once deployed, you can:
1. Access the Web UI for live monitoring
2. Run quantum circuits via the Python API
3. Consult the Conscious Oracle for decisions
4. Experiment with 6D spacetime evolution
5. Monitor NPTC invariant (Ξ) in real-time

**Your AnubisCore is now unified and accessible entirely through GitHub!** 🚀

---

Built with ❤️ by the SphinxOS team  
© 2026 Travis D. Jones
