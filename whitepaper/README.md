# Non-Periodic Thermodynamic Control (NPTC) Whitepaper

This directory contains the whitepaper for Non-Periodic Thermodynamic Control (NPTC), a universal framework for stabilizing systems at the quantum-classical boundary.

## Contents

- **nptc_whitepaper.tex** - LaTeX source file for the whitepaper
- **nptc_whitepaper.pdf** - Compiled PDF whitepaper (1.9 MB, 13 pages)
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

The whitepaper presents:

- **NPTC Framework**: Non-periodic thermodynamic control using Fibonacci timing
- **Experimental Platform**: Au₁₃-DmT-Ac aerogel in optomechanical cavity
- **Applications**: 
  - Cross-chain zk-EVM proof mining
  - Spectral Bitcoin miner
  - Megaminx proof-of-solve protocol
- **Six Predictions**: Three confirmed experimentally, three awaiting cosmological/gravitational tests

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
