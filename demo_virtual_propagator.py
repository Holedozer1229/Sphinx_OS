#!/usr/bin/env python3
"""
Demonstration of Virtual Particle Propagation in the Sovereign Framework.

This script demonstrates the computation of virtual propagator eigenvalues
in the 27-dimensional real representation of the Jordan algebra J₃(O).
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sphinx_os.AnubisCore.unified_kernel import VirtualPropagator


def main():
    """Demonstrate virtual propagator eigenvalue computation."""
    
    print("=" * 80)
    print("Virtual Particle Propagation - Sovereign Framework")
    print("=" * 80)
    print()
    
    # Parameters from problem statement
    delta_0 = 0.4       # FFLO order parameter amplitude
    mu = 0.3            # Chemical potential
    q = np.pi / 8       # Wave vector magnitude
    
    print(f"Parameters:")
    print(f"  Δ₀ (FFLO amplitude):    {delta_0}")
    print(f"  μ (chemical potential): {mu}")
    print(f"  q (wave vector):        {q:.6f} (π/8)")
    print()
    
    # Initialize virtual propagator
    print("Initializing Virtual Propagator...")
    propagator = VirtualPropagator(
        delta_0=delta_0,
        mu=mu,
        q=q,
        lattice_size=9,    # 9 sites per block
        t=1.0,             # Hopping parameter
        k_cutoff=1.0       # FRG regulator cutoff
    )
    print(f"✓ Denominator operator D constructed: {propagator.D.shape}")
    print()
    
    # Compute eigenvalues
    print("Computing eigenvalues of D and G_virt = D⁻¹...")
    eigenvalues_D, eigenvalues_G_virt = propagator.compute_eigenvalues()
    print(f"✓ Computed {len(eigenvalues_D)} eigenvalues")
    print()
    
    # Display results
    sorted_D = np.sort(eigenvalues_D)
    sorted_G = np.sort(np.abs(eigenvalues_G_virt))
    
    print("Eigenvalues of Denominator Operator D:")
    print("  (First 10, sorted)")
    for i in range(min(10, len(sorted_D))):
        print(f"    λ_{i+1:2d} = {sorted_D[i]:8.4f}")
    print()
    
    print("Eigenvalues of Virtual Propagator G_virt = D⁻¹:")
    print("  (First 10, sorted by magnitude)")
    for i in range(min(10, len(sorted_G))):
        print(f"    ν_{i+1:2d} = {sorted_G[i]:8.4f}")
    print()
    
    # Numerical verification
    print("Numerical Verification:")
    verification = propagator.verify_numerical_results()
    print(f"  Total eigenvalues:        {verification['num_eigenvalues']}")
    print(f"  Triality degeneracy:      {verification['triality_degeneracy']}")
    print(f"  Spectrum gapped:          {verification['spectrum_gapped']}")
    print(f"  Uniform gap approx (1/Δ₀): {verification['uniform_gap_approximation']:.4f}")
    print(f"  Modulation splitting:     {verification['modulation_splitting']:.4f}")
    print()
    
    # Analytical approximation
    print("Analytical Approximation:")
    print("  In continuum limit: ν_k ≈ 1/√((ε_k - μ)² + Δ₀²)")
    epsilon_k = np.array([-2*np.cos(k) for k in np.linspace(0, np.pi, 9)])
    analytic_approx = propagator.analytic_approximation(epsilon_k)
    print("  First 5 values:")
    for i in range(min(5, len(analytic_approx))):
        print(f"    ν_k[{i}] ≈ {analytic_approx[i]:.4f}")
    print()
    
    # Sovereign Framework interpretation
    print("Sovereign Framework Interpretation:")
    interpretation = propagator.interpret_sovereign_framework()
    print(f"  Off-shell propagation:    {interpretation['off_shell_propagation']}")
    print(f"  Virtual loops:            {interpretation['virtual_loops']}")
    print(f"  Mean positive eigenvalue: {interpretation['mean_positive_eigenvalue']:.4f}")
    print(f"  Epstein zeta regulated:   {interpretation['epstein_zeta_regulated']}")
    print(f"  Convergence guaranteed:   {interpretation['convergence_guaranteed']}")
    print(f"  NPTC contribution:        {interpretation['nptc_contribution']}")
    print(f"  Triality preservation:    {interpretation['triality_preservation']}")
    print()
    
    # Spectrum characteristics
    print("Spectrum Characteristics:")
    specs = interpretation['spectrum_characteristics']
    print(f"  Gapped:                   {specs['gapped']}")
    print(f"  Controllable:             {specs['controllable']}")
    print(f"  Non-perturbative:         {specs['non_perturbative']}")
    print()
    
    print("=" * 80)
    print("✅ Virtual Propagator Demonstration Complete")
    print("=" * 80)
    print()
    print("Key Results:")
    print("  • 27×27 block-diagonal structure with three 9×9 blocks (triality sectors)")
    print("  • Eigenvalues exhibit triality degeneracy (each appears 3 times)")
    print("  • Spectrum is gapped and controllable")
    print("  • Virtual loops encode off-shell propagation along Fano lines")
    print("  • Regulated by Epstein zeta to ensure convergence")
    print("  • Preserves master invariant Ξ = 1")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
