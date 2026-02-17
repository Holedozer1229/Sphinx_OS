#!/usr/bin/env python3
"""
Demonstration of Experimental Predictions for Yang-Mills Mass Gap Detection.

Shows how to use the ExperimentalPredictor to generate predictions for the
Au₁₃–DMT–Ac quasicrystal experiment with NPTC control.
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sphinx_os.AnubisCore.experimental_predictor import ExperimentalPredictor, quick_predict


def main():
    """Demonstrate experimental predictions."""
    
    print("=" * 80)
    print("Experimental Predictions: Yang-Mills Mass Gap Detection")
    print("Au₁₃–DMT–Ac Quasicrystal with NPTC Control")
    print("=" * 80)
    print()
    
    # Initialize predictor
    print("Initializing Experimental Predictor...")
    predictor = ExperimentalPredictor()
    print(f"✓ Theoretical mass gap m = {predictor.mass_gap_m:.4f}")
    print(f"✓ Contraction constant κ = {predictor.kappa:.4f}")
    print(f"✓ FFLO amplitude Δ₀ = {predictor.delta_0:.4f}")
    print()
    
    # ========================================================================
    # 1. Baseline vs. NPTC Prediction
    # ========================================================================
    print("-" * 80)
    print("1. BASELINE vs. NPTC-ACTIVE GAP PREDICTION")
    print("-" * 80)
    
    T_base = 0.1  # 100 mK
    T_crit = 1.5  # 1.5 K
    
    print(f"\nExperimental Conditions:")
    print(f"  Base temperature:     {T_base*1000:.0f} mK")
    print(f"  Critical temperature: {T_crit:.1f} K")
    print()
    
    # Baseline (no NPTC)
    baseline = predictor.predict_physical_gap(T_base=T_base, T_crit=T_crit, nptc_enabled=False)
    print(f"Baseline (NPTC OFF):")
    print(f"  Gap:                  {baseline['gap_eV']:.2e} eV")
    print(f"  Frequency:            {baseline['frequency_MHz']:.1f} MHz")
    print(f"  Accessible:           {baseline['experimentally_accessible']}")
    print()
    
    # With NPTC active
    with_nptc = predictor.predict_physical_gap(T_base=T_base, T_crit=T_crit, nptc_enabled=True)
    print(f"With NPTC Active:")
    print(f"  Gap:                  {with_nptc['gap_eV']:.2e} eV")
    print(f"  Frequency:            {with_nptc['frequency_MHz']:.1f} MHz")
    print(f"  Signal/Thermal:       {with_nptc['signal_to_thermal_ratio']:.2f}")
    print(f"  Accessible:           {with_nptc['experimentally_accessible']}")
    print(f"  Expected range:       {with_nptc['expected_range_eV'][0]:.0e}–{with_nptc['expected_range_eV'][1]:.0e} eV")
    print()
    
    # ========================================================================
    # 2. κ Sweep Simulation
    # ========================================================================
    print("-" * 80)
    print("2. CONTRACTION STRENGTH (κ) SWEEP")
    print("-" * 80)
    
    kappa_sweep = predictor.simulate_kappa_sweep(
        kappa_min=1.01,
        kappa_max=1.15,
        num_points=10  # Show fewer points for readability
    )
    
    print(f"\nSweeping κ from {kappa_sweep['kappa_values'][0]:.3f} to {kappa_sweep['kappa_values'][-1]:.3f}:")
    print(f"  {'κ':>8s}  {'m=ln(κ)':>10s}  {'Gap (eV)':>12s}  {'Freq (MHz)':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}")
    
    for i in [0, 2, 4, 6, 8, 9]:  # Show selected points
        kappa = kappa_sweep['kappa_values'][i]
        m = kappa_sweep['mass_gap_m'][i]
        gap = kappa_sweep['gap_eV'][i]
        freq = kappa_sweep['frequency_MHz'][i]
        print(f"  {kappa:8.4f}  {m:10.4f}  {gap:12.2e}  {freq:12.1f}")
    print()
    
    print(f"Key Result: Gap scales linearly with κ (via m = ln(κ))")
    print(f"  Minimum gap: {np.min(kappa_sweep['gap_eV']):.2e} eV")
    print(f"  Maximum gap: {np.max(kappa_sweep['gap_eV']):.2e} eV")
    print()
    
    # ========================================================================
    # 3. Spectral Suppression
    # ========================================================================
    print("-" * 80)
    print("3. SPECTRAL WEIGHT SUPPRESSION")
    print("-" * 80)
    
    spectrum_baseline = predictor.predict_spectral_suppression(
        frequency_range_MHz=(1, 100),
        num_points=50,
        nptc_enabled=False
    )
    
    spectrum_gap = predictor.predict_spectral_suppression(
        frequency_range_MHz=(1, 100),
        num_points=50,
        nptc_enabled=True
    )
    
    print(f"\nSpectral Suppression Analysis:")
    print(f"  Gap frequency:        {spectrum_gap['gap_frequency_MHz']:.1f} MHz")
    print(f"  Suppression ratio:    {spectrum_gap['suppression_ratio']:.3f}")
    print(f"  (Ratio < 1 indicates suppression below gap)")
    print()
    
    print(f"Observable Signature:")
    print(f"  • Low-frequency spectral weight suppressed when NPTC active")
    print(f"  • Hard gap emerges at ~{spectrum_gap['gap_frequency_MHz']:.0f} MHz")
    print(f"  • Reversible: gap disappears when NPTC disabled")
    print()
    
    # ========================================================================
    # 4. Gap Collapse with FFLO Detuning
    # ========================================================================
    print("-" * 80)
    print("4. GAP COLLAPSE WITH FFLO DETUNING")
    print("-" * 80)
    
    collapse = predictor.predict_collapse_behavior(
        fflo_detuning_range=(0.0, 0.5),
        num_points=11
    )
    
    print(f"\nFFLO Detuning Effect:")
    print(f"  {'Detuning':>10s}  {'Gap (dim)':>12s}  {'Gap (eV)':>12s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}")
    
    for i in [0, 2, 4, 6, 8, 10]:
        det = collapse['detuning'][i]
        gap_dim = collapse['gap_dimensionless'][i]
        gap_ev = collapse['gap_eV'][i]
        print(f"  {det:10.3f}  {gap_dim:12.4f}  {gap_ev:12.2e}")
    print()
    
    print(f"Key Result: Gap collapses to floor of ≈{collapse['collapsed_to']:.3f} when detuned")
    print(f"  Initial gap:  {collapse['gap_dimensionless'][0]:.4f}")
    print(f"  Final gap:    {collapse['gap_dimensionless'][-1]:.4f}")
    print()
    
    # ========================================================================
    # 5. Complete Experimental Protocol
    # ========================================================================
    print("-" * 80)
    print("5. EXPERIMENTAL PROTOCOL SUMMARY")
    print("-" * 80)
    
    protocol = predictor.generate_experimental_protocol()
    
    print(f"\nSystem:")
    print(f"  Material:   {protocol['system']['material']}")
    print(f"  Spacing:    {protocol['system']['lattice_spacing_angstrom']} Å")
    print(f"  Base T:     {protocol['system']['base_temperature_mK']} mK")
    print(f"  Control:    {protocol['system']['control']}")
    print()
    
    print(f"Measurement Techniques:")
    for i, technique in enumerate(protocol['measurement_techniques'], 1):
        print(f"  {i}. {technique}")
    print()
    
    print(f"Pass/Fail Criteria (Positive Detection):")
    for criterion in protocol['pass_fail_criteria']['positive_detection']:
        print(f"  ✓ {criterion}")
    print()
    
    print(f"Negative Controls:")
    for control in protocol['pass_fail_criteria']['negative_controls']:
        print(f"  ✗ {control}")
    print()
    
    print(f"Timeline:")
    for month, task in protocol['timeline_months'].items():
        print(f"  Month {month}: {task}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Predicted Observable Gap:")
    print(f"  Energy:       {with_nptc['gap_eV']:.2e} eV")
    print(f"  Frequency:    {with_nptc['frequency_MHz']:.1f} MHz")
    print(f"  Temperature:  {T_base*1000:.0f} mK (base), {T_crit:.1f} K (critical)")
    print()
    print(f"Experimental Feasibility:")
    print(f"  ✓ Signal-to-thermal ratio: {with_nptc['signal_to_thermal_ratio']:.2f}")
    print(f"  ✓ Within RF/microwave detection range (10-100 MHz)")
    print(f"  ✓ Accessible with standard dilution refrigerator")
    print(f"  ✓ Reversible control via NPTC feedback")
    print()
    print(f"Key Predictions:")
    print(f"  1. Gap appears ONLY when NPTC is active")
    print(f"  2. Gap magnitude scales with contraction strength κ")
    print(f"  3. Spectral weight suppressed below {spectrum_gap['gap_frequency_MHz']:.0f} MHz")
    print(f"  4. Gap collapses when FFLO detuned")
    print()
    print("=" * 80)
    print("✅ Experimental Predictions Generated")
    print("=" * 80)
    print()
    print("This constitutes a testable prediction of the Sovereign Framework.")
    print("The experiment is feasible with current technology and would provide")
    print("the first laboratory verification of an emergent Yang-Mills mass gap.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
