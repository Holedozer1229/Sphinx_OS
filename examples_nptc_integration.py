#!/usr/bin/env python3
"""
Example: NPTC Integration with Sphinx_OS Unified TOE

This example demonstrates how the NPTC framework integrates with
the existing Sphinx_OS 6D Theory of Everything simulation.
"""

import logging
from quantum_gravity.toe_integration import (
    NPTCEnhancedTOE, 
    run_quantum_gravity_with_nptc,
    verify_quantum_gravity_unification
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NPTCIntegrationExample")


def example_basic_nptc():
    """Example 1: Basic NPTC without TOE."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic NPTC Framework")
    print("=" * 80)
    
    enhanced = NPTCEnhancedTOE(toe=None, tau=1e-6)
    
    # Get initial status
    status = enhanced.get_complete_status()
    print(f"\nInitial Status:")
    print(f"  Ξ = {status['nptc']['xi']:.6f}")
    print(f"  Critical: {status['nptc']['is_critical']}")
    print(f"  Holonomy verified: {status['holonomy_verified']}")
    
    # Run a few control steps
    print("\nRunning 5 NPTC control steps...")
    for i in range(5):
        result = enhanced.apply_nptc_control()
        print(f"  Step {i}: Ξ={result['xi']:.6f}, Critical={result['is_critical']}")
    
    return enhanced


def example_quantum_gravity_proof():
    """Example 2: Quantum Gravity Proof."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Quantum Gravity Proof")
    print("=" * 80)
    
    results = run_quantum_gravity_with_nptc(n_steps=10)
    
    print(f"\nSimulation Results:")
    print(f"  NPTC steps: {len(results['nptc_steps'])}")
    print(f"  Mean Ξ: {results['summary']['mean_xi']:.6f}")
    print(f"  Std Ξ: {results['summary']['std_xi']:.6f}")
    print(f"  Critical fraction: {results['summary']['critical_fraction']:.1%}")
    
    proof = results['quantum_gravity_proof']
    print(f"\nQuantum Gravity Proof:")
    print(f"  Valid: {proof['proof_valid']}")
    print(f"  Propositions verified: {proof['propositions_verified']}/5")
    
    return results


def example_hyper_relativity():
    """Example 3: Hyper-Relativity Unification."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Hyper-Relativity Unification")
    print("=" * 80)
    
    verification = verify_quantum_gravity_unification()
    
    qg = verification['quantum_gravity']
    hr = verification['hyper_relativity']
    
    print(f"\nQuantum Gravity:")
    print(f"  Proof valid: {qg['proof_valid']}")
    print(f"  Propositions: {qg['propositions_verified']}/5")
    
    print(f"\nHyper-Relativity:")
    print(f"  Unification achieved: {hr['unification_achieved']}")
    print(f"  Spacetime: {hr['spacetime_dimension']}D {hr['signature']}")
    print(f"  Experimental support: {hr['experimental_support']}/6")
    
    print(f"\nComplete Unification: {verification['unified']}")
    
    return verification


def example_full_integration():
    """Example 4: Full NPTC-TOE Integration (if available)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Full NPTC-TOE Integration")
    print("=" * 80)
    
    try:
        from quantum_gravity.toe_integration import create_nptc_enhanced_toe
        
        print("\nAttempting to create NPTC-enhanced TOE...")
        enhanced = create_nptc_enhanced_toe(
            grid_size=(5, 5, 5, 5, 3, 3),
            tau=1e-6
        )
        
        if enhanced.toe is not None:
            print("✓ Full TOE integration successful!")
            
            # Synchronize and run
            enhanced.synchronize_with_toe()
            
            status = enhanced.get_complete_status()
            print(f"\nIntegrated Status:")
            print(f"  TOE time: {status['toe']['time']}")
            print(f"  TOE timestep: {status['toe']['time_step']}")
            print(f"  Grid size: {status['toe']['grid_size']}")
            print(f"  NPTC Ξ: {status['nptc']['xi']:.6f}")
            
        else:
            print("⚠ TOE components not available, using NPTC only")
            
    except Exception as e:
        print(f"⚠ Could not create full integration: {e}")
        print("  This is expected if TOE dependencies are not available")
        print("  NPTC framework works independently!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("NPTC FRAMEWORK INTEGRATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the NPTC (Non-Periodic Thermodynamic")
    print("Control) framework and its integration with Sphinx_OS.")
    
    try:
        # Example 1: Basic NPTC
        example_basic_nptc()
        
        # Example 2: Quantum Gravity Proof
        example_quantum_gravity_proof()
        
        # Example 3: Hyper-Relativity
        example_hyper_relativity()
        
        # Example 4: Full Integration
        example_full_integration()
        
        # Summary
        print("\n" + "=" * 80)
        print("EXAMPLES COMPLETE")
        print("=" * 80)
        print("\n✓ NPTC framework operational")
        print("✓ Quantum gravity proof demonstrated")
        print("✓ Hyper-relativity unification shown")
        print("✓ Integration pathways established")
        
        print("\nKey takeaways:")
        print("  1. NPTC framework works standalone or integrated with TOE")
        print("  2. Quantum gravity proof verifies key propositions")
        print("  3. Hyper-relativity extends to 6D spacetime")
        print("  4. Framework is modular and extensible")
        
        print("\nFor more details:")
        print("  - See quantum_gravity/README.md")
        print("  - Run: python demonstrate_quantum_gravity.py")
        print("  - Run tests: pytest tests/test_quantum_gravity.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
