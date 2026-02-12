"""
Integration between NPTC Framework and Unified 6D TOE.

This module bridges the NPTC quantum gravity framework with the existing
Sphinx_OS Unified 6D Theory of Everything simulation.
"""

import numpy as np
from typing import Dict, Optional
import logging

from .nptc_framework import NPTCFramework
from .quantum_gravity_proof import QuantumGravityProof
from .hyper_relativity import HyperRelativityUnification

logger = logging.getLogger("SphinxOS.NPTCIntegration")

# Try to import TOE, but don't fail if not available
TOE_AVAILABLE = False
Unified6DTOE = None


class NPTCEnhancedTOE:
    """
    Enhanced Unified 6D TOE with NPTC Framework.
    
    Integrates NPTC control with the existing 6D spacetime simulation,
    providing quantum gravity unification and hyper-relativity extensions.
    """
    
    def __init__(self, toe: Optional[Unified6DTOE] = None, tau: float = 1e-6):
        """
        Initialize NPTC-enhanced TOE.
        
        Args:
            toe: Optional Unified6DTOE instance
            tau: NPTC fundamental timescale
        """
        self.toe = toe
        self.nptc = NPTCFramework(tau=tau)
        self.proof = QuantumGravityProof(nptc=self.nptc)
        self.unification = HyperRelativityUnification(nptc=self.nptc)
        
        logger.info("NPTC-Enhanced TOE initialized")
        
    def extract_geometric_complexity(self) -> float:
        """
        Extract geometric complexity from TOE quantum state.
        
        Uses entanglement entropy as a proxy for C_geom.
        
        Returns:
            Geometric complexity value
        """
        if self.toe is None:
            return 1.0
            
        # Compute entanglement entropy from quantum state
        psi = self.toe.quantum_state.flatten()
        probs = np.abs(psi)**2
        probs = probs[probs > 1e-15]  # Remove near-zero probabilities
        
        # von Neumann entropy
        S = -np.sum(probs * np.log(probs))
        
        # Normalize to get dimensionless complexity
        C_geom = S / np.log(len(probs)) if len(probs) > 1 else 0.0
        
        return C_geom
    
    def compute_effective_frequency(self) -> float:
        """
        Compute effective frequency from TOE dynamics.
        
        Uses wormhole resonance or NPTC spectral gap.
        
        Returns:
            Effective frequency (Hz)
        """
        if self.toe is not None:
            # Use resonance frequency from TOE config
            omega = 2 * np.pi * self.toe.CONFIG.get("resonance_frequency", 1e3)
        else:
            # Use NPTC spectral gap
            omega = self.nptc.omega_eff
            
        return omega
    
    def synchronize_with_toe(self):
        """
        Synchronize NPTC parameters with current TOE state.
        
        Updates NPTC based on TOE quantum state and fields.
        """
        if self.toe is None:
            logger.warning("No TOE instance to synchronize with")
            return
            
        # Extract geometric complexity from entanglement
        C_geom = self.extract_geometric_complexity()
        self.nptc.update_geometric_complexity(C_geom)
        
        # Update effective frequency
        omega_eff = self.compute_effective_frequency()
        self.nptc.omega_eff = omega_eff
        
        logger.info(f"Synchronized with TOE: C_geom={C_geom:.4f}, ω_eff={omega_eff:.2e} Hz")
    
    def apply_nptc_control(self) -> Dict:
        """
        Apply NPTC control step and update TOE if present.
        
        Returns:
            Control step results
        """
        # Synchronize with TOE state
        if self.toe is not None:
            self.synchronize_with_toe()
        
        # Perform NPTC control step
        result = self.nptc.control_step()
        
        # Update TOE fields based on NPTC feedback (if TOE exists)
        if self.toe is not None and result['is_critical']:
            # Modulate wormhole coupling based on NPTC invariant
            deviation = abs(result['xi'] - 1.0)
            if hasattr(self.toe, 'kappa_worm'):
                self.toe.kappa_worm *= (1.0 - 0.1 * deviation)
                
        return result
    
    def run_nptc_enhanced_simulation(self, n_steps: int) -> Dict:
        """
        Run NPTC-enhanced TOE simulation.
        
        Args:
            n_steps: Number of NPTC control steps
            
        Returns:
            Simulation results with NPTC metrics
        """
        results = {
            'nptc_steps': [],
            'toe_integrated': self.toe is not None,
            'quantum_gravity_proof': None,
            'hyper_relativity': None
        }
        
        logger.info(f"Running NPTC-enhanced simulation for {n_steps} steps...")
        
        # Run NPTC control loop
        for step in range(n_steps):
            nptc_result = self.apply_nptc_control()
            results['nptc_steps'].append(nptc_result)
            
            # Optionally evolve TOE
            if self.toe is not None and step % 10 == 0:
                # Evolve TOE for a few timesteps
                # (Note: actual TOE evolution would require proper initialization)
                logger.debug(f"TOE sync at step {step}")
        
        # Generate quantum gravity proof
        logger.info("Generating quantum gravity proof...")
        proof_summary = self.proof.generate_proof()
        results['quantum_gravity_proof'] = proof_summary
        
        # Compute hyper-relativity unification
        logger.info("Computing hyper-relativity unification...")
        unification_summary = self.unification.generate_full_unification()
        results['hyper_relativity'] = unification_summary
        
        # Summary statistics
        xi_values = [r['xi'] for r in results['nptc_steps']]
        results['summary'] = {
            'mean_xi': np.mean(xi_values),
            'std_xi': np.std(xi_values),
            'critical_fraction': sum(r['is_critical'] for r in results['nptc_steps']) / n_steps,
            'proof_valid': proof_summary['proof_valid'],
            'unification_achieved': unification_summary['unification_achieved']
        }
        
        logger.info("NPTC-enhanced simulation complete")
        logger.info(f"  Mean Ξ: {results['summary']['mean_xi']:.6f}")
        logger.info(f"  Critical fraction: {results['summary']['critical_fraction']:.2%}")
        logger.info(f"  Proof valid: {results['summary']['proof_valid']}")
        logger.info(f"  Unification: {results['summary']['unification_achieved']}")
        
        return results
    
    def get_complete_status(self) -> Dict:
        """
        Get complete status of NPTC-enhanced TOE.
        
        Returns:
            Complete status dictionary
        """
        xi = self.nptc.compute_invariant()
        
        status = {
            'nptc': {
                'xi': xi.value,
                'is_critical': xi.is_critical(),
                'omega_eff': self.nptc.omega_eff,
                'T_eff': self.nptc.T_eff,
                'C_geom': self.nptc.C_geom,
                'current_step': self.nptc.current_step
            },
            'toe_integrated': self.toe is not None,
            'holonomy_verified': self.nptc.verify_holonomy_identity()['verified']
        }
        
        if self.toe is not None:
            status['toe'] = {
                'time': self.toe.time,
                'time_step': self.toe.time_step,
                'grid_size': self.toe.grid_size
            }
        
        return status


def create_nptc_enhanced_toe(grid_size=(5, 5, 5, 5, 3, 3), tau=1e-6):
    """
    Factory function to create NPTC-enhanced TOE with proper initialization.
    
    Args:
        grid_size: 6D grid dimensions
        tau: NPTC fundamental timescale
        
    Returns:
        NPTCEnhancedTOE instance
    """
    if not TOE_AVAILABLE:
        logger.warning("TOE components not available")
        logger.warning("Creating NPTC framework without TOE integration")
        return NPTCEnhancedTOE(toe=None, tau=tau)
    
    try:
        from sphinx_os.core.adaptive_grid import AdaptiveGrid
        from sphinx_os.core.spin_network import SpinNetwork
        from sphinx_os.core.tetrahedral_lattice import TetrahedralLattice
        
        # Initialize TOE components
        adaptive_grid = AdaptiveGrid(base_grid_size=grid_size)
        spin_network = SpinNetwork(grid_size=grid_size)
        lattice = TetrahedralLattice()
        
        # Create TOE
        toe = Unified6DTOE(adaptive_grid, spin_network, lattice)
        
        # Create NPTC-enhanced version
        enhanced = NPTCEnhancedTOE(toe=toe, tau=tau)
        
        logger.info("Created NPTC-enhanced TOE with full integration")
        return enhanced
        
    except ImportError as e:
        logger.warning(f"Could not initialize full TOE: {e}")
        logger.warning("Creating NPTC framework without TOE integration")
        return NPTCEnhancedTOE(toe=None, tau=tau)


# Convenience functions

def run_quantum_gravity_with_nptc(n_steps: int = 100):
    """
    Run quantum gravity proof with NPTC framework.
    
    Args:
        n_steps: Number of NPTC control steps
        
    Returns:
        Complete results
    """
    enhanced = NPTCEnhancedTOE(toe=None)
    return enhanced.run_nptc_enhanced_simulation(n_steps)


def verify_quantum_gravity_unification():
    """
    Verify quantum gravity unification using NPTC framework.
    
    Returns:
        Verification results
    """
    enhanced = NPTCEnhancedTOE(toe=None)
    
    # Verify quantum gravity proof
    proof_summary = enhanced.proof.generate_proof()
    
    # Verify hyper-relativity unification
    unif_summary = enhanced.unification.generate_full_unification()
    
    return {
        'quantum_gravity': proof_summary,
        'hyper_relativity': unif_summary,
        'unified': proof_summary['proof_valid'] and unif_summary['unification_achieved']
    }
