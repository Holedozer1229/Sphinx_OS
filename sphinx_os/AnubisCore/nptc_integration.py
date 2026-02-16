"""
NPTC Integration - Non-Periodic Thermodynamic Control for AnubisCore

Integrates the NPTC framework into the unified kernel:
- NPTC invariant computation (Ξ = (ℏω/kT) · C_geom)
- Fibonacci scheduling
- Icosahedral Laplacian
- Quantum-classical boundary control
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("SphinxOS.AnubisCore.NPTCController")


class NPTCController:
    """
    NPTC Controller for maintaining quantum-classical boundary.
    
    Controls the system to maintain NPTC invariant Ξ ≈ 1.
    """
    
    def __init__(self, tau: float = 1e-6, T_eff: float = 1.5):
        """
        Initialize NPTC Controller.
        
        Args:
            tau: Control timescale (seconds)
            T_eff: Effective temperature (Kelvin)
        """
        self.tau = tau
        self.T_eff = T_eff
        self.xi_history = []
        self.control_steps = 0
        
        logger.info(f"NPTC Controller initialized (tau={tau}, T_eff={T_eff}K)")
        
        # Try to import NPTC framework
        try:
            from quantum_gravity.nptc_framework import NPTCFramework
            self.nptc = NPTCFramework(tau=tau, T_eff=T_eff)
            logger.info("✅ NPTC Framework integrated")
        except ImportError as e:
            logger.warning(f"Could not import NPTC Framework: {e}")
            self.nptc = None
    
    def apply_control(
        self,
        quantum_state: Optional[np.ndarray] = None,
        spacetime_metric: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Apply NPTC control to maintain quantum-classical boundary.
        
        Args:
            quantum_state: Current quantum state
            spacetime_metric: Current spacetime metric
            
        Returns:
            NPTC control results
        """
        self.control_steps += 1
        logger.debug(f"Applying NPTC control, step {self.control_steps}")
        
        if self.nptc is not None:
            try:
                # Compute NPTC invariant
                xi = self.nptc.compute_invariant()
                self.xi_history.append(xi.value)
                
                # Check if at boundary
                at_boundary = xi.is_critical()
                
                # Apply feedback control if needed
                if not at_boundary:
                    adjustment = self._compute_control_adjustment(xi.value)
                else:
                    adjustment = 0.0
                
                return {
                    "xi": xi.value,
                    "at_boundary": at_boundary,
                    "adjustment": adjustment,
                    "control_steps": self.control_steps,
                    "mean_xi": np.mean(self.xi_history) if self.xi_history else 0.0
                }
            except Exception as e:
                logger.warning(f"Error in NPTC control: {e}")
        
        # Fallback control
        logger.warning("Using fallback NPTC control")
        xi_fallback = 1.0 + 0.1 * np.random.randn()
        self.xi_history.append(xi_fallback)
        
        return {
            "xi": xi_fallback,
            "at_boundary": abs(xi_fallback - 1.0) < 0.1,
            "adjustment": 0.0,
            "control_steps": self.control_steps,
            "mean_xi": np.mean(self.xi_history) if self.xi_history else 0.0,
            "fallback": True
        }
    
    def _compute_control_adjustment(self, xi: float) -> float:
        """Compute control adjustment to bring Ξ closer to 1."""
        # Simple proportional control
        error = xi - 1.0
        K_p = 0.1  # Proportional gain
        adjustment = -K_p * error
        return adjustment
    
    def get_state(self) -> Dict[str, Any]:
        """Get current NPTC controller state."""
        return {
            "tau": self.tau,
            "T_eff": self.T_eff,
            "control_steps": self.control_steps,
            "xi_history_length": len(self.xi_history),
            "mean_xi": np.mean(self.xi_history) if self.xi_history else 0.0,
            "current_xi": self.xi_history[-1] if self.xi_history else None,
            "has_nptc_framework": self.nptc is not None
        }
    
    def shutdown(self):
        """Shutdown NPTC controller."""
        logger.info("NPTC Controller shutdown")
