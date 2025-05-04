# sphinx_os/core/physics_daemon.py
"""
PhysicsDaemon: Background daemon for real-time field evolution.
"""
import threading
import time
import numpy as np
from ..utils.constants import CONFIG
import logging

logger = logging.getLogger("SphinxOS.PhysicsDaemon")

class PhysicsDaemon(threading.Thread):
    """Background daemon for real-time field evolution."""
    
    def __init__(self, kernel):
        """
        Initialize the PhysicsDaemon.

        Args:
            kernel (AnubisCore): The AnubisCore instance managing the simulation.
        """
        super().__init__(daemon=True)
        self.kernel = kernel
        self.running = True
        self.time_step = 0
        self.lock = threading.Lock()
        logger.info("PhysicsDaemon initialized")

    def run(self):
        """Run the daemon to continuously evolve fields."""
        while self.running:
            start_time = time.time()
            with self.lock:
                self.kernel.nugget_field = self._evolve_field(
                    self.kernel.nugget_field, CONFIG["m_nugget"], CONFIG["lambda_nugget"]
                )
                self.kernel.higgs_field = self._evolve_field(
                    self.kernel.higgs_field, CONFIG["m_higgs"], CONFIG["lambda_higgs"]
                )
                self.kernel.toe.evolve_fermion_fields()
                self.kernel.toe.compute_rydberg_effect()  # Ensure Rydberg effects are updated
                self.kernel.metric, self.kernel.inverse_metric = self.kernel.toe.compute_quantum_metric()
            self.time_step += 1
            elapsed = time.time() - start_time
            time.sleep(max(0, CONFIG["physics_refresh_interval"] - elapsed))
            logger.debug("PhysicsDaemon step %d completed in %.3f seconds", self.time_step, elapsed)

    def _evolve_field(self, field: np.ndarray, mass: float, coupling: float) -> np.ndarray:
        """
        Evolve a scalar field using the 6D Klein-Gordon equation.

        Args:
            field (np.ndarray): Scalar field to evolve.
            mass (float): Mass parameter.
            coupling (float): Coupling constant.

        Returns:
            np.ndarray: Evolved field.
        """
        laplacian = sum(np.gradient(np.gradient(field, self.kernel.adaptive_grid.deltas[i], axis=i),
                                   self.kernel.adaptive_grid.deltas[i], axis=i) for i in range(6))
        field_new = field + CONFIG["dt"] * (laplacian - mass**2 * field + coupling * field**3)
        field_new = np.clip(field_new, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        return np.nan_to_num(field_new, nan=0.0)
