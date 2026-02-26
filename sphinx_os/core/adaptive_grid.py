# sphinx_os/core/adaptive_grid.py
"""
AdaptiveGrid: Dynamically refined 6D grid for spacetime simulation.
"""
import numpy as np
from typing import Tuple
from ..utils.constants import CONFIG
import logging

logger = logging.getLogger("SphinxOS.AdaptiveGrid")

class AdaptiveGrid:
    """Dynamically refined 6D grid for spacetime simulation."""
    
    def __init__(self, base_grid_size: Tuple[int, ...], max_refinement: int = 2):
        """
        Initialize the AdaptiveGrid.

        Args:
            base_grid_size (Tuple[int, ...]): Base dimensions of the 6D grid.
            max_refinement (int): Maximum refinement level.
        """
        self.base_grid_size = base_grid_size
        self.max_refinement = max_refinement
        self.refinement_levels = np.zeros(base_grid_size, dtype=np.int8)
        self.base_deltas = np.array([CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']], dtype=np.float64)
        self.deltas = self.base_deltas.copy()
        self.coordinates = self._generate_coordinates()
        logger.info("AdaptiveGrid initialized with base grid size %s", base_grid_size)

    def _generate_coordinates(self) -> np.ndarray:
        """Generate 6D coordinates based on current deltas.

        Returns:
            np.ndarray: Coordinate array with shape (*grid_size, 6).
        """
        dims = [np.linspace(0, self.deltas[i] * self.base_grid_size[i], self.base_grid_size[i], dtype=np.float64)
                for i in range(6)]
        return np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1)

    def refine(self, ricci_scalar: np.ndarray):
        """
        Refine grid based on curvature.

        Args:
            ricci_scalar (np.ndarray): Ricci scalar field with shape matching grid_size.
        """
        logger.debug("Refining grid based on Ricci scalar")
        threshold = np.percentile(np.abs(ricci_scalar), 90)
        mask = (np.abs(ricci_scalar) > threshold) & (self.refinement_levels < self.max_refinement)
        self.refinement_levels[mask] += 1
        refinement_factor = 2 ** np.max(self.refinement_levels)
        self.deltas = np.array([max(d / refinement_factor, 1e-15) for d in self.base_deltas])
        self.coordinates = self._generate_coordinates()
        logger.debug("Grid refined with max refinement level %d", np.max(self.refinement_levels))
