# sphinx_os/core/tetrahedral_lattice.py
"""
TetrahedralLattice: Represents a 6D tetrahedral lattice for spacetime coordinates.
"""
import numpy as np
from ..utils.constants import CONFIG
import logging

logger = logging.getLogger("SphinxOS.TetrahedralLattice")

class TetrahedralLattice:
    """Represents a 6D tetrahedral lattice for spacetime coordinates."""
    
    def __init__(self, adaptive_grid: AdaptiveGrid):
        """
        Initialize the lattice.

        Args:
            adaptive_grid (AdaptiveGrid): The adaptive grid instance.
        """
        self.grid_size = adaptive_grid.base_grid_size
        self.deltas = adaptive_grid.deltas
        self.coordinates = adaptive_grid.coordinates
        self.tetrahedra = []
        logger.info("TetrahedralLattice initialized with grid size %s", self.grid_size)

    def _define_tetrahedra(self):
        """Define tetrahedral structure for the lattice."""
        logger.debug("Defining tetrahedral structure")
        self.tetrahedra = []
        for t_idx in range(self.grid_size[0]):
            for v_idx in range(self.grid_size[4]):
                for u_idx in range(self.grid_size[5]):
                    for x_idx in range(self.grid_size[1] - 1):
                        for y_idx in range(self.grid_size[2] - 1):
                            for z_idx in range(self.grid_size[3] - 1):
                                vertices = [
                                    (t_idx, x_idx + dx, y_idx + dy, z_idx + dz, v_idx, u_idx)
                                    for dx in [0, 1] for dy in [0, 1] for dz in [0, 1]
                                ]
                                tetrahedra = [
                                    (vertices[0], vertices[1], vertices[3], vertices[7]),
                                    (vertices[0], vertices[2], vertices[3], vertices[7]),
                                    (vertices[0], vertices[2], vertices[6], vertices[7]),
                                    (vertices[0], vertices[4], vertices[6], vertices[7]),
                                    (vertices[0], vertices[4], vertices[5], vertices[7])
                                ]
                                self.tetrahedra.extend(tetrahedra)
        logger.debug("Defined %d tetrahedra", len(self.tetrahedra))

    def compute_barycentric_coordinates(self, point: np.ndarray, tetrahedron: Tuple[Tuple[int, ...], ...]) -> np.ndarray:
        """
        Compute barycentric coordinates for a point in a tetrahedron, using x, y, z dimensions.

        Args:
            point (np.ndarray): The point coordinates (shape: (6,)).
            tetrahedron (Tuple[Tuple[int, ...], ...]): The tetrahedron vertices.

        Returns:
            np.ndarray: Barycentric coordinates (shape: (4,)).
        """
        v0, v1, v2, v3 = [self.coordinates[vert] for vert in tetrahedron]
        p = point[1:4]  # Use x, y, z dimensions for tetrahedron
        v0, v1, v2, v3 = v0[1:4], v1[1:4], v2[1:4], v3[1:4]
        T = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        p_minus_v0 = p - v0
        try:
            b = np.linalg.solve(T, p_minus_v0)
            b0 = 1 - np.sum(b)
            bary_coords = np.array([b0, b[0], b[1], b[2]])
        except np.linalg.LinAlgError:
            bary_coords = np.array([0.25, 0.25, 0.25, 0.25])
        bary_coords = np.clip(bary_coords, 0, 1)
        bary_coords /= np.sum(bary_coords) + 1e-15
        return bary_coords

    def interpolate_field(self, field: np.ndarray, point: np.ndarray) -> float:
        """
        Interpolate field value at a given point.

        Args:
            field (np.ndarray): The field to interpolate.
            point (np.ndarray): The point coordinates (shape: (6,)).

        Returns:
            float: Interpolated field value.
        """
        for tetrahedron in self.tetrahedra:
            bary_coords = self.compute_barycentric_coordinates(point, tetrahedron)
            if np.all(bary_coords >= -1e-5) and np.all(bary_coords <= 1 + 1e-5):
                field_values = [field[vert] for vert in tetrahedron]
                return float(sum(bary_coords[i] * field_values[i] for i in range(4)))
        # Safeguard division by ensuring deltas are non-zero
        safe_deltas = np.where(self.deltas == 0, 1e-15, self.deltas)
        idx = tuple(np.clip(np.round(point / safe_deltas).astype(int), 0, np.array(self.grid_size) - 1))
        return float(field[idx])
