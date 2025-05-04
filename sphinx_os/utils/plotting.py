# sphinx_os/utils/plotting.py
"""
Plotting: Utility for visualizing spacetime metrics and simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List

logger = logging.getLogger("SphinxOS.Plotting")

class SpacetimePlotter:
    """Utility for plotting spacetime metrics and simulation results."""
    
    def __init__(self, metric: np.ndarray):
        """
        Initialize the SpacetimePlotter.

        Args:
            metric (np.ndarray): The spacetime metric tensor.
        """
        self.metric = metric
        logger.info("SpacetimePlotter initialized")

    def add_gate_path(self, qubit_path: List[np.ndarray], color: str = 'blue') -> None:
        """
        Add a qubit gate path to the visualization, projecting 6D coordinates to 3D (x, y, z).

        Args:
            qubit_path (List[np.ndarray]): List of qubit positions in 6D space.
            color (str): Color of the path.
        """
        logger.debug("Visualizing gate path with color %s", color)
        fig = plt.figure(figsize=(8, 6), facecolor='#0A0B2E')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0A0B2E')
        path = np.array(qubit_path)  # Shape: (N, 6)
        # Project 6D coordinates to 3D (using x, y, z dimensions)
        ax.plot(path[:, 1], path[:, 2], path[:, 3], color=color, linewidth=2, zorder=2)
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.set_zlabel("Z", color='white')
        ax.set_title("Qubit Gate Path in Spacetime (Projected)", color='white')
        plt.savefig('gate_path.png', facecolor='#0A0B2E', edgecolor='none')
        plt.close()
        logger.debug("Gate path visualized with color %s", color)

    def show_ricci_heatmap(self, ricci_scalar: np.ndarray, grid_size: tuple) -> None:
        """
        Visualize the Ricci scalar as a heatmap.

        Args:
            ricci_scalar (np.ndarray): Ricci scalar field.
            grid_size (tuple): Grid dimensions.
        """
        logger.debug("Visualizing Ricci scalar heatmap")
        if ricci_scalar.shape != grid_size:
            logger.error("Ricci scalar shape %s does not match grid size %s", ricci_scalar.shape, grid_size)
            raise ValueError("Ricci scalar shape does not match grid size")
        # Take a 2D slice (e.g., at t=0, z=0, v=0, u=0)
        slice_2d = ricci_scalar[0, :, :, 0, 0, 0]
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0A0B2E')
        ax.set_facecolor('#0A0B2E')
        im = ax.imshow(slice_2d, cmap='plasma', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Ricci Scalar', color='white')
        ax.set_title('Ricci Scalar Heatmap (t=0, z=0, v=0, u=0)', color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.tick_params(colors='white')
        plt.savefig('ricci_heatmap.png', facecolor='#0A0B2E', edgecolor='none')
        plt.close()

    def show_rydberg_effect(self, rydberg_effect: np.ndarray, grid_size: tuple) -> None:
        """
        Visualize the Rydberg effect as a heatmap.

        Args:
            rydberg_effect (np.ndarray): Rydberg effect field.
            grid_size (tuple): Grid dimensions.
        """
        logger.debug("Visualizing Rydberg effect heatmap")
        if rydberg_effect.shape != grid_size:
            logger.error("Rydberg effect shape %s does not match grid size %s", rydberg_effect.shape, grid_size)
            raise ValueError("Rydberg effect shape does not match grid size")
        # Take a 2D slice (e.g., at t=0, z=0, v=0, u=0)
        slice_2d = rydberg_effect[0, :, :, 0, 0, 0]
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0A0B2E')
        ax.set_facecolor('#0A0B2E')
        im = ax.imshow(slice_2d, cmap='inferno', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Rydberg Effect', color='white')
        ax.set_title('Rydberg Effect Heatmap (t=0, z=0, v=0, u=0)', color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.tick_params(colors='white')
        plt.savefig('rydberg_effect_heatmap.png', facecolor='#0A0B2E', edgecolor='none')
        plt.close()
