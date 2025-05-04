# sphinx_os/utils/helpers.py
"""
Helpers: Utility functions for quantum and spacetime calculations.
"""
import numpy as np
import logging
from typing import List

logger = logging.getLogger("SphinxOS.Helpers")

def compute_entanglement_entropy(field: np.ndarray, grid_size: tuple) -> float:
    """
    Compute entanglement entropy for a field.

    Args:
        field (np.ndarray): Field to compute entropy for (e.g., quantum_state, electron_field).
        grid_size (tuple): Dimensions of the grid.

    Returns:
        float: Entanglement entropy.
    """
    logger.debug("Computing entanglement entropy for field with shape %s", field.shape)
    total_points = np.prod(grid_size)
    # Handle fields with additional dimensions (e.g., electron_field has shape (*grid_size, 4))
    if field.shape[:len(grid_size)] != grid_size:
        # Reshape or reduce dimensions (e.g., for electron_field, compute norm over Dirac components)
        if len(field.shape) > len(grid_size):
            field = np.linalg.norm(field, axis=tuple(range(len(grid_size), len(field.shape))))
        else:
            logger.error("Field shape %s does not match grid size %s", field.shape, grid_size)
            raise ValueError("Field shape does not match grid size")
    # Flatten the field to a 1D array for entropy calculation
    probs = np.abs(field.flatten())**2
    probs = probs / (np.sum(probs) + 1e-15)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)

def construct_6d_gamma_matrices(metric: np.ndarray) -> List[np.ndarray]:
    """
    Construct 6D gamma matrices for Dirac equation.

    Args:
        metric (np.ndarray): 6D metric tensor at a point (shape: (6, 6)).

    Returns:
        List[np.ndarray]: List of 6D gamma matrices.
    """
    logger.debug("Constructing 6D gamma matrices")
    # Define 4D gamma matrices (Clifford algebra in 4D)
    I = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Standard 4D gamma matrices (Minkowski basis)
    gamma_0 = np.kron(I, sigma_z)  # Shape: (4, 4)
    gamma_1 = np.kron(sigma_x, sigma_x)
    gamma_2 = np.kron(sigma_x, sigma_y)
    gamma_3 = np.kron(sigma_x, sigma_z)
    gamma_4 = np.kron(sigma_y, I)  # Extra dimension (compact)
    gamma_5 = np.kron(sigma_z, I)  # Extra dimension (compact)

    gamma_matrices = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5]

    # Adjust gamma matrices with metric components
    for mu in range(6):
        metric_component = metric[mu, mu]
        if metric_component < 0:
            logger.warning("Negative metric component detected at index %d: %.3e", mu, metric_component)
            metric_component = abs(metric_component)  # Ensure positive for sqrt
        gamma_matrices[mu] *= np.sqrt(metric_component)

    return gamma_matrices

def compute_schumann_frequencies(N: int) -> List[float]:
    """
    Compute Schumann resonance frequencies.

    Args:
        N (int): Number of frequencies to compute.

    Returns:
        List[float]: List of Schumann frequencies in Hz.
    """
    logger.debug("Computing %d Schumann frequencies", N)
    base_freq = 7.83  # Fundamental Schumann frequency (Hz)
    frequencies = [base_freq * (n + 1) for n in range(N)]
    return frequencies
