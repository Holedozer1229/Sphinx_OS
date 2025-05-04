# sphinx_os/utils/helpers.py
"""
Helper functions for SphinxOS.
"""
import numpy as np
from scipy.linalg import svdvals
import logging
from typing import List, Tuple

logger = logging.getLogger("SphinxOS.Helpers")

def compute_entanglement_entropy(field: np.ndarray, grid_size: Tuple[int, ...]) -> float:
    """
    Compute entanglement entropy from a field.

    Args:
        field (np.ndarray): Quantum field.
        grid_size (Tuple[int, ...]): Grid dimensions.

    Returns:
        float: Mean entanglement entropy.
    """
    entropy = np.zeros(grid_size[:4], dtype=np.float64)
    for idx in np.ndindex(grid_size[:4]):
        local_state = field[idx].flatten()
        local_state = np.nan_to_num(local_state, nan=0.0)
        norm = np.linalg.norm(local_state)
        if norm > 1e-15:
            local_state /= norm
        psi_matrix = local_state.reshape(2, 2)
        schmidt_coeffs = svdvals(psi_matrix)
        probs = schmidt_coeffs**2
        probs = probs[probs > 1e-15]
        entropy[idx] = -np.sum(probs * np.log(probs)) if probs.size > 0 else 0
    if np.any(np.isnan(entropy)):
        logger.warning("NaN detected in entanglement entropy calculation")
    return np.mean(entropy)

def construct_6d_gamma_matrices(metric: np.ndarray) -> List[np.ndarray]:
    """
    Construct 6D Dirac gamma matrices.

    Args:
        metric (np.ndarray): Metric tensor.

    Returns:
        List[np.ndarray]: Gamma matrices.
    """
    gamma_flat = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex64),
        np.array([[0, 0, 0, 1], [0, 0,
