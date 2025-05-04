# sphinx_os/core/spin_network.py
"""
SpinNetwork: Simulates a quantum spin network in 6D spacetime.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import logging
from typing import List, Dict, Tuple
from ..utils.constants import hbar, m_n, c, CONFIG, sigma
from ..utils.helpers import compute_entanglement_entropy

logger = logging.getLogger("SphinxOS.SpinNetwork")

class SpinNetwork:
    """Simulates a quantum spin network in 6D spacetime."""
    def __init__(self, grid_size: Tuple[int, ...]):
        """
        Initialize the spin network.

        Args:
            grid_size (Tuple[int, ...]): Dimensions of the 6D grid.
        """
        self.grid_size = grid_size
        self.total_points = np.prod(grid_size)
        self.state = np.ones(self.total_points, dtype=np.complex128) / np.sqrt(self.total_points)
        self.indices = np.arange(self.total_points).reshape(grid_size)
        self.ctc_buffer = []
        self.ctc_steps = CONFIG["time_delay_steps"]
        self.ctc_factor = CONFIG["ctc_feedback_factor"]

    def evolve(self, dt: float, lambda_field: np.ndarray, metric: np.ndarray, inverse_metric: np.ndarray,
               deltas: List[float], phi_N: np.ndarray, higgs_field: np.ndarray, em_fields: Dict,
               electron_field: np.ndarray, quark_field: np.ndarray) -> int:
        """
        Evolve the spin network state.

        Args:
            dt (float): Time step.
            lambda_field (np.ndarray): Lambda field.
            metric (np.ndarray): Metric tensor.
            inverse_metric (np.ndarray): Inverse metric tensor.
            deltas (List[float]): Grid spacings.
            phi_N (np.ndarray): Nugget field.
            higgs_field (np.ndarray): Higgs field.
            em_fields (Dict): Electromagnetic fields.
            electron_field (np.ndarray): Electron field.
            quark_field (np.ndarray): Quark field.

        Returns:
            int: Number of internal steps taken.
        """
        H = self._build_sparse_hamiltonian(lambda_field, metric, inverse_metric, deltas, phi_N,
                                          higgs_field, em_fields, electron_field, quark_field)
        state_flat = self.state.flatten()
        max_attempts = 3
        current_dt = dt
        for attempt in range(max_attempts):
            try:
                if len(self.ctc_buffer) >= self.ctc_steps:
                    state_past = self.ctc_buffer[-self.ctc_steps].flatten()
                    state_current = state_flat.copy()
                    for _ in range(3):
                        sol = solve_ivp(lambda t, y: -1j * H.dot(y) / hbar, [0, current_dt], state_current,
                                        method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                        if not sol.success:
                            raise ValueError("solve_ivp failed in SpinNetwork evolve")
                        state_evolved = sol.y[:, -1]
                        state_current = (1 - self.ctc_factor) * state_evolved + self.ctc_factor * state_past
                        norm = np.linalg.norm(state_current)
                        if norm > 0:
                            state_current /= norm
                        else:
                            logger.warning("Zero norm in SpinNetwork state evolution")
                            state_current = state_flat.copy()
                else:
                    sol = solve_ivp(lambda t, y: -1j * H.dot(y) / hbar, [0, current_dt], state_flat,
                                    method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                    if not sol.success:
                        raise ValueError("solve_ivp failed in SpinNetwork initial evolve")
                    state_current = sol.y[:, -1]
                    norm = np.linalg.norm(state_current)
                    if norm > 0:
                        state_current /= norm
                    else:
                        logger.warning("Zero norm in SpinNetwork initial state evolution")
                        state_current = state_flat.copy()
                break
            except Exception as e:
                logger.warning(f"SpinNetwork evolve failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
                logger.info(f"Retrying with reduced dt={current_dt}")
        self.state = state_current.reshape(self.grid_size)
        self.state = np.clip(self.state, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        self.ctc_buffer.append(self.state.copy())
        if len(self.ctc_buffer) > self.ctc_steps:
            self.ctc_buffer.pop(0)
        return len(sol.t) - 1

    def _build_sparse_hamiltonian(self, lambda_field: np.ndarray, metric: np.ndarray, inverse_metric: np.ndarray,
                                 deltas: List[float], phi_N: np.ndarray, higgs_field: np.ndarray, em_fields: Dict,
                                 electron_field: np.ndarray, quark_field: np.ndarray) -> csr_matrix:
        """
        Build the sparse Hamiltonian.

        Args:
            lambda_field (np.ndarray): Lambda field.
            metric (np.ndarray): Metric tensor.
            inverse_metric (np.ndarray): Inverse metric tensor.
            deltas (List[float]): Grid spacings.
            phi_N (np.ndarray): Nugget field.
            higgs_field (np.ndarray): Higgs field.
            em_fields (Dict): Electromagnetic fields.
            electron_field (np.ndarray): Electron field.
            quark_field (np.ndarray): Quark field.

        Returns:
            csr_matrix: Sparse Hamiltonian matrix.
        """
        from ..utils.constants import kappa, lambda_higgs, e, m_e, m_q
        state_grid = self.state.reshape(self.grid_size)
        N = self.total_points
        kinetic_term = np.zeros_like(state_grid, dtype=np.complex128)
        for mu in range(6):
            grad_mu = np.gradient(state_grid, deltas[mu], axis=mu)
            laplacian_mu = np.gradient(grad_mu, deltas[mu], axis=mu)
            kinetic_term += inverse_metric[..., mu, mu] * laplacian_mu
        kinetic_energy = -hbar**2 / (2 * m_n) * kinetic_term.flatten()

        potential_energy = np.zeros(N, dtype=np.complex128)
        phi_N_flat = phi_N.flatten()
        higgs_norm = np.abs(higgs_field.flatten())**2
        em_potential = np.abs(em_fields["A"][..., 0].flatten())
        for i in range(N):
            V_phi = kappa * phi_N_flat[i]**2
            V_higgs = lambda_higgs * higgs_norm[i]
            V_em = e * em_potential[i]
            potential_energy[i] = V_phi + V_higgs + V_em

        connection = self._compute_affine_connection(metric, inverse_metric, deltas)
        spin_gravity = np.zeros(N, dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            psi_e = electron_field[idx]
            spin_e = np.einsum('i,ij,j->', psi_e.conj(), sigma[2], psi_e).real
            psi_q = quark_field[idx]
            spin_q = 0
            for f in range(2):
                for c in range(3):
                    psi_q_fc = psi_q[f, c]
                    spin_q += np.einsum('i,ij,j->', psi_q_fc.conj(), sigma[2], psi_q_fc).real
            spin_density = (m_e * spin_e + m_q * spin_q) / (hbar * c)
            for mu in range(6):
                spin_gravity[i] += spin_density * connection[idx + (mu, mu, mu)]

        lambda_perturbation = lambda_field.flatten() * 1e-3
        diagonal = kinetic_energy + potential_energy + hbar * c * spin_gravity * 1e-3 + lambda_perturbation
        if np.any(np.isnan(diagonal)):
            logger.warning("NaN detected in Hamiltonian diagonal")

        rows, cols, data = [], [], []
        for idx in np.ndindex(self.grid_size):
            i = self.indices[idx]
            for mu in range(6):
                idx_plus = list(idx)
                idx_minus = list(idx)
                if idx[mu] < self.grid_size[mu] - 1:
                    idx_plus[mu] += 1
                    j = self.indices[tuple(idx_plus)]
                    coupling = inverse_metric[idx][mu, mu] * hbar * c / deltas[mu]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])
                if idx[mu] > 0:
                    idx_minus[mu] -= 1
                    j = self.indices[tuple(idx_minus)]
                    coupling = inverse_metric[idx][mu, mu] * hbar * c / deltas[mu]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])

        H = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)
        H += csr_matrix((diagonal, (np.arange(N), np.arange(N))), dtype=np.complex128)
        return H

    def _compute_affine_connection(self, metric: np.ndarray, inverse_metric: np.ndarray,
                                  deltas: List[float]) -> np.ndarray:
        """
        Compute the affine connection.

        Args:
            metric (np.ndarray): Metric tensor.
            inverse_metric (np.ndarray): Inverse metric tensor.
            deltas (List[float]): Grid spacings.

        Returns:
            np.ndarray: Affine connection tensor.
        """
        connection = np.zeros((*self.grid_size, 6, 6, 6), dtype=np.float64)
        for idx in np.ndindex(self.grid_size):
            if all(0 < i < s - 1 for i, s in zip(idx, self.grid_size)):
                for rho in range(6):
                    for mu in range(6):
                        for nu in range(6):
                            dg_mu = np.gradient(metric[..., mu, nu], deltas[rho], axis=rho)[idx]
                            dg_nu = np.gradient(metric[..., rho, mu], deltas[nu], axis=nu)[idx]
                            dg_rho = np.gradient(metric[..., rho, nu], deltas[mu], axis=mu)[idx]
                            connection[idx + (rho, mu, nu)] = 0.5 * np.einsum('rs,s->r', inverse_metric[idx],
                                                                              dg_mu + dg_nu - dg_rho)
        if np.any(np.isnan(connection)):
            logger.warning("NaN detected in affine connection")
        return np.nan_to_num(connection, nan=0.0)
