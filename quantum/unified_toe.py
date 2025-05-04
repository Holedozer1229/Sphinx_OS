# sphinx_os/quantum/unified_toe.py
"""
Unified6DTOE: Implements a 6D Theory of Everything simulation.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, List, Optional
from ..utils.constants import (
    G, c, hbar, e, epsilon_0, m_e, m_q, m_h, m_n, v_higgs, l_p, kappa,
    lambda_higgs, RS, sigma, lambda_matrices, f_su2, f_su3, CONFIG
)
from ..utils.helpers import compute_entanglement_entropy, construct_6d_gamma_matrices, compute_schumann_frequencies
from ..utils.lattice import TetrahedralLattice
from ..core.spin_network import SpinNetwork

logger = logging.getLogger("SphinxOS.Unified6DTOE")

class Unified6DTOE:
    """Unified 6D Theory of Everything simulation."""
    def __init__(self):
        """Initialize the TOE simulation."""
        self.grid_size = CONFIG["grid_size"]
        self.total_points = np.prod(self.grid_size)
        self.dt = CONFIG["dt"]
        self.deltas = [CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']]
        self.time = 0.0

        self.spin_network = SpinNetwork(self.grid_size)
        self.lattice = TetrahedralLattice(self.grid_size)
        self.wormhole_nodes = self._generate_wormhole_nodes()

        self.bit_states = np.array([self._repeating_curve(sum(idx)) for idx in np.ndindex(self.grid_size)],
                                  dtype=np.int8).reshape(self.grid_size)
        self.temporal_entanglement = np.zeros(self.grid_size, dtype=np.complex128)
        self.quantum_state = np.ones(self.grid_size, dtype=np.complex128) / np.sqrt(self.total_points)
        self.higgs_field = np.ones(self.grid_size, dtype=np.complex128) * v_higgs
        self.electron_field = np.zeros((*self.grid_size, 4), dtype=np.complex128)
        self.quark_field = np.zeros((*self.grid_size, 2, 3, 4), dtype=np.complex128)
        self.phi_N = np.zeros(self.grid_size, dtype=np.float64)
        self.lambda_field = np.zeros(self.grid_size, dtype=np.float64)
        self.phi_range = np.linspace(-10.0, 10.0, 201, dtype=np.float64)
        self.d_phi = self.phi_range[1] - self.phi_range[0]
        self.phi_wave_functions = np.zeros((*self.grid_size, len(self.phi_range)), dtype=np.complex128)
        for idx in np.ndindex(self.grid_size):
            psi = np.exp(-self.phi_range**2)
            psi /= np.sqrt(np.sum(np.abs(psi)**2 * self.d_phi))
            self.phi_wave_functions[idx] = psi

        self.setup_symbolic_calculations()
        self.metric, self.inverse_metric = self.compute_quantum_metric()
        self.connection = self._compute_affine_connection()
        self.riemann_tensor = self._compute_riemann_tensor()
        self.ricci_tensor, self.ricci_scalar = self._compute_curvature()
        self.stress_energy = self._compute_stress_energy()
        self.einstein_tensor = self._compute_einstein_tensor()

        self.em_fields = self._initialize_em_fields()
        self.weak_fields = self._initialize_weak_fields()
        self.strong_fields = self._initialize_strong_fields()

        self.history = []
        self.fermion_history = []
        self.phi_N_history = []
        self.higgs_norm_history = []
        self.entanglement_history = []
        self.ricci_scalar_history = []
        self.spin_density_history = []
        self.lambda_history = []
        self.j4_history = []
        self.flux_history = []

    def _repeating_curve(self, index: int) -> int:
        """Generate a repeating binary pattern."""
        return 1 if index % 2 == 0 else 0

    def setup_symbolic_calculations(self):
        """Set up symbolic metric calculations."""
        self.t, self.x, self.y, self.z, self.v, self.u = sp.symbols('t x y z v u')
        self.a, self.c_sym, self.m, self.kappa_sym = sp.symbols('a c m kappa', positive=True)
        self.phi_N_sym = sp.Function('phi_N')(self.t, self.x, self.y, self.z, self.v, self.u)
        r = sp.sqrt(self.x**2 + self.y**2 + self.z**2 + 1e-10)
        schwarzschild = 1 - RS / r
        scaled_l_p = l_p * 1e30
        time_scale = 1e-16
        spatial_scale = 1.0
        compact_scale = 1e18
        g_diag = [
            sp.sympify(time_scale * (-self.c_sym**2 * (1 + self.kappa_sym * self.phi_N_sym) * schwarzschild)),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.phi_N_sym))),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.phi_N_sym))),
            sp.sympify(spatial_scale * (self.a**2 * (1 + self.kappa_sym * self.phi_N_sym))),
            sp.sympify(compact_scale * (scaled_l_p**2)),
            sp.sympify(compact_scale * (scaled_l_p**2))
        ]
        try:
            self.g = sp.diag(*g_diag)
        except Exception as e:
            logger.warning(f"sp.diag failed: {e}, falling back to manual construction")
            self.g = sp.zeros(6, 6)
            for i in range(6):
                self.g[i, i] = g_diag[i]
        self.g_inv = self.g.inv()
        self.metric_scale_factors = np.array([time_scale, spatial_scale, spatial_scale, spatial_scale, compact_scale, compact_scale])

    def _generate_wormhole_nodes(self) -> np.ndarray:
        """Generate wormhole node coordinates."""
        coords = np.zeros((*self.grid_size, 6), dtype=np.float64)
        dims = [np.linspace(0, CONFIG[f"d{dim}"] * size, size, dtype=np.float64)
                for dim, size in zip(['t', 'x', 'x', 'x', 'v', 'u'], self.grid_size)]
        T, X, Y, Z, V, U = np.meshgrid(*dims, indexing='ij')
        R, r = 1.5 * self.deltas[1], 0.5 * self.deltas[1]
        coords[..., 0] = (R + r * np.cos(CONFIG["omega"] * T)) * np.cos(X / self.deltas[1])
        coords[..., 1] = (R + r * np.cos(CONFIG["omega"] * T)) * np.sin(Y / self.deltas[1])
        coords[..., 2] = r * np.sin(CONFIG["omega"] * Z)
        coords[..., 3] = r * np.cos(CONFIG["omega"] * V)
        coords[..., 4] = r * np.sin(CONFIG["omega"] * U)
        coords[..., 5] = c * T
        return np.nan_to_num(coords, nan=0.0)

    def compute_quantum_metric(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 6D quantum metric."""
        metric = np.zeros((*self.grid_size, 6, 6), dtype=np.float64)
        coords = self.lattice.coordinates
        for idx in np.ndindex(self.grid_size):
            subs_dict = {
                self.t: coords[idx][0], self.x: coords[idx][1], self.y: coords[idx][2],
                self.z: coords[idx][3], self.v: coords[idx][4], self.u: coords[idx][5],
                self.a: CONFIG["a_godel"], self.c_sym: c, self.m: m_n,
                self.kappa_sym: kappa, self.phi_N_sym: self.phi_N[idx]
            }
            g = np.array(sp.N(self.g.subs(subs_dict)), dtype=np.float64)
            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                logger.warning(f"NaN/Inf detected in metric at idx {idx}: {g}")
                g = np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
            metric[idx] = 0.5 * (g + g.T)
        cond = np.linalg.cond(metric)
        if np.any(cond > 1e10):
            logger.warning(f"High condition number detected: {cond.max()}")
            inverse_metric = pinv(metric, rcond=1e-10)
        else:
            inverse_metric = np.linalg.inv(metric + 1e-5 * np.eye(6, dtype=np.float64))
        scale_matrix = np.diag(1.0 / self.metric_scale_factors)
        for idx in np.ndindex(self.grid_size):
            inverse_metric[idx] = scale_matrix @ inverse_metric[idx] @ scale_matrix
        return np.nan_to_num(metric, nan=0.0), np.nan_to_num(inverse_metric, nan=0.0)

    def _compute_affine_connection(self) -> np.ndarray:
        """Compute the affine connection."""
        connection = np.zeros((*self.grid_size, 6, 6, 6), dtype=np.float64)
        dg_dmu = [np.gradient(self.metric[..., i, j], self.deltas[k], axis=k)
                  for k in range(6) for i in range(6) for j in range(6)]
        dg_dmu = np.array(dg_dmu).reshape(6, 6, 6, *self.grid_size)
        for rho in range(6):
            for mu in range(6):
                for nu in range(6):
                    term = (dg_dmu[mu, nu, :] + dg_dmu[nu, mu, :] -
                            np.sum(dg_dmu[:, mu, nu] * self.inverse_metric[..., rho, :], axis=-1))
                    connection[..., rho, mu, nu] = 0.5 * term.transpose(1, 2, 3, 4, 5, 0)
        if np.any(np.isnan(connection)):
            logger.warning("NaN detected in affine connection")
        return np.nan_to_num(connection, nan=0.0)

    def _compute_riemann_tensor(self) -> np.ndarray:
        """Compute the Riemann curvature tensor."""
        riemann = np.zeros((*self.grid_size, 6, 6, 6, 6), dtype=np.complex128)
        for rho in range(6):
            for sigma in range(6):
                for mu in range(6):
                    for nu in range(6):
                        grad_nu_sigma = np.gradient(self.connection[..., rho, nu, sigma], self.deltas[nu], axis=nu)
                        grad_mu_sigma = np.gradient(self.connection[..., rho, mu, sigma], self.deltas[mu], axis=mu)
                        term1 = np.einsum('...km,...mn->...kn', self.connection[..., rho, :, mu],
                                          self.connection[..., :, nu, sigma])
                        term2 = np.einsum('...kn,...mn->...km', self.connection[..., rho, :, nu],
                                          self.connection[..., :, mu, sigma])
                        riemann[..., rho, sigma, mu, nu] = (grad_nu_sigma - grad_mu_sigma + term1 - term2)
        max_val = np.max(np.abs(riemann))
        if max_val > 1e5:
            riemann /= max_val
            logger.info(f"Riemann tensor scaled by {max_val} to prevent overflow")
        if np.any(np.isnan(riemann)):
            logger.warning("NaN detected in Riemann tensor")
        return np.nan_to_num(riemann, nan=0.0)

    def _compute_curvature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Ricci tensor and scalar."""
        ricci_tensor = np.einsum('...rsmn,...mn->...rs', self.riemann_tensor, self.inverse_metric)
        ricci_scalar = np.einsum('...mn,...mn->...', self.inverse_metric, ricci_tensor)
        if np.any(np.isnan(ricci_tensor)) or np.any(np.isnan(ricci_scalar)):
            logger.warning("NaN detected in curvature computation")
        return np.nan_to_num(ricci_tensor, nan=0.0), np.nan_to_num(ricci_scalar, nan=0.0)

    def _compute_stress_energy(self) -> np.ndarray:
        """Compute the stress-energy tensor."""
        T = np.zeros((*self.grid_size, 6, 6), dtype=np.complex128)
        F = self.em_fields["F"]
        F_nu_alpha = np.einsum('...nu, ...beta, ...alpha->...nu alpha', F, self.inverse_metric, self.metric)
        T_em = (np.einsum('...mu alpha,...nu alpha->...mu nu', F, F_nu_alpha) -
                0.25 * self.metric * np.einsum('...ab,...ab->...', F, F)) / (4 * np.pi * epsilon_0)
        T += T_em
        T += self.em_fields["J4"][..., np.newaxis, np.newaxis] * self.metric
        quantum_amplitude = np.abs(self.quantum_state)**2
        T[..., 0, 0] += -self.phi_N / c**2 + quantum_amplitude
        for i in range(1, 6):
            T[..., i, i] += quantum_amplitude / 5
        if np.any(np.isnan(T)):
            logger.warning("NaN detected in stress-energy tensor")
        return np.nan_to_num(T, nan=0.0)

    def _compute_einstein_tensor(self) -> np.ndarray:
        """Compute the Einstein tensor."""
        ricci_tensor, ricci_scalar = self._compute_curvature()
        einstein_tensor = ricci_tensor - 0.5 * self.metric * ricci_scalar[..., np.newaxis, np.newaxis]
        if np.any(np.isnan(einstein_tensor)):
            logger.warning("NaN detected in Einstein tensor")
        return np.nan_to_num(einstein_tensor, nan=0.0)

    def _initialize_em_fields(self) -> Dict:
        """Initialize electromagnetic fields."""
        A = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        r = np.linalg.norm(self.wormhole_nodes[..., :3], axis=-1) + 1e-15
        A[..., 0] = CONFIG["charge"] / (4 * np.pi * epsilon_0 * r)
        F = np.zeros((*self.grid_size, 6, 6), dtype=np.complex128)
        J = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        base_J = CONFIG["charge"] * c / (4 * np.pi * r**3)
        omega_res = 2 * np.pi * CONFIG["resonance_frequency"]
        resonance = 1 + CONFIG["resonance_amplitude"] * np.sin(omega_res * self.time)
        J[..., 0] = base_J * resonance
        for mu in range(6):
            for nu in range(6):
                grad_A_nu = np.gradient(A[..., nu], self.deltas[mu], axis=mu)
                grad_A_mu = np.gradient(A[..., mu], self.deltas[nu], axis=nu)
                F[..., mu, nu] = grad_A_nu - grad_A_mu
        J_norm = np.einsum('...m,...m->...', J, J)
        k = 2 * np.pi * 1e21
        x = k * self.lambda_field
        coupling = (-x**2 * np.cos(x) + 2 * x * np.sin(x) + 2 * np.cos(x)) * CONFIG["j4_scaling_factor"]
        J4 = J_norm**2 * coupling
        if np.any(np.isnan(J4)):
            logger.warning("NaN detected in J4 computation")
        return {"A": A, "F": F, "J": J, "J4": J4}

    def _initialize_weak_fields(self) -> Dict:
        """Initialize weak gauge fields."""
        W = np.random.normal(0, 1e-3, (*self.grid_size, 3, 6)).astype(np.complex128) * hbar * c / self.deltas[1]
        F_W = np.zeros((*self.grid_size, 3, 6, 6), dtype=np.complex128)
        for a in range(3):
            for mu in range(6):
                for nu in range(6):
                    dW_mu = np.gradient(W[..., a, nu], self.deltas[mu], axis=mu)
                    dW_nu = np.gradient(W[..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_weak"] * np.einsum('abc,...b,...c->...a', f_su2, W[..., mu], W[..., nu])[..., a]
                    F_W[..., a, mu, nu] = dW_mu - dW_nu + nonlinear
        return {"W": W, "F": F_W}

    def _initialize_strong_fields(self) -> Dict:
        """Initialize strong gauge fields."""
        G = np.random.normal(0, 1e-3, (*self.grid_size, 8, 6)).astype(np.complex128) * hbar * c / self.deltas[1]
        F_G = np.zeros((*self.grid_size, 8, 6, 6), dtype=np.complex128)
        for a in range(8):
            for mu in range(6):
                for nu in range(6):
                    dG_mu = np.gradient(G[..., a, nu], self.deltas[mu], axis=mu)
                    dG_nu = np.gradient(G[..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_strong"] * np.einsum('abc,...b,...c->...a', f_su3, G[..., mu], G[..., nu])[..., a]
                    F_G[..., a, mu, nu] = dG_mu - dG_nu + nonlinear
        return {"G": G, "F": F_G}

    def evolve_gauge_fields(self):
        """Evolve gauge fields."""
        for a in range(8):
            for mu in range(6):
                for nu in range(6):
                    dG_mu = np.gradient(self.strong_fields['G'][..., a, nu], self.deltas[mu], axis=mu)
                    dG_nu = np.gradient(self.strong_fields['G'][..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_strong"] * np.einsum('abc,...b,...c->...a', f_su3,
                                                              self.strong_fields['G'][..., mu],
                                                              self.strong_fields['G'][..., nu])[..., a]
                    self.strong_fields['F'][..., a, mu, nu] = dG_mu - dG_nu + nonlinear
        for a in range(3):
            for mu in range(6):
                for nu in range(6):
                    dW_mu = np.gradient(self.weak_fields['W'][..., a, nu], self.deltas[mu], axis=mu)
                    dW_nu = np.gradient(self.weak_fields['W'][..., a, mu], self.deltas[nu], axis=nu)
                    nonlinear = CONFIG["g_weak"] * np.einsum('abc,...b,...c->...a', f_su2,
                                                            self.weak_fields['W'][..., mu],
                                                            self.weak_fields['W'][..., nu])[..., a]
                    self.weak_fields['F'][..., a, mu, nu] = dW_mu - dW_nu + nonlinear
        if np.any(np.isnan(self.strong_fields['F'])) or np.any(np.isnan(self.weak_fields['F'])):
            logger.warning("NaN detected in gauge fields")

    def evolve_phi_wave_functions(self):
        """Evolve scalar field wave functions."""
        F_squared = np.einsum('...mn,...mn->...', self.em_fields["F"], self.em_fields["F"])
        phase_factor = np.exp(-1j * CONFIG["alpha_phi"] * (F_squared + self.em_fields["J4"])[..., np.newaxis] *
                              self.phi_range * self.dt / hbar)
        self.phi_wave_functions *= phase_factor
        kinetic_coeff = -hbar**2 / (2 * m_n * self.d_phi**2)
        second_deriv = (self.phi_wave_functions[..., 2:] - 2 * self.phi_wave_functions[..., 1:-1] +
                        self.phi_wave_functions[..., :-2]) / self.d_phi**2
        self.phi_wave_functions[..., 1:-1] += (-1j * kinetic_coeff * second_deriv * self.dt / hbar)
        self.phi_wave_functions[..., 0] = self.phi_wave_functions[..., 1]
        self.phi_wave_functions[..., -1] = self.phi_wave_functions[..., -2]
        norm = np.sqrt(np.sum(np.abs(self.phi_wave_functions)**2 * self.d_phi, axis=-1))[..., np.newaxis]
        if np.any(norm == 0):
            logger.warning("Zero norm detected in phi wave functions")
            norm = np.where(norm == 0, 1e-15, norm)
        self.phi_wave_functions /= norm
        self.phi_wave_functions = np.clip(self.phi_wave_functions, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])

    def update_phi_N_from_wave_functions(self):
        """Update phi_N from wave functions."""
        self.phi_N = np.sum(self.phi_range * np.abs(self.phi_wave_functions)**2 * self.d_phi, axis=-1)
        self.phi_N = np.clip(self.phi_N, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        if np.any(np.isnan(self.phi_N)):
            logger.warning("NaN detected in phi_N")

    def evolve_higgs_field(self) -> int:
        """Evolve the Higgs field."""
        def higgs_deriv(t, h_flat):
            h = h_flat.reshape(self.grid_size)
            d2_higgs = sum(np.gradient(np.gradient(h, self.deltas[i], axis=i), self.deltas[i], axis=i)
                           for i in range(6))
            h_norm = np.abs(h)**2
            dV_dH = -m_h * c**2 * h + lambda_higgs * h_norm * h
            return (-d2_higgs + dV_dH).flatten()

        h_flat = self.higgs_field.flatten()
        max_attempts = 3
        current_dt = self.dt
        for attempt in range(max_attempts):
            try:
                sol = solve_ivp(higgs_deriv, [0, current_dt], h_flat, method='RK45',
                                rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                if not sol.success:
                    raise ValueError("solve_ivp failed in evolve_higgs_field")
                break
            except Exception as e:
                logger.warning(f"evolve_higgs_field failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
                logger.info(f"Retrying with reduced dt={current_dt}")
        self.higgs_field = sol.y[:, -1].reshape(self.grid_size)
        self.higgs_field = np.nan_to_num(self.higgs_field, nan=v_higgs)
        self.higgs_field = np.clip(self.higgs_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        if np.any(np.isnan(self.higgs_field)):
            logger.warning("NaN detected in Higgs field")
        return len(sol.t) - 1

    def evolve_fermion_fields(self) -> int:
        """Evolve fermion fields."""
        total_steps = 0
        for idx in np.ndindex(self.grid_size):
            psi_e = self.electron_field[idx]
            H_e = self.dirac_hamiltonian(psi_e, idx, quark=False)
            max_attempts = 3
            current_dt = self.dt
            for attempt in range(max_attempts):
                try:
                    sol_e = solve_ivp(lambda t, y: -1j * H_e.dot(y) / hbar, [0, current_dt], psi_e,
                                      method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                    if not sol_e.success:
                        raise ValueError("solve_ivp failed in evolve_fermion_fields for electron")
                    break
                except Exception as e:
                    logger.warning(f"evolve_fermion_fields (electron) failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        raise
                    current_dt *= 0.5
                    logger.info(f"Retrying with reduced dt={current_dt}")
            self.electron_field[idx] = sol_e.y[:, -1]
            total_steps += len(sol_e.t) - 1

            for f in range(2):
                for c in range(3):
                    psi_q = self.quark_field[idx, f, c]
                    H_q = self.dirac_hamiltonian(psi_q, idx, quark=True, flavor=f, color=c)
                    for attempt in range(max_attempts):
                        try:
                            sol_q = solve_ivp(lambda t, y: -1j * H_q.dot(y) / hbar, [0, current_dt], psi_q,
                                              method='RK45', rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                            if not sol_q.success:
                                raise ValueError("solve_ivp failed in evolve_fermion_fields for quark")
                            break
                        except Exception as e:
                            logger.warning(f"evolve_fermion_fields (quark) failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                            if attempt == max_attempts - 1:
                                raise
                            current_dt *= 0.5
                            logger.info(f"Retrying with reduced dt={current_dt}")
                    self.quark_field[idx, f, c] = sol_q.y[:, -1]
                    total_steps += len(sol_q.t) - 1

        self.electron_field = np.nan_to_num(self.electron_field, nan=0.0)
        self.quark_field = np.nan_to_num(self.quark_field, nan=0.0)
        self.electron_field = np.clip(self.electron_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        self.quark_field = np.clip(self.quark_field, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        if np.any(np.isnan(self.electron_field)) or np.any(np.isnan(self.quark_field)):
            logger.warning("NaN detected in fermion fields")
        return total_steps

    def dirac_hamiltonian(self, psi: np.ndarray, idx: Tuple[int, ...], quark: bool = False,
                         flavor: Optional[int] = None, color: Optional[int] = None) -> np.ndarray:
        """Compute the Dirac Hamiltonian."""
        gamma_mu = construct_6d_gamma_matrices(self.metric[idx])
        mass = m_q if quark else m_e
        field = self.quark_field[..., flavor, color] if quark else self.electron_field
        D_mu_psi = [np.gradient(field, self.deltas[i], axis=i)[idx] if i < len(psi.shape) else psi for i in range(6)]
        H_psi = -1j * c * sum(gamma_mu[0] @ gamma_mu[i] @ D_mu_psi[i] for i in range(1, 6))
        H_psi += (mass * c**2 / hbar) * gamma_mu[0] @ psi
        H_psi -= 1j * e * sum(self.em_fields["A"][idx][mu] * gamma_mu[mu] @ psi for mu in range(6))
        if quark and flavor is not None and color is not None:
            T_a = lambda_matrices
            strong_term = sum(CONFIG["g_strong"] * self.strong_fields['G'][idx][a, mu] * T_a[a][color, color] * psi
                              for a in range(8) for mu in range(6))
            H_psi += -1j * strong_term
        if np.any(np.isnan(H_psi)):
            logger.warning("NaN detected in Dirac Hamiltonian")
        return np.nan_to_num(H_psi, nan=0.0)

    def compute_lambda(self, t: float, coords: np.ndarray, N: int = 3) -> np.ndarray:
        """Compute the lambda field."""
        frequencies = compute_schumann_frequencies(N)
        omega = [2 * np.pi * f for f in frequencies]
        lambda_field = np.zeros(self.grid_size, dtype=np.float64)
        x = coords[..., 1] / self.deltas[1]
        for n in range(N):
            A_n = 1e-21
            term = (-x**2 * np.cos(omega[n] * t) + 2 * x * np.sin(omega[n] * t) +
                    2 * np.cos(omega[n] * t))
            lambda_field += A_n * term
        if np.any(np.isnan(lambda_field)):
            logger.warning("NaN detected in lambda field")
        return np.nan_to_num(lambda_field, nan=0.0)

    def compute_rio_pattern(self, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Rio pattern."""
        P = np.abs(self.quantum_state)**2
        F = np.sqrt(np.sum([np.gradient(self.phi_N, self.deltas[mu], axis=mu)**2 for mu in range(6)], axis=0))
        phi_shifted = np.roll(self.phi_N, shift=[1, 1, 0, 0, 0, 0], axis=tuple(range(6)))
        M = np.cos(CONFIG["alpha_phi"] * P) * np.cos(CONFIG["alpha_phi"] * phi_shifted)
        return M * F, F

    def compute_quantum_flux(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the quantum flux."""
        psi = self.quantum_state
        psi_conj = np.conj(psi)
        J = np.zeros((*self.grid_size, 6), dtype=np.complex128)
        for mu in range(6):
            grad_psi = np.gradient(psi, self.deltas[mu], axis=mu)
            grad_psi_conj = np.gradient(psi_conj, self.deltas[mu], axis=mu)
            J[..., mu] = (hbar / (2 * m_n * 1j)) * (psi_conj * grad_psi - psi * grad_psi_conj)
        J_mag = np.sqrt(np.sum(np.abs(J)**2, axis=-1))
        if np.any(np.isnan(J_mag)):
            logger.warning("NaN detected in quantum flux magnitude")
        return J, J_mag

    def adjust_time_step(self, steps_taken: int):
        """Adjust the time step dynamically."""
        target_steps = 10
        if steps_taken > CONFIG["max_steps_per_dt"]:
            self.dt *= 0.5
            logger.info(f"Reducing dt to {self.dt} due to excessive steps ({steps_taken})")
        elif steps_taken > target_steps * 1.5:
            self.dt *= 0.9
            logger.debug(f"Reducing dt to {self.dt} (steps: {steps_taken})")
        elif steps_taken < target_steps * 0.5 and steps_taken > 0:
            self.dt *= 1.1
            logger.debug(f"Increasing dt to {self.dt} (steps: {steps_taken})")
        self.dt = max(CONFIG["dt_min"], min(self.dt, CONFIG["dt_max"]))
        logger.debug(f"Adjusted dt: {self.dt}")

    def quantum_walk(self, iteration: int):
        """Perform a quantum walk iteration."""
        import time
        t_start = self.time
        t_end = t_start + self.dt
        max_attempts = 3
        current_dt = self.dt
        total_steps = 0
        for attempt in range(max_attempts):
            try:
                self.time = t_start
                self.lambda_field = self.compute_lambda(self.time, self.lattice.coordinates)

                steps = self.spin_network.evolve(current_dt, self.lambda_field, self.metric, self.inverse_metric,
                                                self.deltas, self.phi_N, self.higgs_field, self.em_fields,
                                                self.electron_field, self.quark_field)
                total_steps += steps

                prob = np.abs(self.quantum_state)**2
                V_j4_phi = CONFIG["flux_coupling"] * self.em_fields["J4"] * self.phi_N
                def quantum_deriv(t, q_flat):
                    q = q_flat.reshape(self.grid_size)
                    kinetic = np.sum([np.gradient(np.gradient(q, self.deltas[mu], axis=mu), self.deltas[mu], axis=mu)
                                      for mu in range(6)], axis=0)
                    return (-hbar**2 / (2 * m_n) * kinetic + V_j4_phi * q).flatten()
                q_flat = self.quantum_state.flatten()
                sol = solve_ivp(quantum_deriv, [0, current_dt], q_flat, method='RK45',
                                rtol=CONFIG["rtol"], atol=CONFIG["atol"])
                if not sol.success:
                    raise ValueError("solve_ivp failed in quantum_walk")
                self.quantum_state = sol.y[:, -1].reshape(self.grid_size)
                norm = np.linalg.norm(self.quantum_state)
                if norm > 0:
                    self.quantum_state /= norm
                else:
                    logger.warning("Zero norm in quantum state evolution")
                self.quantum_state = np.clip(self.quantum_state, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
                total_steps += len(sol.t) - 1

                if len(self.fermion_history) >= CONFIG["time_delay_steps"]:
                    past_field = self.fermion_history[-CONFIG["time_delay_steps"]][1]
                    self.electron_field += CONFIG["ctc_feedback_factor"] * (past_field - self.electron_field) * current_dt

                self.temporal_entanglement = CONFIG["entanglement_factor"] * prob
                flip_mask = np.random.random(self.grid_size) < np.abs(self.em_fields["A"][..., 0] * CONFIG["em_strength"] *
                                                                      self.temporal_entanglement)
                self.bit_states[flip_mask] = 1 - self.bit_states[flip_mask]

                self.em_fields = self._initialize_em_fields()
                self.evolve_gauge_fields()
                self.evolve_phi_wave_functions()
                self.update_phi_N_from_wave_functions()

                steps = self.evolve_higgs_field()
                total_steps += steps

                steps = self.evolve_fermion_fields()
                total_steps += steps

                self.metric, self.inverse_metric = self.compute_quantum_metric()
                self.connection = self._compute_affine_connection()
                self.riemann_tensor = self._compute_riemann_tensor()
                self.ricci_tensor, self.ricci_scalar = self._compute_curvature()
                self.stress_energy = self._compute_stress_energy()
                self.einstein_tensor = self._compute_einstein_tensor()

                spin_density = np.zeros(self.grid_size, dtype=np.float64)
                for idx in np.ndindex(self.grid_size):
                    psi_e = self.electron_field[idx]
                    spin_e = np.einsum('i,ij,j->', psi_e.conj(), sigma[2], psi_e).real
                    spin_density[idx] = spin_e

                _, J_mag = self.compute_quantum_flux()

                timestamp = time.perf_counter_ns()
                self.history.append((timestamp, self.bit_states.copy()))
                self.fermion_history.append((timestamp, self.electron_field.copy()))
                self.phi_N_history.append(self.phi_N[0, 0, 0, 0, 0, 0])
                self.higgs_norm_history.append(np.mean(np.abs(self.higgs_field)))
                self.entanglement_history.append(compute_entanglement_entropy(self.electron_field, self.grid_size))
                rio_pattern, _ = self.compute_rio_pattern(iteration)
                self.ricci_scalar_history.append(self.ricci_scalar[0, 0, 0, 0, 0, 0].real)
                self.lambda_history.append(self.lambda_field[0, 0, 0, 0, 0, 0])
                self.spin_density_history.append(spin_density[0, :, :, 0, 0, 0])
                self.j4_history.append(self.em_fields["J4"][0, 0, 0, 0, 0, 0])
                self.flux_history.append(J_mag[0, :, :, 0, 0, 0])

                logger.info(f"Iteration {iteration}: Ricci Scalar = {self.ricci_scalar[0, 0, 0, 0, 0, 0].real:.6e}, "
                            f"Higgs Norm = {self.higgs_norm_history[-1]:.6e}, J4 = {self.j4_history[-1]:.6e}, "
                            f"Flux Mag = {np.mean(J_mag):.6e}")

                self.time = t_end
                break
            except Exception as e:
                logger.warning(f"quantum_walk failed with dt={current_dt}: {e}, attempt {attempt+1}/{max_attempts}")
                if attempt == max_attempts - 1:
                    raise
                current_dt *= 0.5
                logger.info(f"Retrying with reduced dt={current_dt}")
                self.dt = current_dt
        self.adjust_time_step(total_steps)

    def visualize(self, iteration: int):
        """Visualize simulation results."""
        fig = plt.figure(figsize=(18, 12))

        ax1 = fig.add_subplot(231, projection='3d')
        x, y, z = self.wormhole_nodes[0, :, :, :, 0, 0, :3].reshape(-1, 3).T
        sc = ax1.scatter(x, y, z, c=self.bit_states[0, :, :, :, 0, 0].flatten(), cmap='viridis')
        plt.colorbar(sc, label='Bit State')
        ax1.set_title('Spacetime Grid (t=0, v=u=0)')

        ax2 = fig.add_subplot(232)
        ax2.plot(self.phi_N_history, label='φ_N')
        ax2.set_title('Nugget Field Evolution')
        ax2.legend()

        ax3 = fig.add_subplot(233)
        ax3.plot(self.higgs_norm_history, label='Higgs Norm', color='orange')
        ax3.set_title('Higgs Field Norm')
        ax3.legend()

        ax4 = fig.add_subplot(234)
        ax4.plot(self.ricci_scalar_history, label='Ricci Scalar', color='red')
        ax4.set_title('Ricci Scalar Evolution')
        ax4.legend()

        ax5 = fig.add_subplot(235)
        ax5.plot(self.entanglement_history, label='Entanglement Entropy', color='green')
        ax5.set_title('Entanglement Entropy')
        ax5.legend()

        ax6 = fig.add_subplot(236)
        spin_slice = self.spin_density_history[-1]
        im = ax6.imshow(spin_slice, cmap='plasma')
        plt.colorbar(im, ax=ax6, label='Electron Spin Density')
        ax6.set_title('Spin Density (t=0, v=u=0)')

        plt.tight_layout()
        plt.savefig(f'toe_6d_iter_{iteration}.png')
        plt.close()

    def visualize_quantum_flux(self, iteration: int):
        """Visualize quantum flux."""
        flux_slice = self.flux_history[-1]
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(flux_slice, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Quantum Flux Magnitude')
        ax.set_title('Quantum Flux with Capacitor Resonance (t=0, z=0, v=0, u=0)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for i in range(0, self.grid_size[1], 2):
            for j in range(0, self.grid_size[2], 2):
                rect = plt.Rectangle((i-0.4, j-0.4), 0.8, 0.8, fill=False, edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
                if i < self.grid_size[1] - 1:
                    ax.plot([i+0.4, i+1.6], [j, j], color='gray', linestyle='-', linewidth=1)
                if j < self.grid_size[2] - 1:
                    ax.plot([i, i], [j+0.4, j+1.6], color='gray', linestyle='-', linewidth=1)

        plt.savefig(f'quantum_flux_iter_{iteration}.png')
        plt.close()
