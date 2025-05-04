# sphinx_os/utils/constants.py
"""
Physical constants and configuration for SphinxOS.
"""
import numpy as np

# Physical Constants
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
e = 1.60217662e-19
epsilon_0 = 8.854187817e-12
mu_0 = 1 / (epsilon_0 * c**2)
m_e = 9.1093837e-31
m_q = 2.3e-30
m_h = 2.23e-30
m_n = 1.67e-28
g_w = 0.653
g_s = 1.221
v_higgs = 246e9 * e / c**2
l_p = np.sqrt(hbar * G / c**3)
kappa = 1e-8
lambda_higgs = 0.5
alpha = 1 / 137.0
yukawa_e = 2.9e-6
yukawa_q = 1.2e-5
RS = 2.0 * G * m_n / c**2

# Pauli and Gell-Mann Matrices
sigma = [
    np.array([[0, 1], [1, 0]], dtype=np.complex64),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    np.array([[1, 0], [0, -1]], dtype=np.complex64)
]
lambda_matrices = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex64),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex64),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex64),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3, dtype=np.complex64)
]
f_su2 = np.zeros((3, 3, 3), dtype=np.float64)
for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: f_su2[a, b, c] = 1
for a, b, c in [(2, 1, 0), (0, 2, 1), (1, 0, 2)]: f_su2[a, b, c] = -1
f_su3 = np.zeros((8, 8, 8), dtype=np.float64)
f_su3[0, 1, 2] = 1; f_su3[0, 2, 1] = -1
f_su3[0, 3, 4] = 0.5; f_su3[0, 4, 3] = -0.5
f_su3[0, 5, 6] = 0.5; f_su3[0, 6, 5] = -0.5
f_su3[1, 3, 5] = 0.5; f_su3[1, 5, 3] = -0.5
f_su3[1, 4, 6] = -0.5; f_su3[1, 6, 4] = 0.5
f_su3[2, 3, 6] = 0.5; f_su3[2, 6, 3] = -0.5
f_su3[2, 4, 5] = 0.5; f_su3[2, 5, 4] = -0.5
f_su3[3, 4, 7] = np.sqrt(3)/2; f_su3[3, 7, 4] = -np.sqrt(3)/2
f_su3[5, 6, 7] = np.sqrt(3)/2; f_su3[5, 7, 6] = -np.sqrt(3)/2

# Configuration
CONFIG = {
    "grid_size": (5, 5, 5, 5, 3, 3),
    "max_iterations": 100,
    "time_delay_steps": 3,
    "ctc_feedback_factor": 5.0,
    "entanglement_factor": 0.2,
    "charge": e,
    "em_strength": 3.0,
    "dt": 1e-12,
    "dx": l_p * 1e5,
    "dv": l_p * 1e3,
    "du": l_p * 1e3,
    "alpha_em": alpha,
    "alpha_phi": 1e-3,
    "m_nugget": m_n,
    "m_higgs": m_h,
    "m_electron": m_e,
    "m_quark": m_q,
    "vev_higgs": v_higgs,
    "yukawa_e": yukawa_e,
    "yukawa_q": yukawa_q,
    "g_strong": g_s * 1e-5,
    "g_weak": g_w * 1e-5,
    "omega": 3,
    "a_godel": 1.0,
    "entanglement_coupling": 1e-6,
    "j4_scaling_factor": 1e-20,
    "sample_rate": 22050,
    "log_tensors": True,
    "flux_coupling": 1e-3,
    "resonance_frequency": 1e6,
    "resonance_amplitude": 0.1,
    "field_clamp_max": 1e6,
    "rtol": 1e-6,
    "atol": 1e-9,
    "dt_min": 1e-15,
    "dt_max": 1e-9,
    "max_steps_per_dt": 1000,
    "shots": 1024
}
