# sphinx_os/utils/constants.py
"""
Constants: Physical and simulation constants for SphinxOS.
"""
import numpy as np

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8  # Speed of light (m/s)
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
e = 1.602176634e-19  # Elementary charge (C)
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
m_e = 9.1093837015e-31  # Electron mass (kg)
m_q = 2.3e-30  # Quark mass (approximate, kg)
m_h = 2.23e-30  # Higgs mass (approximate, kg)
m_n = 1.67492749804e-27  # Neutron mass (kg)
v_higgs = 246e9 / c**2  # Higgs VEV (kg)
l_p = np.sqrt(hbar * G / c**3)  # Planck length (m)
kappa = 1e-8  # Coupling constant
lambda_higgs = 0.5  # Higgs self-coupling
RS = 2 * G * m_n / c**2  # Schwarzschild radius for neutron mass (m)

# Pauli matrices for spin calculations
sigma = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),  # sigma_x
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # sigma_y
    np.array([[1, 0], [0, -1]], dtype=np.complex128)  # sigma_z
]

# Gell-Mann matrices for SU(3) (strong force)
lambda_matrices = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128),
    np.array([[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(3), 0], [0, 0, -2/np.sqrt(3)]], dtype=np.complex128)
]

# Structure constants for SU(2) and SU(3)
f_su2 = np.zeros((3, 3, 3))
f_su2[0, 1, 2] = f_su2[1, 2, 0] = f_su2[2, 0, 1] = 1
f_su2[0, 2, 1] = f_su2[2, 1, 0] = f_su2[1, 0, 2] = -1

f_su3 = np.zeros((8, 8, 8))
f_su3[0, 1, 2] = 1
f_su3[0, 3, 6] = f_su3[0, 4, 5] = 0.5
f_su3[1, 3, 5] = -0.5
f_su3[1, 4, 6] = 0.5
f_su3[2, 3, 4] = 0.5 * np.sqrt(3)
f_su3[2, 5, 6] = -0.5 * np.sqrt(3)
f_su3[3, 4, 7] = 0.5 * np.sqrt(3)
f_su3[5, 6, 7] = 0.5 * np.sqrt(3)
# Antisymmetric
for i in range(8):
    for j in range(8):
        for k in range(8):
            f_su3[j, i, k] = -f_su3[i, j, k]

# Simulation configuration
CONFIG = {
    "grid_size": (2, 2, 2, 2, 2, 2),  # 6D grid size
    "num_qubits": 64,  # Number of qubits for quantum simulation (updated to 64 as requested)
    "qubit_count": 64,  # Same as num_qubits (for compatibility)
    "shots": 1024,  # Number of measurement shots
    "max_iterations": 10,  # Number of simulation iterations
    "dt": 1e-3,  # Base time step (s)
    "dt_min": 1e-5,  # Minimum time step (s)
    "dt_max": 1e-2,  # Maximum time step (s)
    "max_steps_per_dt": 1000,  # Maximum solver steps per dt
    "rtol": 1e-3,  # Relative tolerance for solvers
    "atol": 1e-6,  # Absolute tolerance for solvers
    "physics_refresh_interval": 0.1,  # Physics daemon refresh interval (s)
    "deltat": 1e-3,  # Time step size (s)
    "deltax": 1e-10,  # Spatial step size (m)
    "deltav": 1e-10,  # Compact dimension step size (m)
    "deltay": 1e-10,  # Spatial step size (m)
    "deltaz": 1e-10,  # Spatial step size (m)
    "deltau": 1e-10,  # Compact dimension step size (m)
    "dt": 1e-3,  # Time step (s)
    "dx": 1e-10,  # Spatial step (m)
    "dy": 1e-10,  # Spatial step (m)
    "dz": 1e-10,  # Spatial step (m)
    "dv": 1e-10,  # Compact dimension step (m)
    "du": 1e-10,  # Compact dimension step (m)
    "omega": 1e3,  # Frequency parameter (rad/s)
    "charge": 1e-19,  # Charge for EM fields (C)
    "resonance_frequency": 1e6,  # Resonance frequency (Hz)
    "resonance_amplitude": 0.1,  # Resonance amplitude
    "j4_scaling_factor": 1e-5,  # J4 coupling factor
    "g_weak": 0.653,  # Weak coupling constant
    "g_strong": 1.221e-5,  # Strong coupling constant
    "a_godel": 1e-9,  # Gödel rotation parameter (m)
    "alpha_phi": 1e-2,  # Phi field coupling
    "flux_coupling": 1e-5,  # Quantum flux coupling
    "entanglement_factor": 0.1,  # Entanglement scaling factor
    "ctc_feedback_factor": 5.0,  # Closed timelike curve feedback factor
    "time_delay_steps": 3,  # Steps for CTC delay
    "em_strength": 1e-3,  # EM field strength factor
    "field_clamp_max": 1e6,  # Maximum field value for clamping
    "log_tensors": False,  # Whether to log tensor data
    "m_nugget": 1e-30,  # Nugget field mass (kg)
    "lambda_nugget": 0.1,  # Nugget field coupling
    "m_higgs": 2.23e-30,  # Higgs mass (kg)
    "lambda_h": 0.5,  # Higgs coupling (alias for lambda_higgs)
    # Rydberg gate parameters
    "rydberg_blockade_radius": 1e-6,  # Blockade radius in meters
    "rydberg_interaction_strength": 1e6,  # Interaction strength in Hz
    "rydberg_coupling": 1e-3,  # Coupling factor for Rydberg effects
    "rydberg_decoherence_factor": 1.1  # Decoherence increase factor for Rydberg gates
}
