"""
J4 Scalar Quantum Gravity Framework

Implements scalar quantum gravity as an emergent phenomenon within the 27-dimensional
exceptional Jordan algebra J₃(𝕆), extending the Jones Quantum Gravity Resolution
framework with a propagating scalar field φ on the entanglement-induced curved spacetime.

The scalar field satisfies the generalized Klein-Gordon equation on the emergent metric:

    (-□_g + m² + ξ·R_K)φ = 0

where:
    □_g = g^{ij} ∇_i ∇_j  (d'Alembertian from entanglement metric)
    R_K             = scalar curvature derived from modular Hamiltonian K
    ξ               = non-minimal coupling constant (ξ=1/6 for conformal coupling)
    m               = scalar field mass (m=0 for massless graviton-like mode)

Key physical results:
    1. Scalar propagator G(x,y) from modular Hamiltonian spectrum
    2. Vacuum energy density E_vac = Tr[K] / (2·dim)
    3. Hawking-like temperature T_H = κ/(2π) from spectral gap κ
    4. Unitarity preservation via entanglement islands

References:
    - T. Jones, "Jones Quantum Gravity Resolution" (2026)
    - B. DeWitt, "Quantum Theory of Gravity. I" Phys. Rev. 160, 1113 (1967)
    - S. Fulling, "Aspects of Quantum Field Theory in Curved Spacetime" (1989)
    - D. Page, "Average entropy of a subsystem" PRL 71, 1291 (1993)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp, quad
from scipy.special import jv as bessel_j
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .jones_quantum_gravity import (
    JonesQuantumGravityResolution,
    ModularHamiltonian,
    DeterministicPageCurve,
    EntanglementMetric,
)

logger = logging.getLogger("SphinxOS.J4ScalarQG")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScalarFieldMode:
    """A normal mode of the scalar field on the emergent spacetime."""
    frequency: float           # ω_n  (eigenfrequency)
    wavenumber: float          # k_n  (effective wavenumber from K-spectrum)
    amplitude: complex         # a_n  (mode amplitude)
    occupation_number: float   # <n_n>  (Bose-Einstein occupation)


@dataclass
class ScalarPropagator:
    """Feynman propagator G_F(x,y) for the scalar field."""
    x_points: np.ndarray       # Source-point grid
    values: np.ndarray         # G_F values (complex)
    spectral_function: np.ndarray  # A(ω) = Im G_F / π


@dataclass
class J4Results:
    """Complete numerical results of the J4 Scalar QG computation."""
    # Spectral quantities
    spectral_gap_kappa: float
    hawking_temperature: float
    vacuum_energy_density: float
    scalar_curvature: float

    # Scalar field spectrum
    modes: List[ScalarFieldMode]
    zero_point_energy: float

    # Propagator
    propagator: ScalarPropagator

    # Page-curve / entropy
    max_entropy: float
    nuclearity_bound: float
    saturation_x: float

    # Unitarity checks
    unitarity_preserved: bool
    island_count: int

    # Graviton-scalar coupling
    coupling_xi: float
    conformal_coupling_xi: float  # = 1/6 in 3+1 D

    # Metadata
    dimension: int
    seed: int


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

class J4ScalarQuantumGravity:
    """
    J4 Scalar Quantum Gravity: scalar field on emergent spacetime from J₃(𝕆).

    The framework proceeds in three stages:
      1. Build emergent curved spacetime from the modular Hamiltonian K.
      2. Quantise a scalar field φ on that spacetime.
      3. Extract physical observables: Hawking temperature, vacuum energy,
         propagator, and Page curve saturation.
    """

    def __init__(
        self,
        dimension: int = 27,
        mass: float = 0.0,
        coupling_xi: float = 1.0 / 6.0,
        contraction_strength: float = 1.0,
        rotation_angle: float = np.pi / 6,
        freeze_threshold: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialise J4 Scalar Quantum Gravity framework.

        Args:
            dimension:           Operator-space dimension (27 for J₃(𝕆)).
            mass:                Scalar field mass m (m=0 → massless).
            coupling_xi:         Non-minimal coupling ξ (1/6 → conformal).
            contraction_strength: Strength of contraction operator C.
            rotation_angle:      CTC rotation angle for operator U.
            freeze_threshold:    Freeze threshold for operator F.
            seed:                Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.dimension = dimension
        self.mass = mass
        self.coupling_xi = coupling_xi
        self.seed = seed

        logger.info("=" * 60)
        logger.info("J4 Scalar Quantum Gravity Initialisation")
        logger.info("=" * 60)
        logger.info(f"  dimension   = {dimension}")
        logger.info(f"  mass        = {mass}")
        logger.info(f"  coupling ξ  = {coupling_xi:.6f}")

        # ── Base Jones QG framework ──────────────────────────────────────────
        self.jones = JonesQuantumGravityResolution(
            dimension=dimension,
            contraction_strength=contraction_strength,
            rotation_angle=rotation_angle,
            freeze_threshold=freeze_threshold,
        )
        self.K = self.jones.modular_hamiltonian.K        # 27×27 modular Hamiltonian
        self.K_eigenvalues = self.jones.modular_hamiltonian.spectrum.eigenvalues
        self.K_eigenvectors = self.jones.modular_hamiltonian.spectrum.eigenvectors

        # ── Derived geometric quantities ─────────────────────────────────────
        self.kappa = self.jones.modular_hamiltonian.get_spectral_gap()
        self.hawking_temperature = self.kappa / (2.0 * np.pi)
        self.vacuum_energy_density = np.trace(self.K) / (2.0 * dimension)
        self.scalar_curvature = self._compute_scalar_curvature()

        logger.info(f"  κ (spectral gap)       = {self.kappa:.8f}")
        logger.info(f"  T_H (Hawking temp.)    = {self.hawking_temperature:.8f}")
        logger.info(f"  E_vac                  = {self.vacuum_energy_density:.8f}")
        logger.info(f"  R_K (scalar curvature) = {self.scalar_curvature:.8f}")

        # ── Scalar field modes ────────────────────────────────────────────────
        self.modes = self._build_scalar_modes()
        self.zero_point_energy = sum(0.5 * m.frequency for m in self.modes)

        logger.info(f"  Normal modes           = {len(self.modes)}")
        logger.info(f"  Zero-point energy E₀   = {self.zero_point_energy:.8f}")
        logger.info("Initialisation complete")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_scalar_curvature(self) -> float:
        """
        Compute scalar curvature R_K from the modular Hamiltonian.

        R_K = Tr(K²) / dim - (Tr K / dim)²

        This measures the variance of the modular energy, analogous to the
        Ricci scalar in the entanglement metric.
        """
        mean_K = np.trace(self.K) / self.dimension
        mean_K2 = np.trace(self.K @ self.K) / self.dimension
        return float(mean_K2 - mean_K ** 2)

    def _bose_einstein(self, omega: float) -> float:
        """Bose-Einstein occupation at Hawking temperature T_H."""
        if self.hawking_temperature < 1e-12:
            return 0.0
        exp_arg = omega / self.hawking_temperature
        if exp_arg > 700:
            return 0.0
        return 1.0 / (np.exp(exp_arg) - 1.0 + 1e-30)

    def _build_scalar_modes(self) -> List[ScalarFieldMode]:
        """
        Build normal modes of the scalar field from the K-spectrum.

        The scalar field is expanded in eigenmodes of the operator:
            Ω² = K + m²·I + ξ·R_K·I

        Each eigenvalue ω_n² gives a mode frequency ω_n.
        """
        Omega2_eigenvalues = (
            self.K_eigenvalues
            + self.mass ** 2
            + self.coupling_xi * self.scalar_curvature
        )
        # Regularise: ensure non-negative
        Omega2_eigenvalues = np.maximum(Omega2_eigenvalues, 0.0)
        frequencies = np.sqrt(Omega2_eigenvalues)

        modes = []
        for n, (omega, k) in enumerate(
            zip(frequencies, self.K_eigenvalues)
        ):
            # Effective wavenumber k_n from K-eigenvalue
            k_n = np.sqrt(max(k, 0.0))
            # Amplitude: Gaussian random (vacuum fluctuation)
            amp = complex(np.random.randn(), np.random.randn()) / np.sqrt(
                2.0 * omega + 1e-12
            )
            occ = self._bose_einstein(omega)
            modes.append(
                ScalarFieldMode(
                    frequency=float(omega),
                    wavenumber=float(k_n),
                    amplitude=amp,
                    occupation_number=float(occ),
                )
            )
        return modes

    # ── Public computation methods ───────────────────────────────────────────

    def compute_propagator(
        self,
        n_points: int = 200,
        omega_max: float = None,
    ) -> ScalarPropagator:
        """
        Compute the scalar Feynman propagator G_F(ω) in frequency space.

        G_F(ω) = Σ_n |φ_n|² / (ω² - ω_n² + iε)

        Returns the retarded Green's function on the modular-Hamiltonian spectrum.

        Args:
            n_points:  Number of frequency points.
            omega_max: Maximum frequency (default: 1.5 × max mode frequency).

        Returns:
            ScalarPropagator with spectral function A(ω).
        """
        freqs = np.array([m.frequency for m in self.modes])
        if omega_max is None:
            omega_max = 1.5 * freqs.max()

        omega_grid = np.linspace(0.0, omega_max, n_points)
        epsilon = 1e-3  # small imaginary regulator

        G = np.zeros(n_points, dtype=complex)
        for n, mode in enumerate(self.modes):
            omega_n2 = mode.frequency ** 2
            weight = abs(mode.amplitude) ** 2
            G += weight / (omega_grid ** 2 - omega_n2 + 1j * epsilon)

        spectral_function = -np.imag(G) / np.pi

        logger.info(
            f"Propagator computed: {n_points} points, "
            f"ω ∈ [0, {omega_max:.4f}]"
        )
        return ScalarPropagator(
            x_points=omega_grid,
            values=G,
            spectral_function=spectral_function,
        )

    def compute_stress_energy(self) -> Dict[str, float]:
        """
        Compute the stress-energy tensor components T_{μν} of the scalar field.

        In the modular framework, the stress-energy is projected onto
        the operator-space trace:

            T_00  = Σ_n ω_n (n_n + 1/2)   (energy density)
            T_ii  = Σ_n k_n²/(3 ω_n) (n_n + 1/2)   (pressure, isotropic)
            w     = T_ii / T_00   (equation of state)

        Returns:
            Dictionary with energy density, pressure, and equation-of-state w.
        """
        T00 = 0.0
        Tii = 0.0
        for mode in self.modes:
            occupancy = mode.occupation_number + 0.5  # includes ZPE
            T00 += mode.frequency * occupancy
            if mode.frequency > 1e-12:
                Tii += (mode.wavenumber ** 2 / (3.0 * mode.frequency)) * occupancy

        T00 /= self.dimension
        Tii /= self.dimension
        w = Tii / (T00 + 1e-30)

        result = {
            "energy_density_T00": T00,
            "pressure_Tii": Tii,
            "equation_of_state_w": w,
        }
        logger.info(
            f"Stress-energy: T00={T00:.6f}, Tii={Tii:.6f}, w={w:.6f}"
        )
        return result

    def compute_hawking_radiation_spectrum(
        self, n_points: int = 300
    ) -> Dict[str, np.ndarray]:
        """
        Compute the Hawking radiation spectrum of the scalar field.

        dN/dω = (1/(2π)) · [Bose-Einstein factor at T_H] / ω

        The scalar field excitations near the spectral gap κ correspond
        to Hawking quanta emitted at temperature T_H = κ/(2π).

        Args:
            n_points: Number of frequency points.

        Returns:
            Dictionary with frequency grid and spectral number density.
        """
        omega_min = self.kappa * 0.01
        omega_max = self.kappa * 20.0
        omega = np.linspace(omega_min, omega_max, n_points)

        # Planck-like spectrum
        if self.hawking_temperature > 1e-12:
            exp_arg = omega / self.hawking_temperature
            # Clamp to avoid overflow
            exp_arg = np.minimum(exp_arg, 700.0)
            dN_domega = 1.0 / (2.0 * np.pi * omega * (np.exp(exp_arg) - 1.0 + 1e-30))
        else:
            dN_domega = np.zeros_like(omega)

        # Peak frequency (Wien's displacement: ω_peak ≈ 2.82 T_H)
        wien_peak = 2.821 * self.hawking_temperature

        result = {
            "omega": omega,
            "dN_domega": dN_domega,
            "T_H": self.hawking_temperature,
            "kappa": self.kappa,
            "wien_peak": wien_peak,
        }
        logger.info(
            f"Hawking spectrum: T_H={self.hawking_temperature:.6f}, "
            f"Wien peak ω*={wien_peak:.6f}"
        )
        return result

    def compute_entanglement_entropy_flow(
        self, n_points: int = 200
    ) -> Dict[str, np.ndarray]:
        """
        Compute entanglement entropy flow S(x) for the scalar field.

        S(x) = S_vN(ρ_scalar(x)) = Σ_n [-(λ_n log λ_n)]

        where λ_n are occupation fractions of mode n at position x.

        Args:
            n_points: Number of position points.

        Returns:
            Dictionary with position grid and entropy S(x).
        """
        x_vals, S_base = self.jones.page_curve.compute_page_curve(n_points)

        # Scalar field adds a correction δS = Σ_n log(1 + <n_n>)  (bosonic entropy)
        delta_S_scalar = sum(
            np.log(1.0 + mode.occupation_number + 1e-30)
            for mode in self.modes
        ) / self.dimension

        S_total = S_base + delta_S_scalar

        result = {
            "x": x_vals,
            "S_base": S_base,
            "S_scalar_correction": np.full_like(S_base, delta_S_scalar),
            "S_total": S_total,
            "delta_S_scalar": delta_S_scalar,
        }
        logger.info(
            f"Entropy flow: max S_total={S_total.max():.6f}, "
            f"scalar correction δS={delta_S_scalar:.6f}"
        )
        return result

    def run_full_analysis(self) -> J4Results:
        """
        Run the complete J4 Scalar Quantum Gravity analysis.

        Returns:
            J4Results data container with all numerical observables.
        """
        logger.info("\n" + "=" * 60)
        logger.info("J4 SCALAR QUANTUM GRAVITY — FULL ANALYSIS")
        logger.info("=" * 60)

        # Page curve
        page_data = self.jones.compute_page_curve(n_points=200)

        # Islands
        islands = self.jones.find_entanglement_islands(tolerance=0.5)

        # Propagator
        propagator = self.compute_propagator()

        # Unitarity: satisfied if modular nuclearity bound is compatible
        unitarity = page_data["verification"]["satisfies_bound"] or len(islands) > 0

        results = J4Results(
            spectral_gap_kappa=self.kappa,
            hawking_temperature=self.hawking_temperature,
            vacuum_energy_density=self.vacuum_energy_density,
            scalar_curvature=self.scalar_curvature,
            modes=self.modes,
            zero_point_energy=self.zero_point_energy,
            propagator=propagator,
            max_entropy=page_data["max_entropy"],
            nuclearity_bound=page_data["verification"]["nuclearity_bound"],
            saturation_x=page_data["saturation_point"],
            unitarity_preserved=unitarity,
            island_count=len(islands),
            coupling_xi=self.coupling_xi,
            conformal_coupling_xi=1.0 / 6.0,
            dimension=self.dimension,
            seed=self.seed,
        )

        logger.info("\nKey Results:")
        logger.info(f"  κ               = {results.spectral_gap_kappa:.8f}")
        logger.info(f"  T_H             = {results.hawking_temperature:.8f}")
        logger.info(f"  E_vac           = {results.vacuum_energy_density:.8f}")
        logger.info(f"  R_K             = {results.scalar_curvature:.8f}")
        logger.info(f"  E₀ (ZPE)        = {results.zero_point_energy:.8f}")
        logger.info(f"  S_max           = {results.max_entropy:.8f}")
        logger.info(f"  Unitarity       = {results.unitarity_preserved}")
        logger.info(f"  Islands         = {results.island_count}")

        return results

    # ── Visualisation ────────────────────────────────────────────────────────

    def generate_whitepaper_figures(
        self, output_dir: str = "whitepaper/images"
    ) -> Dict[str, str]:
        """
        Generate all figures for the J4 Scalar QG whitepaper.

        Figures:
          1. j4_scalar_spectrum.png        — Scalar field mode spectrum
          2. j4_hawking_spectrum.png       — Hawking radiation dN/dω
          3. j4_propagator_spectral.png    — Spectral function A(ω)
          4. j4_entropy_flow.png           — Entanglement entropy S(x)
          5. j4_modular_eigenvalues.png    — Modular Hamiltonian eigenvalues

        Args:
            output_dir: Directory for saving figures.

        Returns:
            Dictionary mapping figure keys to file paths.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        plots: Dict[str, str] = {}

        freqs = np.array([m.frequency for m in self.modes])
        occs = np.array([m.occupation_number for m in self.modes])

        # ── 1. Scalar field mode spectrum ────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].stem(
            range(self.dimension), freqs, linefmt="C0-", markerfmt="C0o",
            basefmt="k-"
        )
        axes[0].set_xlabel("Mode index $n$", fontsize=13)
        axes[0].set_ylabel(r"$\omega_n$  (frequency)", fontsize=13)
        axes[0].set_title("Scalar Field Normal-Mode Frequencies", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        axes[1].stem(
            range(self.dimension), occs, linefmt="C1-", markerfmt="C1s",
            basefmt="k-"
        )
        axes[1].set_xlabel("Mode index $n$", fontsize=13)
        axes[1].set_ylabel(r"$\langle n_n \rangle$  (occupation)", fontsize=13)
        axes[1].set_title(
            f"Bose–Einstein Occupation at $T_H = {self.hawking_temperature:.4f}$",
            fontsize=14,
        )
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            "J4 Scalar QG: Normal Modes on Emergent Spacetime", fontsize=15, y=1.01
        )
        plt.tight_layout()
        path = f"{output_dir}/j4_scalar_spectrum.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["scalar_spectrum"] = path
        logger.info(f"Saved: {path}")

        # ── 2. Hawking radiation spectrum ────────────────────────────────────
        hspec = self.compute_hawking_radiation_spectrum(n_points=400)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hspec["omega"] / self.kappa,
            hspec["dN_domega"] * self.kappa,
            "C2-", linewidth=2, label=r"$\kappa^{-1}\,dN/d\omega$",
        )
        ax.axvline(
            x=hspec["wien_peak"] / self.kappa,
            color="C3", linestyle="--", linewidth=1.5,
            label=f"Wien peak $\\omega^*/\\kappa = {hspec['wien_peak']/self.kappa:.3f}$",
        )
        ax.set_xlabel(r"$\omega / \kappa$", fontsize=13)
        ax.set_ylabel(r"$\kappa^{-1}\,dN/d\omega$", fontsize=13)
        ax.set_title(
            f"Hawking Radiation Spectrum  ($T_H = \\kappa / 2\\pi = "
            f"{self.hawking_temperature:.5f}$)",
            fontsize=14,
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/j4_hawking_spectrum.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["hawking_spectrum"] = path
        logger.info(f"Saved: {path}")

        # ── 3. Propagator spectral function ──────────────────────────────────
        prop = self.compute_propagator(n_points=400)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(prop.x_points, prop.spectral_function, "C4-", linewidth=1.5)
        axes[0].set_xlabel(r"$\omega$", fontsize=13)
        axes[0].set_ylabel(r"$A(\omega) = -\mathrm{Im}\,G_F / \pi$", fontsize=13)
        axes[0].set_title("Scalar Propagator Spectral Function", fontsize=14)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(prop.x_points, np.real(prop.values), "C5-", linewidth=1.5,
                     label=r"Re $G_F$")
        axes[1].plot(prop.x_points, np.imag(prop.values), "C6--", linewidth=1.5,
                     label=r"Im $G_F$")
        axes[1].set_xlabel(r"$\omega$", fontsize=13)
        axes[1].set_ylabel(r"$G_F(\omega)$", fontsize=13)
        axes[1].set_title("Real and Imaginary Parts of $G_F$", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/j4_propagator_spectral.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["propagator_spectral"] = path
        logger.info(f"Saved: {path}")

        # ── 4. Entanglement entropy flow ──────────────────────────────────────
        ent = self.compute_entanglement_entropy_flow(n_points=300)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ent["x"], ent["S_base"], "C0-", linewidth=2, label="$S_{\\rm base}(x)$ (Jones QG)")
        ax.plot(ent["x"], ent["S_total"], "C1--", linewidth=2,
                label=f"$S_{{\\rm total}}(x)$ (+scalar $\\delta S={ent['delta_S_scalar']:.4f}$)")
        ax.axhline(
            y=self.jones.page_curve.nuclearity_bound,
            color="C3", linestyle=":", linewidth=2,
            label=f"Nuclearity bound $\\ln(\\dim\\mathcal{{H}}_R)={self.jones.page_curve.nuclearity_bound:.4f}$",
        )
        ax.set_xlabel(r"$x$  (position in operator space)", fontsize=13)
        ax.set_ylabel(r"$S(x)$  (entanglement entropy)", fontsize=13)
        ax.set_title("Entanglement Entropy Flow with Scalar Field Correction", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/j4_entropy_flow.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["entropy_flow"] = path
        logger.info(f"Saved: {path}")

        # ── 5. Modular Hamiltonian eigenvalue distribution ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        eigs = np.sort(self.K_eigenvalues)
        axes[0].stem(
            range(len(eigs)), eigs, linefmt="C7-", markerfmt="C7D",
            basefmt="k-"
        )
        axes[0].axhline(y=self.kappa, color="C3", linestyle="--", linewidth=1.5,
                        label=f"$\\kappa = {self.kappa:.4f}$")
        axes[0].set_xlabel("Index", fontsize=13)
        axes[0].set_ylabel(r"Eigenvalue $\lambda_n$", fontsize=13)
        axes[0].set_title("Modular Hamiltonian Eigenvalue Spectrum", fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(eigs, bins=15, color="C8", edgecolor="black", alpha=0.75)
        axes[1].axvline(x=self.kappa, color="C3", linestyle="--", linewidth=1.5,
                        label=f"$\\kappa = {self.kappa:.4f}$ (spectral gap)")
        axes[1].set_xlabel(r"Eigenvalue $\lambda$", fontsize=13)
        axes[1].set_ylabel("Count", fontsize=13)
        axes[1].set_title("Eigenvalue Distribution of $K$", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/j4_modular_eigenvalues.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["modular_eigenvalues"] = path
        logger.info(f"Saved: {path}")

        logger.info(f"Generated {len(plots)} whitepaper figures in {output_dir}")
        return plots
