"""
Weyl Semimetal: Paired Weyl Nodes, Berry Phase Flux Monopoles and Antipoles
============================================================================

TaAs-like 2-band lattice model with:
  - Paired Weyl nodes at k_W = ±arccos(M/t_z) on the Γ–Z line
  - Chirality χ = +1 (monopole / source) and χ = −1 (antipole / sink)
  - Berry curvature Ω(k) as an effective magnetic field in momentum space
  - Chern number C = ±1 per node via Fukui–Hatsugai–Suzuki (FHS) lattice method
  - Berry phase γ = π for loops encircling a single Weyl node

Minimal TaAs-like Hamiltonian on a 3-D lattice
-----------------------------------------------
    H(k) = d_x(k) σ_x + d_y(k) σ_y + d_z(k) σ_z

    d_x(k) = t_x sin(k_x)
    d_y(k) = t_y sin(k_y)
    d_z(k) = M - t_z cos(k_z) - 2t[2 - cos(k_x) - cos(k_y)]

Weyl nodes (d_x = d_y = d_z = 0) on Γ–Z axis (k_x = k_y = 0):
    k_z^(±) = ±arccos(M / t_z)

Berry curvature for the lower band — solid-angle formula:
    Ω_α(k) = −(1/2) d̂ · (∂_β d̂ × ∂_γ d̂)   (α,β,γ cyclic)

Chern number (FHS):
    C(k_z) = (1/2π) Im ln ∏_{i,j} F(i,j;k_z)
    F(i,j) = U_x(i,j) U_y(i+1,j) U_x*(i,j+1) U_y*(i,j)
    U_μ(k) = ⟨u(k)|u(k+δk_μ)⟩ / |⟨u(k)|u(k+δk_μ)⟩|

Monopole flux:
    Φ = ∮_{sphere} Ω · dS = 2π χ

References
----------
Wan et al., Phys. Rev. B 83, 205101 (2011)   — Weyl semimetal theory
Berry, Proc. R. Soc. Lond. A 392, 45 (1984) — Berry phase
Fukui, Hatsugai & Suzuki, JPSJ 74, 1674 (2005) — lattice Chern number
Xu et al., Science 349, 613 (2015)           — TaAs Fermi arcs (experiment)
Lv et al., Phys. Rev. X 5, 031013 (2015)    — TaAs Weyl nodes (experiment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.optimize import brentq

logger = logging.getLogger("SphinxOS.WeylNodes")

# Pauli matrices
_σx = np.array([[0, 1], [1, 0]], dtype=complex)
_σy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_σz = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WeylNode:
    """A single Weyl node in momentum space."""
    position: np.ndarray   # 3-vector (k_x, k_y, k_z) in the BZ [−π, π]³
    chirality: int         # +1 (monopole / source) or −1 (antipole / sink)
    label: str             # human-readable name, e.g. "W1+" or "W1−"
    chern_number: int = 0  # computed Chern number (should equal chirality)
    berry_flux: float = 0.0  # ∮ Ω · dS around this node (should equal 2π χ)

    def __str__(self) -> str:
        chi_str = f"+{self.chirality}" if self.chirality > 0 else str(self.chirality)
        return (
            f"WeylNode({self.label}  k={np.round(self.position, 4)}  "
            f"χ={chi_str}  C={self.chern_number}  Φ={self.berry_flux:.4f})"
        )


@dataclass
class BerryPhaseResult:
    """Berry phase γ computed around a closed loop in k-space."""
    loop_radius: float        # radius of the loop in the k_x–k_y plane
    loop_center_kz: float     # k_z position of the loop
    gamma: float              # Berry phase in [0, π]  (or [−π, π])
    enclosed_chirality: int   # total chirality of nodes enclosed
    n_points: int             # number of k-points on the loop


@dataclass
class ChernProfile:
    """Chern number C(k_z) as a function of k_z."""
    kz_values: np.ndarray     # k_z grid
    chern_numbers: np.ndarray # C(k_z), integer-valued
    node_kz: np.ndarray       # k_z positions of Weyl nodes
    node_chiralities: np.ndarray  # chiralities of the nodes


# ---------------------------------------------------------------------------
# TaAs-like 2-band Hamiltonian
# ---------------------------------------------------------------------------

class TaAsLikeHamiltonian:
    """
    Qi–Hughes–Zhang (QHZ) 2-band lattice Hamiltonian for a TaAs-like Weyl semimetal.

    H(k) = d_x σ_x + d_y σ_y + d_z σ_z

    d_x = A sin k_x
    d_y = A sin k_y
    d_z = M + B(cos k_x + cos k_y + cos k_z)

    Weyl nodes on Γ–Z axis (k_x = k_y = 0):
        cos(k_z^node) = −(M + 2B) / B   →   k_z = ±arccos(−(M+2B)/B)

    Defaults: A=1, B=1, M=−2  →  nodes at k_z = ±π/2 ≈ ±1.5708 rad

    The kx–ky slice BETWEEN the nodes has d_z changing sign across the torus,
    giving a non-trivial Chern number C = ±1 (proper Weyl semimetal topology).

    Chirality convention (physical / Fermi-arc):
        χ = −sign(det J)  where J_{ij} = ∂d_i/∂k_j
    This ensures χ = +1 for the monopole (outward Berry flux Φ = +2π) and
    χ = −1 for the antipole (inward Berry flux Φ = −2π).
    """

    def __init__(
        self,
        M: float = -2.0,
        A: float = 1.0,
        B: float = 1.0,
    ):
        """
        Args:
            M: Mass parameter.  Nodes exist for −3B < M < −B.
            A: Fermi-velocity prefactor (off-diagonal hoppings).
            B: Diagonal hopping (controls gap and node position).
        """
        self.M, self.A, self.B = M, A, B
        # Also expose legacy aliases so existing callers still work
        self.t_x = A
        self.t_y = A
        self.t_z = B

        arg = -(M + 2 * B) / B
        if abs(arg) >= 1.0:
            raise ValueError(
                f"No Weyl nodes: |-(M+2B)/B| = {abs(arg):.3f} ≥ 1. "
                "Use −3B < M < −B."
            )
        self._kz_node = float(np.arccos(arg))
        logger.info(
            "TaAsLikeHamiltonian (QHZ): Weyl nodes at k_z = ±%.4f rad  (%.2f°)",
            self._kz_node, np.degrees(self._kz_node),
        )

    # ── d-vector ────────────────────────────────────────────────────────────

    def d_vector(self, k: np.ndarray) -> np.ndarray:
        """
        Return the 3-component d-vector at momentum k = (k_x, k_y, k_z).

        d_x = A sin k_x
        d_y = A sin k_y
        d_z = M + B(cos k_x + cos k_y + cos k_z)

        Args:
            k: Array of shape (3,) or (N, 3).

        Returns:
            d: shape (3,) or (N, 3).
        """
        k = np.asarray(k, dtype=float)
        scalar = k.ndim == 1
        if scalar:
            k = k[None, :]
        kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
        dx = self.A * np.sin(kx)
        dy = self.A * np.sin(ky)
        dz = self.M + self.B * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        d = np.stack([dx, dy, dz], axis=-1)
        return d[0] if scalar else d

    # ── Hamiltonian and energies ─────────────────────────────────────────────

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        Return the 2×2 Hamiltonian matrix at k.

        Args:
            k: shape (3,)

        Returns:
            H: shape (2, 2) complex
        """
        dx, dy, dz = self.d_vector(k)
        return dx * _σx + dy * _σy + dz * _σz

    def energies(self, k: np.ndarray) -> Tuple[float, float]:
        """
        Return (E_minus, E_plus) band energies at k.

        E_± = ± |d(k)|
        """
        d = self.d_vector(k)
        mag = float(np.linalg.norm(d))
        return -mag, +mag

    def lower_band_state(self, k: np.ndarray) -> np.ndarray:
        """
        Return the lower-band eigenstate |u_−(k)⟩ at k.

        Args:
            k: shape (3,)

        Returns:
            u: normalised 2-spinor
        """
        H = self.hamiltonian(k)
        _, vecs = np.linalg.eigh(H)
        return vecs[:, 0]   # lower eigenvalue first (eigh returns ascending)

    # ── Weyl node finder ────────────────────────────────────────────────────

    def find_weyl_nodes(self) -> List[WeylNode]:
        """
        Locate Weyl nodes analytically (on Γ–Z axis) and assign chirality.

        Nodes at k_x = k_y = 0, k_z = ±arccos(−(M+2B)/B).
        Labels: χ = +1 node → "W1+" (monopole),  χ = −1 → "W1−" (antipole).

        Returns:
            List of two WeylNode objects with opposite chirality.
        """
        kz_plus  = +self._kz_node
        kz_minus = -self._kz_node

        nodes = []
        for kz, raw_label in [(kz_plus, "W+"), (kz_minus, "W−")]:
            k0  = np.array([0.0, 0.0, kz])
            chi = self._chirality_at_node(k0)
            label = "W1+" if chi > 0 else "W1−"
            nodes.append(WeylNode(position=k0, chirality=chi, label=label))
            logger.info("Found Weyl node %-4s at kz=%.4f  χ=%+d", label, kz, chi)

        return nodes

    def _chirality_at_node(self, k0: np.ndarray, eps: float = 1e-3) -> int:
        """
        Compute chirality χ = sign(det J) of the d-vector map at a Weyl node.

        The Jacobian J_{ij} = ∂d_i/∂k_j evaluated at k0.

        Args:
            k0: Weyl node position.
            eps: Finite difference step.

        Returns:
            +1 or −1
        """
        J = np.zeros((3, 3), dtype=float)
        for j in range(3):
            kp = k0.copy(); kp[j] += eps
            km = k0.copy(); km[j] -= eps
            J[:, j] = (self.d_vector(kp) - self.d_vector(km)) / (2 * eps)
        det = float(np.linalg.det(J))
        # Physical convention: χ = −sign(det J)
        # Ensures χ = +1 ↔ monopole (outward flux Φ = +2π),
        #         χ = −1 ↔ antipole  (inward  flux Φ = −2π)
        return -int(np.sign(det)) if abs(det) > 1e-10 else 0


# ---------------------------------------------------------------------------
# Berry curvature
# ---------------------------------------------------------------------------

class BerryCurvatureField:
    """
    Berry curvature Ω(k) for the lower band of a 2-band Hamiltonian.

    Ω_z(k) = −(1/2) d̂ · (∂_{k_x} d̂ × ∂_{k_y} d̂)   (solid-angle formula)
    """

    def __init__(self, hamiltonian: TaAsLikeHamiltonian):
        self.H = hamiltonian

    # ── Solid-angle formula ─────────────────────────────────────────────────

    def _d_hat_and_derivatives(
        self, k: np.ndarray, eps: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (d̂, ∂_{k_x} d̂, ∂_{k_y} d̂) at k using central differences.
        """
        d = self.H.d_vector(k)
        mag = np.linalg.norm(d)
        d_hat = d / (mag + 1e-30)

        def d_hat_at(kk):
            dd = self.H.d_vector(kk)
            return dd / (np.linalg.norm(dd) + 1e-30)

        kpx = k + np.array([eps, 0, 0])
        kmx = k - np.array([eps, 0, 0])
        d_dx = (d_hat_at(kpx) - d_hat_at(kmx)) / (2 * eps)

        kpy = k + np.array([0, eps, 0])
        kmy = k - np.array([0, eps, 0])
        d_dy = (d_hat_at(kpy) - d_hat_at(kmy)) / (2 * eps)

        return d_hat, d_dx, d_dy

    def omega_z(self, k: np.ndarray, eps: float = 1e-4) -> float:
        """
        Compute Ω_z(k) = −(1/2) d̂ · (∂_x d̂ × ∂_y d̂).

        Args:
            k: momentum vector (3,)

        Returns:
            Ω_z (float)
        """
        d_hat, d_dx, d_dy = self._d_hat_and_derivatives(k, eps)
        cross = np.cross(d_dx, d_dy)
        return float(-0.5 * np.dot(d_hat, cross))

    def omega_vector(self, k: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Return all three components (Ω_x, Ω_y, Ω_z) at k.

        Uses cyclic solid-angle formula for each component.
        """
        def dhat_deriv(k_, axis, ep):
            kp = k_.copy(); kp[axis] += ep
            km = k_.copy(); km[axis] -= ep
            def dh(kk):
                dd = self.H.d_vector(kk)
                return dd / (np.linalg.norm(dd) + 1e-30)
            return (dh(kp) - dh(km)) / (2 * ep)

        d_hat = self.H.d_vector(k)
        d_hat = d_hat / (np.linalg.norm(d_hat) + 1e-30)

        ddk = [dhat_deriv(k, ax, eps) for ax in range(3)]

        omega = np.array([
            -0.5 * np.dot(d_hat, np.cross(ddk[1], ddk[2])),
            -0.5 * np.dot(d_hat, np.cross(ddk[2], ddk[0])),
            -0.5 * np.dot(d_hat, np.cross(ddk[0], ddk[1])),
        ])
        return omega

    # ── 2-D plane map ───────────────────────────────────────────────────────

    def compute_kxky_plane(
        self,
        kz: float,
        n_points: int = 60,
        k_range: float = np.pi,
    ) -> Dict:
        """
        Compute Ω_z on the k_x–k_y plane at fixed k_z.

        Args:
            kz:       Fixed k_z value (radians).
            n_points: Grid resolution in each direction.
            k_range:  Half-width of the BZ window.

        Returns:
            Dict with keys: kx, ky, omega_z (all 2-D arrays).
        """
        kx_vals = np.linspace(-k_range, k_range, n_points)
        ky_vals = np.linspace(-k_range, k_range, n_points)
        KX, KY = np.meshgrid(kx_vals, ky_vals)
        OZ = np.zeros_like(KX)

        for i in range(n_points):
            for j in range(n_points):
                k = np.array([KX[i, j], KY[i, j], kz])
                OZ[i, j] = self.omega_z(k)

        return {"kx": KX, "ky": KY, "omega_z": OZ, "kz": kz}

    # ── Monopole flux (sphere integral) ────────────────────────────────────

    def monopole_flux(
        self,
        node: WeylNode,
        radius: float = 0.3,
        n_theta: int = 16,
        n_phi: int = 32,
    ) -> float:
        """
        Compute Berry flux through a sphere of *radius* centred on *node*.

        Φ = ∮ Ω · dS ≈ Σ_{θ,φ} Ω(k) · k̂ sin(θ) Δθ Δφ · r²

        For a Weyl node with chirality χ:  Φ = 2π χ

        Args:
            node:    WeylNode whose flux we measure.
            radius:  Sphere radius in k-space (must not enclose other nodes).
            n_theta, n_phi: Angular resolution of the sphere.

        Returns:
            Φ (float, expected ≈ ±2π).
        """
        k0 = node.position
        theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
        phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        dtheta = theta_vals[1] - theta_vals[0]
        dphi = phi_vals[1] - phi_vals[0]

        flux = 0.0
        for theta in theta_vals:
            for phi in phi_vals:
                # Point on sphere
                k_hat = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                k = k0 + radius * k_hat
                omega = self.omega_vector(k)
                # dS element = r² sin(θ) dθ dφ  in direction k̂
                dS = radius ** 2 * np.sin(theta) * dtheta * dphi
                flux += np.dot(omega, k_hat) * dS

        logger.info(
            "Monopole flux for %s: Φ = %.4f  (expected %+.4f)",
            node.label, flux, 2 * np.pi * node.chirality,
        )
        return float(flux)


# ---------------------------------------------------------------------------
# Chern number (FHS lattice method)
# ---------------------------------------------------------------------------

class ChernNumberCalculator:
    """
    Fukui–Hatsugai–Suzuki (2005) lattice method for the Chern number.

    C(k_z) = (1/2π) Im ln ∏_{i,j} F_{ij}

    F_{ij} = U_x(i,j) · U_y(i+1,j) · U_x*(i,j+1) · U_y*(i,j)
    U_μ(k) = ⟨u(k)|u(k+δk_μ)⟩ / |⟨u(k)|u(k+δk_μ)⟩|

    The result is exactly integer-valued on the lattice.
    """

    def __init__(self, hamiltonian: TaAsLikeHamiltonian):
        self.H = hamiltonian

    def _link_variable(
        self,
        states: np.ndarray,
        i: int,
        j: int,
        di: int,
        dj: int,
        N: int,
    ) -> complex:
        """
        Compute link variable U between state at (i,j) and (i+di, j+dj) (periodic).
        """
        u1 = states[i % N, j % N]
        u2 = states[(i + di) % N, (j + dj) % N]
        inner = complex(np.dot(u1.conj(), u2))
        mag = abs(inner)
        return (inner / mag) if mag > 1e-12 else 1.0 + 0j

    def chern_number_at_kz(
        self,
        kz: float,
        n_points: int = 40,
    ) -> int:
        """
        Compute the Chern number of the lower band on the k_x–k_y torus at k_z.

        Args:
            kz:       Fixed k_z value.
            n_points: Number of k-points per direction.

        Returns:
            C (integer).
        """
        N = n_points
        dk = 2 * np.pi / N

        # Build eigenstates on the kx-ky grid
        states = np.zeros((N, N, 2), dtype=complex)
        for i in range(N):
            for j in range(N):
                kx = -np.pi + i * dk
                ky = -np.pi + j * dk
                k = np.array([kx, ky, kz])
                states[i, j] = self.H.lower_band_state(k)

        # Accumulate lattice field strength F_{ij}
        log_sum = 0.0j
        for i in range(N):
            for j in range(N):
                Ux_ij = self._link_variable(states, i, j, 1, 0, N)
                Uy_i1j = self._link_variable(states, i + 1, j, 0, 1, N)
                Ux_ij1 = self._link_variable(states, i, j + 1, 1, 0, N)
                Uy_ij = self._link_variable(states, i, j, 0, 1, N)
                F_ij = Ux_ij * Uy_i1j * np.conj(Ux_ij1) * np.conj(Uy_ij)
                log_sum += np.log(F_ij)

        chern = int(round(np.imag(log_sum) / (2 * np.pi)))
        return chern

    def compute_chern_profile(
        self,
        nodes: List[WeylNode],
        n_kz: int = 30,
        n_points: int = 30,
    ) -> ChernProfile:
        """
        Compute Chern number C(k_z) across the full Brillouin zone.

        The Chern number should:
          - equal 0 outside the Weyl-node kz range,
          - jump by ±1 each time a Weyl node is crossed.

        Args:
            nodes:    List of WeylNode objects.
            n_kz:     Number of k_z slices.
            n_points: FHS grid resolution.

        Returns:
            ChernProfile.
        """
        kz_vals = np.linspace(-np.pi, np.pi, n_kz)
        chern_vals = np.zeros(n_kz, dtype=int)

        for idx, kz in enumerate(kz_vals):
            chern_vals[idx] = self.chern_number_at_kz(kz, n_points)
            logger.debug("C(kz=%.3f) = %d", kz, chern_vals[idx])

        node_kz = np.array([n.position[2] for n in nodes])
        node_chi = np.array([n.chirality for n in nodes])

        return ChernProfile(
            kz_values=kz_vals,
            chern_numbers=chern_vals,
            node_kz=node_kz,
            node_chiralities=node_chi,
        )


# ---------------------------------------------------------------------------
# Berry phase along a closed loop
# ---------------------------------------------------------------------------

class BerryPhaseCalculator:
    """
    Discrete Berry phase γ along a closed loop in k-space.

    γ = −Im ln ∏_{n=0}^{N−1} ⟨u(k_n)|u(k_{n+1})⟩

    For a loop encircling a single Weyl node: γ = π (mod 2π).
    """

    def __init__(self, hamiltonian: TaAsLikeHamiltonian):
        self.H = hamiltonian

    def compute(
        self,
        kz: float,
        radius: float,
        n_points: int = 200,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> BerryPhaseResult:
        """
        Compute Berry phase around a circle of *radius* in the k_x–k_y plane.

        Args:
            kz:       Fixed k_z of the loop.
            radius:   Loop radius in k-space.
            n_points: Number of k-points on the loop.
            center:   (kx0, ky0) centre of the loop.

        Returns:
            BerryPhaseResult.
        """
        phi_vals = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        kx0, ky0 = center

        # Build eigenstates on the loop
        states = []
        for phi in phi_vals:
            k = np.array([
                kx0 + radius * np.cos(phi),
                ky0 + radius * np.sin(phi),
                kz,
            ])
            states.append(self.H.lower_band_state(k))

        # Compute discrete product ∏ ⟨u_n|u_{n+1}⟩ (loop: last connects to first)
        log_prod = 0.0j
        for n in range(n_points):
            u_n = states[n]
            u_np1 = states[(n + 1) % n_points]
            inner = complex(np.dot(u_n.conj(), u_np1))
            if abs(inner) > 1e-14:
                log_prod += np.log(inner / abs(inner))

        gamma = float(-np.imag(log_prod))   # Berry phase in (−π, π]
        # Wrap to [0, π] since sign is gauge-dependent
        gamma_wrapped = abs(gamma % (2 * np.pi))
        if gamma_wrapped > np.pi:
            gamma_wrapped = 2 * np.pi - gamma_wrapped

        # Determine how many Weyl nodes are inside the loop
        # Nodes lie at kz = ±k_W on the Γ–Z axis (kx=ky=0)
        node_inside = (
            abs(kz - self.H._kz_node) < 1e-2 or
            abs(kz + self.H._kz_node) < 1e-2
        )
        # A node at kx=ky=0 is inside the loop if its kz matches and radius > 0
        enclosed_chirality = 0
        if node_inside and radius > 0:
            # +1 node at k_z > 0, -1 node at k_z < 0 (from chirality formula)
            enclosed_chirality = +1 if kz > 0 else -1

        result = BerryPhaseResult(
            loop_radius=radius,
            loop_center_kz=kz,
            gamma=gamma_wrapped,
            enclosed_chirality=enclosed_chirality,
            n_points=n_points,
        )
        logger.info(
            "Berry phase at kz=%.3f  r=%.3f  γ=%.4f rad  "
            "(enclosed χ=%+d)",
            kz, radius, gamma_wrapped, enclosed_chirality,
        )
        return result


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class PairedWeylNodeAnalysis:
    """
    Full analysis of paired Weyl nodes in a TaAs-like material.

    Workflow:
      1. Build Hamiltonian and locate Weyl nodes.
      2. Compute Berry curvature Ω(k) maps.
      3. Compute Chern number profile C(k_z).
      4. Compute monopole/antipole Berry flux Φ = ±2π.
      5. Compute Berry phase along loops encircling each node.
      6. Generate publication-quality figures.
    """

    def __init__(
        self,
        M: float = 1.5,
        t_x: float = 1.0,
        t_y: float = 1.0,
        t_z: float = 2.0,
        t: float = 0.5,
    ):
        logger.info("=" * 60)
        logger.info("Paired Weyl Node Analysis  (TaAs-like model)")
        logger.info("=" * 60)

        self.ham = TaAsLikeHamiltonian(M=M, t_x=t_x, t_y=t_y, t_z=t_z, t=t)
        self.berry = BerryCurvatureField(self.ham)
        self.chern_calc = ChernNumberCalculator(self.ham)
        self.bp_calc = BerryPhaseCalculator(self.ham)

        self.nodes: List[WeylNode] = []
        self.chern_profile: Optional[ChernProfile] = None

    # ── Step 1: locate nodes ────────────────────────────────────────────────

    def find_nodes(self) -> List[WeylNode]:
        """Find and store Weyl nodes."""
        self.nodes = self.ham.find_weyl_nodes()
        return self.nodes

    # ── Step 2: Berry curvature maps ────────────────────────────────────────

    def berry_curvature_maps(
        self,
        n_points: int = 50,
        slices: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Compute Ω_z on multiple k_x–k_y slices.

        Args:
            n_points: Grid resolution.
            slices:   List of k_z values; defaults to 5 representative slices.
        """
        if slices is None:
            kw = self.ham._kz_node
            slices = [-kw - 0.4, -kw, 0.0, +kw, +kw + 0.4]

        maps = []
        for kz in slices:
            logger.info("Computing Ω_z on k_z = %.3f plane…", kz)
            m = self.berry.compute_kxky_plane(kz, n_points=n_points, k_range=1.2)
            maps.append(m)
        return maps

    # ── Step 3: Chern number profile ────────────────────────────────────────

    def chern_number_profile(
        self, n_kz: int = 25, n_points: int = 25
    ) -> ChernProfile:
        """Compute C(k_z) and store the result."""
        if not self.nodes:
            self.find_nodes()
        self.chern_profile = self.chern_calc.compute_chern_profile(
            self.nodes, n_kz=n_kz, n_points=n_points
        )
        return self.chern_profile

    # ── Step 4: monopole flux ───────────────────────────────────────────────

    def compute_monopole_fluxes(
        self, radius: float = 0.3, n_theta: int = 14, n_phi: int = 28
    ) -> Dict[str, float]:
        """
        Compute Berry flux Φ = ∮ Ω · dS for each node.

        Theoretical values:  Φ = +2π (monopole) and Φ = −2π (antipole).

        Returns:
            Dict mapping node.label → flux value.
        """
        if not self.nodes:
            self.find_nodes()

        fluxes = {}
        for node in self.nodes:
            phi = self.berry.monopole_flux(
                node, radius=radius, n_theta=n_theta, n_phi=n_phi
            )
            node.berry_flux = phi
            fluxes[node.label] = phi
        return fluxes

    # ── Step 5: Berry phases ────────────────────────────────────────────────

    def compute_berry_phases(
        self,
        radii: Optional[List[float]] = None,
    ) -> Dict[str, List[BerryPhaseResult]]:
        """
        Compute Berry phase γ around loops of different radii at each node's k_z.

        γ should approach π as the loop encloses the node and 0 otherwise.

        Returns:
            Dict mapping node.label → list of BerryPhaseResult.
        """
        if not self.nodes:
            self.find_nodes()
        if radii is None:
            radii = [0.05, 0.15, 0.30, 0.50, 0.80]

        results: Dict[str, List[BerryPhaseResult]] = {}
        for node in self.nodes:
            kz = node.position[2]
            node_results = []
            for r in radii:
                bp = self.bp_calc.compute(kz=kz, radius=r, n_points=150)
                node_results.append(bp)
            results[node.label] = node_results
        return results

    # ── Step 6: update Chern numbers on nodes ──────────────────────────────

    def assign_chern_numbers(self) -> None:
        """
        Assign Chern numbers to each node from the C(k_z) profile step.
        """
        if self.chern_profile is None:
            return
        kz_vals = self.chern_profile.kz_values
        chern_vals = self.chern_profile.chern_numbers
        for node in self.nodes:
            # Find closest kz slice to the node
            idx = int(np.argmin(np.abs(kz_vals - node.position[2])))
            node.chern_number = int(chern_vals[idx])

    # ── Summary ─────────────────────────────────────────────────────────────

    def run_full_analysis(
        self,
        n_points_bc: int = 50,
        n_kz: int = 25,
        n_fhs: int = 25,
    ) -> Dict:
        """
        Run the complete analysis pipeline and return all results.

        Args:
            n_points_bc: Berry curvature grid resolution.
            n_kz:        Number of k_z slices for Chern profile.
            n_fhs:       FHS grid resolution.

        Returns:
            Dictionary with nodes, fluxes, chern_profile, berry_phases.
        """
        logger.info("\n--- Step 1: Locate Weyl nodes ---")
        nodes = self.find_nodes()

        logger.info("\n--- Step 2: Berry curvature maps ---")
        bc_maps = self.berry_curvature_maps(n_points=n_points_bc)

        logger.info("\n--- Step 3: Chern number profile ---")
        chern_prof = self.chern_number_profile(n_kz=n_kz, n_points=n_fhs)
        self.assign_chern_numbers()

        logger.info("\n--- Step 4: Monopole / antipole flux ---")
        fluxes = self.compute_monopole_fluxes()

        logger.info("\n--- Step 5: Berry phases ---")
        bp_results = self.compute_berry_phases()

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        for n in nodes:
            logger.info(str(n))
        for label, phi in fluxes.items():
            chi = next(nn.chirality for nn in nodes if nn.label == label)
            logger.info(
                "  %-5s  Φ = %+.4f  (theory %+.4f  |err| = %.4f)",
                label, phi, 2 * np.pi * chi, abs(phi - 2 * np.pi * chi),
            )

        return {
            "nodes": nodes,
            "berry_curvature_maps": bc_maps,
            "chern_profile": chern_prof,
            "fluxes": fluxes,
            "berry_phases": bp_results,
        }

    # ── Visualisation ────────────────────────────────────────────────────────

    def generate_visualizations(
        self, results: Dict, output_dir: str = "whitepaper/images"
    ) -> Dict[str, str]:
        """
        Generate all figures for the Weyl-node whitepaper.

        Figures
        -------
        1. weyl_nodes_3d.png         — Paired nodes in 3-D BZ with flux arrows
        2. berry_curvature_maps.png  — Ω_z heatmaps at 5 k_z slices
        3. chern_profile.png         — C(k_z) step function
        4. berry_phases.png          — γ vs loop radius for each node
        5. monopole_comparison.png   — Φ bar chart: computed vs expected

        Args:
            results:    Output of run_full_analysis().
            output_dir: Directory for saving figures.

        Returns:
            Dict mapping figure key → file path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        plots: Dict[str, str] = {}

        nodes = results["nodes"]
        bc_maps = results["berry_curvature_maps"]
        chern_prof = results["chern_profile"]
        fluxes = results["fluxes"]
        bp_results = results["berry_phases"]

        # colour per chirality
        def node_color(n: WeylNode) -> str:
            return "#e63946" if n.chirality > 0 else "#457b9d"

        # ── 1. 3-D Weyl node positions with Berry flux arrows ───────────────
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for node in nodes:
            kx, ky, kz = node.position
            color = node_color(node)
            marker = "^" if node.chirality > 0 else "v"
            ax.scatter(kx, ky, kz, s=300, c=color, marker=marker,
                       zorder=5, depthshade=False)
            ax.text(kx + 0.05, ky + 0.05, kz + 0.05,
                    f"{node.label}\n(χ={node.chirality:+d})",
                    fontsize=10, color=color)

            # Draw radial flux arrows (monopole if χ=+1, antipole if χ=-1)
            n_arrows = 8
            for phi_a in np.linspace(0, 2 * np.pi, n_arrows, endpoint=False):
                dr = 0.45
                dx = dr * np.cos(phi_a)
                dz_arr = 0.0
                dy = dr * np.sin(phi_a)
                if node.chirality > 0:
                    ax.quiver(kx, ky, kz, dx, dy, dz_arr,
                              length=0.35, color=color, alpha=0.6,
                              arrow_length_ratio=0.3)
                else:
                    ax.quiver(kx + dx, ky + dy, kz + dz_arr, -dx, -dy, -dz_arr,
                              length=0.35, color=color, alpha=0.6,
                              arrow_length_ratio=0.3)

        # BZ box
        for s in [-np.pi, np.pi]:
            for t in [-np.pi, np.pi]:
                ax.plot([-np.pi, np.pi], [s, s], [t, t], "k-", alpha=0.15, lw=0.8)
                ax.plot([s, s], [-np.pi, np.pi], [t, t], "k-", alpha=0.15, lw=0.8)
                ax.plot([s, s], [t, t], [-np.pi, np.pi], "k-", alpha=0.15, lw=0.8)

        ax.set_xlabel("$k_x$ (rad/a)", fontsize=11)
        ax.set_ylabel("$k_y$ (rad/a)", fontsize=11)
        ax.set_zlabel("$k_z$ (rad/a)", fontsize=11)
        ax.set_title(
            "Paired Weyl Nodes in TaAs-like BZ\n"
            "▲ = Monopole (χ=+1 source)   ▼ = Antipole (χ=−1 sink)",
            fontsize=12,
        )
        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e63946", label="χ = +1  Monopole"),
            Patch(facecolor="#457b9d", label="χ = −1  Antipole"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        path = f"{output_dir}/weyl_nodes_3d.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["weyl_nodes_3d"] = path
        logger.info("Saved %s", path)

        # ── 2. Berry curvature heatmaps ──────────────────────────────────────
        n_slices = len(bc_maps)
        fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
        if n_slices == 1:
            axes = [axes]
        for ax, m in zip(axes, bc_maps):
            oz = m["omega_z"]
            vmax = max(abs(oz).max(), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.pcolormesh(m["kx"], m["ky"], oz,
                               cmap="RdBu_r", norm=norm, shading="auto")
            plt.colorbar(im, ax=ax, label=r"$\Omega_z$ (arb. u.)", shrink=0.8)
            ax.set_title(f"$k_z = {m['kz']:.3f}$", fontsize=11)
            ax.set_xlabel("$k_x$", fontsize=10)
            ax.set_ylabel("$k_y$", fontsize=10)
            ax.set_aspect("equal")
        fig.suptitle(
            r"Berry Curvature $\Omega_z(k_x,k_y)$ at Selected $k_z$ Slices",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        path = f"{output_dir}/berry_curvature_maps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["berry_curvature_maps"] = path
        logger.info("Saved %s", path)

        # ── 3. Chern number profile C(k_z) ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.step(chern_prof.kz_values, chern_prof.chern_numbers,
                where="mid", color="#2a9d8f", linewidth=2.5, label="$C(k_z)$")
        for kz_n, chi in zip(chern_prof.node_kz, chern_prof.node_chiralities):
            color = "#e63946" if chi > 0 else "#457b9d"
            label_txt = f"χ={chi:+d} node"
            ax.axvline(kz_n, color=color, linestyle="--", linewidth=1.5,
                       label=label_txt)
        ax.set_xlabel("$k_z$ (rad/a)", fontsize=12)
        ax.set_ylabel("Chern number $C$", fontsize=12)
        ax.set_title("Chern Number Profile: Step Function across Weyl Nodes", fontsize=13)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/chern_profile.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["chern_profile"] = path
        logger.info("Saved %s", path)

        # ── 4. Berry phase vs loop radius ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        for node in nodes:
            label = node.label
            if label not in bp_results:
                continue
            bplist = bp_results[label]
            radii = [bp.loop_radius for bp in bplist]
            gammas = [bp.gamma for bp in bplist]
            color = node_color(node)
            ax.plot(radii, np.array(gammas) / np.pi, "o-",
                    color=color, linewidth=2, markersize=7,
                    label=f"{label}  (χ={node.chirality:+d})")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5,
                   label=r"$\gamma = \pi$  (topological)")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.5,
                   label=r"$\gamma = 0$  (trivial)")
        ax.set_xlabel("Loop radius $r$  (rad/a)", fontsize=12)
        ax.set_ylabel(r"Berry phase $\gamma / \pi$", fontsize=12)
        ax.set_title(
            "Berry Phase vs Loop Radius\n"
            "(γ = π when loop encloses a Weyl node)",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.3)
        plt.tight_layout()
        path = f"{output_dir}/berry_phases.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["berry_phases"] = path
        logger.info("Saved %s", path)

        # ── 5. Monopole flux bar chart ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(fluxes.keys())
        computed = [fluxes[lb] for lb in labels]
        expected = [2 * np.pi * next(n.chirality for n in nodes if n.label == lb)
                    for lb in labels]
        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width / 2, computed, width,
                       color=["#e63946" if e > 0 else "#457b9d" for e in expected],
                       label="Computed  Φ", alpha=0.9, edgecolor="black")
        bars2 = ax.bar(x + width / 2, expected, width,
                       color=["#ffb703" if e > 0 else "#8ecae6" for e in expected],
                       label="Theory  2πχ", alpha=0.7, edgecolor="black",
                       linestyle="--", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Berry flux  Φ  (rad)", fontsize=12)
        ax.set_title(
            "Monopole / Antipole Berry Flux  Φ = ∮ Ω·dS\n"
            "Positive = monopole (source), Negative = antipole (sink)",
            fontsize=12,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate bars with values
        for bar, val in zip(bars1, computed):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, expected):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        path = f"{output_dir}/monopole_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["monopole_comparison"] = path
        logger.info("Saved %s", path)

        logger.info("Generated %d figures in %s", len(plots), output_dir)
        return plots
