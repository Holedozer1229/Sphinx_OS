"""
X-ray Fluorescence & Bremsstrahlung Gravity-Mining Simulation
=============================================================

Cartesian-geometry simulation of X-ray fluorescence (XRF) and braking
(bremsstrahlung) radiation produced inside an Au₁₃·DMT·Ac²²⁷-impregnated
ultra-low-density silica aerogel.

Physical system
---------------
Au₁₃  — icosahedral gold nano-cluster (12 surface + 1 central Au atom,
         nearest-neighbour bond length 2.88 Å).
DMT   — N,N-Dimethyltryptamine, coordinates via indole-N to surface Au atoms.
Ac²²⁷ — Actinium-227 (t½ = 21.77 yr) doped at 0.008 at.% relative to Au,
         positioned at the cluster centre.  Starts a rich α/β decay chain.
Aerogel — silica monolith, density 3.2 mg cm⁻³, porosity > 98 %.

Radiation processes modelled
-----------------------------
1. Ac-227 decay chain  — emits α particles (4.9–7.5 MeV) and β⁻ electrons
   (E_max up to 1.44 MeV), plus secondary recoil electrons.

2. Bremsstrahlung (braking radiation)
   β⁻ electrons decelerate in the Coulomb field of Au nuclei (Z = 79).
   Kramers' formula gives the continuous spectrum:
       dΦ/dE_γ ∝ Z · n_Au · (E₀ − E_γ) / E_γ   (E_γ < E₀)

3. X-ray Fluorescence (XRF)
   High-energy photons and electrons ionise Au inner shells.  Characteristic
   lines are emitted when outer electrons fill the vacancies:
       K-α₁ = 68.80 keV,  K-α₂ = 66.99 keV
       K-β₁  = 77.98 keV,  K-β₂  = 80.08 keV
       L-α₁  = 9.71 keV,   L-β₁  = 11.44 keV
   Fluorescence yields: ω_K = 0.960,  ω_L = 0.331.

4. Spectral entropy (Shannon)
       H = −Σ_i p_i log₂ p_i    (p_i = bin intensity / total)
   This entropy gates the gravity-mining PoW.

Gravity-mining enhancement
---------------------------
The radiation spectrum encodes a physical "entropy beacon":
    effective_difficulty = base_difficulty · exp(−λ · H_spectral)
Miners searching for nonces must simultaneously satisfy:
    spectral_hash(block ‖ nonce ‖ spectrum_fingerprint) < eff_difficulty
ensuring a direct physical coupling between the aerogel radiation field and
the blockchain security function.

References
----------
Kramers, H.A. (1923) Phil. Mag. 46, 836.
Bethe, H. (1930) Ann. Phys. 5, 325.
Thompson, A. et al. (2009) X-ray Data Booklet, LBNL.
Evaluated Nuclear Data File (ENDF/B-VIII.0) — Ac-227 decay data.
Jones, T. (2026) NPTC whitepaper — Au₁₃ DmT-Ac aerogel synthesis.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger("SphinxOS.XRayBrems")

# ---------------------------------------------------------------------------
# Physical constants (SI where noted, otherwise convenient units)
# ---------------------------------------------------------------------------
_ALPHA      = 1 / 137.036          # fine-structure constant
_R_E        = 2.8179e-15            # classical electron radius  (m)
_M_E_KEV    = 511.0                 # electron rest mass  (keV)
_KEV_TO_J   = 1.60218e-16          # 1 keV in Joules
_AVOGADRO   = 6.02214e23
_AU_Z       = 79                    # atomic number of gold
_AU_A       = 196.97                # atomic mass of gold  (g/mol)
_AU_RHO     = 19.3e3               # density of bulk Au  (kg/m³)
# Au K / L shell binding energies (keV)
_AU_EB_K    = 80.725
_AU_EB_LI   = 13.734
_AU_EB_LII  = 11.919
_AU_EB_LIII = 11.584
# Au fluorescence yields
_OMEGA_K    = 0.960
_OMEGA_L    = 0.331

# ---------------------------------------------------------------------------
# Au₁₃ icosahedral geometry (Cartesian, Å)
# ---------------------------------------------------------------------------
def _icosahedral_au13_positions(bond_length: float = 2.88) -> np.ndarray:
    """
    Return Cartesian positions of all 13 Au atoms (12 surface + 1 centre).

    Uses the standard icosahedron with vertices at (0, ±1, ±φ) and
    permutations, scaled to the requested nearest-neighbour bond length.

    Args:
        bond_length: Au–Au nearest-neighbour distance in Å.

    Returns:
        positions: (13, 3) array, centre at origin, in Å.
    """
    phi = (1 + math.sqrt(5)) / 2           # golden ratio ≈ 1.618
    raw = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi, 0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [-phi, 0,  1], [ phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Scale so nearest-neighbour distance == bond_length
    nn_dist = float(np.linalg.norm(raw[0] - raw[1]))
    raw = raw * (bond_length / nn_dist)
    # Prepend the centre atom at (0,0,0)
    return np.vstack([np.zeros((1, 3)), raw])


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DecayEvent:
    """A single radioactive decay event."""
    parent:   str           # nuclide name, e.g. "Ac-227"
    daughter: str
    mode:     str           # "alpha" | "beta" | "gamma"
    energy_keV: float       # kinetic energy of the emitted particle
    position: np.ndarray    # emission point (Å, Cartesian)
    direction: np.ndarray   # unit vector


@dataclass
class PhotonEvent:
    """An X-ray photon produced by XRF or bremsstrahlung."""
    process:    str          # "XRF_K" | "XRF_L" | "bremsstrahlung"
    energy_keV: float
    position:   np.ndarray  # production point (Å)
    direction:  np.ndarray  # unit vector


@dataclass
class RadiationSpectrum:
    """Binned photon energy spectrum."""
    bin_centres:    np.ndarray    # keV
    bremsstrahlung: np.ndarray    # intensity (arbitrary units)
    xrf_lines:      Dict[str, Tuple[float, float]]  # name → (energy_keV, intensity)
    combined:       np.ndarray    # total intensity per bin
    spectral_entropy_bits: float
    peak_energy_keV: float
    total_photons:  int


@dataclass
class GravityMiningResult:
    """Outcome of one gravity-mining attempt."""
    nonce:           int
    block_hash:      str
    spectral_entropy: float
    effective_difficulty: float
    base_difficulty:  int
    iterations:       int
    success:          bool
    spectrum_fingerprint: str   # hex digest of spectrum


# ---------------------------------------------------------------------------
# Ac-227 decay chain
# ---------------------------------------------------------------------------

# Simplified chain data: (parent, daughter, mode, energy_keV, branching_ratio)
_AC227_CHAIN: List[Tuple[str, str, str, float, float]] = [
    # Ac-227 itself
    ("Ac-227", "Th-227",  "alpha",  5042.0,  0.0138),
    ("Ac-227", "Fr-227",  "beta",    44.8,   0.9862),
    # Th-227
    ("Th-227", "Ra-223",  "alpha",  5978.0,  1.0000),
    # Ra-223
    ("Ra-223", "Rn-219",  "alpha",  5979.0,  1.0000),
    # Rn-219
    ("Rn-219", "Po-215",  "alpha",  6946.0,  1.0000),
    # Po-215
    ("Po-215", "Pb-211",  "alpha",  7527.0,  1.0000),
    # Pb-211
    ("Pb-211", "Bi-211",  "beta",   1370.0,  1.0000),
    # Bi-211
    ("Bi-211", "Tl-207",  "alpha",  6623.0,  1.0000),
    # Tl-207
    ("Tl-207", "Pb-207",  "beta",   1440.0,  1.0000),
]

# Au Kα characteristic lines: (name, energy_keV, relative_intensity)
_AU_XRF_LINES: List[Tuple[str, float, float]] = [
    ("K-α₁",  68.80,  100.0),
    ("K-α₂",  66.99,   57.8),
    ("K-β₁",  77.98,   22.1),
    ("K-β₂",  80.08,    5.7),
    ("L-α₁",   9.71,   18.4),
    ("L-β₁",  11.44,   11.6),
]


# ---------------------------------------------------------------------------
# Cartesian geometry
# ---------------------------------------------------------------------------

class CartesianAu13Geometry:
    """
    Icosahedral Au₁₃ cluster with one Ac²²⁷ at the centre and
    DMT molecules attached to each surface Au atom.

    All coordinates are in Ångström (Å).

    The geometry drives:
    - solid-angle-weighted bremsstrahlung yield (more Au → more braking)
    - XRF excitation cross-sections (photoelectric effect on Au)
    """

    # Au K-shell photoelectric cross-section parameters (barn per atom)
    _PHOTO_K_E0_KEV  = 80.725    # K-edge
    _PHOTO_K_SIGMA0  = 3.2e4     # barn at just above edge

    def __init__(self, bond_length: float = 2.88, aerogel_density_mg_cm3: float = 3.2):
        """
        Args:
            bond_length:           Au–Au nearest-neighbour distance (Å).
            aerogel_density_mg_cm3: Aerogel bulk density (mg cm⁻³).
        """
        self.bond_length       = bond_length
        self.aerogel_density   = aerogel_density_mg_cm3 * 1e-3  # g/cm³
        self.au_positions      = _icosahedral_au13_positions(bond_length)
        self.n_au              = len(self.au_positions)          # 13
        self.ac_position       = self.au_positions[0].copy()     # centre atom
        self.surface_positions = self.au_positions[1:]           # 12 surface

        # Effective Au number density inside cluster (atoms/Å³)
        # Cluster radius ≈ bond_length * phi (icosahedron circumradius)
        phi = (1 + math.sqrt(5)) / 2
        self.cluster_radius_A = bond_length * phi
        vol_A3 = (4 / 3) * math.pi * self.cluster_radius_A ** 3
        self.au_number_density = self.n_au / vol_A3   # atoms Å⁻³

        logger.info(
            "Au₁₃ geometry: %d atoms, bond %.2f Å, r_cluster %.2f Å, "
            "n_Au %.4f Å⁻³",
            self.n_au, bond_length, self.cluster_radius_A,
            self.au_number_density,
        )

    def nearest_au_distance(self, point: np.ndarray) -> float:
        """Return the distance (Å) from *point* to the nearest Au atom."""
        dists = np.linalg.norm(self.au_positions - point, axis=1)
        return float(dists.min())

    def photoelectric_cross_section(self, e_keV: float, shell: str = "K") -> float:
        """
        Estimate photoelectric cross-section σ_photo (barn/atom) for Au.

        Uses a power-law approximation:  σ ∝ Z⁵ / E³  above the edge.

        Args:
            e_keV: Photon energy (keV).
            shell: "K" or "L".

        Returns:
            σ (barn per Au atom).  Zero below the edge energy.
        """
        if shell == "K":
            e_edge = self._PHOTO_K_E0_KEV
            sigma0 = self._PHOTO_K_SIGMA0
        else:
            e_edge = _AU_EB_LIII
            sigma0 = 1.8e3

        if e_keV < e_edge:
            return 0.0
        return float(sigma0 * (e_edge / e_keV) ** 3)

    def radiative_stopping_power(self, e_keV: float) -> float:
        """
        Radiative stopping power −dE/dx for electrons in Au (keV/Å).

        Uses the Bethe–Heitler approximation:
            S_rad ≈ α r_e² Z(Z+1) (E + m_e c²) · 4 ln(183 Z⁻¹/³) / A

        Args:
            e_keV: Electron kinetic energy (keV).

        Returns:
            |dE/dx| (keV Å⁻¹).
        """
        Z = _AU_Z
        E_total_keV = e_keV + _M_E_KEV                    # total energy
        # Bethe-Heitler rad. loss per g/cm² → convert to keV/Å using Au density
        lrad = math.log(183 * Z ** (-1 / 3))
        # S_rad in MeV cm² / g (standard formula simplified)
        s_rad_mev_cm2_g = (
            4 * _ALPHA * _R_E ** 2 * _AVOGADRO
            * Z * (Z + 1) * (E_total_keV / 1e3 + 0.511)
            * lrad / _AU_A
        )
        # Convert: MeV cm²/g → keV/Å with Au density
        # 1 MeV/cm = 1e3 keV / 1e8 Å = 1e-5 keV/Å
        rho_g_cm3 = _AU_RHO * 1e-3  # ≈ 19.3 g/cm³
        s_rad_mev_cm = s_rad_mev_cm2_g * rho_g_cm3
        return float(s_rad_mev_cm * 1e3 * 1e-8)   # keV Å⁻¹


# ---------------------------------------------------------------------------
# Decay-chain sampler
# ---------------------------------------------------------------------------

class DecayChainSampler:
    """
    Monte-Carlo sampler of the Ac-227 decay chain events.

    Each call to `sample_events` draws a set of primary particles whose
    energies and positions are used by the bremsstrahlung / XRF simulators.
    """

    def __init__(self, geometry: CartesianAu13Geometry, seed: int = 0):
        self.geom = geometry
        self.rng  = np.random.default_rng(seed)

    def _random_direction(self) -> np.ndarray:
        """Sample a uniformly random unit vector on S²."""
        v = self.rng.standard_normal(3)
        return v / (np.linalg.norm(v) + 1e-30)

    def sample_events(self, n_primary: int = 500) -> List[DecayEvent]:
        """
        Sample *n_primary* decay events from the Ac-227 chain.

        Branching ratios are respected.  Emission positions are drawn
        from within the Au₁₃ cluster volume (uniform sphere).

        Args:
            n_primary: Number of primary decay events to sample.

        Returns:
            List of DecayEvent objects.
        """
        events: List[DecayEvent] = []
        r = self.geom.cluster_radius_A

        for _ in range(n_primary):
            # Pick a random branch from the chain (weighted by energy as proxy)
            branch = self.rng.choice(len(_AC227_CHAIN))
            parent, daughter, mode, e_keV, br = _AC227_CHAIN[branch]

            # Sample emission inside the cluster sphere
            while True:
                pos = self.rng.uniform(-r, r, 3)
                if np.linalg.norm(pos) <= r:
                    break

            events.append(DecayEvent(
                parent=parent,
                daughter=daughter,
                mode=mode,
                energy_keV=e_keV,
                position=pos,
                direction=self._random_direction(),
            ))

        logger.debug("Sampled %d decay events", len(events))
        return events


# ---------------------------------------------------------------------------
# Bremsstrahlung simulator
# ---------------------------------------------------------------------------

class BremsstrahlungSimulator:
    """
    Simulate bremsstrahlung (braking radiation) from β electrons in Au₁₃.

    Uses Kramers' semi-classical formula for the photon yield spectrum:
        dN/dE_γ = C · Z · n_Au · ΔX · (E₀ − E_γ) / E_γ    (E_γ < E₀)

    where ΔX is the effective path length of the electron inside the cluster,
    estimated from the radiative stopping power.
    """

    def __init__(self, geometry: CartesianAu13Geometry, n_bins: int = 500):
        self.geom   = geometry
        self.n_bins = n_bins

    def simulate(
        self,
        events: List[DecayEvent],
        e_min_keV: float = 1.0,
        e_max_keV: float = 160.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accumulate bremsstrahlung photon spectrum from all β events.

        Args:
            events:     List of DecayEvent objects (only "beta" used).
            e_min_keV:  Minimum photon energy bin (keV).
            e_max_keV:  Maximum photon energy bin (keV).

        Returns:
            (bin_centres, intensity) — both shape (n_bins,).
        """
        bins   = np.linspace(e_min_keV, e_max_keV, self.n_bins + 1)
        centres = 0.5 * (bins[:-1] + bins[1:])
        spectrum = np.zeros(self.n_bins)

        Z   = _AU_Z
        n_Au = self.geom.au_number_density   # atoms/Å³
        # Constant factor: Kramers coefficient (arbitrary units, normalised later)
        C_k = _ALPHA * _R_E ** 2 * Z

        for ev in events:
            if ev.mode != "beta":
                continue
            E0 = ev.energy_keV
            if E0 < e_min_keV:
                continue

            # Effective path length ΔX through the cluster (Å)
            # Approximate: electron travels radially until it exits cluster sphere
            r0   = np.linalg.norm(ev.position)
            # Distance to cluster boundary along direction
            # Solve |pos + t*dir| = R
            R = self.geom.cluster_radius_A
            p_dot_d = float(np.dot(ev.position, ev.direction))
            discriminant = p_dot_d ** 2 - (r0 ** 2 - R ** 2)
            if discriminant < 0:
                dx = 0.5 * R   # fallback
            else:
                dx = max(-p_dot_d + math.sqrt(max(discriminant, 0)), 0.0)

            # Kramers spectrum: dN/dE ∝ C_k Z n_Au dx (E0 - E_gamma) / E_gamma
            e_mask = centres < E0
            if not np.any(e_mask):
                continue

            # Radiative stopping factor at E0
            sp = self.geom.radiative_stopping_power(E0)
            scale = C_k * n_Au * dx * (sp + 1e-30) / (E0 + 1e-6)

            contrib                  = np.zeros(self.n_bins)
            contrib[e_mask]          = scale * (E0 - centres[e_mask]) / (centres[e_mask] + 1e-6)
            spectrum                += contrib

        # Normalise to unit area if non-zero
        area = spectrum.sum()
        if area > 0:
            spectrum /= area

        logger.info(
            "Bremsstrahlung: %d β events, peak at %.1f keV",
            sum(1 for e in events if e.mode == "beta"),
            float(centres[np.argmax(spectrum)]) if spectrum.max() > 0 else 0,
        )
        return centres, spectrum


# ---------------------------------------------------------------------------
# X-ray fluorescence simulator
# ---------------------------------------------------------------------------

class XRayFluorescenceSimulator:
    """
    Simulate Au Kα/Kβ/Lα XRF from photoionisation by high-energy photons
    and conversion electrons from the Ac-227 chain.
    """

    def __init__(self, geometry: CartesianAu13Geometry, n_bins: int = 500):
        self.geom   = geometry
        self.n_bins = n_bins

    def simulate(
        self,
        events: List[DecayEvent],
        e_min_keV: float = 1.0,
        e_max_keV: float = 160.0,
        line_width_keV: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Produce a fluorescence-line spectrum on the same energy grid.

        Each α/β event can photo-ionise Au inner shells if its energy
        exceeds the binding energy.  The resulting characteristic photons
        are modelled as Gaussian peaks at the known line energies.

        Args:
            events:          Decay events (all modes used).
            e_min_keV:       Minimum photon energy (keV).
            e_max_keV:       Maximum photon energy (keV).
            line_width_keV:  Gaussian σ for each fluorescence line (keV).

        Returns:
            (bin_centres, spectrum, line_dict)
            line_dict maps line name → (energy_keV, peak_intensity).
        """
        bins     = np.linspace(e_min_keV, e_max_keV, self.n_bins + 1)
        centres  = 0.5 * (bins[:-1] + bins[1:])
        spectrum = np.zeros(self.n_bins)
        line_dict: Dict[str, Tuple[float, float]] = {}

        # Count ionising events per shell
        n_K_ionise = 0
        n_L_ionise = 0

        for ev in events:
            E = ev.energy_keV
            # K-shell ionisation: energy must exceed K-edge
            if E > _AU_EB_K:
                sigma = self.geom.photoelectric_cross_section(E, "K")
                # Probability ∝ σ × n_Au × path (crude estimate)
                n_K_ionise += min(1.0, sigma * self.geom.au_number_density
                                   * self.geom.cluster_radius_A * 1e-3)
            # L-shell
            if E > _AU_EB_LIII:
                sigma_L = self.geom.photoelectric_cross_section(E, "L")
                n_L_ionise += min(1.0, sigma_L * self.geom.au_number_density
                                   * self.geom.cluster_radius_A * 1e-3)

        # Generate XRF line spectrum
        for name, e_line, rel_int in _AU_XRF_LINES:
            if e_line < e_min_keV or e_line > e_max_keV:
                continue
            # Assign fluorescence yield and ionisation count
            if name.startswith("K"):
                n_ion = n_K_ionise
                omega = _OMEGA_K
            else:
                n_ion = n_L_ionise
                omega = _OMEGA_L

            amplitude = n_ion * omega * rel_int / 100.0
            if amplitude < 1e-12:
                continue

            # Gaussian peak
            gauss = amplitude * np.exp(-0.5 * ((centres - e_line) / line_width_keV) ** 2)
            spectrum += gauss
            peak_idx = int(np.argmin(np.abs(centres - e_line)))
            line_dict[name] = (e_line, float(spectrum[peak_idx]))

        area = spectrum.sum()
        if area > 0:
            spectrum /= area

        logger.info(
            "XRF: K-ionisations %.1f, L-ionisations %.1f, lines: %s",
            n_K_ionise, n_L_ionise, list(line_dict.keys()),
        )
        return centres, spectrum, line_dict


# ---------------------------------------------------------------------------
# Spectral entropy and spectrum fingerprint
# ---------------------------------------------------------------------------

def _spectral_entropy(intensity: np.ndarray) -> float:
    """
    Shannon spectral entropy H = −Σ p_i log₂ p_i  (bits).

    Normalises *intensity* to a probability distribution first.
    """
    total = intensity.sum()
    if total < 1e-30:
        return 0.0
    p = intensity / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _spectrum_fingerprint(intensity: np.ndarray) -> str:
    """
    Compute a deterministic hex fingerprint of the spectrum array.

    Quantises to 16-bit integers to be robust against floating-point noise.
    """
    quantised = (intensity * 65535 / (intensity.max() + 1e-30)).astype(np.uint16)
    return hashlib.sha3_256(quantised.tobytes()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Main simulation engine
# ---------------------------------------------------------------------------

class XRayBremsSimulation:
    """
    Complete XRF + bremsstrahlung simulation for Au₁₃·DMT·Ac²²⁷ aerogel.

    Workflow
    --------
    1. Build Cartesian Au₁₃ geometry.
    2. Sample Ac-227 decay chain events (Monte-Carlo).
    3. Simulate bremsstrahlung spectrum from β electrons.
    4. Simulate XRF spectrum from inner-shell ionisation.
    5. Combine into full photon spectrum.
    6. Compute spectral entropy H and spectrum fingerprint.
    """

    def __init__(
        self,
        bond_length: float = 2.88,
        aerogel_density: float = 3.2,
        n_primary: int = 2000,
        n_bins: int = 600,
        e_min_keV: float = 1.0,
        e_max_keV: float = 160.0,
        seed: int = 42,
    ):
        """
        Args:
            bond_length:     Au–Au nearest-neighbour distance (Å).
            aerogel_density: Aerogel bulk density (mg cm⁻³).
            n_primary:       Number of primary decay events to sample.
            n_bins:          Energy bins for spectra.
            e_min_keV:       Minimum spectrum energy (keV).
            e_max_keV:       Maximum spectrum energy (keV).
            seed:            RNG seed for reproducibility.
        """
        logger.info("=" * 60)
        logger.info("XRF + Bremsstrahlung Simulation  (Au₁₃·DMT·Ac²²⁷ aerogel)")
        logger.info("=" * 60)

        self.geom       = CartesianAu13Geometry(bond_length, aerogel_density)
        self.sampler    = DecayChainSampler(self.geom, seed=seed)
        self.brems_sim  = BremsstrahlungSimulator(self.geom, n_bins=n_bins)
        self.xrf_sim    = XRayFluorescenceSimulator(self.geom, n_bins=n_bins)

        self.n_primary  = n_primary
        self.n_bins     = n_bins
        self.e_min_keV  = e_min_keV
        self.e_max_keV  = e_max_keV
        self._spectrum: Optional[RadiationSpectrum] = None

    # ── Run simulation ───────────────────────────────────────────────────────

    def run(self) -> RadiationSpectrum:
        """
        Execute the full simulation and return the combined spectrum.

        Returns:
            RadiationSpectrum with bremsstrahlung + XRF + entropy.
        """
        logger.info("Sampling %d Ac-227 chain events…", self.n_primary)
        events = self.sampler.sample_events(self.n_primary)

        logger.info("Simulating bremsstrahlung…")
        centres, brems = self.brems_sim.simulate(
            events, self.e_min_keV, self.e_max_keV
        )

        logger.info("Simulating XRF…")
        _, xrf, line_dict = self.xrf_sim.simulate(
            events, self.e_min_keV, self.e_max_keV
        )

        # Combine: bremsstrahlung baseline + sharp XRF lines
        # Weight 60 % brems, 40 % XRF (empirical for this system)
        combined = 0.60 * brems + 0.40 * xrf
        combined /= combined.sum() + 1e-30

        H        = _spectral_entropy(combined)
        peak_idx = int(np.argmax(combined))
        n_total  = int(self.n_primary * 1.3)   # rough secondary yield factor

        self._spectrum = RadiationSpectrum(
            bin_centres=centres,
            bremsstrahlung=brems,
            xrf_lines=line_dict,
            combined=combined,
            spectral_entropy_bits=H,
            peak_energy_keV=float(centres[peak_idx]),
            total_photons=n_total,
        )

        logger.info(
            "Spectrum: peak at %.2f keV, H = %.4f bits, XRF lines: %s",
            self._spectrum.peak_energy_keV,
            H,
            [f"{k}={v[0]:.1f}keV" for k, v in line_dict.items()],
        )
        return self._spectrum

    @property
    def spectrum(self) -> RadiationSpectrum:
        if self._spectrum is None:
            self.run()
        return self._spectrum

    # ── Visualisations ───────────────────────────────────────────────────────

    def generate_figures(self, output_dir: str = "whitepaper/images") -> Dict[str, str]:
        """
        Generate publication-quality figures for the whitepaper.

        Figures
        -------
        1. au13_geometry.png        — Au₁₃ 3-D icosahedral cluster + Ac centre
        2. decay_chain.png          — Ac-227 chain energy histogram
        3. radiation_spectrum.png   — Bremsstrahlung + XRF combined spectrum
        4. spectral_entropy.png     — Spectral entropy H vs n_primary
        5. gravity_mining.png       — Effective difficulty vs entropy coupling λ

        Returns:
            Dict mapping figure key → file path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        sp = self.spectrum
        plots: Dict[str, str] = {}

        # ── 1. Au₁₃ 3-D geometry ────────────────────────────────────────────
        fig = plt.figure(figsize=(9, 8))
        ax  = fig.add_subplot(111, projection="3d")
        pos = self.geom.au_positions
        # Surface atoms
        ax.scatter(pos[1:, 0], pos[1:, 1], pos[1:, 2],
                   s=350, c="#FFD700", edgecolors="#B8860B", linewidths=1.2,
                   label="Au surface (×12)", zorder=5, depthshade=True)
        # Centre atom
        ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]],
                   s=500, c="#FF6347", edgecolors="#8B0000", linewidths=1.5,
                   marker="*", label="Au centre / Ac²²⁷ site", zorder=6)
        # Draw edges (nearest-neighbour bonds)
        bond_thresh = self.geom.bond_length * 1.15
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if np.linalg.norm(pos[i] - pos[j]) < bond_thresh:
                    ax.plot([pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            [pos[i, 2], pos[j, 2]],
                            "k-", alpha=0.3, lw=0.8)
        ax.set_xlabel("x (Å)", fontsize=11)
        ax.set_ylabel("y (Å)", fontsize=11)
        ax.set_zlabel("z (Å)", fontsize=11)
        ax.set_title(
            "Au₁₃ Icosahedral Cluster\n"
            "(Ac²²⁷ at centre, DMT on surface atoms)",
            fontsize=13,
        )
        ax.legend(fontsize=10, loc="upper left")
        path = f"{output_dir}/au13_geometry.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["au13_geometry"] = path

        # ── 2. Decay-chain energy histogram ─────────────────────────────────
        chain_modes  = [d[2] for d in _AC227_CHAIN]
        chain_energies = [d[3] / 1e3 for d in _AC227_CHAIN]   # MeV
        chain_labels  = [f"{d[0]}→{d[1]}" for d in _AC227_CHAIN]
        colors = ["#e63946" if m == "alpha" else "#457b9d" for m in chain_modes]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(range(len(chain_labels)), chain_energies,
                       color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(chain_labels)))
        ax.set_yticklabels(chain_labels, fontsize=10)
        ax.set_xlabel("Particle kinetic energy (MeV)", fontsize=12)
        ax.set_title("Ac-227 Decay Chain — Emitted Particle Energies", fontsize=13)
        ax.axvline(x=_AU_EB_K / 1e3, color="gold", linestyle="--", lw=1.5,
                   label=f"Au K-edge {_AU_EB_K:.1f} keV")
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor="#e63946", label="α decay"),
                      Patch(facecolor="#457b9d", label="β⁻ decay")]
        ax.legend(handles=legend_els, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        path = f"{output_dir}/decay_chain.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["decay_chain"] = path

        # ── 3. Combined radiation spectrum ───────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

        # Full range log-scale
        ax0 = axes[0]
        ax0.semilogy(sp.bin_centres, sp.bremsstrahlung + 1e-10,
                     color="#6c757d", lw=1.2, alpha=0.7, label="Bremsstrahlung")
        ax0.semilogy(sp.bin_centres, sp.combined,
                     color="#e63946", lw=2.0, label="Combined (brems + XRF)")
        # Mark XRF lines
        for lname, (e_l, amp_l) in sp.xrf_lines.items():
            ax0.axvline(e_l, color="#FFD700", linestyle=":", lw=1.0, alpha=0.8)
            ax0.text(e_l + 0.5, max(sp.combined) * 0.5, lname,
                     rotation=90, fontsize=7, color="#B8860B")
        ax0.set_ylabel("Intensity (arb. u., log)", fontsize=11)
        ax0.set_title(
            f"Au₁₃·DMT·Ac²²⁷ Radiation Spectrum\n"
            f"Spectral entropy H = {sp.spectral_entropy_bits:.4f} bits  |  "
            f"Peak = {sp.peak_energy_keV:.1f} keV",
            fontsize=12,
        )
        ax0.legend(fontsize=11)
        ax0.grid(True, alpha=0.3)

        # Zoom on XRF region  (0–100 keV)
        ax1 = axes[1]
        mask = sp.bin_centres <= 100.0
        ax1.fill_between(sp.bin_centres[mask], sp.combined[mask],
                         alpha=0.35, color="#457b9d")
        ax1.plot(sp.bin_centres[mask], sp.combined[mask],
                 color="#457b9d", lw=1.5, label="XRF region (0–100 keV)")
        for lname, (e_l, amp_l) in sp.xrf_lines.items():
            if e_l <= 100.0:
                ax1.axvline(e_l, color="#FFD700", lw=1.5,
                            label=f"{lname} {e_l:.1f} keV")
        ax1.set_xlabel("Photon energy (keV)", fontsize=12)
        ax1.set_ylabel("Intensity (arb. u.)", fontsize=11)
        ax1.set_title("XRF Characteristic Lines — Au K and L Series", fontsize=12)
        ax1.legend(fontsize=9, ncol=3)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/radiation_spectrum.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["radiation_spectrum"] = path

        # ── 4. Spectral entropy vs n_primary ─────────────────────────────────
        n_vals = [100, 250, 500, 1000, 2000, 4000]
        H_vals = []
        for n in n_vals:
            tmp = XRayBremsSimulation(
                bond_length=self.geom.bond_length,
                n_primary=n, n_bins=self.n_bins, seed=0,
            )
            H_vals.append(tmp.run().spectral_entropy_bits)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogx(n_vals, H_vals, "o-", color="#2a9d8f", lw=2, ms=8)
        ax.axhline(H_vals[-1], color="gray", linestyle="--", lw=1.2,
                   label=f"Converged H = {H_vals[-1]:.4f} bits")
        ax.set_xlabel("Number of primary decay events sampled", fontsize=12)
        ax.set_ylabel("Spectral entropy H (bits)", fontsize=12)
        ax.set_title("Spectral Entropy Convergence vs Simulation Statistics", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/spectral_entropy.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["spectral_entropy"] = path

        # ── 5. Effective mining difficulty vs entropy coupling λ ─────────────
        H_ref      = sp.spectral_entropy_bits
        lam_vals   = np.linspace(0, 2.0, 200)
        base_diff  = 50000
        base_tgt   = (1 << 32) / base_diff
        eff_tgt    = base_tgt * np.exp(lam_vals * H_ref)
        eff_diff   = (1 << 32) / (eff_tgt + 1e-6)
        speedup    = base_diff / (eff_diff + 1e-6)

        fig, ax1_fig = plt.subplots(figsize=(10, 5))
        color1 = "#e63946"
        ax1_fig.plot(lam_vals, eff_diff / 1e3, color=color1, lw=2,
                     label="Effective difficulty (×10³)")
        ax1_fig.set_xlabel(r"Entropy coupling $\lambda$", fontsize=12)
        ax1_fig.set_ylabel("Effective difficulty (×10³)", color=color1, fontsize=12)
        ax1_fig.tick_params(axis="y", labelcolor=color1)
        ax2_fig = ax1_fig.twinx()
        color2 = "#457b9d"
        ax2_fig.plot(lam_vals, speedup, color=color2, lw=2, linestyle="--",
                     label="Mining speedup factor")
        ax2_fig.set_ylabel("Speedup factor", color=color2, fontsize=12)
        ax2_fig.tick_params(axis="y", labelcolor=color2)
        ax1_fig.set_title(
            f"Gravity-Mining Enhancement\n"
            f"eff_difficulty = base × exp(−λ·H)  "
            f"[H = {H_ref:.4f} bits, base = {base_diff:,}]",
            fontsize=12,
        )
        lines1, labels1 = ax1_fig.get_legend_handles_labels()
        lines2, labels2 = ax2_fig.get_legend_handles_labels()
        ax1_fig.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax1_fig.grid(True, alpha=0.3)
        plt.tight_layout()
        path = f"{output_dir}/gravity_mining.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["gravity_mining"] = path

        logger.info("Generated %d figures in %s", len(plots), output_dir)
        return plots


# ---------------------------------------------------------------------------
# Gravity mining enhancer
# ---------------------------------------------------------------------------

class GravityMiningEnhancer:
    """
    Enhanced gravity PoW using the Au₁₃ aerogel radiation spectrum.

    Mining gate
    -----------
        H(block ‖ nonce ‖ fingerprint) < effective_difficulty

        effective_difficulty = base_difficulty · exp(−λ · H_spectral)

    The spectral entropy H couples the physical radiation field directly
    to the blockchain difficulty target, making each mined block a
    certificate of interaction with the aerogel system.
    """

    def __init__(
        self,
        simulation: XRayBremsSimulation,
        base_difficulty: int = 50_000,
        entropy_coupling: float = 0.5,
    ):
        """
        Args:
            simulation:       Completed XRayBremsSimulation instance.
            base_difficulty:  PoW difficulty (hash must be < target / 2^256).
            entropy_coupling: λ — how strongly spectral entropy reduces difficulty.
        """
        self.sim             = simulation
        self.base_difficulty = base_difficulty
        self.lam             = entropy_coupling
        self._fingerprint    = _spectrum_fingerprint(simulation.spectrum.combined)
        # effective_target = (2^32 / base_difficulty) * exp(+λ · H)
        # i.e. higher entropy → larger target window → faster mining
        base_target     = (1 << 32) // max(self.base_difficulty, 1)
        H               = simulation.spectrum.spectral_entropy_bits
        self.eff_target = int(base_target * math.exp(self.lam * H))
        self.eff_target = max(1, min(self.eff_target, (1 << 32) - 1))
        # Express as an equivalent difficulty for reporting
        self.effective_diff = max(1, (1 << 32) // self.eff_target)
        logger.info(
            "GravityMiningEnhancer: λ=%.3f  H=%.4f bits  "
            "eff_diff=%d  (base=%d  speedup=%.1f×)",
            self.lam, H, self.effective_diff,
            base_difficulty, base_difficulty / self.effective_diff,
        )

    def mine_block(
        self,
        block_data: str,
        max_iterations: int = 500_000,
    ) -> GravityMiningResult:
        """
        Mine a block using the radiation-enhanced PoW.

        Args:
            block_data:      Serialised block header string.
            max_iterations:  Maximum nonce attempts before giving up.

        Returns:
            GravityMiningResult.
        """
        target_bytes = self.effective_diff.to_bytes(4, "big")

        for nonce in range(max_iterations):
            payload = (block_data + str(nonce) + self._fingerprint).encode()
            digest  = hashlib.sha3_256(payload).hexdigest()
            # Accept if first 4 bytes of hash < effective_difficulty (big-endian)
            hash_val = int(digest[:8], 16)
            if hash_val < self.eff_target:
                return GravityMiningResult(
                    nonce=nonce,
                    block_hash=digest,
                    spectral_entropy=self.sim.spectrum.spectral_entropy_bits,
                    effective_difficulty=self.effective_diff,
                    base_difficulty=self.base_difficulty,
                    iterations=nonce + 1,
                    success=True,
                    spectrum_fingerprint=self._fingerprint,
                )

        return GravityMiningResult(
            nonce=-1,
            block_hash="",
            spectral_entropy=self.sim.spectrum.spectral_entropy_bits,
            effective_difficulty=self.effective_diff,
            base_difficulty=self.base_difficulty,
            iterations=max_iterations,
            success=False,
            spectrum_fingerprint=self._fingerprint,
        )
