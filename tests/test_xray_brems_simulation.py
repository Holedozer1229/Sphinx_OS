"""
Tests for XRay + Bremsstrahlung Gravity-Mining Simulation.

Covers:
  - Au₁₃ Cartesian geometry construction
  - Ac-227 decay-chain event sampling
  - Bremsstrahlung spectrum properties
  - XRF characteristic line detection
  - Spectral entropy calculation
  - Gravity-mining PoW (success, difficulty, speedup)
"""

import math
import pytest
import numpy as np

from sphinx_os.AnubisCore.xray_brems_simulation import (
    CartesianAu13Geometry,
    DecayChainSampler,
    BremsstrahlungSimulator,
    XRayFluorescenceSimulator,
    XRayBremsSimulation,
    GravityMiningEnhancer,
    RadiationSpectrum,
    _icosahedral_au13_positions,
    _spectral_entropy,
    _spectrum_fingerprint,
    _AU_XRF_LINES,
)


# ---------------------------------------------------------------------------
# Cartesian geometry
# ---------------------------------------------------------------------------

class TestCartesianAu13Geometry:

    def test_icosahedral_positions_count(self):
        pos = _icosahedral_au13_positions(2.88)
        assert pos.shape == (13, 3), "Must return 13 Au atoms"

    def test_centre_at_origin(self):
        pos = _icosahedral_au13_positions(2.88)
        assert np.allclose(pos[0], [0, 0, 0], atol=1e-10), "Centre atom at origin"

    def test_nearest_neighbour_distance(self):
        bl = 2.88
        pos = _icosahedral_au13_positions(bl)
        surface = pos[1:]
        dists = []
        for i in range(len(surface)):
            for j in range(i + 1, len(surface)):
                dists.append(np.linalg.norm(surface[i] - surface[j]))
        nn = min(dists)
        assert abs(nn - bl) < 0.01, f"Nearest-neighbour {nn:.4f} Å ≠ {bl} Å"

    def test_geometry_number_density_positive(self):
        geom = CartesianAu13Geometry(bond_length=2.88)
        assert geom.au_number_density > 0

    def test_photoelectric_cross_section_above_edge(self):
        geom = CartesianAu13Geometry()
        # K-edge 80.7 keV  — 100 keV should have finite cross section
        sigma = geom.photoelectric_cross_section(100.0, shell="K")
        assert sigma > 0

    def test_photoelectric_cross_section_below_edge(self):
        geom = CartesianAu13Geometry()
        sigma = geom.photoelectric_cross_section(50.0, shell="K")
        assert sigma == 0.0, "Below K-edge: σ must be zero"

    def test_radiative_stopping_power_positive(self):
        geom = CartesianAu13Geometry()
        sp = geom.radiative_stopping_power(500.0)   # 500 keV electron
        assert sp > 0


# ---------------------------------------------------------------------------
# Decay-chain sampler
# ---------------------------------------------------------------------------

class TestDecayChainSampler:

    def setup_method(self):
        self.geom    = CartesianAu13Geometry()
        self.sampler = DecayChainSampler(self.geom, seed=0)

    def test_event_count(self):
        events = self.sampler.sample_events(100)
        assert len(events) == 100

    def test_event_positions_inside_cluster(self):
        events = self.sampler.sample_events(200)
        R = self.geom.cluster_radius_A
        for ev in events:
            assert np.linalg.norm(ev.position) <= R + 1e-9

    def test_directions_unit_vectors(self):
        events = self.sampler.sample_events(50)
        for ev in events:
            assert abs(np.linalg.norm(ev.direction) - 1.0) < 1e-9

    def test_modes_are_valid(self):
        events = self.sampler.sample_events(300)
        valid_modes = {"alpha", "beta"}
        for ev in events:
            assert ev.mode in valid_modes

    def test_energies_positive(self):
        events = self.sampler.sample_events(100)
        for ev in events:
            assert ev.energy_keV > 0


# ---------------------------------------------------------------------------
# Bremsstrahlung simulator
# ---------------------------------------------------------------------------

class TestBremsstrahlungSimulator:

    def setup_method(self):
        self.geom    = CartesianAu13Geometry()
        self.sampler = DecayChainSampler(self.geom, seed=1)
        self.sim     = BremsstrahlungSimulator(self.geom, n_bins=300)
        self.events  = self.sampler.sample_events(500)

    def test_spectrum_length(self):
        centres, spectrum = self.sim.simulate(self.events)
        assert len(centres) == 300
        assert len(spectrum) == 300

    def test_spectrum_non_negative(self):
        _, spectrum = self.sim.simulate(self.events)
        assert np.all(spectrum >= 0)

    def test_spectrum_normalised(self):
        _, spectrum = self.sim.simulate(self.events)
        if spectrum.sum() > 0:
            assert abs(spectrum.sum() - 1.0) < 1e-6

    def test_spectrum_decreasing_trend(self):
        """Bremsstrahlung should generally decrease with photon energy."""
        centres, spectrum = self.sim.simulate(self.events, e_min_keV=5.0, e_max_keV=100.0)
        low  = spectrum[:50].mean()
        high = spectrum[-50:].mean()
        # Low-energy portion should dominate (Kramers' formula)
        assert low >= high, "Bremsstrahlung should decrease with energy"


# ---------------------------------------------------------------------------
# XRF simulator
# ---------------------------------------------------------------------------

class TestXRayFluorescenceSimulator:

    def setup_method(self):
        self.geom    = CartesianAu13Geometry()
        self.sampler = DecayChainSampler(self.geom, seed=2)
        self.sim     = XRayFluorescenceSimulator(self.geom, n_bins=400)
        self.events  = self.sampler.sample_events(1000)

    def test_spectrum_non_negative(self):
        centres, spectrum, lines = self.sim.simulate(self.events)
        assert np.all(spectrum >= 0)

    def test_xrf_lines_detected(self):
        _, _, lines = self.sim.simulate(self.events)
        # At least K-α₁ should always appear (highest branching)
        assert len(lines) > 0

    def test_xrf_line_energies_in_range(self):
        centres, _, lines = self.sim.simulate(self.events, e_min_keV=1.0, e_max_keV=160.0)
        for name, (e_keV, _) in lines.items():
            assert 1.0 <= e_keV <= 160.0, f"Line {name} at {e_keV} keV out of range"

    def test_kα1_energy_correct(self):
        """Au K-α₁ must be near 68.8 keV."""
        _, _, lines = self.sim.simulate(self.events)
        if "K-α₁" in lines:
            e, _ = lines["K-α₁"]
            assert abs(e - 68.80) < 0.1


# ---------------------------------------------------------------------------
# Spectral entropy and fingerprint
# ---------------------------------------------------------------------------

class TestSpectralEntropy:

    def test_uniform_distribution(self):
        n = 128
        uniform = np.ones(n) / n
        H = _spectral_entropy(uniform)
        assert abs(H - math.log2(n)) < 1e-4   # H_uniform = log2(N)

    def test_delta_distribution(self):
        delta = np.zeros(50)
        delta[25] = 1.0
        H = _spectral_entropy(delta)
        assert H < 0.01   # perfectly ordered → H ≈ 0

    def test_entropy_positive(self):
        rng = np.random.default_rng(0)
        rand = np.abs(rng.standard_normal(200))
        H = _spectral_entropy(rand)
        assert H > 0

    def test_fingerprint_deterministic(self):
        arr = np.linspace(0, 1, 100)
        fp1 = _spectrum_fingerprint(arr)
        fp2 = _spectrum_fingerprint(arr)
        assert fp1 == fp2

    def test_fingerprint_sensitive(self):
        a = np.ones(100)
        b = a.copy()
        b[50] += 1e-4
        assert _spectrum_fingerprint(a) != _spectrum_fingerprint(b)


# ---------------------------------------------------------------------------
# Full simulation pipeline
# ---------------------------------------------------------------------------

class TestXRayBremsSimulation:

    @pytest.fixture(scope="class")
    def sim_result(self):
        sim = XRayBremsSimulation(n_primary=500, n_bins=200, seed=7)
        return sim.run()

    def test_spectrum_type(self, sim_result):
        assert isinstance(sim_result, RadiationSpectrum)

    def test_peak_energy_in_au_kα_region(self, sim_result):
        # Peak should be near Au K-α (≈68–69 keV) dominated by XRF
        assert 60.0 <= sim_result.peak_energy_keV <= 90.0

    def test_spectral_entropy_positive(self, sim_result):
        assert sim_result.spectral_entropy_bits > 0

    def test_spectral_entropy_bounded(self, sim_result):
        # For 200 bins: max entropy = log2(200) ≈ 7.6 bits
        assert sim_result.spectral_entropy_bits <= math.log2(200) + 0.1

    def test_combined_spectrum_normalised(self, sim_result):
        assert abs(sim_result.combined.sum() - 1.0) < 1e-5

    def test_xrf_lines_present(self, sim_result):
        assert len(sim_result.xrf_lines) >= 1


# ---------------------------------------------------------------------------
# Gravity mining
# ---------------------------------------------------------------------------

class TestGravityMiningEnhancer:

    @pytest.fixture(scope="class")
    def miner(self):
        sim = XRayBremsSimulation(n_primary=500, n_bins=200, seed=9)
        return GravityMiningEnhancer(sim, base_difficulty=5_000, entropy_coupling=0.5)

    def test_effective_difficulty_lower_than_base(self, miner):
        """Higher entropy → larger target window → lower effective difficulty."""
        assert miner.effective_diff <= miner.base_difficulty

    def test_speedup_positive(self, miner):
        speedup = miner.base_difficulty / miner.effective_diff
        assert speedup >= 1.0

    def test_mine_success(self, miner):
        """With difficulty=5000 and λ=0.5, H≈7 bits, should mine quickly."""
        result = miner.mine_block("TEST_BLOCK_DATA", max_iterations=50_000)
        assert result.success, (
            f"Mining should succeed within 50k iters "
            f"(eff_diff={miner.effective_diff}, eff_tgt={miner.eff_target})"
        )

    def test_mine_iterations_reasonable(self, miner):
        result = miner.mine_block("TEST2", max_iterations=50_000)
        if result.success:
            # Expected iterations ≈ effective_difficulty ≈ base/speedup
            assert result.iterations <= 10 * miner.effective_diff + 1000

    def test_fingerprint_is_hex(self, miner):
        fp = miner._fingerprint
        assert len(fp) == 32
        int(fp, 16)   # should not raise

    def test_different_blocks_different_hashes(self, miner):
        r1 = miner.mine_block("BLOCK_A", max_iterations=50_000)
        r2 = miner.mine_block("BLOCK_B", max_iterations=50_000)
        if r1.success and r2.success:
            assert r1.block_hash != r2.block_hash
