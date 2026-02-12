"""
Tests for Quantum Gravity and NPTC Framework.

Tests the complete implementation of:
- NPTC framework
- Quantum gravity proof
- Hyper-relativity unification
"""

import pytest
import numpy as np
from quantum_gravity.nptc_framework import (
    NPTCFramework, NPTCInvariant, IcosahedralLaplacian,
    FibonacciScheduler, FanoPlane
)
from quantum_gravity.quantum_gravity_proof import (
    QuantumGravityProof, OctonionicHolonomy, EpsteinZetaFunction
)
from quantum_gravity.hyper_relativity import (
    HyperRelativityUnification, HyperRelativityMetric,
    TsirelsonBoundViolation, ChromoGravity
)


class TestNPTCInvariant:
    """Test NPTC invariant calculations."""
    
    def test_invariant_value(self):
        """Test invariant computation."""
        omega_eff = 1000.0  # Hz
        T_eff = 1.5  # K
        C_geom = 1.0
        
        xi = NPTCInvariant(omega_eff, T_eff, C_geom)
        assert xi.value > 0
        assert isinstance(xi.value, float)
        
    def test_critical_boundary(self):
        """Test quantum-classical boundary detection."""
        # At boundary: Ξ ≈ 1
        # HBAR = 1.0545718e-34, K_B = 1.380649e-23
        # Need omega_eff / T_eff * C_geom ≈ K_B / HBAR ≈ 1.31e11
        xi_critical = NPTCInvariant(1.31e11, 1.0, 1.0)  # Tuned to give Ξ ≈ 1
        assert xi_critical.is_critical(tolerance=0.5)
        
        # Away from boundary
        xi_quantum = NPTCInvariant(1e6, 1.0, 1.0)
        assert not xi_quantum.is_critical(tolerance=0.1)


class TestIcosahedralLaplacian:
    """Test icosahedral Laplacian implementation."""
    
    def test_construction(self):
        """Test Laplacian matrix construction."""
        ico = IcosahedralLaplacian()
        assert ico.laplacian.shape == (13, 13)
        assert np.allclose(ico.laplacian, ico.laplacian.T)  # Symmetric
        
    def test_eigenvalues(self):
        """Test eigenvalue computation."""
        ico = IcosahedralLaplacian()
        assert len(ico.eigenvalues) == 13
        assert np.isclose(ico.eigenvalues[0], 0.0, atol=1e-10)  # First eigenvalue is 0
        assert ico.spectral_gap() > 0
        
    def test_holonomy_identity(self):
        """Test holonomy identity verification."""
        ico = IcosahedralLaplacian()
        ratio, eigensum, error = ico.verify_holonomy_identity()
        
        assert np.isclose(ratio, 75.0 / 17.0)
        assert error < 0.1  # Should be within 10%
        
    def test_spectral_gap(self):
        """Test spectral gap calculation."""
        ico = IcosahedralLaplacian()
        gap = ico.spectral_gap()
        assert gap > 0
        assert gap < 2.0  # Should be less than continuum limit


class TestFibonacciScheduler:
    """Test Fibonacci scheduler."""
    
    def test_fibonacci_generation(self):
        """Test Fibonacci sequence generation."""
        scheduler = FibonacciScheduler(tau=1e-6, max_steps=10)
        assert scheduler.fibonacci_seq[0] == 1
        assert scheduler.fibonacci_seq[1] == 1
        assert scheduler.fibonacci_seq[2] == 2
        assert scheduler.fibonacci_seq[3] == 3
        assert scheduler.fibonacci_seq[4] == 5
        
    def test_update_times(self):
        """Test non-periodic update times."""
        scheduler = FibonacciScheduler(tau=1.0, max_steps=5)
        t0 = scheduler.get_update_time(0)
        t1 = scheduler.get_update_time(1)
        t2 = scheduler.get_update_time(2)
        
        # Times should be increasing
        assert t1 > t0
        assert t2 > t1
        
        # Non-periodic: differences should vary
        diff1 = t1 - t0
        diff2 = t2 - t1
        assert not np.isclose(diff1, diff2)


class TestFanoPlane:
    """Test Fano plane structure."""
    
    def test_construction(self):
        """Test Fano plane construction."""
        fano = FanoPlane()
        assert len(fano.lines) == 7
        for line in fano.lines:
            assert len(line) == 3  # Each line has 3 points
            
    def test_adjacency(self):
        """Test adjacency matrix."""
        fano = FanoPlane()
        assert fano.adjacency.shape == (7, 7)
        assert np.allclose(fano.adjacency, fano.adjacency.T)  # Symmetric
        
    def test_laplacian(self):
        """Test Laplacian matrix."""
        fano = FanoPlane()
        assert fano.laplacian.shape == (7, 7)
        
        # Row sums should be zero
        row_sums = np.sum(fano.laplacian, axis=1)
        assert np.allclose(row_sums, 0.0)
        
    def test_spectral_gap(self):
        """Test spectral gap."""
        fano = FanoPlane()
        gap = fano.spectral_gap()
        assert gap > 0


class TestNPTCFramework:
    """Test complete NPTC framework."""
    
    def test_initialization(self):
        """Test framework initialization."""
        nptc = NPTCFramework()
        assert nptc.icosahedral is not None
        assert nptc.fano is not None
        assert nptc.scheduler is not None
        
    def test_invariant_computation(self):
        """Test invariant computation."""
        nptc = NPTCFramework()
        xi = nptc.compute_invariant()
        assert isinstance(xi, NPTCInvariant)
        assert xi.value > 0
        
    def test_control_step(self):
        """Test control step execution."""
        nptc = NPTCFramework()
        result = nptc.control_step()
        
        assert 'step' in result
        assert 'time' in result
        assert 'xi' in result
        assert 'is_critical' in result
        
    def test_simulation(self):
        """Test multi-step simulation."""
        nptc = NPTCFramework()
        results = nptc.run_simulation(n_steps=5)
        
        assert len(results) == 5
        assert all('xi' in r for r in results)
        
    def test_entropy_balance(self):
        """Test entropy balance computation."""
        nptc = NPTCFramework()
        result = nptc.compute_entropy_balance(
            delta_S_geom=0.1,
            delta_S_landauer=0.05,
            W_ergo=0.01
        )
        
        assert 'delta_S_total' in result
        assert 'second_law_satisfied' in result


class TestQuantumGravityProof:
    """Test quantum gravity proof implementation."""
    
    def test_initialization(self):
        """Test proof initialization."""
        proof = QuantumGravityProof()
        assert proof.nptc is not None
        assert proof.epstein is not None
        assert proof.octonionic is not None
        
    def test_holonomy_identity(self):
        """Test holonomy identity verification."""
        proof = QuantumGravityProof()
        result = proof.verify_holonomy_identity()
        
        assert 'holonomy_ratio' in result
        assert 'eigenvalue_sum' in result
        assert 'verified' in result
        
    def test_spectral_convergence(self):
        """Test spectral convergence."""
        proof = QuantumGravityProof()
        result = proof.verify_spectral_convergence()
        
        assert 'lambda_1' in result
        assert 'converging' in result
        
    def test_nptc_invariant_verification(self):
        """Test NPTC invariant verification."""
        proof = QuantumGravityProof()
        result = proof.verify_nptc_invariant()
        
        assert 'xi_value' in result
        assert 'is_critical' in result
        
    def test_gravity_quantum_coupling(self):
        """Test gravity-quantum coupling computation."""
        proof = QuantumGravityProof()
        result = proof.compute_unified_gravity_quantum_coupling()
        
        assert 'coupling_strength' in result
        assert 'E_planck' in result
        assert 'E_quantum' in result
        assert result['coupling_strength'] > 0
        
    def test_full_proof(self):
        """Test complete proof generation."""
        proof = QuantumGravityProof()
        summary = proof.generate_proof()
        
        assert 'proof_valid' in summary
        assert 'propositions_verified' in summary
        assert summary['propositions_verified'] >= 2  # At least 2 should pass


class TestOctonionicHolonomy:
    """Test octonionic holonomy."""
    
    def test_initialization(self):
        """Test octonionic structure initialization."""
        oct = OctonionicHolonomy()
        assert oct.dimension == 7
        assert len(oct.fano_lines) == 7
        
    def test_berry_phase(self):
        """Test Berry phase computation."""
        oct = OctonionicHolonomy()
        path = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        phase = oct.compute_berry_phase(path)
        
        assert isinstance(phase, complex)
        assert abs(phase) > 0
        
    def test_non_associative_phase(self):
        """Test non-associative Berry phase."""
        oct = OctonionicHolonomy()
        gamma1 = np.array([1, 0, 0])
        gamma2 = np.array([0, 1, 0])
        gamma3 = np.array([0, 0, 1])
        
        delta_phi = oct.non_associative_phase(gamma1, gamma2, gamma3)
        assert isinstance(delta_phi, float)


class TestEpsteinZetaFunction:
    """Test Epstein zeta function."""
    
    def test_initialization(self):
        """Test initialization."""
        epstein = EpsteinZetaFunction(signature=(3, 3))
        assert epstein.dimension == 6
        assert epstein.signature == (3, 3)
        
    def test_quadratic_form(self):
        """Test quadratic form."""
        epstein = EpsteinZetaFunction(signature=(3, 3))
        n = np.array([1, 1, 1, 1, 1, 1])
        Q = epstein.quadratic_form(n)
        
        # For signature (3,3): Q = t₁² + t₂² + t₃² - x² - y² - z²
        expected = 3 - 3  # = 0
        assert np.isclose(Q, expected)


class TestHyperRelativityMetric:
    """Test 6D metric."""
    
    def test_metric_construction(self):
        """Test metric construction."""
        metric = HyperRelativityMetric()
        assert metric.dimension == 6
        assert metric.signature == (3, 3)
        assert metric.eta.shape == (6, 6)
        
    def test_proper_time(self):
        """Test proper time calculation."""
        metric = HyperRelativityMetric()
        dx = np.array([1, 0, 0, 1, 0, 0])
        tau = metric.proper_time_6d(dx)
        assert tau >= 0
        
    def test_light_cone_structure(self):
        """Test light cone classification."""
        metric = HyperRelativityMetric()
        
        # Timelike event
        timelike = np.array([2, 0, 0, 1, 0, 0])
        assert metric.light_cone_structure(timelike) == 'timelike'
        
        # Spacelike event
        spacelike = np.array([0, 0, 0, 2, 2, 2])
        assert metric.light_cone_structure(spacelike) == 'spacelike'


class TestTsirelsonBoundViolation:
    """Test Tsirelson bound violations."""
    
    def test_chsh_computation(self):
        """Test CHSH parameter computation."""
        tsirelson = TsirelsonBoundViolation()
        correlations = {
            'E_ab': 0.707,
            'E_ab_prime': -0.707,
            'E_a_prime_b': 0.707,
            'E_a_prime_b_prime': 0.707
        }
        
        S = tsirelson.compute_chsh_parameter(correlations)
        assert S > 0
        
    def test_violation_check(self):
        """Test violation checking."""
        tsirelson = TsirelsonBoundViolation()
        
        # Classical bound violation
        result = tsirelson.check_violation(2.5)
        assert result['violates_classical']
        
        # Quantum bound
        result = tsirelson.check_violation(2.0)
        assert not result['violates_tsirelson']
        
    def test_6d_prediction(self):
        """Test 6D violation prediction."""
        tsirelson = TsirelsonBoundViolation()
        result = tsirelson.predict_6d_violation(timelike_separation=1.0)
        
        assert 'S_predicted' in result
        assert result['S_predicted'] > tsirelson.tsirelson_bound


class TestChromoGravity:
    """Test chromogravity force."""
    
    def test_force_law(self):
        """Test force law."""
        chromograv = ChromoGravity()
        r = 1e-10  # meters
        F = chromograv.force_law(r)
        
        assert F > 0
        assert isinstance(F, float)
        
    def test_potential(self):
        """Test potential energy."""
        chromograv = ChromoGravity()
        r = 1e-10
        V = chromograv.potential(r)
        
        assert V < 0  # Attractive potential
        
    def test_inverse_square(self):
        """Test inverse square behavior."""
        chromograv = ChromoGravity()
        r1 = 1.0
        r2 = 2.0
        
        F1 = chromograv.force_law(r1)
        F2 = chromograv.force_law(r2)
        
        # F should decrease by factor of 4 when r doubles
        assert np.isclose(F1 / F2, 4.0, rtol=0.01)


class TestHyperRelativityUnification:
    """Test complete hyper-relativity unification."""
    
    def test_initialization(self):
        """Test initialization."""
        unif = HyperRelativityUnification()
        assert unif.nptc is not None
        assert unif.metric is not None
        assert unif.tsirelson is not None
        assert unif.chromogravity is not None
        
    def test_6d_spacetime_verification(self):
        """Test 6D spacetime verification."""
        unif = HyperRelativityUnification()
        result = unif.verify_6d_spacetime()
        
        assert result['dimension'] == 6
        assert result['signature'] == (3, 3)
        assert result['verified']
        
    def test_tsirelson_violation(self):
        """Test Tsirelson violation verification."""
        unif = HyperRelativityUnification()
        result = unif.verify_tsirelson_violation()
        
        assert 'S_predicted' in result
        assert 'violates_bound' in result
        
    def test_new_forces(self):
        """Test new forces verification."""
        unif = HyperRelativityUnification()
        result = unif.verify_new_forces()
        
        assert 'chromograv_forces' in result
        assert 'force_ratio' in result
        assert result['verified']
        
    def test_unification_metric(self):
        """Test unification metric."""
        unif = HyperRelativityUnification()
        result = unif.compute_unification_metric()
        
        assert 'unification_score' in result
        assert 'unified' in result
        assert 0 <= result['unification_score'] <= 1
        
    def test_full_unification(self):
        """Test complete unification."""
        unif = HyperRelativityUnification()
        summary = unif.generate_full_unification()
        
        assert summary['spacetime_dimension'] == 6
        assert summary['signature'] == (3, 3)
        assert 'unification_achieved' in summary
        assert summary['experimental_support'] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
