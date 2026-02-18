"""
Test suite for Jones Quantum Gravity Resolution framework.

Tests cover:
1. Modular Hamiltonian construction
2. Spectral properties
3. Entanglement islands
4. Page curve computation
5. Geodesic flow
6. Visualization generation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_gravity.jones_quantum_gravity import (
    ExceptionalJordanAlgebra,
    ContractionOperator,
    TrialityOperator,
    CTCRotationOperator,
    FreezingOperator,
    ModularHamiltonian,
    DeterministicPageCurve,
    EntanglementMetric,
    JonesQuantumGravityResolution,
    ModularSpectrum,
    EntanglementIsland
)


class TestExceptionalJordanAlgebra:
    """Test exceptional Jordan algebra J_3(O)."""
    
    def test_initialization(self):
        """Test Jordan algebra initialization."""
        algebra = ExceptionalJordanAlgebra(dimension=27)
        assert algebra.dimension == 27
        assert algebra.block_size == 9
        
    def test_jordan_product_commutative(self):
        """Test Jordan product commutativity: A·B = B·A."""
        algebra = ExceptionalJordanAlgebra(dimension=27)
        A = algebra.create_hermitian_element()
        B = algebra.create_hermitian_element()
        
        AB = algebra.jordan_product(A, B)
        BA = algebra.jordan_product(B, A)
        
        assert np.allclose(AB, BA), "Jordan product should be commutative"
    
    def test_hermitian_element(self):
        """Test that created elements are Hermitian."""
        algebra = ExceptionalJordanAlgebra(dimension=27)
        H = algebra.create_hermitian_element()
        
        assert H.shape == (27, 27)
        assert np.allclose(H, H.T), "Element should be Hermitian (symmetric)"


class TestComponentOperators:
    """Test component operators C, T, U, F."""
    
    def test_contraction_operator(self):
        """Test contraction operator properties."""
        C = ContractionOperator(dimension=27, contraction_strength=1.0)
        
        assert C.operator.shape == (27, 27)
        # Should be positive definite (all eigenvalues positive)
        eigenvals = np.linalg.eigvalsh(C.operator)
        assert np.all(eigenvals > 0), "Contraction operator should be positive definite"
    
    def test_triality_operator(self):
        """Test triality operator cyclic permutation."""
        T = TrialityOperator(dimension=27)
        
        assert T.operator.shape == (27, 27)
        
        # Test that it's a permutation (orthogonal)
        T_T_inv = T.operator.T @ T.operator
        assert np.allclose(T_T_inv, np.eye(27)), "Triality should be orthogonal"
        
        # Test cyclic property: T^3 = I
        T3 = T.operator @ T.operator @ T.operator
        assert np.allclose(T3, np.eye(27)), "Triality cubed should be identity"
    
    def test_ctc_rotation_operator(self):
        """Test CTC rotation operator."""
        U = CTCRotationOperator(dimension=27, rotation_angle=np.pi/6)
        
        assert U.operator.shape == (27, 27)
        
        # Should be approximately orthogonal
        U_T_U = U.operator.T @ U.operator
        assert np.allclose(U_T_U, np.eye(27), atol=1e-10), "CTC rotation should be orthogonal"
    
    def test_freezing_operator(self):
        """Test freezing operator properties."""
        F = FreezingOperator(dimension=27, freeze_threshold=0.1)
        
        assert F.operator.shape == (27, 27)
        
        # Should be diagonal positive definite
        assert np.allclose(F.operator, np.diag(np.diag(F.operator))), "Freezing should be diagonal"
        eigenvals = np.linalg.eigvalsh(F.operator)
        assert np.all(eigenvals > 0), "Freezing operator should be positive definite"


class TestModularHamiltonian:
    """Test modular Hamiltonian construction and properties."""
    
    @pytest.fixture
    def hamiltonian(self):
        """Create a modular Hamiltonian for testing."""
        return ModularHamiltonian(dimension=27)
    
    def test_initialization(self, hamiltonian):
        """Test modular Hamiltonian initialization."""
        assert hamiltonian.dimension == 27
        assert hamiltonian.Delta is not None
        assert hamiltonian.K is not None
        assert hamiltonian.spectrum is not None
    
    def test_modular_operator_positive_definite(self, hamiltonian):
        """Test that modular operator Δ is positive definite."""
        eigenvals = np.linalg.eigvalsh(hamiltonian.Delta)
        assert np.all(eigenvals > 0), "Modular operator Δ should be positive definite"
    
    def test_modular_hamiltonian_hermitian(self, hamiltonian):
        """Test that modular Hamiltonian K is Hermitian."""
        assert np.allclose(hamiltonian.K, hamiltonian.K.T), "K should be Hermitian"
    
    def test_spectral_gap_positive(self, hamiltonian):
        """Test that spectral gap κ is positive."""
        kappa = hamiltonian.get_spectral_gap()
        assert kappa > 0, "Spectral gap κ should be positive"
    
    def test_spectrum_properties(self, hamiltonian):
        """Test modular spectrum properties."""
        spectrum = hamiltonian.spectrum
        
        assert len(spectrum.eigenvalues) == 27
        assert spectrum.eigenvalues.shape == (27,)
        assert spectrum.eigenvectors.shape == (27, 27)
        assert np.isclose(spectrum.kappa, spectrum.eigenvalues.min())
    
    def test_block_spectral_gaps(self, hamiltonian):
        """Test block spectral gap computation."""
        gaps = hamiltonian.compute_block_spectral_gaps(block_size=3)
        
        # Should be 9×9 grid for 27D space with block_size=3
        assert gaps.shape == (9, 9)
        # All gaps should be real and finite
        assert np.all(np.isfinite(gaps))
    
    def test_find_islands(self, hamiltonian):
        """Test entanglement island detection."""
        islands = hamiltonian.find_islands(tolerance=0.5)
        
        # Should return a list (possibly empty)
        assert isinstance(islands, list)
        
        # If islands found, check properties
        for island in islands:
            assert isinstance(island, EntanglementIsland)
            assert island.location.shape == (27,)
            assert island.projection.shape == (27, 27)
            assert island.rank_reduction > 0
            assert island.entropy_contribution >= 0


class TestDeterministicPageCurve:
    """Test Page curve computation."""
    
    @pytest.fixture
    def page_curve(self):
        """Create a Page curve calculator for testing."""
        hamiltonian = ModularHamiltonian(dimension=27)
        return DeterministicPageCurve(hamiltonian)
    
    def test_initialization(self, page_curve):
        """Test Page curve initialization."""
        assert page_curve.K is not None
        assert page_curve.dimension == 27
    
    def test_modular_density(self, page_curve):
        """Test modular density K(x) computation."""
        # Test at various points
        for x in [0.0, 0.5, 1.0]:
            K_x = page_curve.modular_density(x)
            assert np.isfinite(K_x), f"K({x}) should be finite"
            assert K_x >= 0, f"K({x}) should be non-negative"
    
    def test_entropy_monotonic(self, page_curve):
        """Test that entropy S(x) is monotonically increasing."""
        x_vals = np.linspace(0, 1, 10)
        S_vals = [page_curve.entropy(x) for x in x_vals]
        
        # Check monotonicity
        for i in range(len(S_vals) - 1):
            assert S_vals[i+1] >= S_vals[i], "Entropy should be monotonically increasing"
    
    def test_entropy_initial_condition(self, page_curve):
        """Test that S(0) = 0."""
        S_0 = page_curve.entropy(0.0)
        assert abs(S_0) < 1e-6, "Entropy at x=0 should be zero"
    
    def test_compute_page_curve(self, page_curve):
        """Test Page curve computation."""
        x_vals, S_vals = page_curve.compute_page_curve(n_points=50)
        
        assert len(x_vals) == 50
        assert len(S_vals) == 50
        assert x_vals[0] == 0.0
        assert x_vals[-1] == 1.0
        assert S_vals[0] <= S_vals[-1]  # Generally increasing
    
    def test_nuclearity_bound(self, page_curve):
        """Test nuclearity bound computation."""
        bound = page_curve.nuclearity_bound()
        
        assert bound > 0
        assert np.isclose(bound, np.log(27))  # ln(dim H_R)
    
    def test_verify_nuclearity(self, page_curve):
        """Test nuclearity verification."""
        result = page_curve.verify_nuclearity()
        
        assert 'max_entropy' in result
        assert 'nuclearity_bound' in result
        assert 'satisfies_bound' in result
        assert 'margin' in result
        
        assert result['max_entropy'] >= 0
        assert result['nuclearity_bound'] > 0


class TestEntanglementMetric:
    """Test entanglement metric and geodesics."""
    
    @pytest.fixture
    def metric(self):
        """Create entanglement metric for testing."""
        hamiltonian = ModularHamiltonian(dimension=27)
        page_curve = DeterministicPageCurve(hamiltonian)
        return EntanglementMetric(page_curve)
    
    def test_initialization(self, metric):
        """Test metric initialization."""
        assert metric.page_curve is not None
        assert metric.dimension == 27
    
    def test_metric_tensor_symmetric(self, metric):
        """Test that metric tensor is symmetric."""
        x = np.array([0.5, 0.5, 0.5])
        g = metric.metric_tensor(x, epsilon=1e-5)
        
        assert g.shape == (3, 3)
        assert np.allclose(g, g.T), "Metric tensor should be symmetric"
    
    def test_christoffel_symbols(self, metric):
        """Test Christoffel symbol computation."""
        x = np.array([0.5, 0.5, 0.5])
        Gamma = metric.christoffel_symbols(x, epsilon=1e-5)
        
        assert Gamma.shape == (3, 3, 3)
        assert np.all(np.isfinite(Gamma)), "Christoffel symbols should be finite"
    
    def test_geodesic_computation(self, metric):
        """Test geodesic trajectory computation."""
        x0 = np.array([0.3, 0.5, 0.7])
        v0 = np.array([0.1, -0.05, 0.08])
        
        result = metric.compute_geodesic(x0, v0, t_span=(0, 1), n_points=20)
        
        assert 't' in result
        assert 'trajectory' in result
        assert 'velocities' in result
        assert 'success' in result
        
        if result['success']:
            assert len(result['t']) == 20
            assert result['trajectory'].shape[0] == 20
            assert result['trajectory'].shape[1] == 3
    
    def test_project_to_3d(self, metric):
        """Test 3D projection."""
        # Test with already 3D data
        trajectory_3d = np.random.randn(50, 3)
        projected = metric.project_to_3d(trajectory_3d)
        
        assert projected.shape == (50, 3)
        assert np.allclose(projected, trajectory_3d)
        
        # Test with higher dimensional data
        trajectory_high = np.random.randn(50, 10)
        projected = metric.project_to_3d(trajectory_high)
        
        assert projected.shape == (50, 3)


class TestJonesQuantumGravityResolution:
    """Test complete Jones Quantum Gravity Resolution framework."""
    
    @pytest.fixture
    def jones(self):
        """Create Jones framework for testing."""
        return JonesQuantumGravityResolution(dimension=27)
    
    def test_initialization(self, jones):
        """Test framework initialization."""
        assert jones.jordan_algebra is not None
        assert jones.modular_hamiltonian is not None
        assert jones.page_curve is not None
        assert jones.metric is not None
    
    def test_analyze_spectral_structure(self, jones):
        """Test spectral analysis."""
        result = jones.analyze_spectral_structure()
        
        assert 'spectral_gap_kappa' in result
        assert 'eigenvalue_range' in result
        assert 'eigenvalue_mean' in result
        assert 'eigenvalue_std' in result
        assert 'dimension' in result
        
        assert result['spectral_gap_kappa'] > 0
        assert result['dimension'] == 27
    
    def test_find_entanglement_islands(self, jones):
        """Test island finding."""
        islands = jones.find_entanglement_islands(tolerance=0.5)
        
        assert isinstance(islands, list)
        # Should find at least some islands with reasonable tolerance
        # (but this depends on the random seed, so we don't assert a specific number)
    
    def test_compute_page_curve(self, jones):
        """Test Page curve computation."""
        result = jones.compute_page_curve(n_points=50)
        
        assert 'x' in result
        assert 'S' in result
        assert 'verification' in result
        assert 'max_entropy' in result
        assert 'saturation_point' in result
        
        assert len(result['x']) == 50
        assert len(result['S']) == 50
        assert result['max_entropy'] >= 0
    
    def test_compute_geodesic_flow(self, jones):
        """Test geodesic flow computation."""
        x0 = np.array([0.3, 0.5, 0.7])
        v0 = np.array([0.1, -0.05, 0.08])
        
        result = jones.compute_geodesic_flow(x0=x0, v0=v0, n_points=20)
        
        assert 'geodesic' in result
        assert 'trajectory_3d' in result
        assert 'success' in result
    
    def test_generate_visualizations(self, jones, tmp_path):
        """Test visualization generation."""
        # Use temporary directory for output
        plots = jones.generate_visualizations(output_dir=str(tmp_path))
        
        # Should generate multiple plots
        assert isinstance(plots, dict)
        assert len(plots) >= 2  # At minimum heatmap and page curve
        
        # Check that files were created
        for plot_type, filename in plots.items():
            filepath = Path(filename)
            assert filepath.exists(), f"Plot file {filename} should exist"
    
    def test_generate_full_analysis(self, jones):
        """Test full analysis generation."""
        results = jones.generate_full_analysis()
        
        assert 'spectral_analysis' in results
        assert 'islands' in results
        assert 'page_curve' in results
        assert 'geodesic_flow' in results
        
        # Check that all components ran
        assert results['spectral_analysis']['spectral_gap_kappa'] > 0
        assert isinstance(results['islands'], list)
        assert results['page_curve']['max_entropy'] >= 0


class TestIntegration:
    """Integration tests for the complete framework."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from initialization to analysis."""
        # Initialize framework
        jones = JonesQuantumGravityResolution(dimension=27)
        
        # Run spectral analysis
        spectral = jones.analyze_spectral_structure()
        assert spectral['spectral_gap_kappa'] > 0
        
        # Find islands
        islands = jones.find_entanglement_islands(tolerance=0.5)
        assert isinstance(islands, list)
        
        # Compute Page curve
        page = jones.compute_page_curve(n_points=30)
        assert len(page['x']) == 30
        assert page['max_entropy'] >= 0
        
        # Compute geodesic
        x0 = np.array([0.5, 0.5, 0.5])
        v0 = np.array([0.1, 0.0, 0.0])
        geodesic = jones.compute_geodesic_flow(x0=x0, v0=v0, n_points=20)
        assert 'trajectory_3d' in geodesic
    
    def test_parameter_variations(self):
        """Test that framework works with different parameters."""
        # Test different contraction strengths
        for strength in [0.5, 1.0, 2.0]:
            jones = JonesQuantumGravityResolution(
                dimension=27,
                contraction_strength=strength
            )
            result = jones.analyze_spectral_structure()
            assert result['spectral_gap_kappa'] > 0
        
        # Test different rotation angles
        for angle in [np.pi/12, np.pi/6, np.pi/4]:
            jones = JonesQuantumGravityResolution(
                dimension=27,
                rotation_angle=angle
            )
            result = jones.analyze_spectral_structure()
            assert result['spectral_gap_kappa'] > 0
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        jones1 = JonesQuantumGravityResolution(dimension=27)
        result1 = jones1.analyze_spectral_structure()
        
        np.random.seed(42)
        jones2 = JonesQuantumGravityResolution(dimension=27)
        result2 = jones2.analyze_spectral_structure()
        
        # Results should be identical with same seed
        assert np.isclose(result1['spectral_gap_kappa'], result2['spectral_gap_kappa'])


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
