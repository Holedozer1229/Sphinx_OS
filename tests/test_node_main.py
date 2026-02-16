# tests/test_node_main.py
"""
Unit tests for SphinxSkynet Node (node_main.py).
"""
import unittest
import numpy as np


class TestNode(unittest.TestCase):
    """Test suite for the Node class."""

    def setUp(self):
        """Set up test nodes."""
        from node_main import Node, NUM_NODES, ANCILLA_DIM
        self.node = Node(0)
        self.num_nodes = NUM_NODES
        self.ancilla_dim = ANCILLA_DIM

    def test_node_initialization(self):
        """Test Node initializes with correct shapes and properties."""
        self.assertEqual(self.node.id, 0)
        self.assertGreaterEqual(self.node.phi_total, 0)
        self.assertLessEqual(self.node.phi_total, 10)
        self.assertGreaterEqual(self.node.delta_lambda, 0)
        self.assertLessEqual(self.node.delta_lambda, 0.1)

    def test_hypercube_state_shape(self):
        """Test hypercube state is 12 faces x 50 layers."""
        self.assertEqual(self.node.hypercube_state.shape, (12, 50))

    def test_ancilla_state_shape(self):
        """Test ancilla projection tensor shape."""
        self.assertEqual(self.node.ancilla_state.shape, (12, 50, self.ancilla_dim))

    def test_full_state_shape(self):
        """Test combined state includes hypercube + ancilla dimensions."""
        expected_shape = (12, 50, 1 + self.ancilla_dim)
        self.assertEqual(self.node.full_state.shape, expected_shape)

    def test_density_matrix_symmetric(self):
        """Test entanglement density matrix is symmetric."""
        np.testing.assert_array_almost_equal(self.node.rho_s, self.node.rho_s.T)

    def test_density_matrix_normalized(self):
        """Test entanglement density matrix trace is 1."""
        self.assertAlmostEqual(np.trace(self.node.rho_s), 1.0, places=10)

    def test_density_matrix_shape(self):
        """Test density matrix has correct shape."""
        self.assertEqual(self.node.rho_s.shape, (self.num_nodes, self.num_nodes))


class TestTraversableAncillaryWormhole(unittest.TestCase):
    """Test suite for the TraversableAncillaryWormhole class."""

    def setUp(self):
        """Set up two nodes and a wormhole between them."""
        from node_main import Node, TraversableAncillaryWormhole, ALPHA, BETA
        self.node_i = Node(0)
        self.node_j = Node(1)
        self.wormhole = TraversableAncillaryWormhole(self.node_i, self.node_j)
        self.alpha = ALPHA
        self.beta = BETA

    def test_wormhole_initialization(self):
        """Test wormhole initializes with correct parameters."""
        self.assertIs(self.wormhole.node_i, self.node_i)
        self.assertIs(self.wormhole.node_j, self.node_j)
        self.assertEqual(self.wormhole.alpha, self.alpha)
        self.assertEqual(self.wormhole.beta, self.beta)
        self.assertIsNone(self.wormhole.metric)

    def test_compute_laplacian(self):
        """Test Laplacian computation returns a non-negative scalar."""
        laplacian = self.wormhole.compute_laplacian()
        self.assertIsInstance(laplacian, float)
        self.assertGreaterEqual(laplacian, 0.0)

    def test_compute_metric(self):
        """Test metric computation returns a finite number."""
        metric = self.wormhole.compute_metric()
        self.assertIsNotNone(metric)
        self.assertTrue(np.isfinite(metric))
        self.assertEqual(self.wormhole.metric, metric)

    def test_propagate_computes_metric_if_none(self):
        """Test propagate auto-computes metric when not yet computed."""
        self.assertIsNone(self.wormhole.metric)
        result = self.wormhole.propagate(1.0)
        self.assertIsNotNone(self.wormhole.metric)
        self.assertAlmostEqual(result, self.wormhole.metric)

    def test_propagate_scales_signal(self):
        """Test propagate scales input signal by the metric."""
        self.wormhole.compute_metric()
        signal = 2.5
        result = self.wormhole.propagate(signal)
        self.assertAlmostEqual(result, signal * self.wormhole.metric)

    def test_same_node_laplacian_zero(self):
        """Test Laplacian is zero when comparing a node to itself."""
        from node_main import TraversableAncillaryWormhole
        w = TraversableAncillaryWormhole(self.node_i, self.node_i)
        self.assertAlmostEqual(w.compute_laplacian(), 0.0)


class TestHyperZkProof(unittest.TestCase):
    """Test suite for the hyper zk-EVM proof stub."""

    def test_proof_generation_returns_true(self):
        """Test the proof stub always returns True."""
        from node_main import generate_hyperzk_proof, Node
        node = Node(0)
        self.assertTrue(generate_hyperzk_proof(node))


class TestAllNodes(unittest.TestCase):
    """Test suite for the global node list."""

    def test_all_nodes_count(self):
        """Test that all_nodes contains NUM_NODES nodes."""
        from node_main import all_nodes, NUM_NODES
        self.assertEqual(len(all_nodes), NUM_NODES)

    def test_all_nodes_unique_ids(self):
        """Test that all node IDs are unique and sequential."""
        from node_main import all_nodes
        ids = [node.id for node in all_nodes]
        self.assertEqual(ids, list(range(len(all_nodes))))


if __name__ == "__main__":
    unittest.main()
