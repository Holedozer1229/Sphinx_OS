# tests/test_spin_network.py
import unittest
import numpy as np
from sphinx_os.core.spin_network import SpinNetwork
from sphinx_os.utils.constants import CONFIG

class TestSpinNetwork(unittest.TestCase):
    def test_initialization(self):
        grid_size = (2, 2, 2, 2, 2, 2)
        network = SpinNetwork(grid_size)
        self.assertEqual(network.grid_size, grid_size)
        self.assertEqual(network.total_points, np.prod(grid_size))
        self.assertAlmostEqual(np.linalg.norm(network.state.flatten()), 1.0)

    def test_evolve(self):
        grid_size = (2, 2, 2, 2, 2, 2)
        network = SpinNetwork(grid_size)
        lambda_field = np.zeros(grid_size)
        metric = np.array([np.eye(6)] * np.prod(grid_size)).reshape(*grid_size, 6, 6)
        inverse_metric = metric
        deltas = [CONFIG[f"d{dim}"] for dim in ['t', 'x', 'x', 'x', 'v', 'u']]
        phi_N = np.zeros(grid_size)
        higgs_field = np.ones(grid_size) * CONFIG["vev_higgs"]
        em_fields = {"A": np.zeros((*grid_size, 6)), "F": np.zeros((*grid_size, 6, 6)),
                     "J": np.zeros((*grid_size, 6)), "J4": np.zeros(grid_size)}
        electron_field = np.zeros((*grid_size, 4))
        quark_field = np.zeros((*grid_size, 2, 3, 4))
        steps = network.evolve(1e-12, lambda_field, metric, inverse_metric, deltas, phi_N,
                               higgs_field, em_fields, electron_field, quark_field)
        self.assertGreater(steps, 0)
        self.assertAlmostEqual(np.linalg.norm(network.state.flatten()), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
