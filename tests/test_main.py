# tests/test_main.py
import unittest
from sphinx_os.main import SphinxOS

class TestSphinxOS(unittest.TestCase):
    def test_initialization(self):
        os = SphinxOS()
        self.assertEqual(os.grid_size, (5, 5, 5, 5, 3, 3))
        self.assertTrue(hasattr(os, 'anubis_core'))
        self.assertTrue(hasattr(os, 'toe'))

    def test_emulate_on_hardware(self):
        os = SphinxOS()
        result = os.emulate_on_hardware()
        self.assertIn('counts', result)
        self.assertIn('fidelity', result)
        self.assertIn('S', result)
        self.assertEqual(sum(result['counts'].values()), 1024)
        self.assertTrue(abs(result['S']) > 2)  # Should violate CHSH

if __name__ == '__main__':
    unittest.main()
