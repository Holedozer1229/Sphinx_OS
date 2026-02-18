#!/usr/bin/env python3
"""
Test suite for Jones Quantum Pirate Miner
Tests core functionality without requiring display
"""

import sys
import os
import unittest
import numpy as np

# Set SDL to dummy driver for headless testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Import game module
import quantum_pirate_miner as qpm


class TestUFTDerivation(unittest.TestCase):
    """Test UFT (Unified Field Theory) derivation functions"""
    
    def test_compute_schmidt(self):
        """Test Schmidt decomposition"""
        matrix = np.array([[1, 0.5], [0.5, 1]], dtype=float)
        lambdas = qpm.compute_schmidt(matrix)
        
        # Lambdas should be normalized
        self.assertAlmostEqual(np.sum(lambdas**2), 1.0, places=5)
        # Should have positive values
        self.assertTrue(np.all(lambdas >= 0))
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation"""
        lambdas = np.array([0.6, 0.8])
        lambdas /= np.linalg.norm(lambdas)
        
        entropy = qpm.entanglement_entropy(lambdas)
        
        # Entropy should be non-negative
        self.assertGreaterEqual(entropy, 0.0)
        # Should be finite
        self.assertTrue(np.isfinite(entropy))
    
    def test_seven_fold_warp(self):
        """Test seven-fold warp calculation"""
        lambdas = np.array([0.5, 0.5, 0.5])
        lambdas /= np.linalg.norm(lambdas)
        
        w7 = qpm.seven_fold_warp(lambdas)
        
        # Should be positive (abs value)
        self.assertGreaterEqual(w7, 0.0)
        # Should be finite
        self.assertTrue(np.isfinite(w7))
    
    def test_warp_integral(self):
        """Test warp integral"""
        matrix = np.array([[1, 0.5], [0.5, 1]], dtype=float)
        w7 = 2.0
        
        integral = qpm.warp_integral(matrix, w7)
        
        # Should be proportional to trace
        expected = w7 * np.trace(matrix)
        self.assertAlmostEqual(integral, expected, places=5)
    
    def test_inertial_mass_reduction(self):
        """Test inertial mass reduction"""
        m_reduced = qpm.inertial_mass_reduction(m=1.0, t=0.0)
        
        # Should be positive
        self.assertGreater(m_reduced, 0.0)
        # Should not exceed original mass
        self.assertLessEqual(m_reduced, 1.0)
    
    def test_derive_uft(self):
        """Test complete UFT derivation"""
        result = qpm.derive_uft(0.0)
        
        # Should have all required keys
        self.assertIn("entropy_S", result)
        self.assertIn("warp_W7", result)
        self.assertIn("integral_I", result)
        self.assertIn("m_reduced", result)
        
        # All values should be finite
        for key, value in result.items():
            self.assertTrue(np.isfinite(value), f"{key} is not finite: {value}")


class TestJ4Function(unittest.TestCase):
    """Test J4 (Jones 4-fold modular flow)"""
    
    def test_j4_basic(self):
        """Test J4 basic functionality"""
        result = qpm.J4(1.0)
        
        # Should be finite
        self.assertTrue(np.isfinite(result))
    
    def test_j4_zero(self):
        """Test J4 at zero"""
        result = qpm.J4(0.0)
        
        # At zero, should be zero
        self.assertAlmostEqual(result, 0.0, places=5)


class TestEntanglementEngine(unittest.TestCase):
    """Test entanglement engine"""
    
    def setUp(self):
        """Set up test engine"""
        self.engine = qpm.EntanglementEngine(width=64, height=48)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.width, 64)
        self.assertEqual(self.engine.height, 48)
        self.assertEqual(len(self.engine.pairs), 0)
        self.assertEqual(self.engine.wormholes_active, 0)
    
    def test_create_epr_pair(self):
        """Test EPR pair creation"""
        pair = self.engine.create_epr_pair(
            0, 0,
            (10, 20),
            (30, 40)
        )
        
        # Pair should be entangled
        self.assertTrue(pair.entangled)
        # Should have correct coordinates
        self.assertEqual(pair.coord_time, (10, 20))
        self.assertEqual(pair.coord_space, (30, 40))
        # Wormholes should increase
        self.assertEqual(self.engine.wormholes_active, 1)
        # Graph nodes should be created
        self.assertEqual(len(self.engine.graph_nodes), 2)
    
    def test_collapse_pair(self):
        """Test EPR pair collapse"""
        pair = self.engine.create_epr_pair(0, 0, (10, 20), (30, 40))
        
        self.engine.collapse(pair)
        
        # Pair should no longer be entangled
        self.assertFalse(pair.entangled)
        # Wormholes should decrease
        self.assertEqual(self.engine.wormholes_active, 0)
    
    def test_create_ghz_state(self):
        """Test GHZ state creation"""
        ghz = self.engine.create_ghz_state(num_qubits=3)
        
        # Should have correct number of qubits
        self.assertEqual(len(ghz.qubits), 3)
        # Should have full coherence initially
        self.assertEqual(ghz.coherence, 1.0)
        # Should be in the list
        self.assertIn(ghz, self.engine.ghz_states)
    
    def test_update(self):
        """Test engine update"""
        ghz = self.engine.create_ghz_state(num_qubits=3)
        initial_coherence = ghz.coherence
        
        # Update for 1 second
        self.engine.update(1.0)
        
        # Coherence should decrease
        self.assertLess(ghz.coherence, initial_coherence)


class TestTreasureMap(unittest.TestCase):
    """Test treasure map"""
    
    def setUp(self):
        """Set up test map"""
        self.engine = qpm.EntanglementEngine()
        self.tmap = qpm.TreasureMap(self.engine, width=64, height=48)
    
    def test_initialization(self):
        """Test map initialization"""
        self.assertEqual(self.tmap.width, 64)
        self.assertEqual(self.tmap.height, 48)
        # Should have generated treasures
        self.assertEqual(len(self.tmap.treasures), 30)
        # Phase field should have correct shape
        self.assertEqual(self.tmap.phase_field.shape, (48, 64))
    
    def test_treasure_properties(self):
        """Test treasure properties"""
        for treasure in self.tmap.treasures:
            # Should have all required keys
            self.assertIn("id", treasure)
            self.assertIn("x", treasure)
            self.assertIn("y", treasure)
            self.assertIn("value", treasure)
            self.assertIn("rarity", treasure)
            self.assertIn("color", treasure)
            self.assertIn("collected", treasure)
            
            # Position should be within bounds
            self.assertGreaterEqual(treasure["x"], 0)
            self.assertLess(treasure["x"], 64)
            self.assertGreaterEqual(treasure["y"], 0)
            self.assertLess(treasure["y"], 48)
            
            # Value should be positive
            self.assertGreater(treasure["value"], 0)
            
            # Should not be collected initially
            self.assertFalse(treasure["collected"])
    
    def test_get_treasure_at(self):
        """Test getting treasure at position"""
        # Get first treasure
        treasure = self.tmap.treasures[0]
        x, y = treasure["x"], treasure["y"]
        
        # Should find it
        found = self.tmap.get_treasure_at(x, y)
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], treasure["id"])
        
        # Should not find at wrong position
        not_found = self.tmap.get_treasure_at(100, 100)
        self.assertIsNone(not_found)
    
    def test_collect_treasure(self):
        """Test treasure collection"""
        treasure = self.tmap.treasures[0]
        tid = treasure["id"]
        expected_value = treasure["value"]
        
        # Collect it
        value = self.tmap.collect(tid)
        
        # Should return correct value
        self.assertAlmostEqual(value, expected_value, places=5)
        
        # Should be marked as collected
        self.assertTrue(treasure["collected"])
        
        # Should not find it anymore
        found = self.tmap.get_treasure_at(treasure["x"], treasure["y"])
        self.assertIsNone(found)
        
        # Collecting again should return 0
        value2 = self.tmap.collect(tid)
        self.assertEqual(value2, 0.0)


class TestECS(unittest.TestCase):
    """Test Entity Component System"""
    
    def setUp(self):
        """Reset ECS state"""
        qpm._entities.clear()
        qpm._components.clear()
        qpm._systems.clear()
        qpm._next_entity_id = 0
    
    def test_create_entity(self):
        """Test entity creation"""
        eid1 = qpm.create_entity()
        eid2 = qpm.create_entity()
        
        # Should have unique IDs
        self.assertNotEqual(eid1, eid2)
        # Should be in entities dict
        self.assertIn(eid1, qpm._entities)
        self.assertIn(eid2, qpm._entities)
    
    def test_add_get_component(self):
        """Test adding and getting components"""
        eid = qpm.create_entity()
        pos = qpm.Position(10.0, 20.0)
        
        qpm.add_component(eid, pos)
        
        # Should be able to get it back
        retrieved = qpm.get(eid, qpm.Position)
        self.assertEqual(retrieved.x, 10.0)
        self.assertEqual(retrieved.y, 20.0)
    
    def test_query(self):
        """Test component querying"""
        # Create entities with different components
        e1 = qpm.create_entity()
        qpm.add_component(e1, qpm.Position(1, 2))
        qpm.add_component(e1, qpm.Player())
        
        e2 = qpm.create_entity()
        qpm.add_component(e2, qpm.Position(3, 4))
        
        e3 = qpm.create_entity()
        qpm.add_component(e3, qpm.Player())
        
        # Query for Position
        pos_entities = qpm.query(qpm.Position)
        self.assertEqual(len(pos_entities), 2)
        self.assertIn(e1, pos_entities)
        self.assertIn(e2, pos_entities)
        
        # Query for Player
        player_entities = qpm.query(qpm.Player)
        self.assertEqual(len(player_entities), 2)
        self.assertIn(e1, player_entities)
        self.assertIn(e3, player_entities)
        
        # Query for both Position and Player
        both_entities = qpm.query(qpm.Position, qpm.Player)
        self.assertEqual(len(both_entities), 1)
        self.assertIn(e1, both_entities)
    
    def test_system_registration_and_tick(self):
        """Test system registration and execution"""
        call_count = [0]
        
        def test_system(dt):
            call_count[0] += 1
        
        qpm.register_system(test_system)
        
        # Tick should call the system
        qpm.tick(0.016)
        self.assertEqual(call_count[0], 1)
        
        qpm.tick(0.016)
        self.assertEqual(call_count[0], 2)


class TestDataClasses(unittest.TestCase):
    """Test data classes"""
    
    def test_epr_pair(self):
        """Test EPRPair dataclass"""
        pair = qpm.EPRPair(
            time_id=1,
            space_id=2,
            coord_time=(10, 20),
            coord_space=(30, 40)
        )
        
        self.assertEqual(pair.time_id, 1)
        self.assertEqual(pair.space_id, 2)
        self.assertTrue(pair.entangled)
        self.assertFalse(pair.collapsed_time)
    
    def test_graph_node(self):
        """Test GraphNode dataclass"""
        node = qpm.GraphNode(
            id=1,
            position=(10, 20),
            entangled_with=[2, 3]
        )
        
        self.assertEqual(node.id, 1)
        self.assertEqual(node.position, (10, 20))
        self.assertEqual(len(node.entangled_with), 2)
        self.assertIsNone(node.collapse_time)
    
    def test_temporal_ghz(self):
        """Test TemporalGHZ dataclass"""
        ghz = qpm.TemporalGHZ(
            qubits=[1, 2, 3],
            phase=1.57
        )
        
        self.assertEqual(len(ghz.qubits), 3)
        self.assertAlmostEqual(ghz.phase, 1.57, places=2)
        self.assertEqual(ghz.coherence, 1.0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUFTDerivation))
    suite.addTests(loader.loadTestsFromTestCase(TestJ4Function))
    suite.addTests(loader.loadTestsFromTestCase(TestEntanglementEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTreasureMap))
    suite.addTests(loader.loadTestsFromTestCase(TestECS))
    suite.addTests(loader.loadTestsFromTestCase(TestDataClasses))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
