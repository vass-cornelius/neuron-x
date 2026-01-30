import sys
import os
import unittest
import networkx as nx

# Add parent directory to path to import neuron_x
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuron_x import NeuronX

class TestUniversalEntropy(unittest.TestCase):
    def setUp(self):
        # Initialize NeuronX without LLM
        self.brain = NeuronX(persistence_path="test_memory_vault_entropy")
        # Clear graph for clean slate
        self.brain.graph = nx.DiGraph()
        self.brain.graph.add_node("Self", reinforcement_sum=0.0, entropy_sum=0.0, metabolic_fluidity=0.5)

    def tearDown(self):
        # Cleanup
        if os.path.exists("test_memory_vault_entropy"):
            import shutil
            shutil.rmtree("test_memory_vault_entropy")

    def test_universal_decay(self):
        # Create an edge with minimal weight
        self.brain.graph.add_edge("Self", "Test1", weight=1.0)
        
        # Manually trigger apply_entropy with standard params
        # Fluidity 0.5 -> Decay = 0.01 * 1.5 = 0.015
        self.brain._apply_entropy(reinforced_edges=[], fluidity=0.5, density=0.0)
        
        updated_weight = self.brain.graph["Self"]["Test1"]["weight"]
        self.assertTrue(updated_weight < 1.0, f"Weight should decrease: {updated_weight}")
        self.assertAlmostEqual(updated_weight, 1.0 - 0.015, places=3)

    def test_pruning(self):
        # Create an edge on the verge of death
        self.brain.graph.add_edge("Self", "Weak", weight=0.16)
        
        # Apply entropy. 0.16 - 0.015 = 0.145 < 0.15
        self.brain._apply_entropy(reinforced_edges=[], fluidity=0.5, density=0.0)
        
        self.assertFalse(self.brain.graph.has_edge("Self", "Weak"), "Edge should be pruned")

    def test_dynamic_floor_stiff(self):
        # Stiff Floor: Low density -> Floor 3.2
        # Edge at 3.3 should decay but STOP at 3.2
        self.brain.graph.add_edge("A", "B", weight=3.3)
        self.brain.graph.add_edge("B", "C", weight=3.2) # Should stay 3.2
        
        # Fluidity 0.0 -> Decay 0.01
        self.brain._apply_entropy(reinforced_edges=[], fluidity=0.0, density=0.0)
        
        w_ab = self.brain.graph["A"]["B"]["weight"]
        w_bc = self.brain.graph["B"]["C"]["weight"]
        
        # 3.3 - 0.01 = 3.29 (> 3.2, allowed)
        self.assertAlmostEqual(w_ab, 3.29, places=2)
        
        # 3.2 - 0.01 = 3.19 (< 3.2, clamped)
        self.assertEqual(w_bc, 3.2, "Weight should not drop below stiff floor")

    def test_dynamic_floor_fluid(self):
        # Fluid Floor: High Density -> Floor 2.5
        # Edge at 3.0 should decay
        
        # To get floor 2.5, we need density such that 3.2 - (d * 70) <= 2.5
        # 0.7 <= d * 70 => d >= 0.01
        
        self.brain.graph.add_edge("A", "B", weight=3.0)
        
        self.brain._apply_entropy(reinforced_edges=[], fluidity=0.0, density=0.01)
        
        w_ab = self.brain.graph["A"]["B"]["weight"]
        
        # Should decay below 3.0
        self.assertTrue(w_ab < 3.0, f"Weight should decay below 3.0: {w_ab}")

    def test_metabolic_fluidity(self):
        # High Fluidity -> High Decay rate
        self.brain.graph.add_edge("A", "B", weight=1.0)
        
        # Fluidity 2.0 -> Decay = 0.01 * 3.0 = 0.03
        self.brain._apply_entropy(reinforced_edges=[], fluidity=2.0, density=0.0)
        
        w_ab = self.brain.graph["A"]["B"]["weight"]
        self.assertAlmostEqual(w_ab, 0.97, places=3)

if __name__ == '__main__':
    unittest.main()
