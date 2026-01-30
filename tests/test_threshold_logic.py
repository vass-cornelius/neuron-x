import unittest
import sys
import os
import networkx as nx

# Add parent directory to path to import neuron_x
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuron_x import NeuronX

class TestThresholdLogic(unittest.TestCase):
    def setUp(self):
        self.brain = NeuronX(persistence_path="test_threshold_vault")
        self.brain.graph = nx.DiGraph()
        # Initialize Self node
        self.brain._initialize_self_node()
        
    def tearDown(self):
        import shutil
        if os.path.exists("test_threshold_vault"):
            shutil.rmtree("test_threshold_vault")

    def calculate_threshold(self, reinforcement, entropy):
        # Helper to isolate the logic being tested
        # This mirrors the logic we plan to implement in neuron_x.py
        # BUT for the test to pass on the *modified* code, we should call the actual method.
        # However, the method is embedded in _get_relevant_memories.
        # We can simulate the state and check the debug logs? 
        # Or better, we can extract the logic or just reimplement the check here if we want to unit test the formula.
        # Ideally, we test the actual method. But _get_relevant_memories returns memories, it doesn't return the threshold.
        # We can inspect the logs or just trust the math if valid.
        
        # Let's subclass or monkeypatch for testing?
        # Or just manually set the values on the Self node and run the logic snippet?
        # Let's write a targeted test that runs a modified version of the logic 
        # OR better yet, we can modify the test to *access* the logic if we extract it.
        # For now, let's just forcefully run the logic block as it would appear in the code.
        
        # Actually, let's test the EFFECT.
        # If threshold is high (0.4), weak matches should be filtered.
        # If threshold is low (0.15), weak matches should appear.
        return 0

    def test_threshold_logic_direct(self):
        """
        Directly validates the mathematical formula we want to implement.
        This effectively tests the 'specification'.
        """
        # Proposed Formula: ratio = entropy / (reinforcement + 1.0)
        # threshold = max(0.15, 0.4 - (ratio * 0.25))

        def logic(r, e):
            ratio = e / (r + 1.0)
            return max(0.15, 0.4 - (ratio * 0.25))

        # Case 1: Initial State (0, 0)
        # Ratio = 0 / 1 = 0
        # Threshold = 0.4 - 0 = 0.4
        t1 = logic(0.0, 0.0)
        self.assertAlmostEqual(t1, 0.4, places=2, msg="Initial state should have high threshold")

        # Case 2: Stagnation (0, 10.0)
        # Ratio = 10 / 1 = 10
        # Threshold = 0.4 - 2.5 = -2.1 -> clamped to 0.15
        t2 = logic(0.0, 10.0)
        self.assertEqual(t2, 0.15, "High entropy should collapse threshold to floor")

        # Case 3: Active Learning (10, 1)
        # Ratio = 1 / 11 = 0.09
        # Threshold = 0.4 - 0.02 = 0.38
        t3 = logic(10.0, 1.0)
        self.assertTrue(t3 > 0.35, "Active learning should maintain high standards")

    def test_current_behavior_failure(self):
        """
        This test is designed to FAIL on the current codebase,
        proving that the current logic is flawed (ratio=1.0 when r=0).
        """
        # Set brain state to 0,0
        self.brain.graph.nodes["Self"]["reinforcement_sum"] = 0.0
        self.brain.graph.nodes["Self"]["entropy_sum"] = 0.0
        
        # Disable CrossEncoder to isolate Threshold Logic check
        self.brain.cross_encoder = None
        
        # We need to expose the internal threshold variable or verify its effect.
        # Since we can't easily see local vars, let's verify if a weak memory is retrieved.
        # We'll use a controlled vectors setup.
        
        # Query Vector: [1, 0, 0]
        # Memory Vector: [0.3, 0.9, 0] -> Sim approx 0.3 (Cos sim)
        
        # Current Logic (0,0): Ratio=1.0 -> Threshold = 0.4 - 0.25 = 0.15.
        # So a 0.3 sim memory WOULD be retrieved.
        
        # New Logic (0,0): Threshold = 0.4.
        # So a 0.3 sim memory WOULD NOT be retrieved.
        
        self.brain.encoder.encode = lambda x: [1.0, 0.0, 0.0] if x == "query" else [0.3, 0.9, 0.0]
        
        # Setup Cache AND Graph (Critical to avoid KeyError)
        self.brain.vector_cache = {
            "Memory_Weak": [0.3, 0.9, 0.0], # Sim = 0.316
            "Self": [0.0, 0.0, 0.0]
        }
        
        # Bypass graph checks
        self.brain.graph.add_node("Memory_Weak", content="Weak Stuff", status="ACTIVE")
        
        # Run retrieval
        results = self.brain._get_relevant_memories("query", top_k=1)
        
        # IF Fixed: Results should be empty (0.3 < 0.4)
        # IF Broken: Results should have it (0.3 > 0.15)
        
        print(f"DEBUG: Retrieved {len(results)} memories.")
        self.assertEqual(len(results), 0, "Should NOT retrieve weak memories in initial state (0,0)")

if __name__ == '__main__':
    unittest.main()
