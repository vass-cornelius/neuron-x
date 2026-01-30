
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import networkx as nx
import random

# Mock heavy dependencies
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["models"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["rich.logging"] = MagicMock()
sys.modules["dotenv"] = MagicMock() 

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the class to test
import neuron_x
from neuron_x import NeuronX

class TestLoopFix(unittest.TestCase):
    def setUp(self):
        # Prevent logging errors by mocking the module-level logger
        neuron_x.logger = MagicMock()
        
        # Patch __init__ to avoid loading everything
        with patch.object(NeuronX, '__init__', return_value=None):
            self.neuron = NeuronX()
            # Manually set up attributes needed for generate_proactive_thought
            self.neuron.llm_client = MagicMock()
            self.neuron.path = "/tmp/test"
            self.neuron.graph = nx.DiGraph()
            self.neuron.focus_history = []
            self.neuron.MAX_FOCUS_HISTORY = 5
            self.neuron.thought_buffer = []
            self.neuron.MAX_THOUGHT_RECURSION = 5
            self.neuron.goals = []
            self.neuron.vector_cache = {}
            self.neuron.encoder = MagicMock()
            self.neuron.encoder.encode.return_value = [0.1, 0.2] # Dummy vector
            
            # Mock methods called inside generate_proactive_thought
            self.neuron._sync_with_disk = MagicMock()
            self.neuron.get_bg_goal = MagicMock(return_value=None)
            self.neuron.get_identity_summary = MagicMock(return_value="Summary")
            self.neuron._get_relevant_memories = MagicMock(return_value=[])
            self.neuron._validate_thought = MagicMock(return_value=True)
            self.neuron._broadcast_thought = MagicMock()
            self.neuron._check_for_loops = MagicMock(return_value=False)
            
            # Mock LLM response
            response = MagicMock()
            response.text = "Mock thought"
            self.neuron.llm_client.models.generate_content.return_value = response

    def test_dissonance_loop_prevention(self):
        # Setup: One dissonant node
        self.neuron.graph.add_node("ConflictA", status="DISSONANT")
        self.neuron.graph.add_node("NormalB")
        
        # 1. First Call: Should pick ConflictA
        # Implementation Detail: internal logic uses random.choice on dissonant_nodes
        # We need to ensure random.choice works deterministically or we check the result
        
        # Override random to ensure if it picks, it picks ours (only one choice anyway)
        
        processed_thought = self.neuron.generate_proactive_thought()
        
        # Verify history was updated
        self.assertIn("ConflictA", self.neuron.focus_history)
        print(f"History after run 1: {self.neuron.focus_history}")
        
        # 2. Second Call: Should NOT pick ConflictA because it is in history
        # It should fall back to Wandering -> NormalB
        # (Assuming NormalB is not in history)
        
        processed_thought_2 = self.neuron.generate_proactive_thought()
        
        self.assertIn("NormalB", self.neuron.focus_history)
        # ConflictA should still be in history (at pos 0)
        self.assertEqual(self.neuron.focus_history[0], "ConflictA")
        self.assertEqual(self.neuron.focus_history[1], "NormalB")
        
        print(f"History after run 2: {self.neuron.focus_history}")

if __name__ == '__main__':
    unittest.main()
