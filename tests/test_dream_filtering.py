
import unittest
import networkx as nx
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuron_x import NeuronX

class TestDreamFiltering(unittest.TestCase):
    def setUp(self):
        with patch('neuron_x.SentenceTransformer'), \
             patch('neuron_x.CrossEncoder'), \
             patch('neuron_x.os.makedirs'), \
             patch('neuron_x.os.path.exists', return_value=False), \
             patch('neuron_x.NeuronX.save_graph'):
            
            self.neuron = NeuronX(persistence_path="./test_memory_vault_dream")
            self.neuron.llm_client = MagicMock() # Needs LLM client to dream
            self.neuron.graph = nx.DiGraph()
            self.neuron.graph.add_node("Self", type="Identity")

    def test_dream_excludes_rejected(self):
        """Ensure REJECTED nodes are never selected for dreaming."""
        # Add normal nodes
        for i in range(5):
            self.neuron.graph.add_node(f"Node_{i}", status="ACTIVE")
        
        # Add a rejected node
        self.neuron.graph.add_node("Legacy_Raphael", status="REJECTED")
        
        # Mock random to ensure we'd pick it if it were in the list
        # But we can't easily force random to pick from a list that shouldn't contain it.
        # Instead, we'll verify the list comprehension logic directly by inspecting the function's scope if possible, 
        # or better: we just run it and assert 'Legacy_Raphael' is NEVER the subject or object.
        
        # Actually, let's just inspect the logic by mocking the internal selection or simply running it
        # and checking the call args to llm_client.
        
        # Better yet, let's force the graph to have ONLY rejected nodes + 2 valid nodes.
        # Then we ensure only the valid ones are picked.
        
        self.neuron.graph = nx.DiGraph()
        self.neuron.graph.add_node("Self")
        self.neuron.graph.add_node("Valid_A")
        self.neuron.graph.add_node("Valid_B")
        self.neuron.graph.add_node("Valid_C")
        self.neuron.graph.add_node("Valid_D") 
        self.neuron.graph.add_node("Valid_E") # Need at least 5 nodes total for dream to trigger
        
        self.neuron.graph.add_node("Rejected_X", status="REJECTED")
        self.neuron.graph.add_node("Rejected_Y", status="REJECTED")

        # Trace the LLM call
        self.neuron.llm_client.models.generate_content.return_value.text = '{"predicate": "linked", "reasoning": "test"}'
        
        # Act
        self.neuron._dream_cycle()
        
        # Assert
        # Check carefully if we can inspect what was chosen. 
        # The prompt contains the chosen nodes.
        call_args = self.neuron.llm_client.models.generate_content.call_args
        if call_args:
            prompt_content = call_args[1]['contents']
            print(f"Dream Prompt: {prompt_content}")
            self.assertNotIn("Rejected_X", prompt_content)
            self.assertNotIn("Rejected_Y", prompt_content)
        else:
            self.fail("LLM was not called, maybe not enough nodes?")

if __name__ == '__main__':
    unittest.main()
