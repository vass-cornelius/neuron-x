
import sys
import os
import json
import networkx as nx
import logging
from typing import Dict, Any

# Mocking necessary parts to test logic without full dependencies
# We want to test the `consolidate` logic specifically.

# Stubbing the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEURON-X-TEST")

class MockEncoder:
    def encode(self, text):
        return [0.1, 0.2, 0.3] # Dummy vector
    def tolist(self):
        return [0.1, 0.2, 0.3]

class NeuronXTest:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.working_memory = []
        self.encoder = MockEncoder()
        self.llm_client = None # Not needed for this test
        self.vector_cache = {}
        self.files_dir = "./test_data"
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)
        self.graph_file = os.path.join(self.files_dir, "test_graph.gexf")
    
    def _sync_with_disk(self):
        pass

    def save_graph(self):
        pass
    
    def _rebuild_vector_cache(self):
        pass

    def _extract_triples_batch(self, memories):
        return None # Force fallback or manual injection for test

    def _extract_semantic_triples(self, text):
        # We will bypass this by injecting triples directly into working_memory 
        # or by mocking the extraction result if we were testing extraction.
        # But here we want to test CONSOLIDATION logic.
        # The consolidate function re-extracts. 
        # So we need to mock this method to return what we want.
        return []

    # Copying the relevant parts of consolidate from neuron_x.py
    # NOTE: In a real scenario, we would import the class. 
    # But since I modified the file on disk, I can try to import it!
    # However, imports might be messy with dependencies.
    # Let's try to import the actual module first, if possible.
    pass

# Strategy Change: Import the actual NeuronX class and mock its dependencies.
# This ensures we test the ACTUAL code I just modified.

sys.path.append(os.getcwd())
try:
    from neuron_x import NeuronX
except ImportError:
    print("Could not import NeuronX. Make sure you are in the project root.")
    sys.exit(1)

# Mocking the heavy dependencies
import unittest
from unittest.mock import MagicMock

class TestPromotion(unittest.TestCase):
    def setUp(self):
        # Initialize NeuronX with mocked components
        self.bot = NeuronX()
        self.bot.llm_client = None
        self.bot.encoder = MagicMock()
        self.bot.encoder.encode.return_value = MagicMock()
        self.bot.encoder.encode.return_value.tolist.return_value = [0.1]*384
        self.bot.cross_encoder = None
        
        # Reset Graph
        self.bot.graph = nx.DiGraph()

    def test_hypothesis_promotion(self):
        print("\n--- Testing Hypothesis Promotion ---")
        # 1. Setup Semantic Hypothesis
        subj = "User"
        obj = "Piano"
        self.bot.graph.add_node(subj)
        self.bot.graph.add_node(obj)
        self.bot.graph.add_edge(subj, obj, relation="plays", weight=0.2, category="HYPOTHESIS")
        
        print(f"Initial State: {self.bot.graph[subj][obj]}")
        
        # 2. Simulate Factual Reinforcement
        # formatting memory to be extracted
        # Since we can't easily mock the internal _extract_semantic_triples regexes heavily,
        # We will inject the processing logic by mocking the extraction method to return our target triple.
        
        target_triple = [{"subject": subj, "predicate": "plays", "object": obj, "category": "FACTUAL", "source": "User_Interaction"}]
        
        # We mock _extract_triples_batch to return our desired triple
        # The logic in consolidate calls _extract_triples_batch first if llm_client exists.
        # But llm_client is None. So it falls back to regex.
        # We need to mock _extract_semantic_triples too or just ensure our logic flows.
        
        # EASIER: We can just use the internal logic or mock the extraction.
        self.bot._extract_triples_batch = MagicMock(return_value=target_triple)
        self.bot.llm_client = True # Enable the batch path
        
        self.bot.working_memory = [{"text": "I play the Piano.", "source": "User_Interaction", "vector": [0.1]*384}]
        
        # Run Consolidate
        self.bot.consolidate()
        
        # 3. Verify
        edge_data = self.bot.graph[subj][obj]
        print(f"Final State: {edge_data}")
        
        self.assertEqual(edge_data['category'], "FACTUAL", "Category should be promoted to FACTUAL")
        self.assertTrue(edge_data['weight'] > 0.2, "Weight should increase")

    def test_node_reawakening(self):
        print("\n--- Testing Node Re-Awakening ---")
        # 1. Setup Rejected Node
        node_name = "Unicorn"
        self.bot.graph.add_node(node_name, status="REJECTED")
        
        print(f"Initial Status: {self.bot.graph.nodes[node_name].get('status')}")
        
        # 2. Reinforce with FACTUAL evidence
        target_triple = [{"subject": node_name, "predicate": "is_a", "object": "MythicalCreature", "category": "FACTUAL", "source": "User_Interaction"}]
        
        self.bot._extract_triples_batch = MagicMock(return_value=target_triple)
        self.bot.llm_client = True 
        
        self.bot.working_memory = [{"text": "A Unicorn is a mythical creature.", "source": "User_Interaction", "vector": [0.1]*384}]
        
        # Run Consolidate
        self.bot.consolidate()
        
        # 3. Verify
        node_data = self.bot.graph.nodes[node_name]
        print(f"Final Status: {node_data.get('status')}")
        
        self.assertIsNone(node_data.get('status'), "Status should be cleared (None)")

if __name__ == '__main__':
    unittest.main()
