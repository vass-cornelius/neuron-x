
import unittest
import networkx as nx
import numpy as np
import json
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuron_x import NeuronX

class TestBiologicalAnchor(unittest.TestCase):
    def setUp(self):
        with patch('neuron_x.SentenceTransformer') as mock_encoder, \
             patch('neuron_x.CrossEncoder'), \
             patch('neuron_x.os.makedirs'), \
             patch('neuron_x.os.path.exists', return_value=False), \
             patch('neuron_x.NeuronX.save_graph'): # Mock save_graph
            
            # Setup encoder to return generic vectors
            mock_encoder.return_value.encode.side_effect = lambda x: np.array([0.1, 0.1, 0.1])
            
            self.neuron = NeuronX(persistence_path="./test_memory_vault_bio")
            self.neuron.graph = nx.DiGraph()
            self.neuron.graph.add_node("Self", type="Identity")
            
            # Setup vector cache
            self.neuron.vector_cache = {
                "Noah": np.array([1.0, 0.0, 0.0]), # Match query for "Noah"
                "MentorX": np.array([1.0, 0.0, 0.0])
            }

    def test_biological_priority(self):
        """Ensure Biological relations outrank stronger Social relations."""
        # Setup:
        # Noah is Child (Biological) but weight 1.0
        # MentorX is Mentor (Social) but weight 1.5
        
        # 1. Add Nodes
        self.neuron.graph.add_node("Noah")
        self.neuron.graph.add_node("MentorX")
        
        # 2. Add Edges
        # Self -> Noah (child_of) w=1.0. With boost (x2) -> 2.0
        self.neuron.graph.add_edge("Self", "Noah", relation="parent_of", weight=1.0)
        
        # Self -> MentorX (mentors) w=1.5. No boost -> 1.5
        self.neuron.graph.add_edge("Self", "MentorX", relation="mentors", weight=1.5)
        
        # 3. Retrieve
        # We search for "Family" or just generic query that pulls both.
        # Since we mocked encoder to always return same vector, all nodes are candidates.
        # We need to ensure _get_relevant_memories ranks them.
        
        # Trick: We need the nodes to be retrieved first. 
        # The retrieval logic strictly uses vector sim first.
        # Both have same vector cache in my mock, so they tie on similarity.
        # Then it extracts context.
        # Context extraction sorts triples by weight.
        
        with patch.object(self.neuron.encoder, 'encode', return_value=np.array([1.0, 0.0, 0.0])):
            context = self.neuron._get_relevant_memories("query", top_k=5)
            
        print("\nContext Retrieved:")
        for line in context:
            print(line)
            
        # We expect Noah to appear BEFORE MentorX in the context list because 2.0 > 1.5
        # The list returned is a list of strings "Triples".
        
        found_noah_idx = -1
        found_mentor_idx = -1
        
        for i, line in enumerate(context):
            if "parent_of" in line and "Noah" in line:
                found_noah_idx = i
            if "mentors" in line and "MentorX" in line:
                found_mentor_idx = i
                
        self.assertNotEqual(found_noah_idx, -1, "Noah should be found")
        self.assertNotEqual(found_mentor_idx, -1, "MentorX should be found")
        
        # Check Ranking
        self.assertLess(found_noah_idx, found_mentor_idx, "Biological Noah should be ranked higher than MentorX")

if __name__ == '__main__':
    unittest.main()
