
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

class TestEntityMerging(unittest.TestCase):
    def setUp(self):
        # Create a dummy NeuronX without loading real models on init if possible,
        # or mock the heavy components.
        # Since NeuronX loads models in __init__, we need to patch them.
        with patch('neuron_x.SentenceTransformer') as mock_encoder, \
             patch('neuron_x.CrossEncoder') as mock_cross_encoder, \
             patch('neuron_x.os.makedirs'), \
             patch('neuron_x.os.path.exists', return_value=False), \
             patch('neuron_x.NeuronX.save_graph'):  # Mock save_graph to prevent file writes
            
            mock_encoder.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            self.neuron = NeuronX(persistence_path="./test_memory_vault")
            self.neuron.cross_encoder = mock_cross_encoder.return_value
            # Reset graph
            self.neuron.graph = nx.DiGraph()
            self.neuron.graph.add_node("Self", type="Identity")

    def test_merge_typos_auto(self):
        """Test automatic merging for high vector sim + high name sim (Typos)"""
        # Node A: "Raphael"
        # Node B: "Rafael"
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.99, 0.05, 0.0]) # Very similar
        
        self.neuron.graph.add_node("Raphael", vector=json.dumps(vec_a.tolist()))
        self.neuron.graph.add_node("Rafael", vector=json.dumps(vec_b.tolist()))
        
        self.neuron.vector_cache = {
            "Raphael": vec_a,
            "Rafael": vec_b
        }
        
        # Act
        self.neuron._merge_similar_entities()
        
        # Assert
        # Should be merged into one. "Rafael" (alpha order) or "Raphael"? 
        # Logic says alphabetical usually wins if edges are equal, or the one with edges.
        nodes = list(self.neuron.graph.nodes())
        print(f"Nodes after merge: {nodes}")
        self.assertTrue("Raphael" in nodes or "Rafael" in nodes)
        self.assertFalse("Raphael" in nodes and "Rafael" in nodes)

    def test_merge_synonyms_cross_encoder(self):
        """Test merging via CrossEncoder for high vec sim + low name sim (Synonyms)"""
        # Node A: "The Boss"
        # Node B: "Der Chef" (German)
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.95, 0.1, 0.0]) # High semantic sim
        
        self.neuron.graph.add_node("The Boss", vector=json.dumps(vec_a.tolist()))
        self.neuron.graph.add_node("Der Chef", vector=json.dumps(vec_b.tolist()))
        
        self.neuron.vector_cache = {
            "The Boss": vec_a,
            "Der Chef": vec_b
        }
        
        # Mock CrossEncoder to say YES (score > 0.85)
        self.neuron.cross_encoder.predict.return_value = 0.90
        
        # Act
        self.neuron._merge_similar_entities()
        
        # Assert
        nodes = list(self.neuron.graph.nodes())
        self.assertEqual(len([n for n in nodes if n != "Self"]), 1)
        # Should call predict
        self.neuron.cross_encoder.predict.assert_called()

    def test_no_merge_distinct(self):
        """Test strict rejection when CrossEncoder says NO"""
        # Node A: "Brother"
        # Node B: "Sister"
        # Vectors might be close in some spaces
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.92, 0.0, 0.0]) 
        
        self.neuron.graph.add_node("Brother", vector=json.dumps(vec_a.tolist()))
        self.neuron.graph.add_node("Sister", vector=json.dumps(vec_b.tolist()))
        
        self.neuron.vector_cache = {
            "Brother": vec_a,
            "Sister": vec_b
        }
        
        # Mock CrossEncoder to say NO (score < 0.85)
        self.neuron.cross_encoder.predict.return_value = 0.10
        
        # Act
        self.neuron._merge_similar_entities()
        
        # Assert
        self.assertTrue("Brother" in self.neuron.graph)
        self.assertTrue("Sister" in self.neuron.graph)

if __name__ == '__main__':
    # Suppress logging for clean test output
    import logging
    logging.disable(logging.CRITICAL)
    unittest.main()
