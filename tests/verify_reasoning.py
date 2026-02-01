
import os
import networkx as nx
import numpy as np
from neuron_x import NeuronX
import shutil
import json

def test_reasoning_integration():
    # Setup a dummy graph
    brain = NeuronX()
    
    # Clear existing graph for test isolation (creating a temp one effectively)
    brain.graph = nx.DiGraph()
    brain.vector_cache = {}
    
    # Add nodes
    brain.graph.add_node("Self", type="Identity", vector=json.dumps([0.1]*384))
    brain.graph.add_node("A", vector=json.dumps([0.1]*384))
    brain.graph.add_node("B", vector=json.dumps([0.1]*384))
    
    # Add Edge with Reasoning
    brain.graph.add_edge("A", "B", relation="test_relation", weight=0.5, category="HYPOTHESIS", reasoning="Because I said so")
    
    # Mock vector cache to ensure 'A' is retrieved
    brain.vector_cache["A"] = np.array([0.1]*384)
    brain.vector_cache["B"] = np.array([0.1]*384)
    
    # Mock encoder to always return stored vector (simplification)
    class MockEncoder:
        def encode(self, text):
            return np.array([0.1]*384)
    brain.encoder = MockEncoder()
    
    # Test Retrieval
    context = brain._get_relevant_memories("A", top_k=5)
    
    print("Retrieved Context:")
    found = False
    for item in context:
        print(item)
        if "(Reason: Because I said so)" in item:
            found = True
            
    if found:
        print("\nSUCCESS: Reasoning metadata found in context.")
    else:
        print("\nFAILURE: Reasoning metadata missing.")

if __name__ == "__main__":
    test_reasoning_integration()
