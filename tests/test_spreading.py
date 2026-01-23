import os
import shutil
import json
import networkx as nx
import numpy as np
from neuron_x import NeuronX

def test_spreading_activation():
    # Setup a clean test environment
    test_path = "./test_spreading_vault"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    
    brain = NeuronX(persistence_path=test_path)
    
    # Define some facts that create a chain
    # We'll mock the extraction result for testing retrieval logic
    node1 = "Kaelen"
    node2 = "Wood Elf Rogue"
    node3 = "Baldur's Gate"
    
    for n in [node1, node2, node3]:
        vec = brain.encoder.encode(n).tolist()
        brain.graph.add_node(n, content=n, vector=json.dumps(vec))
    
    brain.graph.add_edge(node1, node2, relation="is_a", category="FACTUAL")
    brain.graph.add_edge(node1, node3, relation="located_in", category="FACTUAL")
    
    # Update cache
    brain._rebuild_vector_cache()
    
    # Check graph structure
    print("[DEBUG] Nodes:", brain.graph.nodes())
    print("[DEBUG] Edges:", brain.graph.edges(data=True))

    # Query about the class, which should find the person, and then the location
    query = "Where is the Wood Elf?"
    print(f"\n[DEBUG] Query: {query}")
    results = brain._get_relevant_memories(query, top_k=3)
    
    print("Found Memories:")
    for res in results:
        print(f"  - {res}")
    
    # Check if we got the location
    has_location = any("Baldur's Gate" in res for res in results)
    has_kaelen = any("Kaelen" in res for res in results)
    
    print(f"\n[RESULT] Has Kaelen: {has_kaelen}, Has Location: {has_location}")
    
    if has_location:
        print("[SUCCESS] Spreading activation successfully retrieved the location across nodes!")
    else:
        print("[FAILURE] Location not found in retrieved memories.")

    # Cleanup
    shutil.rmtree(test_path)

if __name__ == "__main__":
    try:
        test_spreading_activation()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
