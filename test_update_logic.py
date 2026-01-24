
import os
import shutil
import networkx as nx

os.environ["NEURON_X_LOG_LEVEL"] = "DEBUG"
from neuron_x import NeuronX

# Setup temp directory
TEST_DIR = "./test_update_logic"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

# Initialize NeuronX
print("Initializing NeuronX...")
brain = NeuronX(persistence_path=TEST_DIR)

# 1. Learn Initial Fact: Age is 3.
print("\n--- Step 1: Learning 'My age is 3 years old' ---")
# Using "My age is..." matches regex: (?:my|our)\s+(\w+)\s+is\s+(.+?)(?:\.|$|,)
brain.perceive("My age is 3 years old.", source="User_Interaction") 
brain.consolidate()

# Check Graph
graph_file = os.path.join(TEST_DIR, "synaptic_graph.gexf")
graph = nx.read_gexf(graph_file)
print("Edges from Self:")
if "Self" in graph:
    for neighbor in graph.neighbors("Self"):
        if neighbor.startswith("Memory_"): continue
        edge = graph["Self"][neighbor]
        print(f"  -> {neighbor} [{edge.get('relation')}] (weight: {edge.get('weight')})")
else:
    print("  (Self node not found)")

# 2. Reinforce Fact: Age is 3 (Again).
print("\n--- Step 2: Reinforcing 'My age is 3 years old' ---")
brain.perceive("My age is 3 years old.", source="User_Interaction")
brain.consolidate()

# Check Graph Again
graph = nx.read_gexf(graph_file)
print("Edges from Self (After Reinforcement):")
if "Self" in graph:
    for neighbor in graph.neighbors("Self"):
        if neighbor.startswith("Memory_"): continue
        edge = graph["Self"][neighbor]
        print(f"  -> {neighbor} [{edge.get('relation')}] (weight: {edge.get('weight')})")
else:
    print("  (Self node not found)")
