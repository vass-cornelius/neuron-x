
import os
import shutil
import logging
from neuron_x import NeuronX
import networkx as nx
import time

# Configure logging to see neuron_x output
logging.basicConfig(level=logging.INFO)

# Setup test environment
TEST_DIR = "./repro_test_env"
# Clean start
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

print(f"Initializing NeuronX in {TEST_DIR}...")
brain = NeuronX(persistence_path=TEST_DIR)

# Test 1: Introspection (Should FAIL/WARN)
print("\n--- Test 1: source='Introspection' ---")
brain.perceive("My status is testing_introspection.", source="Introspection")
brain.consolidate()

# Check Graph
graph_file = os.path.join(TEST_DIR, "synaptic_graph.gexf")
found_introspection = False
if os.path.exists(graph_file):
    g = nx.read_gexf(graph_file)
    for u, v, data in g.edges(data=True):
        if v == 'testing_introspection':
            found_introspection = True
            print(f"FOUND Edge: {u} -> {v} ({data})")

if not found_introspection:
    print("RESULT: 'Introspection' source was DROPPED (As Expected). CHECK LOGS for 'Unknown source' warning.")
else:
    print("RESULT: 'Introspection' source was PERSISTED (Unexpected!).")

# Test 2: Self_Reflection (Should SUCCEED)
print("\n--- Test 2: source='Self_Reflection' ---")
brain.perceive("My status is testing_reflection.", source="Self_Reflection")
brain.consolidate()

g = nx.read_gexf(graph_file)
found_reflection = False
for u, v, data in g.edges(data=True):
    if v == 'testing_reflection':
        found_reflection = True
        print(f"FOUND Edge: {u} -> {v} ({data})")

if found_reflection:
    print("RESULT: 'Self_Reflection' source was PERSISTED (SUCCESS).")
else:
    print("RESULT: 'Self_Reflection' source was DROPPED (FAILURE).")

# Clean up
# shutil.rmtree(TEST_DIR)
