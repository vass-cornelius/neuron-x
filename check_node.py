
import networkx as nx
import os

GRAPH_PATH = "./memory_vault/synaptic_graph.gexf"

if os.path.exists(GRAPH_PATH):
    graph = nx.read_gexf(GRAPH_PATH)
    if "Physical Feedback" in graph.nodes:
        print(f"Node 'Physical Feedback' found. Data: {graph.nodes['Physical Feedback']}")
    else:
        print("Node 'Physical Feedback' NOT found in the graph.")
        # Check for similar names
        for node in graph.nodes:
            if "Physical" in node:
                print(f"Found similar node: {node}")
else:
    print("Graph file not found.")
