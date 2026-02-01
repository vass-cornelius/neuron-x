
import networkx as nx
import os

GRAPH_PATH = "./memory_vault/synaptic_graph.gexf"

if os.path.exists(GRAPH_PATH):
    try:
        graph = nx.read_gexf(GRAPH_PATH)
        max_w = 0.0
        count_gt_1 = 0
        total_edges = len(graph.edges())
        
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight", 0.0))
            if w > max_w:
                max_w = w
            if w > 1.0:
                count_gt_1 += 1
                if count_gt_1 <= 5:
                    print(f"Edge > 1.0: {u} -> {v} ({data.get('relation')}) w={w}")

        print(f"Total Edges: {total_edges}")
        print(f"Max Weight: {max_w}")
        print(f"Edges > 1.0: {count_gt_1}")
        
        # Check specific edge if it exists
        if graph.has_edge("Self", "Physical Feedback"):
            print(f"Self -> Physical Feedback: {graph['Self']['Physical Feedback']}")
            
    except Exception as e:
        print(f"Error reading graph: {e}")
else:
    print("Graph file not found.")
