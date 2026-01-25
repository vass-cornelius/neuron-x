import networkx as nx
import os

graph_file = "memory_vault/synaptic_graph.gexf"

if not os.path.exists(graph_file):
    print(f"Graph file not found at {graph_file}")
    exit(1)

try:
    g = nx.read_gexf(graph_file)
    if "Self" in g:
        self_data = g.nodes["Self"]
        r_sum = self_data.get("reinforcement_sum", "Not Found")
        e_sum = self_data.get("entropy_sum", "Not Found")
        
        print(f"--- SYSTEM STATE ---")
        print(f"Reinforcement Sum: {r_sum}")
        print(f"Entropy Sum:       {e_sum}")
        
        # Calculate current threshold if values are numbers
        try:
            r = float(r_sum)
            e = float(e_sum)
            if r < 1e-6:
                ratio = 1.0 # Infinite entropy effectively
            else:
                ratio = e / r
            
            thresh = max(0.15, 0.4 - (ratio * 0.25))
            print(f"Current Ratio (E/R): {ratio:.2f}")
            print(f"Current Threshold:   {thresh:.3f}")
            
            if thresh < 0.25:
                print("MODE: DAYDREAMING (Creative/Low Filter)")
            else:
                print("MODE: STUDYING (Focused/High Filter)")
                
        except (ValueError, TypeError):
            print("Could not calculate threshold (values might be strings or missing).")
            
    else:
        print("'Self' node not found in graph.")
except Exception as e:
    print(f"Failed to read graph: {e}")
