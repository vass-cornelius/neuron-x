import networkx as nx
import os
from collections import defaultdict

# Path to your persistent memory
graph_file = "./memory_vault/synaptic_graph.gexf"

def display_graph_analysis(graph):
    """Display comprehensive analysis of the knowledge graph."""
    print("=" * 70)
    print("🧠 NEURON-X BRAIN MAP")
    print("=" * 70)
    
    # Statistics
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    print(f"\n📊 Statistics:")
    print(f"   • Total Entities (Nodes): {num_nodes}")
    print(f"   • Total Relationships (Edges): {num_edges}")
    
    # Group edges by relation type
    relations_by_type = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        relation = data.get('relation', 'connected_to')
        relations_by_type[relation].append((u, v))
    
    print(f"   • Relationship Types: {len(relations_by_type)}")
    print()
    
    # Display relationships grouped by type
    print("🔗 RELATIONSHIP TYPES:")
    print("-" * 70)
    for relation_type, edges in sorted(relations_by_type.items()):
        print(f"\n   [{relation_type.upper()}] ({len(edges)} relationships):")
        for u, v in edges:
            # Truncate long node names
            u_display = u[:30] + "..." if len(u) > 30 else u
            v_display = v[:30] + "..." if len(v) > 30 else v
            
            # Get edge data
            edge_data = graph[u][v]
            weight = edge_data.get('weight', 1.0)
            category = edge_data.get('category', 'FACTUAL')
            
            cat_color = {
                "FACTUAL": "cyan",
                "INFERENCE": "yellow",
                "PROPOSAL": "magenta"
            }.get(category, "white")
            
            print(f"      ({u_display}) --[{relation_type}]--> ({v_display}) [bold {cat_color}][{category}][/bold {cat_color}] weight: {weight:.1f}")
    
    print("\n" + "=" * 70)
    
    # Show entity-focused view (nodes that are not Memory_* or Concept_*)
    print("\n🎯 ENTITY-FOCUSED VIEW:")
    print("-" * 70)
    
    entities = [n for n in graph.nodes() if not n.startswith("Memory_") 
                and not n.startswith("Concept_") and n != "Self" and n != "Knowledge"]
    
    if entities:
        print(f"\nExtracted Entities: {len(entities)}")
        for entity in sorted(entities):
            print(f"\n   📌 {entity}")
            
            # Show outgoing edges
            out_edges = list(graph.out_edges(entity, data=True))
            if out_edges:
                print(f"      Outgoing:")
                for _, v, data in out_edges:
                    relation = data.get('relation', 'connected_to')
                    v_display = v[:30] + "..." if len(v) > 30 else v
                    print(f"         --[{relation}]--> {v_display}")
            
            # Show incoming edges
            in_edges = list(graph.in_edges(entity, data=True))
            if in_edges:
                print(f"      Incoming:")
                for u, _, data in in_edges:
                    relation = data.get('relation', 'connected_to')
                    u_display = u[:30] + "..." if len(u) > 30 else u
                    print(f"         {u_display} --[{relation}]-->")
    else:
        print("\n   No extracted entities found yet.")
        print("   (Run the D&D test to see entities like 'Kaelen', 'Aethervale', etc.)")
    
    print("\n" + "=" * 70)

if os.path.exists(graph_file):
    graph = nx.read_gexf(graph_file)
    display_graph_analysis(graph)
else:
    print("❌ No brain file found. Have you run the bridge and typed 'exit' yet?")