import os
import json
import networkx as nx
from neuron_x import NeuronX

def test_hallucination_handling():
    # Setup temporary vault for test
    test_vault = "./test_memory_vault"
    if not os.path.exists(test_vault):
        os.makedirs(test_vault)
    
    # Wipe existing graph if any
    graph_file = os.path.join(test_vault, "synaptic_graph.gexf")
    if os.path.exists(graph_file):
        os.remove(graph_file)
        
    brain = NeuronX(persistence_path=test_vault, llm_client=True) # Dummy client to trigger LLM path
    
    # 1. Simulate a state where a hallucination exists (PROPOSAL category)
    # This is what might happen after AI suggests "Noahs Papa heißt Ronny Bolten."
    brain.graph.add_node("Ronny Bolten", content="Ronny Bolten", type="Entity")
    brain.graph.add_edge("Noah", "Ronny Bolten", relation="has_father", category="PROPOSAL", weight=0.5)
    brain.save_graph()
    brain._rebuild_vector_cache()
    
    print(f"Initial weight of (Noah)--[has_father]-->(Ronny Bolten): {brain.graph['Noah']['Ronny Bolten']['weight']}")
    
    # 2. Simulate User Rejection: "wtf. nein das ist komplett falsch. ronny bolten gibt es nicht."
    # We call perceive with the Rejection
    # For this test, we need to mock or ensure the LLM (if present) extracts the rejection.
    # Since we can't easily mock the Gemini client here without it being messy, 
    # let's manually trigger the consolidation with a rejection triple.
    
    rejection_triple = {
        "subject": "Ronny Bolten",
        "predicate": "is_hallucination",
        "object": "true",
        "category": "FACTUAL",
        "source": "User_Interaction"
    }
    
    # Manually add to working memory as if extracted
    brain.working_memory = [{"text": "Ronny Bolten exists not", "vector": [0]*384}] 
    
    # Mocking _extract_triples_with_llm to return our rejection triple
    original_extract = brain._extract_triples_with_llm
    brain._extract_triples_with_llm = lambda text, source: [rejection_triple]
    
    print("Simulating consolidation of rejection...")
    brain.consolidate()
    
    # 3. Verify weight reduction
    new_weight = brain.graph["Noah"]["Ronny Bolten"]["weight"]
    print(f"New weight after rejection: {new_weight}")
    
    # 4. Verify Retrieval Filtering
    # Ronny Bolten should have weight 0 or very low, and status REJECTED
    print(f"Node 'Ronny Bolten' status: {brain.graph.nodes['Ronny Bolten'].get('status')}")
    
    # Query for Noah's father
    memories = brain._get_relevant_memories("Who is Noah's father?")
    print("Relevant memories retrieved for 'Who is Noah's father?':")
    for m in memories:
        print(f" - {m}")
        
    assert "Ronny Bolten" not in "".join(memories), "Failure: Hallucinated name still retrieved after rejection!"
    assert brain.graph.nodes["Ronny Bolten"].get("status") == "REJECTED", "Failure: Node status not set to REJECTED"
    print("SUCCESS: Hallucination handled correctly.")

if __name__ == "__main__":
    test_hallucination_handling()
