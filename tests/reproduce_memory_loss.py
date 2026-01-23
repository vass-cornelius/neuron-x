import os
import shutil
from neuron_x import NeuronX

def test_memory_retention():
    # Setup a clean test environment
    test_path = "./test_memory_vault"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    
    brain = NeuronX(persistence_path=test_path)
    
    # 1. Perceive a unique fact
    fact = "The Golden Dragon of Mauren guards a hoard of Swiss Francs."
    brain.perceive(fact)
    
    # 2. Trigger consolidation manually
    # Add some garbage to fill up working memory or just call consolidate directly
    for i in range(5):
        brain.perceive(f"Irrelevant thought {i}")
    
    print("DEBUG: Calling consolidate")
    brain.consolidate()
    print("DEBUG: Consolidate finished")
    
    # 3. Verify retrieval
    query = "What does the dragon in Mauren guard?"
    print(f"DEBUG: Calling _get_relevant_memories with query: {query}")
    results = brain._get_relevant_memories(query)
    print("DEBUG: _get_relevant_memories finished")
    
    print("\nSearch Query:", query)
    print("Found Memories:")
    for res in results:
        print(f"  - {res}")
    
    assert any("Swiss Francs" in res for res in results) or any("Dragon" in res for res in results), "Failed to retrieve the correct memory!"
    print("\n[SUCCESS] Memory retention and semantic retrieval verified!")

    # Cleanup
    shutil.rmtree(test_path)

if __name__ == "__main__":
    try:
        test_memory_retention()
    except Exception as e:
        print(f"\n[FAILURE] {e}")
        exit(1)
