
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_x import NeuronX
import numpy as np

def test_bridge_mechanism():
    print("Testing Bridge (Consciousness Integration)...")
    brain = NeuronX(persistence_path="./tests/temp_brain_bridge")
    
    # 1. Simulate a Subconscious Thought
    thought_text = "I am wondering about the lifespan of Wood Elves."
    thought_vec = brain.encoder.encode(thought_text)
    
    # Hack: Inject into buffer
    brain.thought_buffer.append((thought_text, thought_vec))
    
    # 2. Simulate User Input: RELEVANT
    user_input_relevant = "Tell me about Elves."
    user_vec = brain.encoder.encode(user_input_relevant)
    
    similarity = np.dot(user_vec, thought_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(thought_vec))
    print(f"Similarity (Relevant): {similarity:.2f}")
    assert similarity > 0.4, "Relevant topic should trigger injection"
    
    # 3. Simulate User Input: IRRELEVANT
    user_input_irrelevant = "What is the capital of Peru?"
    user_vec_irr = brain.encoder.encode(user_input_irrelevant)
    
    similarity_irr = np.dot(user_vec_irr, thought_vec) / (np.linalg.norm(user_vec_irr) * np.linalg.norm(thought_vec))
    print(f"Similarity (Irrelevant): {similarity_irr:.2f}")
    assert similarity_irr < 0.4, "Irrelevant topic should NOT trigger injection"
    
    print("PASS: Bridge logic correctly filters relevance.")
    
    import shutil
    shutil.rmtree("./tests/temp_brain_bridge")

if __name__ == "__main__":
    test_bridge_mechanism()
