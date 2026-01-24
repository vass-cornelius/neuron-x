
import os
import shutil
import logging
from neuron_x import NeuronX, read_codebase_file


# Configure logging
logging.basicConfig(level=logging.INFO)

def test_read_codebase_file_direct():
    """Test the helper function directly."""
    print("\n--- Testing read_codebase_file (Direct) ---")
    
    # Positive Case
    content = read_codebase_file("neuron_x.py")
    assert "class NeuronX:" in content
    print("SUCCESS: Read neuron_x.py")
    
    # Negative Case (External)
    content = read_codebase_file("../../../etc/passwd")
    assert "Access denied" in content
    print("SUCCESS: Blocked path traversal")
    
    # Negative Case (Non-existent)
    content = read_codebase_file("ghost.py")
    assert "File not found" in content
    print("SUCCESS: Handled missing file")

if __name__ == "__main__":
    test_read_codebase_file_direct()
    
    print("\n[NOTE] To test the LLM integration fully, run the cognitive core and observe logs.")
