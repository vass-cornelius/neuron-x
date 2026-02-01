def read_codebase_file(filename: str) -> str:
    """
    Reads the content of a file from the current neuron-x codebase directory.
    Useful for understanding the system's own architecture and constraints.
    
    Args:
        filename: The name of the file to read (e.g., 'neuron_x.py', 'consciousness_loop.py', 'gemini_interface.py', 'models.py').
                  Must be relative to the project root.
    """
    try:
        # Security: Restrict to current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.abspath(os.path.join(base_dir, filename))
        
        if not target_path.startswith(base_dir):
            return f"Access denied: {filename} is outside the codebase directory."
            
        if not os.path.exists(target_path):
            return f"File not found: {filename}"
            
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return f"--- FILE: {filename} ---\n{content}\n--- END OF FILE ---"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"