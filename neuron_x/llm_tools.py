import os
import logging
from pathlib import Path

logger = logging.getLogger("neuron-x")



def read_codebase_file(filename: str) -> str:
    """
    Reads the content of a file from the current neuron-x codebase directory.
    Useful for understanding the system's own architecture and constraints.
    
    Args:
        filename: The name of the file to read (e.g., 'neuron_x.py', 'consciousness_loop.py', 'gemini_interface.py', 'models.py').
                  Must be relative to the project root.
    """
    if not filename:
        return "Error: No filename provided"
    
    # Security: prevent path traversal
    if ".." in filename or filename.startswith("/"):
        return f"Error: Invalid filename '{filename}'. Use relative paths from project root."
    
    # Get project root
    project_root = Path.cwd()
    file_path = project_root / filename
    
    # Check if it's a skill file
    if ".agent/skills/" in str(filename):
        logger.info(f"[SKILLS] LLM is reading skill file: {filename}")
    
    # Security: ensure file is within project
    try:
        file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return f"Error: File '{filename}' is outside project root"
    
    if not file_path.exists():
        return f"Error: File '{filename}' not found"
    
    try:
        content = file_path.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"Error reading '{filename}': {e}"