import os
import shutil
import glob
from pathlib import Path
from typing import List, Optional

def list_files(path: str = ".") -> List[str]:
    """
    List contents of a directory.
    
    Args:
        path: The directory path to list. Defaults to current directory.
        
    Returns:
        List of file and directory names.
    """
    try:
        p = Path(path)
        if not p.exists():
            return [f"Error: Path '{path}' does not exist."]
        return [str(f.relative_to(p.parent if p.is_absolute() else ".")) for f in p.iterdir()]
    except Exception as e:
        return [f"Error listing files: {e}"]

def read_file(path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        path: Path to the file.
        
    Returns:
        Content of the file or an error message.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str, overwrite: bool = True) -> str:
    """
    Write or overwrite a file with given content.
    
    Args:
        path: Path where the file should be written.
        content: The text content to write.
        overwrite: Whether to overwrite if file exists.
        
    Returns:
        A success or error message.
    """
    try:
        p = Path(path)
        if p.exists() and not overwrite:
            return f"Error: File '{path}' already exists and overwrite is False."
        
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully written to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"

def find_files(pattern: str, search_path: str = ".") -> List[str]:
    """
    Search for files matching a glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., '**/*.py').
        search_path: Directory to start search from.
        
    Returns:
        List of matching file paths.
    """
    try:
        return [str(p) for p in Path(search_path).glob(pattern)]
    except Exception as e:
        return [f"Error finding files: {e}"]
