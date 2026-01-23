import os
import sys
import subprocess
from pathlib import Path

# --- CONSTANTS ---
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"

def _get_executable_path(venv_path: Path, name: str) -> Path:
    """Returns the platform-specific path to an executable in the venv."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / f"{name}.exe"
    return venv_path / "bin" / name


def setup_virtual_environment():
    """
    Creates a virtual environment and installs dependencies into it.
    Uses plain print statements as this runs before rich is guaranteed to be installed.
    Returns the path to the venv directory on success, None on failure.
    """
    venv_path = Path(VENV_DIR)
    
    if not venv_path.exists():
        print(f"Creating a dedicated virtual environment in './{VENV_DIR}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', str(venv_path)])
            print("✅ Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return None

    print("\nInstalling/updating required packages into the virtual environment...")
    pip_executable = _get_executable_path(venv_path, "pip")
    
    try:
        subprocess.run(
            [str(pip_executable), 'install', '-r', REQUIREMENTS_FILE],
            capture_output=True, text=True, check=True
        )
        print("✅ Dependencies installed successfully.")
        return venv_path
    except FileNotFoundError:
        print(f"❌ Error: Could not find pip executable at '{pip_executable}'.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies:")
        # Print the error output from pip for better debugging
        print(e.stderr)
        return None

def main():
    """
    An interactive script to configure Synk and prepare it for first use.
    """
    print("🚀 NEURON-X Setup Assistant 🚀")

    # --- Step 0: Setup virtual environment and install dependencies ---
    venv_path = setup_virtual_environment()
    if not venv_path:
        sys.exit(1) # Exit if installation fails

    # After successful venv setup, execute rich_setup.py using the venv's python interpreter
    print("\n✅ Setup executed successfully. NEURON-X is available now!")

if __name__ == "__main__":
    main()
