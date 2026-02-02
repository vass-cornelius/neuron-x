import subprocess
import logging

logger = logging.getLogger("neuron-x.plugins.system_access")

# List of dangerous commands to block
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:", # Fork bomb
    "chmod 777 /",
    "chown",
]

def run_command(command: str, timeout: int = 120) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute.
        timeout: Safety timeout in seconds (default: 120)
        
    Returns:
        The combined stdout and stderr of the command.
    """
    # Basic safety check
    cmd_lower = command.lower()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return f"Error: Command '{command}' is blocked for safety reasons."
            
    try:
        logger.info(f"Executing command: {command} (timeout={timeout}s)")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
            
        if not output.strip():
            return f"Command executed successfully (no output). Exit code: {result.returncode}"
            
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {e}"
