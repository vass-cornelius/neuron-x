import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neuron_x.plugins.system_access import SystemAccessPlugin

def test_system_access():
    print("--- Testing SystemAccess Plugin ---")
    plugin = SystemAccessPlugin()
    tools = plugin.get_tools()
    
    # 1. Test list_files
    print("\n[Testing list_files]")
    files = tools["list_files"](".")
    print(f"Files in current dir: {len(files)}")
    assert len(files) > 0
    
    # 2. Test write_file and read_file
    print("\n[Testing write_file and read_file]")
    test_file = "tmp_test_plugin.txt"
    content = "Hello from automated test"
    res_write = tools["write_file"](test_file, content)
    print(res_write)
    assert "Successfully" in res_write
    
    res_read = tools["read_file"](test_file)
    print(f"Read content: {res_read}")
    assert res_read == content
    
    # 3. Test find_files
    print("\n[Testing find_files]")
    found = tools["find_files"](test_file)
    print(f"Found files: {found}")
    assert test_file in found[0]
    
    # 4. Test run_command
    print("\n[Testing run_command]")
    res_cmd = tools["run_command"]("echo 'Plugin works'")
    print(f"Command output: {res_cmd.strip()}")
    assert "Plugin works" in res_cmd
    
    # 5. Test dangerous command block
    print("\n[Testing safety block]")
    res_danger = tools["run_command"]("rm -rf /")
    print(f"Dangerous output: {res_danger}")
    assert "blocked" in res_danger.lower()
    
    # Cleanup
    os.remove(test_file)
    print("\nCleanup done. All tests passed!")

if __name__ == "__main__":
    try:
        test_system_access()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
