"""
Verification test for autonomous plugin tool usage in the conscious loop.

This test verifies that plugin tools are properly passed to CognitiveCore
and would be available during autonomous thought generation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuron_x.plugin_manager import PluginManager
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore
from neuron_x.const import DEFAULT_PERSISTENCE_PATH

def test_plugin_tools_integration():
    """Test that CognitiveCore can access plugin tools."""
    print("\n=== Testing Plugin Tools in CognitiveCore ===")
    
    # Setup
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    manager.discover_plugins()
    manager.load_plugin("http_fetcher")
    
    # Create CognitiveCore with plugin tools
    persistence_path = Path(DEFAULT_PERSISTENCE_PATH)
    smith = GraphSmith(persistence_path)
    vault = VectorVault()
    
    core = CognitiveCore(
        smith,
        vault,
        llm_client=None,  # No LLM for this test
        plugin_tools_getter=manager.get_all_tools
    )
    
    print("✓ CognitiveCore created with plugin_tools_getter")
    
    # Verify tools are accessible
    assert core.plugin_tools_getter is not None
    print("✓ plugin_tools_getter is set")
    
    # Get tools
    tools = core.plugin_tools_getter()
    print(f"✓ Retrieved {len(tools)} tool(s): {list(tools.keys())}")
    
    assert "fetch_file" in tools
    assert "fetch_url_content" in tools
    print("✓ Plugin tools are accessible via getter")
    
    # Verify tools are callable
    assert callable(tools["fetch_file"])
    assert callable(tools["fetch_url_content"])
    print("✓ Plugin tools are callable")
    
    print("\n=== Integration Test Passed ===")
    print("The conscious loop will have access to:")
    for tool_name in tools:
        print(f"  - {tool_name}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(test_plugin_tools_integration())
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
