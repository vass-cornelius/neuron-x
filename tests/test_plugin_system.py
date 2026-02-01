"""
Test script for the NeuronX plugin system.

This script verifies that:
1. Plugins can be discovered
2. Plugins can be loaded and unloaded
3. Tools from plugins are accessible
4. The HTTP fetcher plugin works correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuron_x.plugin_manager import PluginManager
from neuron_x.plugin_base import BasePlugin

def test_plugin_discovery():
    """Test that plugins can be discovered."""
    print("\n=== Testing Plugin Discovery ===")
    
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    
    available = manager.discover_plugins()
    print(f"✓ Discovered {len(available)} plugin(s): {available}")
    
    assert "http_fetcher" in available, "http_fetcher plugin should be discovered"
    print("✓ http_fetcher plugin found")

def test_plugin_loading():
    """Test that plugins can be loaded and unloaded."""
    print("\n=== Testing Plugin Loading ===")
    
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    manager.discover_plugins()
    
    # Load the plugin
    manager.load_plugin("http_fetcher")
    print("✓ Loaded http_fetcher plugin")
    
    # Verify it's in the registry
    plugin = manager.get_plugin("http_fetcher")
    assert isinstance(plugin, BasePlugin)
    print(f"✓ Plugin instance: {plugin.metadata.name} v{plugin.metadata.version}")
    
    # List plugins
    plugins = manager.list_plugins()
    assert plugins["http_fetcher"]["loaded"] is True
    print(f"✓ Plugin status: loaded={plugins['http_fetcher']['loaded']}")
    
    # Unload the plugin
    manager.unload_plugin("http_fetcher")
    print("✓ Unloaded http_fetcher plugin")
    
    # Verify it's removed
    plugins = manager.list_plugins()
    assert plugins["http_fetcher"]["loaded"] is False
    print("✓ Plugin unloaded successfully")

def test_plugin_tools():
    """Test that tools from plugins are accessible."""
    print("\n=== Testing Plugin Tools ===")
    
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    manager.discover_plugins()
    manager.load_plugin("http_fetcher")
    
    # Get all tools
    tools = manager.get_all_tools()
    print(f"✓ Retrieved {len(tools)} tool(s): {list(tools.keys())}")
    
    assert "fetch_file" in tools, "fetch_file tool should be available"
    assert "fetch_url_content" in tools, "fetch_url_content tool should be available"
    print("✓ All expected tools are present")
    
    # Verify tools are callable
    assert callable(tools["fetch_file"])
    assert callable(tools["fetch_url_content"])
    print("✓ Tools are callable")

def test_http_fetcher_functionality():
    """Test the HTTP fetcher plugin functionality."""
    print("\n=== Testing HTTP Fetcher Plugin ===")
    
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    manager.discover_plugins()
    manager.load_plugin("http_fetcher")
    
    tools = manager.get_all_tools()
    fetch_url_content = tools["fetch_url_content"]
    
    # Test fetching example.com
    print("Fetching https://example.com ...")
    result = fetch_url_content("https://example.com", timeout=10)
    
    assert "example.com" in result.lower() or "example domain" in result.lower(), \
        "Should retrieve example.com content"
    print("✓ Successfully fetched content from example.com")
    print(f"  Preview: {result[:200]}...")

def test_error_handling():
    """Test error handling for invalid requests."""
    print("\n=== Testing Error Handling ===")
    
    plugin_dir = project_root / "neuron_x" / "plugins"
    manager = PluginManager(plugin_dir)
    manager.discover_plugins()
    manager.load_plugin("http_fetcher")
    
    tools = manager.get_all_tools()
    fetch_url_content = tools["fetch_url_content"]
    
    # Test invalid URL scheme
    result = fetch_url_content("ftp://example.com")
    assert "Error" in result and "Invalid URL scheme" in result
    print("✓ Invalid URL scheme handled correctly")
    
    # Test non-existent domain
    result = fetch_url_content("https://this-domain-definitely-does-not-exist-12345.com", timeout=5)
    assert "Error" in result
    print("✓ Non-existent domain handled correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("NeuronX Plugin System Test Suite")
    print("=" * 60)
    
    try:
        test_plugin_discovery()
        test_plugin_loading()
        test_plugin_tools()
        test_http_fetcher_functionality()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
