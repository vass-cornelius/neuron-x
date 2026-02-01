# NeuronX Plugins

This directory contains plugins that extend the capabilities of the NeuronX bot.

## Creating a Plugin

Plugins can be either:
1. **Single-file plugins**: A single `.py` file in this directory
2. **Package plugins**: A subdirectory with an `__init__.py` file

### Basic Plugin Structure

```python
from neuron_x.plugin_base import BasePlugin, PluginMetadata
from typing import Callable
from collections.abc import Mapping

class MyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="Does something useful",
            author="Your Name",
            dependencies=["requests>=2.28.0"],  # Optional
            capabilities=["http", "data_processing"]  # Optional tags
        )
    
    def get_tools(self) -> Mapping[str, Callable]:
        """Return tools that the LLM can invoke."""
        return {
            "my_tool": self._my_tool_impl
        }
    
    def _my_tool_impl(self, argument: str) -> str:
        """
        Tool description that the LLM will see.
        
        Args:
            argument: Description of the argument
            
        Returns:
            Description of what is returned
        """
        # Your implementation here
        return f"Processed: {argument}"
    
    def on_load(self) -> None:
        """Called when plugin is loaded - initialize resources here."""
        super().on_load()
        # Your initialization code
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded - cleanup resources here."""
        super().on_unload()
        # Your cleanup code
```

### Tool Function Guidelines

1. **Comprehensive docstrings**: The LLM uses docstrings to understand how to use your tool
2. **Type hints**: Always use type hints for parameters and return values
3. **Error handling**: Handle errors gracefully and return meaningful messages
4. **Descriptive names**: Use clear, action-oriented names (e.g., `fetch_file`, `send_email`)

### Dependencies

If your plugin requires external packages, list them in the `dependencies` field:

```python
dependencies=["requests>=2.28.0", "beautifulsoup4"]
```

The plugin system will check if dependencies are available before loading the plugin.

## Example Plugins

- `http_fetcher/`: Fetches files from online sources via HTTP/HTTPS

## Plugin Management

Plugins are automatically discovered and can be managed via the `PluginManager`:

```python
from neuron_x.bridge import NeuronBridge

bridge = NeuronBridge()

# List all plugins
plugins = bridge.list_plugins()

# Load a plugin
bridge.load_plugin("my_plugin")

# Unload a plugin
bridge.unload_plugin("my_plugin")
```
