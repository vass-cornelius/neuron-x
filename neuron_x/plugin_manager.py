"""
Plugin manager for dynamic loading, unloading, and orchestration of NeuronX plugins.

This module provides the central plugin registry and lifecycle management,
allowing plugins to be discovered, loaded, and integrated with the bot at runtime.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable
from collections.abc import Mapping
import logging

from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins")


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin doesn't exist."""
    pass


class PluginManager:
    """
    Central manager for plugin discovery, loading, and lifecycle management.
    
    The PluginManager handles:
    - Scanning plugin directories for available plugins
    - Dynamically importing and instantiating plugin classes
    - Maintaining a registry of loaded plugins
    - Collecting tools from all active plugins
    - Managing plugin lifecycle (load/unload hooks)
    
    Attributes:
        plugin_dir: Path to the directory containing plugins
        _registry: Internal mapping of plugin names to instances
        _available: Cache of discovered but not yet loaded plugins
    
    Example:
        >>> manager = PluginManager(Path("neuron_x/plugins"))
        >>> manager.discover_plugins()
        >>> manager.load_plugin("http_fetcher")
        >>> tools = manager.get_all_tools()
    """
    
    def __init__(self, plugin_dir: Path) -> None:
        """
        Initialize the plugin manager.
        
        Args:
            plugin_dir: Path to the directory containing plugin modules
        """
        self.plugin_dir = Path(plugin_dir)
        self._registry: dict[str, BasePlugin] = {}
        self._available: dict[str, Path] = {}
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created plugin directory: {self.plugin_dir}")
    
    def discover_plugins(self) -> list[str]:
        """
        Scan the plugin directory for available plugins.
        
        Discovers both single-file plugins (*.py) and package-based plugins
        (directories with __init__.py). Plugins are cached in _available but
        not loaded until explicitly requested.
        
        Returns:
            List of discovered plugin names
        """
        self._available.clear()
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
            return []
        
        # Find single-file plugins (*.py, excluding __init__.py and __pycache__)
        for py_file in self.plugin_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            plugin_name = py_file.stem
            self._available[plugin_name] = py_file
            logger.debug(f"Discovered plugin: {plugin_name} (file: {py_file})")
        
        # Find package-based plugins (directories with __init__.py)
        for plugin_pkg in self.plugin_dir.iterdir():
            if not plugin_pkg.is_dir() or plugin_pkg.name.startswith("__"):
                continue
            
            init_file = plugin_pkg / "__init__.py"
            if init_file.exists():
                plugin_name = plugin_pkg.name
                self._available[plugin_name] = plugin_pkg
                logger.debug(f"Discovered plugin: {plugin_name} (package: {plugin_pkg})")
        
        logger.info(f"Discovered {len(self._available)} plugin(s): {list(self._available.keys())}")
        return list(self._available.keys())
    
    def load_plugin(self, name: str) -> None:
        """
        Load and initialize a plugin by name.
        
        This method:
        1. Checks if the plugin is already loaded
        2. Locates the plugin module/package
        3. Dynamically imports it
        4. Finds and instantiates the plugin class
        5. Validates it implements BasePlugin
        6. Checks dependencies via is_available()
        7. Calls the on_load() lifecycle hook
        8. Registers the plugin in the active registry
        
        Args:
            name: Plugin name (without .py extension)
            
        Raises:
            PluginNotFoundError: If the plugin doesn't exist
            PluginLoadError: If loading or initialization fails
        """
        # Check if already loaded
        if name in self._registry:
            logger.info(f"Plugin '{name}' is already loaded")
            return
        
        # Ensure we have discovered plugins
        if not self._available:
            self.discover_plugins()
        
        # Check if plugin exists
        if name not in self._available:
            raise PluginNotFoundError(f"Plugin '{name}' not found in {self.plugin_dir}")
        
        plugin_path = self._available[name]
        
        try:
            # Import the plugin module
            module = self._import_plugin_module(name, plugin_path)
            
            # Find the plugin class (should inherit from BasePlugin)
            plugin_class = self._find_plugin_class(module, name)
            
            # Instantiate the plugin
            plugin_instance = plugin_class()
            
            # Validate it implements BasePlugin
            if not isinstance(plugin_instance, BasePlugin):
                raise PluginLoadError(
                    f"Plugin '{name}' does not inherit from BasePlugin"
                )
            
            # Check if dependencies are available
            if not plugin_instance.is_available():
                raise PluginLoadError(
                    f"Plugin '{name}' dependencies not satisfied. "
                    f"Required: {plugin_instance.metadata.dependencies}"
                )
            
            # Call lifecycle hook
            plugin_instance.on_load()
            
            # Register the plugin
            self._registry[name] = plugin_instance
            logger.info(
                f"Loaded plugin: {plugin_instance.metadata.name} "
                f"v{plugin_instance.metadata.version}"
            )
            
        except Exception as e:
            if isinstance(e, (PluginNotFoundError, PluginLoadError)):
                raise
            raise PluginLoadError(f"Failed to load plugin '{name}': {e}") from e
    
    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and clean up its resources.
        
        This method:
        1. Retrieves the plugin from the registry
        2. Calls the on_unload() lifecycle hook
        3. Removes it from the active registry
        
        Args:
            name: Plugin name to unload
            
        Raises:
            PluginNotFoundError: If the plugin is not currently loaded
        """
        if name not in self._registry:
            raise PluginNotFoundError(f"Plugin '{name}' is not loaded")
        
        plugin = self._registry[name]
        
        try:
            # Call lifecycle hook
            plugin.on_unload()
        except Exception as e:
            logger.error(f"Error during plugin unload for '{name}': {e}")
        finally:
            # Always remove from registry even if unload hook fails
            del self._registry[name]
            logger.info(f"Unloaded plugin: {name}")
    
    def get_plugin(self, name: str) -> BasePlugin:
        """
        Retrieve a loaded plugin instance.
        
        Args:
            name: Plugin name
            
        Returns:
            The plugin instance
            
        Raises:
            PluginNotFoundError: If the plugin is not loaded
        """
        if name not in self._registry:
            raise PluginNotFoundError(f"Plugin '{name}' is not loaded")
        return self._registry[name]
    
    def list_plugins(self) -> dict[str, dict[str, Any]]:
        """
        List all available plugins with their status.
        
        Returns:
            Dictionary mapping plugin names to info dictionaries with keys:
            - 'loaded': bool indicating if plugin is currently active
            - 'metadata': PluginMetadata if loaded, else None
            - 'path': Path to plugin file/directory
        """
        # Ensure we have discovered plugins
        if not self._available:
            self.discover_plugins()
        
        result: dict[str, dict[str, Any]] = {}
        
        # Add all discovered plugins
        for name, path in self._available.items():
            is_loaded = name in self._registry
            result[name] = {
                "loaded": is_loaded,
                "metadata": self._registry[name].metadata if is_loaded else None,
                "path": str(path),
            }
        
        return result
    
    def get_all_tools(self) -> dict[str, Callable[..., Any]]:
        """
        Collect all tools from all loaded plugins.
        
        Combines tool dictionaries from all active plugins into a single
        mapping. If multiple plugins provide tools with the same name,
        the last one wins (with a warning logged).
        
        Returns:
            Dictionary mapping tool names to callable functions
        """
        all_tools: dict[str, Callable[..., Any]] = {}
        
        for plugin_name, plugin in self._registry.items():
            try:
                plugin_tools = plugin.get_tools()
                
                # Check for naming conflicts
                for tool_name in plugin_tools:
                    if tool_name in all_tools:
                        logger.warning(
                            f"Tool name conflict: '{tool_name}' from plugin "
                            f"'{plugin_name}' overwrites existing tool"
                        )
                
                all_tools.update(plugin_tools)
                logger.debug(
                    f"Registered {len(plugin_tools)} tool(s) from plugin '{plugin_name}'"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to get tools from plugin '{plugin_name}': {e}"
                )
        
        return all_tools
    
    def _import_plugin_module(self, name: str, path: Path) -> Any:
        """
        Dynamically import a plugin module.
        
        Args:
            name: Plugin name (used as module name)
            path: Path to plugin file or directory
            
        Returns:
            The imported module
        """
        # Construct module name to avoid conflicts
        module_name = f"neuron_x.plugins.{name}"
        
        # If it's a package (directory), import the __init__.py
        if path.is_dir():
            spec = importlib.util.spec_from_file_location(
                module_name, path / "__init__.py"
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
        
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Could not load module spec for '{name}'")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def _find_plugin_class(self, module: Any, name: str) -> type[BasePlugin]:
        """
        Find the plugin class within a module.
        
        Looks for a class that inherits from BasePlugin. If multiple are found,
        prefers one named with a "Plugin" suffix or the capitalized plugin name.
        
        Args:
            module: The imported plugin module
            name: Plugin name (for better class matching)
            
        Returns:
            The plugin class
            
        Raises:
            PluginLoadError: If no plugin class is found
        """
        plugin_classes: list[type[BasePlugin]] = []
        
        # Scan module for BasePlugin subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's a class and inherits from BasePlugin
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePlugin)
                and attr is not BasePlugin
            ):
                plugin_classes.append(attr)
        
        if not plugin_classes:
            raise PluginLoadError(
                f"No BasePlugin subclass found in plugin '{name}'"
            )
        
        # If multiple classes, prefer one with matching name
        if len(plugin_classes) > 1:
            # Try exact match or "Plugin" suffix
            preferred_names = [
                name.capitalize() + "Plugin",
                name.upper() + "Plugin",
                name.title().replace("_", "") + "Plugin",
            ]
            
            for cls in plugin_classes:
                if cls.__name__ in preferred_names:
                    return cls
            
            # Fall back to first one with a warning
            logger.warning(
                f"Multiple plugin classes found in '{name}', using {plugin_classes[0].__name__}"
            )
        
        return plugin_classes[0]
