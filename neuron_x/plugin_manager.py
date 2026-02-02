"""
Plugin manager for dynamic loading, unloading, and orchestration of NeuronX plugins.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from collections.abc import Mapping
import logging

from neuron_x.plugin_base import BasePlugin, PluginMetadata, PluginContext

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
    """
    
    def __init__(self, plugin_dir: Path) -> None:
        self.plugin_dir = Path(plugin_dir)
        self._registry: dict[str, BasePlugin] = {}
        self._available: dict[str, Path] = {}
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created plugin directory: {self.plugin_dir}")
    
    def discover_plugins(self) -> list[str]:
        self._available.clear()
        
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
            return []
        
        for py_file in self.plugin_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            plugin_name = py_file.stem
            self._available[plugin_name] = py_file
        
        for plugin_pkg in self.plugin_dir.iterdir():
            if not plugin_pkg.is_dir() or plugin_pkg.name.startswith("__"):
                continue
            
            init_file = plugin_pkg / "__init__.py"
            if init_file.exists():
                plugin_name = plugin_pkg.name
                self._available[plugin_name] = plugin_pkg
        
        return list(self._available.keys())
    
    def load_plugin(self, name: str, context: Optional[PluginContext] = None) -> None:
        """
        Load and initialize a plugin by name.
        """
        if name in self._registry:
            logger.info(f"Plugin '{name}' is already loaded")
            return
        
        if not self._available:
            self.discover_plugins()
        
        if name not in self._available:
            raise PluginNotFoundError(f"Plugin '{name}' not found in {self.plugin_dir}")
        
        plugin_path = self._available[name]
        
        try:
            module = self._import_plugin_module(name, plugin_path)
            plugin_class = self._find_plugin_class(module, name)
            
            # Instantiate the plugin
            plugin_instance = plugin_class()
            
            if not isinstance(plugin_instance, BasePlugin):
                raise PluginLoadError(f"Plugin '{name}' does not inherit from BasePlugin")
            
            # Inject context if provided
            if context:
                plugin_instance.set_context(context)
            
            if not plugin_instance.is_available():
                raise PluginLoadError(f"Plugin '{name}' dependencies not satisfied.")
            
            # Call lifecycle hook
            plugin_instance.on_load()
            
            self._registry[name] = plugin_instance
            logger.info(f"Loaded plugin: {plugin_instance.metadata.name} v{plugin_instance.metadata.version}")
            
        except Exception as e:
            if isinstance(e, (PluginNotFoundError, PluginLoadError)):
                raise
            raise PluginLoadError(f"Failed to load plugin '{name}': {e}") from e
    
    def unload_plugin(self, name: str) -> None:
        if name not in self._registry:
            raise PluginNotFoundError(f"Plugin '{name}' is not loaded")
        
        plugin = self._registry[name]
        try:
            plugin.on_unload()
        except Exception as e:
            logger.error(f"Error during plugin unload for '{name}': {e}")
        finally:
            del self._registry[name]
            logger.info(f"Unloaded plugin: {name}")

    def unload_all(self) -> None:
        """Unload all currently loaded plugins."""
        plugin_names = list(self._registry.keys())
        for name in plugin_names:
            try:
                self.unload_plugin(name)
            except Exception as e:
                logger.error(f"Error unloading plugin '{name}' during shutdown: {e}")
    
    def get_plugin(self, name: str) -> BasePlugin:
        if name not in self._registry:
            raise PluginNotFoundError(f"Plugin '{name}' is not loaded")
        return self._registry[name]
    
    def list_plugins(self) -> dict[str, dict[str, Any]]:
        if not self._available:
            self.discover_plugins()
        
        result: dict[str, dict[str, Any]] = {}
        for name, path in self._available.items():
            is_loaded = name in self._registry
            result[name] = {
                "loaded": is_loaded,
                "metadata": self._registry[name].metadata if is_loaded else None,
                "path": str(path),
            }
        return result
    
    def get_all_tools(self) -> dict[str, Callable[..., Any]]:
        all_tools: dict[str, Callable[..., Any]] = {}
        for plugin_name, plugin in self._registry.items():
            try:
                plugin_tools = plugin.get_tools()
                for tool_name in plugin_tools:
                    if tool_name in all_tools:
                        logger.warning(f"Tool conflict: '{tool_name}' from '{plugin_name}'")
                all_tools.update(plugin_tools)
            except Exception as e:
                logger.error(f"Failed to get tools from plugin '{plugin_name}': {e}")
        return all_tools
    
    def _import_plugin_module(self, name: str, path: Path) -> Any:
        module_name = f"neuron_x.plugins.{name}"
        if path.is_dir():
            spec = importlib.util.spec_from_file_location(module_name, path / "__init__.py")
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
        
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Could not load module spec for '{name}'")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    def _find_plugin_class(self, module: Any, name: str) -> type[BasePlugin]:
        plugin_classes: list[type[BasePlugin]] = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin):
                plugin_classes.append(attr)
        
        if not plugin_classes:
            raise PluginLoadError(f"No BasePlugin subclass found in plugin '{name}'")
        
        if len(plugin_classes) > 1:
            preferred_names = [name.capitalize() + "Plugin", name.title().replace("_", "") + "Plugin"]
            for cls in plugin_classes:
                if cls.__name__ in preferred_names:
                    return cls
        return plugin_classes[0]
