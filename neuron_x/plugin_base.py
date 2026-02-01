"""
Base classes and interfaces for the NeuronX plugin system.

This module defines the core abstractions that all plugins must implement,
along with supporting data structures for plugin metadata and lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from collections.abc import Mapping
import logging

logger = logging.getLogger("neuron-x.plugins")


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin's identity, capabilities, and requirements.
    
    Attributes:
        name: Unique identifier for the plugin
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable description of plugin functionality
        author: Plugin creator/maintainer
        dependencies: List of required Python packages (e.g., ["requests>=2.28.0"])
        capabilities: Tags describing what the plugin can do (e.g., ["http", "filesystem"])
    """
    name: str
    version: str
    description: str
    author: str = "Unknown"
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


class BasePlugin(ABC):
    """
    Abstract base class for all NeuronX plugins.
    
    Plugins extend the bot's capabilities by providing tools that can be invoked
    by the LLM during conversations. Each plugin must:
    
    1. Define metadata via the `metadata` property
    2. Implement `get_tools()` to expose callable functions
    3. Optionally implement lifecycle hooks (`on_load`, `on_unload`)
    4. Optionally implement `is_available()` to check dependencies
    
    Example:
        >>> class MyPlugin(BasePlugin):
        ...     @property
        ...     def metadata(self) -> PluginMetadata:
        ...         return PluginMetadata(
        ...             name="my_plugin",
        ...             version="1.0.0",
        ...             description="Does something useful"
        ...         )
        ...     
        ...     def get_tools(self) -> Mapping[str, Callable]:
        ...         return {"my_tool": self._my_tool_impl}
        ...     
        ...     def _my_tool_impl(self, arg: str) -> str:
        ...         return f"Processed: {arg}"
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.
        
        This property must be implemented by all plugins to provide
        identification and capability information.
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        """
        Return a mapping of tool names to callable functions.
        
        Tools are functions that can be invoked by the LLM. Each tool should:
        - Have clear, descriptive names
        - Include comprehensive docstrings (used by LLM to understand usage)
        - Have type-annotated parameters and return values
        - Handle errors gracefully
        
        Returns:
            Dictionary mapping tool names to callable functions
            
        Example:
            >>> def get_tools(self):
            ...     return {
            ...         "fetch_data": self._fetch_data,
            ...         "process_file": self._process_file,
            ...     }
        """
        pass
    
    def on_load(self) -> None:
        """
        Lifecycle hook called when the plugin is loaded.
        
        Use this to initialize resources, validate configuration,
        or perform setup tasks. If initialization fails, raise an
        exception to prevent the plugin from loading.
        
        Raises:
            Exception: If initialization fails
        """
        logger.info(f"Loading plugin: {self.metadata.name} v{self.metadata.version}")
    
    def on_unload(self) -> None:
        """
        Lifecycle hook called before the plugin is unloaded.
        
        Use this to clean up resources, close connections,
        or save state. This method should not raise exceptions.
        """
        logger.info(f"Unloading plugin: {self.metadata.name}")
    
    def is_available(self) -> bool:
        """
        Check if the plugin's dependencies are satisfied.
        
        Override this method to perform custom availability checks,
        such as verifying required packages are installed or
        checking for required environment variables.
        
        Returns:
            True if the plugin can be used, False otherwise
        """
        # Default implementation: check if required packages are importable
        for dep in self.metadata.dependencies:
            # Extract package name (strip version specifiers)
            pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
            try:
                __import__(pkg_name)
            except ImportError:
                logger.warning(
                    f"Plugin '{self.metadata.name}' is missing dependency: {dep}"
                )
                return False
        return True
