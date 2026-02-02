"""
Base classes and interfaces for the NeuronX plugin system.

This module defines the core abstractions that all plugins must implement,
along with supporting data structures for plugin metadata and lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import Mapping
import logging

logger = logging.getLogger("neuron-x.plugins")


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin's identity, capabilities, and requirements.
    """
    name: str
    version: str
    description: str
    author: str = "Unknown"
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


class PluginContext:
    """
    Execution context provided to plugins.
    Allows plugins to interact back with the cognitive core.
    """
    def __init__(
        self, 
        interact_func: Optional[Callable[[str], str]] = None,
        restart_func: Optional[Callable[[Optional[dict]], None]] = None,
        transcribe_func: Optional[Callable[[str], str]] = None,
        get_tool_func: Optional[Callable[[str], Optional[Callable]]] = None
    ):
        self.interact = interact_func
        self.restart = restart_func
        self.transcribe = transcribe_func
        self._get_tool_func = get_tool_func

    def get_tool(self, name: str) -> Optional[Callable]:
        """Retrieves a specific tool by name from the environment."""
        if self._get_tool_func:
            return self._get_tool_func(name)
        return None


class BasePlugin(ABC):
    """
    Abstract base class for all NeuronX plugins.
    """
    
    def __init__(self):
        self.context: Optional[PluginContext] = None

    def set_context(self, context: PluginContext) -> None:
        """Sets the execution context for the plugin."""
        self.context = context

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        pass
    
    @abstractmethod
    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        pass
    
    def on_load(self) -> None:
        logger.info(f"Loading plugin: {self.metadata.name} v{self.metadata.version}")
    
    def on_unload(self) -> None:
        logger.info(f"Unloading plugin: {self.metadata.name}")
    
    def is_available(self) -> bool:
        for dep in self.metadata.dependencies:
            pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
            try:
                __import__(pkg_name)
            except ImportError:
                logger.warning(
                    f"Plugin '{self.metadata.name}' is missing dependency: {dep}"
                )
                return False
        return True
