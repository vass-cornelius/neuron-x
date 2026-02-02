from typing import Mapping, Callable, Any
from neuron_x.plugin_base import BasePlugin, PluginMetadata
from . import filesystem as fs
from . import shell

class SystemAccessPlugin(BasePlugin):
    """
    Plugin providing system access tools (filesystem and shell).
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="system_access",
            version="1.0.0",
            description="Provides tools for file system interaction and shell command execution.",
            author="Antigravity",
            capabilities=["filesystem", "terminal"]
        )
    
    def get_tools(self) -> Mapping[str, Callable[... , Any]]:
        return {
            "list_files": fs.list_files,
            "read_file": fs.read_file,
            "write_file": fs.write_file,
            "find_files": fs.find_files,
            "run_command": shell.run_command,
        }
    
    def on_load(self) -> None:
        super().on_load()
        # Any specific initialization if needed
    
    def on_unload(self) -> None:
        super().on_unload()
        # Any cleanup if needed
