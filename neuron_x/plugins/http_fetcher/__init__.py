"""
HTTP Fetcher Plugin for NeuronX.

This plugin provides tools for fetching files and content from online sources
via HTTP/HTTPS. It supports downloading files, retrieving web page content,
and handling common HTTP operations.
"""

from pathlib import Path
from typing import Any, Callable
from collections.abc import Mapping
import logging

from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.http_fetcher")


class HttpFetcherPlugin(BasePlugin):
    """
    Plugin for fetching files and content from HTTP/HTTPS sources.
    
    Capabilities:
    - Fetch file content from URLs
    - Download files to local filesystem
    - Handle timeouts and errors gracefully
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="http_fetcher",
            version="1.0.0",
            description="Fetch files and content from online sources via HTTP/HTTPS",
            author="NeuronX Team",
            dependencies=["requests>=2.28.0"],
            capabilities=["http", "download", "web"],
        )
    
    def on_load(self) -> None:
        """Initialize the plugin."""
        super().on_load()
        # Validate that requests is available
        try:
            import requests
            self._requests = requests
            logger.info("HTTP Fetcher plugin initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import requests: {e}")
            raise
    
    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        """
        Return available tools for fetching HTTP content.
        
        Returns:
            Dictionary of tool functions
        """
        return {
            "fetch_file": self.fetch_file,
            "fetch_url_content": self.fetch_url_content,
        }
    
    def fetch_file(
        self,
        url: str,
        destination: str | None = None,
        timeout: int = 30,
    ) -> str:
        """
        Fetch a file from a URL and optionally save it to disk.
        
        This tool downloads content from HTTP/HTTPS URLs. If a destination path
        is provided, the file is saved to disk. Otherwise, the content is returned
        as a string (useful for text files, JSON, etc.).
        
        Args:
            url: The URL to fetch (must be http:// or https://)
            destination: Optional local path to save the file. If not provided,
                        content is returned as a string.
            timeout: Request timeout in seconds (default: 30)
        
        Returns:
            Success message with file location or file content
            
        Examples:
            - fetch_file("https://example.com/data.json") → Returns JSON content
            - fetch_file("https://example.com/image.png", "/tmp/image.png") → Saves file
        """
        try:
            logger.info(f"Fetching URL: {url}")
            
            # Validate URL scheme
            if not url.startswith(("http://", "https://")):
                return f"Error: Invalid URL scheme. Must be http:// or https://. Got: {url}"
            
            # Make the request
            response = self._requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # If destination is provided, save to file
            if destination:
                dest_path = Path(destination)
                
                # Create parent directories if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content to file
                with dest_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = dest_path.stat().st_size
                logger.info(f"Saved {file_size} bytes to {dest_path}")
                return f"Successfully downloaded file to {dest_path} ({file_size:,} bytes)"
            
            # Otherwise, return content as string
            content = response.text
            content_size = len(content)
            
            # Truncate very large responses
            max_length = 10000
            if content_size > max_length:
                content = content[:max_length]
                logger.warning(f"Content truncated from {content_size} to {max_length} characters")
                return (
                    f"Retrieved {content_size:,} characters (truncated to {max_length:,}):\n\n"
                    f"{content}\n\n... [truncated]"
                )
            
            logger.info(f"Retrieved {content_size} characters from {url}")
            return f"Retrieved content from {url} ({content_size:,} characters):\n\n{content}"
            
        except self._requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout} seconds for URL: {url}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except self._requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code} for URL: {url}"
            logger.error(error_msg)
            return f"Error: {error_msg} - {e.response.reason}"
        
        except self._requests.exceptions.RequestException as e:
            error_msg = f"Request failed for URL: {url}"
            logger.error(f"{error_msg}: {e}")
            return f"Error: {error_msg} - {str(e)}"
        
        except OSError as e:
            error_msg = f"Failed to write file to {destination}"
            logger.error(f"{error_msg}: {e}")
            return f"Error: {error_msg} - {str(e)}"
        
        except Exception as e:
            error_msg = f"Unexpected error fetching {url}"
            logger.error(f"{error_msg}: {e}")
            return f"Error: {error_msg} - {str(e)}"
    
    def fetch_url_content(self, url: str, timeout: int = 30) -> str:
        """
        Fetch and return the text content of a URL (HTTP/HTTPS only).
        
        This is a simplified version of fetch_file that always returns content
        as a string, useful for quickly retrieving web page content, API responses,
        or text files.
        
        NOTE: This tool is for REMOTE URLs only. To read local codebase files,
        use the read_codebase_file tool instead.
        
        Args:
            url: The URL to fetch (must be http:// or https://)
            timeout: Request timeout in seconds (default: 30)
        
        Returns:
            The text content from the URL or an error message
            
        Example:
            - fetch_url_content("https://api.example.com/data") → Returns API response
        """
        # Reject file:// URLs
        if url.startswith('file://'):
            return (
                "Error: file:// URLs are not supported. "
                "To read local codebase files, use the read_codebase_file tool instead. "
                "Example: read_codebase_file('neuron_x/cognition.py')"
            )
        
        return self.fetch_file(url=url, destination=None, timeout=timeout)
