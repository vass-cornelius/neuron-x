# OPTIMIZER_PROTECTED
"""
Safety configuration and enforcement for the self-optimization system.

This module defines the safety boundaries and validation rules that prevent
the optimizer from causing system damage through infinite loops, broken code,
or modification of critical system components.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger("neuron-x.optimizer")


@dataclass
class SafetyConfig:
    """
    Configuration for optimizer safety mechanisms.
    
    Attributes:
        protected_paths: Paths that cannot be modified by the optimizer
        protected_markers: Comments that mark code as protected
        max_execution_time: Timeout in seconds for test execution
        backup_retention_days: How long to keep optimization backups
        require_tests: Whether all tests must pass before commit
        malicious_patterns: Regex patterns for dangerous code
        project_root: Root directory of the project
    """
    
    protected_paths: Sequence[str] = field(default_factory=lambda: [
        "neuron_x/optimizer",
        "neuron_x/bridge.py",
    ])
    
    protected_markers: Sequence[str] = field(default_factory=lambda: [
        "# OPTIMIZER_PROTECTED",
        "# DO NOT OPTIMIZE",
    ])
    
    max_execution_time: int = 30  # seconds
    backup_retention_days: int = 30
    require_tests: bool = True
    cache_expiry_hours: int = 48  # How long to cache attempted optimizations
    
    malicious_patterns: Sequence[str] = field(default_factory=lambda: [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"os\.system\s*\(",
        r"subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True",
        r"compile\s*\(",
    ])
    
    project_root: Path = field(default_factory=lambda: Path.cwd())
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects and load env vars."""
        import os
        
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        
        # Load cache expiry from environment variable if set
        if 'OPTIMIZER_CACHE_EXPIRY_HOURS' in os.environ:
            try:
                self.cache_expiry_hours = int(os.environ['OPTIMIZER_CACHE_EXPIRY_HOURS'])
            except ValueError:
                logger.warning(f"Invalid OPTIMIZER_CACHE_EXPIRY_HOURS value, using default: {self.cache_expiry_hours}")



class SafetyEnforcer:
    """
    Enforces safety rules for code optimization.
    
    This class validates optimization requests against the safety configuration
    and prevents dangerous or protected code from being modified.
    """
    
    def __init__(self, config: SafetyConfig | None = None) -> None:
        """
        Initialize the safety enforcer.
        
        Args:
            config: Safety configuration. Uses defaults if not provided.
        """
        self.config = config or SafetyConfig()
        self._malicious_regex = [re.compile(pattern) for pattern in self.config.malicious_patterns]
    
    def is_protected(self, file_path: Path | str) -> bool:
        """
        Check if a file is in a protected zone.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is protected, False otherwise
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Convert to absolute path relative to project root
        if not file_path.is_absolute():
            file_path = self.config.project_root / file_path
        
        # Check if path matches any protected paths
        try:
            relative_path = file_path.relative_to(self.config.project_root)
            path_str = str(relative_path)
        except ValueError:
            # Path is outside project root
            logger.warning(f"Path {file_path} is outside project root")
            return True
        
        for protected in self.config.protected_paths:
            if path_str.startswith(protected):
                logger.info(f"File {path_str} is in protected zone: {protected}")
                return True
        
        # Check file content for protection markers
        if file_path.exists():
            try:
                content = file_path.read_text()
                for marker in self.config.protected_markers:
                    if marker in content:
                        logger.info(f"File {path_str} has protection marker: {marker}")
                        return True
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return True  # Err on the side of caution
        
        return False
    
    def check_malicious_patterns(self, code: str) -> list[str]:
        """
        Scan code for potentially dangerous patterns.
        
        Args:
            code: Source code to scan
            
        Returns:
            List of found malicious patterns (empty if safe)
        """
        found_patterns: list[str] = []
        
        for regex in self._malicious_regex:
            matches = regex.findall(code)
            if matches:
                found_patterns.append(f"Found dangerous pattern: {regex.pattern}")
        
        return found_patterns
    
    def validate_optimization_request(
        self,
        file_path: Path | str,
        optimized_code: str | None = None
    ) -> tuple[bool, str]:
        """
        Validate an optimization request before processing.
        
        Args:
            file_path: Path to the file to optimize
            optimized_code: The optimized code (optional, for malicious pattern check)
            
        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        # Check if file is protected
        if self.is_protected(file_path):
            return False, f"File {file_path} is in a protected zone and cannot be optimized"
        
        # Check for malicious patterns in optimized code
        if optimized_code:
            malicious = self.check_malicious_patterns(optimized_code)
            if malicious:
                return False, f"Optimized code contains dangerous patterns: {'; '.join(malicious)}"
        
        return True, ""
    
    def get_timeout(self) -> int:
        """Get the configured execution timeout in seconds."""
        return self.config.max_execution_time
    
    def requires_tests(self) -> bool:
        """
        Check if tests are required for optimizations.
        
        Returns:
            True if tests must pass before committing changes
        """
        return self.config.require_tests
