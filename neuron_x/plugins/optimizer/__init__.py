"""
Optimizer Plugin for NeuronX.

This plugin exposes self-optimization capabilities to the LLM through the
tool system, enabling the bot to request and perform code optimizations.
"""

import logging
from pathlib import Path
from typing import Callable, Any
from collections.abc import Mapping

from neuron_x.plugin_base import BasePlugin, PluginMetadata
from neuron_x.optimizer import SelfOptimizer, SafetyConfig

logger = logging.getLogger("neuron-x.plugins")


class OptimizerPlugin(BasePlugin):
    """
    Plugin that provides self-optimization capabilities.
    
    This plugin allows the LLM to:
    - Identify optimization opportunities in the codebase
    - Execute optimizations with full safety checks
    - Roll back failed optimizations
    - View optimization history
    """
    
    def __init__(self) -> None:
        """Initialize the optimizer plugin."""
        super().__init__()
        self._optimizer: SelfOptimizer | None = None
        self._project_root = Path(__file__).parent.parent.parent.parent
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="optimizer",
            version="1.0.0",
            description="Self-optimization plugin with Draft-Test-Commit workflow",
            author="NeuronX System",
            dependencies=[],
            capabilities=["code_analysis", "code_optimization", "rollback"]
        )
    
    def get_tools(self) -> Mapping[str, Callable[..., Any]]:
        """
        Return optimizer tools for LLM use.
        
        Returns:
            Dictionary of tool names to callable functions
        """
        return {
            "list_optimization_opportunities": self.list_optimization_opportunities,
            "optimize_module": self.optimize_module,
            "rollback_optimization": self.rollback_optimization,
            "get_optimization_history": self.get_optimization_history,
            "clear_optimization_cache": self.clear_optimization_cache,
        }
    
    def on_load(self) -> None:
        """Initialize the optimizer when plugin is loaded."""
        super().on_load()
        
        # Create safety config that protects critical files
        safety_config = SafetyConfig(project_root=self._project_root)
        
        # Initialize the optimizer
        self._optimizer = SelfOptimizer(
            project_root=self._project_root,
            safety_config=safety_config
        )
        
        logger.info("Optimizer plugin initialized successfully")
    
    def on_unload(self) -> None:
        """Clean up when plugin is unloaded."""
        if self._optimizer:
            self._optimizer.cleanup()
        super().on_unload()
    
    # Tool Methods (exposed to LLM)
    
    def list_optimization_opportunities(
        self,
        module_path: str | None = None,
        limit: int = 10
    ) -> str:
        """
        Scan for optimization opportunities in the codebase.
        
        Args:
            module_path: Optional specific module to analyze (e.g., "neuron_x/cognition.py")
            limit: Maximum number of opportunities to return
            
        Returns:
            Human-readable report of optimization opportunities
        """
        if not self._optimizer:
            return "Error: Optimizer not initialized"
        
        try:
            opportunities = self._optimizer.identify_opportunities(module_path)
            
            if not opportunities:
                return "No optimization opportunities found."
            
            # Limit results
            opportunities = opportunities[:limit]
            
            # Format report
            lines = [f"Found {len(opportunities)} optimization opportunities:\n"]
            
            for i, opp in enumerate(opportunities, 1):
                target = f"{opp.class_name}.{opp.function_name}" if opp.class_name else opp.function_name
                
                # Show relative path from project root, not just filename
                try:
                    rel_path = opp.module_path.relative_to(Path.cwd())
                except ValueError:
                    rel_path = opp.module_path.name
                
                lines.append(f"{i}. [{opp.priority}/10] {target} in {rel_path}")
                lines.append(f"   Type: {opp.opportunity_type.value}")
                lines.append(f"   Issue: {opp.description}")
                if opp.suggestions:
                    lines.append(f"   Suggestions: {', '.join(opp.suggestions)}")
                lines.append(f"   Module path for optimize_module: '{rel_path}'")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to list optimization opportunities: {e}")
            return f"Error: {e}"
    
    def optimize_module(
        self,
        module_path: str,
        function_name: str,
        optimized_code: str,
        class_name: str | None = None,
        require_tests: bool = True
    ) -> str:
        """
        Optimize a specific function in a module.
        
        This executes the full Draft-Test-Commit workflow:
        1. Validates the optimized code for safety
        2. Creates a draft and runs tests
        3. Commits changes if all validations pass
        4. Creates automatic backup for rollback
        
        Args:
            module_path: **RELATIVE** path to the module from project root
                         Examples: "neuron_x/cognition.py", "neuron_x/storage.py"
                         NOT absolute paths like "/Users/.../cognition.py"
            function_name: Name of the function to optimize
            optimized_code: The complete optimized function code
            class_name: Class name if optimizing a method
            require_tests: Whether to require tests to pass (defaults to True). 
                           Set to False if no tests exist for this module.
            
        Returns:
            Success message or error description
        """
        if not self._optimizer:
            return "Error: Optimizer not initialized"
        
        try:
            # Create a cache key for this optimization attempt
            from pathlib import Path
            from datetime import datetime
            module_path_obj = Path(module_path)
            cache_key = f"{module_path}::{class_name + '.' if class_name else ''}{function_name}"
            
            record = self._optimizer.optimize_function(
                module_path=module_path,
                function_name=function_name,
                optimized_code=optimized_code,
                class_name=class_name,
                require_tests=require_tests
            )
            
            # Record the attempt in cache
            self._optimizer._attempted_cache[cache_key] = {
                "timestamp": datetime.now().isoformat(),
                "attempts": self._optimizer._attempted_cache.get(cache_key, {}).get("attempts", 0) + 1,
                "last_result": "success" if record.success else "failed"
            }
            self._optimizer._save_cache()
            
            if record.success:
                msg = f"✓ Successfully optimized {function_name}"
                
                # Report if tests were skipped or not run
                tests_found_in_results = False
                if record.validation_results:
                    test_results = [r for r in record.validation_results if r.level.value == "tests"]
                    if test_results:
                        tests_found_in_results = True
                        tr = test_results[0]
                        if tr.output and "Skipped" in tr.output:
                            msg += f" (Tests skipped: {tr.output})"
                
                if not tests_found_in_results and not require_tests:
                    msg += " (Tests manually bypassed)"
                
                if record.backup_id:
                    msg += f"\n  Backup ID: {record.backup_id}"
                return msg
            else:
                # Include detailed error information for LLM to learn from
                error_parts = [f"✗ Optimization failed: {record.error_message}"]
                
                # If there are validation results, include them
                if record.validation_results:
                    error_parts.append("\nValidation Details:")
                    for result in record.validation_results:
                        status = "✓" if result.success else "✗"
                        label = result.level.value
                        
                        # Special handling for skipped tests in failure report
                        msg = result.error_message or "OK"
                        if label == "tests" and result.output and "Skipped" in result.output:
                            status = "-"
                            msg = result.output
                            
                        error_parts.append(f"  {status} {label}: {msg}")
                        
                        # Include test output if available and failed
                        if label == "tests" and result.output and not result.success:
                            error_parts.append(f"\nTest Output:\n{result.output}")
                
                return "\n".join(error_parts)
                
        except Exception as e:
            logger.error(f"Optimization failed with exception: {e}", exc_info=True)
            return f"Error: {e}"
    
    def rollback_optimization(self, backup_id: str) -> str:
        """
        Roll back to a previous backup.
        
        Args:
            backup_id: ID of the backup to restore (from optimization history)
            
        Returns:
            Success message or error description
        """
        if not self._optimizer:
            return "Error: Optimizer not initialized"
        
        try:
            success = self._optimizer.rollback_optimization(backup_id)
            
            if success:
                return f"✓ Successfully rolled back to backup {backup_id}"
            else:
                return f"✗ Rollback failed: Backup {backup_id} not found"
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return f"Error: {e}"
    
    def get_optimization_history(
        self,
        limit: int = 10,
        successful_only: bool = False
    ) -> str:
        """
        Get history of optimization attempts.
        
        Args:
            limit: Maximum number of records to return
            successful_only: Whether to show only successful optimizations
            
        Returns:
            Human-readable history report
        """
        if not self._optimizer:
            return "Error: Optimizer not initialized"
        
        try:
            records = self._optimizer.get_optimization_history(limit, successful_only)
            
            if not records:
                return "No optimization history found."
            
            lines = [f"Optimization History (showing {len(records)} records):\n"]
            
            for i, record in enumerate(records, 1):
                status = "✓ SUCCESS" if record.success else "✗ FAILED"
                lines.append(f"{i}. [{record.timestamp.strftime('%Y-%m-%d %H:%M')}] {status}")
                lines.append(f"   Function: {record.function_name} in {record.module_path.name}")
                if record.backup_id:
                    lines.append(f"   Backup ID: {record.backup_id}")
                if not record.success and record.error_message:
                    lines.append(f"   Error: {record.error_message}")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get optimization history: {e}")
            return f"Error: {e}"
    
    def clear_optimization_cache(self) -> str:
        """
        Clear the cache of attempted optimizations.
        
        This allows the system to retry previously attempted optimizations.
        Useful after code has been manually fixed or 48 hours haven't passed yet.
        
        Returns:
            Success message with count of cleared entries
        """
        if not self._optimizer:
            return "Error: Optimizer not initialized"
        
        try:
            count = self._optimizer.clear_cache()
            return f"✓ Cleared {count} cached optimization attempts"
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return f"Error: {e}"
