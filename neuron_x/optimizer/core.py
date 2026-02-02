# OPTIMIZER_PROTECTED
"""
Core self-optimization orchestrator.

This module implements the main SelfOptimizer class that coordinates the
entire Draft-Test-Commit workflow for code optimization.
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timedelta

from neuron_x.optimizer.safety import SafetyConfig, SafetyEnforcer
from neuron_x.optimizer.draft_manager import DraftManager, Draft
from neuron_x.optimizer.commit_manager import CommitManager, CommitResult
from neuron_x.optimizer.validator import ValidationEngine, ValidationResult
from neuron_x.optimizer.analyzer import CodeAnalyzer, OptimizationOpportunity
from neuron_x.optimizer.ast_modifier import ASTModifier, ASTModificationResult
from neuron_x.optimizer.llm_generator import LLMCodeGenerator

logger = logging.getLogger("neuron-x.optimizer")


@dataclass
class OptimizationRecord:
    """Record of an optimization attempt."""
    
    module_path: Path
    function_name: str
    timestamp: datetime
    success: bool
    draft_id: str | None = None
    backup_id: str | None = None
    error_message: str = ""
    validation_results: list[ValidationResult] | None = None


class SelfOptimizer:
    """
    Main orchestrator for the self-optimization system.
    
    This class coordinates the Draft-Test-Commit workflow:
    1. Analysis: Identify optimization opportunities
    2. Drafting: Generate improved code (via LLM or manual input)
    3. Validation: Multi-level safety checks
    4. Commit: Atomic code deployment with backup
    
    The optimizer enforces strict safety rules to prevent system corruption.
    """
    
    def __init__(
        self,
        project_root: Path | str,
        safety_config: SafetyConfig | None = None,
        llm_client: Any | None = None
    ) -> None:
        """
        Initialize the self-optimizer.
        
        Args:
            project_root: Root directory of the project
            safety_config: Safety configuration (uses defaults if not provided)
            llm_client: Optional LLM client for generating optimizations
        """
        self.project_root = Path(project_root)
        
        # Initialize components
        self.safety = SafetyEnforcer(safety_config or SafetyConfig(project_root=self.project_root))
        self.draft_manager = DraftManager()
        self.commit_manager = CommitManager(retention_days=self.safety.config.backup_retention_days)
        self.validator = ValidationEngine(self.project_root, timeout=self.safety.get_timeout())
        self.analyzer = CodeAnalyzer(self.project_root)
        self.ast_modifier = ASTModifier()
        self.llm_client = llm_client
        
        # LLM code generator (if client available)
        self.code_generator = LLMCodeGenerator(llm_client) if llm_client else None
        
        # Audit log
        self._optimization_history: list[OptimizationRecord] = []
        
        # Cache for attempted optimizations
        self._cache_file = self.project_root / "tmp" / "optimizer" / "attempted_cache.json"
        self._attempted_cache: dict[str, dict] = self._load_cache()
    
    def identify_opportunities(
        self,
        module_path: Path | str | None = None
    ) -> list[OptimizationOpportunity]:
        """
        Scan for optimization opportunities.
        
        Args:
            module_path: Specific module to scan, or None to scan entire project
            
        Returns:
            List of identified optimization opportunities
        """
        if module_path:
            # Scan specific module
            module_path = Path(module_path)
            if not module_path.is_absolute():
                module_path = self.project_root / module_path
            
            # Check if protected
            if self.safety.is_protected(module_path):
                logger.warning(f"Cannot analyze protected module: {module_path}")
                return []
            
            return self.analyzer.scan_module(module_path)
        else:
            # Scan entire project
            results = self.analyzer.scan_directory(
                self.project_root / "neuron_x",
                exclude_patterns=["test_", "__pycache__", ".venv", "optimizer"]
            )
            
            # Flatten results
            all_opportunities: list[OptimizationOpportunity] = []
            for opportunities in results.values():
                all_opportunities.extend(opportunities)
            
            # Sort by priority
            all_opportunities.sort(key=lambda x: x.priority, reverse=True)
            
            # Filter out recently attempted optimizations
            all_opportunities = self._filter_attempted(all_opportunities)
            
            return all_opportunities
    
    def _load_cache(self) -> dict[str, dict]:
        """Load the attempted optimizations cache from disk."""
        if not self._cache_file.exists():
            return {}
        
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Clean up expired entries using configured expiry time
            expiry_hours = self.safety.config.cache_expiry_hours
            cutoff = (datetime.now() - timedelta(hours=expiry_hours)).isoformat()
            cache = {
                k: v for k, v in cache.items()
                if v.get('timestamp', '') > cutoff
            }
            
            logger.debug(f"Loaded {len(cache)} attempted optimizations from cache (expiry: {expiry_hours}h)")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load optimization cache: {e}")
            return {}
    
    def _save_cache(self) -> None:
        """Save the attempted optimizations cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._attempted_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save optimization cache: {e}")
    
    def _get_cache_key(self, opp: OptimizationOpportunity) -> str:
        """Generate a unique cache key for an optimization opportunity."""
        # Format: module:function:class:opportunity_type
        class_part = f":{opp.class_name}" if opp.class_name else ""
        return f"{opp.module_path.name}:{opp.function_name}{class_part}:{opp.opportunity_type.value}"
    
    def _filter_attempted(self, opportunities: list[OptimizationOpportunity]) -> list[OptimizationOpportunity]:
        """Filter out recently attempted optimizations."""
        filtered = []
        for opp in opportunities:
            key = self._get_cache_key(opp)
            if key not in self._attempted_cache:
                filtered.append(opp)
            else:
                attempt_data = self._attempted_cache[key]
                logger.debug(f"Filtered out previously attempted: {key} (last attempt: {attempt_data.get('timestamp')})")
        
        if len(filtered) < len(opportunities):
            logger.info(f"Filtered {len(opportunities) - len(filtered)} previously attempted optimizations")
        
        return filtered
    
    def _record_attempt(self, opp: OptimizationOpportunity, success: bool) -> None:
        """Record an optimization attempt in the cache."""
        key = self._get_cache_key(opp)
        self._attempted_cache[key] = {
            "timestamp": datetime.now().isoformat(),
            "attempts": self._attempted_cache.get(key, {}).get("attempts", 0) + 1,
            "last_result": "success" if success else "failed"
        }
        self._save_cache()
    
    def clear_cache(self) -> int:
        """Clear the attempted optimizations cache. Returns number of entries cleared."""
        count = len(self._attempted_cache)
        self._attempted_cache = {}
        self._save_cache()
        logger.info(f"Cleared {count} entries from optimization cache")
        return count
    
    def optimize_function(
        self,
        module_path: Path | str,
        function_name: str,
        optimized_code: str,
        class_name: str | None = None,
        use_ast: bool = True,
        require_tests: bool | None = None
    ) -> OptimizationRecord:
        """
        Execute the Draft-Test-Commit workflow for a function optimization.
        
        This is the main entry point for optimization operations.
        
        Args:
            module_path: Path to the module to optimize
            function_name: Name of the function to optimize
            optimized_code: The optimized function code
            class_name: Class name if optimizing a method
            use_ast: Whether to use AST-based modification (safer)
            require_tests: Whether to require tests to pass (overrides safety config)
            
        Returns:
            OptimizationRecord with details of the attempt
        """
        if isinstance(module_path, str):
            module_path = Path(module_path)
        
        if not module_path.is_absolute():
            module_path = self.project_root / module_path
        
        logger.info(f"Starting optimization of {function_name} in {module_path}")
        
        # Create optimization record
        record = OptimizationRecord(
            module_path=module_path,
            function_name=function_name,
            timestamp=datetime.now(),
            success=False
        )
        
        try:
            # Step 1: Safety Pre-Check
            is_valid, error_msg = self.safety.validate_optimization_request(
                module_path, optimized_code
            )
            if not is_valid:
                record.error_message = error_msg
                logger.error(f"Safety check failed: {error_msg}")
                self._optimization_history.append(record)
                return record
            
            # Step 2: Read Original Code
            if not module_path.exists():
                record.error_message = f"Module not found: {module_path}"
                logger.error(record.error_message)
                self._optimization_history.append(record)
                return record
            
            original_code = module_path.read_text(encoding='utf-8')
            
            # Step 3: Generate Modified Code
            if use_ast:
                # Use AST for surgical replacement
                modification_result = self.ast_modifier.modify_function(
                    original_code, function_name, optimized_code, class_name
                )
                
                if not modification_result.success:
                    record.error_message = f"AST modification failed: {modification_result.error_message}"
                    logger.error(record.error_message)
                    self._optimization_history.append(record)
                    return record
                
                final_code = modification_result.modified_source
            else:
                # Use full replacement (less safe)
                final_code = optimized_code
            
            # Step 4: Create Draft
            draft = self.draft_manager.create_draft(
                module_path, original_code, final_code
            )
            record.draft_id = draft.draft_id
            
            # Step 5: Validate
            logger.info("Validating optimized code...")
            draft_paths = self.draft_manager.get_draft_paths(draft.draft_id)
            if not draft_paths:
                record.error_message = "Failed to get draft paths"
                logger.error(record.error_message)
                self._optimization_history.append(record)
                return record
            
            # Determine if tests are required (parameter overrides config)
            should_run_tests = require_tests if require_tests is not None else self.safety.requires_tests()
            
            validation_results = self.validator.validate_all(
                final_code,
                module_path,
                draft_file=draft_paths.optimized,
                require_tests=should_run_tests
            )
            record.validation_results = validation_results
            
            if not self.validator.is_valid(validation_results):
                # Validation failed - collect errors
                errors = [
                    f"{r.level.value}: {r.error_message}"
                    for r in validation_results if not r.success
                ]
                record.error_message = "; ".join(errors)
                logger.error(f"Validation failed: {record.error_message}")
                self.draft_manager.cleanup_draft(draft.draft_id)
                self._optimization_history.append(record)
                return record
            
            # Step 6: Commit
            logger.info("Committing optimized code...")
            commit_result = self.commit_manager.commit_changes(
                module_path, final_code, create_backup=True
            )
            
            if not commit_result.success:
                record.error_message = commit_result.error_message
                logger.error(f"Commit failed: {commit_result.error_message}")
                self.draft_manager.cleanup_draft(draft.draft_id)
                self._optimization_history.append(record)
                return record
            
            # Success!
            record.success = True
            record.backup_id = commit_result.backup_id
            logger.info(f"Successfully optimized {function_name} in {module_path}")
            
            # Cleanup draft
            self.draft_manager.cleanup_draft(draft.draft_id)
            
        except Exception as e:
            record.error_message = f"Unexpected error: {e}"
            logger.error(f"Optimization failed with exception: {e}", exc_info=True)
        
        finally:
            self._optimization_history.append(record)
        
        return record
    
    def rollback_optimization(self, backup_id: str) -> bool:
        """
        Rollback to a previous backup.
        
        Args:
            backup_id: ID of the backup to restore
            
        Returns:
            True if rollback was successful, False otherwise
        """
        backup_info = self.commit_manager.get_backup_info(backup_id)
        if not backup_info:
            logger.error(f"Backup {backup_id} not found")
            return False
        
        return self.commit_manager.rollback(backup_info.original_path, backup_id)
    
    def get_optimization_history(
        self,
        limit: int | None = None,
        successful_only: bool = False
    ) -> list[OptimizationRecord]:
        """
        Get optimization history.
        
        Args:
            limit: Maximum number of records to return
            successful_only: Whether to return only successful optimizations
            
        Returns:
            List of optimization records
        """
        records = self._optimization_history
        
        if successful_only:
            records = [r for r in records if r.success]
        
        # Sort by timestamp (newest first)
        records = sorted(records, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            records = records[:limit]
        
        return records
    
    def auto_optimize_opportunity(
        self,
        opportunity: OptimizationOpportunity
    ) -> OptimizationRecord:
        """
        Automatically generate and apply an optimization using LLM.
        
        This is a convenience method that:
        1. Reads the original code
        2. Uses LLM to generate optimized version
        3. Executes the full Draft-Test-Commit workflow
        
        Args:
            opportunity: The optimization opportunity to address
            
        Returns:
            OptimizationRecord with result of the optimization attempt
        """
        if not self.code_generator:
            logger.error("Cannot auto-optimize without LLM code generator")
            return OptimizationRecord(
                module_path=opportunity.module_path,
                function_name=opportunity.function_name,
                timestamp=datetime.now(),
                success=False,
                error_message="LLM code generator not available"
            )
        
        logger.info(f"Auto-optimizing {opportunity.function_name} in {opportunity.module_path.name}")
        
        # Read original code
        try:
            original_code = opportunity.module_path.read_text(encoding='utf-8')
        except Exception as e:
            record = OptimizationRecord(
                module_path=opportunity.module_path,
                function_name=opportunity.function_name,
                timestamp=datetime.now(),
                success=False,
                error_message=f"Failed to read file: {e}"
            )
            self._record_attempt(opportunity, success=False)
            return record
        
        # Generate optimized code using LLM
        optimized_code = self.code_generator.generate_optimized_code(
            opportunity,
            original_code
        )
        
        if not optimized_code:
            record = OptimizationRecord(
                module_path=opportunity.module_path,
                function_name=opportunity.function_name,
                timestamp=datetime.now(),
                success=False,
                error_message="LLM failed to generate optimized code"
            )
            self._record_attempt(opportunity, success=False)
            return record
        
        # Execute optimization workflow
        record = self.optimize_function(
            module_path=opportunity.module_path,
            function_name=opportunity.function_name,
            optimized_code=optimized_code,
            class_name=opportunity.class_name,
            use_ast=True
        )
        
        # Record the attempt in cache
        self._record_attempt(opportunity, success=record.success)
        
        return record
    
    def cleanup(self) -> None:
        """Clean up temporary files and old backups."""
        self.draft_manager.cleanup_all()
        self.commit_manager.cleanup_old_backups()
