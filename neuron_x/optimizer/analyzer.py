# OPTIMIZER_PROTECTED
"""
Code analyzer for identifying optimization opportunities.

This module scans Python code to identify potential improvements like
complexity reduction, missing type hints, inefficient algorithms, etc.
"""

import ast
import inspect
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Sequence, Callable, Any
from enum import Enum

logger = logging.getLogger("neuron-x.optimizer")


class OpportunityType(Enum):
    """Types of optimization opportunities."""
    
    HIGH_COMPLEXITY = "high_complexity"
    MISSING_TYPE_HINTS = "missing_type_hints"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    CODE_DUPLICATION = "code_duplication"
    PERFORMANCE = "performance"


@dataclass
class OptimizationOpportunity:
    """
    Represents a potential code optimization.
    
    Attributes:
        opportunity_type: Type of optimization
        module_path: Path to the module
        function_name: Name of the function/method to optimize
        class_name: Class name if it's a method
        priority: Priority score (1-10, higher = more important)
        description: Human-readable description
        line_number: Line number where the issue starts
        suggestions: List of suggested improvements
    """
    
    opportunity_type: OpportunityType
    module_path: Path
    function_name: str
    priority: int
    description: str
    class_name: str | None = None
    line_number: int | None = None
    suggestions: list[str] = field(default_factory=list)


class CodeAnalyzer:
    """
    Analyzes Python code to identify optimization opportunities.
    
    Uses AST analysis and static code inspection to find potential improvements.
    """
    
    def __init__(self, project_root: Path | str) -> None:
        """
        Initialize the code analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
    
    def scan_module(self, module_path: Path | str) -> list[OptimizationOpportunity]:
        """
        Scan a Python module for optimization opportunities.
        
        Args:
            module_path: Path to the module to analyze
            
        Returns:
            List of identified optimization opportunities
        """
        if isinstance(module_path, str):
            module_path = Path(module_path)
        
        if not module_path.exists():
            logger.error(f"Module not found: {module_path}")
            return []
        
        try:
            source = module_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception as e:
            logger.error(f"Failed to parse {module_path}: {e}")
            return []
        
        opportunities: list[OptimizationOpportunity] = []
        
        # Analyze all functions and methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                class_name = self._get_parent_class(tree, node)
                func_opportunities = self._analyze_function(
                    module_path, node, class_name
                )
                opportunities.extend(func_opportunities)
        
        # Sort by priority (highest first)
        opportunities.sort(key=lambda x: x.priority, reverse=True)
        
        # Enhanced logging - show top opportunities
        if opportunities:
            logger.info(f"Found {len(opportunities)} optimization opportunities in {module_path}")
            for i, opp in enumerate(opportunities[:3], 1):
                target = f"{opp.class_name}.{opp.function_name}" if opp.class_name else opp.function_name
                logger.info(f"  #{i} [{opp.priority}/10] {target}: {opp.opportunity_type.value}")
            if len(opportunities) > 3:
                logger.info(f"  ... and {len(opportunities) - 3} more")
        return opportunities
    
    def scan_directory(
        self,
        directory: Path | str,
        exclude_patterns: Sequence[str] | None = None
    ) -> dict[Path, list[OptimizationOpportunity]]:
        """
        Recursively scan a directory for optimization opportunities.
        
        Args:
            directory: Directory to scan
            exclude_patterns: Patterns to exclude (e.g., "test_", "__pycache__")
            
        Returns:
            Dictionary mapping file paths to their opportunities
        """
        if isinstance(directory, str):
            directory = Path(directory)
        
        exclude_patterns = exclude_patterns or ["test_", "__pycache__", ".venv"]
        results: dict[Path, list[OptimizationOpportunity]] = {}
        
        for py_file in directory.rglob("*.py"):
            # Check exclusions
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            
            opportunities = self.scan_module(py_file)
            if opportunities:
                results[py_file] = opportunities
        
        return results
    
    def _analyze_function(
        self,
        module_path: Path,
        func_node: ast.FunctionDef,
        class_name: str | None
    ) -> list[OptimizationOpportunity]:
        """Analyze a function for optimization opportunities."""
        opportunities: list[OptimizationOpportunity] = []
        
        # Check complexity
        complexity = self._calculate_complexity(func_node)
        if complexity > 10:
            opportunities.append(OptimizationOpportunity(
                opportunity_type=OpportunityType.HIGH_COMPLEXITY,
                module_path=module_path,
                function_name=func_node.name,
                class_name=class_name,
                priority=min(complexity // 2, 10),
                description=f"High cyclomatic complexity ({complexity})",
                line_number=func_node.lineno,
                suggestions=[
                    "Extract helper functions",
                    "Simplify conditional logic",
                    "Reduce nesting levels"
                ]
            ))
        
        # Check type hints
        if not self._has_type_hints(func_node):
            opportunities.append(OptimizationOpportunity(
                opportunity_type=OpportunityType.MISSING_TYPE_HINTS,
                module_path=module_path,
                function_name=func_node.name,
                class_name=class_name,
                priority=3,
                description="Missing type hints",
                line_number=func_node.lineno,
                suggestions=["Add type hints to parameters and return value"]
            ))
        
        # Check for nested loops (potential O(n²) issues)
        if self._has_nested_loops(func_node):
            opportunities.append(OptimizationOpportunity(
                opportunity_type=OpportunityType.INEFFICIENT_ALGORITHM,
                module_path=module_path,
                function_name=func_node.name,
                class_name=class_name,
                priority=7,
                description="Contains nested loops - potential O(n²) complexity",
                line_number=func_node.lineno,
                suggestions=[
                    "Consider using set/dict for O(1) lookups",
                    "Use list comprehensions where appropriate",
                    "Investigate vectorized operations if using numpy"
                ]
            ))
        
        # Check error handling
        if not self._has_error_handling(func_node):
            opportunities.append(OptimizationOpportunity(
                opportunity_type=OpportunityType.MISSING_ERROR_HANDLING,
                module_path=module_path,
                function_name=func_node.name,
                class_name=class_name,
                priority=4,
                description="No error handling detected",
                line_number=func_node.lineno,
                suggestions=["Add try/except blocks for error handling"]
            ))
        
        return opportunities
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Each branch point adds to complexity
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _has_type_hints(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        # Check return type
        has_return = func_node.returns is not None
        
        # Check argument types
        has_args = all(
            arg.annotation is not None
            for arg in func_node.args.args
            if arg.arg != 'self'
        )
        
        return has_return or has_args
    
    def _has_nested_loops(self, func_node: ast.FunctionDef) -> bool:
        """Check if function contains nested loops."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                # Check if there's a loop inside this loop
                for child in ast.walk(node):
                    if child is not node and isinstance(child, (ast.For, ast.While)):
                        return True
        return False
    
    def _has_error_handling(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has any try/except blocks."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Try):
                return True
        return False
    
    def _get_parent_class(
        self,
        tree: ast.Module,
        func_node: ast.FunctionDef
    ) -> str | None:
        """Find the parent class of a function, if any."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return node.name
        return None
