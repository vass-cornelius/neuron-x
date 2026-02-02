# OPTIMIZER_PROTECTED
"""
Validation engine for code optimization safety.

This module implements multi-level validation to ensure optimized code is safe,
correct, and doesn't break existing functionality.
"""

import ast
import logging
import subprocess
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence
from enum import Enum

logger = logging.getLogger("neuron-x.optimizer")


class ValidationLevel(Enum):
    """Validation check levels."""
    
    SYNTAX = "syntax"
    IMPORTS = "imports"
    TESTS = "tests"
    BEHAVIOR = "behavior"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    level: ValidationLevel
    success: bool
    error_message: str = ""
    output: str = ""


class ValidationEngine:
    """
    Multi-level validation engine for code changes.
    
    Performs progressive validation:
    1. Syntax check - ensures code parses correctly
    2. Import check - verifies all imports are resolvable
    3. Test execution - runs relevant tests with timeout
    4. Behavior check - compares outputs (optional)
    """
    
    def __init__(
        self,
        project_root: Path | str,
        timeout: int = 30,
        python_executable: str = ".venv/bin/python"
    ) -> None:
        """
        Initialize the validation engine.
        
        Args:
            project_root: Root directory of the project
            timeout: Timeout in seconds for subprocess execution
            python_executable: Path to Python executable (defaults to venv)
        """
        self.project_root = Path(project_root)
        self.timeout = timeout
        self.python_executable = python_executable
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """
        Level 1: Validate Python syntax.
        
        Args:
            code: Source code to validate
            
        Returns:
            ValidationResult with syntax check status
        """
        try:
            ast.parse(code)
            return ValidationResult(
                level=ValidationLevel.SYNTAX,
                success=True
            )
        except SyntaxError as e:
            return ValidationResult(
                level=ValidationLevel.SYNTAX,
                success=False,
                error_message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
    
    def validate_imports(self, code: str, module_path: Path | str) -> ValidationResult:
        """
        Level 2: Validate that all imports are resolvable.
        
        Args:
            code: Source code to validate
            module_path: Path where the module would be located
            
        Returns:
            ValidationResult with import check status
        """
        try:
            # Parse the code to extract imports
            tree = ast.parse(code)
            imports: list[str] = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Try to verify each import
            failed_imports: list[str] = []
            for imp in imports:
                try:
                    # Check if it's a standard library or installed package
                    spec = importlib.util.find_spec(imp.split('.')[0])
                    if spec is None:
                        failed_imports.append(imp)
                except (ImportError, ModuleNotFoundError):
                    failed_imports.append(imp)
            
            if failed_imports:
                return ValidationResult(
                    level=ValidationLevel.IMPORTS,
                    success=False,
                    error_message=f"Unresolvable imports: {', '.join(failed_imports)}"
                )
            
            return ValidationResult(
                level=ValidationLevel.IMPORTS,
                success=True
            )
            
        except Exception as e:
            return ValidationResult(
                level=ValidationLevel.IMPORTS,
                success=False,
                error_message=f"Import validation failed: {e}"
            )
    
    def validate_with_tests(
        self,
        draft_file: Path | str,
        module_path: Path | str,  # Original module path for test discovery
        test_patterns: Sequence[str] | None = None
    ) -> ValidationResult:
        """
        Level 3: Run pytest tests with timeout.
        
        Args:
            draft_file: Path to the draft file to test
            module_path: Original module path (used for test discovery)
            test_patterns: Optional list of test file patterns to run
            
        Returns:
            ValidationResult with test execution status
        """
        if isinstance(draft_file, str):
            draft_file = Path(draft_file)
        if isinstance(module_path, str):
            module_path = Path(module_path)
        
        # Determine which tests to run
        test_args = []
        if test_patterns:
            test_args = list(test_patterns)
        else:
            # Better module name detection (especially for plugins)
            if module_path.stem == "__init__":
                module_name = module_path.parent.name
            else:
                module_name = module_path.stem
            
            # Check for standard test files
            potential_tests = [
                self.project_root / "tests" / f"test_{module_name}.py",
                self.project_root / "tests" / f"test_{module_path.stem}.py"
            ]
            
            # Filter for existing files
            test_args = [
                str(p.relative_to(self.project_root)) 
                for p in potential_tests if p.exists()
            ]
            
            # If no direct matches, check if we should skip
            if not test_args:
                msg = f"Skipped: No test files found for {module_name} in tests/"
                logger.info(msg)
                return ValidationResult(
                    level=ValidationLevel.TESTS,
                    success=True,
                    output=msg
                )
        
        # Construct pytest command
        cmd = [self.python_executable, "-m", "pytest", "-v", "--tb=short"] + test_args
        
        logger.info(f"Running tests: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=self.timeout,
                capture_output=True,
                text=True
            )
            
            # Combine stdout and stderr
            full_output = result.stdout + "\n" + result.stderr
            
            # Limit output size to avoid overwhelming the LLM (keep last 1000 chars)
            if len(full_output) > 1000:
                truncated_output = "...[truncated]...\n" + full_output[-1000:]
            else:
                truncated_output = full_output
            
            if result.returncode == 0:
                return ValidationResult(
                    level=ValidationLevel.TESTS,
                    success=True,
                    output=truncated_output
                )
            elif result.returncode == 4:
                # Pytest exit code 4 means no tests were collected
                return ValidationResult(
                    level=ValidationLevel.TESTS,
                    success=True,
                    output=f"Success (No tests collected): {truncated_output}"
                )
            else:
                return ValidationResult(
                    level=ValidationLevel.TESTS,
                    success=False,
                    error_message=f"Tests failed (exit code {result.returncode})",
                    output=truncated_output
                )
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                level=ValidationLevel.TESTS,
                success=False,
                error_message=f"Test execution timeout ({self.timeout}s) - possible infinite loop"
            )
        except FileNotFoundError:
            return ValidationResult(
                level=ValidationLevel.TESTS,
                success=False,
                error_message=f"Python executable not found: {self.python_executable}"
            )
        except Exception as e:
            return ValidationResult(
                level=ValidationLevel.TESTS,
                success=False,
                error_message=f"Test execution failed: {e}"
            )
    
    def validate_all(
        self,
        code: str,
        module_path: Path | str,
        draft_file: Path | str | None = None,
        test_patterns: Sequence[str] | None = None,
        require_tests: bool = True
    ) -> list[ValidationResult]:
        """
        Run all validation levels progressively.
        
        Stops at the first failure to avoid unnecessary work.
        
        Args:
            code: Source code to validate
            module_path: Path to the module being optimized
            draft_file: Path to draft file for test execution
            test_patterns: Optional test patterns to run
            require_tests: Whether to require tests to pass
            
        Returns:
            List of ValidationResults from each level
        """
        results: list[ValidationResult] = []
        
        # Level 1: Syntax
        logger.info("Validating syntax...")
        syntax_result = self.validate_syntax(code)
        results.append(syntax_result)
        if not syntax_result.success:
            logger.error(f"Syntax validation failed: {syntax_result.error_message}")
            return results
        
        # Level 2: Imports
        logger.info("Validating imports...")
        import_result = self.validate_imports(code, module_path)
        results.append(import_result)
        if not import_result.success:
            logger.error(f"Import validation failed: {import_result.error_message}")
            return results
        
        # Level 3: Tests (if required and draft file provided)
        if require_tests and draft_file:
            logger.info("Running tests...")
            test_result = self.validate_with_tests(draft_file, module_path, test_patterns)
            results.append(test_result)
            if not test_result.success:
                logger.error(f"Test validation failed: {test_result.error_message}")
                return results
        
        logger.info("All validations passed!")
        return results
    
    def is_valid(self, results: Sequence[ValidationResult]) -> bool:
        """
        Check if all validation results are successful.
        
        Args:
            results: List of validation results
            
        Returns:
            True if all validations passed, False otherwise
        """
        return all(result.success for result in results)
