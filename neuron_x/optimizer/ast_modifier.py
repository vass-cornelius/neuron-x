# OPTIMIZER_PROTECTED
"""
AST-based code modification for surgical function replacement.

This module uses Python's Abstract Syntax Tree (ast) module to perform
precise code modifications without disrupting the surrounding code structure.
"""

import ast
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("neuron-x.optimizer")


@dataclass
class ASTModificationResult:
    """Result of an AST modification operation."""
    
    success: bool
    modified_source: str = ""
    error_message: str = ""


class ASTModifier:
    """
    Performs surgical code modifications using Abstract Syntax Trees.
    
    This class can replace individual functions or class methods while
    preserving imports, global variables, and other code structure.
    Uses Python 3.9+ built-in ast.unparse for AST-to-source conversion.
    """
    
    def __init__(self) -> None:
        """Initialize the AST modifier."""
        pass
    
    def parse_module(self, source: str) -> ast.Module | None:
        """
        Parse Python source code into an AST.
        
        Args:
            source: Python source code
            
        Returns:
            AST Module node, or None if parsing fails
        """
        try:
            tree = ast.parse(source)
            return tree
        except SyntaxError as e:
            logger.error(f"Syntax error parsing source: {e}")
            return None
    
    def parse_function(self, function_source: str) -> ast.FunctionDef | None:
        """
        Parse a function definition from source code.
        
        Handles common LLM formatting issues:
        - Strips markdown code fences (```python)
        - Normalizes indentation (dedents to column 0)
        
        Args:
            function_source: Source code of a single function
            
        Returns:
            FunctionDef AST node, or None if parsing fails
        """
        import textwrap
        
        # Strip markdown code fences
        lines = function_source.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]  # Remove closing fence
        
        cleaned = '\n'.join(lines)
        
        # Normalize indentation (dedent to column 0)
        cleaned = textwrap.dedent(cleaned)
        
        # Unescape common LLM mistakes (escaped quotes, newlines, etc.)
        # LLMs sometimes escape quotes when they shouldn't
        cleaned = cleaned.replace(r'\"', '"')
        cleaned = cleaned.replace(r"\'", "'")
        cleaned = cleaned.replace(r'\n', '\n')  # Literal \n to actual newline
        
        try:
            tree = ast.parse(cleaned)
            
            # Extract the function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node
            
            logger.error("No function definition found in source")
            logger.debug(f"Parsed code (first 200 chars):\n{cleaned[:200]}")
            return None
            
        except SyntaxError as e:
            # Save failed code to file for debugging
            self._save_failed_code(cleaned, f"syntax_error_{e}")
            
            # Show detailed error with code context
            code_preview = cleaned[:300] if len(cleaned) > 300 else cleaned
            logger.error(f"Syntax error parsing function: {e}")
            logger.error(f"Problematic code preview:\n{code_preview}")
            logger.debug(f"Full code:\n{cleaned}")
            return None
    
    def _save_failed_code(self, code: str, reason: str) -> None:
        """
        Save failed optimization code to file for debugging.
        
        Args:
            code: The code that failed to parse
            reason: Brief description of why it failed
        """
        from datetime import datetime
        
        failed_dir = Path("tmp/optimizer/failed")
        failed_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Sanitize reason for filename
        safe_reason = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(reason)[:50])
        filename = failed_dir / f"failed_{timestamp}_{safe_reason}.py"
        
        try:
            filename.write_text(code, encoding='utf-8')
            logger.info(f"Saved failed code to {filename} for debugging")
        except Exception as ex:
            logger.warning(f"Failed to save debug file: {ex}")
    
    def replace_function(
        self,
        module_ast: ast.Module,
        func_name: str,
        new_func_ast: ast.FunctionDef,
        class_name: str | None = None
    ) -> bool:
        """
        Replace a function in the module AST.
        
        Args:
            module_ast: Module AST to modify
            func_name: Name of the function to replace
            new_func_ast: New function AST node
            class_name: If replacing a method, the class name
            
        Returns:
            True if replacement was successful, False otherwise
        """
        replaced = False
        
        if class_name:
            # Replace method in a class
            for node in module_ast.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.FunctionDef) and item.name == func_name:
                            node.body[i] = new_func_ast
                            replaced = True
                            logger.info(f"Replaced method {class_name}.{func_name}")
                            break
                    break
        else:
            # Replace top-level function
            for i, node in enumerate(module_ast.body):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    module_ast.body[i] = new_func_ast
                    replaced = True
                    logger.info(f"Replaced function {func_name}")
                    break
        
        if not replaced:
            target = f"{class_name}.{func_name}" if class_name else func_name
            logger.warning(f"Function {target} not found for replacement")
        
        return replaced
    
    def validate_ast(self, tree: ast.Module) -> tuple[bool, str]:
        """
        Validate that an AST is well-formed.
        
        Args:
            tree: AST Module to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to compile the AST
            compile(tree, "<ast>", "exec")
            return True, ""
        except Exception as e:
            return False, f"AST validation failed: {e}"
    
    def generate_source(self, tree: ast.Module) -> str | None:
        """
        Convert an AST back to Python source code.
        
        Uses ast.unparse (Python 3.9+) which is built-in.
        
        Args:
            tree: AST Module to convert
            
        Returns:
            Python source code, or None if conversion fails
        """
        try:
            # Use built-in ast.unparse (Python 3.9+)
            source = ast.unparse(tree)
            return source
        except Exception as e:
            logger.error(f"Failed to convert AST to source: {e}")
            return None
    
    def modify_function(
        self,
        original_source: str,
        func_name: str,
        new_func_source: str,
        class_name: str | None = None
    ) -> ASTModificationResult:
        """
        High-level function replacement operation.
        
        This method orchestrates the entire replacement process:
        1. Parse original source to AST
        2. Parse new function to AST
        3. Replace the function in the tree
        4. Validate the modified AST
        5. Generate source from modified AST
        
        Args:
            original_source: Original module source code
            func_name: Name of function to replace
            new_func_source: Source code of new function
            class_name: Optional class name if replacing a method
            
        Returns:
            ASTModificationResult with success status and modified source
        """
        # Parse original module
        module_ast = self.parse_module(original_source)
        if module_ast is None:
            return ASTModificationResult(
                success=False,
                error_message="Failed to parse original source"
            )
        
        # Parse new function
        new_func_ast = self.parse_function(new_func_source)
        if new_func_ast is None:
            return ASTModificationResult(
                success=False,
                error_message="Failed to parse new function"
            )
        
        # Replace function in AST
        if not self.replace_function(module_ast, func_name, new_func_ast, class_name):
            return ASTModificationResult(
                success=False,
                error_message=f"Failed to find and replace function {func_name}"
            )
        
        # Validate modified AST
        is_valid, validation_error = self.validate_ast(module_ast)
        if not is_valid:
            return ASTModificationResult(
                success=False,
                error_message=validation_error
            )
        
        # Generate source code
        modified_source = self.generate_source(module_ast)
        if modified_source is None:
            return ASTModificationResult(
                success=False,
                error_message="Failed to generate source from AST"
            )
        
        return ASTModificationResult(
            success=True,
            modified_source=modified_source
        )
