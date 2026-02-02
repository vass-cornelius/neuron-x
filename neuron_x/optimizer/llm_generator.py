# OPTIMIZER_PROTECTED
"""
LLM-based code generation for optimizations.

This module uses the LLM to generate improved code based on
identified optimization opportunities.
"""

import logging
from pathlib import Path
from typing import Any
from textwrap import dedent

from google import genai
from google.genai import types

from neuron_x.optimizer.analyzer import OptimizationOpportunity

logger = logging.getLogger("neuron-x.optimizer")


class LLMCodeGenerator:
    """
    Uses LLM to generate optimized code based on opportunities.
    
    This class takes an optimization opportunity and the original code,
    then asks the LLM to generate an improved version following best practices.
    """
    
    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the code generator.
        
        Args:
            llm_client: Google GenAI client for code generation
        """
        self.llm_client = llm_client
    
    def generate_optimized_code(
        self,
        opportunity: OptimizationOpportunity,
        original_code: str
    ) -> str | None:
        """
        Generate optimized code for a given opportunity.
        
        Args:
            opportunity: The identified optimization opportunity
            original_code: The complete original module source code
            
        Returns:
            Generated optimized function code, or None if generation fails
        """
        if not self.llm_client:
            logger.error("LLM client not available for code generation")
            return None
        
        # Extract the specific function to optimize
        function_context = self._extract_function_context(
            original_code,
            opportunity.function_name,
            opportunity.class_name
        )
        
        if not function_context:
            logger.error(f"Could not extract context for {opportunity.function_name}")
            return None
        
        # Build the optimization prompt
        prompt = self._build_optimization_prompt(opportunity, function_context)
        
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temp for more consistent code
                    max_output_tokens=2000,
                    system_instruction=dedent("""
                        You are an expert Python developer specializing in code optimization.
                        
                        CRITICAL RULES:
                        1. Return ONLY the optimized function code (no markdown, no explanations)
                        2. Preserve the function signature exactly (name, parameters)
                        3. Maintain all existing functionality
                        4. Add type hints if missing
                        5. Add docstrings if missing
                        6. Follow PEP 8 and Python best practices
                        7. Do NOT include import statements (they will be preserved separately)
                        8. Do NOT add any text before or after the function
                        
                        Your output should start with 'def' or '@' (for decorators).
                    """).strip()
                )
            )
            
            if not response.text:
                logger.error("LLM returned empty response for code generation")
                return None
            
            generated_code = response.text.strip()
            
            # Validate that it looks like a function
            if not (generated_code.startswith("def ") or generated_code.startswith("@")):
                logger.warning(f"Generated code doesn't look like a function: {generated_code[:100]}")
                # Try to extract if wrapped in markdown
                if "```python" in generated_code:
                    parts = generated_code.split("```python")
                    if len(parts) > 1:
                        generated_code = parts[1].split("```")[0].strip()
                elif "```" in generated_code:
                    parts = generated_code.split("```")
                    if len(parts) >= 3:
                        generated_code = parts[1].strip()
            
            logger.info(f"Generated optimized code for {opportunity.function_name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate optimized code: {e}", exc_info=True)
            return None
    
    def _extract_function_context(
        self,
        source_code: str,
        function_name: str,
        class_name: str | None = None
    ) -> str | None:
        """
        Extract the specific function from the source code.
        
        Args:
            source_code: Full module source
            function_name: Name of the function
            class_name: Optional class name if it's a method
            
        Returns:
            The function code, or None if not found
        """
        import ast
        
        try:
            tree = ast.parse(source_code)
            
            # Search for the function
            for node in ast.walk(tree):
                if class_name:
                    # Looking for a method
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == function_name:
                                # Get the source lines
                                lines = source_code.split('\n')
                                start_line = item.lineno - 1
                                end_line = item.end_lineno if hasattr(item, 'end_lineno') else start_line + 10
                                return '\n'.join(lines[start_line:end_line])
                else:
                    # Looking for a top-level function
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        lines = source_code.split('\n')
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                        return '\n'.join(lines[start_line:end_line])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract function context: {e}")
            return None
    
    def _build_optimization_prompt(
        self,
        opportunity: OptimizationOpportunity,
        function_code: str
    ) -> str:
        """
        Build the prompt for code optimization.
        
        Args:
            opportunity: The optimization opportunity
            function_code: The current function code
            
        Returns:
            The complete prompt
        """
        opp_type = opportunity.opportunity_type.value
        suggestions = "\n".join(f"- {s}" for s in opportunity.suggestions) if opportunity.suggestions else "None"
        
        prompt = dedent(f"""
            OPTIMIZATION TASK:
            
            Current Function:
            ```python
            {function_code}
            ```
            
            Issue Identified: {opportunity.description}
            Optimization Type: {opp_type}
            Priority: {opportunity.priority}/10
            
            Suggested Improvements:
            {suggestions}
            
            Please provide an optimized version of this function that addresses the identified issues.
            Remember: Return ONLY the optimized function code.
        """).strip()
        
        return prompt
