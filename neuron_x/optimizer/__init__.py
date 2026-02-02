# OPTIMIZER_PROTECTED
"""
Self-Optimization System for NeuronX.

This module implements a safe self-optimization system with Draft-Test-Commit workflow.
The optimizer can analyze code, propose improvements, validate changes, and safely commit
optimizations while maintaining multiple safety mechanisms.

Key Components:
- SelfOptimizer: Main orchestrator for the optimization workflow
- SafetyConfig: Configuration and enforcement of safety rules
- CodeAnalyzer: Identifies optimization opportunities
- DraftManager: Manages temporary code versions
- ValidationEngine: Multi-level validation of changes
- CommitManager: Atomic commits with rollback capability
- ASTModifier: Surgical code modifications using AST
"""

from neuron_x.optimizer.core import SelfOptimizer
from neuron_x.optimizer.safety import SafetyConfig, SafetyEnforcer

__all__ = ["SelfOptimizer", "SafetyConfig", "SafetyEnforcer"]
