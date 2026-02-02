# Optimizer Plugin

Self-optimization plugin for NeuronX with Draft-Test-Commit workflow.

## Features

- **Code Analysis**: Identifies optimization opportunities (complexity, type hints, algorithms)
- **Safe Optimization**: Multi-level validation with AST-based modifications
- **Rollback**: Automatic backups with easy rollback
- **Audit Trail**: Complete history of all optimization attempts

## Tools Exposed

### `list_optimization_opportunities(module_path=None, limit=10)`
Scans codebase for optimization opportunities.

**Example**:
```
list_optimization_opportunities("neuron_x/cognition.py")
```

### `optimize_module(module_path, function_name, optimized_code, class_name=None)`
Executes Draft-Test-Commit workflow for a function optimization.

**Example**:
```
optimize_module(
    "neuron_x/cognition.py",
    "consolidate",
    "<optimized function code>",
    class_name="CognitiveCore"
)
```

### `rollback_optimization(backup_id)`
Rolls back to a previous backup.

### `get_optimization_history(limit=10, successful_only=False)`
Views optimization history.

## Safety Features

- Protected zones (cannot modify optimizer itself or bridge.py)
- 30-second timeout for test execution
- Multi-level validation (syntax → imports → tests)
- Automatic backups before any commit
- Malicious pattern detection

## See Also

- Implementation: `neuron_x/optimizer/`
- Plugin base: `neuron_x/plugin_base.py`
