
import sys
from pathlib import Path
from textwrap import dedent

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neuron_x.optimizer.core import SelfOptimizer
from neuron_x.plugins.optimizer import OptimizerPlugin

def test_skip_logic():
    print("=== Testing Skip Logic ===")
    
    # 1. Test core skip logic (automatic)
    optimizer = SelfOptimizer(project_root)
    
    # Create a module with NO test file
    test_module = project_root / "tmp" / "no_tests_module.py"
    test_module.parent.mkdir(parents=True, exist_ok=True)
    test_module.write_text("def my_func():\n    return 42")
    
    print("\nAttempting optimization of module with NO tests...")
    optimized_code = "def my_func():\n    return 43"
    
    record = optimizer.optimize_function(
        test_module,
        "my_func",
        optimized_code,
        require_tests=True # Should still skip because none found
    )
    
    assert record.success
    test_result = [r for r in record.validation_results if r.level.value == "tests"][0]
    print(f"✓ Core success: {record.success}")
    print(f"✓ Test result output: {test_result.output}")
    assert "Skipped" in test_result.output
    
    # 2. Test plugin reporting
    plugin = OptimizerPlugin()
    plugin.on_load()
    
    print("\nAttempting plugin optimization of module with NO tests...")
    result = plugin.optimize_module(
        "tmp/no_tests_module.py",
        "my_func",
        "def my_func():\n    return 44",
        require_tests=True
    )
    
    print(f"Plugin result:\n{result}")
    assert "Successfully optimized" in result
    assert "Tests skipped" in result
    
    # 3. Test manual bypass
    print("\nAttempting plugin optimization with manual require_tests=False...")
    result = plugin.optimize_module(
        "tmp/no_tests_module.py",
        "my_func",
        "def my_func():\n    return 45",
        require_tests=False
    )
    
    print(f"Plugin result:\n{result}")
    assert "Successfully optimized" in result
    assert "Tests manually bypassed" in result
    
    # Cleanup
    test_module.unlink()
    optimizer.cleanup()
    print("\n=== All Skip Logic Tests Passed ===")

if __name__ == "__main__":
    test_skip_logic()
