
import sys
import os
import shutil
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_x.cognition import CognitiveCore
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from models import GoalPriority, Goal

from pathlib import Path

class TestGoalFixes(unittest.TestCase):
    def setUp(self):
        self.test_path = Path("./tests/temp_goal_test")
        if self.test_path.exists():
            shutil.rmtree(self.test_path)
        self.test_path.mkdir(parents=True, exist_ok=True)
        
        self.smith = GraphSmith(self.test_path)
        self.vault = MagicMock(spec=VectorVault)
        self.vault.vector_cache = {}
        self.vault.cross_encoder = None
        self.core = CognitiveCore(self.smith, self.vault)

    def tearDown(self):
        if os.path.exists(self.test_path):
            shutil.rmtree(self.test_path)

    def test_add_goal_deduplication(self):
        # 1. Add a goal
        desc = "Test deduplication"
        self.core.add_goal(desc, priority=GoalPriority.MEDIUM)
        initial_count = len(self.core.goals)
        
        # 2. Add the same goal again
        self.core.add_goal(desc, priority=GoalPriority.MEDIUM)
        
        # Should still be the same
        self.assertEqual(len(self.core.goals), initial_count, "Goal count should not increase for exact duplicate description")

        # 3. Add a fuzzy duplicate
        fuzzy_desc = "  Test deduplication  "
        self.core.add_goal(fuzzy_desc, priority=GoalPriority.MEDIUM)
        self.assertEqual(len(self.core.goals), initial_count, "Goal count should not increase for whitespace variant")

        # 4. Add a "contained" duplicate
        contained_desc = "Self-improvement opportunity detected: Fix something"
        self.core.add_goal(contained_desc, priority=GoalPriority.MEDIUM)
        self.core.add_goal("Self-improvement opportunity detected", priority=GoalPriority.MEDIUM)
        self.assertTrue(any(g.description == contained_desc for g in self.core.goals))
        self.assertFalse(any(g.description == "Self-improvement opportunity detected" for g in self.core.goals), "Should have deduplicated the shorter containment")

    def test_get_bg_goal_persistence(self):
        # 1. Add a goal
        self.core.add_goal("Persistent Goal", priority=GoalPriority.HIGH)
        
        # 2. Pick it - should mark as IN_PROGRESS
        g1 = self.core.get_bg_goal()
        self.assertEqual(g1.description, "Persistent Goal")
        self.assertEqual(g1.status, "IN_PROGRESS")
        self.assertEqual(self.core.active_goal_id, g1.id)
        
        # 3. Pick again - should return the same one
        g2 = self.core.get_bg_goal()
        self.assertEqual(g1.id, g2.id, "Should persist with the same goal")
        
        # 4. Resolve it (simulated)
        g1.status = "COMPLETED"
        
        # 5. Pick again - should pick a different one (default goals)
        g3 = self.core.get_bg_goal()
        self.assertNotEqual(g1.id, g3.id, "Should pick a new goal after completion")
        self.assertEqual(self.core.active_goal_id, g3.id)

    def test_preventive_self_improvement_check(self):
        import os
        os.environ['AUTONOMOUS_OPT_PROBABILITY'] = '1.0' # Always trigger
        
        self.core.plugin_tools_getter = MagicMock(return_value={
            'list_optimization_opportunities': MagicMock(return_value="Found: Fix something")
        })
        self.core.llm_client = MagicMock()
        
        # Mock generate_content to return a simple thought
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="I am thinking.")]))]
        self.core.llm_client.models.generate_content.return_value = mock_response
        
        # Mock vault.encode
        self.core.vault.encode.return_value = __import__("numpy").zeros(384)
        
        # 1. Run thought cycle
        self.core.generate_proactive_thought()
        
        # Should have added a self-improvement goal
        self.assertTrue(any("Self-improvement" in g.description for g in self.core.goals))
        initial_goals = len(self.core.goals)
        
        # 2. Run again - should NOT add another self-improvement goal
        self.core.generate_proactive_thought()
        self.assertEqual(len(self.core.goals), initial_goals, "Should not add duplicate self-improvement goal")

if __name__ == "__main__":
    unittest.main()
