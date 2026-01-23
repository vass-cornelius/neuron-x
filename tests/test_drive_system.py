
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_x import NeuronX
from models import GoalPriority
import shutil

def test_drive_system():
    # Setup - use a temp path
    test_path = "./tests/temp_brain"
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    
    print("Initializing NeuronX with Drive System...")
    brain = NeuronX(persistence_path=test_path)
    
    # 1. Test Default Goals
    print("\n[TEST 1] Default Goals")
    assert len(brain.goals) == 2, f"Expected 2 default goals, found {len(brain.goals)}"
    print("PASS: Default goals initialized.")
    
    # 2. Test Goal Addition
    print("\n[TEST 2] Add Goal")
    brain.add_goal("Find the Holy Grail", priority=GoalPriority.CRITICAL)
    assert len(brain.goals) == 3
    print("PASS: Goal added.")
    
    # 3. Test Goal Prioritization (Probabilistic)
    print("\n[TEST 3] Goal Prioritization (Stochastic)")
    # Add a low priority goal
    brain.add_goal("Sweep the floor", priority=GoalPriority.LOW)
    
    # Run Monte Carlo simulation
    results = {"CRITICAL": 0, "LOW": 0}
    for _ in range(100):
        g = brain.get_bg_goal()
        # We only care about the ones we added for this test
        if g.description == "Find the Holy Grail":
            results["CRITICAL"] += 1
        elif g.description == "Sweep the floor":
            results["LOW"] += 1
            
    print(f"Distribution (n=100): {results}")
    
    # CRITICAL (weight 10) vs LOW (weight 1) -> Should be roughly 10:1 ratio
    # but there are other default goals too.
    # Just ensure CRITICAL > LOW
    assert results["CRITICAL"] > results["LOW"], f"CRITICAL ({results['CRITICAL']}) should be more frequent than LOW ({results['LOW']})"
    print("PASS: Higher priority selected more often.")
    
    top_goal = brain.goals[2] # manually grab the critical one for next test
    top_goal_id = top_goal.id
    
    # 4. Test Completion logic (Simulated)
    # This just tests getting the next one if we change status
    # We simulate the REGEX logic manually
    # In the real app, ">> GOAL RESOLVED:" triggers this
    for g in brain.goals:
        if g.id == top_goal_id:
            g.status = "COMPLETED"
    
    next_goal = brain.get_bg_goal()
    print(f"Goal '{top_goal.description}' Status: {top_goal.status}")
    print(f"Next Goal: {next_goal.description} [{next_goal.priority}]")
    
    # Should NOT be the completed one
    assert top_goal.status == "COMPLETED"
    assert next_goal.description != "Find the Holy Grail"
    print("PASS: Completed goal skipped.")

    # Cleanup
    shutil.rmtree(test_path)
    print("\nAll Drive System tests passed!")

if __name__ == "__main__":
    test_drive_system()
