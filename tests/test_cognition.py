"""
Tests for neuron_x.cognition module (CognitiveCore).
"""
import pytest
import networkx as nx
from pathlib import Path
from neuron_x.cognition import CognitiveCore
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from models import Goal

@pytest.fixture
def temp_storage_path(tmp_path):
    """Create a temporary storage directory."""
    return tmp_path / "test_cognition"

@pytest.fixture
def graph_smith(temp_storage_path):
    """Create a GraphSmith instance."""
    return GraphSmith(temp_storage_path)

@pytest.fixture
def vector_vault():
    """Create a VectorVault instance."""
    return VectorVault()

@pytest.fixture
def cognitive_core(graph_smith, vector_vault):
    """Create a CognitiveCore instance for testing."""
    return CognitiveCore(
        persistence=graph_smith,
        memory=vector_vault,
        llm_client=None  # No LLM needed for basic tests
    )


class TestCognitiveInit:
    """Test CognitiveCore initialization."""
    
    def test_init_creates_instance(self, cognitive_core):
        """Test that CognitiveCore initializes correctly."""
        assert cognitive_core is not None
        assert isinstance(cognitive_core.goals, list)
    
    def test_init_loads_existing_graph(self, graph_smith, vector_vault):
        """Test that initialization loads existing graph."""
        # Create and save a graph
        test_graph = nx.DiGraph()
        test_graph.add_node("TestNode", type="CONCEPT")
        graph_smith.save_graph(test_graph)
        
        # Create cognitive core - should load the graph
        core = CognitiveCore(graph_smith, vector_vault)
        graph = core.smith.load_graph()
        assert "TestNode" in graph.nodes()


class TestPerception:
    """Test perception functionality."""
    
    def test_perceive_adds_to_buffer(self, cognitive_core):
        """Test that perceive adds memories to working_memory."""
        initial_size = len(cognitive_core.working_memory)
        cognitive_core.perceive("Test memory text")
        
        assert len(cognitive_core.working_memory) > initial_size
        assert any("Test memory text" in m.get("text", "") for m in cognitive_core.working_memory)


class TestGoalManagement:
    """Test goal/drive system."""
    
    def test_add_goal(self, cognitive_core):
        """Test adding a new goal."""
        initial_count = len(cognitive_core.goals)
        from neuron_x.const import GoalPriority
        cognitive_core.add_goal("Test goal description", priority=GoalPriority.HIGH)
        
        assert len(cognitive_core.goals) == initial_count + 1
        assert any(g.description == "Test goal description" for g in cognitive_core.goals)
    
    def test_get_bg_goal_returns_goal(self, cognitive_core):
        """Test that get_bg_goal returns a goal when available."""
        from neuron_x.const import GoalPriority
        cognitive_core.add_goal("Test goal", priority=GoalPriority.HIGH)
        goal = cognitive_core.get_bg_goal()
        
        assert goal is not None
        assert isinstance(goal, Goal)


class TestConsolidation:
    """Test memory consolidation."""
    
    def test_consolidate_processes_buffer(self, cognitive_core):
        """Test that consolidate processes the working_memory."""
        cognitive_core.perceive("Subject relates to Object")
        cognitive_core.perceive("Another fact about Subject")
        
        buffer_size_before = len(cognitive_core.working_memory)
        cognitive_core.consolidate()
        
        # Buffer should be cleared after consolidation
        assert len(cognitive_core.working_memory) < buffer_size_before


class TestIdentity:
    """Test identity/self-concept retrieval."""
    
    def test_get_identity_summary(self, cognitive_core):
        """Test identity summary generation."""
        graph = nx.DiGraph()
        graph.add_node("Self", type="CONCEPT")
        graph.add_node("AI", type="CONCEPT")
        graph.add_edge("Self", "AI", relation="IS_A", weight=1.0)
        
        summary = cognitive_core.get_identity_summary(graph)
        assert isinstance(summary, str)
        assert "nodes" in summary
