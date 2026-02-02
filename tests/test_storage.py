"""
Tests for neuron_x.storage module (GraphSmith).
"""
import json
import pytest
import networkx as nx
from pathlib import Path
from neuron_x.storage import GraphSmith
from models import Goal
from neuron_x.const import DEFAULT_GRAPH_FILENAME, DEFAULT_GOALS_FILENAME 

@pytest.fixture
def temp_storage_path(tmp_path):
    """Create a temporary storage directory."""
    return tmp_path / "test_storage"

@pytest.fixture
def graph_smith(temp_storage_path):
    """Create a GraphSmith instance for testing."""
    return GraphSmith(temp_storage_path)

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = nx.DiGraph()
    graph.add_node("Alice", type="CONCEPT")
    graph.add_node("Bob", type="CONCEPT")
    graph.add_edge("Alice", "Bob", relation="KNOWS", weight=1.0)
    return graph

@pytest.fixture
def sample_goals():
    """Create sample goals for testing."""
    return [
        Goal(id="1", description="Test goal 1", priority="HIGH", status="PENDING"),
        Goal(id="2", description="Test goal 2", priority="MEDIUM", status="IN_PROGRESS")
    ]


class TestGraphSmithInit:
    """Test GraphSmith initialization."""
    
    def test_init_creates_directory(self, temp_storage_path):
        """Test that initialization creates the storage directory."""
        smith = GraphSmith(temp_storage_path)
        assert temp_storage_path.exists()
        assert smith.path == temp_storage_path
    
    def test_init_sets_file_paths(self, graph_smith, temp_storage_path):
        """Test that file paths are correctly set."""
        assert graph_smith.graph_file == temp_storage_path / DEFAULT_GRAPH_FILENAME
        assert graph_smith.goals_file == temp_storage_path / DEFAULT_GOALS_FILENAME


class TestGraphOperations:
    """Test graph save/load operations."""
    
    def test_save_and_load_graph(self, graph_smith, sample_graph):
        """Test saving and loading a graph."""
        graph_smith.save_graph(sample_graph)
        loaded_graph = graph_smith.load_graph()
        
        assert len(loaded_graph.nodes()) == 2
        assert len(loaded_graph.edges()) == 1
        assert "Alice" in loaded_graph.nodes()
        assert "Bob" in loaded_graph.nodes()
    
    def test_load_nonexistent_graph(self, graph_smith):
        """Test loading when no graph file exists."""
        graph = graph_smith.load_graph()
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 0
    
    def test_sync_if_needed_no_changes(self, graph_smith, sample_graph):
        """Test sync_if_needed returns None when no external changes."""
        graph_smith.save_graph(sample_graph)
        graph_smith.load_graph()  # Sets last_sync_time
        
        result = graph_smith.sync_if_needed(sample_graph)
        assert result is None


class TestGoalsOperations:
    """Test goals save/load operations."""
    
    def test_save_and_load_goals(self, graph_smith, sample_goals):
        """Test saving and loading goals."""
        graph_smith.save_goals(sample_goals)
        loaded_goals = graph_smith.load_goals()
        
        assert len(loaded_goals) == 2
        assert loaded_goals[0].description == "Test goal 1"
        assert loaded_goals[1].priority == "MEDIUM"
    
    def test_load_nonexistent_goals(self, graph_smith):
        """Test loading when no goals file exists."""
        goals = graph_smith.load_goals()
        assert isinstance(goals, list)
        assert len(goals) == 0


class TestThoughtStream:
    """Test thought stream save/load operations."""
    
    def test_save_and_load_thought_stream(self, graph_smith):
        """Test saving and loading thought stream data."""
        data = {"thought": "test thought", "priority": "HIGH"}
        graph_smith.save_thought_stream(data, "test_stream.json")
        
        loaded_data = graph_smith.load_thought_stream("test_stream.json")
        assert loaded_data == data
    
    def test_load_nonexistent_stream(self, graph_smith):
        """Test loading non-existent thought stream."""
        result = graph_smith.load_thought_stream("nonexistent.json")
        assert result is None
