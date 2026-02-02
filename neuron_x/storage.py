import json
import logging
import os
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
from neuron_x.const import DEFAULT_GRAPH_FILENAME, DEFAULT_GOALS_FILENAME
from models import Goal
logger = logging.getLogger('neuron-x')

class GraphSmith:
    """
    Repository class responsible for the persistence and retrieval of the 
    Knowledge Graph and Goal System.
    """

    def __init__(self, persistence_path: Path):
        self.path = persistence_path
        self.graph_file = self.path / DEFAULT_GRAPH_FILENAME
        self.goals_file = self.path / DEFAULT_GOALS_FILENAME
        self.last_sync_time: float = 0.0
        try:
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f'Critical error: Could not create storage directory {self.path}: {e}')
            raise

    def load_graph(self) -> nx.DiGraph:
        """Loads the graph from disk and updates the sync timestamp."""
        if self.graph_file.exists():
            try:
                graph = nx.read_gexf(str(self.graph_file))
                self.last_sync_time = self.graph_file.stat().st_mtime
                logger.info(f'[bold blue][GRAPH_SMITH][/bold blue] Knowledge Graph loaded | [bold cyan]{len(graph.nodes())}[/bold cyan] nodes.')
                return graph
            except Exception as e:
                logger.error(f'Failed to load graph: {e}')
                return nx.DiGraph()
        else:
            logger.info('[bold blue][GRAPH_SMITH][/bold blue] No existing graph found. Starting fresh.')
            return nx.DiGraph()

    def _atomic_write(self, graph: nx.DiGraph, file_path: Path) -> None:
        """Writes graph to a temp file, then atomically renames it."""
        temp_path = file_path.with_suffix('.tmp')
        try:
            nx.write_gexf(graph, str(temp_path))
            temp_path.replace(file_path)
        except Exception as e:
            logger.error(f'Failed to save atomic graph: {e}')
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def save_graph(self, graph: nx.DiGraph) -> None:
        """Saves current graph state safely."""
        try:
            self._atomic_write(graph, self.graph_file)
            self.last_sync_time = self.graph_file.stat().st_mtime
        except Exception as e:
            logger.error(f'Failed to save graph: {e}')

    def sync_if_needed(self, current_graph: nx.DiGraph) -> Optional[nx.DiGraph]:
        """
        Checks if the file on disk has been modified by another process.
        Returns the new graph if reloaded, else None.
        """
        try:
            if self.graph_file.exists():
                mtime = self.graph_file.stat().st_mtime
                if mtime > self.last_sync_time:
                    logger.info('[bold blue][GRAPH_SMITH][/bold blue] External update detected. Hot-reloading...')
                    return self.load_graph()
        except (OSError, IOError) as e:
            logger.error(f'Error checking graph sync status: {e}')
        return None

    def load_goals(self) -> List[Goal]:
        """Loads goals from disk."""
        if self.goals_file.exists():
            try:
                with open(self.goals_file, 'r') as f:
                    data = json.load(f)
                    goals = [Goal(**g) for g in data]
                logger.info(f'[bold blue][GRAPH_SMITH][/bold blue] Drives restored | [bold cyan]{len(goals)}[/bold cyan] active goals.')
                return goals
            except Exception as e:
                logger.error(f'Failed to load goals: {e}')
                return []
        return []

    def save_goals(self, goals: List[Goal]) -> None:
        """Persists current goals to disk safely using atomic write."""
        temp_path = self.goals_file.with_suffix('.tmp')
        try:
            goal_data = [g.model_dump() if hasattr(g, 'model_dump') else g.dict() for g in goals]
            with open(temp_path, 'w') as f:
                json.dump(goal_data, f, indent=2)
            temp_path.replace(self.goals_file)
        except Exception as e:
            logger.error(f'Failed to save goals safely: {e}')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def save_thought_stream(self, data: Dict[str, Any], filename: str) -> None:
        """Saves data to a JSON stream file safely using atomic write."""
        stream_path = self.path / filename
        temp_path = stream_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f)
            temp_path.replace(stream_path)
        except Exception as e:
            logger.error(f'Failed to save thought stream: {e}')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def load_thought_stream(self, filename: str) -> Optional[Dict[str, Any]]:
        """Loads data from a JSON stream file with error handling."""
        stream_path = self.path / filename
        if stream_path.exists():
            try:
                with open(stream_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f'Could not load thought stream {filename}: {e}')
                return None
        return None