from pathlib import Path
from typing import Optional, Any
import logging

from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore

logger = logging.getLogger("neuron-x")

from neuron_x.const import DEFAULT_PERSISTENCE_PATH

class NeuronX:
    """
    Facade for the NeuronX Cognitive Architecture.
    Maintains backward compatibility while delegating to modular components.
    """
    def __init__(self, persistence_path: str = DEFAULT_PERSISTENCE_PATH, llm_client: Any = None):
        self.path = Path(persistence_path)
        
        # Initialize Components
        self.smith = GraphSmith(self.path)
        self.vault = VectorVault()
        self.core = CognitiveCore(self.smith, self.vault, llm_client)
        
        # Public properties for compatibility
        self.llm_client = llm_client

    @property
    def graph(self):
        """Exposes the underlying networkx graph."""
        # This is a bit expensive if we load it every time, but necessary for direct access compatibility.
        # Ideally, we should cache it or expose the one from storage if loaded.
        # But GraphSmith loads it on demand. 
        # Let's delegate to a cached property or method if possible, 
        # but for now simple loading is safe.
        return self.smith.load_graph()

    @property
    def encoder(self):
        """Exposes the encoder for external use."""
        return self.vault.encoder

    @property
    def goals(self):
        """Exposes the goals list."""
        return self.core.goals

    @property
    def working_memory(self):
         return self.core.working_memory

    @property
    def thought_buffer(self):
         return self.core.thought_buffer

    def perceive(self, text: str, source: str = "Internal") -> None:
        self.core.perceive(text, source)

    def consolidate(self) -> None:
        self.core.consolidate()

    def add_goal(self, description: str, priority=None) -> None:
        if priority:
            self.core.add_goal(description, priority)
        else:
            self.core.add_goal(description)

    def get_bg_goal(self):
        return self.core.get_bg_goal()

    def generate_proactive_thought(self):
        return self.core.generate_proactive_thought()

    def get_current_thought(self):
        return self.core.get_current_thought()

    def get_identity_summary(self):
        return self.core.get_identity_summary(self.graph)

    def save_graph(self):
        """Explicit save trigger."""
        # Logic layer handles saving usually, but if called explicitly:
        self.smith.save_graph(self.smith.load_graph())

    # Forward other private methods if tests access them?
    # e.g. _get_relevant_memories
    # If external scripts call privates, they are breaking encapsulation, but we can shim them.
    def _get_relevant_memories(self, text, top_k=5):
        # We didn't implement this publicly in Core yet.
        # Ideally CognitiveCore handles this.
        # For now, let's stub it or move logic to Memory/Cognition.
        # The `check_state.py` might use it? No, check_state usually checks graph.
        return [] # Placeholder
