from pathlib import Path
from typing import Optional, Any
import logging
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore
logger = logging.getLogger('neuron-x')
from neuron_x.const import DEFAULT_PERSISTENCE_PATH

class NeuronX:
    """
    Facade for the NeuronX Cognitive Architecture.
    Maintains backward compatibility while delegating to modular components.
    """

    def __init__(self, persistence_path: str=DEFAULT_PERSISTENCE_PATH, llm_client: Any=None, core: Optional[CognitiveCore]=None, smith: Optional[GraphSmith]=None, vault: Optional[VectorVault]=None):
        """
        Facade for the NeuronX Cognitive Architecture.
        Allows injection of existing components to prevent redundant initialization.
        """
        self.path = Path(persistence_path)
        self.llm_client = llm_client
        self.smith = smith if smith else GraphSmith(self.path)
        self.vault = vault if vault else VectorVault()
        if core:
            self.core = core
        else:
            self.core = CognitiveCore(self.smith, self.vault, llm_client, plugin_tools_getter=None)

    @property
    def graph(self):
        """Exposes the underlying networkx graph."""
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

    def perceive(self, text: str, source: str='Internal') -> None:
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
        self.smith.save_graph(self.smith.load_graph())

    def _get_relevant_memories(self, text, top_k=5):
        """Delegates memory retrieval to the CognitiveCore."""
        return self.core._get_relevant_memories(text, top_k=top_k)