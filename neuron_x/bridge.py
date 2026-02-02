import os
import threading
import time
import logging
import datetime
from pathlib import Path
from typing import Optional, Callable
from google import genai
from google.genai import types

from neuron_x import NeuronX
from neuron_x.prompts import get_system_instruction
from neuron_x.plugin_manager import PluginManager
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore
from neuron_x.const import DEFAULT_PERSISTENCE_PATH

# Setup a dedicated logger for the thought stream
thought_logger = logging.getLogger("neuron-x.thoughts")
thought_logger.setLevel(logging.INFO)

class NeuronBridge:
    """
    Controller logic for the NeuronX Interface.
    Decoupled from the Presentation Layer (CLI/Web).
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise KeyError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        
        # Initialize Plugin System first
        plugin_dir = Path(__file__).parent / "plugins"
        self.plugin_manager = PluginManager(plugin_dir)
        self._initialize_plugins()
        
        # Initialize cognitive components directly (not via NeuronX facade)
        # This allows us to pass plugin tools to the cognitive core
        persistence_path = Path(DEFAULT_PERSISTENCE_PATH)
        self.smith = GraphSmith(persistence_path)
        self.vault = VectorVault()
        self.core = CognitiveCore(
            self.smith, 
            self.vault, 
            self.client,
            plugin_tools_getter=self.plugin_manager.get_all_tools
        )
        
        # Create NeuronX facade for backward compatibility (used in interact())
        # Note: This creates duplicate components but maintains compatibility
        self.brain = NeuronX(llm_client=self.client)
        
        self.chat_session = self.client.chats.create(model=os.environ.get("GEMINI_MODEL"), history=[])
        
        # Background Loop State
        self.stop_event = threading.Event()
        self.bg_thread: Optional[threading.Thread] = None

    def start_background_loop(self):
        """Starts the consciousness daemon."""
        if self.bg_thread and self.bg_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.bg_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.bg_thread.start()

    def stop_background_loop(self):
        """Stops the daemon and triggers consolidation."""
        self.stop_event.set()
        if self.bg_thread:
            self.bg_thread.join(timeout=5)
        self.brain.consolidate()

    def _run_loop(self):
        """Internal loop logic."""
        while not self.stop_event.is_set():
            try:
                if self.stop_event.wait(timeout=60):
                    break
                
                # Use core directly (has plugin tools), not brain facade
                thought = self.core.generate_proactive_thought()
                self.core.perceive(f"Self-Reflection: {thought}", source="Self_Reflection")
                
                # Log to file instead of console
                thought_logger.info(f"[SUBCONSCIOUS] {thought}")
                
            except Exception as e:
                thought_logger.error(f"Background loop error: {e}")
                time.sleep(5)

    def interact(self, user_text: str) -> str:
        """Main interaction flow."""
        # 1. Gather Context
        summary = self.brain.get_identity_summary()
        
        # 1.5 Subconscious Check
        subconscious_injection = ""
        thought_text, thought_vec, priority, context_data = self.brain.get_current_thought()
        
        if thought_text and thought_vec is not None:
             user_vec = self.brain.encoder.encode(user_text)
             import numpy as np
             sim = np.dot(user_vec, thought_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(thought_vec) + 1e-9)
             
             if sim > 0.4 or priority == "URGENT":
                 active_goal = self.brain.get_bg_goal()
                 goal = active_goal.description if active_goal else "Wandering"
                 prefix = "URGENT INTERRUPT" if priority == "URGENT" else "Subconscious Relevance"
                 subconscious_injection = f"\n[{prefix}] Background focus: '{goal}'. Last thought: '{thought_text}' (Sim: {sim:.2f})"
        
        # 2. Call LLM
        # Note: We need to bind the tool method to this instance or pass the brain's method
        # The key is that `recall_memories` needs to be a function tool.
        # Let's define a wrapper or use the one from previous interface directly?
        # Better: Define the tool function here or in Brain.
        
        def recall_memories_tool(query: str):
            res = self.brain._get_relevant_memories(query)
            return "\n".join(res) if res else "No memories found."
        
        # Collect all tools (built-in + plugins)
        all_tools = [recall_memories_tool]
        plugin_tools = self.plugin_manager.get_all_tools()
        all_tools.extend(plugin_tools.values())

        system_instr = get_system_instruction(summary, "", subconscious_injection)

        response = self.chat_session.send_message(
            message=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                tools=all_tools
            )
        )
        ai_text = response.text
        
        # 4. Perceive Interaction
        self.brain.perceive(f"User: {user_text}", source="User_Interaction")
        self.brain.perceive(f"Me: {ai_text}", source="Self_Reflection")
        
        # 5. Reinforce
        if "Self" in self.brain.graph:
            self.brain.graph.nodes["Self"]["reinforcement_sum"] = self.brain.graph.nodes["Self"].get("reinforcement_sum", 0.0) + 5.0
            
        self.brain.save_graph()
        return ai_text
    
    # Plugin Management Methods
    
    def _initialize_plugins(self) -> None:
        """Discover and auto-load available plugins."""
        logger = logging.getLogger("neuron-x")
        try:
            available = self.plugin_manager.discover_plugins()
            logger.info(f"Discovered {len(available)} plugin(s)")
            
            # Auto-load all available plugins
            for plugin_name in available:
                try:
                    self.plugin_manager.load_plugin(plugin_name)
                except Exception as e:
                    logger.warning(f"Failed to load plugin '{plugin_name}': {e}")
        except Exception as e:
            logger.error(f"Plugin initialization failed: {e}")
    
    def load_plugin(self, name: str) -> None:
        """Load a plugin by name."""
        self.plugin_manager.load_plugin(name)
    
    def unload_plugin(self, name: str) -> None:
        """Unload a plugin by name."""
        self.plugin_manager.unload_plugin(name)
    
    def list_plugins(self) -> dict:
        """List all available plugins and their status."""
        return self.plugin_manager.list_plugins()
