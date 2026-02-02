import os
import threading
import time
import logging
import datetime
from pathlib import Path
from typing import Optional, Callable, Any, List
from google import genai
from google.genai import types

from neuron_x import NeuronX
from neuron_x.prompts import get_system_instruction
from neuron_x.plugin_manager import PluginManager
from neuron_x.plugin_base import PluginContext
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore
from neuron_x.const import DEFAULT_PERSISTENCE_PATH, THOUGHT_LOOP_INTERVAL
from neuron_x.llm_tools import read_codebase_file

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
        self.model_id = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
        
        # Initialize Plugin System first
        plugin_dir = Path(__file__).parent / "plugins"
        self.plugin_manager = PluginManager(plugin_dir)
        
        # Initialize cognitive components directly
        persistence_path = Path(DEFAULT_PERSISTENCE_PATH)
        self.smith = GraphSmith(persistence_path)
        self.vault = VectorVault()
        self.core = CognitiveCore(
            self.smith, 
            self.vault, 
            self.client,
            plugin_tools_getter=self.plugin_manager.get_all_tools
        )
        
        # Create NeuronX facade for backward compatibility
        self.brain = NeuronX(llm_client=self.client)
        
        # Chat Session with automatic function calling
        self.chat_session = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                tools=self._get_all_tools_list(),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
            )
        )
        
        # Now initialize plugins with the bridge context
        self._initialize_plugins()

        # Background Loop State
        self.stop_event = threading.Event()
        self.bg_thread: Optional[threading.Thread] = None

    def _get_all_tools_list(self) -> List[Callable]:
        """Returns a list of all available tools for the LLM."""
        def recall_memories_tool(query: str):
            res = self.brain._get_relevant_memories(query)
            return "\n".join(res) if res else "No memories found."
            
        all_tools = [recall_memories_tool, read_codebase_file]
        plugin_tools = self.plugin_manager.get_all_tools()
        all_tools.extend(plugin_tools.values())
        return all_tools

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
        
        # Cleanup plugins (Discord etc.)
        logger = logging.getLogger("neuron-x")
        logger.info("Unloading all plugins...")
        self.plugin_manager.unload_all()
        
        self.brain.consolidate()

    def _run_loop(self):
        """Internal loop logic."""
        while not self.stop_event.is_set():
            try:
                if self.stop_event.wait(timeout=THOUGHT_LOOP_INTERVAL):
                    break
                
                thought = self.core.generate_proactive_thought()
                self.core.perceive(f"Self-Reflection: {thought}", source="Self_Reflection")
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
        
        system_instr = get_system_instruction(summary, "", subconscious_injection)

        # Update tools in case new plugins were loaded
        current_tools = self._get_all_tools_list()

        response = self.chat_session.send_message(
            message=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                tools=current_tools
            )
        )
        
        # Extract text safely from response parts
        ai_text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    ai_text += part.text
        
        if not ai_text:
            ai_text = "[System: Executed tools, but no final text response was generated.]"
        
        # 4. Perceive Interaction
        self.brain.perceive(f"User: {user_text}", source="User_Interaction")
        self.brain.perceive(f"Me: {ai_text}", source="Self_Reflection")
        
        if "Self" in self.brain.graph:
            self.brain.graph.nodes["Self"]["reinforcement_sum"] = self.brain.graph.nodes["Self"].get("reinforcement_sum", 0.0) + 5.0
            
        self.brain.save_graph()
        return ai_text
    
    # Plugin Management Methods
    
    def _initialize_plugins(self) -> None:
        """Discover and auto-load available plugins."""
        logger = logging.getLogger("neuron-x")
        
        def trigger_restart():
            import sys
            import os
            self.stop_background_loop()
            os.execv(sys.executable, [sys.executable] + sys.argv)

        context = PluginContext(
            interact_func=self.interact,
            restart_func=trigger_restart
        )
        try:
            available = self.plugin_manager.discover_plugins()
            logger.info(f"Discovered {len(available)} plugin(s)")
            
            for plugin_name in available:
                try:
                    self.plugin_manager.load_plugin(plugin_name, context=context)
                except Exception as e:
                    logger.warning(f"Failed to load plugin '{plugin_name}': {e}")
        except Exception as e:
            logger.error(f"Plugin initialization failed: {e}")
    
    def load_plugin(self, name: str) -> None:
        context = PluginContext(interact_func=self.interact)
        self.plugin_manager.load_plugin(name, context=context)
    
    def unload_plugin(self, name: str) -> None:
        self.plugin_manager.unload_plugin(name)
    
    def list_plugins(self) -> dict:
        return self.plugin_manager.list_plugins()
