import os
import threading
import time
import json
import logging
import datetime
import sys
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict
from google import genai
from google.genai import types

from neuron_x import NeuronX
from neuron_x.prompts import get_system_instruction
from neuron_x.plugin_manager import PluginManager
from neuron_x.plugin_base import PluginContext
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from neuron_x.cognition import CognitiveCore
from neuron_x.const import (
    DEFAULT_PERSISTENCE_PATH, 
    THOUGHT_LOOP_INTERVAL,
    DEFAULT_RESUME_FILENAME
)
from neuron_x.llm_tools import read_codebase_file

# Setup a dedicated logger
logger = logging.getLogger("neuron-x")
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
        
        # Track active conversation source for tools like restart
        self.active_source: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
        # Initialize Plugin System
        plugin_dir = Path(__file__).parent / "plugins"
        self.plugin_manager = PluginManager(plugin_dir)
        
        # Initialize cognitive components
        persistence_path = Path(DEFAULT_PERSISTENCE_PATH)
        self.smith = GraphSmith(persistence_path)
        self.vault = VectorVault()
        self.core = CognitiveCore(
            self.smith, 
            self.vault, 
            self.client,
            plugin_tools_getter=self.plugin_manager.get_all_tools
        )
        
        # Create NeuronX facade (Injecting existing core/smith/vault to prevent double-loading)
        self.brain = NeuronX(
            llm_client=self.client, 
            core=self.core, 
            smith=self.smith, 
            vault=self.vault
        )
        
        # Chat Session
        self.chat_session = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                tools=self._get_all_tools_list(),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
            )
        )
        
        self._initialize_plugins()
        self._check_for_resume()

        self.stop_event = threading.Event()
        self.bg_thread: Optional[threading.Thread] = None

    def _get_all_tools_list(self) -> List[Callable]:
        """Returns a list of all available tools for the LLM."""
        def recall_memories_tool(query: str):
            """Retrieves semantically relevant memories and relational context."""
            res = self.brain._get_relevant_memories(query)
            return "\n".join(res) if res else "No memories found for this query."
            
        def restart_neuron_x(reason: Optional[str] = None, todos: Optional[str] = None) -> str:
            """
            Triggers a graceful restart of the entire NeuronX system.
            Use this when plugins have been updated or system state needs refresh.
            """
            logger.info(f"System restart requested via Tool. Reason: {reason}")
            
            with self.lock:
                resume_state = {
                    "plugin": self.active_source.get("plugin"),
                    "channel_id": self.active_source.get("channel_id"),
                    "recipient": self.active_source.get("recipient"),
                    "reason": reason,
                    "todos": todos
                }
            
            # Start restart in a separate thread to allow tool response to return
            threading.Thread(target=self._trigger_restart, args=(resume_state,), daemon=True).start()
            
            msg = "Restart initiated."
            if todos:
                msg += f" I will resume with these tasks: {todos}"
            return msg + " Back online in a few seconds."

        # Core tools
        all_tools = [recall_memories_tool, restart_neuron_x, read_codebase_file]
        
        # Plugin tools (collision detection handled in PluginManager)
        plugin_tools = self.plugin_manager.get_all_tools()
        for name, tool in plugin_tools.items():
            if name not in ["restart_neuron_x"]: # Skip plugin-specific restart tools to avoid collision
                all_tools.append(tool)
                
        return all_tools
    
    def _get_tool_by_name(self, name: str) -> Optional[Callable]:
        """Internal helper to find a tool by its registered name."""
        # Check built-in tools
        if name == "recall_memories_tool":
            return lambda query: "\n".join(self.brain._get_relevant_memories(query))
        if name == "read_codebase_file":
            return read_codebase_file
            
        # Check plugin tools
        plugin_tools = self.plugin_manager.get_all_tools()
        return plugin_tools.get(name)

    def _trigger_restart(self, resume_state: Optional[dict] = None):
        """Internal method to perform the actual restart."""
        time.sleep(1) # Small delay to allow messages to flush
        
        if resume_state:
            try:
                resume_file = Path(DEFAULT_PERSISTENCE_PATH) / DEFAULT_RESUME_FILENAME
                with open(resume_file, "w") as f:
                    json.dump(resume_state, f, indent=2)
                logger.info(f"Saved resume state: {resume_state}")
            except Exception as e:
                logger.error(f"Failed to save resume state: {e}")

        self.stop_background_loop()
        logger.info("RESTARTING PROCESS...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def start_background_loop(self):
        if self.bg_thread and self.bg_thread.is_alive():
            return
        self.stop_event.clear()
        self.bg_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.bg_thread.start()

    def stop_background_loop(self):
        self.stop_event.set()
        if self.bg_thread:
            self.bg_thread.join(timeout=5)
        self.plugin_manager.unload_all()
        self.brain.consolidate()

    def _run_loop(self):
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

    def interact(self, user_text: str, context_metadata: Optional[Dict] = None) -> str:
        """Main interaction flow."""
        # Update active source for tools
        if context_metadata:
            with self.lock:
                self.active_source = context_metadata
        
        summary = self.brain.get_identity_summary()
        
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
        current_tools = self._get_all_tools_list()

        try:
            response = self.chat_session.send_message(
                message=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instr,
                    tools=current_tools
                )
            )
            
            ai_text = response.text if response.candidates else ""
            
            if not ai_text:
                # Fallback: if tool calls were made but no text summary generated, force one
                logger.info("No text response in first turn, requesting summary...")
                response = self.chat_session.send_message(
                    "Verstanden. Bitte gib mir eine kurze Rückmeldung zu den ausgeführten Aktionen oder eine abschließende Antwort."
                )
                ai_text = response.text if response.candidates else "[System: Aktionen ausgeführt.]"
            
            self.brain.perceive(f"User: {user_text}", source="User_Interaction")
            self.brain.perceive(f"Me: {ai_text}", source="Self_Reflection")
            
            if "Self" in self.brain.graph:
                # Move this to CognitiveCore.consolidate logic eventually
                pass
                
            self.brain.save_graph()
            return ai_text
            
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            return f"Entschuldigung, es gab einen Fehler bei der Verarbeitung: {str(e)}"
    
    def _check_for_resume(self) -> None:
        resume_file = Path(DEFAULT_PERSISTENCE_PATH) / DEFAULT_RESUME_FILENAME
        if not resume_file.exists():
            return

        try:
            with open(resume_file, "r") as f:
                state = json.load(f)
            
            logger.info(f"Processing resume state: {state}")
            
            plugin_name = state.get("plugin")
            channel_id = state.get("channel_id")
            recipient = state.get("recipient")
            todos = state.get("todos")
            reason = state.get("reason")
            
            follow_up = "Ich bin wieder online."
            if reason:
                follow_up += f" (Neustart-Grund: {reason})"
            if todos:
                follow_up += f"\n\nMeine nächsten To-Dos:\n{todos}"
            
            tool_name = None
            target = None
            
            if plugin_name == "discord_connector" and channel_id:
                tool_name = "send_discord_message"
                target = channel_id
            elif plugin_name == "signal_connector" and (recipient or recipient == "0"):
                tool_name = "send_signal_message"
                target = recipient if recipient else "0"
            
            if tool_name:
                send_func = self._get_tool_by_name(tool_name)
                if send_func:
                    send_func(target, follow_up)
            
            resume_file.unlink()
        except Exception as e:
            logger.error(f"Failed to process resume state: {e}")

    def _initialize_plugins(self) -> None:
        """Discover and auto-load available plugins."""
        context = PluginContext(
            interact_func=self.interact,
            restart_func=self._trigger_restart,
            transcribe_func=self.transcribe_audio,
            get_tool_func=self._get_tool_by_name
        )
        try:
            available = self.plugin_manager.discover_plugins()
            for plugin_name in available:
                try:
                    self.plugin_manager.load_plugin(plugin_name, context=context)
                except Exception as e:
                    logger.warning(f"Failed to load plugin '{plugin_name}': {e}")
        except Exception as e:
            logger.error(f"Plugin initialization failed: {e}")

    def transcribe_audio(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return "[Error: File not found]"
            audio_file = self.client.files.upload(file=file_path)
            prompt = "Transcribe this audio message exactly. Output ONLY the text."
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[audio_file, prompt]
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"[Error: {e}]"
