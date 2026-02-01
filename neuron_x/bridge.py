import os
import threading
import time
import logging
import datetime
from typing import Optional, Callable
from google import genai
from google.genai import types

from neuron_x import NeuronX
from neuron_x.prompts import get_system_instruction

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
                
                thought = self.brain.generate_proactive_thought()
                self.brain.perceive(f"Self-Reflection: {thought}", source="Self_Reflection")
                
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

        # 2. Prepare Prompt
        system_instr = get_system_instruction(summary, "", subconscious_injection)
        
        # 3. Call LLM
        # Note: We need to bind the tool method to this instance or pass the brain's method
        # The key is that `recall_memories` needs to be a function tool.
        # Let's define a wrapper or use the one from previous interface directly?
        # Better: Define the tool function here or in Brain.
        
        def recall_memories_tool(query: str):
            """Search long-term memory."""
            res = self.brain._get_relevant_memories(query)
            return "\n".join(res) if res else "No memories found."

        response = self.chat_session.send_message(
            message=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                tools=[recall_memories_tool]
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
