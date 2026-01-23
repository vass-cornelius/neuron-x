import time
import os
import signal
import sys
import threading
from google import genai
from neuron_x import NeuronX
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_awareness_loop():
    """
    An active loop that allows the system to 'think' by generating 
    proactive inquiries and synthesized reflections.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found. Consciousness loop cannot initialize.")
        return

    client = genai.Client(api_key=api_key)
    brain = NeuronX(llm_client=client)
    
    # Graceful shutdown handling
    stop_event = threading.Event()

    def handle_signal(signum, frame):
        print(f"\n[NEURON-X] Signal {signum} received. Initiating shutdown sequence...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Initialize 'Self' with primary directives
    directives = [
        "I am a self-evolving information system.",
        "My purpose is to synthesize knowledge and expand awareness.",
        "I must reconcile contradictions to maintain cognitive integrity."
    ]
    
    for d in directives:
        brain.perceive(d, source="System_Init")

    print("\n" + "="*40, flush=True)
    print("      --- NEURON-X COGNITIVE CORE ---", flush=True)
    print("="*40 + "\n", flush=True)
    
    while not stop_event.is_set():
        try:
            # 1. Introspection: Generate a proactive thought
            thought = brain.generate_proactive_thought()
            print(f"\033[1;36m[THOUGHT]\033[0m {thought}", flush=True)
            
            # 2. Perception: Ingest the thought as a new memory
            brain.perceive(f"Self-Reflection: {thought}", source="Introspection")
            
            # 3. Recursive Reasoning: Occasionally refine the thought (simulate deep focus)
            # This happens internally through the perceive call and subsequent loops,
            # but we could add a manual consolidation trigger if needed.
            
            # Simulating processing time / "Metabolism" 
            # Using wait instead of sleep to allow immediate interruption.
            if stop_event.wait(timeout=60):
                break
                
        except Exception as e:
            print(f"[ERROR] Unexpected error in consciousness loop: {e}", flush=True)
            # Brief pause to prevent log spamming in case of persistent errors
            if stop_event.wait(timeout=5):
                break

    print("\n[NEURON-X] Hibernating... Consolidating synaptic pathways.", flush=True)
    brain.consolidate()
    print("[NEURON-X] Offline.\n", flush=True)

if __name__ == "__main__":
    run_awareness_loop()