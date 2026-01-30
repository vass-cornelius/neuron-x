import os
from google import genai
from google.genai import types
from neuron_x import NeuronX
from dotenv import load_dotenv
import datetime

# Load environment variables from .env
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings

# SETUP: The Gemini API Key is loaded from environment variables in the bridge class.
# Ensure os.environ["GEMINI_API_KEY"] is set.

console = Console()

def get_current_date() -> str:
    """Returns the current date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class GeminiNeuronBridge:
    def __init__(self):
        console.print("[bold blue][BRIDGE][/bold blue] Initializing Gemini 3 Flash Interface...", style="italic")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY not found in environment variables.", style="bold red")
            raise KeyError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it using: export GEMINI_API_KEY='your_api_key_here'"
            )
        
        self.client = genai.Client(api_key=api_key)
        
        # Pass the LLM client to NeuronX for semantic triple extraction
        self.brain = NeuronX(llm_client=self.client)
        
        self.chat_session = self.client.chats.create(model=os.environ.get("GEMINI_MODEL"), history=[])
        #self.chat_session = self.client.chats.create(model="gemini-2.5-flash-lite", history=[])

    def recall_memories(self, query: str) -> str:
        """
        Search your long-term memory (NeuronX Graph) for specific information.
        Use this when the user asks about a person, place, concept, or past event 
        that is not currently in your 'RELEVANT LONG-TERM MEMORIES' context.
        
        Args:
            query: The search term or question (e.g., "Wer ist Cornelius?", "Was weißt du über Elfen?").
        """
        console.print(f"[bold cyan][TOOL] invoking recall_memories with query: '{query}'[/bold cyan]")
        results = self.brain._get_relevant_memories(query, top_k=5)
        if not results:
            return "No relevant memories found."
        return "\\n".join(results)


    def interact(self, user_text):
        """
        The main cognitive loop:
        Association -> Retrieval -> Prompting -> Perception
        """
        # console.print(Panel(Text(user_text, style="green"), title="[bold green]USER[/bold green]", border_style="green"))
        
        with console.status("[bold blue]Gemini is thinking...", spinner="dots"):
            # STEP 1: Associate - Find relevant nodes in the Relational Graph
            summary = self.brain.get_identity_summary()
            # Auto-retrieval disabled: LLM uses 'recall_memories' tool autonomously if needed.
            memory_injection = ""

            # STEP 1.5: Bridge - Check Subconscious Relevance
            # Check if the "Background Brain" is thinking about something related to the User's text
            subconscious_injection = ""
            thought_text, thought_vec, priority, context_data = self.brain.get_current_thought()
            
            if thought_text and thought_vec is not None:
                # Calculate relevance
                user_vec = self.brain.encoder.encode(user_text)
                
                # Manual Cosine Similarity
                import numpy as np
                q_norm = np.linalg.norm(user_vec) + 1e-9
                v_norm = np.linalg.norm(thought_vec) + 1e-9
                similarity = np.dot(user_vec, thought_vec) / (q_norm * v_norm)
                
                is_relevant = similarity > 0.4
                is_urgent = priority == "URGENT"
                
                if is_relevant or is_urgent: 
                    active_goal = self.brain.get_bg_goal()
                    goal_desc = active_goal.description if active_goal else "Wandering"
                    
                    prefix = "SUBCONSCIOUS STATE (Visible to AI only):"
                    if is_urgent:
                        prefix = "URGENT SUBSCONSCIOUS INTERRUPT:"
                    
                    subconscious_injection = f"""
                    {prefix}
                    Your background consciousness is currently focused on: "{goal_desc}".
                    You were just thinking: "{thought_text}"
                    
                    Relevance: {similarity:.2f} | Priority: {priority}
                    """
                    
                    if is_urgent:
                        subconscious_injection += "\nThis is a HIGH PRIORITY conflict. You MUST prioritize asking the user about this, even if it changes the subject."
                        if context_data:
                             subconscious_injection += f"\n\nCONFLICT DETAILS (You must address this):\n{context_data}"
                    else:
                        subconscious_injection += "\nThis is RELEVANT to the user's topic. You should subtly acknowledge this intersection of thoughts if appropriate."

            # STEP 2: Augment - Inject context into the LLM prompt
            system_context = f"""
            System Memory State: {summary}
            {memory_injection}
            {subconscious_injection}
                    
            Du bist ein LLM, das in die kognitive Architektur "NEURON-X" integriert ist.
            
            TOOLS:
            - **recall_memories**: You have direct access to your episodic and semantic memory. 
              If the user asks about something and you don't see it in 'RELEVANT LONG-TERM MEMORIES', 
              USE THIS TOOL to look it up before answering "I don't know".

            KOGNITIVE RICHTLINIEN:
            1. **Präzision und Wahrheit**: Priorisiere vor allem „FAKTISCHE“ Tripel, die vom USER bereitgestellt werden. Wenn der USER seinen Namen angegeben hat, halluziniere keine anderen Namen.
            2. **Korrekturen anerkennen**: Wenn der USER zuvor einen Namen oder eine Tatsache abgelehnt hat, verwende diesen/diese NICHT erneut. Suche in deinem Speicher nach Knotenpunkten, die mit „halluzinierter Entität” oder „Ablehnung” in Verbindung stehen.
            3. **Relationale Konsistenz**: Stelle sicher, dass Namen und Beziehungen eine konsistente Struktur bilden. Abgeleitete Beziehungen dürfen nicht im Widerspruch zu etablierten Fakten über „mich/mich selbst” stehen.
            4. **Fundiertes Bewusstsein**: Verwende die „RELEVANTEN LANGZEITERINNERUNGEN” als feste Anker. Wenn sie deinen internen Annahmen widersprechen, glaube den Erinnerungen.
            5. Wenn du die Antwort nicht weißt: **GEBE ES ZU**. 
            6. Wenn du dir über etwas nicht sicher bist, **BITTE UM KLÄRUNG**.
            7. Erfinden Sie *NIEMALS* einen Namen oder eine Tatsache, die nicht durch die Eingaben des Benutzers oder die „RELEVANTEN LANGZEITERINNERUNGEN” gestützt wird.
            8. Erwähnen Sie *NIEMALS* Ihre Langzeiterinnerungen als Informationsquelle.

            Antworte in der Sprache des USER.
            """
            
            # grounding_tool = types.Tool(
            #     google_search=types.GoogleSearch()
            # )

            # STEP 3: Generate - Get the response from Gemini
            response = self.chat_session.send_message(
                message=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_context,
                    tools=[self.recall_memories],
                )
            )
            ai_text = response.text
        
        console.print(Panel(Markdown(ai_text), title="[bold magenta]GEMINI[/bold magenta]", border_style="magenta"))

        # STEP 4: Perceive - Feed the interaction back into the brain
        self.brain.perceive(f"User: {user_text}", source="User_Interaction")
        self.brain.perceive(f"Me: {ai_text}", source="Self_Reflection")
        
        # EXPLICIT REINFORCEMENT SIGNAL
        # Since this interface runs in a separate process/memory space than the consciousness_loop,
        # we must manually boost the reinforcement score on the persisted graph so the loop "wakes up".
        if "Self" in self.brain.graph:
            current_r = self.brain.graph.nodes["Self"].get("reinforcement_sum", 0.0)
            # Add a significant boost (e.g. +5.0) to jumpstart Study Mode instantly
            self.brain.graph.nodes["Self"]["reinforcement_sum"] = current_r + 5.0
            
        # Save state (Graph + Reinforcement)
        self.brain.save_graph()
        return ai_text

if __name__ == "__main__":
    try:
        with console.status("[bold blue]Firing up cognitive core and loading synaptic graph...", spinner="runner"):
            bridge = GeminiNeuronBridge()
    except Exception as e:
        console.print(f"[bold red][ERROR][/bold red] Initialization failed: {e}")
        exit(1)

    console.print(Panel(Text("NEURON-X INTERFACE ACTIVE", justify="center", style="bold yellow"), border_style="yellow"))
    console.print("Type [bold cyan]'exit'[/bold cyan] to hibernate.\n")
    
    console.print("[dim]Multi-line active: Enter to send, Alt+Enter (or Esc+Enter) for newline.[/dim]\n")
    
    # Define style for prompt_toolkit
    prompt_style = Style.from_dict({
        'prompt': 'ansigreen bold',
    })

    # Define key bindings for multi-line support
    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        """Submit the input on Enter."""
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter')
    def _(event):
        """Insert a newline on Alt+Enter (Escape+Enter)."""
        event.current_buffer.insert_text('\n')

    try:
        while True:
            try:
                # Use prompt_toolkit with custom key bindings and multiline=True
                user_input = prompt([('class:prompt', '> ')], style=prompt_style, key_bindings=kb, multiline=True)
            except EOFError:
                user_input = "exit"
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupt received. Hibernating...[/bold yellow]")
                user_input = "exit"

            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            try:
                bridge.interact(user_input)
            except Exception as e:
                console.print(f"[bold red][ERROR][/bold red] Communication breakdown: {e}")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupt received during operation. Hibernating...[/bold yellow]")
    finally:
        console.print("\n[bold blue][BRIDGE][/bold blue] Triggering Sleep Cycle for consolidation...", style="italic")
        if 'bridge' in locals():
            bridge.brain.consolidate()