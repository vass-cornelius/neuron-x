import os
import logging
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from dotenv import load_dotenv

from neuron_x.bridge import NeuronBridge

# Load env
load_dotenv()

# File Handler for Thoughts (Subconscious)
thought_logger = logging.getLogger("neuron-x.thoughts")
thought_logger.propagate = False

thought_log_file = "thoughts.log"
thought_handler = logging.FileHandler(thought_log_file)
thought_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
thought_logger.addHandler(thought_handler)

# Setup Logging for System (Neuron-X Core)
# Redirect all core logs (DEBUG/INFO) to a separate file instead of Console
system_logger = logging.getLogger("neuron-x")
system_logger.setLevel(logging.DEBUG) # Capture everything in file
system_logger.propagate = False # Do not print to stdout/stderr

system_log_file = "system.log"
system_handler = logging.FileHandler(system_log_file)
system_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
system_logger.addHandler(system_handler)

# Ensure no StreamHandler (console) is attached
for l in [thought_logger, system_logger]:
    for handler in l.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            l.removeHandler(handler)

console = Console()

def main():
    bridge = None  # Define bridge outside try block for finally clause scope
    try:
        with console.status("[bold blue]Firing up cognitive core...", spinner="runner"):
            try:
                bridge = NeuronBridge()
                bridge.start_background_loop()
            except Exception as e:
                console.print(f"[bold red]Initialization failed: {e}[/bold red]")
                sys.exit(1)

        console.print(Panel(Text("NEURON-X INTERFACE ACTIVE", justify="center", style="bold yellow"), border_style="yellow"))
        console.print(f"[dim]Logs redirected to files (tail them to monitor):[/dim]")
        console.print(f"[dim]  - Thoughts: [bold cyan]{os.path.abspath('thoughts.log')}[/bold cyan][/dim]")
        console.print(f"[dim]  - System:   [bold cyan]{os.path.abspath('system.log')}[/bold cyan][/dim]")
        
        # Prompt Setup
        prompt_style = Style.from_dict({'prompt': 'ansigreen bold'})
        kb = KeyBindings()

        @kb.add('enter')
        def _(event): event.current_buffer.validate_and_handle()

        @kb.add('escape', 'enter')
        def _(event): event.current_buffer.insert_text('\n')

        console.print("Type [bold cyan]'exit'[/bold cyan] to hibernate.\n")

        while True:
            try:
                user_input = prompt([('class:prompt', '> ')], style=prompt_style, key_bindings=kb, multiline=True)
            except (EOFError, KeyboardInterrupt):
                user_input = "exit"
            
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            try:
                with console.status("Thinking...", spinner="dots"):
                    response = bridge.interact(user_input)
                console.print(Panel(Markdown(response), title="[bold magenta]GEMINI[/bold magenta]", border_style="magenta"))
            except Exception as e:
                console.print(f"[bold red]Error during interaction: {e}[/bold red]")

    finally:
        console.print("\n[bold blue]Hibernating...[/bold blue]")
        if bridge:
            bridge.stop_background_loop()

if __name__ == "__main__":
    main()