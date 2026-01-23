#!/usr/bin/env python3
"""
Test script to demonstrate semantic triple extraction.
This simulates what happens during consolidation without needing to run the full interface.
"""

from neuron_x import NeuronX
from rich.console import Console

console = Console()

def test_triple_extraction():
    """Test the semantic triple extraction with D&D examples."""
    
    # Initialize NEURON-X
    brain = NeuronX()
    
    # D&D Test Data
    test_prompts = [
        "In our D&D campaign, I am playing a Level 5 Wood Elf Rogue named 'Kaelen'.",
        "Kaelen specializes in 'Infiltration' and carries a legendary dagger called 'The Whisper'.",
        "The setting of our game is a floating city called 'Aethervale' which is powered by ancient crystals."
    ]
    
    console.print("\n[bold yellow]🧪 SEMANTIC TRIPLE EXTRACTION TEST[/bold yellow]\n")
    console.print("[bold cyan]Testing with D&D Campaign Data...[/bold cyan]\n")
    
    # Feed the data into working memory
    for prompt in test_prompts:
        console.print(f"[green]INPUT:[/green] {prompt}")
        brain.perceive(prompt, source="Test")
    
    console.print("\n" + "="*70)
    console.print("[bold magenta]⚙️  TRIGGERING CONSOLIDATION (SLEEP CYCLE)...[/bold magenta]")
    console.print("="*70 + "\n")
    
    # Trigger consolidation
    brain.consolidate()
    
    console.print("\n" + "="*70)
    console.print("[bold green]✅ CONSOLIDATION COMPLETE[/bold green]")
    console.print("="*70 + "\n")
    
    console.print("[bold cyan]💡 Now run:[/bold cyan] python graph_inspector.py")
    console.print("[bold cyan]   to see the extracted semantic relationships![/bold cyan]\n")

if __name__ == "__main__":
    test_triple_extraction()
