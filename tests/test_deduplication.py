#!/usr/bin/env python3
"""
Test edge deduplication with weight reinforcement.
Shows how repeated facts get reinforced instead of duplicated.
"""

import os
from google import genai
from ..neuron_x import NeuronX
from rich.console import Console

console = Console()

def test_deduplication():
    """Test that duplicate facts get reinforced with weights."""
    
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY not found", style="bold red")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Initialize NEURON-X with LLM client
    brain = NeuronX(llm_client=client)
    
    # Mention the same fact multiple times
    test_prompts = [
        "User: Mein Sohn heißt Noah.",
        "Me: Ich habe notiert: Dein Sohn heißt Noah.",
        "User: Noah ist mein Sohn.",
        "Me: Ja, Noah ist dein Sohn.",
    ]
    
    console.print("\n[bold yellow]🧪 DEDUPLICATION & REINFORCEMENT TEST[/bold yellow]\n")
    console.print("[bold cyan]Testing with repeated facts...[/bold cyan]\n")
    
    for prompt in test_prompts:
        console.print(f"[green]INPUT:[/green] {prompt}")
        brain.perceive(prompt, source="Test")
    
    console.print("\n" + "="*70)
    console.print("[bold magenta]⚙️  CONSOLIDATING (WATCH FOR REINFORCEMENT)...[/bold magenta]")
    console.print("="*70 + "\n")
    
    brain.consolidate()
    
    console.print("\n" + "="*70)
    console.print("[bold green]✅ COMPLETE[/bold green]")
    console.print("="*70 + "\n")
    
    console.print("[bold cyan]💡 Expected:[/bold cyan]")
    console.print("   First mention: [green]Added[/green] (User) --[has_son]--> (Noah)")
    console.print("   Repeated mentions: [yellow]Reinforced[/yellow] with increasing weight\n")
    
    console.print("[bold cyan]💡 Now run:[/bold cyan] python graph_inspector.py")
    console.print("[bold cyan]   Look for weight annotations like [weight: 2.0][/bold cyan]\n")

if __name__ == "__main__":
    test_deduplication()
