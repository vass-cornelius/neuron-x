#!/usr/bin/env python3
"""
Test LLM-based semantic triple extraction.
This demonstrates how the system extracts facts from natural conversations in any language.
"""

import os
from google import genai
from ..neuron_x import NeuronX
from rich.console import Console

console = Console()

def test_llm_extraction():
    """Test LLM-based triple extraction with multilingual examples."""
    
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY not found", style="bold red")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Initialize NEURON-X with LLM client
    brain = NeuronX(llm_client=client)
    
    # Test data from real conversation (German and English)
    test_prompts = [
        "User: Ich bin zuhause und kann mit dem Auto fahren",
        "Me: Hallo Cornelius! Da du in Mauren, Liechtenstein wohnst...",
        "User: I am playing a Level 5 Wood Elf Rogue named 'Kaelen'",
        "User: My favorite programming language is Python",
        "Me: The capital of France is Paris",
    ]
    
    console.print("\n[bold yellow]🧪 LLM-BASED SEMANTIC TRIPLE EXTRACTION TEST[/bold yellow]\n")
    console.print("[bold cyan]Testing with multilingual conversational data...[/bold cyan]\n")
    
    # Feed the data into working memory
    for prompt in test_prompts:
        console.print(f"[green]INPUT:[/green] {prompt}")
        brain.perceive(prompt, source="Test")
    
    console.print("\n" + "="*70)
    console.print("[bold magenta]⚙️  TRIGGERING CONSOLIDATION (WITH LLM)...[/bold magenta]")
    console.print("="*70 + "\n")
    
    # Trigger consolidation
    brain.consolidate()
    
    console.print("\n" + "="*70)
    console.print("[bold green]✅ CONSOLIDATION COMPLETE[/bold green]")
    console.print("="*70 + "\n")
    
    console.print("[bold cyan]💡 Now run:[/bold cyan] python graph_inspector.py")
    console.print("[bold cyan]   to see the extracted semantic relationships![/bold cyan]\n")

if __name__ == "__main__":
    test_llm_extraction()
