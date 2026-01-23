#!/usr/bin/env python3
"""
Test specifically designed to check for "tunnel vision" during triple extraction.
It provides a complex scenario with multiple entities, possession, and environmental context.
"""

import os
from google import genai
from neuron_x import NeuronX
from rich.console import Console
from rich.table import Table

console = Console()

def test_tunnel_vision():
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY not found")
        return
    
    client = genai.Client(api_key=api_key)
    brain = NeuronX(llm_client=client)
    
    # Complex scenario text
    scenario = """
    In the misty town of Shadowfell, the merchant Elara sells enchanted lanterns. 
    Kaelen, a wood elf rogue, is hiding in the shadows of the 'Broken Keg' tavern, watching a hooded stranger.
    The stranger carries a silver locket that glows in the dark. 
    It is midnight, and the rain is starting to fall on the cobblestone streets.
    Elara is worried about the 최근 bandits sightings in the nearby Whispering Woods.
    """
    
    console.print("\n[bold yellow]🧪 TUNNEL VISION BREADTH TEST[/bold yellow]\n")
    console.print(f"[bold cyan]Scenario:[/bold cyan]\n{scenario.strip()}\n")
    
    with console.status("[bold blue]Extracting triples with broad scanning..."):
        triples = brain._extract_triples_with_llm(scenario, source="Test_Scenario")
    
    if not triples:
        console.print("[bold red]FAILED:[/bold red] No triples extracted.")
        return

    table = Table(title="Extracted Semantic Triples")
    table.add_column("Subject", style="cyan")
    table.add_column("Predicate", style="magenta")
    table.add_column("Object", style="green")
    table.add_column("Category", style="yellow")
    
    for t in triples:
        table.add_row(t.get('subject', ''), t.get('predicate', ''), t.get('object', ''), t.get('category', ''))
    
    console.print(table)
    
    # Check for breadth
    all_text = " ".join([f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]).lower()
    expected_entities = ['shadowfell', 'elara', 'kaelen', 'stranger', 'locket', 'tavern', 'rain', 'midnight', 'woods']
    
    found_expected = [ent for ent in expected_entities if ent in all_text]
    
    console.print(f"\n[bold]Breadth Score:[/bold] {len(found_expected)}/{len(expected_entities)} expected entities/concepts detected.")
    
    if len(found_expected) >= 6:
         console.print("[bold green]✅ BREADTH TEST PASSED[/bold green] - Extracted information beyond just the primary subject.")
    else:
         console.print("[bold red]❌ BREADTH TEST FAILED[/bold red] - Extraction seems too narrow.")

if __name__ == "__main__":
    test_tunnel_vision()
