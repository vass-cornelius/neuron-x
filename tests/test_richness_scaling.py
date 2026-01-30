#!/usr/bin/env python3
"""
Test Sensory Density Index (SDI) Scaling
Verifies that richer, more complex memories result in higher reinforcement weights.
"""

import os
import time
import shutil
import numpy as np
from neuron_x import NeuronX
from rich.console import Console
import logging

# Configure minimal logging to avoid noise
logging.basicConfig(level=logging.ERROR)
console = Console()

def test_sdi_scaling():
    console.print("\n[bold yellow]🧪 TESTING SENSORY DENSITY INDEX (SDI)[/bold yellow]\n")

    # 1. Setup minimal NeuronX
    # Use a temp directory
    test_dir = "./test_sdi_vault"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize without LLM (we will manually inject triples or generic extraction)
    # Since we don't have an LLM, consolidate will fallback to regex extraction.
    # We need to construct inputs that regex can parse to ensure triples are created.
    brain = NeuronX(persistence_path=test_dir)
    
    # 2. Define Inputs
    
    # Low Density Input (Simple, repetitive)
    # "The cat is a cat." -> TTR low, Length low
    # Regex: "X is a Y" -> (The cat, is_a, cat)
    low_density_text = "The cat is a cat category." 
    
    # High Density Input (Complex, unique words)
    # "The iridescent emerald dragon is a legendary beast." 
    # Regex: "X is a Y" -> (The iridescent emerald dragon, is_a, legendary beast)
    high_density_text = "The iridescent emerald dragon is a legendary beast of power."

    console.print(f"[cyan]Input 1 (Low Density):[/cyan] {low_density_text}")
    console.print(f"[cyan]Input 2 (High Density):[/cyan] {high_density_text}")

    # 3. Perceive & Check SDI
    brain.perceive(low_density_text)
    mem_low = brain.working_memory[-1]
    sdi_low = mem_low.get('sdi')
    console.print(f"[blue]SDI Low:[/blue] {sdi_low:.4f} (Expected ~0.3-0.5)")
    
    brain.perceive(high_density_text)
    mem_high = brain.working_memory[-1]
    sdi_high = mem_high.get('sdi')
    console.print(f"[blue]SDI High:[/blue] {sdi_high:.4f} (Expected > SDI Low)")

    if sdi_high <= sdi_low:
        console.print("[bold red]❌ FAILURE: High density input did not get higher SDI![/bold red]")
        return

    # Mock _extract_triples_batch to bypass regex limitations
    # We want to force extraction of specific triples linked to specific memory indices.
    # Memory 0: Low Density (User)
    # Memory 1: High Density (User)
    
    def mock_extract(memories):
        triples = []
        for i, m in enumerate(memories):
            # We assume the order matches our input sequence
            # But wait, working_memory accumulates. 
            # In this test, we have exactly 2 memories if we start fresh.
            if "cat" in m['text']:
                 triples.append({
                     "subject": "The cat",
                     "predicate": "is_a",
                     "object": "feline",
                     "category": "FACTUAL",
                     "index": i,
                     "source": "User_Interaction"
                 })
            elif "dragon" in m['text']:
                 triples.append({
                     "subject": "The dragon",
                     "predicate": "is_a",
                     "object": "legendary beast",
                     "category": "FACTUAL",
                     "index": i, # precise index needed for lookup
                     "source": "User_Interaction"
                 })
            elif "Artificial" in m['text']:
                 triples.append({
                     "subject": "Artificial Intelligence",
                     "predicate": "is_a",
                     "object": "Construct",
                     "category": "FACTUAL",
                     "index": i,
                     "source": "Self_Reflection"
                 })
        return triples

    brain._extract_triples_batch = mock_extract
    
    # 3.5 Test AI Damping
    # Feed an AI memory that is IDENTICAL or LONGER than the high density user memory
    # "The iridescent emerald dragon is a legendary beast of power." (10 words)
    # AI: "Artificial Intelligence is a complex construct of digital neurons and code." (11 words)
    ai_text = "Artificial Intelligence is a complex construct of digital neurons and code."
    brain.perceive(ai_text, source="Self_Reflection")
    
    mem_ai = brain.working_memory[-1]
    sdi_ai = mem_ai.get('sdi')
    console.print(f"[blue]SDI AI:[/blue] {sdi_ai:.4f} (Expected < SDI High due to damping)")
    
    if sdi_ai < sdi_high:
        console.print("[bold green]✅ SUCCESS: AI Damping is active![/bold green]")
    else:
        console.print("[bold red]❌ FAILURE: AI Damping failed or insufficient.[/bold red]")

    # 4. Consolidate & Check Weights
    console.print("\n[magenta]Consolidating...[/magenta]")
    
    # We need to manually set llm_client to something truthy so it calls _extract_triples_batch?
    # No, the code checks: if self.llm_client: try batch...
    # So we must fake an llm_client OR modify consolidate to use our mock regardless.
    # Let's fake llm_client.
    brain.llm_client = True 

    brain.consolidate()
    
    # Check Graph
    # Low density edge: ("The cat", "feline")
    # High density edge: ("The dragon", "legendary beast")
    
    edges = list(brain.graph.edges(data=True)) 
    
    w_low = 0.0
    w_high = 0.0
    
    for u, v, data in edges:
        # console.print(f"Edge: {u} -> {v} (Weight: {data['weight']})")
        if "cat" in u.lower() and "feline" in v.lower():
            w_low = data['weight']
        if "dragon" in u.lower() and "beast" in v.lower():
            w_high = data['weight']
            
    console.print(f"\n[green]Weight Low:[/green]  {w_low:.4f} (Expected ~1.3-1.4)")
    console.print(f"[green]Weight High:[/green] {w_high:.4f} (Expected > Low)")
    
    if w_high > w_low:
         console.print("[bold green]✅ SUCCESS: Richer memory created stronger synaptic connection![/bold green]")
    else:
         console.print(f"[bold red]❌ FAILURE: Weights are not scaled correctly based on SDI. (Low:{w_low}, High:{w_high})[/bold red]")

    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_sdi_scaling()
