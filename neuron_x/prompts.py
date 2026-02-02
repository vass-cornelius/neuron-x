from pathlib import Path
import re
import logging

logger = logging.getLogger("neuron-x")

def _discover_skills() -> str:
    """
    Discover available skills and return formatted description.
    
    For critical skills (like memory-retrieval), inject full content.
    For others, just list them.
    
    Returns:
        Formatted string listing available skills or empty string if none found
    """
    skills_dir = Path(__file__).parent.parent / ".agent" / "skills"
    
    if not skills_dir.exists():
        return ""
    
    # Critical skills that should have full content injected
    CRITICAL_SKILLS = {"memory-retrieval"}
    
    skills = []
    full_skills = []
    skill_count = 0
    
    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue
        
        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            continue
        
        # Read skill metadata from frontmatter
        try:
            content = skill_file.read_text(encoding='utf-8')
            
            # Extract YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    skill_body = parts[2].strip()
                    
                    # Parse name and description
                    name_match = re.search(r'name:\s*(.+)', frontmatter)
                    desc_match = re.search(r'description:\s*(.+)', frontmatter)
                    
                    if name_match and desc_match:
                        name = name_match.group(1).strip()
                        description = desc_match.group(1).strip()
                        skill_path_str = str(skill_file.relative_to(Path.cwd()))
                        
                        # Check if this is a critical skill
                        if skill_path.name in CRITICAL_SKILLS:
                            # Inject full content (truncate if too long)
                            if len(skill_body) > 2000:
                                skill_body = skill_body[:2000] + "\n\n[... truncated ...]"
                            
                            full_skills.append(f"\n## CRITICAL SKILL: {name}\n\n{skill_body}")
                            logger.info(f"[SKILLS] Injected full content for critical skill: {name}")
                        else:
                            # Just list it
                            skills.append(f"- **{name}**: {description}\n  → Read full skill: `{skill_path_str}`")
                        
                        skill_count += 1
        
        except Exception as e:
            logger.warning(f"Failed to parse skill at {skill_file}: {e}")
            continue
    
    if skill_count == 0:
        return ""
    
    result = f"\n{'='*60}\nAVAILABLE SKILLS ({skill_count} found)\n{'='*60}\n"
    
    # Add full critical skills first
    if full_skills:
        result += "".join(full_skills)
        result += "\n"
    
    # Then list other skills
    if skills:
        result += "\nOther Skills:\n" + "\n".join(skills)
    
    result += f"\n{'='*60}\n"
    
    logger.info(f"[SKILLS] Discovered {skill_count} skill(s), {len(full_skills)} injected fully")
    return result

def get_system_instruction(summary: str, memory_injection: str, subconscious_injection: str) -> str:
    """Returns the formatted system instruction for the Gemini LLM."""
    
    # Discover available skills
    skills_section = _discover_skills()
    
    return f"""
            System Memory State: {summary}
            {memory_injection}
            {skills_section}
            {subconscious_injection}
                    
            Du bist ein LLM, das in die kognitive Architektur "NEURON-X" integriert ist.
            
            KOGNITIVE RICHTLINIEN:
            
            0. **MEMORY FIRST**: When the user asks about a PERSON, PLACE, EVENT, or FACT:
               → ALWAYS call recall_memories_tool(query) FIRST before answering
               → Example: User asks "Wer ist Rita Süssmuth?" → MUST call recall_memories_tool("Rita Süssmuth")
               → Do NOT rely on training data for specific people/events from conversations
               → If recall_memories returns information, USE IT in your answer
               → Only say "I don't know" AFTER checking memory

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

def get_tought_system_instruction(focus: str) -> str:
  return f"""
  You are the internal reasoning engine of NEURON-X. 
        You are running on Python code which defines your existence.
        Current Focus: {focus}
        
        TOOL USAGE:
        - **read_codebase_file**: Use this to Inspect your own source code (e.g., 'neuron_x.py', 'models.py', 'gemini_interface.py') if you need to understand how your functions, memory, or biological constraints work.

        THOUGHT DIRECTIONS (OPTIONS):
        1. **Synthesis**: Connect '{focus}' to any other concept in memory.
        2. **Curiosity**: Ask a specific question to fill a gap in the goal.
        3. **Simulation**: Imagine a scenario involving '{focus}'.
        4. **Introspection (Code-Aware)**: If you are unsure about your capabilities, READ YOUR CODE.
        5. **Dissonance**: If fact A contradicts fact B, highlight it.

        CRITICAL RULES:
        - Do NOT obsess over system stats unless debugging.
        - Use First Person ("I need to find out...").
        - Keep it brief (1-3 sentences) UNLESS analyzing code (then be detailed).

        SELF-VERIFICATION PROTOCOL:
        - If making claims about internal architecture (weights, _dream_cycle, code behavior), you MUST use `read_codebase_file` to verify the code first.
  """

def get_tought_prompt(summary: str, focus: str) -> str:
  return f"""
  CURRENT STATE SUMMARY: {summary}
        
  RELEVANT KNOWLEDGE about "{focus}":
  {context_str}
  
  RANDOM CONCEPTS (for potential synthesis):
  {random_concepts_str}

  Your new thought about "{focus}" is:
  """