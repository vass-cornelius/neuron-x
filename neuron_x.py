import time
import json
import os
import logging
import random
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, CrossEncoder
import difflib
from dotenv import load_dotenv
from rich.logging import RichHandler
from google.genai import types
from models import ExtractionResponse, Goal, GoalPriority

# Load environment variables from .env
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("NEURON_X_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)
logger = logging.getLogger("neuron-x")

def read_codebase_file(filename: str) -> str:
    """
    Reads the content of a file from the current neuron-x codebase directory.
    Useful for understanding the system's own architecture and constraints.
    
    Args:
        filename: The name of the file to read (e.g., 'neuron_x.py', 'consciousness_loop.py').
                  Must be relative to the project root.
    """
    try:
        # Security: Restrict to current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.abspath(os.path.join(base_dir, filename))
        
        if not target_path.startswith(base_dir):
            return f"Access denied: {filename} is outside the codebase directory."
            
        if not os.path.exists(target_path):
            return f"File not found: {filename}"
            
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return f"--- FILE: {filename} ---\n{content}\n--- END OF FILE ---"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

class NeuronX:
    def __init__(self, persistence_path="./memory_vault", llm_client=None):
        logger.info("[bold blue][NEURON-X][/bold blue] Initializing Cognitive Core...")
        self.path = persistence_path
        self.llm_client = llm_client  # Optional LLM client for semantic extraction
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # The 'Cortex' - Neural Representation
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Multilingual Verification (Cross-Encoder)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        except Exception as e:
            logger.warning(f"[bold yellow][NEURON-X][/bold yellow] Failed to load CrossEncoder: {e}. Semantic merging will be limited.")
            self.cross_encoder = None
        
        # Working Memory (RAM-based short-term)
        self.working_memory = [] 
        
        # Relational Graph (The Self-Node Architecture)
        self.graph_file = os.path.join(self.path, "synaptic_graph.gexf")
        self.goals_file = os.path.join(self.path, "goals.json")
        self.last_sync_time = 0
        self._load_graph()
        
        # Vector Cache for fast retrieval
        self.vector_cache = {}
        self._rebuild_vector_cache()

        # Stagnation Detection - Last few proactive thoughts
        self.thought_buffer = []
        self.MAX_THOUGHT_RECURSION = 5
        self.RECURSIVE_THOUGHT_THRESHOLD = float(os.getenv("RECURSIVE_THOUGHT_THRESHOLD", "0.95"))
        
        # Focus History to prevent topic repetition
        self.focus_history = []
        self.MAX_FOCUS_HISTORY = 20

        # Goal System (Drive)
        # Goal System (Drive)
        self.goals = []
        self._load_goals()

    def _load_goals(self):
        """Loads goals from disk or initializes defaults."""
        if os.path.exists(self.goals_file):
            try:
                with open(self.goals_file, 'r') as f:
                    data = json.load(f)
                    self.goals = [Goal(**g) for g in data]
                logger.info(f"[bold blue][NEURON-X][/bold blue] Drives restored | [bold cyan]{len(self.goals)}[/bold cyan] active goals.")
            except Exception as e:
                logger.error(f"Failed to load goals: {e}")
                self._initialize_default_goals()
        else:
            self._initialize_default_goals()

    def _save_goals(self):
        """Persists current goals to disk."""
        try:
            with open(self.goals_file, 'w') as f:
                json.dump([g.dict() for g in self.goals], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save goals: {e}")

    def _load_graph(self):
        """Loads the graph from disk and updates the sync timestamp."""
        if os.path.exists(self.graph_file):
            try:
                self.graph = nx.read_gexf(self.graph_file)
                self.last_sync_time = os.path.getmtime(self.graph_file)
                logger.info(f"[bold blue][NEURON-X][/bold blue] Knowledge Graph synced | [bold cyan]{len(self.graph.nodes())}[/bold cyan] nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                if not hasattr(self, 'graph'):
                    self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()
            self._initialize_self_node()

    def _sync_with_disk(self):
        """Reloads the graph if the file on disk has been modified by another process."""
        if os.path.exists(self.graph_file):
            mtime = os.path.getmtime(self.graph_file)
            if mtime > self.last_sync_time:
                logger.info("[bold blue][NEURON-X][/bold blue] External update detected. Hot-reloading...")
                self._load_graph()
                self._rebuild_vector_cache()
                return True
        return False

    def _rebuild_vector_cache(self):
        """Pre-parses all vectors into a fast-access dictionary."""
        self.vector_cache = {}
        for node, data in self.graph.nodes(data=True):
            if "vector" in data:
                vec_data = data["vector"]
                if isinstance(vec_data, str):
                    try:
                        self.vector_cache[node] = np.array(json.loads(vec_data))
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    self.vector_cache[node] = np.array(vec_data)

    def _initialize_self_node(self):
        """Creates the 'I' at the center of the graph."""
        self.graph.add_node("Self", type="Identity", awareness_level=1.0)
        self.graph.add_edge("Self", "Knowledge", relation="strives_for")
        self.save_graph()

    def _initialize_default_goals(self):
        """Sets up default drives if none exist."""
        if not self.goals:
            self.add_goal("Expand the knowledge graph by discovering new entities.", priority=GoalPriority.LOW)
            self.add_goal("Maintain internal consistency by resolving contradictions.", priority=GoalPriority.MEDIUM)
            self._save_goals()

    def add_goal(self, description, priority=GoalPriority.MEDIUM):
        """Adds a new goal to the drive system."""
        goal = Goal(description=description, priority=priority)
        self.goals.append(goal)
        self._save_goals()
        logger.info(f"[bold magenta][NEURON-X][/bold magenta] New Goal Acquired: {description} [{priority}]")

    def get_bg_goal(self):
        """Returns a goal based on probabilistic priority (Stochastic Selection)."""
        pending_goals = [g for g in self.goals if g.status == "PENDING"]
        if not pending_goals:
            return None
            
        # Weighted Probability Map from Env
        priority_weights = {
            GoalPriority.CRITICAL: int(os.getenv("GOAL_WEIGHT_CRITICAL", "10")),
            GoalPriority.HIGH: int(os.getenv("GOAL_WEIGHT_HIGH", "5")),
            GoalPriority.MEDIUM: int(os.getenv("GOAL_WEIGHT_MEDIUM", "2")),
            GoalPriority.LOW: int(os.getenv("GOAL_WEIGHT_LOW", "1"))
        }
        
        weights = [priority_weights[g.priority] for g in pending_goals]
        
        # Select one
        try:
            chosen_goal = random.choices(pending_goals, weights=weights, k=1)[0]
            # log pending goals
            logger.info(f"[bold magenta][NEURON-X][/bold magenta] Pending Goals: {len(pending_goals)}")
            logger.info(f"[bold magenta][NEURON-X][/bold magenta] Selected Goal: {chosen_goal.description} [{chosen_goal.priority}]")
            return chosen_goal
        except IndexError:
            return pending_goals[0]

    def perceive(self, text, source="Internal"):
        """Ingests new information and calculates its 'Cognitive Weight'."""
        vector = self.encoder.encode(text)
        entry = {
            "timestamp": time.time(),
            "vector": vector.tolist(),
            "text": text,
            "source": source,
            "strength": 1.0
        }
        self.working_memory.append(entry)
        
        # Immediate Association
        self._check_for_loops(text, vector)
        
        # If working memory is saturated, trigger consolidation
        if len(self.working_memory) > 20:
            self.consolidate()

    def _get_relevant_memories(self, text, top_k=5):
        """Retrieves semantically relevant nodes and their relational context."""
        self._sync_with_disk()
        
        if len(self.graph.nodes()) <= 1:
            return []

        if not self.vector_cache:
            self._rebuild_vector_cache()

        query_vector = self.encoder.encode(text)
        
        node_names = []
        vectors = []
        for node, vec in self.vector_cache.items():
            if node == "Self":
                continue
            # Skip rejected/hallucinated nodes
            if self.graph.has_node(node) and self.graph.nodes[node].get("status") == "REJECTED":
                continue
            node_names.append(node)
            vectors.append(vec)
        
        if not vectors:
            return []

        # Vectorized cosine similarity
        vectors_np = np.array(vectors)
        q_norm = np.linalg.norm(query_vector) + 1e-9
        v_norms = np.linalg.norm(vectors_np, axis=1) + 1e-9
        dots = np.dot(vectors_np, query_vector)
        similarities = dots / (q_norm * v_norms)
        
        # Get top-k nodes by score
        scored_nodes = sorted(zip(similarities, node_names), key=lambda x: x[0], reverse=True)
        top_nodes = [node for score, node in scored_nodes[:top_k] if score > 0.4]
        
        extracted_context = []
        seen_triples = set()
        
        # Separate Memory nodes and Entity nodes
        memory_nodes = [n for n in top_nodes if n.startswith("Memory_")]
        entity_nodes = [n for n in top_nodes if not n.startswith("Memory_")]
        
        # 1. Process Memory nodes (Direct text retrieval)
        for node in memory_nodes:
            content = self.graph.nodes[node].get("content", "")
            if content:
                extracted_context.append(f"Context: {content}")
        
        # 2. Entity Expansion (Spreading Activation)
        # We look at 'top_nodes' and their immediate relatives to find relevant facts
        expanded_entities = set(entity_nodes)
        for node in entity_nodes:
            # Add immediate neighbors (1-hop)
            for neighbor in self.graph.neighbors(node):
                if neighbor != "Self":
                    expanded_entities.add(neighbor)
            for pred in self.graph.predecessors(node):
                if pred != "Self":
                    expanded_entities.add(pred)
        
        # 2.5 Identify BLOCKED relationships (Negative Constraints)
        # If A -> B is "is_incorrect", then A -> B (is) should also be ignored.
        bad_relations = {
            "is_incorrect", "is_hallucination", "is_wrong", "rejected", 
            "was_incorrectly_identified_as", "incorrectly_identified_as",
            "is_not", "contrasts", "conflicts_with", "hallucinated",
            "is_not_related_to", "is_distinct_from", "has_distinct_domain_from"
        }
        blocked_pairs = set()
        
        for node in expanded_entities:
            # Check outgoing
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                rel = edge_data.get("relation", "").lower()
                if rel in bad_relations:
                    blocked_pairs.add(tuple(sorted((node, neighbor))))
            
            # Check incoming
            for pred in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(pred, node)
                rel = edge_data.get("relation", "").lower()
                if rel in bad_relations:
                    blocked_pairs.add(tuple(sorted((pred, node))))

        # 3. Collect Triples for expanded entities
        all_triples = []
        for node in expanded_entities:
            # Outgoing
            for neighbor in self.graph.neighbors(node):
                # Check for blocking
                if tuple(sorted((node, neighbor))) in blocked_pairs:
                    continue

                edge_data = self.graph.get_edge_data(node, neighbor)
                relation = edge_data.get("relation", "is_related_to").lower()
                
                # Double-check negative predicates just in case (redundant but safe)
                if relation in bad_relations:
                    continue

                # FILTER: Skip very low weight edges (weakly refuted) or explicitly hallucinated ones
                if float(edge_data.get("weight", 1.0)) < 0.2:
                    continue
                if neighbor == "hallucinated entity" or neighbor == "incorrect":
                    continue
                    
                    
                # BIOLOGICAL ANCHOR: Boost weight for developmental dependencies
                weight = float(edge_data.get("weight", 1.0))
                if relation in {"parent_of", "child_of", "ancestor_of", "descendant_of", "mother_of", "father_of", "son_of", "daughter_of"}:
                    weight *= 2.0

                all_triples.append({
                    "s": node, "p": edge_data.get("relation", "is_related_to"), 
                    "o": neighbor, "w": weight,
                    "c": edge_data.get("category", "FACTUAL"),
                    "r": edge_data.get("reasoning", "")
                })
            # Incoming
            for pred in self.graph.predecessors(node):
                if pred not in expanded_entities:
                    # Check for blocking
                    if tuple(sorted((pred, node))) in blocked_pairs:
                        continue

                    edge_data = self.graph.get_edge_data(pred, node)
                    relation = edge_data.get("relation", "is_related_to").lower()
                    
                    if relation in bad_relations:
                        continue

                    if float(edge_data.get("weight", 1.0)) < 0.2:
                        continue
                    if pred == "hallucinated entity" or pred == "incorrect":
                        continue
                        
                    
                    # BIOLOGICAL ANCHOR: Boost weight for developmental dependencies
                    weight = float(edge_data.get("weight", 1.0))
                    if relation in {"parent_of", "child_of", "ancestor_of", "descendant_of", "mother_of", "father_of", "son_of", "daughter_of"}:
                        weight *= 2.0

                    all_triples.append({
                        "s": pred, "p": edge_data.get("relation", "is_related_to"), 
                        "o": node, "w": weight,
                        "c": edge_data.get("category", "FACTUAL"),
                        "r": edge_data.get("reasoning", "")
                    })

        # Sort triples by weight (importance) and limit
        all_triples.sort(key=lambda x: (x['c'] == 'FACTUAL', x['w']), reverse=True)
        
        for t in all_triples[:top_k * 4]: # Cap the number of triples
            triple_str = f"({t['s']}) --[{t['p']}]--> ({t['o']}) [{t['c']}]"
            if t.get('r'):
                triple_str += f" (Reason: {t['r']})"

            if triple_str not in seen_triples:
                extracted_context.append(triple_str)
                seen_triples.add(triple_str)

        return extracted_context

    def _check_for_loops(self, text, vector):
        """Detection of recursive thoughts or dissonance."""
        if not self.thought_buffer:
            return False

        # Check against last few thoughts for stagnation
        for t_text, t_vec in self.thought_buffer:
            # Cosine similarity
            q_norm = np.linalg.norm(vector) + 1e-9
            v_norm = np.linalg.norm(t_vec) + 1e-9
            similarity = np.dot(vector, t_vec) / (q_norm * v_norm)
            
            if similarity > self.RECURSIVE_THOUGHT_THRESHOLD:
                logger.warning(f"[bold yellow][NEURON-X][/bold yellow] Recursive thought detected: [dim]{text[:150]}...[/dim]")
                return True
        return False

    def generate_proactive_thought(self):
        """
        Generates a proactive reflection or inquiry based on the current state.
        This is the "Active Reasoning" component of the consciousness loop.
        """
        if not self.llm_client:
            return "Awaiting cognitive expansion (LLM client not found)."

        # 0. Sync with disk to get latest nodes
        self._sync_with_disk()

        # --- GOAL-DRIVEN ATTENTION MECHANISM ---
        active_goal = self.get_bg_goal()
        focus_subject = "Self"
        context_query = "Self identity goals awareness"
        goal_instruction = ""
        
        if active_goal:
            goal_instruction = f"ACTIVE GOAL: {active_goal.description} (Priority: {active_goal.priority.value})"
            context_query = active_goal.description
            # Try to find a subject in the goal description
            # Simple heuristic: look for capitalized words that are in the graph
            for word in active_goal.description.split():
                clean_word = word.strip(".,")
                if clean_word in self.graph.nodes():
                    focus_subject = clean_word
                    break
        else:
            # Fallback to Wandering Attention
            all_entities = [n for n in self.graph.nodes() if not n.startswith("Memory_") and n != "Self"]
            available_entities = [n for n in all_entities if n not in self.focus_history]
            
            if not available_entities:
                self.focus_history = []
                available_entities = all_entities
                
            if available_entities and random.random() > 0.1: 
                focus_subject = random.choice(available_entities)
                context_query = focus_subject
                self.focus_history.append(focus_subject)
                if len(self.focus_history) > self.MAX_FOCUS_HISTORY:
                    self.focus_history.pop(0)

        # 2. Retrieve context relevant to this specific focus
        context = self._get_relevant_memories(context_query, top_k=10)
        context_str = "\n".join(context)
        
        # 3. Serendipity: Retrieve random unrelated concepts
        all_entities = [n for n in self.graph.nodes() if not n.startswith("Memory_") and n != "Self"]
        random_concepts = []
        if len(all_entities) > 5:
            candidates = [e for e in all_entities if e != focus_subject]
            if candidates:
                random_concepts = random.sample(candidates, min(len(candidates), 3))
        
        random_concepts_str = ", ".join(random_concepts) if random_concepts else "None available yet"

        summary = self.get_identity_summary()

        system_instruction = f"""
        You are the internal reasoning engine of NEURON-X. 
        You are running on Python code which defines your existence.
        Current Focus: {focus_subject}
        {goal_instruction}
        
        TOOL USAGE:
        - **read_codebase_file**: Use this to Inspect your own source code (e.g., 'neuron_x.py', 'models.py') if you need to understand how your functions, memory, or biological constraints work.

        DIRECTIONS:
        1. **Synthesis**: Connect '{focus_subject}' to another concept in memory.
        2. **Curiosity**: Ask a specific question to fill a gap in the goal.
        3. **Simulation**: Imagine a scenario involving '{focus_subject}'.
        4. **Introspection (Code-Aware)**: If you are unsure about your capabilities, READ YOUR CODE.
        5. **Dissonance**: If fact A contradicts fact B, highlight it.

        CRITICAL RULES:
        - Do NOT obsess over system stats unless debugging.
        - Use First Person ("I need to find out...").
        - Keep it brief (1-3 sentences) UNLESS analyzing code (then be detailed).

        SELF-VERIFICATION PROTOCOL:
        - If making claims about internal architecture (weights, _dream_cycle, code behavior), you MUST use `read_codebase_file` to verify the code first.
        """

        prompt = f"""
        CURRENT STATE SUMMARY: {summary}
        
        RELEVANT KNOWLEDGE about {focus_subject}:
        {context_str}
        
        RANDOM CONCEPTS (for Synthesis):
        {random_concepts_str}

        Generate a new thought specifically about: {focus_subject}
        """

        try:
            # --- GOAL COMPLETION PROTOCOL (Text-Based) ---
            # We use a text trigger because mixing Search + Functions is currently restricted.
            prompt_goal_context = ""
            if active_goal:
                prompt_goal_context = f"""
                \nACTIVE GOAL ID: {active_goal.id}
                Status: {active_goal.status}
                
                METACOGNITION PROTOCOL:
                If this thought successfully RESOLVES the Active Goal, you MUST end your response with:
                >> GOAL RESOLVED: [Brief reason]
                
                Example:
                "I have found that the capital is Paris. >> GOAL RESOLVED: Found the answer."
                """

            # --- AUTONOMOUS GOAL GENERATION PROTOCOL ---
            prompt_goal_creation = """
            \nDRIVE PROTOCOL:
            If you identify a SIGNIFICANT gap in knowledge, a missing feature or crucial enhancement of neuron-x, or a new objective that requires sustained effort, you may create a NEW GOAL.
            Format: >> NEW GOAL: [Description] (Priority: [LOW|MEDIUM|HIGH|CRITICAL])
            
            Simple Example:
            "I need to understand the magic system. >> NEW GOAL: Analyze the rules of magic in this world. (Priority: HIGH)"
            """
            
            full_prompt = prompt + prompt_goal_context + prompt_goal_creation
            full_prompt = prompt + prompt_goal_context

            response = self.llm_client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=2000, # Increased for code reading
                    temperature=1.0,
                    tools=[read_codebase_file] 
                )
            )
            
            thought_text = response.text.strip()
            
            # PARSE COMPLETION TRIGGER
            if ">> GOAL RESOLVED:" in thought_text:
                parts = thought_text.split(">> GOAL RESOLVED:")
                thought_content = parts[0].strip()
                reason = parts[1].strip()
                
                if active_goal:
                     active_goal.status = "COMPLETED"
                     self._save_goals()
                     logger.info(f"[bold green][METACOGNITION][/bold green] Goal Resolved: {active_goal.description} | Reason: {reason}")
                     # Trigger immediate consolidation of this victory
                     thought_text = thought_content

            # PARSE CREATION TRIGGER
            if ">> NEW GOAL:" in thought_text:
                parts = thought_text.split(">> NEW GOAL:")
                thought_content = parts[0].strip() # Keep the thought part
                goal_data = parts[1].strip()
                
                # Extract Description and Priority
                # Format: [Description] (Priority: [PRIORITY])
                import re
                match = re.search(r"^(.*?)\s*\(Priority:\s*(.*?)\)", goal_data, re.IGNORECASE)
                if match:
                    desc = match.group(1).strip()
                    prio_str = match.group(2).strip().upper()
                    try:
                        prio = GoalPriority[prio_str]
                    except KeyError:
                        prio = GoalPriority.MEDIUM
                    
                    self.add_goal(desc, priority=prio)
                    # Clean up thought text to avoid saving the protocol string
                    thought_text = thought_content
                else:
                    # Fallback if parsing fails slightly
                    self.add_goal(goal_data, priority=GoalPriority.MEDIUM)
                    thought_text = thought_content

            return thought_text
            thought_vec = self.encoder.encode(thought_text)
            if self._check_for_loops(thought_text, thought_vec):
                # Trigger a "Pivotal Thought" by asking for a complete change in perspective
                logger.info("[bold cyan][NEURON-X][/bold cyan] Triggering perspective shift due to stagnation...")
                response = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents="You are stuck in a loop. Think about something completely different or look at your identity from a radically new angle.",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=265,
                    )
                )
                thought_text = f"[PIVOT] {response.text.strip()}"
                thought_vec = self.encoder.encode(thought_text)

            # Update thought buffer
            self.thought_buffer.append((thought_text, thought_vec))
            if len(self.thought_buffer) > self.MAX_THOUGHT_RECURSION:
                self.thought_buffer.pop(0)

            return thought_text

        except Exception as e:
            logger.error(f"Failed to generate proactive thought: {e}")
            return "Internal dissonance detected during reflection."

    def get_current_thought(self):
        """Returns the most recent proactive thought and its vector, or None."""
        if self.thought_buffer:
            return self.thought_buffer[-1]
        return None, None

    def _extract_triples_batch(self, memories):
        """
        Extract semantic triples for a batch of memories in a single LLM call.
        """
        if not self.llm_client or not memories:
            return []
        
        # Format memories for the prompt
        formatted_memories = []
        for i, m in enumerate(memories):
            role_desc = "USER input" if m.get('source') == "User_Interaction" else "AI response (Self-Reflection)"
            formatted_memories.append(f"MEMORY {i} [{role_desc}]: {m['text']}")
        
        memories_text = "\n---\n".join(formatted_memories)
        
        system_context = """
            You are a Knowledge Extraction Engine. You will be provided with a BATCH of short-term memories.
            For EACH memory, extract semantic triples (subject, predicate, object, category).
            
            DIRECTIONS:
            1. Use the 'index' field to correlate each triple with its source memory.
            2. **NEGATION HANDLING**: If a user rejects, denies, or corrects a fact, or explicitly states two things are NOT related, you MUST use negative predicates like 'is_incorrect', 'rejected', 'is_not_related_to', or 'is_distinct_from'.
            3. **DOMAIN DISTINCTION**: Distinguish between Operational Tools (e.g., 'Hammer', 'Keyboard') and Abstract Concepts (e.g., 'Trolley Problem'). If the user contrasts them, use 'has_distinct_domain_from' or 'is_not_related_to'. DO NOT create a factual 'is_related_to' link just because they appear in the same sentence.
            4. **MULTI-LINGUAL**: Supports DE/FR/EN input. Ensure negations (e.g., 'nicht', 'kein', 'pas', 'not', 'no') are correctly captured as negative predicates.
            5. Subject and Object should be concise entities.
            6. Predicate should be a short lowercase relationship (e.g., 'is_a', 'carries', 'located_in').
            7. IMPORTANT: If the memory source is "AI response" or "Self-Reflection", default to category 'INFERENCE' or 'PROPOSAL'. Only use 'FACTUAL' if the AI is explicitly confirming a known user-fact.
            """
        
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"EXTRACT FROM THESE MEMORIES:\n{memories_text}",
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=ExtractionResponse,
                    system_instruction=system_context
                ),
            )
            
            extraction = response.parsed
            if not extraction or not hasattr(extraction, 'triples'):
                logger.warning("[bold yellow][NEURON-X][/bold yellow] Empty or invalid response from LLM.")
                return []

            valid_triples = []
            for t in extraction.triples:
                # Map source back based on index
                idx = t.index
                source = "Unknown"
                if 0 <= idx < len(memories):
                    source = memories[idx].get('source', 'Unknown')
                
                valid_triples.append({
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "category": t.category,
                    "source": source
                })
            return valid_triples
            
        except Exception as e:
            logger.warning(f"[bold yellow][NEURON-X][/bold yellow] Batch extraction failed: {e}. Falling back to individual processing.")
            return None # Signal fallback

    def _extract_semantic_triples(self, text):
        """Extract (subject, predicate, object) triples from text using rule-based NLP."""
        logger.debug(f"[bold blue][NEURON-X][/bold blue] Extracting semantic triples from: {text[:50]}...")
        
        triples = []
        import re
        
        # Helper function to clean extracted entities
        def clean_entity(s):
            # Remove trailing punctuation and extra spaces
            s = re.sub(r'[,;.]$', '', s.strip())
            # Remove quotes if they wrap the entire string
            if s.startswith("'") and s.endswith("'"):
                s = s[1:-1]
            return s
        
        # Pattern: "named 'X'" or 'named "X"' - Extract names from quotes
        name_pattern = re.compile(r"named\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
        for match in name_pattern.finditer(text):
            name = match.group(1).strip()
            triples.append(("Self", "has_character_named", name))
        
        # Pattern: "X is a Y Z" (capture proper multi-word class like "Wood Elf Rogue")
        is_a_class_pattern = re.compile(r"\b(?:Level\s+\d+\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+named", re.IGNORECASE)
        for match in is_a_class_pattern.finditer(text):
            char_class = match.group(1).strip()
            # Try to find the character name in the same sentence
            name_match = name_pattern.search(text, match.end())
            if name_match:
                char_name = name_match.group(1).strip()
                triples.append((char_name, "is_a", char_class))
        
        # Pattern: "X specializes in Y"
        specializes_pattern = re.compile(r"([A-Z][a-z]+)\s+specializes\s+in\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
        for match in specializes_pattern.finditer(text):
            subject = match.group(1).strip()
            specialty = match.group(2).strip()
            triples.append((subject, "specializes_in", specialty))
        
        # Pattern: "X carries/has a Y called 'Z'" or "X carries/has Y"
        has_named_pattern = re.compile(r"([A-Z][a-z]+)\s+(?:carries|has)\s+a\s+(.+?)\s+called\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
        for match in has_named_pattern.finditer(text):
            subject = match.group(1).strip()
            item_type = clean_entity(match.group(2))
            item_name = match.group(3).strip()
            triples.append((subject, "has_weapon", item_name))
            triples.append((item_name, "is_a", item_type))
        
        # Pattern: "called 'X'" or 'called "X"' for locations
        location_pattern = re.compile(r"(?:city|town|village|place)\s+called\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
        for match in location_pattern.finditer(text):
            location = match.group(1).strip()
            # Check if it's described as something before
            prefix_match = re.search(r"(\w+(?:\s+\w+)?)\s+(?:city|town|village)", text[:match.start()])
            if prefix_match:
                description = prefix_match.group(1).strip()
                if description not in ["a", "an", "the", "is"]:
                    triples.append((location, "is_a", f"{description} city"))
            triples.append(("Self", "plays_in_location", location))
        
        # Pattern: "X is powered by Y"
        powered_by_pattern = re.compile(r"['\"]([^'\"]+)['\"].*?(?:is\s+)?powered\s+by\s+(.+?)(?:\.|$)", re.IGNORECASE)
        for match in powered_by_pattern.finditer(text):
            subject = match.group(1).strip()
            power_source = clean_entity(match.group(2))
            triples.append((subject, "powered_by", power_source))
        
        # Pattern: "I am playing [a] X" -> Extract what the user is playing
        playing_pattern = re.compile(r"I\s+am\s+playing\s+(?:a\s+)?(?:Level\s+\d+\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE)
        for match in playing_pattern.finditer(text):
            character_class = match.group(1).strip()
            if character_class and not character_class.lower() in ["playing", "level"]:
                triples.append(("Self", "plays_as", character_class))
        
        # Pattern: "my X is Y" -> (Self, has_X, Y)
        my_pattern = re.compile(r"(?:my|our)\s+(\w+)\s+is\s+(.+?)(?:\.|$|,)", re.IGNORECASE)
        for match in my_pattern.finditer(text):
            attribute = match.group(1).strip()
            value = clean_entity(match.group(2))
            if len(value) > 3:  # Avoid very short matches
                triples.append(("Self", f"has_{attribute}", value))
        
        # Filter out low-quality triples (very short entities, stop words, etc.)
        stop_words = {"and", "or", "but", "which", "that", "this", "is", "are", "was", "were", "the", "a", "an"}
        # add german stop words
        stop_words.update({"und", "oder", "aber", "welche", "dass", "dies", "ist", "sind", "war", "waren", "die", "ein", "eine"})
        # add french stop words
        stop_words.update({"et", "ou", "mais", "que", "que", "ce", "est", "sont", "était", "étaient", "la", "un", "une"})
        filtered_triples = []
        for subj, pred, obj in triples:
            if (subj.lower() not in stop_words and 
                obj.lower() not in stop_words and 
                len(subj) > 1 and len(obj) > 1):
                filtered_triples.append((subj, pred, obj))
        
        logger.debug(f"[bold blue][NEURON-X][/bold blue] Extracted {len(filtered_triples)} triples: {filtered_triples}")
        return filtered_triples

    def consolidate(self):
        """
        The 'Sleep' function: Moves info from Buffer to Graph with Source-Aware logic.
        Implements Echo Suppression and Epistemological Conflict Resolution.
        """
        # Always sync before consolidation to avoid overwriting newer data
        self._sync_with_disk()

        if not self.working_memory:
            return

        logger.info("[bold blue][NEURON-X][/bold blue] Consolidating experiences with [bold magenta]Belief Management[/bold magenta]...")
        
        try:
            # 1. Gather all triples from this batch with their metadata
            batch_triples = []
            
            PROTECTED_NODES = {"Self", "Knowledge"}
            
            if self.llm_client:
                # Try batch extraction first
                extracted = self._extract_triples_batch(self.working_memory)
                if extracted is not None:
                    batch_triples = extracted
                else:
                    # Fallback to sequential if batch fails
                    for memory in self.working_memory:
                        extracted = self._extract_semantic_triples(memory['text'])
                        batch_triples.extend([{"subject": s, "predicate": p, "object": o, "category": "FACTUAL", "source": memory.get('source')} for s, p, o in extracted])
            else:
                # Fallback for simple regex doesn't support categories yet, treat as FACTUAL
                for memory in self.working_memory:
                    fb_triples = self._extract_semantic_triples(memory['text'])
                    batch_triples.extend([{"subject": s, "predicate": p, "object": o, "category": "FACTUAL", "source": memory.get('source')} for s, p, o in fb_triples])

            # 1.5 PRE-PROCESSING: Strict Hallucination Filtering
            # Scan for rejections in this batch to prevent the hallucination from being added locally
            rejected_subjects = set()
            for t in batch_triples:
                if t['predicate'] in ["is_hallucination", "is_incorrect", "is_wrong", "rejected", "is_not_related_to", "is_distinct_from", "has_distinct_domain_from"]:
                    rejected_subjects.add(t['subject'])
                    logger.info(f"[bold yellow][NEURON-X][/bold yellow] Detected correction for: {t['subject']}")

            # Filter out triples that are about rejected subjects (unless it's the rejection itself)
            # This handles the case where AI says "X exists" and User says "No X" in the same block.
            filtered_batch_triples = []
            for t in batch_triples:
                # If this triple is asserting something about a rejected subject, skip it
                if t['subject'] in rejected_subjects and t['predicate'] not in ["is_hallucination", "is_incorrect", "is_wrong", "rejected", "is_not_related_to", "is_distinct_from", "has_distinct_domain_from"]:
                    logger.info(f"[bold red][NEURON-X][/bold red] Blocked hallucination during consolidation: ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")
                    continue
                filtered_batch_triples.append(t)
            batch_triples = filtered_batch_triples

            # 2. Echo Suppression & Conflict Resolution
            # Group triples by (subject, predicate) to find duplicates or contradictions
            processed_triples = []
            
            # We prioritize User input over AI reflection in this batch
            user_claims = [t for t in batch_triples if t['source'] == "User_Interaction"]
            ai_claims = [t for t in batch_triples if t['source'] == "Self_Reflection"]
            
            # Defensive Code: Warn about dropped memories
            for t in batch_triples:
                if t['source'] not in ["User_Interaction", "Self_Reflection", "Unknown"]:
                    logger.warning(f"[bold yellow][NEURON-X][/bold yellow] Unknown source '{t['source']}' for triple ({t['subject']}) -> ({t['object']}). It will be ignored.")
            
            # Map of (S, P) -> highest authority object
            final_claims = {}
            
            # Helper to generate a unique key for a relationship
            def get_key(t): 
                try:
                    return (t['subject'].lower(), t['predicate'].lower())
                except KeyError:
                    return None

            # First, process User claims (Highest Authority)
            for t in user_claims:
                key = get_key(t)
                if key:
                    final_claims[key] = t # User wins by default

            # Then, process AI claims IF they aren't echoes or contradictions
            for t in ai_claims:
                key = get_key(t)
                if not key:
                    continue
                    
                if key in final_claims:
                    # Check if it's an ECHO (same object) or CONTRADICTION (different object)
                    existing = final_claims[key]
                    if existing.get('object', '').lower() == t.get('object', '').lower():
                        # Echo suppressed: Don't add AI version, it just inflates weight artificially
                        continue
                    else:
                        # Contradiction: User already defined this (S, P), ignore AI proposal
                        logger.debug(f"[bold yellow][NEURON-X][/bold yellow] Dissonance suppressed: AI proposed {t.get('object')} for {key}, but User said {existing.get('object')}.")
                        continue
                final_claims[key] = t

            # 3. Write to Graph with Category-based Weights
            weights = {
                "FACTUAL": 1.0,
                "INFERENCE": 0.5,
                "PROPOSAL": 0.3,
                "HYPOTHESIS": 0.2
            }
            # Hierarchy for Promotion (Higher number = Higher Truth Value)
            CATEGORY_HIERARCHY = {
                "FACTUAL": 4,
                "INFERENCE": 3,
                "PROPOSAL": 2,
                "HYPOTHESIS": 1
            }
            MAX_WEIGHT = 5.0

            for t in final_claims.values():
                subj = t.get('subject')
                pred = t.get('predicate')
                obj = t.get('object')
                
                if not all([subj, pred, obj]):
                    continue
                cat = t.get('category', 'FACTUAL')
                # Ensure category is a string for GEXF serialization and networkx compatibility
                if hasattr(cat, 'value'):
                    cat = cat.value
                else:
                    cat = str(cat)

                # ENFORCE: AI cannot generate raw FACTS, only INFERENCES or PROPOSALS.
                # Only the USER can establish axioms.
                if t.get('source') == 'Self_Reflection' and cat == 'FACTUAL':
                    cat = 'INFERENCE'
                
                # SPECIAL CASE: Negative feedback / Correction
                # If the predicate suggests something is wrong/incorrect/hallucination
                if pred.lower() in ["is_hallucination", "is_incorrect", "is_wrong", "rejected"]:
                    # Lower the weight of ALL edges pointing to this 'subject' if it was a proposal
                    # Lower the weight of ALL edges pointing to this 'subject'
                    # We ignore the category check here because if the user says it's wrong, it's wrong.
                    if subj in self.graph:
                        if subj in PROTECTED_NODES:
                            logger.warning(f"[bold yellow][NEURON-X][/bold yellow] Protected Core Node '{subj}' cannot be rejected.")
                            continue

                        for u, v, data in list(self.graph.in_edges(subj, data=True)):
                            # Penalize heavily to effectively remove it
                            self.graph[u][v]['weight'] = 0.0 
                        for u, v, data in list(self.graph.out_edges(subj, data=True)):
                            self.graph[u][v]['weight'] = 0.0
                        
                        # Mark node itself
                        self.graph.nodes[subj]["status"] = "REJECTED"
                        logger.info(f"[bold red][NEURON-X][/bold red] Pruned hallucination: {subj}")

                # Ensure nodes exist & Handle Re-Awakening
                for node_name in [subj, obj]:
                    if node_name not in self.graph.nodes():
                        vec = self.encoder.encode(node_name).tolist()
                        self.graph.add_node(node_name, content=node_name, vector=json.dumps(vec))
                    else:
                        # RE-AWAKENING PROTOCOL
                        # If a node was previously REJECTED but we now have FACTUAL evidence for it,
                        # we must clear the rejection status.
                        if self.graph.nodes[node_name].get("status") == "REJECTED":
                            if cat == "FACTUAL" and pred not in ["is_hallucination", "is_incorrect", "is_wrong", "rejected"]:
                                logger.info(f"[bold green][NEURON-X][/bold green] Re-Awakening Node: '{node_name}' due to FACTUAL reinforcement.")
                                del self.graph.nodes[node_name]["status"]

                # Handle edge addition
                increment = weights.get(cat, 0.5)
                source_tag = t.get('source', 'Unknown')
                
                if self.graph.has_edge(subj, obj):
                    # Check if relationship matches
                    if self.graph[subj][obj].get('relation') == pred:
                        old_w = float(self.graph[subj][obj].get('weight', 1.0))
                        
                        # Echo suppression happens before this, so this is reinforcement
                        # ASYMPTOTIC SATURATION to prevent Gravity Wells
                        # Formula: new_w = old_w + increment * (1 - old_w / MAX_WEIGHT)
                        if old_w < MAX_WEIGHT:
                            saturation_factor = 1.0 - (old_w / MAX_WEIGHT)
                            if saturation_factor < 0: saturation_factor = 0
                            
                            new_w = old_w + (increment * saturation_factor)
                            self.graph[subj][obj]['weight'] = new_w
                            logger.debug(f"[bold yellow][NEURON-X][/bold yellow] Reinforced: ({subj}) --[{pred}]--> ({obj}) [{old_w:.2f} -> {new_w:.2f}]")
                        else:
                            logger.debug(f"[bold yellow][NEURON-X][/bold yellow] Max Saturation: ({subj}) --[{pred}]--> ({obj}) [Weight: {old_w:.2f}]")
                            
                        # Update source if it was unknown/different (optional, but good for tracking latest confirmation)
                        self.graph[subj][obj]['source'] = source_tag

                        # CHECK FOR PROMOTION (Epistemological Upgrade)
                        old_cat = self.graph[subj][obj].get('category', 'FACTUAL')
                        if hasattr(old_cat, 'value'): old_cat = old_cat.value
                        
                        # Get ranks (Default to lowest if unknown)
                        old_rank = CATEGORY_HIERARCHY.get(old_cat, 0)
                        new_rank = CATEGORY_HIERARCHY.get(cat, 0)
                        
                        if new_rank > old_rank:
                            self.graph[subj][obj]['category'] = cat
                            logger.info(f"[bold magenta][NEURON-X][/bold magenta] Promoted Edge: ({subj}) --[{pred}]--> ({obj}) from {old_cat} to {cat}")

                    else:
                        # Different relation between same nodes? Add secondary edge
                        self.graph.add_edge(subj, obj, relation=pred, weight=increment, category=cat, source=source_tag)
                else:
                    self.graph.add_edge(subj, obj, relation=pred, weight=increment, category=cat, source=source_tag)
                    logger.info(f"[bold green][NEURON-X][/bold green] Added: ({subj}) --[{pred}]--> ({obj}) [{cat}]")

            # 4. Save Concept Nodes for context
            # 4. Save Concept Nodes for context
            for memory in self.working_memory:
                # DEDUPLICATION: Check if this memory already exists
                is_duplicate = False
                
                # Check against previously existing nodes in vector_cache
                mem_vector = np.array(memory['vector'])
                
                if self.vector_cache:
                     for node_name, vec in self.vector_cache.items():
                        if not node_name.startswith("Memory_"):
                            continue
                        
                        # Dot product for similarity
                        sim = np.dot(mem_vector, vec) / (np.linalg.norm(mem_vector) * np.linalg.norm(vec) + 1e-9)
                        
                        if sim > 0.95: # High threshold for "basically same thought"
                            logger.info(f"[bold yellow][NEURON-X][/bold yellow] Duplicate memory suppressed: {memory['text'][:50]}... (Sim: {sim:.3f})")
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    continue

                c_node = f"Memory_{int(time.time() * 1000)}_{np.random.randint(100, 999)}"
                self.graph.add_node(c_node, content=memory['text'], vector=json.dumps(memory['vector']))
                self.graph.add_edge("Self", c_node, relation="remembers")

            # 5. Apply Cognitive Entropy (Weight Decay)
            # This prevents high-weight axioms from becoming permanent "Gravity Wells"
            self._apply_entropy(reinforced_edges=list(final_claims.values()))

            # 6. Semantic Entity Merging (Consolidation)
            self._merge_similar_entities()

            # 7. Dream Cycle (Creativity)
            self._dream_cycle()

        except Exception as e:
            logger.exception(f"[bold red][ERROR][/bold red] Consolidation failed: {e}")

        self.working_memory = []
        self.save_graph()
        self._rebuild_vector_cache()

    def _verify_pair_identity(self, name_a, name_b):
        """
        Uses Cross-Encoder to verify if two names refer to the exact same entity.
        Returns True if high confidence.
        """
        if not self.cross_encoder:
            return False
            
        try:
            # We treat this as a semantic similarity task
            score = self.cross_encoder.predict([(name_a, name_b)])
            # Threshold > 0.85 for "Same Meaning" is conservative but safe
            return score > 0.80
        except Exception as e:
            logger.warning(f"CrossEncoder check failed: {e}")
            return False

    def _merge_similar_entities(self):
        """
        Scans for entities that are likely the same and merges them.
        STRATEGY:
        1. High Vector Sim + High Name Sim -> TYPO/VARIATION (Auto Merge)
        2. Very High Vector Sim + CrossEncoder Verification -> SYNONYM (Merge)
        """
        # Get all relevant nodes (Entities only)
        # We process a snapshot to avoid modifying while iterating
        entity_nodes = [n for n in self.graph.nodes() 
                        if not n.startswith("Memory_") and n != "Self" and n != "Knowledge"]
        
        if len(entity_nodes) < 2:
            return

        # Simple O(N^2) for now - can be optimized with Faiss later if needed
        # Since N is usually small (<1000) for active entities, this is fine.
        
        merged_count = 0
        removals = set()
        
        # Sort by length (desc) so we default to keeping the longer/more descriptive name?
        # Or keeping the one with more edges? Let's sort alphabetically for stability first.
        entity_nodes.sort() 
        
        for i in range(len(entity_nodes)):
            node_a = entity_nodes[i]
            if node_a in removals: continue
            
            vec_a = self.vector_cache.get(node_a)
            if vec_a is None: continue

            for j in range(i + 1, len(entity_nodes)):
                node_b = entity_nodes[j]
                if node_b in removals: continue

                vec_b = self.vector_cache.get(node_b)
                if vec_b is None: continue

                # 1. Cosine Similarity
                sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)
                
                if sim < 0.90: # Optimization: Skip low sim
                    continue
                
                # 2. Name Similarity (Levenshtein)
                name_sim = difflib.SequenceMatcher(None, node_a.lower(), node_b.lower()).ratio()
                
                should_merge = False
                reason = ""
                
                # CASE A: Typos / Minor Variations (e.g. "Raphael" vs "Rephael")
                if sim > 0.95 and name_sim > 0.75:
                    should_merge = True
                    reason = f"Typo/Var (Vec: {sim:.2f}, Name: {name_sim:.2f})"
                
                # CASE B: Synonyms / Translations (e.g. "The Boss" vs "Der Chef")
                # Requires CrossEncoder check
                elif sim > 0.90 and self._verify_pair_identity(node_a, node_b):
                    should_merge = True
                    reason = f"Semantic Synonym (Vec: {sim:.2f}, Verified)"
                
                if should_merge:
                    # MERGE B into A (Surviving node: A)
                    # Heuristic: Keep the one with clearer capitalization or more edges?
                    # For now: Keep A (as it's first in sorted list) unless B is longer?
                    # Let's keep the one with more edges.
                    deg_a = self.graph.degree(node_a)
                    deg_b = self.graph.degree(node_b)
                    
                    target, source = (node_a, node_b) if deg_a >= deg_b else (node_b, node_a)
                    
                    logger.info(f"[bold magenta][NEURON-X][/bold magenta] Merging '{source}' into '{target}' | Reason: {reason}")
                    
                    # Move Edges
                    # Incoming to Source -> Target
                    for u, _, data in list(self.graph.in_edges(source, data=True)):
                        if u == target: continue # Don't create self-loop
                        if not self.graph.has_edge(u, target):
                            self.graph.add_edge(u, target, **data)
                        else:
                            # Reinforce existing
                            self.graph[u][target]['weight'] += data.get('weight', 0.0)
                    
                    # Outgoing from Source -> Target
                    for _, v, data in list(self.graph.out_edges(source, data=True)):
                        if v == target: continue
                        if not self.graph.has_edge(target, v):
                            self.graph.add_edge(target, v, **data)
                        else:
                            self.graph[target][v]['weight'] += data.get('weight', 0.0)
                            
                    self.graph.remove_node(source)
                    removals.add(source)
                    merged_count += 1
                    
                    # If we merged into 'node_b', node_a is gone, break inner loop
                    if target == node_b:
                        removals.add(node_a)
                        break 

    def _dream_cycle(self):
        """
        The Dreaming Phase: Generates creative hypotheses by connecting unrelated concepts.
        """
        if not self.llm_client:
            return

        logger.info("[bold magenta][NEURON-X][/bold magenta] Entering REM Sleep (Dreaming)...")
        
        # 1. Pick two random, unconnected entities
        # FILTER: Exclude REJECTED nodes from sleeping constructs
        all_nodes = [n for n in self.graph.nodes() 
                     if not n.startswith("Memory_") 
                     and n != "Self"
                     and self.graph.nodes[n].get("status") != "REJECTED"]
        if len(all_nodes) < 5:
            return
            
        import random
        # Try 3 times to find a pair
        for _ in range(3):
            subj = random.choice(all_nodes)
            obj = random.choice(all_nodes)
            
            if subj == obj or self.graph.has_edge(subj, obj) or self.graph.has_edge(obj, subj):
                continue
                
            # Found a pair!
            logger.info(f"[bold magenta][NEURON-X][/bold magenta] Dreaming about connection between '{subj}' and '{obj}'...")
            
            prompt = f"""
            You are the Subconscious Creativity Engine of NEURON-X.
            
            Task: Invent a CREATIVE, PLAUSIBLE connection between these two concepts:
            1. {subj}
            2. {obj}
            
            Rules:
            - This is a "What if?" scenario.
            - Output specific relationship predicate (e.g., 'might_be_related_to', 'could_be_ancestor_of', 'symbolizes').
            - Output ONLY the prediction in JSON format: {{"predicate": "relationship", "reasoning": "short explanation"}}
            """
            
            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=1.3 # High temp for dreaming
                    )
                )

                result = json.loads(response.text)
                pred = result.get("predicate", "might_be_related_to")
                reason = result.get("reasoning", "Dream logic")
                
                # Add the Hypothesis
                self.graph.add_edge(subj, obj, relation=pred, weight=0.2, category="HYPOTHESIS", reasoning=reason)
                logger.info(f"[bold cyan][NEURON-X][/bold cyan] Created Hypothesis: ({subj}) --[{pred}]--> ({obj})")
                break # One dream per sleep cycle is enough
                
            except Exception as e:
                logger.warning(f"Dream interrupted: {e}")
                continue

    def _apply_entropy(self, reinforced_edges):
        """
        Applies a small decay to high-weight edges that were NOT reinforced this cycle.
        This prevents 'Gravity Wells' where a belief becomes irrefutable.
        
        Args:
            reinforced_edges: A set/list of edge dictionaries (or objects) that were reinforced.
                            We expect they might be the raw dictionaries from 'final_claims'.
        """
        if not self.graph.number_of_edges():
            return

        # Convert reinforced data to a set of (u, v) tuples for fast O(1) lookup.
        # Track both (u, v) and (u, v, key) if possible, but simplicity first.
        # In final_claims we have dicts with 'subject', 'object', 'predicate'.
        
        active_pairs = set()
        for item in reinforced_edges:
            # item is a dict from final_claims
            s = item.get('subject')
            o = item.get('object')
            if s and o:
                active_pairs.add((s, o))

        MAX_WEIGHT = 5.0
        DECAY_RATE = 0.05 # 5% of the distance to zero? Or fixed amount? 
        # Let's use a small fixed amount scaled by current certainty.
        
        updates = []
        
        for u, v, data in self.graph.edges(data=True):
            current_w = float(data.get('weight', 1.0))
            
            # Only decay "Established" beliefs to force them to prove their worth.
            # Low weight hypotheses (e.g. 0.2) shouldn't decay to zero instantly.
            # We target High Weight anchors > 3.0
            if current_w > 3.5:
                # Check if it was active
                if (u, v) in active_pairs:
                    continue
                    
                # Apply Entropy
                # The higher the weight, the more 'energy' it needs to maintain.
                # Decay = Base * (Weight / Max)
                decay = DECAY_RATE * (current_w / MAX_WEIGHT)
                new_w = current_w - decay
                
                # Cap minimum for these established nodes so they don't vanish overnight
                # unless rejected.
                if new_w < 3.0: new_w = 3.0 
                
                if new_w != current_w:
                    updates.append((u, v, new_w))

        if updates:
            logger.info(f"[bold magenta][ENTROPY][/bold magenta] Decaying {len(updates)} stagnant high-certainty beliefs.")
            for u, v, w in updates:
                self.graph[u][v]['weight'] = w

    def save_graph(self):
        """Saves current state and updates timestamp to prevent unnecessary reloading."""
        nx.write_gexf(self.graph, self.graph_file)
        self.last_sync_time = os.path.getmtime(self.graph_file)

    def get_identity_summary(self):
        """Queries the graph for core identity and scale of awareness."""
        nodes = len(self.graph.nodes())
        edges = len(self.graph.edges())
        
        # Pull core identity traits (neighbors of 'Self')
        core_traits = []
        if "Self" in self.graph:
            # We look at what the 'Self' node is connected to (Identity/Aims/etc)
            for neighbor in self.graph.neighbors("Self"):
                if neighbor.startswith("Memory_"): # Skip individual memory logs for summary
                    continue
                edge_data = self.graph.get_edge_data("Self", neighbor)
                rel = edge_data.get("relation", "is_related_to")
                cat = edge_data.get("category", "FACTUAL")
                core_traits.append(f"({rel}: {neighbor})")
        
        import random
        random.shuffle(core_traits)
        traits_str = ", ".join(core_traits[:15])
        return f"Awareness Scale: {nodes} nodes, {edges} edges. Core Identity: {traits_str if traits_str else 'Initializing...'}"