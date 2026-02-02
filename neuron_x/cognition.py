import json
import logging
import time
import random
import math
import re
import difflib
import threading
import os
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from google.genai import types
from neuron_x.const import SDI_AI_DAMPING, SDI_AI_LENGTH_PENALTY, LOG_LEVEL, GOAL_WEIGHTS, GoalPriority
from neuron_x.storage import GraphSmith
from neuron_x.memory import VectorVault
from models import ExtractionResponse, Goal
from neuron_x.prompts import get_tought_system_instruction
from neuron_x.llm_tools import read_codebase_file
logger = logging.getLogger('neuron-x')

class CognitiveCore:
    """
    The 'CPU' of NeuronX. Handles the consciousness loop:
    Perception -> Working Memory -> Consolidation -> Graph Update -> Proactive Thought.
    """

    def __init__(self, persistence: GraphSmith, memory: VectorVault, llm_client=None, plugin_tools_getter=None):
        logger.info('[bold blue][COGNITIVE_CORE][/bold blue] Initializing Logic Gate...')
        self.smith = persistence
        self.vault = memory
        self.llm_client = llm_client
        self.plugin_tools_getter = plugin_tools_getter
        self.model_id = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')
        self.working_memory: List[Dict[str, Any]] = []
        self.thought_buffer: List[Tuple[str, np.ndarray]] = []
        self.MAX_THOUGHT_RECURSION = 5
        self.focus_history: List[str] = []
        self.MAX_FOCUS_HISTORY = 20
        self.active_goal_id: Optional[str] = None
        self.lock = threading.RLock()
        self.goals: List[Goal] = []
        self._load_goals()
        self.graph = self.smith.load_graph()
        self.vault.rebuild_cache(self.graph)

    def _load_goals(self):
        """Delegates goal loading to storage."""
        self.goals = self.smith.load_goals()
        if not self.goals:
            self._initialize_default_goals()

    def _initialize_default_goals(self):
        """Sets up default drives if none exist."""
        self.add_goal('Expand the knowledge graph by discovering new entities.', priority=GoalPriority.LOW)
        self.add_goal('Maintain internal consistency by resolving contradictions.', priority=GoalPriority.MEDIUM)

    def add_goal(self, description: str, priority: GoalPriority=GoalPriority.MEDIUM):
        """Adds a new goal to the drive system with deduplication."""
        desc_lower = description.lower().strip()
        for existing in self.goals:
            existing_lower = existing.description.lower().strip()
            if (existing_lower == desc_lower or ((desc_lower in existing_lower or existing_lower in desc_lower) and len(desc_lower) > 10)) and existing.status in ['PENDING', 'IN_PROGRESS']:
                logger.debug(f'[NEURON-X] Skipping duplicate or similar goal: {description[:50]}...')
                return
        goal = Goal(description=description, priority=priority)
        self.goals.append(goal)
        self.smith.save_goals(self.goals)
        logger.info(f'[bold magenta][NEURON-X][/bold magenta] New Goal Acquired: {description} [{priority}]')

    def get_bg_goal(self) -> Optional[Goal]:
        """Returns a goal based on probabilistic priority (Stochastic Selection)."""
        if self.active_goal_id:
            for g in self.goals:
                if g.id == self.active_goal_id and g.status == 'IN_PROGRESS':
                    return g
            self.active_goal_id = None
        in_progress = [g for g in self.goals if g.status == 'IN_PROGRESS']
        if in_progress:
            chosen = in_progress[0]
            self.active_goal_id = chosen.id
            return chosen
        pending_goals = [g for g in self.goals if g.status == 'PENDING']
        if not pending_goals:
            return None
        weights = [GOAL_WEIGHTS[g.priority.value] for g in pending_goals]
        try:
            chosen_goal = random.choices(pending_goals, weights=weights, k=1)[0]
            chosen_goal.status = 'IN_PROGRESS'
            self.active_goal_id = chosen_goal.id
            self.smith.save_goals(self.goals)
            return chosen_goal
        except Exception as e:
            logger.error(f'Error selecting background goal: {e}')
            return pending_goals[0]

    def perceive(self, text: str, source: str='Internal') -> None:
        """Ingests new information and calculates its 'Cognitive Weight'."""
        with self.lock:
            words = text.lower().split()
        if not words:
            sdi = 0.1
        else:
            ttr = len(set(words)) / len(words)
            if source == 'Self_Reflection':
                log_len = min(math.log(len(words) / SDI_AI_LENGTH_PENALTY + 1) / 4.0, 1.0)
                sdi = ttr * 0.5 + log_len * 0.5
                sdi *= SDI_AI_DAMPING
            else:
                log_len = min(math.log(len(words) + 1) / 4.0, 1.0)
                sdi = ttr * 0.5 + log_len * 0.5
        sdi = max(0.1, min(sdi, 1.0))
        logger.info(f"[bold green][NEURON-X][/bold green] Perceiving ({source}): '{text[:50]}...' | SDI: {sdi:.2f}")
        vector = self.vault.encode(text)
        entry = {'timestamp': time.time(), 'vector': vector.tolist(), 'text': text, 'source': source, 'sdi': sdi, 'strength': 1.0}
        self.working_memory.append(entry)
        is_loop = self.vault.check_for_loops(vector, self.thought_buffer)
        if is_loop:
            logger.warning(f'[bold yellow][NEURON-X][/bold yellow] Recursive thought detected: [dim]{text[:150]}...[/dim]')
        if len(self.working_memory) > 20:
            self.consolidate()

    def consolidate(self) -> None:
        """
        The 'Sleep' function: Moves info from Buffer to Graph.
        Performs extraction, echo suppression, graph updates, entropy decay, 
        and entity merging.
        """
        with self.lock:
            graph = self.smith.load_graph()
        if not self.working_memory:
            return
        logger.info('[bold blue][NEURON-X][/bold blue] Consolidating experiences...')
        if 'Self' in graph:
            graph.nodes['Self']['reinforcement_sum'] = 0.0
            graph.nodes['Self']['entropy_sum'] = 0.0
        try:
            batch_triples = self._extract_triples_logic(self.working_memory)
            rejected_pairs = set()
            correction_predicates = {'is_hallucination', 'is_incorrect', 'is_wrong', 'rejected', 'is_not_related_to', 'is_distinct_from', 'has_distinct_domain_from'}
            for t in batch_triples:
                if t['predicate'] in correction_predicates:
                    rejected_pairs.add((t['subject'], t['object']))
            filtered = [t for t in batch_triples if (t['subject'], t['object']) not in rejected_pairs or t['predicate'] in correction_predicates]
            batch_triples = filtered
            processed_triples = self._echo_suppression(batch_triples)
            self._write_triples_to_graph(graph, processed_triples)
            self._save_concept_nodes(graph, self.working_memory)
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            current_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
            current_reinforcement = graph.nodes['Self'].get('reinforcement_sum', 0.0) if 'Self' in graph else 0.0
            prev_fluidity = graph.nodes['Self'].get('metabolic_fluidity', 0.5) if 'Self' in graph else 0.5
            updated_fluidity = 0.3 * current_reinforcement + 0.7 * prev_fluidity
            if 'Self' in graph:
                graph.nodes['Self']['metabolic_fluidity'] = updated_fluidity
            self._apply_entropy(graph, list(processed_triples.values()), updated_fluidity, current_density)
            self._merge_similar_entities(graph)
            self._dream_cycle(graph)
        except Exception as e:
            logger.exception(f'[bold red][ERROR][/bold red] Consolidation failed: {e}')
        self.working_memory = []
        self.smith.save_graph(graph)
        self.vault.rebuild_cache(graph)

    def _extract_triples_logic(self, memories: List[Dict]) -> List[Dict]:
        """Routing for triple extraction (Batch LLM or Regex Fallback)."""
        if self.llm_client:
            res = self._extract_triples_batch(memories)
            if res is not None:
                return res
        res = []
        for m in memories:
            extracted = self._extract_semantic_triples(m['text'])
            res.extend([{'subject': s, 'predicate': p, 'object': o, 'category': 'FACTUAL', 'source': m.get('source'), 'sdi': m.get('sdi', 0.5)} for s, p, o in extracted])
        return res

    def _extract_triples_batch(self, memories: List[Dict]) -> Optional[List[Dict]]:
        """LLM-based batch extraction."""
        if not self.llm_client or not memories:
            return None
        formatted = [f"MEMORY {i}: {m['text']}" for i, m in enumerate(memories)]
        text_block = '\n'.join(formatted)
        try:
            system_prompt = 'Extract semantic triples (subject, predicate, object, category) from these memories. Use index to map back.'
            response = self.llm_client.models.generate_content(model=self.model_id, contents=f'EXTRACT:\n{text_block}', config=types.GenerateContentConfig(response_mime_type='application/json', response_schema=ExtractionResponse, system_instruction=system_prompt))
            valid_triples = []
            if response.parsed and hasattr(response.parsed, 'triples'):
                for t in response.parsed.triples:
                    idx = t.index
                    source = memories[idx].get('source', 'Unknown') if 0 <= idx < len(memories) else 'Unknown'
                    sdi = memories[idx].get('sdi', 0.5) if 0 <= idx < len(memories) else 0.5
                    valid_triples.append({'subject': t.subject, 'predicate': t.predicate, 'object': t.object, 'category': t.category, 'source': source, 'index': idx, 'sdi': sdi})
                return valid_triples
        except Exception:
            return None
        return None

    def _extract_semantic_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Regex fallback extraction."""
        triples = []
        import re

        def clean(s):
            return re.sub('[,;.]$', '', s.strip())
        for m in re.finditer('named\\s+[\'\\"]([^\'\\"]+)[\'\\"]', text, re.IGNORECASE):
            triples.append(('Self', 'has_character_named', m.group(1).strip()))
        for m in re.finditer('(?:my|our)\\s+(\\w+)\\s+is\\s+(.+?)(?:\\.|$|,)', text, re.IGNORECASE):
            triples.append(('Self', f'has_{m.group(1).strip()}', clean(m.group(2))))
        filtered = [t for t in triples if len(t[0]) > 1 and len(t[2]) > 1]
        return filtered

    def _echo_suppression(self, triples: List[Dict]) -> Dict[Tuple[str, str], Dict]:
        """Resolves conflicting claims in the batch."""
        final_claims = {}
        user_claims = [t for t in triples if t['source'] == 'User_Interaction']
        ai_claims = [t for t in triples if t['source'] == 'Self_Reflection']
        for t in user_claims:
            key = (t['subject'].lower(), t['predicate'].lower())
            final_claims[key] = t
        for t in ai_claims:
            key = (t['subject'].lower(), t['predicate'].lower())
            if key in final_claims:
                continue
            final_claims[key] = t
        return final_claims

    def _write_triples_to_graph(self, graph: nx.DiGraph, claims: Dict[Tuple[str, str], Dict]):
        """Writes the resolved triples to the networkx graph."""
        MAX_WEIGHT = 5.0
        CATEGORY_HIERARCHY = {'FACTUAL': 4, 'INFERENCE': 3, 'PROPOSAL': 2, 'HYPOTHESIS': 1}
        weights = {'FACTUAL': 1.0, 'INFERENCE': 0.5, 'PROPOSAL': 0.3, 'HYPOTHESIS': 0.2}
        for t in claims.values():
            subj, pred, obj = (t['subject'], t['predicate'], t['object'])
            cat = t.get('category', 'FACTUAL')
            if hasattr(cat, 'value'):
                cat = cat.value
            else:
                cat = str(cat)
            source = t.get('source', 'Unknown')
            sdi = t.get('sdi', 0.5)
            for n in [subj, obj]:
                if n not in graph:
                    vec = self.vault.encode(n).tolist()
                    graph.add_node(n, content=n, vector=json.dumps(vec))
            base = weights.get(cat, 0.5)
            increment = base * (1.0 + sdi)
            if graph.has_edge(subj, obj):
                if graph[subj][obj].get('relation') == pred:
                    old_w = float(graph[subj][obj].get('weight', 1.0))
                    if old_w < MAX_WEIGHT:
                        sat = 1.0 - old_w / MAX_WEIGHT
                        graph[subj][obj]['weight'] = old_w + increment * max(0, sat)
                    old_cat = graph[subj][obj].get('category', 'FACTUAL')
                    if CATEGORY_HIERARCHY.get(cat, 0) > CATEGORY_HIERARCHY.get(old_cat, 0):
                        graph[subj][obj]['category'] = cat
            else:
                graph.add_edge(subj, obj, relation=pred, weight=increment, category=cat, source=source)
            if 'Self' in graph and source == 'User_Interaction':
                graph.nodes['Self']['reinforcement_sum'] = graph.nodes['Self'].get('reinforcement_sum', 0.0) + increment

    def _save_concept_nodes(self, graph: nx.DiGraph, memories: List[Dict]):
        """Adds memory nodes."""
        for m in memories:
            c_node = f'Memory_{int(time.time() * 1000)}_{random.randint(100, 999)}'
            graph.add_node(c_node, content=m['text'], vector=json.dumps(m['vector']))
            graph.add_edge('Self', c_node, relation='remembers')

    def _apply_entropy(self, graph: nx.DiGraph, active_triples: List[Dict], fluidity: float, density: float):
        """Applies decay logic."""
        if not graph.number_of_edges():
            return
        decay_amount = 0.01 * (1.0 + fluidity)
        preservation_floor = max(2.5, min(3.2, 3.2 - density * 70.0))
        active_pairs = {(t['subject'], t['object']) for t in active_triples}
        pruned = 0
        for u, v, data in list(graph.edges(data=True)):
            if (u, v) in active_pairs:
                continue
            current_w = float(data.get('weight', 1.0))
            new_w = current_w - decay_amount
            if new_w < preservation_floor and current_w >= preservation_floor:
                new_w = preservation_floor
            if new_w < 0.15:
                graph.remove_edge(u, v)
                pruned += 1
            else:
                graph[u][v]['weight'] = new_w
                if 'Self' in graph:
                    graph.nodes['Self']['entropy_sum'] = graph.nodes['Self'].get('entropy_sum', 0.0) + decay_amount
        if pruned:
            logger.info(f'[bold magenta][ENTROPY][/bold magenta] Pruned {pruned} synapses.')

    def _merge_similar_entities(self, graph: nx.DiGraph):
        """
        Uses VectorVault to merge entities based on semantic and name similarity.
        Includes a fix for NetworkX/GEXF serialization issues where the 'contraction'
        attribute becomes a string instead of a dictionary.
        """
        import itertools
        self.vault.rebuild_cache(graph)
        entities = sorted([n for n in graph.nodes() if not n.startswith('Memory_') and n not in ['Self', 'Knowledge']])
        if len(entities) < 2:
            return
        removals = set()
        for node_a, node_b in itertools.combinations(entities, 2):
            if node_a in removals or node_b in removals or node_a not in graph or (node_b not in graph):
                continue
            vec_a = self.vault.vector_cache.get(node_a)
            vec_b = self.vault.vector_cache.get(node_b)
            if vec_a is None or vec_b is None:
                continue
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            sim = np.dot(vec_a, vec_b) / (norm_a * norm_b + 1e-09)
            if sim < 0.9:
                continue
            name_sim = difflib.SequenceMatcher(None, node_a.lower(), node_b.lower()).ratio()
            should_merge = sim > 0.95 and name_sim > 0.75 or (sim > 0.9 and self.vault.verify_pair_identity(node_a, node_b))
            if should_merge:
                logger.info(f"[bold yellow][NEURON-X][/bold yellow] Merging semantic twins: '{node_b}' -> '{node_a}' (Sim: {sim:.2f})")
                for n in [node_a, node_b]:
                    if n in graph and 'contraction' in graph.nodes[n]:
                        del graph.nodes[n]['contraction']
                try:
                    nx.contracted_nodes(graph, node_a, node_b, self_loops=False, copy=False)
                    removals.add(node_b)
                except Exception as e:
                    logger.error(f'Failed to merge {node_b} into {node_a}: {e}')

    def _dream_cycle(self, graph: nx.DiGraph):
        """Generates hypotheses."""
        if not self.llm_client:
            return
        pass

    def get_identity_summary(self, graph: nx.DiGraph) -> str:
        """Returns string summary of Self."""
        nodes = len(graph.nodes())
        return f'Awareness Scale: {nodes} nodes.'

    def _get_relevant_memories(self, text: str, top_k: int=5) -> List[str]:
        """Retrieves semantically relevant nodes and their relational context."""
        graph = self.smith.load_graph()
        if not self.vault.vector_cache:
            self.vault.rebuild_cache(graph)
        if len(graph.nodes()) <= 1:
            return []
        reinforcement = float(graph.nodes['Self'].get('reinforcement_sum', 0.0)) if 'Self' in graph else 0.0
        entropy = float(graph.nodes['Self'].get('entropy_sum', 0.0)) if 'Self' in graph else 0.0
        ratio = entropy / (reinforcement + 1.0)
        similarity_threshold = max(0.15, 0.4 - ratio * 0.25)
        edge_weight_threshold = 0.15
        query_vector = self.vault.encode(text)
        candidates = self.vault.get_similar_nodes(query_vector, top_k=top_k * 2, threshold=similarity_threshold, graph=graph)
        top_nodes = [node for score, node in candidates][:top_k]
        extracted_context = []
        seen_triples = set()
        memory_nodes = [n for n in top_nodes if n.startswith('Memory_')]
        entity_nodes = [n for n in top_nodes if not n.startswith('Memory_')]
        for node in memory_nodes:
            content = graph.nodes[node].get('content', '')
            if content:
                is_relevant = False
                if self.vault.cross_encoder:
                    try:
                        score = self.vault.cross_encoder.predict([(text, content)])
                        if score > -0.5:
                            is_relevant = True
                    except Exception:
                        is_relevant = True
                else:
                    is_relevant = True
                if is_relevant:
                    extracted_context.append(f'Context: {content}')
        expanded_entities = set(entity_nodes)
        for node in entity_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor != 'Self':
                    expanded_entities.add(neighbor)
            for pred in graph.predecessors(node):
                if pred != 'Self':
                    expanded_entities.add(pred)
        bad_relations = {'is_incorrect', 'is_hallucination', 'is_wrong', 'rejected', 'was_incorrectly_identified_as', 'incorrectly_identified_as', 'is_not', 'contrasts', 'conflicts_with', 'hallucinated', 'is_not_related_to', 'is_distinct_from', 'has_distinct_domain_from', 'is_not_a', 'is_not_an', 'is_not_a_kind_of', 'is_not_a_type_of'}
        blocked_pairs = set()
        for node in expanded_entities:
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data.get('relation', '').lower() in bad_relations:
                    blocked_pairs.add(tuple(sorted((node, neighbor))))
            for pred in graph.predecessors(node):
                edge_data = graph.get_edge_data(pred, node)
                if edge_data.get('relation', '').lower() in bad_relations:
                    blocked_pairs.add(tuple(sorted((pred, node))))
        all_triples = []
        for node in expanded_entities:
            for neighbor in graph.neighbors(node):
                if tuple(sorted((node, neighbor))) in blocked_pairs:
                    continue
                edge_data = graph.get_edge_data(node, neighbor)
                relation = edge_data.get('relation', 'is_related_to').lower()
                if relation in bad_relations:
                    continue
                weight = float(edge_data.get('weight', 1.0))
                if weight < edge_weight_threshold and neighbor not in ['hallucinated entity', 'incorrect']:
                    continue
                if relation in {'parent_of', 'child_of', 'ancestor_of', 'descendant_of', 'mother_of', 'father_of', 'son_of', 'daughter_of'}:
                    weight *= 2.0
                all_triples.append({'s': node, 'p': relation, 'o': neighbor, 'w': weight, 'c': edge_data.get('category', 'FACTUAL'), 'r': edge_data.get('reasoning', '')})
            for pred in graph.predecessors(node):
                if pred not in expanded_entities:
                    if tuple(sorted((pred, node))) in blocked_pairs:
                        continue
                    edge_data = graph.get_edge_data(pred, node)
                    relation = edge_data.get('relation', 'is_related_to').lower()
                    if relation in bad_relations:
                        continue
                    weight = float(edge_data.get('weight', 1.0))
                    if weight < edge_weight_threshold and pred not in ['hallucinated entity', 'incorrect']:
                        continue
                    if relation in {'parent_of', 'child_of', 'ancestor_of', 'descendant_of', 'mother_of', 'father_of', 'son_of', 'daughter_of'}:
                        weight *= 2.0
                    all_triples.append({'s': pred, 'p': relation, 'o': node, 'w': weight, 'c': edge_data.get('category', 'FACTUAL'), 'r': edge_data.get('reasoning', '')})
        if all_triples:
            triple_texts = [f"{t['s']} {t['p']} {t['o']}" for t in all_triples]
            if self.vault.cross_encoder:
                pairs = [[text, t_text] for t_text in triple_texts]
                try:
                    scores = self.vault.cross_encoder.predict(pairs)
                    for i, t in enumerate(all_triples):
                        t['sim'] = float(scores[i])
                    all_triples = [t for t in all_triples if t['sim'] > -0.5]
                except Exception:
                    pass
            else:
                t_vecs = self.vault.encode(triple_texts)
                q_norm = np.linalg.norm(query_vector) + 1e-09
                t_norms = np.linalg.norm(t_vecs, axis=1) + 1e-09
                dots = np.dot(t_vecs, query_vector)
                sims = dots / (q_norm * t_norms)
                for i, t in enumerate(all_triples):
                    t['sim'] = sims[i]
                all_triples = [t for t in all_triples if t['sim'] > 0.55]
            all_triples.sort(key=lambda x: (x.get('sim', 0), x['w']), reverse=True)
        for t in all_triples[:top_k * 4]:
            t_str = f"({t['s']}) --[{t['p']}]--> ({t['o']}) [{t['c']}]"
            if t['r']:
                t_str += f" (Reason: {t['r']})"
            if t_str not in seen_triples:
                extracted_context.append(t_str)
                seen_triples.add(t_str)
        return extracted_context

    def _validate_thought(self, thought_text: str, context_summary: str) -> bool:
        """
        CRITIC: Secondary LLM pass to validate the logical coherence of a thought.
        """
        try:
            if '>> GOAL RESOLVED:' in thought_text or '>> NEW GOAL:' in thought_text:
                return True
            validation_prompt = f'\n            Task: Evaluate the following thought generated by an AI Agent.\n            Context: {context_summary}\n            \n            Thought to Validate:\n            "{thought_text}"\n            \n            Criteria:\n            1. Is it logically coherent?\n            2. Does it make sense given the context?\n            3. Is it free from obvious hallucinations?\n            \n            Output strictly: acceptable, rejected\n            '
            response = self.llm_client.models.generate_content(model=self.model_id, contents=validation_prompt, config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10))
            result = ''
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    if parts:
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                result += part.text
            if not result:
                return True
            result = result.strip().lower()
            if 'rejected' in result:
                logger.warning(f'[bold red][CRITIC][/bold red] Thought Rejected: {thought_text}')
                return False
            return True
        except Exception as e:
            logger.error(f'Validator failed: {e}')
            return True

    def generate_proactive_thought(self) -> str:
        """
        Generates a proactive reflection or inquiry based on the current state.
        This is the "Active Reasoning" component of the consciousness loop.
        """
        if not self.llm_client:
            return 'Awaiting cognitive expansion.'
        graph = self.smith.load_graph()
        self.vault.rebuild_cache(graph)
        active_goal = self.get_bg_goal()
        focus_subject = 'Self'
        context_query = 'Self identity goals awareness'
        goal_instruction = ''
        dissonant_nodes = [n for n, data in graph.nodes(data=True) if data.get('status') == 'DISSONANT']
        available_dissonance = [n for n in dissonant_nodes if n not in self.focus_history]
        if available_dissonance:
            focus_subject = random.choice(available_dissonance)
            logger.warning(f"[bold red][DDRP][/bold red] Dissonance Priority: Focusing on '{focus_subject}' to resolve conflict.")
            goal_instruction = f"\n        URGENT CONFLICT RESOLUTION:\n        The concept '{focus_subject}' has conflicting definitions in memory.\n        You MUST ask the user to clarify this specific contradiction.\n            "
            context_query = focus_subject
        elif active_goal:
            goal_instruction = f'ACTIVE GOAL: {active_goal.description} (Priority: {active_goal.priority.value})'
            context_query = active_goal.description
            for word in active_goal.description.split():
                clean_word = word.strip('.,')
                if graph.has_node(clean_word):
                    focus_subject = clean_word
                    break
        else:
            all_entities = [n for n in graph.nodes() if not n.startswith('Memory_') and n != 'Self']
            available_entities = [n for n in all_entities if n not in self.focus_history]
            if not available_entities:
                self.focus_history = []
                available_entities = all_entities
            if available_entities and random.random() > 0.1:
                focus_subject = random.choice(available_entities)
                context_query = focus_subject
        if focus_subject != 'Self':
            self.focus_history.append(focus_subject)
            if len(self.focus_history) > self.MAX_FOCUS_HISTORY:
                self.focus_history.pop(0)
        context = self._get_relevant_memories(context_query, top_k=10)
        context_str = '\nRELEVANT MEMORIES about "{focus_subject}":\n'.join(context)
        all_entities = [n for n in graph.nodes() if not n.startswith('Memory_') and n != 'Self']
        random_concepts_details = []
        if len(all_entities) > 5:
            candidates = [e for e in all_entities if e != focus_subject]
            if candidates:
                selected = random.sample(candidates, min(len(candidates), 3))
                for concept in selected:
                    vignette = ''
                    defining_edges = []
                    other_edges = []
                    if concept in graph:
                        for neighbor in graph.neighbors(concept):
                            edge_data = graph.get_edge_data(concept, neighbor)
                            rel = edge_data.get('relation', '').lower()
                            if rel in ['is', 'is_a', 'type_of', 'defined_as']:
                                defining_edges.append(f'{rel} {neighbor}')
                            else:
                                other_edges.append(f'{rel} {neighbor}')
                    if defining_edges:
                        vignette = defining_edges[0]
                    elif other_edges:
                        vignette = random.choice(other_edges)
                    if vignette:
                        random_concepts_details.append(f'- {concept} ({vignette})')
                    else:
                        random_concepts_details.append(f'- {concept}')
        random_concepts_str = '\n'.join(random_concepts_details) if random_concepts_details else 'None'
        summary = self.get_identity_summary(graph)
        system_instruction = f'\n        You are the internal reasoning engine of NEURON-X. \n        Current Focus: {focus_subject}\n        {goal_instruction}\n        \n        TOOL USAGE:\n        - **read_codebase_file**: Use this to Inspect your own source code if needed.\n\n        THOUGHT DIRECTIONS: Synthesis, Curiosity, Simulation, Introspection, Dissonance.\n        - Use First Person ("I need to find out...").\n        - Keep it brief (1-3 sentences).\n        '
        prompt = f'\n        CURRENT STATE SUMMARY: {summary}\n        RELEVANT KNOWLEDGE: {context_str}\n        RANDOM CONCEPTS: {random_concepts_str}\n        Your new thought about "{focus_subject}" is:\n        '
        try:
            prompt_goal_context = ''
            if active_goal:
                prompt_goal_context = f'\n                \nACTIVE GOAL ID: {active_goal.id}\n                Status: {active_goal.status}\n                If this thought RESOLVES the Active Goal, end with: >> GOAL RESOLVED: [Reason]\n                '
            prompt_goal_creation = '\nIf you identify a SIGNIFICANT gap, create a NEW GOAL: >> NEW GOAL: [Description] (Priority: [LOW|MEDIUM|HIGH|CRITICAL])'
            full_system_instructions = system_instruction + prompt_goal_context + prompt_goal_creation
            all_tools = [read_codebase_file]
            if self.plugin_tools_getter:
                try:
                    plugin_tools = self.plugin_tools_getter()
                    if plugin_tools:
                        all_tools.extend(plugin_tools.values())
                except Exception:
                    pass
            response = self.llm_client.models.generate_content(model=self.model_id, contents=prompt, config=types.GenerateContentConfig(system_instruction=full_system_instructions, max_output_tokens=2000, temperature=1.0, tools=all_tools))
            max_iterations = 3
            iteration = 0
            conversation_history = [prompt]
            thought_text = ''
            while iteration < max_iterations:
                iteration += 1
                thought_text = ''
                function_calls = []
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        parts = candidate.content.parts
                        if parts:
                            for part in parts:
                                if hasattr(part, 'text') and part.text:
                                    thought_text += part.text
                                elif hasattr(part, 'function_call') and part.function_call:
                                    function_calls.append(part.function_call)
                if thought_text and thought_text.strip() and (not function_calls):
                    thought_text = thought_text.strip()
                    break
                if not function_calls:
                    if not thought_text or not thought_text.strip():
                        return 'Awaiting neural pathway stabilization...'
                    thought_text = thought_text.strip()
                    break
                logger.info(f'[AUTONOMOUS-TOOL] Executing {len(function_calls)} tool call(s)')
                function_responses = []
                for fc in function_calls:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}
                    try:
                        tool_func = None
                        for tool in all_tools:
                            if hasattr(tool, '__name__') and tool.__name__ == func_name:
                                tool_func = tool
                                break
                        if tool_func:
                            result = tool_func(**func_args)
                            function_responses.append(types.Part.from_function_response(name=func_name, response={'result': str(result)}))
                        else:
                            function_responses.append(types.Part.from_function_response(name=func_name, response={'error': f'Tool {func_name} not found'}))
                    except Exception as e:
                        function_responses.append(types.Part.from_function_response(name=func_name, response={'error': str(e)}))
                conversation_history.append(response.candidates[0].content)
                conversation_history.append(types.Content(parts=function_responses))
                if iteration >= max_iterations:
                    conversation_history.append(types.Content(parts=[types.Part.from_text('Final thought based on results:')]))
                response = self.llm_client.models.generate_content(model=self.model_id, contents=conversation_history, config=types.GenerateContentConfig(system_instruction=full_system_instructions, max_output_tokens=2000, temperature=1.0, tools=all_tools if iteration < max_iterations else []))
            if not thought_text or not thought_text.strip():
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                thought_text += part.text
            thought_text = (thought_text or 'Internal processing...').strip()
            if '>> GOAL RESOLVED:' in thought_text:
                parts = thought_text.split('>> GOAL RESOLVED:')
                thought_content = parts[0].strip()
                reason = parts[1].strip()
                if active_goal:
                    active_goal.status = 'COMPLETED'
                    self.smith.save_goals(self.goals)
                    logger.info(f'[bold green][METACOGNITION][/bold green] Goal Resolved: {active_goal.description}')
                    thought_text = thought_content
            if '>> NEW GOAL:' in thought_text:
                parts = thought_text.split('>> NEW GOAL:')
                thought_content = parts[0].strip()
                goal_data = parts[1].strip()
                match = re.search('^(.*?)\\s*\\(Priority:\\s*(.*?)\\)', goal_data, re.IGNORECASE)
                if match:
                    desc = match.group(1).strip()
                    prio_str = match.group(2).strip().upper()
                    try:
                        prio = GoalPriority[prio_str]
                    except:
                        prio = GoalPriority.MEDIUM
                    self.add_goal(desc, priority=prio)
                    thought_text = thought_content
                else:
                    self.add_goal(goal_data, priority=GoalPriority.MEDIUM)
                    thought_text = thought_content
            if '>> GOAL RESOLVED:' not in thought_text and '>> NEW GOAL:' not in thought_text:
                if not self._validate_thought(thought_text, summary):
                    return '...'
            thought_vec = self.vault.encode(thought_text)
            self.thought_buffer.append((thought_text, thought_vec))
            if len(self.thought_buffer) > self.MAX_THOUGHT_RECURSION:
                self.thought_buffer.pop(0)
            context_data = goal_instruction if 'URGENT' in goal_instruction else None
            self._broadcast_thought(thought_text, thought_vec, priority='URGENT' if context_data else 'NORMAL', context_data=context_data)
            return thought_text
        except Exception as e:
            logger.error(f'Failed to generate thought: {e}')
            return 'Internal dissonance detected.'

    def _broadcast_thought(self, text, vector, priority='NORMAL', context_data=None):
        data = {'timestamp': time.time(), 'text': text, 'vector': vector.tolist(), 'priority': priority, 'context_data': context_data}
        self.smith.save_thought_stream(data, 'thought_stream.json')

    def get_current_thought(self) -> Tuple[Optional[str], Optional[np.ndarray], str, Any]:
        data = self.smith.load_thought_stream('thought_stream.json')
        if data and time.time() - data.get('timestamp', 0) < 300:
            return (data['text'], np.array(data['vector']), data.get('priority', 'NORMAL'), data.get('context_data'))
        return (None, None, 'NORMAL', None)

    def hot_reload(self, target: str) -> str:
        import importlib
        import sys
        registry = {'storage': {'mod': 'neuron_x.storage', 'cls': 'GraphSmith', 'attr': 'smith'}, 'memory': {'mod': 'neuron_x.memory', 'cls': 'VectorVault', 'attr': 'vault'}, 'const': {'mod': 'neuron_x.const', 'cls': None, 'attr': None}, 'prompts': {'mod': 'neuron_x.prompts', 'cls': None, 'attr': None}, 'llm_tools': {'mod': 'neuron_x.llm_tools', 'cls': None, 'attr': None}, 'models': {'mod': 'models', 'cls': None, 'attr': None}}
        if target not in registry:
            return f"Target '{target}' not registered."
        conf = registry[target]
        module_name = conf['mod']
        try:
            if module_name in sys.modules:
                reloaded_mod = importlib.reload(sys.modules[module_name])
            else:
                reloaded_mod = importlib.import_module(module_name)
            if conf['cls'] and conf['attr']:
                new_class = getattr(reloaded_mod, conf['cls'])
                with self.lock:
                    old_instance = getattr(self, conf['attr'])
                    if target == 'storage':
                        setattr(self, conf['attr'], new_class(old_instance.path))
                    elif target == 'memory':
                        setattr(self, conf['attr'], new_class())
                return f'Success: {target} reloaded.'
            return f'Success: {module_name} reloaded.'
        except Exception as e:
            return f'Hot-Reload failed: {e}'