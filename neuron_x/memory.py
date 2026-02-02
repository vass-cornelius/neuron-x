import json
import logging
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from neuron_x.const import RECURSIVE_THOUGHT_THRESHOLD
logger = logging.getLogger('neuron-x')

class VectorVault:
    """
    Service class responsible for Vector Embeddings, Similarity Search, 
    and Semantic Verification.
    """

    def __init__(self):
        logger.info('[bold blue][VECTOR_VAULT][/bold blue] Loading Neural Models...')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        except Exception as e:
            logger.warning(f'[bold yellow][VECTOR_VAULT][/bold yellow] Failed to load CrossEncoder: {e}. Semantic merging will be limited.')
            self.cross_encoder = None
        self.vector_cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """Encodes text into a vector."""
        return self.encoder.encode(text)

    def rebuild_cache(self, graph: nx.DiGraph) -> None:
        """Parscache all vectors from the graph into a fast-access dictionary."""
        self.vector_cache = {}
        for node, data in graph.nodes(data=True):
            if 'vector' in data:
                vec_data = data['vector']
                if isinstance(vec_data, str):
                    try:
                        self.vector_cache[node] = np.array(json.loads(vec_data))
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    self.vector_cache[node] = np.array(vec_data)

    def check_for_loops(self, vector: np.ndarray, thought_buffer: List[Tuple[str, np.ndarray]]) -> bool:
        """Detection of recursive thoughts or dissonance."""
        if not thought_buffer:
            return False
        for t_text, t_vec in thought_buffer:
            q_norm = np.linalg.norm(vector) + 1e-09
            v_norm = np.linalg.norm(t_vec) + 1e-09
            similarity = np.dot(vector, t_vec) / (q_norm * v_norm)
            if similarity > RECURSIVE_THOUGHT_THRESHOLD:
                return True
        return False

    def verify_pair_identity(self, name_a: str, name_b: str) -> bool:
        """
        Uses Cross-Encoder to verify if two names refer to the exact same entity.
        """
        if not self.cross_encoder:
            return False
        try:
            score = self.cross_encoder.predict([(name_a, name_b)])
            return score > 0.8
        except Exception as e:
            logger.warning(f'CrossEncoder check failed: {e}')
            return False

    def get_similar_nodes(self, query_vector: np.ndarray, top_k: int=5, threshold: float=0.0, graph: Optional[nx.DiGraph]=None, query_text: Optional[str]=None) -> List[Tuple[float, str]]:
        """
        Finds top_k similar nodes from the cache using hybrid search (Vector + Keyword).
        """
        if not self.vector_cache:
            return []
        
        node_names = []
        vectors = []
        query_keywords = set()
        
        if query_text:
            clean_text = query_text.lower().replace('?', '').replace('!', '').replace('.', '').replace(',', '')
            # Extract keywords longer than 2 chars
            query_keywords = {w for w in clean_text.split() if len(w) > 2}
            
        for node, vec in self.vector_cache.items():
            if node == 'Self':
                continue
            if graph and graph.has_node(node) and (graph.nodes[node].get('status') == 'REJECTED'):
                continue
            node_names.append(node)
            vectors.append(vec)
            
        if not vectors:
            return []
            
        vectors_np = np.array(vectors)
        q_norm = np.linalg.norm(query_vector) + 1e-09
        v_norms = np.linalg.norm(vectors_np, axis=1) + 1e-09
        dots = np.dot(vectors_np, query_vector)
        similarities = dots / (q_norm * v_norms)
        
        final_scores = []
        for i, node in enumerate(node_names):
            score = similarities[i]
            if query_text and query_keywords:
                node_lower = node.lower()
                # Exact name match boost
                if any((kw == node_lower or node_lower in kw for kw in query_keywords)):
                    score += 1.0
                elif graph and graph.has_node(node):
                    content = graph.nodes[node].get('content', '').lower()
                    # Content keyword match boost
                    if any((kw in content for kw in query_keywords)):
                        score += 0.5
            final_scores.append(score)
            
        scored_nodes = sorted(zip(final_scores, node_names), key=lambda x: x[0], reverse=True)
        # We return results above the threshold, or always the best if it's a keyword match
        return [(score, node) for score, node in scored_nodes[:top_k] if score > threshold or score > 0.5]
