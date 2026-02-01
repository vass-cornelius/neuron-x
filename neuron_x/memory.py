import json
import logging
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer, CrossEncoder # type: ignore

from neuron_x.const import RECURSIVE_THOUGHT_THRESHOLD

logger = logging.getLogger("neuron-x")

class VectorVault:
    """
    Service class responsible for Vector Embeddings, Similarity Search, 
    and Semantic Verification.
    """
    def __init__(self):
        logger.info("[bold blue][VECTOR_VAULT][/bold blue] Loading Neural Models...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        except Exception as e:
            logger.warning(f"[bold yellow][VECTOR_VAULT][/bold yellow] Failed to load CrossEncoder: {e}. Semantic merging will be limited.")
            self.cross_encoder = None
            
        self.vector_cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """Encodes text into a vector."""
        return self.encoder.encode(text)

    def rebuild_cache(self, graph: nx.DiGraph) -> None:
        """Parscache all vectors from the graph into a fast-access dictionary."""
        self.vector_cache = {}
        for node, data in graph.nodes(data=True):
            if "vector" in data:
                vec_data = data["vector"]
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
            # Cosine similarity
            q_norm = np.linalg.norm(vector) + 1e-9
            v_norm = np.linalg.norm(t_vec) + 1e-9
            similarity = np.dot(vector, t_vec) / (q_norm * v_norm)
            
            if similarity > RECURSIVE_THOUGHT_THRESHOLD:
                # We don't log here to avoid side effects in a calculation function, 
                # but returning True signals the loop.
                return True
        return False

    def verify_pair_identity(self, name_a: str, name_b: str) -> bool:
        """
        Uses Cross-Encoder to verify if two names refer to the exact same entity.
        Returns True if high confidence.
        """
        if not self.cross_encoder:
            return False
            
        try:
            score = self.cross_encoder.predict([(name_a, name_b)])
            return score > 0.80
        except Exception as e:
            logger.warning(f"CrossEncoder check failed: {e}")
            return False
            
    def get_similar_nodes(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.0, graph: Optional[nx.DiGraph] = None) -> List[Tuple[float, str]]:
        """
        Finds top_k similar nodes from the cache.
        Returns list of (score, node_name).
        """
        if not self.vector_cache:
            return []
            
        node_names = []
        vectors = []
        
        for node, vec in self.vector_cache.items():
            if node == "Self":
                continue
            # Check graph status if provided
            if graph and graph.has_node(node) and graph.nodes[node].get("status") == "REJECTED":
                continue
                
            node_names.append(node)
            vectors.append(vec)
            
        if not vectors:
            return []

        vectors_np = np.array(vectors)
        q_norm = np.linalg.norm(query_vector) + 1e-9
        v_norms = np.linalg.norm(vectors_np, axis=1) + 1e-9
        dots = np.dot(vectors_np, query_vector)
        similarities = dots / (q_norm * v_norms)
        
        # Zip and sort
        scored_nodes = sorted(zip(similarities, node_names), key=lambda x: x[0], reverse=True)
        
        # Filter and cut
        return [(score, node) for score, node in scored_nodes[:top_k] if score > threshold]
