import os
from enum import Enum

# Logging Configuration
LOG_LEVEL = os.getenv("NEURON_X_LOG_LEVEL", "DEBUG").upper()

class GoalPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

# Default Paths
ENV_MODE = os.getenv("ENV", "PROD").upper()

if ENV_MODE == "DEV":
    DEFAULT_PERSISTENCE_PATH = "./memory_vault_dev"
else:
    DEFAULT_PERSISTENCE_PATH = "./memory_vault"

DEFAULT_GRAPH_FILENAME = "synaptic_graph.gexf"
DEFAULT_GOALS_FILENAME = "goals.json"
DEFAULT_THOUGHT_STREAM_FILENAME = "thought_stream.json"
DEFAULT_RESUME_FILENAME = "resume_state.json"

# Goal Weights
GOAL_WEIGHTS = {
    "CRITICAL": int(os.getenv("GOAL_WEIGHT_CRITICAL", "10")),
    "HIGH": int(os.getenv("GOAL_WEIGHT_HIGH", "5")),
    "MEDIUM": int(os.getenv("GOAL_WEIGHT_MEDIUM", "2")),
    "LOW": int(os.getenv("GOAL_WEIGHT_LOW", "1"))
}

# Thresholds
RECURSIVE_THOUGHT_THRESHOLD = float(os.getenv("RECURSIVE_THOUGHT_THRESHOLD", "0.95"))
SDI_AI_DAMPING = float(os.getenv("NEURON_SDI_AI_DAMPING", "0.8"))
SDI_AI_LENGTH_PENALTY = float(os.getenv("NEURON_SDI_AI_LENGTH_PENALTY", "2.0"))
THOUGHT_LOOP_INTERVAL = int(os.getenv("NEURON_THOUGHT_LOOP_INTERVAL", "60"))
