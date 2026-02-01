from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class TripleCategory(str, Enum):
    FACTUAL = "FACTUAL"
    INFERENCE = "INFERENCE"
    PROPOSAL = "PROPOSAL"
    HYPOTHESIS = "HYPOTHESIS"

from neuron_x.const import GoalPriority

class Goal(BaseModel):
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex, description="Unique identifier for the goal.")
    description: str = Field(description="The specific objective of this goal.")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, description="The urgency of this goal.")
    status: str = Field(default="PENDING", description="PENDING, IN_PROGRESS, COMPLETED, or FAILED")
    created_at: float = Field(default_factory=lambda: __import__("time").time(), description="Timestamp of creation")

class SemanticTriple(BaseModel):
    subject: str = Field(description="The subject of the relationship (e.g., 'Kaelen').")
    predicate: str = Field(description="The relationship between subject and object (e.g., 'is_a', 'has_weapon').")
    object: str = Field(description="The object or value of the relationship (e.g., 'Wood Elf Rogue', 'Sunblade').")
    category: TripleCategory = Field(description="The epistemological category of the triple.")
    index: int = Field(default=0, description="The index of the memory this triple was extracted from (0-based).")

class ExtractionResponse(BaseModel):
    triples: List[SemanticTriple] = Field(description="A flat list of all extracted triples across all provided memories.")
