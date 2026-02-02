# OPTIMIZER_PROTECTED
"""
Draft management for code optimizations.

This module handles the creation, storage, and cleanup of draft code versions
during the optimization process. All drafts are stored in a temporary directory
with unique identifiers to prevent conflicts.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger("neuron-x.optimizer")


class DraftPaths(NamedTuple):
    """Paths for a draft optimization."""
    
    draft_id: str
    original: Path
    optimized: Path


@dataclass
class Draft:
    """
    Represents a code optimization draft.
    
    Attributes:
        draft_id: Unique identifier for this draft
        module_path: Original module path being optimized
        original_code: Original source code
        optimized_code: Optimized source code
        created_at: Timestamp when draft was created
    """
    
    draft_id: str
    module_path: Path
    original_code: str
    optimized_code: str
    created_at: datetime


class DraftManager:
    """
    Manages temporary code drafts during optimization.
    
    This class creates isolated draft versions of code for testing and validation
    before committing changes to the actual codebase.
    """
    
    def __init__(self, draft_dir: Path | str = Path("./tmp/optimizer/drafts")) -> None:
        """
        Initialize the draft manager.
        
        Args:
            draft_dir: Directory for storing drafts
        """
        self.draft_dir = Path(draft_dir)
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        self._active_drafts: dict[str, Draft] = {}
    
    def create_draft(
        self,
        module_path: Path | str,
        original_code: str,
        optimized_code: str
    ) -> Draft:
        """
        Create a new optimization draft.
        
        Args:
            module_path: Path to the original module
            original_code: Original source code
            optimized_code: Optimized source code
            
        Returns:
            Draft object with unique ID and file paths
        """
        if isinstance(module_path, str):
            module_path = Path(module_path)
        
        # Generate unique draft ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        module_name = module_path.stem
        draft_id = f"{module_name}_{timestamp}"
        
        # Create draft
        draft = Draft(
            draft_id=draft_id,
            module_path=module_path,
            original_code=original_code,
            optimized_code=optimized_code,
            created_at=datetime.now()
        )
        
        # Write draft files
        paths = self._get_draft_paths(draft_id)
        paths.original.write_text(original_code)
        paths.optimized.write_text(optimized_code)
        
        # Track active draft
        self._active_drafts[draft_id] = draft
        
        logger.info(f"Created draft {draft_id} for {module_path}")
        return draft
    
    def get_draft(self, draft_id: str) -> Draft | None:
        """
        Retrieve a draft by ID.
        
        Args:
            draft_id: Unique draft identifier
            
        Returns:
            Draft object if found, None otherwise
        """
        return self._active_drafts.get(draft_id)
    
    def get_draft_paths(self, draft_id: str) -> DraftPaths | None:
        """
        Get file paths for a draft.
        
        Args:
            draft_id: Unique draft identifier
            
        Returns:
            DraftPaths with original and optimized file paths, or None if not found
        """
        if draft_id not in self._active_drafts:
            return None
        return self._get_draft_paths(draft_id)
    
    def _get_draft_paths(self, draft_id: str) -> DraftPaths:
        """Internal method to construct draft paths."""
        return DraftPaths(
            draft_id=draft_id,
            original=self.draft_dir / f"{draft_id}_original.py",
            optimized=self.draft_dir / f"{draft_id}_optimized.py"
        )
    
    def cleanup_draft(self, draft_id: str) -> bool:
        """
        Remove a draft and its associated files.
        
        Args:
            draft_id: Unique draft identifier
            
        Returns:
            True if draft was cleaned up, False if not found
        """
        if draft_id not in self._active_drafts:
            logger.warning(f"Draft {draft_id} not found for cleanup")
            return False
        
        paths = self._get_draft_paths(draft_id)
        
        # Remove files
        paths.original.unlink(missing_ok=True)
        paths.optimized.unlink(missing_ok=True)
        
        # Remove from tracking
        del self._active_drafts[draft_id]
        
        logger.info(f"Cleaned up draft {draft_id}")
        return True
    
    def cleanup_all(self) -> int:
        """
        Clean up all active drafts.
        
        Returns:
            Number of drafts cleaned up
        """
        draft_ids = list(self._active_drafts.keys())
        count = 0
        
        for draft_id in draft_ids:
            if self.cleanup_draft(draft_id):
                count += 1
        
        return count
    
    def list_drafts(self) -> list[str]:
        """
        List all active draft IDs.
        
        Returns:
            List of draft IDs
        """
        return list(self._active_drafts.keys())
