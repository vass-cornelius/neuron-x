# OPTIMIZER_PROTECTED
"""
Commit management for safe code deployment.

This module handles the atomic commitment of validated optimizations to the codebase,
including backup creation and rollback capabilities.
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple

logger = logging.getLogger("neuron-x.optimizer")


class BackupInfo(NamedTuple):
    """Information about a code backup."""
    
    backup_id: str
    original_path: Path
    backup_path: Path
    created_at: datetime


@dataclass
class CommitResult:
    """Result of a commit operation."""
    
    success: bool
    backup_id: str | None
    error_message: str = ""


class CommitManager:
    """
    Manages safe commits and rollbacks of optimization changes.
    
    This class ensures atomic file operations with automatic backups,
    enabling safe rollback in case of issues.
    """
    
    def __init__(
        self,
        backup_dir: Path | str = Path("./tmp/optimizer/backups"),
        retention_days: int = 30
    ) -> None:
        """
        Initialize the commit manager.
        
        Args:
            backup_dir: Directory for storing backups
            retention_days: Number of days to keep backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._backups: dict[str, BackupInfo] = {}
    
    def create_backup(self, file_path: Path | str) -> str:
        """
        Create a timestamped backup of a file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Unique backup ID
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If backup creation fails
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
        
        # Generate unique backup ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_id = f"{file_path.stem}_{timestamp}"
        
        # Create backup path
        backup_path = self.backup_dir / f"{backup_id}{file_path.suffix}"
        
        # Copy file to backup
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup {backup_id} for {file_path}")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise IOError(f"Backup creation failed: {e}") from e
        
        # Track backup
        backup_info = BackupInfo(
            backup_id=backup_id,
            original_path=file_path,
            backup_path=backup_path,
            created_at=datetime.now()
        )
        self._backups[backup_id] = backup_info
        
        return backup_id
    
    def commit_changes(
        self,
        file_path: Path | str,
        new_content: str,
        create_backup: bool = True
    ) -> CommitResult:
        """
        Atomically commit changes to a file.
        
        This method writes to a temporary file first, then performs an atomic
        rename to ensure the operation is all-or-nothing.
        
        Args:
            file_path: Path to the file to update
            new_content: New file content
            create_backup: Whether to create a backup before committing
            
        Returns:
            CommitResult with success status and backup ID
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        backup_id: str | None = None
        
        try:
            # Create backup if requested and file exists
            if create_backup and file_path.exists():
                backup_id = self.create_backup(file_path)
            
            # Write to temporary file
            temp_path = file_path.with_suffix(f'.tmp_{datetime.now().timestamp()}')
            temp_path.write_text(new_content, encoding='utf-8')
            
            # Atomic rename (POSIX guarantee)
            temp_path.replace(file_path)
            
            logger.info(f"Successfully committed changes to {file_path}")
            return CommitResult(success=True, backup_id=backup_id)
            
        except Exception as e:
            error_msg = f"Failed to commit changes: {e}"
            logger.error(error_msg)
            
            # Attempt rollback if we created a backup
            if backup_id:
                logger.info("Attempting automatic rollback...")
                rollback_result = self.rollback(file_path, backup_id)
                if rollback_result:
                    error_msg += " (automatic rollback successful)"
                else:
                    error_msg += " (automatic rollback failed)"
            
            return CommitResult(success=False, backup_id=backup_id, error_message=error_msg)
    
    def rollback(self, file_path: Path | str, backup_id: str) -> bool:
        """
        Restore a file from a backup.
        
        Args:
            file_path: Path to the file to restore
            backup_id: ID of the backup to restore from
            
        Returns:
            True if rollback was successful, False otherwise
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        backup_info = self._backups.get(backup_id)
        if not backup_info:
            logger.error(f"Backup {backup_id} not found")
            return False
        
        if not backup_info.backup_path.exists():
            logger.error(f"Backup file {backup_info.backup_path} not found")
            return False
        
        try:
            # Copy backup to original location
            shutil.copy2(backup_info.backup_path, file_path)
            logger.info(f"Successfully rolled back {file_path} to backup {backup_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_backup_info(self, backup_id: str) -> BackupInfo | None:
        """
        Get information about a backup.
        
        Args:
            backup_id: Unique backup identifier
            
        Returns:
            BackupInfo if found, None otherwise
        """
        return self._backups.get(backup_id)
    
    def list_backups(self, file_path: Path | str | None = None) -> list[BackupInfo]:
        """
        List all backups, optionally filtered by file.
        
        Args:
            file_path: Optional path to filter backups by original file
            
        Returns:
            List of BackupInfo objects
        """
        if file_path is None:
            return list(self._backups.values())
        
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        return [
            info for info in self._backups.values()
            if info.original_path == file_path
        ]
    
    def cleanup_old_backups(self) -> int:
        """
        Remove backups older than the retention period.
        
        Returns:
            Number of backups cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0
        
        backup_ids_to_remove = [
            backup_id for backup_id, info in self._backups.items()
            if info.created_at < cutoff_date
        ]
        
        for backup_id in backup_ids_to_remove:
            info = self._backups[backup_id]
            try:
                info.backup_path.unlink(missing_ok=True)
                del self._backups[backup_id]
                removed_count += 1
                logger.info(f"Removed old backup {backup_id}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup_id}: {e}")
        
        return removed_count
