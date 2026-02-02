# OPTIMIZER_PROTECTED
"""
Skill Manager Plugin - Enables autonomous skill creation and discovery.

This plugin provides tools for NeuronX to:
- Create new skills and store them in .agent/skills/
- List available skills
- Search for skills online
- Install skills from URLs
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from collections.abc import Mapping

from neuron_x.plugin_base import BasePlugin, PluginMetadata

logger = logging.getLogger("neuron-x.plugins.skill-manager")


class SkillManagerPlugin(BasePlugin):
    """
    Plugin for managing skills in .agent/skills/.
    
    Provides tools for creating, discovering, and installing skills.
    """
    
    def __init__(self) -> None:
        """Initialize the skill manager plugin."""
        super().__init__()
        self._project_root: Optional[Path] = None
        self._skills_dir: Optional[Path] = None
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="skill_manager",
            version="1.0.0",
            description="Manage autonomous skills in .agent/skills/",
            author="NeuronX Team",
            capabilities=["skills", "automation", "discovery"],
        )
    
    def on_load(self) -> None:
        """Initialize the plugin on load."""
        super().on_load()
        # In the context of NeuronX, the project root is usually the current directory
        self._project_root = Path.cwd()
        self._skills_dir = self._project_root / ".agent" / "skills"
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[SKILL_MANAGER] Plugin initialized at {self._skills_dir}")

    def get_tools(self) -> Mapping[str, Any]:
        """
        Return available tools for this plugin.
        
        Returns:
            Dictionary mapping tool names to callables
        """
        return {
            "create_skill": self.create_skill,
            "list_skills": self.list_skills,
            "search_skills_online": self.search_skills_online,
            "install_skill_from_url": self.install_skill_from_url
        }
    
    def _sanitize_skill_name(self, name: str) -> str:
        """
        Sanitize skill name to prevent directory traversal.
        
        Args:
            name: Skill name input
            
        Returns:
            Sanitized skill name safe for filesystem use
        """
        # Remove any path separators and parent directory references
        name = name.replace('/', '-').replace('\\', '-').replace('..', '')
        # Keep only alphanumeric, hyphens, and underscores
        name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        return name.lower()
    
    def _validate_yaml_frontmatter(self, content: str) -> bool:
        """
        Basic validation of YAML frontmatter format.
        
        Args:
            content: SKILL.md content
            
        Returns:
            True if valid frontmatter exists
        """
        lines = content.strip().split('\n')
        if len(lines) < 3:
            return False
        if lines[0] != '---':
            return False
        
        # Find closing ---
        for i in range(1, len(lines)):
            if lines[i] == '---':
                return True
        return False
    
    def create_skill(
        self,
        skill_name: str,
        description: str,
        instructions: str,
        examples: Optional[str] = None,
        scripts: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new skill and store it in .agent/skills/.
        
        Args:
            skill_name: Name of the skill (will be sanitized)
            description: Brief description of what the skill does
            instructions: Detailed markdown instructions for using the skill
            examples: Optional examples section (markdown)
            scripts: Optional dict of {filename: content} for script files
            
        Returns:
            Success message with skill path
        """
        if not self._skills_dir:
            return "Error: Plugin not initialized"
        
        # Sanitize skill name
        safe_name = self._sanitize_skill_name(skill_name)
        if not safe_name:
            return f"Error: Invalid skill name '{skill_name}'"
        
        skill_dir = self._skills_dir / safe_name
        
        # Check if skill already exists
        if skill_dir.exists():
            return f"Error: Skill '{safe_name}' already exists at {skill_dir}"
        
        try:
            # Create skill directory
            skill_dir.mkdir(parents=True, exist_ok=True)
            
            # Build SKILL.md content
            skill_content = f"""---
name: {safe_name}
description: {description}
---

# {safe_name.replace('-', ' ').title()}

{instructions}
"""
            
            if examples:
                skill_content += f"\n## Examples\n\n{examples}\n"
            
            # Validate frontmatter
            if not self._validate_yaml_frontmatter(skill_content):
                skill_dir.rmdir()
                return "Error: Generated invalid YAML frontmatter"
            
            # Write SKILL.md
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(skill_content, encoding='utf-8')
            
            # Create scripts directory if scripts provided
            if scripts:
                scripts_dir = skill_dir / "scripts"
                scripts_dir.mkdir(exist_ok=True)
                
                for filename, content in scripts.items():
                    script_file = scripts_dir / filename
                    script_file.write_text(content, encoding='utf-8')
                
                logger.info(f"[SKILL_MANAGER] Created {len(scripts)} script(s) for '{safe_name}'")
            
            logger.info(f"[SKILL_MANAGER] Created skill '{safe_name}' at {skill_file}")
            return f"✓ Successfully created skill '{safe_name}' at {skill_file.relative_to(self._project_root)}"
        
        except Exception as e:
            # Cleanup on failure
            if skill_dir.exists():
                import shutil
                shutil.rmtree(skill_dir)
            
            error_msg = f"Failed to create skill: {e}"
            logger.error(f"[SKILL_MANAGER] {error_msg}")
            return f"Error: {error_msg}"
    
    def list_skills(self) -> str:
        """
        List all available skills in .agent/skills/.
        
        Returns:
            Formatted list of skills with names and descriptions
        """
        if not self._skills_dir:
            return "Error: Plugin not initialized"
        
        if not self._skills_dir.exists():
            return "No skills directory found"
        
        skills = []
        
        for skill_dir in self._skills_dir.iterdir():
            if not skill_dir.is_dir() or skill_dir.name.startswith('.'):
                continue
            
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            
            try:
                content = skill_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                # Parse frontmatter
                if lines[0] == '---':
                    name = None
                    description = None
                    
                    for i in range(1, len(lines)):
                        if lines[i] == '---':
                            break
                        if lines[i].startswith('name:'):
                            name = lines[i].split(':', 1)[1].strip()
                        elif lines[i].startswith('description:'):
                            description = lines[i].split(':', 1)[1].strip()
                    
                    if name:
                        skills.append({
                            'name': name,
                            'description': description or 'No description',
                            'path': str(skill_file.relative_to(self._project_root))
                        })
            
            except Exception as e:
                logger.warning(f"[SKILL_MANAGER] Failed to parse {skill_file}: {e}")
                continue
        
        if not skills:
            return "No skills found in .agent/skills/"
        
        # Format output
        output = ["Available Skills:\n"]
        for i, skill in enumerate(skills, 1):
            output.append(f"{i}. **{skill['name']}**")
            output.append(f"   {skill['description']}")
            output.append(f"   Path: {skill['path']}\n")
        
        return '\n'.join(output)
    
    def search_skills_online(self, query: str) -> str:
        """
        Search the web for existing skills.
        """
        return f"Searching for skills matching '{query}'... (Feature not fully implemented)"
    
    def install_skill_from_url(self, url: str, skill_name: str) -> str:
        """
        Download and install a skill from a URL.
        """
        return f"Installing skill '{skill_name}' from {url}... (Feature not fully implemented)"
