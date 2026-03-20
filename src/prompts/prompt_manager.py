"""
Prompt manager for RAG system.
Handles prompt loading, caching, and optimization.
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from .prompt_templates import PromptTemplates, PromptTemplate

class PromptManager:
    """Manages prompts for RAG applications."""
    
    def __init__(self, cache_dir: str = "prompts/cache"):
        self.prompt_templates = PromptTemplates()
        self.cache_dir = cache_dir
        self.prompt_cache = {}
        self.usage_stats = {}
        self._ensure_cache_dir()
        self._load_usage_stats()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_usage_stats(self):
        """Load usage statistics from file."""
        stats_file = os.path.join(self.cache_dir, "usage_stats.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    self.usage_stats = json.load(f)
            except Exception as e:
                print(f"Error loading usage stats: {e}")
                self.usage_stats = {}
    
    def _save_usage_stats(self):
        """Save usage statistics to file."""
        stats_file = os.path.join(self.cache_dir, "usage_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving usage stats: {e}")
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a formatted prompt.
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to format the template
            
        Returns:
            Formatted prompt string
        """
        try:
            # Format the template
            prompt = self.prompt_templates.format_template(template_name, **kwargs)
            
            # Update usage stats
            self._update_usage_stats(template_name)
            
            # Cache the prompt
            cache_key = self._generate_cache_key(template_name, kwargs)
            self.prompt_cache[cache_key] = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "template": template_name
            }
            
            return prompt
            
        except Exception as e:
            raise ValueError(f"Error generating prompt: {e}")
    
    def _update_usage_stats(self, template_name: str):
        """Update usage statistics for a template."""
        if template_name not in self.usage_stats:
            self.usage_stats[template_name] = {
                "count": 0,
                "last_used": None,
                "first_used": None
            }
        
        self.usage_stats[template_name]["count"] += 1
        self.usage_stats[template_name]["last_used"] = datetime.now().isoformat()
        
        if self.usage_stats[template_name]["first_used"] is None:
            self.usage_stats[template_name]["first_used"] = datetime.now().isoformat()
        
        # Save stats periodically
        if self.usage_stats[template_name]["count"] % 10 == 0:
            self._save_usage_stats()
    
    def _generate_cache_key(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Generate cache key for prompt."""
        import hashlib
        key_data = f"{template_name}_{sorted(variables.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_prompt(self, template_name: str, **kwargs) -> Optional[str]:
        """Get a cached prompt if available."""
        cache_key = self._generate_cache_key(template_name, kwargs)
        
        if cache_key in self.prompt_cache:
            cached_data = self.prompt_cache[cache_key]
            return cached_data["prompt"]
        
        return None
    
    def optimize_prompt_for_context(self, 
                                   template_name: str, 
                                   context: str, 
                                   max_length: int = 4000) -> str:
        """
        Optimize a prompt for a specific context length.
        
        Args:
            template_name: Name of the template
            context: Context to include
            max_length: Maximum prompt length
            
        Returns:
            Optimized prompt
        """
        template = self.prompt_templates.get_template(template_name)
        
        # Calculate available space for context
        template_without_context = template.template.replace("{context}", "")
        template_length = len(template_without_context)
        available_context_length = max_length - template_length - 100  # Buffer
        
        if available_context_length <= 0:
            raise ValueError("Template is too long for the specified max_length")
        
        # Truncate context if necessary
        if len(context) > available_context_length:
            context = self._truncate_context(context, available_context_length)
        
        # Format the prompt
        return self.get_prompt(template_name, context=context)
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context intelligently."""
        if len(context) <= max_length:
            return context
        
        # Try to truncate at sentence boundaries
        sentences = context.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated) + len(sentence) + 2 <= max_length:
                truncated += sentence + ". "
            else:
                break
        
        if not truncated:
            # If no sentence fits, truncate at word boundary
            words = context.split()
            truncated = ""
            for word in words:
                if len(truncated) + len(word) + 1 <= max_length:
                    truncated += word + " "
                else:
                    break
        
        return truncated.strip() + "..." if len(truncated) < len(context) else truncated.strip()
    
    def create_custom_template(self, 
                             name: str, 
                             template: str, 
                             description: str = "",
                             variables: List[str] = None,
                             category: str = "custom") -> bool:
        """
        Create a custom prompt template.
        
        Args:
            name: Template name
            template: Template string
            description: Template description
            variables: List of variables in template
            category: Template category
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract variables from template if not provided
            if variables is None:
                variables = self._extract_variables_from_template(template)
            
            custom_template = PromptTemplate(
                name=name,
                template=template,
                description=description,
                variables=variables,
                category=category
            )
            
            self.prompt_templates.add_template(custom_template)
            
            # Save custom template to file
            self._save_custom_template(custom_template)
            
            return True
            
        except Exception as e:
            print(f"Error creating custom template: {e}")
            return False
    
    def _extract_variables_from_template(self, template: str) -> List[str]:
        """Extract variable names from template string."""
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))
    
    def _save_custom_template(self, template: PromptTemplate):
        """Save custom template to file."""
        custom_templates_file = os.path.join(self.cache_dir, "custom_templates.json")
        
        # Load existing custom templates
        custom_templates = {}
        if os.path.exists(custom_templates_file):
            try:
                with open(custom_templates_file, 'r') as f:
                    custom_templates = json.load(f)
            except Exception:
                custom_templates = {}
        
        # Add new template
        custom_templates[template.name] = {
            "template": template.template,
            "description": template.description,
            "variables": template.variables,
            "category": template.category
        }
        
        # Save to file
        try:
            with open(custom_templates_file, 'w') as f:
                json.dump(custom_templates, f, indent=2)
        except Exception as e:
            print(f"Error saving custom template: {e}")
    
    def load_custom_templates(self):
        """Load custom templates from file."""
        custom_templates_file = os.path.join(self.cache_dir, "custom_templates.json")
        
        if not os.path.exists(custom_templates_file):
            return
        
        try:
            with open(custom_templates_file, 'r') as f:
                custom_templates = json.load(f)
            
            for name, template_data in custom_templates.items():
                custom_template = PromptTemplate(
                    name=name,
                    template=template_data["template"],
                    description=template_data.get("description", ""),
                    variables=template_data.get("variables", []),
                    category=template_data.get("category", "custom")
                )
                self.prompt_templates.add_template(custom_template)
                
        except Exception as e:
            print(f"Error loading custom templates: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all templates."""
        return self.usage_stats.copy()
    
    def get_most_used_templates(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most used templates."""
        sorted_templates = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {
                "template": name,
                "count": stats["count"],
                "last_used": stats["last_used"]
            }
            for name, stats in sorted_templates[:limit]
        ]
    
    def clear_cache(self):
        """Clear prompt cache."""
        self.prompt_cache.clear()
    
    def export_templates(self, file_path: str) -> bool:
        """Export all templates to a file."""
        try:
            templates_data = {}
            for name, template in self.prompt_templates.templates.items():
                templates_data[name] = {
                    "template": template.template,
                    "description": template.description,
                    "variables": template.variables,
                    "category": template.category
                }
            
            with open(file_path, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting templates: {e}")
            return False
    
    def import_templates(self, file_path: str) -> bool:
        """Import templates from a file."""
        try:
            with open(file_path, 'r') as f:
                templates_data = json.load(f)
            
            for name, template_data in templates_data.items():
                template = PromptTemplate(
                    name=name,
                    template=template_data["template"],
                    description=template_data.get("description", ""),
                    variables=template_data.get("variables", []),
                    category=template_data.get("category", "imported")
                )
                self.prompt_templates.add_template(template)
            
            return True
            
        except Exception as e:
            print(f"Error importing templates: {e}")
            return False
