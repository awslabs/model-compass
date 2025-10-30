# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection classes for models, profiles, and aliases.

This module provides collection classes that offer convenient access to models,
profiles, and aliases with iteration, containment, indexing support, and find methods.
"""

from typing import List, Iterator, Optional, Dict, Any, TYPE_CHECKING
from .exceptions import ResolutionError

if TYPE_CHECKING:
    from .resolution import ResolutionEngine
    from .proxy import ModelProxy


class ModelsCollection:
    """
    Collection for accessing all available models.
    
    Provides convenient access to models with iteration, containment, and indexing
    support, plus find methods for metadata-based model discovery.
    """
    
    def __init__(self):
        """Initialize empty collection."""
        self._resolution_engine: Optional['ResolutionEngine'] = None
    
    def _set_resolution_engine(self, resolution_engine: Optional['ResolutionEngine']):
        """Set the resolution engine for this collection."""
        self._resolution_engine = resolution_engine
    
    def _ensure_configured(self):
        """Ensure the collection is properly configured."""
        if self._resolution_engine is None:
            from .exceptions import ConfigurationError
            raise ConfigurationError("Library not configured. Call configure() first.")
    
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all model identifiers.
        
        Yields model IDs, profile names, and alias names that can be used
        to access models through this collection.
        
        Returns:
            Iterator over model identifiers
            
        Example:
            >>> for model_id in models:
            ...     print(f"Available model: {model_id}")
        """
        self._ensure_configured()
        
        # Get all available identifiers
        config_manager = self._resolution_engine._config_manager
        
        # Direct model IDs
        model_ids = list(config_manager.get_models().keys())
        
        # Profile names (these resolve to models)
        profile_names = list(config_manager.get_profiles().keys())
        
        # Alias names (these also resolve to models)
        alias_names = list(config_manager.get_aliases().keys())
        
        # Return all identifiers that can resolve to models
        all_identifiers = model_ids + profile_names + alias_names
        return iter(sorted(set(all_identifiers)))
    
    def __getitem__(self, identifier: str) -> 'ModelProxy':
        """
        Access model by identifier.
        
        Args:
            identifier: Model ID, profile name, alias name, or URN
            
        Returns:
            ModelProxy instance for the resolved model
            
        Raises:
            ResolutionError: If identifier cannot be resolved
            
        Example:
            >>> model = models['prod']
            >>> print(f"Using {model.provider}/{model.name}")
        """
        self._ensure_configured()
        from .proxy import ModelProxy
        return ModelProxy(identifier, self._resolution_engine)
    
    def __contains__(self, identifier: str) -> bool:
        """
        Check if model identifier exists and can be resolved.
        
        Args:
            identifier: Model ID, profile name, alias name, or URN to check
            
        Returns:
            True if identifier can be resolved, False otherwise
            
        Example:
            >>> if 'prod' in models:
            ...     print("Production model is available")
        """
        self._ensure_configured()
        
        try:
            # Try to resolve the identifier
            self._resolution_engine.resolve_identifier(identifier)
            return True
        except ResolutionError:
            return False
        except Exception:
            return False
    
    def list(self) -> List[str]:
        """
        List all available model identifiers.
        
        Returns a sorted list of all identifiers that can be used to access
        models, including model IDs, profile names, and alias names.
        
        Returns:
            List of model identifiers
            
        Example:
            >>> available_models = models.list()
            >>> print(f"Available models: {', '.join(available_models)}")
        """
        return list(self)
    
    def find(self, **criteria) -> List['ModelProxy']:
        """
        Find models matching the given metadata criteria.
        
        Searches through all available models and returns those that match
        the specified criteria. Supports exact matching and partial matching.
        
        Args:
            **criteria: Metadata criteria to match against
            
        Returns:
            List of ModelProxy instances matching the criteria
            
        Example:
            >>> high_reasoning_models = models.find(reasoning='high')
            >>> cheap_models = models.find(cost='low', speed='fast')
            >>> openai_models = models.find(provider='openai')
        """
        self._ensure_configured()
        
        matching_models = []
        config_manager = self._resolution_engine._config_manager
        
        # Search through all direct model configurations
        for model_id, model_config in config_manager.get_models().items():
            if self._matches_criteria(model_config, criteria):
                from .proxy import ModelProxy
                matching_models.append(ModelProxy(model_id, self._resolution_engine))
        
        return matching_models
    
    def _matches_criteria(self, model_config, criteria: Dict[str, Any]) -> bool:
        """
        Check if a model configuration matches the given criteria.
        
        Args:
            model_config: ModelConfig instance to check
            criteria: Dictionary of criteria to match against
            
        Returns:
            True if all criteria match, False otherwise
        """
        for key, expected_value in criteria.items():
            # Check provider field
            if key == "provider":
                if model_config.provider.lower() != str(expected_value).lower():
                    return False
                continue
            
            # Check model name field
            if key == "model":
                if model_config.model.lower() != str(expected_value).lower():
                    return False
                continue
            
            # Check metadata
            if key not in model_config.metadata:
                return False
            
            actual_value = model_config.metadata[key]
            
            # Handle different comparison types
            if isinstance(expected_value, str):
                if str(actual_value).lower() != expected_value.lower():
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def get_models_by_provider(self, provider: str) -> List['ModelProxy']:
        """
        Get all models from a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            List of ModelProxy instances from the specified provider
        """
        return self.find(provider=provider)
    
    def get_models_by_capability(self, **capabilities) -> List['ModelProxy']:
        """
        Get models by capability requirements.
        
        This is an alias for the find method with more descriptive naming
        for capability-based searches.
        
        Args:
            **capabilities: Capability requirements (reasoning, speed, cost, etc.)
            
        Returns:
            List of ModelProxy instances matching the capabilities
        """
        return self.find(**capabilities)


class ProfilesCollection:
    """
    Collection for accessing profiles.
    
    Provides convenient access to logical profiles with iteration, containment,
    and indexing support.
    """
    
    def __init__(self):
        """Initialize empty collection."""
        self._resolution_engine: Optional['ResolutionEngine'] = None
    
    def _set_resolution_engine(self, resolution_engine: Optional['ResolutionEngine']):
        """Set the resolution engine for this collection."""
        self._resolution_engine = resolution_engine
    
    def _ensure_configured(self):
        """Ensure the collection is properly configured."""
        if self._resolution_engine is None:
            from .exceptions import ConfigurationError
            raise ConfigurationError("Library not configured. Call configure() first.")
    
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all profile names.
        
        Returns:
            Iterator over profile names
            
        Example:
            >>> for profile_name in profiles:
            ...     print(f"Available profile: {profile_name}")
        """
        self._ensure_configured()
        config_manager = self._resolution_engine._config_manager
        profile_names = list(config_manager.get_profiles().keys())
        return iter(sorted(profile_names))
    
    def __getitem__(self, name: str) -> 'ModelProxy':
        """
        Access profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            ModelProxy instance for the profile
            
        Raises:
            ResolutionError: If profile does not exist
            
        Example:
            >>> prod_model = profiles['prod']
            >>> print(f"Production model: {prod_model}")
        """
        self._ensure_configured()
        
        # Validate that profile exists
        config_manager = self._resolution_engine._config_manager
        if not config_manager.has_profile(name):
            raise ResolutionError(
                f"Profile '{name}' not found",
                identifier=name,
                context={
                    "available_profiles": list(config_manager.get_profiles().keys())
                }
            )
        
        from .proxy import ModelProxy
        return ModelProxy(name, self._resolution_engine)
    
    def __contains__(self, name: str) -> bool:
        """
        Check if profile exists.
        
        Args:
            name: Profile name to check
            
        Returns:
            True if profile exists, False otherwise
            
        Example:
            >>> if 'prod' in profiles:
            ...     print("Production profile is configured")
        """
        self._ensure_configured()
        config_manager = self._resolution_engine._config_manager
        return config_manager.has_profile(name)
    
    def list(self) -> List[str]:
        """
        List all profile names.
        
        Returns:
            List of profile names
            
        Example:
            >>> available_profiles = profiles.list()
            >>> print(f"Available profiles: {', '.join(available_profiles)}")
        """
        return list(self)
    
    def get_profile_details(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a profile.
        
        Args:
            name: Profile name
            
        Returns:
            Dictionary with profile details including model reference and parameters
            
        Raises:
            ResolutionError: If profile does not exist
        """
        self._ensure_configured()
        
        config_manager = self._resolution_engine._config_manager
        profile = config_manager.get_profile(name)
        
        if profile is None:
            raise ResolutionError(
                f"Profile '{name}' not found",
                identifier=name,
                context={
                    "available_profiles": list(config_manager.get_profiles().keys())
                }
            )
        
        return {
            "name": name,
            "model": profile.model,
            "parameters": profile.parameters.copy()
        }


class AliasesCollection:
    """
    Collection for accessing aliases.
    
    Provides convenient access to aliases with iteration, containment, indexing
    support, and alias chain resolution methods.
    """
    
    def __init__(self):
        """Initialize empty collection."""
        self._resolution_engine: Optional['ResolutionEngine'] = None
    
    def _set_resolution_engine(self, resolution_engine: Optional['ResolutionEngine']):
        """Set the resolution engine for this collection."""
        self._resolution_engine = resolution_engine
    
    def _ensure_configured(self):
        """Ensure the collection is properly configured."""
        if self._resolution_engine is None:
            from .exceptions import ConfigurationError
            raise ConfigurationError("Library not configured. Call configure() first.")
    
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all alias names.
        
        Returns:
            Iterator over alias names
            
        Example:
            >>> for alias_name in aliases:
            ...     print(f"Available alias: {alias_name}")
        """
        self._ensure_configured()
        config_manager = self._resolution_engine._config_manager
        alias_names = list(config_manager.get_aliases().keys())
        return iter(sorted(alias_names))
    
    def __getitem__(self, name: str) -> 'ModelProxy':
        """
        Access alias by name.
        
        Args:
            name: Alias name
            
        Returns:
            ModelProxy instance for the resolved alias
            
        Raises:
            ResolutionError: If alias does not exist or cannot be resolved
            CircularAliasError: If circular reference detected
            
        Example:
            >>> main_model = aliases['main-model']
            >>> print(f"Main model: {main_model}")
        """
        self._ensure_configured()
        
        # Validate that alias exists
        config_manager = self._resolution_engine._config_manager
        if not config_manager.has_alias(name):
            raise ResolutionError(
                f"Alias '{name}' not found",
                identifier=name,
                context={
                    "available_aliases": list(config_manager.get_aliases().keys())
                }
            )
        
        from .proxy import ModelProxy
        return ModelProxy(name, self._resolution_engine)
    
    def __contains__(self, name: str) -> bool:
        """
        Check if alias exists.
        
        Args:
            name: Alias name to check
            
        Returns:
            True if alias exists, False otherwise
            
        Example:
            >>> if 'main-model' in aliases:
            ...     print("Main model alias is configured")
        """
        self._ensure_configured()
        config_manager = self._resolution_engine._config_manager
        return config_manager.has_alias(name)
    
    def list(self) -> List[str]:
        """
        List all alias names.
        
        Returns:
            List of alias names
            
        Example:
            >>> available_aliases = aliases.list()
            >>> print(f"Available aliases: {', '.join(available_aliases)}")
        """
        return list(self)
    
    def resolve_chain(self, name: str) -> List[str]:
        """
        Show the full resolution chain for an alias.
        
        Returns the complete chain from the initial alias to the final target,
        including all intermediate aliases.
        
        Args:
            name: Alias name to trace
            
        Returns:
            List of identifiers showing the full resolution path
            
        Raises:
            ResolutionError: If alias cannot be resolved
            CircularAliasError: If circular reference detected
            
        Example:
            >>> chain = aliases.resolve_chain('dev')
            >>> print(f"Resolution chain: {' -> '.join(chain)}")
        """
        self._ensure_configured()
        
        # Validate that alias exists
        config_manager = self._resolution_engine._config_manager
        if not config_manager.has_alias(name):
            raise ResolutionError(
                f"Alias '{name}' not found",
                identifier=name,
                context={
                    "available_aliases": list(config_manager.get_aliases().keys())
                }
            )
        
        # Use the alias resolver to get the full chain
        alias_resolver = self._resolution_engine.get_alias_resolver()
        return alias_resolver.resolve_chain(name)
    
    def get_alias_target(self, name: str) -> str:
        """
        Get the direct target of an alias (without following the chain).
        
        Args:
            name: Alias name
            
        Returns:
            Direct target of the alias
            
        Raises:
            ResolutionError: If alias does not exist
        """
        self._ensure_configured()
        
        config_manager = self._resolution_engine._config_manager
        target = config_manager.get_alias(name)
        
        if target is None:
            raise ResolutionError(
                f"Alias '{name}' not found",
                identifier=name,
                context={
                    "available_aliases": list(config_manager.get_aliases().keys())
                }
            )
        
        return target
    
    def get_dependencies(self, name: str) -> List[str]:
        """
        Get all aliases that depend on the given alias.
        
        This method finds all aliases that directly or indirectly reference
        the given alias, which is useful for understanding the impact of
        changing an alias target.
        
        Args:
            name: Alias name to find dependencies for
            
        Returns:
            List of alias names that depend on the given alias
            
        Raises:
            ResolutionError: If alias does not exist
        """
        self._ensure_configured()
        
        # Validate that alias exists
        config_manager = self._resolution_engine._config_manager
        if not config_manager.has_alias(name):
            raise ResolutionError(
                f"Alias '{name}' not found",
                identifier=name,
                context={
                    "available_aliases": list(config_manager.get_aliases().keys())
                }
            )
        
        # Use the alias resolver to get dependencies
        alias_resolver = self._resolution_engine.get_alias_resolver()
        return alias_resolver.get_alias_dependencies(name)
    
    def validate_all(self) -> List[str]:
        """
        Validate all alias references in the configuration.
        
        Checks all aliases for circular references and invalid targets.
        This method can be used during configuration validation to detect
        issues before they cause runtime errors.
        
        Returns:
            List of validation error messages (empty if all aliases are valid)
        """
        self._ensure_configured()
        
        alias_resolver = self._resolution_engine.get_alias_resolver()
        return alias_resolver.validate_alias_references()