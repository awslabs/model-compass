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
Model proxy classes for property-based access to model configurations.

This module provides the ModelProxy class which offers a clean, property-based
interface for accessing resolved model configurations with convenience methods
and framework compatibility features.
"""

from typing import Dict, Any, Optional, Iterator, Tuple, Union, TYPE_CHECKING
from .data_models import ResolvedModel
from .exceptions import ResolutionError

if TYPE_CHECKING:
    from .resolution import ResolutionEngine


class ModelProxy:
    """
    Proxy object for accessing model configuration with property-based syntax.
    
    Provides a clean interface for accessing resolved model configurations with
    property-based access, convenience properties for common metadata, and
    dictionary-like access for framework compatibility.
    """
    
    def __init__(self, identifier: Optional[str], resolution_engine: 'ResolutionEngine'):
        """
        Initialize ModelProxy with an identifier and resolution engine.
        
        Args:
            identifier: Model identifier to resolve (URN, profile, alias, or physical ID)
            resolution_engine: ResolutionEngine instance for resolving the identifier
        """
        self._identifier = identifier
        self._resolution_engine = resolution_engine
        self._resolved_model = None
        
        # Cache for computed properties
        self._property_cache = {}
    
    def _ensure_resolved(self) -> ResolvedModel:
        """
        Ensure the model has been resolved and return the resolved model.
        
        This method performs lazy resolution - the model is only resolved
        when first accessed, not when the ModelProxy is created.
        """
        if self._resolved_model is None:
            # Handle current profile context
            from .core import _get_current_profile
            current_profile = _get_current_profile()
            
            # If no identifier provided, use current profile or default
            identifier = self._identifier
            if identifier is None:
                if current_profile:
                    identifier = current_profile
                # If still None, resolution engine will use default profile
            
            # Resolve the identifier
            self._resolved_model = self._resolution_engine.resolve_identifier(identifier)
        
        return self._resolved_model
    
    # Core properties from resolved model
    @property
    def provider(self) -> str:
        """The provider name (e.g., 'openai', 'anthropic')"""
        resolved = self._ensure_resolved()
        return resolved.config.provider
    
    @property
    def name(self) -> str:
        """The model name (e.g., 'gpt-4', 'claude-3-sonnet')"""
        resolved = self._ensure_resolved()
        return resolved.config.model
    
    @property
    def base_url(self) -> str:
        """The API base URL"""
        resolved = self._ensure_resolved()
        return resolved.provider_config.base_url
    
    @property
    def timeout(self) -> int:
        """Request timeout in seconds"""
        resolved = self._ensure_resolved()
        return resolved.provider_config.timeout
    
    @property
    def headers(self) -> Dict[str, str]:
        """Additional HTTP headers for requests"""
        resolved = self._ensure_resolved()
        return resolved.provider_config.headers.copy()
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Model parameters (temperature, max_tokens, etc.)"""
        resolved = self._ensure_resolved()
        return resolved.effective_parameters.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Model metadata (reasoning, cost, speed, etc.)"""
        resolved = self._ensure_resolved()
        return resolved.config.metadata.copy()
    
    # Convenience properties for common metadata
    @property
    def is_local(self) -> bool:
        """True if this is a local model (e.g., Ollama, local inference)"""
        # Check if base URL indicates local deployment
        base_url = self.base_url.lower()
        local_indicators = ['localhost', '127.0.0.1', '0.0.0.0', 'local']
        
        if any(indicator in base_url for indicator in local_indicators):
            return True
        
        # Check metadata for local indicator
        deployment = self.metadata.get('deployment', '').lower()
        if deployment in ['local', 'on-premise', 'self-hosted']:
            return True
        
        # Check provider for known local providers
        local_providers = ['ollama', 'local', 'self-hosted']
        if self.provider.lower() in local_providers:
            return True
        
        return False
    
    @property
    def is_cloud(self) -> bool:
        """True if this is a cloud model"""
        return not self.is_local
    
    @property
    def cost_tier(self) -> str:
        """Cost tier: 'low', 'medium', or 'high'"""
        cost = self.metadata.get('cost', '').lower()
        if cost in ['low', 'medium', 'high']:
            return cost
        
        # Fallback logic based on provider and model
        if self.is_local:
            return 'low'  # Local models are typically low cost
        
        # Known expensive models
        expensive_models = ['gpt-4', 'claude-3-opus']
        if any(model in self.name.lower() for model in expensive_models):
            return 'high'
        
        # Default to medium for cloud models
        return 'medium' if self.is_cloud else 'low'
    
    @property
    def reasoning_level(self) -> str:
        """Reasoning level: 'low', 'medium', or 'high'"""
        reasoning = self.metadata.get('reasoning', '').lower()
        if reasoning in ['low', 'medium', 'high']:
            return reasoning
        
        # Fallback logic based on model name
        high_reasoning_models = ['gpt-4', 'claude-3', 'claude-2']
        if any(model in self.name.lower() for model in high_reasoning_models):
            return 'high'
        
        medium_reasoning_models = ['gpt-3.5', 'claude-instant']
        if any(model in self.name.lower() for model in medium_reasoning_models):
            return 'medium'
        
        return 'medium'  # Default to medium
    
    @property
    def speed_tier(self) -> str:
        """Speed tier: 'slow', 'medium', or 'fast'"""
        speed = self.metadata.get('speed', '').lower()
        if speed in ['slow', 'medium', 'fast']:
            return speed
        
        # Fallback logic
        if self.is_local:
            return 'medium'  # Local models vary in speed
        
        # Known fast models
        fast_models = ['gpt-3.5-turbo', 'claude-instant']
        if any(model in self.name.lower() for model in fast_models):
            return 'fast'
        
        # Large models tend to be slower
        if 'gpt-4' in self.name.lower() or 'claude-3-opus' in self.name.lower():
            return 'slow'
        
        return 'medium'  # Default to medium
    
    @property
    def context_length(self) -> Optional[int]:
        """Maximum context length in tokens"""
        context_len = self.metadata.get('context_length')
        if context_len is not None:
            try:
                return int(context_len)
            except (ValueError, TypeError):
                pass
        
        # Fallback based on known models
        if 'gpt-4' in self.name.lower():
            return 8192
        elif 'gpt-3.5' in self.name.lower():
            return 4096
        elif 'claude-3' in self.name.lower():
            return 200000
        elif 'claude-2' in self.name.lower():
            return 100000
        
        return None
    
    # Dictionary-like access for framework compatibility
    def __getitem__(self, key: str) -> Any:
        """
        Allow dict-like access: model['provider']
        
        Supports accessing all properties and metadata through dictionary syntax.
        """
        # Map common keys to properties
        property_map = {
            'provider': self.provider,
            'model': self.name,
            'name': self.name,
            'base_url': self.base_url,
            'timeout': self.timeout,
            'headers': self.headers,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'is_local': self.is_local,
            'is_cloud': self.is_cloud,
            'cost_tier': self.cost_tier,
            'reasoning_level': self.reasoning_level,
            'speed_tier': self.speed_tier,
            'context_length': self.context_length
        }
        
        if key in property_map:
            return property_map[key]
        
        # Check in parameters
        if key in self.parameters:
            return self.parameters[key]
        
        # Check in metadata
        if key in self.metadata:
            return self.metadata[key]
        
        # Check in headers
        if key in self.headers:
            return self.headers[key]
        
        raise KeyError(f"Key '{key}' not found in model configuration")
    
    def __contains__(self, key: str) -> bool:
        """Support 'key in model' syntax"""
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def keys(self) -> Iterator[str]:
        """Return available keys"""
        # Core properties
        core_keys = [
            'provider', 'model', 'name', 'base_url', 'timeout', 'headers',
            'parameters', 'metadata', 'is_local', 'is_cloud', 'cost_tier',
            'reasoning_level', 'speed_tier', 'context_length'
        ]
        
        # Add parameter keys
        param_keys = list(self.parameters.keys())
        
        # Add metadata keys
        metadata_keys = list(self.metadata.keys())
        
        # Add header keys
        header_keys = list(self.headers.keys())
        
        # Return unique keys
        all_keys = set(core_keys + param_keys + metadata_keys + header_keys)
        return iter(sorted(all_keys))
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """Return key-value pairs"""
        for key in self.keys():
            yield key, self[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for framework compatibility.
        
        Returns a dictionary containing all model configuration data
        in a format suitable for use with various ML frameworks.
        """
        return {
            # Core configuration
            'provider': self.provider,
            'model': self.name,
            'base_url': self.base_url,
            'timeout': self.timeout,
            'headers': self.headers,
            
            # Parameters and metadata
            'parameters': self.parameters,
            'metadata': self.metadata,
            
            # Convenience properties
            'is_local': self.is_local,
            'is_cloud': self.is_cloud,
            'cost_tier': self.cost_tier,
            'reasoning_level': self.reasoning_level,
            'speed_tier': self.speed_tier,
            'context_length': self.context_length
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value with optional default (dict-like interface).
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value for key or default if not found
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def update_parameters(self, **kwargs) -> 'ModelProxy':
        """
        Create a new ModelProxy with updated parameters.
        
        This method returns a new ModelProxy instance with the specified
        parameters merged into the existing parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            New ModelProxy instance with updated parameters
        """
        # Ensure we have a resolved model
        resolved = self._ensure_resolved()
        
        # Create a copy of the resolved model with updated parameters
        new_parameters = resolved.effective_parameters.copy()
        new_parameters.update(kwargs)
        
        new_resolved_model = ResolvedModel(
            config=resolved.config,
            provider_config=resolved.provider_config,
            effective_parameters=new_parameters
        )
        
        # Create a new ModelProxy with the updated resolved model
        new_proxy = ModelProxy(self._identifier, self._resolution_engine)
        new_proxy._resolved_model = new_resolved_model
        return new_proxy
    
    # String representations
    def __str__(self) -> str:
        """String representation showing provider/model"""
        return f"{self.provider}/{self.name}"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging"""
        if self._identifier:
            return f"ModelProxy('{self._identifier}' -> {self.provider}/{self.name})"
        else:
            return f"ModelProxy({self.provider}/{self.name})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on provider and model name"""
        if not isinstance(other, ModelProxy):
            return False
        return self.provider == other.provider and self.name == other.name
    
    def __hash__(self) -> int:
        """Hash based on provider and model name"""
        return hash((self.provider, self.name))