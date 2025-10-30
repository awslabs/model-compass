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
Core data classes for the Model Compass library.

This module defines the fundamental data structures used throughout the library
for representing model configurations, profiles, provider settings, and resolved models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """
    Configuration for a specific model.
    
    Attributes:
        provider: The provider name (e.g., 'openai', 'anthropic', 'ollama')
        model: The model name (e.g., 'gpt-4', 'claude-3-sonnet', 'llama3')
        metadata: Model metadata including capabilities and characteristics
        parameters: Model-specific parameters (temperature, max_tokens, etc.)
    """
    provider: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.provider:
            raise ValueError("Provider cannot be empty")
        if not self.model:
            raise ValueError("Model cannot be empty")


@dataclass
class Profile:
    """
    A logical profile that references a model with specific parameters.
    
    Profiles provide a way to group model configurations with specific settings
    for different use cases (e.g., 'prod', 'cheap', 'local').
    
    Attributes:
        model: Reference to a model (can be model ID, alias, or physical identifier)
        parameters: Profile-specific parameters that override model defaults
    """
    model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.model:
            raise ValueError("Model reference cannot be empty")


@dataclass
class ProviderConfig:
    """
    Configuration for a specific provider.
    
    Contains provider-specific settings like base URLs, timeouts, and headers
    that are used when making API calls to the provider.
    
    Attributes:
        base_url: The API base URL for the provider
        timeout: Request timeout in seconds
        headers: Additional HTTP headers to include in requests
    """
    base_url: str
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class ResolvedModel:
    """
    A fully resolved model configuration ready for use.
    
    This represents the final result of the resolution process, combining
    model configuration, provider settings, and effective parameters.
    
    Attributes:
        config: The resolved model configuration
        provider_config: Provider-specific configuration
        effective_parameters: Final parameters after merging defaults and overrides
    """
    config: ModelConfig
    provider_config: ProviderConfig
    effective_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that all required components are present."""
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be a ModelConfig instance")
        if not isinstance(self.provider_config, ProviderConfig):
            raise ValueError("provider_config must be a ProviderConfig instance")