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
Configuration templates and generators for common use cases.

This module provides pre-built configuration templates and generators that make it
easy to get started with popular model configurations without writing YAML/JSON.
All templates are self-contained and do not depend on any static model registry.
"""

from typing import Dict, Any, List, Optional, NamedTuple
from .logging_utils import log_debug, log_info


class ValidationResult(NamedTuple):
    """Result of configuration validation."""
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)."""
        return len(self.errors) == 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if configuration has warnings."""
        return len(self.warnings) > 0


class ConfigurationValidator:
    """
    Configuration validator that checks structure, completeness, and provides helpful feedback.
    
    This class validates user configurations for required fields, correct structure,
    and provides warnings and suggestions for improving configuration completeness.
    """
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required top-level structure
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ValidationResult(errors, warnings, suggestions)
        
        # Check for required 'models' section
        if "models" not in config:
            errors.append("Missing required 'models' section in configuration")
            suggestions.append("Add a 'models' section with at least one model definition")
        elif not isinstance(config["models"], dict):
            errors.append("'models' section must be a dictionary")
        elif len(config["models"]) == 0:
            warnings.append("No models defined in configuration")
            suggestions.append("Add at least one model to the 'models' section")
        else:
            # Validate individual models
            model_errors, model_warnings, model_suggestions = ConfigurationValidator._validate_models(config["models"])
            errors.extend(model_errors)
            warnings.extend(model_warnings)
            suggestions.extend(model_suggestions)
        
        # Check providers section
        if "providers" in config:
            if not isinstance(config["providers"], dict):
                errors.append("'providers' section must be a dictionary")
            else:
                provider_errors, provider_warnings, provider_suggestions = ConfigurationValidator._validate_providers(config["providers"])
                errors.extend(provider_errors)
                warnings.extend(provider_warnings)
                suggestions.extend(provider_suggestions)
        else:
            warnings.append("No 'providers' section found - using default provider settings")
            suggestions.append("Add a 'providers' section to customize API endpoints and authentication")
        
        # Check profiles section
        if "profiles" in config:
            if not isinstance(config["profiles"], dict):
                errors.append("'profiles' section must be a dictionary")
            else:
                profile_errors, profile_warnings, profile_suggestions = ConfigurationValidator._validate_profiles(
                    config["profiles"], config.get("models", {})
                )
                errors.extend(profile_errors)
                warnings.extend(profile_warnings)
                suggestions.extend(profile_suggestions)
        
        # Check aliases section
        if "aliases" in config:
            if not isinstance(config["aliases"], dict):
                errors.append("'aliases' section must be a dictionary")
            else:
                alias_errors, alias_warnings, alias_suggestions = ConfigurationValidator._validate_aliases(
                    config["aliases"], config.get("models", {}), config.get("profiles", {})
                )
                errors.extend(alias_errors)
                warnings.extend(alias_warnings)
                suggestions.extend(alias_suggestions)
        
        # Check for intent resolution readiness
        intent_warnings, intent_suggestions = ConfigurationValidator._check_intent_resolution_readiness(config.get("models", {}))
        warnings.extend(intent_warnings)
        suggestions.extend(intent_suggestions)
        
        return ValidationResult(errors, warnings, suggestions)
    
    @staticmethod
    def _validate_models(models: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """Validate models section."""
        errors = []
        warnings = []
        suggestions = []
        
        for model_id, model_config in models.items():
            if not isinstance(model_config, dict):
                errors.append(f"Model '{model_id}' configuration must be a dictionary")
                continue
            
            # Check required fields
            if "provider" not in model_config:
                errors.append(f"Model '{model_id}' missing required 'provider' field")
            elif not model_config["provider"]:
                errors.append(f"Model '{model_id}' has empty 'provider' field")
            
            if "model" not in model_config:
                errors.append(f"Model '{model_id}' missing required 'model' field")
            elif not model_config["model"]:
                errors.append(f"Model '{model_id}' has empty 'model' field")
            
            # Check metadata for intent resolution
            if "metadata" not in model_config:
                warnings.append(f"Model '{model_id}' missing metadata - intent resolution may not work")
                suggestions.append(f"Add metadata to model '{model_id}' with fields like 'reasoning', 'cost', 'speed', 'capabilities'")
            elif not isinstance(model_config["metadata"], dict):
                errors.append(f"Model '{model_id}' metadata must be a dictionary")
            elif len(model_config["metadata"]) == 0:
                warnings.append(f"Model '{model_id}' has empty metadata - intent resolution may not work")
                suggestions.append(f"Add metadata fields to model '{model_id}' for better intent resolution")
            
            # Check parameters
            if "parameters" in model_config and not isinstance(model_config["parameters"], dict):
                errors.append(f"Model '{model_id}' parameters must be a dictionary")
        
        return errors, warnings, suggestions
    
    @staticmethod
    def _validate_providers(providers: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """Validate providers section."""
        errors = []
        warnings = []
        suggestions = []
        
        for provider_id, provider_config in providers.items():
            if not isinstance(provider_config, dict):
                errors.append(f"Provider '{provider_id}' configuration must be a dictionary")
                continue
            
            # Check required fields
            if "base_url" not in provider_config:
                errors.append(f"Provider '{provider_id}' missing required 'base_url' field")
            elif not provider_config["base_url"]:
                errors.append(f"Provider '{provider_id}' has empty 'base_url' field")
            
            # Check timeout
            if "timeout" in provider_config:
                timeout = provider_config["timeout"]
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    errors.append(f"Provider '{provider_id}' timeout must be a positive number")
            
            # Check headers
            if "headers" in provider_config and not isinstance(provider_config["headers"], dict):
                errors.append(f"Provider '{provider_id}' headers must be a dictionary")
        
        return errors, warnings, suggestions
    
    @staticmethod
    def _validate_profiles(profiles: Dict[str, Any], models: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """Validate profiles section."""
        errors = []
        warnings = []
        suggestions = []
        
        for profile_id, profile_config in profiles.items():
            if not isinstance(profile_config, dict):
                errors.append(f"Profile '{profile_id}' configuration must be a dictionary")
                continue
            
            # Check required model field
            if "model" not in profile_config:
                errors.append(f"Profile '{profile_id}' missing required 'model' field")
            elif not profile_config["model"]:
                errors.append(f"Profile '{profile_id}' has empty 'model' field")
            else:
                # Check if referenced model exists
                model_ref = profile_config["model"]
                if model_ref not in models:
                    warnings.append(f"Profile '{profile_id}' references unknown model '{model_ref}'")
                    suggestions.append(f"Add model '{model_ref}' to the models section or update profile '{profile_id}' to reference an existing model")
            
            # Check parameters
            if "parameters" in profile_config and not isinstance(profile_config["parameters"], dict):
                errors.append(f"Profile '{profile_id}' parameters must be a dictionary")
        
        return errors, warnings, suggestions
    
    @staticmethod
    def _validate_aliases(aliases: Dict[str, Any], models: Dict[str, Any], profiles: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """Validate aliases section."""
        errors = []
        warnings = []
        suggestions = []
        
        for alias_id, alias_target in aliases.items():
            if not isinstance(alias_target, str):
                errors.append(f"Alias '{alias_id}' target must be a string")
                continue
            
            if not alias_target:
                errors.append(f"Alias '{alias_id}' has empty target")
                continue
            
            # Check if target exists (could be model, profile, or another alias)
            if alias_target not in models and alias_target not in profiles and alias_target not in aliases:
                warnings.append(f"Alias '{alias_id}' references unknown target '{alias_target}'")
                suggestions.append(f"Ensure target '{alias_target}' exists in models, profiles, or aliases sections")
        
        return errors, warnings, suggestions
    
    @staticmethod
    def _check_intent_resolution_readiness(models: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Check if configuration is ready for intent resolution."""
        warnings = []
        suggestions = []
        
        if not models:
            return warnings, suggestions
        
        # Check for common metadata fields that enable intent resolution
        common_fields = ["reasoning", "cost", "speed", "capabilities"]
        models_with_metadata = 0
        field_coverage = {field: 0 for field in common_fields}
        
        for model_id, model_config in models.items():
            metadata = model_config.get("metadata", {})
            if metadata:
                models_with_metadata += 1
                for field in common_fields:
                    if field in metadata:
                        field_coverage[field] += 1
        
        # Warn if few models have metadata
        if models_with_metadata < len(models) * 0.5:
            warnings.append(f"Only {models_with_metadata}/{len(models)} models have metadata - intent resolution may be limited")
            suggestions.append("Add metadata to more models to improve intent resolution capabilities")
        
        # Suggest adding common fields if they're missing
        for field, count in field_coverage.items():
            if count == 0:
                suggestions.append(f"Consider adding '{field}' metadata to models for better intent resolution")
            elif count < models_with_metadata * 0.5:
                suggestions.append(f"Consider adding '{field}' metadata to more models for consistent intent resolution")
        
        return warnings, suggestions


class ConfigurationExamples:
    """
    Helper class for generating configuration examples for error messages and suggestions.
    
    This class provides methods to generate helpful configuration examples when
    models are not found or when users need guidance on adding models to their configuration.
    """
    
    @staticmethod
    def get_model_example(model_identifier: str, provider: str = None) -> str:
        """
        Generate a configuration example for a specific model.
        
        Args:
            model_identifier: The model identifier that was not found
            provider: Optional provider hint for better examples
            
        Returns:
            YAML configuration example as a string
        """
        # Try to infer provider from model identifier
        if not provider:
            if "gpt" in model_identifier.lower():
                provider = "openai"
            elif "claude" in model_identifier.lower():
                provider = "anthropic"
            elif any(name in model_identifier.lower() for name in ["llama", "mistral", "phi", "codellama"]):
                provider = "ollama"
        
        if provider == "openai":
            return f"""models:
  {model_identifier}:
    provider: openai
    model: {model_identifier}
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      capabilities: [chat, completion]
    parameters:
      temperature: 0.7
      max_tokens: 2048

providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${{OPENAI_API_KEY}}"
      Content-Type: application/json"""
        
        elif provider == "anthropic":
            return f"""models:
  {model_identifier}:
    provider: anthropic
    model: {model_identifier}
    metadata:
      reasoning: high
      cost: medium
      speed: medium
      capabilities: [chat, completion, vision]
    parameters:
      temperature: 0.7
      max_tokens: 4096

providers:
  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${{ANTHROPIC_API_KEY}}"
      Content-Type: application/json
      anthropic-version: "2023-06-01\""""
        
        elif provider == "ollama":
            return f"""models:
  {model_identifier}:
    provider: ollama
    model: {model_identifier}
    metadata:
      reasoning: medium
      cost: free
      speed: medium
      capabilities: [chat, completion]
      local: true
    parameters:
      temperature: 0.7
      num_predict: 2048

providers:
  ollama:
    base_url: http://localhost:11434
    timeout: 120
    headers:
      Content-Type: application/json"""
        
        else:
            # Generic example
            return f"""models:
  {model_identifier}:
    provider: your-provider
    model: {model_identifier}
    metadata:
      reasoning: medium
      cost: medium
      speed: medium
      capabilities: [chat, completion]
    parameters:
      temperature: 0.7
      max_tokens: 2048

providers:
  your-provider:
    base_url: https://api.your-provider.com/v1
    headers:
      Authorization: "Bearer ${{YOUR_API_KEY}}"
      Content-Type: application/json"""
    
    @staticmethod
    def get_quick_start_instructions() -> str:
        """
        Generate quick-start instructions when no configuration is loaded.
        
        Returns:
            Quick-start instructions as a string
        """
        return """No configuration loaded. To get started:

1. Generate a template:
   import model_compass as mc
   config = mc.ConfigurationTemplates.generate_openai_template()
   mc.configure(config_dict=config)

2. Or create a config file:
   mc.configure(config_path="config.yaml")

3. Or use a quick multi-provider setup:
   config = mc.ConfigurationTemplates.generate_multi_provider_template(
       openai_models=["gpt-4", "gpt-3.5-turbo"],
       anthropic_models=["claude-3-sonnet"],
       ollama_models=["llama3"]
   )
   mc.configure(config_dict=config)"""
    
    @staticmethod
    def get_intent_resolution_example(intent_criteria: Dict[str, str]) -> str:
        """
        Generate an example showing how to add models that match intent criteria.
        
        Args:
            intent_criteria: Dictionary of intent criteria that failed to match
            
        Returns:
            Configuration example for intent resolution
        """
        # Generate example models based on criteria
        example_models = []
        
        reasoning = intent_criteria.get("reasoning", "medium")
        cost = intent_criteria.get("cost", "medium")
        speed = intent_criteria.get("speed", "medium")
        capabilities = intent_criteria.get("capabilities", "chat").split(",")
        
        if reasoning == "high":
            if cost == "low":
                example_models.append(("claude-3-haiku", "anthropic"))
            else:
                example_models.append(("gpt-4", "openai"))
                example_models.append(("claude-3-sonnet", "anthropic"))
        elif reasoning == "medium":
            example_models.append(("gpt-3.5-turbo", "openai"))
            example_models.append(("llama3", "ollama"))
        
        if not example_models:
            example_models = [("gpt-4", "openai")]
        
        model_id, provider = example_models[0]
        
        capabilities_list = ", ".join(capabilities)
        
        return f"""# Add models with matching metadata for intent resolution:
models:
  {model_id}:
    provider: {provider}
    model: {model_id}
    metadata:
      reasoning: {reasoning}
      cost: {cost}
      speed: {speed}
      capabilities: [{capabilities_list}]
    parameters:
      temperature: 0.7
      max_tokens: 2048"""
    
    @staticmethod
    def get_provider_suggestions(available_models: List[str]) -> List[str]:
        """
        Generate suggestions based on available models.
        
        Args:
            available_models: List of available model identifiers
            
        Returns:
            List of helpful suggestions
        """
        suggestions = []
        
        if not available_models:
            suggestions.extend([
                "No models are configured. Use ConfigurationTemplates to generate a starter configuration",
                "Try: config = mc.ConfigurationTemplates.generate_openai_template()",
                "Or load from file: mc.configure(config_path='your-config.yaml')"
            ])
        else:
            suggestions.extend([
                f"Use one of the available models: {', '.join(available_models[:5])}",
                "Check your model identifier spelling and case",
                "Verify the model exists in your configuration file"
            ])
            
            if len(available_models) > 5:
                suggestions.append(f"See all {len(available_models)} available models with mc.list_models()")
        
        return suggestions


class ConfigurationTemplates:
    """
    Configuration template generators for common providers and use cases.
    
    This class provides static methods to generate complete configuration
    dictionaries for popular model providers without depending on any
    static model registry.
    """
    
    @staticmethod
    def generate_openai_template(
        models: Optional[List[str]] = None,
        include_profiles: bool = True,
        default_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a configuration template for OpenAI models.
        
        Args:
            models: List of OpenAI model IDs to include (defaults to common models)
            include_profiles: Whether to include logical profiles
            default_profile: Name of the default profile
            
        Returns:
            Configuration dictionary ready for use with configure()
        """
        if models is None:
            models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
        
        # OpenAI model definitions with comprehensive metadata
        openai_models = {
            "gpt-4": {
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {
                    "reasoning": "high",
                    "cost": "high", 
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "function_calling"],
                    "context_length": 8192,
                    "multimodal": False
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            "gpt-4-turbo": {
                "provider": "openai",
                "model": "gpt-4-turbo",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "function_calling", "vision"],
                    "context_length": 128000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            "gpt-4o": {
                "provider": "openai",
                "model": "gpt-4o",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "function_calling", "vision"],
                    "context_length": 128000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "function_calling"],
                    "context_length": 16385,
                    "multimodal": False
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            }
        }
        
        config = {
            "models": {},
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 30,
                    "headers": {
                        "Authorization": "Bearer ${OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    }
                }
            }
        }
        
        # Add requested models
        for model_id in models:
            if model_id in openai_models:
                config["models"][model_id] = openai_models[model_id]
        
        # Add profiles if requested
        if include_profiles:
            config["profiles"] = {
                "fast": {
                    "model": "gpt-3.5-turbo",
                    "parameters": {"temperature": 0.3, "max_tokens": 1024}
                },
                "balanced": {
                    "model": "gpt-4",
                    "parameters": {"temperature": 0.7, "max_tokens": 2048}
                },
                "creative": {
                    "model": "gpt-4",
                    "parameters": {"temperature": 0.9, "max_tokens": 2048}
                }
            }
            
            if default_profile:
                config["default_profile"] = default_profile
            elif "gpt-3.5-turbo" in models:
                config["default_profile"] = "fast"
        
        log_debug(f"Generated OpenAI config with {len(config['models'])} models")
        return config


    @staticmethod
    def generate_anthropic_template(
        models: Optional[List[str]] = None,
        include_profiles: bool = True,
        default_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a configuration template for Anthropic Claude models.
        
        Args:
            models: List of Anthropic model IDs to include (defaults to Claude 3 models)
            include_profiles: Whether to include logical profiles
            default_profile: Name of the default profile
            
        Returns:
            Configuration dictionary ready for use with configure()
        """
        if models is None:
            models = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        
        # Anthropic model definitions with comprehensive metadata
        anthropic_models = {
            "claude-3-haiku-20240307": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            "claude-3-sonnet-20240229": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            "claude-3-opus-20240229": {
                "provider": "anthropic",
                "model": "claude-3-opus-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "high",
                    "speed": "slow",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            # Convenience aliases for easier reference
            "claude-3-haiku": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            "claude-3-opus": {
                "provider": "anthropic",
                "model": "claude-3-opus-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "high",
                    "speed": "slow",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            }
        }
        
        config = {
            "models": {},
            "providers": {
                "anthropic": {
                    "base_url": "https://api.anthropic.com",
                    "timeout": 60,
                    "headers": {
                        "x-api-key": "${ANTHROPIC_API_KEY}",
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01"
                    }
                }
            }
        }
        
        # Add requested models
        for model_id in models:
            if model_id in anthropic_models:
                config["models"][model_id] = anthropic_models[model_id]
        
        # Add profiles if requested
        if include_profiles:
            config["profiles"] = {
                "fast": {
                    "model": "claude-3-haiku-20240307" if "claude-3-haiku-20240307" in config["models"] else "claude-3-haiku",
                    "parameters": {"temperature": 0.3, "max_tokens": 1024}
                },
                "balanced": {
                    "model": "claude-3-sonnet-20240229" if "claude-3-sonnet-20240229" in config["models"] else "claude-3-sonnet", 
                    "parameters": {"temperature": 0.7, "max_tokens": 2048}
                },
                "powerful": {
                    "model": "claude-3-opus-20240229" if "claude-3-opus-20240229" in config["models"] else "claude-3-opus",
                    "parameters": {"temperature": 0.7, "max_tokens": 4096}
                }
            }
            
            if default_profile:
                config["default_profile"] = default_profile
            elif any(model in models for model in ["claude-3-haiku", "claude-3-haiku-20240307"]):
                config["default_profile"] = "fast"
        
        log_debug(f"Generated Anthropic config with {len(config['models'])} models")
        return config


    @staticmethod
    def generate_ollama_template(
        models: Optional[List[str]] = None,
        include_profiles: bool = True,
        default_profile: Optional[str] = None,
        base_url: str = "http://localhost:11434"
    ) -> Dict[str, Any]:
        """
        Generate a configuration template for Ollama local models.
        
        Args:
            models: List of Ollama model IDs to include (defaults to common models)
            include_profiles: Whether to include logical profiles
            default_profile: Name of the default profile
            base_url: Ollama server URL
            
        Returns:
            Configuration dictionary ready for use with configure()
        """
        if models is None:
            models = ["llama3", "mistral", "codellama", "phi3"]
        
        # Ollama model definitions with comprehensive metadata
        ollama_models = {
            "llama3": {
                "provider": "ollama",
                "model": "llama3",
                "metadata": {
                    "reasoning": "high",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["chat", "completion"],
                    "context_length": 8192,
                    "multimodal": False,
                    "local": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 2048}
            },
            "llama3:70b": {
                "provider": "ollama",
                "model": "llama3:70b",
                "metadata": {
                    "reasoning": "high",
                    "cost": "free",
                    "speed": "slow",
                    "capabilities": ["chat", "completion"],
                    "context_length": 8192,
                    "multimodal": False,
                    "local": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 2048}
            },
            "mistral": {
                "provider": "ollama",
                "model": "mistral",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "fast",
                    "capabilities": ["chat", "completion"],
                    "context_length": 8192,
                    "multimodal": False,
                    "local": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 2048}
            },
            "codellama": {
                "provider": "ollama",
                "model": "codellama",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "code"],
                    "context_length": 16384,
                    "multimodal": False,
                    "local": True
                },
                "parameters": {"temperature": 0.1, "num_predict": 2048}
            },
            "phi3": {
                "provider": "ollama",
                "model": "phi3",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "fast",
                    "capabilities": ["chat", "completion"],
                    "context_length": 4096,
                    "multimodal": False,
                    "local": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 1024}
            },
            "llava": {
                "provider": "ollama",
                "model": "llava",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 4096,
                    "multimodal": True,
                    "local": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 1024}
            }
        }
        
        config = {
            "models": {},
            "providers": {
                "ollama": {
                    "base_url": base_url,
                    "timeout": 120,
                    "headers": {"Content-Type": "application/json"}
                }
            }
        }
        
        # Add requested models
        for model_id in models:
            if model_id in ollama_models:
                config["models"][model_id] = ollama_models[model_id]
        
        # Add profiles if requested
        if include_profiles:
            profiles = {
                "local": {
                    "model": "llama3",
                    "parameters": {"temperature": 0.7, "num_predict": 1024}
                },
                "code": {
                    "model": "codellama",
                    "parameters": {"temperature": 0.1, "num_predict": 2048}
                },
                "fast": {
                    "model": "phi3",
                    "parameters": {"temperature": 0.7, "num_predict": 1024}
                }
            }
            
            # Only add vision profile if llava model is included
            if "llava" in config["models"]:
                profiles["vision"] = {
                    "model": "llava",
                    "parameters": {"temperature": 0.7, "num_predict": 1024}
                }
            
            config["profiles"] = profiles
            
            if default_profile:
                config["default_profile"] = default_profile
            elif "llama3" in models:
                config["default_profile"] = "local"
        
        log_debug(f"Generated Ollama config with {len(config['models'])} models")
        return config


    @staticmethod
    def generate_multi_provider_template(
        openai_models: Optional[List[str]] = None,
        anthropic_models: Optional[List[str]] = None,
        ollama_models: Optional[List[str]] = None,
        include_profiles: bool = True,
        default_profile: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive configuration template that includes multiple providers.
        
        Args:
            openai_models: OpenAI models to include
            anthropic_models: Anthropic models to include
            ollama_models: Ollama models to include
            include_profiles: Whether to include logical profiles
            default_profile: Name of the default profile
            
        Returns:
            Configuration dictionary with multiple providers
        """
        config = {
            "models": {},
            "providers": {}
        }
        
        # Add OpenAI if requested
        if openai_models:
            openai_config = ConfigurationTemplates.generate_openai_template(openai_models, False)
            config["models"].update(openai_config["models"])
            config["providers"].update(openai_config["providers"])
        
        # Add Anthropic if requested
        if anthropic_models:
            anthropic_config = ConfigurationTemplates.generate_anthropic_template(anthropic_models, False)
            config["models"].update(anthropic_config["models"])
            config["providers"].update(anthropic_config["providers"])
        
        # Add Ollama if requested
        if ollama_models:
            ollama_config = ConfigurationTemplates.generate_ollama_template(ollama_models, False)
            config["models"].update(ollama_config["models"])
            config["providers"].update(ollama_config["providers"])
        
        # Add cross-provider profiles
        if include_profiles:
            profiles = {}
            
            # Fast profile - prefer local or cheap cloud
            if ollama_models and "phi3" in config["models"]:
                profiles["fast"] = {"model": "phi3"}
            elif "gpt-3.5-turbo" in config["models"]:
                profiles["fast"] = {"model": "gpt-3.5-turbo"}
            elif "claude-3-haiku" in config["models"]:
                profiles["fast"] = {"model": "claude-3-haiku"}
            
            # Balanced profile - good performance/cost ratio
            if "gpt-4" in config["models"]:
                profiles["balanced"] = {"model": "gpt-4"}
            elif "claude-3-sonnet" in config["models"]:
                profiles["balanced"] = {"model": "claude-3-sonnet"}
            elif ollama_models and "llama3" in config["models"]:
                profiles["balanced"] = {"model": "llama3"}
            
            # Powerful profile - best reasoning
            if "claude-3-opus" in config["models"]:
                profiles["powerful"] = {"model": "claude-3-opus"}
            elif "gpt-4-turbo" in config["models"]:
                profiles["powerful"] = {"model": "gpt-4-turbo"}
            elif ollama_models and "llama3:70b" in config["models"]:
                profiles["powerful"] = {"model": "llama3:70b"}
            
            # Local profile - prefer local models
            if ollama_models and "llama3" in config["models"]:
                profiles["local"] = {"model": "llama3"}
            
            # Code profile - prefer code-specialized models
            if ollama_models and "codellama" in config["models"]:
                profiles["code"] = {"model": "codellama"}
            elif "gpt-4" in config["models"]:
                profiles["code"] = {
                    "model": "gpt-4",
                    "parameters": {"temperature": 0.1}
                }
            
            # Vision profile - prefer multimodal models (only add if model exists)
            if "gpt-4o" in config["models"]:
                profiles["vision"] = {"model": "gpt-4o"}
            elif "claude-3-sonnet" in config["models"]:
                profiles["vision"] = {"model": "claude-3-sonnet"}
            elif ollama_models and "llava" in config["models"]:
                profiles["vision"] = {"model": "llava"}
            
            config["profiles"] = profiles
            
            # Only set default profile if we have profiles and the requested one exists
            if default_profile in profiles:
                config["default_profile"] = default_profile
            elif profiles:
                # Set to first available profile
                config["default_profile"] = list(profiles.keys())[0]
            
            # Add aliases for common patterns
            aliases = {}
            if profiles:
                main_profile = config.get("default_profile", list(profiles.keys())[0])
                aliases.update({
                    "main": main_profile,
                    "default": main_profile,
                })
                if "balanced" in profiles:
                    aliases["prod"] = "balanced"
                if "fast" in profiles:
                    aliases["dev"] = "fast"
                elif profiles:
                    aliases["dev"] = list(profiles.keys())[0]
            
            if aliases:
                config["aliases"] = aliases
        
        log_info(f"Generated multi-provider config", {
            "models": len(config["models"]),
            "providers": len(config["providers"]),
            "profiles": len(config.get("profiles", {}))
        })
        
        return config



# Backward compatibility functions that use the new ConfigurationTemplates class
def generate_openai_config(
    models: Optional[List[str]] = None,
    include_profiles: bool = True,
    default_profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a configuration for OpenAI models.
    
    DEPRECATED: Use ConfigurationTemplates.generate_openai_template() instead.
    """
    return ConfigurationTemplates.generate_openai_template(models, include_profiles, default_profile)


def generate_anthropic_config(
    models: Optional[List[str]] = None,
    include_profiles: bool = True,
    default_profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a configuration for Anthropic models.
    
    DEPRECATED: Use ConfigurationTemplates.generate_anthropic_template() instead.
    """
    return ConfigurationTemplates.generate_anthropic_template(models, include_profiles, default_profile)


def generate_ollama_config(
    models: Optional[List[str]] = None,
    include_profiles: bool = True,
    default_profile: Optional[str] = None,
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Generate a configuration for Ollama local models.
    
    DEPRECATED: Use ConfigurationTemplates.generate_ollama_template() instead.
    """
    return ConfigurationTemplates.generate_ollama_template(models, include_profiles, default_profile, base_url)


def generate_multi_provider_config(
    openai_models: Optional[List[str]] = None,
    anthropic_models: Optional[List[str]] = None,
    ollama_models: Optional[List[str]] = None,
    include_profiles: bool = True,
    default_profile: str = "balanced"
) -> Dict[str, Any]:
    """
    Generate a configuration that includes multiple providers.
    
    DEPRECATED: Use ConfigurationTemplates.generate_multi_provider_template() instead.
    """
    return ConfigurationTemplates.generate_multi_provider_template(
        openai_models, anthropic_models, ollama_models, include_profiles, default_profile
    )


def generate_development_config() -> Dict[str, Any]:
    """
    Generate a development-friendly configuration with local and cloud options.
    
    Returns:
        Configuration optimized for development workflows
    """
    return ConfigurationTemplates.generate_multi_provider_template(
        openai_models=["gpt-3.5-turbo", "gpt-4"],
        anthropic_models=["claude-3-haiku"],
        ollama_models=["llama3", "codellama"],
        include_profiles=True,
        default_profile="fast"
    )


def generate_production_config() -> Dict[str, Any]:
    """
    Generate a production-ready configuration with reliable cloud models.
    
    Returns:
        Configuration optimized for production use
    """
    return ConfigurationTemplates.generate_multi_provider_template(
        openai_models=["gpt-4", "gpt-3.5-turbo"],
        anthropic_models=["claude-3-sonnet", "claude-3-haiku"],
        include_profiles=True,
        default_profile="balanced"
    )