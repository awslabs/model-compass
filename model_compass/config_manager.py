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
Configuration manager for loading and validating configuration files.

This module handles loading configuration from YAML/JSON files and validating
the configuration schema for models, profiles, aliases, and providers.
"""

import json
import yaml
from typing import Dict, Any, Optional, Set, List
from pathlib import Path

from .exceptions import ConfigurationError
from .data_models import ModelConfig, Profile, ProviderConfig
from .logging_utils import (
    log_debug, log_info, log_warning, log_configuration_load, 
    log_configuration_validation, create_error_context
)
from .caching import get_config_cache


class ConfigurationManager:
    """
    Manages loading and validation of configuration files.
    
    Supports both YAML and JSON configuration formats with comprehensive
    validation for models, profiles, aliases, and provider configurations.
    This manager works exclusively with user-provided configuration and does
    not fall back to any static registry.
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration manager with user configuration.
        
        Args:
            config_path: Path to configuration file
            config_dict: Configuration dictionary
            
        Raises:
            ConfigurationError: If no configuration is provided
        """
        self._config = {}
        self._models = {}
        self._profiles = {}
        self._aliases = {}
        self._providers = {}
        self._default_profile = None
        self._resolution_timeout = 30
        self._is_configured = False
        
        # Require configuration during initialization
        if config_path is None and config_dict is None:
            from .config_templates import ConfigurationExamples
            raise ConfigurationError(
                "No configuration provided. Configuration is required for Model Compass to function.",
                context={
                    "quick_start": ConfigurationExamples.get_quick_start_instructions(),
                    "suggestions": [
                        "Provide config_path parameter with path to your configuration file",
                        "Provide config_dict parameter with configuration dictionary",
                        "Use ConfigurationTemplates to generate a starter configuration"
                    ]
                }
            )
        
        # Load configuration
        if config_path is not None:
            self.load_from_file(config_path)
        elif config_dict is not None:
            self.load_from_dict(config_dict)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If file cannot be loaded or is invalid
        """
        log_configuration_load(config_path=config_path)
        
        # Try to get from cache first
        cache = get_config_cache()
        cached_config = cache.get(config_path)
        if cached_config is not None:
            log_debug(f"Using cached configuration for: {config_path}")
            self.load_from_dict(cached_config)
            return
        
        config_file = Path(config_path)
        
        # Check if file exists
        if not config_file.exists():
            error_context = create_error_context(
                config_path=config_path,
                suggestions=[
                    f"Create the configuration file at: {config_path}",
                    "Check if the file path is correct",
                    "Ensure you have read permissions for the file"
                ]
            )
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                config_path=config_path,
                context=error_context
            )
        
        # Check if file is readable
        if not config_file.is_file():
            error_context = create_error_context(
                config_path=config_path,
                suggestions=[
                    "Ensure the path points to a file, not a directory",
                    "Check file permissions",
                    "Verify the file exists and is accessible"
                ]
            )
            raise ConfigurationError(
                f"Configuration path is not a file: {config_path}",
                config_path=config_path,
                context=error_context
            )
        
        try:
            # Read file content
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse based on file extension
            file_extension = config_file.suffix.lower()
            
            if file_extension in ['.yaml', '.yml']:
                log_debug(f"Parsing YAML configuration file: {config_path}")
                try:
                    config_dict = yaml.safe_load(content)
                except yaml.YAMLError as e:
                    error_context = create_error_context(
                        config_path=config_path,
                        yaml_error=str(e),
                        suggestions=[
                            "Check YAML syntax - ensure proper indentation",
                            "Validate YAML structure using an online YAML validator",
                            "Check for missing colons, quotes, or brackets",
                            "Ensure consistent indentation (spaces vs tabs)"
                        ]
                    )
                    raise ConfigurationError(
                        f"Invalid YAML syntax: {str(e)}",
                        config_path=config_path,
                        context=error_context
                    )
            elif file_extension == '.json':
                log_debug(f"Parsing JSON configuration file: {config_path}")
                try:
                    config_dict = json.loads(content)
                except json.JSONDecodeError as e:
                    error_context = create_error_context(
                        config_path=config_path,
                        json_error=str(e),
                        suggestions=[
                            "Check JSON syntax - ensure proper brackets and quotes",
                            "Validate JSON structure using an online JSON validator",
                            "Check for trailing commas (not allowed in JSON)",
                            "Ensure all strings are properly quoted"
                        ]
                    )
                    raise ConfigurationError(
                        f"Invalid JSON syntax: {str(e)}",
                        config_path=config_path,
                        context=error_context
                    )
            else:
                error_context = create_error_context(
                    config_path=config_path,
                    file_extension=file_extension,
                    suggestions=[
                        "Use .yaml or .yml for YAML configuration files",
                        "Use .json for JSON configuration files",
                        "Rename your file with the correct extension"
                    ]
                )
                raise ConfigurationError(
                    f"Unsupported file format: {file_extension}. Use .yaml, .yml, or .json",
                    config_path=config_path,
                    context=error_context
                )
            
            # Handle empty files
            if config_dict is None:
                config_dict = {}
            
            # Load and validate the configuration
            self.load_from_dict(config_dict)
            
            # Cache the successfully parsed configuration
            cache.put(config_path, config_dict)
            
            # Mark as configured
            self._is_configured = True
            
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read configuration file: {str(e)}",
                config_path=config_path,
                context={"error": str(e)}
            )
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        log_configuration_load(config_dict=config_dict)
        
        if not isinstance(config_dict, dict):
            error_context = create_error_context(
                type_received=type(config_dict).__name__,
                suggestions=[
                    "Ensure config_dict is a Python dictionary",
                    "Check that you're not passing a string or other type",
                    "Verify the configuration structure is correct"
                ]
            )
            raise ConfigurationError(
                "Configuration must be a dictionary",
                context=error_context
            )
        
        # Store raw configuration
        self._config = config_dict.copy()
        log_debug(f"Loading configuration with sections: {list(config_dict.keys())}")
        
        # Validate and load each section
        self._validate_and_load_global_settings(config_dict)
        self._validate_and_load_providers(config_dict.get('providers', {}))
        self._validate_and_load_models(config_dict.get('models', {}))
        self._validate_and_load_profiles(config_dict.get('profiles', {}))
        self._validate_and_load_aliases(config_dict.get('aliases', {}))
        
        # Validate cross-references
        self._validate_cross_references()
        
        # Run comprehensive validation and provide feedback
        self._run_comprehensive_validation()
        
        # Mark as configured
        self._is_configured = True
        
        log_info("Configuration loaded successfully", {
            "models": len(self._models),
            "profiles": len(self._profiles),
            "aliases": len(self._aliases),
            "providers": len(self._providers)
        })
    
    def _validate_and_load_global_settings(self, config_dict: Dict[str, Any]) -> None:
        """Validate and load global configuration settings."""
        # Load default profile
        if 'default_profile' in config_dict:
            default_profile = config_dict['default_profile']
            if not isinstance(default_profile, str):
                raise ConfigurationError(
                    "default_profile must be a string",
                    context={"type": type(default_profile).__name__}
                )
            self._default_profile = default_profile
        
        # Load resolution timeout
        if 'resolution_timeout' in config_dict:
            timeout = config_dict['resolution_timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ConfigurationError(
                    "resolution_timeout must be a positive number",
                    context={"value": timeout}
                )
            self._resolution_timeout = int(timeout)
    
    def _validate_and_load_providers(self, providers_config: Dict[str, Any]) -> None:
        """Validate and load provider configurations."""
        log_debug(f"Validating providers section with {len(providers_config)} providers")
        
        if not isinstance(providers_config, dict):
            error_context = create_error_context(
                section="providers",
                type_received=type(providers_config).__name__,
                suggestions=[
                    "Ensure 'providers' section is a dictionary/object",
                    "Check YAML/JSON syntax for the providers section",
                    "Example: providers: { openai: { base_url: '...' } }"
                ]
            )
            raise ConfigurationError(
                "providers section must be a dictionary",
                context=error_context
            )
        
        self._providers = {}
        validation_errors = []
        
        for provider_name, provider_config in providers_config.items():
            if not isinstance(provider_name, str):
                raise ConfigurationError(
                    "Provider names must be strings",
                    context={"provider_name": provider_name}
                )
            
            if not isinstance(provider_config, dict):
                raise ConfigurationError(
                    f"Provider '{provider_name}' configuration must be a dictionary",
                    context={"provider": provider_name, "type": type(provider_config).__name__}
                )
            
            # Validate required fields
            if 'base_url' not in provider_config:
                raise ConfigurationError(
                    f"Provider '{provider_name}' missing required field 'base_url'",
                    context={"provider": provider_name}
                )
            
            base_url = provider_config['base_url']
            if not isinstance(base_url, str) or not base_url.strip():
                raise ConfigurationError(
                    f"Provider '{provider_name}' base_url must be a non-empty string",
                    context={"provider": provider_name, "base_url": base_url}
                )
            
            # Validate optional fields
            timeout = provider_config.get('timeout', 30)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ConfigurationError(
                    f"Provider '{provider_name}' timeout must be a positive number",
                    context={"provider": provider_name, "timeout": timeout}
                )
            
            headers = provider_config.get('headers', {})
            if not isinstance(headers, dict):
                raise ConfigurationError(
                    f"Provider '{provider_name}' headers must be a dictionary",
                    context={"provider": provider_name, "type": type(headers).__name__}
                )
            
            # Validate header values are strings
            for header_name, header_value in headers.items():
                if not isinstance(header_name, str) or not isinstance(header_value, str):
                    raise ConfigurationError(
                        f"Provider '{provider_name}' header names and values must be strings",
                        context={"provider": provider_name, "header": header_name}
                    )
            
            # Create ProviderConfig instance
            try:
                self._providers[provider_name] = ProviderConfig(
                    base_url=base_url.strip(),
                    timeout=int(timeout),
                    headers=headers
                )
                log_debug(f"Loaded provider: {provider_name}")
            except ValueError as e:
                error_msg = f"Invalid provider configuration for '{provider_name}': {str(e)}"
                validation_errors.append(error_msg)
                raise ConfigurationError(
                    error_msg,
                    context=create_error_context(
                        provider=provider_name,
                        suggestions=[
                            "Check provider configuration syntax",
                            "Ensure base_url is a valid URL",
                            "Verify timeout is a positive number",
                            "Check that headers is a dictionary"
                        ]
                    )
                )
        
        log_configuration_validation("providers", len(self._providers), validation_errors)
    
    def _validate_and_load_models(self, models_config: Dict[str, Any]) -> None:
        """Validate and load model configurations."""
        if not isinstance(models_config, dict):
            raise ConfigurationError(
                "models section must be a dictionary",
                context={"type": type(models_config).__name__}
            )
        
        self._models = {}
        
        for model_id, model_config in models_config.items():
            if not isinstance(model_id, str):
                raise ConfigurationError(
                    "Model IDs must be strings",
                    context={"model_id": model_id}
                )
            
            if not isinstance(model_config, dict):
                raise ConfigurationError(
                    f"Model '{model_id}' configuration must be a dictionary",
                    context={"model": model_id, "type": type(model_config).__name__}
                )
            
            # Validate required fields
            if 'provider' not in model_config:
                raise ConfigurationError(
                    f"Model '{model_id}' missing required field 'provider'",
                    context={"model": model_id}
                )
            
            if 'model' not in model_config:
                raise ConfigurationError(
                    f"Model '{model_id}' missing required field 'model'",
                    context={"model": model_id}
                )
            
            provider = model_config['provider']
            model_name = model_config['model']
            
            if not isinstance(provider, str) or not provider.strip():
                raise ConfigurationError(
                    f"Model '{model_id}' provider must be a non-empty string",
                    context={"model": model_id, "provider": provider}
                )
            
            if not isinstance(model_name, str) or not model_name.strip():
                raise ConfigurationError(
                    f"Model '{model_id}' model name must be a non-empty string",
                    context={"model": model_id, "model_name": model_name}
                )
            
            # Validate optional fields
            metadata = model_config.get('metadata', {})
            if not isinstance(metadata, dict):
                raise ConfigurationError(
                    f"Model '{model_id}' metadata must be a dictionary",
                    context={"model": model_id, "type": type(metadata).__name__}
                )
            
            parameters = model_config.get('parameters', {})
            if not isinstance(parameters, dict):
                raise ConfigurationError(
                    f"Model '{model_id}' parameters must be a dictionary",
                    context={"model": model_id, "type": type(parameters).__name__}
                )
            
            # Create ModelConfig instance
            try:
                self._models[model_id] = ModelConfig(
                    provider=provider.strip(),
                    model=model_name.strip(),
                    metadata=metadata,
                    parameters=parameters
                )
            except ValueError as e:
                raise ConfigurationError(
                    f"Invalid model configuration for '{model_id}': {str(e)}",
                    context={"model": model_id}
                )
    
    def _validate_and_load_profiles(self, profiles_config: Dict[str, Any]) -> None:
        """Validate and load profile configurations."""
        if not isinstance(profiles_config, dict):
            raise ConfigurationError(
                "profiles section must be a dictionary",
                context={"type": type(profiles_config).__name__}
            )
        
        self._profiles = {}
        
        for profile_name, profile_config in profiles_config.items():
            if not isinstance(profile_name, str):
                raise ConfigurationError(
                    "Profile names must be strings",
                    context={"profile_name": profile_name}
                )
            
            if not isinstance(profile_config, dict):
                raise ConfigurationError(
                    f"Profile '{profile_name}' configuration must be a dictionary",
                    context={"profile": profile_name, "type": type(profile_config).__name__}
                )
            
            # Validate required fields
            if 'model' not in profile_config:
                raise ConfigurationError(
                    f"Profile '{profile_name}' missing required field 'model'",
                    context={"profile": profile_name}
                )
            
            model_ref = profile_config['model']
            if not isinstance(model_ref, str) or not model_ref.strip():
                raise ConfigurationError(
                    f"Profile '{profile_name}' model reference must be a non-empty string",
                    context={"profile": profile_name, "model_ref": model_ref}
                )
            
            # Validate optional fields
            parameters = profile_config.get('parameters', {})
            if not isinstance(parameters, dict):
                raise ConfigurationError(
                    f"Profile '{profile_name}' parameters must be a dictionary",
                    context={"profile": profile_name, "type": type(parameters).__name__}
                )
            
            # Create Profile instance
            try:
                self._profiles[profile_name] = Profile(
                    model=model_ref.strip(),
                    parameters=parameters
                )
            except ValueError as e:
                raise ConfigurationError(
                    f"Invalid profile configuration for '{profile_name}': {str(e)}",
                    context={"profile": profile_name}
                )
    
    def _validate_and_load_aliases(self, aliases_config: Dict[str, Any]) -> None:
        """Validate and load alias configurations."""
        if not isinstance(aliases_config, dict):
            raise ConfigurationError(
                "aliases section must be a dictionary",
                context={"type": type(aliases_config).__name__}
            )
        
        self._aliases = {}
        
        for alias_name, alias_target in aliases_config.items():
            if not isinstance(alias_name, str):
                raise ConfigurationError(
                    "Alias names must be strings",
                    context={"alias_name": alias_name}
                )
            
            if not isinstance(alias_target, str) or not alias_target.strip():
                raise ConfigurationError(
                    f"Alias '{alias_name}' target must be a non-empty string",
                    context={"alias": alias_name, "target": alias_target}
                )
            
            self._aliases[alias_name] = alias_target.strip()
    
    def _validate_cross_references(self) -> None:
        """Validate cross-references between different configuration sections."""
        # Validate that model providers exist in provider configurations
        for model_id, model_config in self._models.items():
            provider = model_config.provider
            if provider not in self._providers:
                raise ConfigurationError(
                    f"Model '{model_id}' references unknown provider '{provider}'",
                    context={"model": model_id, "provider": provider}
                )
        
        # Validate that default profile exists
        if self._default_profile and self._default_profile not in self._profiles:
            raise ConfigurationError(
                f"Default profile '{self._default_profile}' not found in profiles",
                context={"default_profile": self._default_profile}
            )
        
        # Validate profile model references (basic check - detailed resolution happens later)
        for profile_name, profile in self._profiles.items():
            model_ref = profile.model
            # Check if it's a direct model reference
            if model_ref not in self._models and model_ref not in self._aliases:
                # Could be a physical identifier (provider/model) - we'll validate this during resolution
                if '/' not in model_ref:
                    raise ConfigurationError(
                        f"Profile '{profile_name}' references unknown model '{model_ref}'",
                        context={"profile": profile_name, "model_ref": model_ref}
                    )
        
        # Basic alias validation (detailed circular detection happens during resolution)
        for alias_name, alias_target in self._aliases.items():
            # Check if target exists in any of the valid reference types
            if (alias_target not in self._models and 
                alias_target not in self._profiles and 
                alias_target not in self._aliases and
                '/' not in alias_target):  # Not a physical identifier
                raise ConfigurationError(
                    f"Alias '{alias_name}' references unknown target '{alias_target}'",
                    context={"alias": alias_name, "target": alias_target}
                )
    
    def _run_comprehensive_validation(self) -> None:
        """
        Run comprehensive validation using ConfigurationValidator and log warnings/suggestions.
        """
        from .config_templates import ConfigurationValidator
        
        validation_result = ConfigurationValidator.validate(self._config)
        
        # Log warnings
        for warning in validation_result.warnings:
            log_warning(f"Configuration warning: {warning}")
        
        # Log suggestions as info
        for suggestion in validation_result.suggestions:
            log_info(f"Configuration suggestion: {suggestion}")
        
        # Store validation metrics for performance tracking
        self._validation_metrics = {
            'warnings_count': len(validation_result.warnings),
            'suggestions_count': len(validation_result.suggestions),
            'has_intent_metadata': any(
                model_config.get('metadata', {}) 
                for model_config in self._config.get('models', {}).values()
            )
        }
    
    def has_profile(self, profile_name: str) -> bool:
        """
        Check if a profile exists in the configuration.
        
        Args:
            profile_name: Name of the profile to check
            
        Returns:
            True if profile exists, False otherwise
        """
        return profile_name in self._profiles
    
    def has_model(self, model_id: str) -> bool:
        """
        Check if a model exists in the configuration.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        return model_id in self._models
    
    def has_alias(self, alias_name: str) -> bool:
        """
        Check if an alias exists in the configuration.
        
        Args:
            alias_name: Name of the alias to check
            
        Returns:
            True if alias exists, False otherwise
        """
        return alias_name in self._aliases
    
    def has_provider(self, provider_name: str) -> bool:
        """
        Check if a provider exists in the configuration.
        
        Args:
            provider_name: Name of the provider to check
            
        Returns:
            True if provider exists, False otherwise
        """
        return provider_name in self._providers
    
    def get_models(self) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        return self._models.copy()
    
    def get_profiles(self) -> Dict[str, Profile]:
        """Get all profile configurations."""
        return self._profiles.copy()
    
    def get_aliases(self) -> Dict[str, str]:
        """Get all alias configurations."""
        return self._aliases.copy()
    
    def get_providers(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations."""
        return self._providers.copy()
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration from user configuration only.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            ModelConfig if found, None otherwise
            
        Raises:
            ConfigurationError: If no configuration is loaded
        """
        if not self._is_configured:
            from .config_templates import ConfigurationExamples
            raise ConfigurationError(
                "No configuration loaded. Cannot retrieve model without configuration.",
                context={
                    "model_id": model_id,
                    "quick_start": ConfigurationExamples.get_quick_start_instructions(),
                    "suggestions": [
                        "Load configuration using ConfigurationManager(config_path='...')",
                        "Or use ConfigurationManager(config_dict={...})",
                        "Generate a template with ConfigurationTemplates"
                    ]
                }
            )
        
        return self._models.get(model_id)
    
    def get_profile(self, profile_name: str) -> Optional[Profile]:
        """Get a specific profile configuration."""
        return self._profiles.get(profile_name)
    
    def get_alias(self, alias_name: str) -> Optional[str]:
        """Get a specific alias target."""
        return self._aliases.get(alias_name)
    
    def get_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get a specific provider configuration."""
        return self._providers.get(provider_name)
    
    def get_default_profile(self) -> Optional[str]:
        """Get the default profile name."""
        return self._default_profile
    
    def get_resolution_timeout(self) -> int:
        """Get the resolution timeout in seconds."""
        return self._resolution_timeout
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config.copy()
    
    def is_configured(self) -> bool:
        """
        Check if configuration has been loaded.
        
        Returns:
            True if configuration is loaded, False otherwise
        """
        return self._is_configured
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model identifiers from user configuration.
        
        Returns:
            List of model IDs available in the current configuration
        """
        return list(self._models.keys())
    
    def get_available_profiles(self) -> List[str]:
        """
        Get list of available profile names from user configuration.
        
        Returns:
            List of profile names available in the current configuration
        """
        return list(self._profiles.keys())
    
    def get_available_aliases(self) -> List[str]:
        """
        Get list of available alias names from user configuration.
        
        Returns:
            List of alias names available in the current configuration
        """
        return list(self._aliases.keys())
    
    def validate_configuration_completeness(self) -> 'ValidationResult':
        """
        Validate the current configuration for completeness and provide suggestions.
        
        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        if not self._is_configured:
            from .config_templates import ValidationResult
            return ValidationResult(
                errors=["No configuration loaded"],
                warnings=[],
                suggestions=["Load configuration before validation"]
            )
        
        from .config_templates import ConfigurationValidator
        return ConfigurationValidator.validate(self._config)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for configuration loading and validation.
        
        Returns:
            Dictionary with performance and validation metrics
        """
        if not self._is_configured:
            return {"configured": False}
        
        metrics = {
            "configured": True,
            "models_count": len(self._models),
            "profiles_count": len(self._profiles),
            "aliases_count": len(self._aliases),
            "providers_count": len(self._providers),
        }
        
        # Add validation metrics if available
        if hasattr(self, '_validation_metrics'):
            metrics.update(self._validation_metrics)
        
        return metrics