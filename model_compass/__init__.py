"""
Model Compass - A lightweight, provider-agnostic Python library for LLM configuration and access.

This library provides a unified interface for configuring LLM providers and accessing models
through intent-based URNs, logical profiles, physical identifiers, and flexible alias chains.
"""

from .core import configure, model, profile, models, profiles, aliases, get_current_profile, get_profile_stack, reset_profile_context, reset_configuration, is_configured, enable_performance_optimizations, get_performance_stats, generate_template, validate_config, get_config_suggestions, is_configured_with_details, list_models, list_profiles, list_aliases
from .logging_utils import enable_verbose_logging, disable_verbose_logging, is_verbose_enabled

from .config_templates import (
    generate_openai_config, generate_anthropic_config, generate_ollama_config,
    generate_multi_provider_config, generate_development_config, generate_production_config,
    ConfigurationTemplates, ConfigurationValidator, ConfigurationExamples
)
from .exceptions import (
    LLMConfigError,
    ConfigurationError,
    ResolutionError,
    CircularAliasError,
    NetworkError,
    ProviderError
)
from .data_models import ModelConfig, Profile, ProviderConfig, ResolvedModel
from .proxy import ModelProxy

__version__ = "0.1.0"
__all__ = [
    # Main API functions
    "configure",
    "model",
    "profile",
    # Utility functions
    "is_configured",
    "enable_performance_optimizations",
    "get_performance_stats",
    # New configuration-focused API methods
    "generate_template",
    "validate_config", 
    "get_config_suggestions",
    "is_configured_with_details",
    "list_models",
    "list_profiles", 
    "list_aliases",
    # Logging functions
    "enable_verbose_logging",
    "disable_verbose_logging", 
    "is_verbose_enabled",

    # Configuration templates
    "generate_openai_config",
    "generate_anthropic_config", 
    "generate_ollama_config",
    "generate_multi_provider_config",
    "generate_development_config",
    "generate_production_config",
    "ConfigurationTemplates",
    "ConfigurationValidator", 
    "ConfigurationExamples",
    # Profile context utilities
    "get_current_profile",
    "get_profile_stack",
    "reset_profile_context",
    "reset_configuration",
    # Collections
    "models",
    "profiles", 
    "aliases",
    # Data models
    "ModelConfig",
    "Profile",
    "ProviderConfig", 
    "ResolvedModel",
    "ModelProxy",
    # Exceptions
    "LLMConfigError",
    "ConfigurationError",
    "ResolutionError",
    "CircularAliasError",
    "NetworkError",
    "ProviderError"
]