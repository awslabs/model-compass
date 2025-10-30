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
Core API functions and module-level interface for Model Compass.

This module provides the main public API functions that users interact with,
including configuration loading, model access, and profile switching.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

from .data_models import ModelConfig, Profile, ProviderConfig, ResolvedModel
from .exceptions import LLMConfigError, ConfigurationError
from .proxy import ModelProxy
from .collections import ModelsCollection, ProfilesCollection, AliasesCollection
from .logging_utils import (
    log_debug, log_info, log_warning, log_error, 
    log_resolution_attempt, log_resolution_success,
    log_profile_context_switch, create_error_context,
    enable_verbose_logging, disable_verbose_logging, is_verbose_enabled
)


# Global state for the library
_configuration_manager = None
_resolution_engine = None
_current_profile = None
_profile_stack = []

# Global collection instances
models = ModelsCollection()
profiles = ProfilesCollection()
aliases = AliasesCollection()


def configure(config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Load configuration from file or dictionary.
    
    This function initializes the library with model, profile, and provider
    configurations. It must be called before using other library functions.
    
    Args:
        config_path: Path to YAML or JSON configuration file
        config_dict: Configuration dictionary (alternative to file)
        
    Raises:
        ConfigurationError: If configuration is invalid or cannot be loaded
        ValueError: If both or neither arguments are provided
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> # or
        >>> mc.configure(config_dict={"models": {...}, "profiles": {...}})
    """
    global _configuration_manager, _resolution_engine
    
    # Input validation
    if config_path is None and config_dict is None:
        raise ValueError("Either config_path or config_dict must be provided")
    
    if config_path is not None and config_dict is not None:
        raise ValueError("Cannot provide both config_path and config_dict")
    
    # Validate config_path if provided
    if config_path is not None:
        if not isinstance(config_path, (str, Path)):
            raise ValueError(f"config_path must be a string or Path object, got {type(config_path).__name__}")
        
        config_path = str(config_path)  # Convert Path to string if needed
        if not config_path.strip():
            raise ValueError("config_path cannot be empty")
    
    # Validate config_dict if provided
    if config_dict is not None:
        if not isinstance(config_dict, dict):
            raise ValueError(f"config_dict must be a dictionary, got {type(config_dict).__name__}")
    
    try:
        log_info("Starting configuration loading")
        
        # Import here to avoid circular imports
        from .config_manager import ConfigurationManager
        from .resolution import ResolutionEngine
        
        # Reset any existing state
        _configuration_manager = None
        _resolution_engine = None
        
        # Clear all caches when configuration changes
        from .caching import clear_all_caches
        clear_all_caches()
        
        # Initialize configuration manager with configuration
        if config_path:
            _configuration_manager = ConfigurationManager(config_path=config_path)
        else:
            _configuration_manager = ConfigurationManager(config_dict=config_dict)
        
        # Initialize resolution engine
        log_debug("Initializing resolution engine")
        _resolution_engine = ResolutionEngine(_configuration_manager)
        
        # Initialize global collections
        log_debug("Initializing global collections")
        models._set_resolution_engine(_resolution_engine)
        profiles._set_resolution_engine(_resolution_engine)
        aliases._set_resolution_engine(_resolution_engine)
        
        # Reset profile context to ensure clean state
        reset_profile_context()
        
        log_info("Configuration loading completed successfully")
        
    except Exception as e:
        log_error(f"Configuration loading failed: {str(e)}")
        
        # Clean up partial state on failure
        _configuration_manager = None
        _resolution_engine = None
        reset_profile_context()
        
        if isinstance(e, LLMConfigError):
            raise
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


def model(identifier: Optional[str] = None) -> ModelProxy:
    """
    Get a model proxy object with property-based access.
    
    Returns a ModelProxy instance that provides convenient access to model
    configuration through properties and dictionary-like interface.
    
    Args:
        identifier: Model identifier (URN, profile, alias, or physical ID).
                   If None, uses the current profile or default profile.
                   
    Returns:
        ModelProxy instance for the resolved model
        
    Raises:
        ConfigurationError: If library is not configured
        ResolutionError: If identifier cannot be resolved
        ValueError: If identifier is not a string or None
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> 
        >>> # Get default model
        >>> default_model = mc.model()
        >>> print(f"Using {default_model.provider}/{default_model.name}")
        >>> 
        >>> # Get specific model by profile
        >>> prod_model = mc.model("prod")
        >>> 
        >>> # Get model by intent
        >>> smart_model = mc.model("urn:llm:intent:chat?reasoning=high")
    """
    if _resolution_engine is None:
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    # Validate identifier type
    if identifier is not None and not isinstance(identifier, str):
        raise ValueError(f"identifier must be a string or None, got {type(identifier).__name__}")
    
    # Validate identifier is not empty string
    if identifier is not None and not identifier.strip():
        raise ValueError("identifier cannot be an empty string")
    
    # Normalize identifier (strip whitespace if provided)
    if identifier is not None:
        identifier = identifier.strip()
    
    try:
        log_resolution_attempt(identifier or "default", "model_proxy")
        proxy = ModelProxy(identifier, _resolution_engine)
        log_debug(f"Created model proxy for identifier: {identifier or 'default'}")
        return proxy
    except Exception as e:
        log_error(f"Failed to create model proxy for identifier '{identifier}': {str(e)}")
        
        # Add context to resolution errors
        if isinstance(e, LLMConfigError):
            raise
        raise ConfigurationError(f"Failed to create model proxy: {str(e)}")



@contextmanager
def profile(profile_name: str):
    """
    Temporarily switch to a different profile using a context manager.
    
    This allows temporary switching to different model profiles with proper
    context stack management for nested profile switches. The context manager
    ensures that the previous profile is always restored, even if an exception
    occurs within the context.
    
    Args:
        profile_name: Name of the profile to switch to
        
    Yields:
        str: The profile name that was activated
        
    Raises:
        ConfigurationError: If library is not configured
        ResolutionError: If profile does not exist
        ValueError: If profile_name is None or empty
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> 
        >>> # Default profile
        >>> default_model = mc.model()
        >>> print(f"Default: {default_model}")
        >>> 
        >>> # Temporarily switch to local profile
        >>> with mc.profile("local") as active_profile:
        ...     print(f"Active profile: {active_profile}")
        ...     local_model = mc.model()
        ...     print(f"Local: {local_model}")
        ...     
        ...     # Nested context switching
        ...     with mc.profile("prod"):
        ...         prod_model = mc.model()
        ...         print(f"Nested prod: {prod_model}")
        >>> 
        >>> # Back to default
        >>> default_again = mc.model()
        >>> print(f"Default again: {default_again}")
    """
    global _current_profile, _profile_stack
    
    # Validate inputs
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("Profile name must be a non-empty string")
    
    profile_name = profile_name.strip()
    
    if _resolution_engine is None:
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    # Validate that profile exists
    if not _configuration_manager.has_profile(profile_name):
        from .exceptions import ResolutionError
        available_profiles = list(_configuration_manager.get_profiles().keys())
        raise ResolutionError(
            f"Profile '{profile_name}' not found",
            identifier=profile_name,
            context={
                "available_profiles": available_profiles,
                "current_profile": _current_profile,
                "profile_stack_depth": len(_profile_stack)
            }
        )
    
    # Save current profile to stack
    previous_profile = _current_profile
    _profile_stack.append(previous_profile)
    _current_profile = profile_name
    
    log_profile_context_switch(profile_name, previous_profile, len(_profile_stack))
    
    try:
        yield profile_name
    except Exception as e:
        log_error(f"Exception in profile context '{profile_name}': {str(e)}")
        raise
    finally:
        # Always restore previous profile, even if an exception occurred
        try:
            restored_profile = _profile_stack.pop()
            _current_profile = restored_profile
            log_debug(f"Restored profile context to: {restored_profile or 'default'}")
        except IndexError:
            # This should never happen, but handle gracefully
            log_warning("Profile stack underflow - resetting to default state")
            _current_profile = None
            # Reset the stack to be safe
            _profile_stack = []


def get_current_profile() -> Optional[str]:
    """
    Get the currently active profile.
    
    Returns the name of the profile that is currently active due to
    context manager usage, or None if using the default profile.
    
    Returns:
        Current profile name or None if using default
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> 
        >>> print(mc.get_current_profile())  # None (default)
        >>> 
        >>> with mc.profile("local"):
        ...     print(mc.get_current_profile())  # "local"
    """
    return _current_profile


def get_profile_stack() -> List[str]:
    """
    Get the current profile context stack.
    
    Returns a copy of the current profile stack, showing the nested
    context hierarchy. Useful for debugging nested profile contexts.
    
    Returns:
        List of profile names in the current stack (outermost first)
        
    Example:
        >>> with mc.profile("local"):
        ...     with mc.profile("dev"):
        ...         print(mc.get_profile_stack())  # [None, "local"]
    """
    return _profile_stack.copy()


def reset_profile_context():
    """
    Reset the profile context to default state.
    
    This function clears the current profile and profile stack,
    returning to the default profile state. Useful for testing
    or error recovery scenarios.
    
    Warning:
        This function should be used with caution as it can disrupt
        active context managers. It's primarily intended for testing
        and error recovery scenarios.
    """
    global _current_profile, _profile_stack
    _current_profile = None
    _profile_stack = []


def reset_configuration():
    """
    Reset the entire configuration state to unconfigured.
    
    This function clears all configuration state including the configuration
    manager, resolution engine, and global collections. Useful for testing
    and complete state reset scenarios.
    
    Warning:
        This function should be used with caution as it completely resets
        the library state. It's primarily intended for testing scenarios.
    """
    global _configuration_manager, _resolution_engine
    
    # Reset core state
    _configuration_manager = None
    _resolution_engine = None
    
    # Reset profile context
    reset_profile_context()
    
    # Reset global collections by clearing their resolution engines
    models._set_resolution_engine(None)
    profiles._set_resolution_engine(None)
    aliases._set_resolution_engine(None)


def _get_current_profile() -> Optional[str]:
    """
    Internal function to get the currently active profile.
    
    This is used by the ModelProxy and resolution engine to determine
    which profile should be used for default model resolution.
    
    Returns:
        Current profile name or None if using default
    """
    return _current_profile


def is_configured() -> bool:
    """
    Check if the library has been configured.
    
    This function can be used to verify that the library is ready for use
    before making API calls. Useful for conditional logic and debugging.
    
    Returns:
        True if library is configured, False otherwise
        
    Example:
        >>> import model_compass as mc
        >>> print(mc.is_configured())  # False
        >>> mc.configure("config.yaml")
        >>> print(mc.is_configured())  # True
    """
    return _configuration_manager is not None and _resolution_engine is not None


def enable_performance_optimizations() -> None:
    """
    Enable all performance optimizations for the library.
    
    This function enables:
    - Configuration file caching to avoid re-parsing files
    - Resolution result caching for frequently accessed models
    - Lazy loading of model definitions and provider configurations
    - Optimized intent matching algorithm for large model sets
    
    Call this function after configure() to enable optimizations.
    
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> mc.enable_performance_optimizations()
    """
    if not is_configured():
        raise ConfigurationError("Library must be configured before enabling optimizations")
    
    from .caching import enable_performance_optimizations as enable_caching_optimizations
    enable_caching_optimizations()
    
    log_info("Performance optimizations enabled for Model Compass")


def get_performance_stats() -> Dict[str, Any]:
    """
    Get performance statistics from all caches and optimizations.
    
    Returns:
        Dictionary with comprehensive performance statistics
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure("config.yaml")
        >>> mc.enable_performance_optimizations()
        >>> stats = mc.get_performance_stats()
        >>> print(f"Config cache hit rate: {stats['configuration_cache']['hit_rate']}%")
    """
    from .caching import get_cache_stats
    return get_cache_stats()


def generate_template(provider: str = "multi", **kwargs) -> Dict[str, Any]:
    """
    Generate a configuration template for quick setup.
    
    Convenience method to generate configuration templates without importing
    the ConfigurationTemplates class directly.
    
    Args:
        provider: Provider type ('openai', 'anthropic', 'ollama', 'multi')
        **kwargs: Additional arguments passed to the template generator
        
    Returns:
        Configuration dictionary ready for use with configure()
        
    Example:
        >>> import model_compass as mc
        >>> config = mc.generate_template('openai')
        >>> mc.configure(config_dict=config)
        >>> 
        >>> # Multi-provider setup
        >>> config = mc.generate_template('multi', 
        ...     openai_models=['gpt-4'], 
        ...     anthropic_models=['claude-3-sonnet'])
        >>> mc.configure(config_dict=config)
    """
    from .config_templates import ConfigurationTemplates
    
    if provider == "openai":
        return ConfigurationTemplates.generate_openai_template(**kwargs)
    elif provider == "anthropic":
        return ConfigurationTemplates.generate_anthropic_template(**kwargs)
    elif provider == "ollama":
        return ConfigurationTemplates.generate_ollama_template(**kwargs)
    elif provider == "multi":
        return ConfigurationTemplates.generate_multi_provider_template(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', 'ollama', or 'multi'")


def validate_config(config_dict: Dict[str, Any] = None, config_path: str = None) -> 'ValidationResult':
    """
    Validate a configuration dictionary or file without loading it.
    
    This method allows you to validate configuration before using it with configure().
    Useful for checking configuration files during development or deployment.
    
    Args:
        config_dict: Configuration dictionary to validate
        config_path: Path to configuration file to validate
        
    Returns:
        ValidationResult with errors, warnings, and suggestions
        
    Raises:
        ValueError: If both or neither arguments are provided
        ConfigurationError: If file cannot be read
        
    Example:
        >>> import model_compass as mc
        >>> 
        >>> # Validate dictionary
        >>> config = mc.generate_template('openai')
        >>> result = mc.validate_config(config_dict=config)
        >>> print(f"Valid: {result.is_valid}")
        >>> 
        >>> # Validate file
        >>> result = mc.validate_config(config_path='config.yaml')
        >>> for warning in result.warnings:
        ...     print(f"Warning: {warning}")
    """
    from .config_templates import ConfigurationValidator
    
    if config_dict is None and config_path is None:
        raise ValueError("Either config_dict or config_path must be provided")
    
    if config_dict is not None and config_path is not None:
        raise ValueError("Cannot provide both config_dict and config_path")
    
    if config_path is not None:
        # Load configuration from file for validation
        from .config_manager import ConfigurationManager
        temp_manager = ConfigurationManager(config_path=config_path)
        config_dict = temp_manager.get_raw_config()
    
    return ConfigurationValidator.validate(config_dict)


def get_config_suggestions() -> List[str]:
    """
    Get improvement recommendations for the current configuration.
    
    Analyzes the current configuration and provides suggestions for improving
    completeness, metadata coverage, and intent resolution capabilities.
    
    Returns:
        List of suggestion strings
        
    Raises:
        ConfigurationError: If library is not configured
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure('config.yaml')
        >>> suggestions = mc.get_config_suggestions()
        >>> for suggestion in suggestions:
        ...     print(f"Suggestion: {suggestion}")
    """
    if not is_configured():
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    validation_result = _configuration_manager.validate_configuration_completeness()
    return validation_result.suggestions


def is_configured_with_details() -> Dict[str, Any]:
    """
    Check if the library is configured and provide detailed status information.
    
    Enhanced version of is_configured() that provides detailed information about
    the configuration state, including counts of models, profiles, and validation status.
    
    Returns:
        Dictionary with configuration status and details
        
    Example:
        >>> import model_compass as mc
        >>> status = mc.is_configured_with_details()
        >>> if status['configured']:
        ...     print(f"Loaded {status['models_count']} models")
        >>> else:
        ...     print("Not configured - use mc.configure() first")
    """
    if not is_configured():
        from .config_templates import ConfigurationExamples
        return {
            "configured": False,
            "quick_start": ConfigurationExamples.get_quick_start_instructions(),
            "suggestions": [
                "Use mc.configure(config_path='your-config.yaml')",
                "Or use mc.configure(config_dict={...})",
                "Generate a template with mc.generate_template()"
            ]
        }
    
    # Get detailed configuration metrics
    metrics = _configuration_manager.get_performance_metrics()
    validation_result = _configuration_manager.validate_configuration_completeness()
    
    return {
        "configured": True,
        "models_count": metrics.get("models_count", 0),
        "profiles_count": metrics.get("profiles_count", 0),
        "aliases_count": metrics.get("aliases_count", 0),
        "providers_count": metrics.get("providers_count", 0),
        "validation_warnings": len(validation_result.warnings),
        "validation_suggestions": len(validation_result.suggestions),
        "has_intent_metadata": metrics.get("has_intent_metadata", False)
    }


def list_models() -> List[str]:
    """
    List all available model identifiers from the current configuration.
    
    Returns a list of model IDs that can be used with mc.model().
    
    Returns:
        List of model identifiers
        
    Raises:
        ConfigurationError: If library is not configured
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure('config.yaml')
        >>> models = mc.list_models()
        >>> print(f"Available models: {', '.join(models)}")
    """
    if not is_configured():
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    return _configuration_manager.get_available_models()


def list_profiles() -> List[str]:
    """
    List all available profile names from the current configuration.
    
    Returns a list of profile names that can be used with mc.profile().
    
    Returns:
        List of profile names
        
    Raises:
        ConfigurationError: If library is not configured
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure('config.yaml')
        >>> profiles = mc.list_profiles()
        >>> print(f"Available profiles: {', '.join(profiles)}")
    """
    if not is_configured():
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    return _configuration_manager.get_available_profiles()


def list_aliases() -> List[str]:
    """
    List all available alias names from the current configuration.
    
    Returns a list of alias names that can be used with mc.model().
    
    Returns:
        List of alias names
        
    Raises:
        ConfigurationError: If library is not configured
        
    Example:
        >>> import model_compass as mc
        >>> mc.configure('config.yaml')
        >>> aliases = mc.list_aliases()
        >>> print(f"Available aliases: {', '.join(aliases)}")
    """
    if not is_configured():
        raise ConfigurationError("Library not configured. Call configure() first.")
    
    return _configuration_manager.get_available_aliases()


def _is_configured() -> bool:
    """
    Internal function to check if the library has been configured.
    
    This is used by other modules to verify that the library is ready for use.
    
    Returns:
        True if library is configured, False otherwise
    """
    return is_configured()