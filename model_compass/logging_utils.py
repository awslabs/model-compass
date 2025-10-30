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
Logging utilities for the Model Compass library.

This module provides optional verbose logging support for debugging
configuration issues and understanding library behavior.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


# Library logger
_logger = None
_verbose_enabled = False


def get_logger() -> logging.Logger:
    """
    Get the Model Compass library logger.
    
    Returns:
        Logger instance for the library
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("model_compass")
        _logger.setLevel(logging.WARNING)  # Default to WARNING level
        
        # Add console handler if none exists
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '[%(name)s] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
    
    return _logger


def enable_verbose_logging(level: str = "DEBUG") -> None:
    """
    Enable verbose logging for debugging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> import model_compass as mc
        >>> mc.enable_verbose_logging("DEBUG")
        >>> mc.configure("config.yaml")  # Will show detailed logging
    """
    global _verbose_enabled
    
    logger = get_logger()
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    logger.setLevel(numeric_level)
    
    _verbose_enabled = True
    
    logger.info(f"Verbose logging enabled at {level} level")


def disable_verbose_logging() -> None:
    """
    Disable verbose logging and return to default WARNING level.
    
    Example:
        >>> import model_compass as mc
        >>> mc.disable_verbose_logging()
    """
    global _verbose_enabled
    
    logger = get_logger()
    logger.setLevel(logging.WARNING)
    _verbose_enabled = False


def is_verbose_enabled() -> bool:
    """
    Check if verbose logging is currently enabled.
    
    Returns:
        True if verbose logging is enabled, False otherwise
    """
    return _verbose_enabled


def log_debug(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a debug message with optional context.
    
    Args:
        message: Debug message to log
        context: Optional context dictionary
    """
    logger = get_logger()
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.debug(f"{message} (context: {context_str})")
    else:
        logger.debug(message)


def log_info(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an info message with optional context.
    
    Args:
        message: Info message to log
        context: Optional context dictionary
    """
    logger = get_logger()
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.info(f"{message} (context: {context_str})")
    else:
        logger.info(message)


def log_warning(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a warning message with optional context.
    
    Args:
        message: Warning message to log
        context: Optional context dictionary
    """
    logger = get_logger()
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.warning(f"{message} (context: {context_str})")
    else:
        logger.warning(message)


def log_error(message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error message with optional context.
    
    Args:
        message: Error message to log
        context: Optional context dictionary
    """
    logger = get_logger()
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.error(f"{message} (context: {context_str})")
    else:
        logger.error(message)


def log_configuration_load(config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Log configuration loading attempt.
    
    Args:
        config_path: Path to configuration file being loaded
        config_dict: Configuration dictionary being loaded
    """
    if config_path:
        log_info(f"Loading configuration from file: {config_path}")
    elif config_dict:
        log_info(f"Loading configuration from dictionary with {len(config_dict)} sections")


def log_resolution_attempt(identifier: str, resolution_type: str) -> None:
    """
    Log model resolution attempt.
    
    Args:
        identifier: Identifier being resolved
        resolution_type: Type of resolution (intent, profile, alias, physical, default)
    """
    log_debug(f"Attempting {resolution_type} resolution for identifier: {identifier}")


def log_resolution_success(identifier: str, resolved_model: str, resolution_path: Optional[list] = None) -> None:
    """
    Log successful model resolution.
    
    Args:
        identifier: Original identifier
        resolved_model: Final resolved model (provider/model)
        resolution_path: Path taken during resolution
    """
    context = {"resolved_to": resolved_model}
    if resolution_path:
        context["resolution_path"] = " -> ".join(resolution_path)
    
    log_debug(f"Successfully resolved identifier: {identifier}", context)


def log_alias_chain_resolution(alias: str, chain: list, final_target: str) -> None:
    """
    Log alias chain resolution details.
    
    Args:
        alias: Original alias name
        chain: Full resolution chain
        final_target: Final target after resolution
    """
    chain_str = " -> ".join(chain)
    log_debug(f"Alias chain resolution: {alias} -> {chain_str} -> {final_target}")


def log_intent_matching(service: str, criteria: Dict[str, str], matches: list) -> None:
    """
    Log intent matching process.
    
    Args:
        service: Service type being matched
        criteria: Intent criteria
        matches: List of matching models with scores
    """
    criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
    log_debug(f"Intent matching for service '{service}' with criteria: {criteria_str}")
    
    if matches:
        for model_id, score in matches[:3]:  # Log top 3 matches
            log_debug(f"  Match: {model_id} (score: {score:.3f})")
    else:
        log_debug("  No matches found")


def log_profile_context_switch(profile_name: str, previous_profile: Optional[str], stack_depth: int) -> None:
    """
    Log profile context switching.
    
    Args:
        profile_name: Profile being switched to
        previous_profile: Previous active profile
        stack_depth: Current context stack depth
    """
    context = {
        "previous_profile": previous_profile or "None",
        "stack_depth": stack_depth
    }
    log_debug(f"Switching to profile context: {profile_name}", context)


def log_configuration_validation(section: str, item_count: int, errors: Optional[list] = None) -> None:
    """
    Log configuration validation results.
    
    Args:
        section: Configuration section being validated
        item_count: Number of items in the section
        errors: List of validation errors, if any
    """
    if errors:
        log_warning(f"Configuration validation found {len(errors)} errors in {section} section")
        for error in errors[:5]:  # Log first 5 errors
            log_warning(f"  {error}")
    else:
        log_debug(f"Configuration validation passed for {section} section ({item_count} items)")


def create_error_context(
    identifier: Optional[str] = None,
    resolution_path: Optional[list] = None,
    available_items: Optional[list] = None,
    suggestions: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized error context dictionary.
    
    Args:
        identifier: The identifier that caused the error
        resolution_path: Resolution path attempted
        available_items: List of available items (models, profiles, etc.)
        suggestions: List of suggestions for fixing the error
        **kwargs: Additional context items
        
    Returns:
        Dictionary with error context
    """
    context = {}
    
    if identifier:
        context["identifier"] = identifier
    
    if resolution_path:
        context["resolution_path"] = " -> ".join(str(item) for item in resolution_path)
    
    if available_items:
        # Limit to first 10 items to avoid overwhelming output
        context["available_items"] = available_items[:10]
        if len(available_items) > 10:
            context["total_available"] = len(available_items)
    
    if suggestions:
        context["suggestions"] = suggestions
    
    # Add any additional context
    context.update(kwargs)
    
    return context