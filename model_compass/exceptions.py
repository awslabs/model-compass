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
Exception hierarchy for the Model Compass library.

This module defines all custom exceptions used throughout the library,
providing clear error messages and proper error categorization.
"""


class LLMConfigError(Exception):
    """
    Base exception for all Model Compass library errors.
    
    This is the root exception class that all other library-specific
    exceptions inherit from. It provides a common base for catching
    any library-related error.
    """
    
    def __init__(self, message: str, context: dict = None):
        """
        Initialize the exception with a message and optional context.
        
        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        """Return a formatted error message with context if available."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (context: {context_str})"
        return self.message


class ConfigurationError(LLMConfigError):
    """
    Configuration file or validation errors.
    
    Raised when there are issues with configuration file parsing,
    validation, or when required configuration elements are missing
    or invalid.
    """
    
    def __init__(self, message: str, config_path: str = None, context: dict = None):
        """
        Initialize with configuration-specific context.
        
        Args:
            message: Human-readable error message
            config_path: Path to the configuration file that caused the error
            context: Optional dictionary with additional error context
        """
        context = context or {}
        if config_path:
            context["config_path"] = config_path
        super().__init__(message, context)
        self.config_path = config_path


class ResolutionError(LLMConfigError):
    """
    Model resolution errors.
    
    Raised when the library cannot resolve a model identifier to a
    concrete model configuration. This includes cases where identifiers
    are not found, ambiguous, or invalid.
    """
    
    def __init__(self, message: str, identifier: str = None, resolution_path: list = None, 
                 context: dict = None, suggestions: list = None, available_models: list = None,
                 configuration_examples: list = None, quick_start_instructions: str = None):
        """
        Initialize with resolution-specific context.
        
        Args:
            message: Human-readable error message
            identifier: The identifier that failed to resolve
            resolution_path: List showing the resolution path attempted
            context: Optional dictionary with additional error context
            suggestions: Optional list of suggestions for fixing the error
            available_models: Optional list of available model identifiers
            configuration_examples: Optional list of configuration examples
            quick_start_instructions: Optional quick-start instructions when no config is loaded
        """
        context = context or {}
        if identifier:
            context["identifier"] = identifier
        if resolution_path:
            context["resolution_path"] = " -> ".join(str(item) for item in resolution_path)
        if suggestions:
            context["suggestions"] = suggestions
        if available_models:
            context["available_models"] = available_models
        
        super().__init__(message, context)
        self.identifier = identifier
        self.resolution_path = resolution_path or []
        self.suggestions = suggestions or []
        self.available_models = available_models or []
        self.configuration_examples = configuration_examples or []
        self.quick_start_instructions = quick_start_instructions
    
    def get_detailed_message(self) -> str:
        """
        Get a detailed error message with suggestions and examples.
        
        Returns:
            Formatted error message with context, suggestions, and configuration examples
        """
        msg = str(self)
        
        # Add available models if provided
        if self.available_models:
            msg += f"\n\nAvailable models: {', '.join(self.available_models)}"
        
        # Add suggestions
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        
        # Add configuration examples
        if self.configuration_examples:
            msg += "\n\nConfiguration examples:"
            for i, example in enumerate(self.configuration_examples, 1):
                msg += f"\n\n{i}. {example}"
        
        # Add quick start instructions
        if self.quick_start_instructions:
            msg += f"\n\nQuick start:\n{self.quick_start_instructions}"
        
        return msg
    
    def __str__(self):
        """Return enhanced error message with available models and examples."""
        msg = super().__str__()
        
        # Add available models inline for brief display
        if self.available_models and len(self.available_models) <= 5:
            msg += f"\n\nAvailable models: {', '.join(self.available_models)}"
        elif self.available_models:
            msg += f"\n\nAvailable models: {', '.join(self.available_models[:3])}, ... ({len(self.available_models)} total)"
        
        # Add first configuration example inline
        if self.configuration_examples:
            msg += f"\n\nTo add this model:\n{self.configuration_examples[0]}"
        
        return msg


class CircularAliasError(ResolutionError):
    """
    Circular alias reference detected.
    
    Raised when alias resolution encounters a circular reference,
    preventing infinite loops during alias chain traversal.
    """
    
    def __init__(self, message: str, alias_chain: list = None, context: dict = None, suggestions: list = None):
        """
        Initialize with circular reference context.
        
        Args:
            message: Human-readable error message
            alias_chain: List showing the circular alias chain
            context: Optional dictionary with additional error context
            suggestions: Optional list of suggestions for fixing the error
        """
        context = context or {}
        if alias_chain:
            context["circular_chain"] = " -> ".join(str(item) for item in alias_chain)
        
        # Default suggestions for circular alias errors
        if not suggestions:
            suggestions = [
                "Check your alias definitions for circular references",
                "Ensure each alias points to a concrete model or profile, not back to itself",
                "Use alias chain visualization to identify the circular dependency"
            ]
        
        super().__init__(message, resolution_path=alias_chain, context=context, suggestions=suggestions)
        self.alias_chain = alias_chain or []


class NetworkError(LLMConfigError):
    """
    Network-related errors when communicating with LLM providers.
    
    Raised when there are network connectivity issues, timeouts, or
    other network-level problems when trying to reach provider APIs.
    """
    
    def __init__(self, message: str, provider: str = None, base_url: str = None, 
                 context: dict = None, suggestions: list = None):
        """
        Initialize with network-specific context.
        
        Args:
            message: Human-readable error message
            provider: Provider name that failed
            base_url: Base URL that was unreachable
            context: Optional dictionary with additional error context
            suggestions: Optional list of suggestions for fixing the error
        """
        context = context or {}
        if provider:
            context["provider"] = provider
        if base_url:
            context["base_url"] = base_url
        
        # Default suggestions for network errors
        if not suggestions:
            suggestions = [
                "Check your internet connection",
                "Verify the provider's base URL is correct",
                "Check if the provider service is currently available",
                "Verify firewall settings allow outbound connections"
            ]
            if provider:
                suggestions.append(f"Check {provider} service status page for outages")
        
        if suggestions:
            context["suggestions"] = suggestions
        
        super().__init__(message, context)
        self.provider = provider
        self.base_url = base_url
        self.suggestions = suggestions or []


class ProviderError(LLMConfigError):
    """
    Provider-specific API errors.
    
    Raised when the provider API returns an error response, such as
    authentication failures, rate limiting, invalid requests, or
    provider-specific validation errors.
    """
    
    def __init__(self, message: str, provider: str = None, status_code: int = None,
                 error_code: str = None, context: dict = None, suggestions: list = None):
        """
        Initialize with provider-specific context.
        
        Args:
            message: Human-readable error message
            provider: Provider name that returned the error
            status_code: HTTP status code from the provider
            error_code: Provider-specific error code
            context: Optional dictionary with additional error context
            suggestions: Optional list of suggestions for fixing the error
        """
        context = context or {}
        if provider:
            context["provider"] = provider
        if status_code:
            context["status_code"] = status_code
        if error_code:
            context["error_code"] = error_code
        
        # Default suggestions based on common error patterns
        if not suggestions:
            suggestions = []
            
            if status_code == 401:
                suggestions.extend([
                    "Check your API key is correct and properly configured",
                    "Verify the API key has the necessary permissions",
                    "Ensure the API key is not expired"
                ])
            elif status_code == 403:
                suggestions.extend([
                    "Check if your account has access to the requested model",
                    "Verify your API key permissions",
                    "Check if you've exceeded usage limits"
                ])
            elif status_code == 429:
                suggestions.extend([
                    "You've hit rate limits - wait before retrying",
                    "Consider implementing exponential backoff",
                    "Check your usage tier and limits"
                ])
            elif status_code == 404:
                suggestions.extend([
                    "Check if the model name is correct",
                    "Verify the API endpoint URL is correct",
                    "Ensure the model is available in your region"
                ])
            else:
                suggestions.extend([
                    f"Check {provider} documentation for error details" if provider else "Check provider documentation",
                    "Verify your request parameters are valid",
                    "Try again later if this is a temporary issue"
                ])
        
        if suggestions:
            context["suggestions"] = suggestions
        
        super().__init__(message, context)
        self.provider = provider
        self.status_code = status_code
        self.error_code = error_code
        self.suggestions = suggestions or []