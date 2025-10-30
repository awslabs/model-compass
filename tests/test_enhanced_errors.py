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
Tests for enhanced error handling and user-friendly error messages.

This module tests the improved error handling that provides configuration examples,
helpful suggestions, and context-aware error messages when using user-config-only approach.
"""

import pytest
import model_compass as mc
from model_compass.exceptions import ConfigurationError, ResolutionError
from model_compass.config_templates import ConfigurationTemplates


class TestConfigurationErrors:
    """Test cases for configuration-related error handling."""
    
    def test_no_configuration_error(self):
        """Test clear error messages when no configuration is provided."""
        # Reset to ensure no configuration
        mc.reset_configuration()
        
        # Should get helpful error when trying to use model() without configuration
        with pytest.raises(ConfigurationError) as exc_info:
            mc.model("gpt-4")
        
        error = exc_info.value
        assert "not configured" in str(error).lower()
        assert "configure()" in str(error)
    
    def test_no_configuration_with_suggestions(self):
        """Test that no-configuration errors include helpful suggestions."""
        mc.reset_configuration()
        
        with pytest.raises(ConfigurationError) as exc_info:
            mc.model("test-model")
        
        error = exc_info.value
        # Should suggest using configure()
        assert "configure()" in str(error)
    
    def test_invalid_configuration_dict_error(self):
        """Test error handling for invalid configuration dictionary."""
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_dict="not a dict")
        
        assert "dictionary" in str(exc_info.value)
    
    def test_missing_config_arguments_error(self):
        """Test error when neither config_path nor config_dict is provided."""
        with pytest.raises(ValueError) as exc_info:
            mc.configure()
        
        assert "config_path or config_dict must be provided" in str(exc_info.value)
    
    def test_both_config_arguments_error(self):
        """Test error when both config_path and config_dict are provided."""
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_path="test.yaml", config_dict={})
        
        assert "Cannot provide both" in str(exc_info.value)


class TestModelResolutionErrors:
    """Test cases for model resolution error handling."""
    
    def setup_method(self):
        """Set up a basic configuration for testing."""
        config = {
            "models": {
                "test-model": {
                    "provider": "test-provider",
                    "model": "test-model",
                    "metadata": {
                        "reasoning": "medium",
                        "cost": "low",
                        "speed": "fast"
                    }
                }
            },
            "providers": {
                "test-provider": {
                    "base_url": "http://test.com"
                }
            },
            "profiles": {
                "test-profile": {
                    "model": "test-model"
                }
            },
            "aliases": {
                "test-alias": "test-model"
            }
        }
        mc.configure(config_dict=config)
    
    def teardown_method(self):
        """Clean up after each test."""
        mc.reset_configuration()
    
    def test_model_not_found_error_with_suggestions(self):
        """Test helpful error messages when models are not found."""
        model_proxy = mc.model("nonexistent-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        assert "not found" in str(error)
        
        # Should include available models
        assert hasattr(error, 'available_models')
        assert "test-model" in error.available_models
        
        # Should include suggestions
        assert hasattr(error, 'suggestions')
        assert len(error.suggestions) > 0
        
        # Should include configuration examples
        assert hasattr(error, 'configuration_examples')
        assert len(error.configuration_examples) > 0
    
    def test_profile_not_found_error_with_suggestions(self):
        """Test helpful error messages when profiles are not found."""
        model_proxy = mc.model("nonexistent-profile")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        assert "not found" in str(error)
        
        # Should include available models/profiles in suggestions
        assert hasattr(error, 'suggestions')
        suggestions_text = " ".join(error.suggestions)
        assert "test-profile" in suggestions_text or "test-model" in suggestions_text
    
    def test_physical_identifier_provider_not_found(self):
        """Test error handling for physical identifiers with unknown providers."""
        model_proxy = mc.model("unknown-provider/some-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        assert "Unknown provider" in str(error)
        assert "unknown-provider" in str(error)
        
        # Should include available providers
        assert hasattr(error, 'available_models')  # Contains available providers in this context
        
        # Should include configuration examples
        assert hasattr(error, 'configuration_examples')
        assert len(error.configuration_examples) > 0
        
        # Configuration example should show how to add the provider
        example = error.configuration_examples[0]
        assert "some-model" in example  # The model name from the physical identifier
        assert "providers:" in example
    
    def test_invalid_physical_identifier_format(self):
        """Test error handling for invalid physical identifier format."""
        model_proxy = mc.model("invalid-format-no-slash")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        # Should provide suggestions about available models
        assert hasattr(error, 'suggestions')
        assert len(error.suggestions) > 0
    
    def test_detailed_error_message_method(self):
        """Test the get_detailed_message() method of ResolutionError."""
        model_proxy = mc.model("nonexistent-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        detailed_message = error.get_detailed_message()
        
        # Should include available models
        assert "Available models:" in detailed_message
        assert "test-model" in detailed_message
        
        # Should include suggestions
        assert "Suggestions:" in detailed_message
        
        # Should include configuration examples
        assert "Configuration examples:" in detailed_message


class TestIntentResolutionErrors:
    """Test cases for intent resolution error handling."""
    
    def setup_method(self):
        """Set up configuration with metadata for intent testing."""
        config = ConfigurationTemplates.generate_multi_provider_template(
            openai_models=["gpt-4"],
            anthropic_models=["claude-3-sonnet"],
            ollama_models=["llama3"]
        )
        mc.configure(config_dict=config)
    
    def teardown_method(self):
        """Clean up after each test."""
        mc.reset_configuration()
    
    def test_intent_no_matching_models(self):
        """Test error handling when no models match intent criteria."""
        # Use criteria that won't match any models
        model_proxy = mc.model("urn:llm:intent:chat?reasoning=impossible&cost=negative")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        assert "No models found matching intent criteria" in str(error)
        
        # Should include available models
        assert hasattr(error, 'available_models')
        assert len(error.available_models) > 0
        
        # Should include configuration examples for intent resolution
        assert hasattr(error, 'configuration_examples')
        assert len(error.configuration_examples) > 0
        
        # Configuration example should show how to add models with matching metadata
        example = error.configuration_examples[0]
        assert "reasoning: impossible" in example or "cost: negative" in example
    
    def test_invalid_urn_format_error(self):
        """Test error handling for invalid URN format."""
        model_proxy = mc.model("urn:invalid:format")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        # Should provide suggestions about available models since URN parsing failed
        assert hasattr(error, 'suggestions')
        suggestions_text = " ".join(error.suggestions)
        # Since the URN format is invalid, it falls back to general model suggestions
        assert "available models" in suggestions_text.lower()
    
    def test_intent_resolution_with_custom_metadata(self):
        """Test intent resolution errors with custom metadata fields."""
        # Add a model with custom metadata
        custom_config = {
            "models": {
                "custom-model": {
                    "provider": "custom",
                    "model": "custom-model",
                    "metadata": {
                        "custom_field": "custom_value",
                        "accuracy": "high"
                    }
                }
            },
            "providers": {
                "custom": {
                    "base_url": "http://custom.com"
                }
            }
        }
        mc.configure(config_dict=custom_config)
        
        # Try to match on a custom field that doesn't exist
        model_proxy = mc.model("urn:llm:intent:chat?nonexistent_field=value")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        assert hasattr(error, 'suggestions')
        
        # Should suggest available metadata fields
        suggestions_text = " ".join(error.suggestions)
        # The error message should mention that the field doesn't exist and suggest adding it
        assert "nonexistent_field" in suggestions_text


class TestConfigurationValidationErrors:
    """Test cases for configuration validation error messages."""
    
    def test_validate_config_with_errors(self):
        """Test validation of invalid configuration with helpful messages."""
        invalid_config = {
            "models": {
                "invalid-model": {
                    # Missing required fields
                    "metadata": {}
                }
            }
        }
        
        result = mc.validate_config(config_dict=invalid_config)
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Errors should mention missing required fields
        errors_text = " ".join(result.errors)
        assert "provider" in errors_text or "model" in errors_text
    
    def test_validate_config_with_warnings_and_suggestions(self):
        """Test validation warnings and suggestions for incomplete configuration."""
        minimal_config = {
            "models": {
                "minimal-model": {
                    "provider": "test",
                    "model": "test-model"
                    # No metadata - should generate warnings
                }
            },
            "providers": {
                "test": {
                    "base_url": "http://test.com"
                }
            }
        }
        
        result = mc.validate_config(config_dict=minimal_config)
        assert result.is_valid  # Structure is valid
        assert result.has_warnings
        assert len(result.suggestions) > 0
        
        # Should warn about missing metadata
        warnings_text = " ".join(result.warnings)
        assert "metadata" in warnings_text.lower()
        
        # Should suggest adding metadata fields
        suggestions_text = " ".join(result.suggestions)
        assert "metadata" in suggestions_text.lower()
    
    def test_get_config_suggestions_api(self):
        """Test the get_config_suggestions API method."""
        # Configure with minimal configuration
        minimal_config = {
            "models": {
                "test-model": {
                    "provider": "test",
                    "model": "test-model"
                }
            },
            "providers": {
                "test": {
                    "base_url": "http://test.com"
                }
            }
        }
        mc.configure(config_dict=minimal_config)
        
        suggestions = mc.get_config_suggestions()
        assert len(suggestions) > 0
        
        # Should suggest adding metadata
        suggestions_text = " ".join(suggestions)
        assert "metadata" in suggestions_text.lower()
        
        mc.reset_configuration()
    
    def test_is_configured_with_details_error_state(self):
        """Test is_configured_with_details when not configured."""
        mc.reset_configuration()
        
        status = mc.is_configured_with_details()
        assert not status["configured"]
        assert "quick_start" in status
        assert "suggestions" in status
        
        # Should include helpful quick-start instructions
        assert "configure" in status["quick_start"].lower()
        assert len(status["suggestions"]) > 0


class TestErrorMessageQuality:
    """Test cases for the quality and helpfulness of error messages."""
    
    def test_error_messages_include_context(self):
        """Test that error messages include relevant context information."""
        config = ConfigurationTemplates.generate_openai_template()
        mc.configure(config_dict=config)
        
        model_proxy = mc.model("nonexistent-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        
        # Should include the identifier that failed
        assert hasattr(error, 'identifier')
        assert error.identifier == "nonexistent-model"
        
        # Should include available alternatives
        assert hasattr(error, 'available_models')
        assert len(error.available_models) > 0
        
        mc.reset_configuration()
    
    def test_error_messages_are_actionable(self):
        """Test that error messages provide actionable suggestions."""
        mc.reset_configuration()
        
        with pytest.raises(ConfigurationError) as exc_info:
            mc.model("any-model")
        
        error_message = str(exc_info.value)
        
        # Should tell user what to do
        assert "configure()" in error_message
        
        # Should be specific about the problem
        assert "not configured" in error_message.lower()
    
    def test_configuration_examples_are_valid(self):
        """Test that configuration examples in error messages are valid."""
        config = {"models": {}, "providers": {}}
        mc.configure(config_dict=config)
        
        model_proxy = mc.model("example-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        if hasattr(error, 'configuration_examples') and error.configuration_examples:
            example = error.configuration_examples[0]
            
            # Should look like valid YAML configuration
            assert "models:" in example
            assert "example-model" in example
            assert "provider:" in example
            assert "providers:" in example
        
        mc.reset_configuration()
    
    def test_suggestions_are_relevant(self):
        """Test that suggestions are relevant to the specific error context."""
        config = ConfigurationTemplates.generate_openai_template()
        mc.configure(config_dict=config)
        
        # Test physical identifier error
        model_proxy = mc.model("unknown-provider/some-model")
        
        # Error should be raised when accessing properties (lazy resolution)
        with pytest.raises(ResolutionError) as exc_info:
            _ = model_proxy.provider
        
        error = exc_info.value
        if hasattr(error, 'suggestions'):
            suggestions_text = " ".join(error.suggestions)
            
            # Should mention providers since it's a provider error
            assert "provider" in suggestions_text.lower()
            
            # Should mention available providers
            assert "openai" in suggestions_text
        
        mc.reset_configuration()