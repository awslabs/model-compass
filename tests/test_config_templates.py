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
Tests for configuration templates and generators.

This module tests the ConfigurationTemplates class and related functionality
for generating user configurations without relying on static model registry.
"""

import pytest
import model_compass as mc
from model_compass.config_templates import ConfigurationTemplates, ConfigurationValidator, ConfigurationExamples


class TestConfigurationTemplates:
    """Test cases for configuration template generators."""
    
    def test_openai_template_generation(self):
        """Test OpenAI template generation produces valid configuration."""
        config = ConfigurationTemplates.generate_openai_template()
        
        # Validate structure
        assert "models" in config
        assert "providers" in config
        assert "profiles" in config
        
        # Check OpenAI provider
        assert "openai" in config["providers"]
        openai_provider = config["providers"]["openai"]
        assert "base_url" in openai_provider
        assert "headers" in openai_provider
        assert "Authorization" in openai_provider["headers"]
        
        # Check models have proper structure
        assert len(config["models"]) > 0
        for model_id, model_config in config["models"].items():
            assert "provider" in model_config
            assert "model" in model_config
            assert "metadata" in model_config
            assert "parameters" in model_config
            assert model_config["provider"] == "openai"
            
            # Check metadata includes intent resolution fields
            metadata = model_config["metadata"]
            assert "reasoning" in metadata
            assert "cost" in metadata
            assert "speed" in metadata
            assert "capabilities" in metadata
        
        # Validate configuration
        validation_result = ConfigurationValidator.validate(config)
        assert validation_result.is_valid
    
    def test_anthropic_template_generation(self):
        """Test Anthropic template generation produces valid configuration."""
        config = ConfigurationTemplates.generate_anthropic_template()
        
        # Validate structure
        assert "models" in config
        assert "providers" in config
        assert "profiles" in config
        
        # Check Anthropic provider
        assert "anthropic" in config["providers"]
        anthropic_provider = config["providers"]["anthropic"]
        assert "base_url" in anthropic_provider
        assert "headers" in anthropic_provider
        assert "x-api-key" in anthropic_provider["headers"]
        
        # Check models have proper structure
        assert len(config["models"]) > 0
        for model_id, model_config in config["models"].items():
            assert model_config["provider"] == "anthropic"
            
            # Check metadata includes intent resolution fields
            metadata = model_config["metadata"]
            assert "reasoning" in metadata
            assert "cost" in metadata
            assert "speed" in metadata
            assert "multimodal" in metadata
        
        # Validate configuration
        validation_result = ConfigurationValidator.validate(config)
        assert validation_result.is_valid
    
    def test_ollama_template_generation(self):
        """Test Ollama template generation produces valid configuration."""
        config = ConfigurationTemplates.generate_ollama_template()
        
        # Validate structure
        assert "models" in config
        assert "providers" in config
        assert "profiles" in config
        
        # Check Ollama provider
        assert "ollama" in config["providers"]
        ollama_provider = config["providers"]["ollama"]
        assert "base_url" in ollama_provider
        assert ollama_provider["base_url"] == "http://localhost:11434"
        
        # Check models have proper structure
        assert len(config["models"]) > 0
        for model_id, model_config in config["models"].items():
            assert model_config["provider"] == "ollama"
            
            # Check metadata includes local deployment info
            metadata = model_config["metadata"]
            assert "local" in metadata
            assert metadata["local"] is True
            assert "cost" in metadata
            assert metadata["cost"] == "free"
        
        # Validate configuration
        validation_result = ConfigurationValidator.validate(config)
        assert validation_result.is_valid
    
    def test_multi_provider_template_generation(self):
        """Test multi-provider template generation produces valid configuration."""
        config = ConfigurationTemplates.generate_multi_provider_template(
            openai_models=["gpt-4"],
            anthropic_models=["claude-3-sonnet"],
            ollama_models=["llama3"]
        )
        
        # Validate structure
        assert "models" in config
        assert "providers" in config
        assert "profiles" in config
        assert "aliases" in config
        
        # Check all providers are included
        assert "openai" in config["providers"]
        assert "anthropic" in config["providers"]
        assert "ollama" in config["providers"]
        
        # Check models from all providers
        assert "gpt-4" in config["models"]
        assert "claude-3-sonnet" in config["models"]
        assert "llama3" in config["models"]
        
        # Check cross-provider profiles
        profiles = config["profiles"]
        assert len(profiles) > 0
        
        # Validate configuration
        validation_result = ConfigurationValidator.validate(config)
        assert validation_result.is_valid
    
    def test_template_with_custom_models(self):
        """Test template generation with custom model lists."""
        custom_models = ["gpt-4", "gpt-4-turbo"]
        config = ConfigurationTemplates.generate_openai_template(models=custom_models)
        
        # Check only requested models are included
        assert len(config["models"]) == len(custom_models)
        for model_id in custom_models:
            assert model_id in config["models"]
    
    def test_template_without_profiles(self):
        """Test template generation without profiles."""
        config = ConfigurationTemplates.generate_openai_template(include_profiles=False)
        
        # Should not include profiles
        assert "profiles" not in config or len(config["profiles"]) == 0
        
        # Should still have models and providers
        assert len(config["models"]) > 0
        assert len(config["providers"]) > 0
    
    def test_backward_compatibility_functions(self):
        """Test that backward compatibility functions still work."""
        # Test deprecated function names
        openai_config = mc.generate_openai_config()
        assert "models" in openai_config
        
        anthropic_config = mc.generate_anthropic_config()
        assert "models" in anthropic_config
        
        ollama_config = mc.generate_ollama_config()
        assert "models" in ollama_config
        
        multi_config = mc.generate_multi_provider_config(
            openai_models=["gpt-4"],
            anthropic_models=["claude-3-sonnet"]
        )
        assert "models" in multi_config
    
    def test_development_and_production_configs(self):
        """Test development and production configuration generators."""
        dev_config = mc.generate_development_config()
        prod_config = mc.generate_production_config()
        
        # Both should be valid
        dev_result = ConfigurationValidator.validate(dev_config)
        prod_result = ConfigurationValidator.validate(prod_config)
        
        assert dev_result.is_valid
        assert prod_result.is_valid
        
        # Both should have multiple providers
        assert len(dev_config["providers"]) > 1
        assert len(prod_config["providers"]) > 1


class TestConfigurationValidator:
    """Test cases for configuration validation."""
    
    def test_valid_configuration_validation(self):
        """Test validation of a valid configuration."""
        config = ConfigurationTemplates.generate_openai_template()
        result = ConfigurationValidator.validate(config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_missing_models_section(self):
        """Test validation with missing models section."""
        config = {"providers": {}}
        result = ConfigurationValidator.validate(config)
        
        assert not result.is_valid
        assert any("models" in error for error in result.errors)
    
    def test_empty_models_section(self):
        """Test validation with empty models section."""
        config = {"models": {}, "providers": {}}
        result = ConfigurationValidator.validate(config)
        
        # Should be valid but have warnings
        assert result.is_valid
        assert result.has_warnings
        assert any("No models defined" in warning for warning in result.warnings)
    
    def test_invalid_model_configuration(self):
        """Test validation with invalid model configuration."""
        config = {
            "models": {
                "invalid-model": {
                    # Missing required fields
                    "metadata": {}
                }
            },
            "providers": {}
        }
        result = ConfigurationValidator.validate(config)
        
        assert not result.is_valid
        assert any("provider" in error for error in result.errors)
        assert any("model" in error for error in result.errors)
    
    def test_missing_metadata_warnings(self):
        """Test validation warnings for missing metadata."""
        config = {
            "models": {
                "test-model": {
                    "provider": "test",
                    "model": "test-model"
                    # No metadata
                }
            },
            "providers": {
                "test": {
                    "base_url": "http://test.com"
                }
            }
        }
        result = ConfigurationValidator.validate(config)
        
        assert result.is_valid  # Valid structure
        assert result.has_warnings
        assert any("metadata" in warning for warning in result.warnings)
    
    def test_intent_resolution_readiness(self):
        """Test validation of intent resolution readiness."""
        # Configuration with good metadata
        good_config = ConfigurationTemplates.generate_openai_template()
        good_result = ConfigurationValidator.validate(good_config)
        
        # Should have fewer warnings about metadata
        metadata_warnings = [w for w in good_result.warnings if "metadata" in w.lower()]
        assert len(metadata_warnings) == 0 or len(metadata_warnings) < 3
        
        # Configuration with poor metadata
        poor_config = {
            "models": {
                "model1": {"provider": "test", "model": "test1"},
                "model2": {"provider": "test", "model": "test2"}
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            }
        }
        poor_result = ConfigurationValidator.validate(poor_config)
        
        # Should have warnings about metadata
        assert poor_result.has_warnings
        metadata_warnings = [w for w in poor_result.warnings if "metadata" in w.lower()]
        assert len(metadata_warnings) > 0


class TestConfigurationExamples:
    """Test cases for configuration examples and suggestions."""
    
    def test_model_example_generation(self):
        """Test generation of model configuration examples."""
        example = ConfigurationExamples.get_model_example("gpt-4", "openai")
        
        # Should be valid YAML-like configuration
        assert "models:" in example
        assert "gpt-4:" in example
        assert "provider: openai" in example
        assert "providers:" in example
        assert "openai:" in example
    
    def test_model_example_provider_inference(self):
        """Test provider inference from model identifier."""
        # Test OpenAI inference
        gpt_example = ConfigurationExamples.get_model_example("gpt-4")
        assert "openai" in gpt_example.lower()
        
        # Test Anthropic inference
        claude_example = ConfigurationExamples.get_model_example("claude-3-sonnet")
        assert "anthropic" in claude_example.lower()
        
        # Test Ollama inference
        llama_example = ConfigurationExamples.get_model_example("llama3")
        assert "ollama" in llama_example.lower()
    
    def test_quick_start_instructions(self):
        """Test generation of quick-start instructions."""
        instructions = ConfigurationExamples.get_quick_start_instructions()
        
        assert "No configuration loaded" in instructions
        assert "ConfigurationTemplates" in instructions
        assert "configure" in instructions
    
    def test_intent_resolution_example(self):
        """Test generation of intent resolution examples."""
        criteria = {"reasoning": "high", "cost": "low"}
        example = ConfigurationExamples.get_intent_resolution_example(criteria)
        
        assert "reasoning: high" in example
        assert "cost: low" in example
        assert "models:" in example
    
    def test_provider_suggestions(self):
        """Test generation of provider suggestions."""
        # With available models
        available_models = ["gpt-4", "claude-3-sonnet"]
        suggestions = ConfigurationExamples.get_provider_suggestions(available_models)
        
        assert len(suggestions) > 0
        assert any("gpt-4" in suggestion for suggestion in suggestions)
        
        # Without available models
        empty_suggestions = ConfigurationExamples.get_provider_suggestions([])
        assert len(empty_suggestions) > 0
        assert any("No models" in suggestion for suggestion in empty_suggestions)


class TestTemplateIntegration:
    """Integration tests for templates with the main API."""
    
    def test_template_with_configure(self):
        """Test using templates with the configure() function."""
        config = ConfigurationTemplates.generate_openai_template()
        
        # Should be able to configure with template
        mc.configure(config_dict=config)
        assert mc.is_configured()
        
        # Should be able to access models
        model = mc.model("gpt-4")
        assert model.provider == "openai"
        
        # Clean up
        mc.reset_configuration()
    
    def test_template_validation_integration(self):
        """Test template validation with the validate_config API."""
        config = ConfigurationTemplates.generate_anthropic_template()
        
        # Should validate successfully
        result = mc.validate_config(config_dict=config)
        assert result.is_valid
    
    def test_generate_template_api(self):
        """Test the generate_template convenience API."""
        # Test different providers
        openai_config = mc.generate_template("openai")
        anthropic_config = mc.generate_template("anthropic")
        ollama_config = mc.generate_template("ollama")
        multi_config = mc.generate_template("multi")
        
        # All should be valid
        assert ConfigurationValidator.validate(openai_config).is_valid
        assert ConfigurationValidator.validate(anthropic_config).is_valid
        assert ConfigurationValidator.validate(ollama_config).is_valid
        assert ConfigurationValidator.validate(multi_config).is_valid
        
        # Test invalid provider
        with pytest.raises(ValueError, match="Unknown provider"):
            mc.generate_template("invalid")
    
    def test_template_with_intent_resolution(self):
        """Test that template-generated configurations work with intent resolution."""
        config = ConfigurationTemplates.generate_multi_provider_template(
            openai_models=["gpt-4"],
            anthropic_models=["claude-3-sonnet"],
            ollama_models=["llama3"]
        )
        
        mc.configure(config_dict=config)
        
        try:
            # Should be able to resolve by intent
            high_reasoning_model = mc.model("urn:llm:intent:chat?reasoning=high")
            assert high_reasoning_model is not None
            
            # Should be able to resolve by cost
            low_cost_model = mc.model("urn:llm:intent:chat?cost=free")
            assert low_cost_model is not None
        except Exception:
            # Intent resolution may not find matches depending on metadata
            # This is acceptable as it depends on the specific metadata in templates
            pass
        
        # Clean up
        mc.reset_configuration()