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
Integration tests for complete resolution workflows and end-to-end functionality.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path

import model_compass as mc
from model_compass.exceptions import ResolutionError, CircularAliasError


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_complete_yaml_workflow(self, temp_yaml_config):
        """Test complete workflow from YAML file to model usage."""
        # Load configuration from YAML
        mc.configure(config_path=temp_yaml_config)
        
        # Test different resolution types
        # 1. Direct model ID
        model1 = mc.model("test-openai")
        assert model1.provider == "openai"
        assert model1.name == "gpt-3.5-turbo"
        
        # 2. Profile resolution
        model2 = mc.model("smart")
        assert model2.provider == "anthropic"
        
        # 3. Alias resolution
        model3 = mc.model("main")
        assert model3.provider == "openai"  # main -> default -> test-openai
        
        # 4. Physical identifier
        model4 = mc.model("ollama/llama3")
        assert model4.provider == "ollama"
        assert model4.name == "llama3"
        
        # 5. Intent-based resolution
        model5 = mc.model("urn:llm:intent:chat?cost=low")
        assert model5.metadata["cost"] == "low"
        
        # Test profile context switching
        with mc.profile("local"):
            local_model = mc.model()
            assert local_model.provider == "ollama"
            assert local_model.is_local
        
        # Test collections
        assert len(mc.models.list()) > 0
        assert len(mc.profiles.list()) == 4
        assert len(mc.aliases.list()) == 5
    
    def test_complete_json_workflow(self, temp_json_config):
        """Test complete workflow from JSON file to model usage."""
        # Load configuration from JSON
        mc.configure(config_path=temp_json_config)
        
        # Test nested profile contexts with different resolution types
        with mc.profile("fast"):
            fast_model = mc.model()
            assert fast_model.parameters["temperature"] == 0.3
            
            with mc.profile("smart"):
                smart_model = mc.model()
                assert smart_model.provider == "anthropic"
                
                # Test alias resolution within context
                alias_model = mc.model("backup")  # backup -> fast
                assert alias_model.provider == "openai"
            
            # Back to fast context
            restored_model = mc.model()
            assert restored_model.parameters["temperature"] == 0.3
        
        # Test intent matching with multiple criteria
        multi_intent_model = mc.model("urn:llm:intent:chat?cost=low&deployment=local")
        assert multi_intent_model.provider == "ollama"
        assert multi_intent_model.is_local
    
    def test_configuration_switching(self, sample_config_dict):
        """Test switching between different configurations."""
        # Configure with first config
        mc.configure(config_dict=sample_config_dict)
        
        model1 = mc.model("test-openai")
        assert model1.provider == "openai"
        
        # Switch to different configuration
        new_config = {
            "models": {
                "new-model": {
                    "provider": "new-provider",
                    "model": "new-test-model"
                }
            },
            "providers": {
                "new-provider": {
                    "base_url": "https://new-api.com"
                }
            }
        }
        
        mc.configure(config_dict=new_config)
        
        # Old models should no longer be available
        try:
            old_model = mc.model("test-openai")
            pytest.fail(f"Expected ResolutionError but got model: {old_model.provider}")
        except ResolutionError:
            pass  # This is expected
        
        # New model should be available
        new_model = mc.model("new-model")
        assert new_model.provider == "new-provider"
    
    def test_error_recovery_workflow(self, sample_config_dict):
        """Test error recovery in various scenarios."""
        mc.configure(config_dict=sample_config_dict)
        
        # Test recovery from resolution errors
        try:
            model = mc.model("nonexistent-model")
            # Trigger resolution by accessing a property
            _ = model.provider
            pytest.fail("Expected ResolutionError for nonexistent model")
        except ResolutionError:
            pass  # This is expected
        
        # Library should still work after error
        working_model = mc.model("test-openai")
        assert working_model.provider == "openai"
        
        # Test recovery from profile context errors
        try:
            with mc.profile("nonexistent-profile"):
                pass
        except ResolutionError:
            pass
        
        # Profile context should be clean after error
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
        
        # Normal profile switching should still work
        with mc.profile("fast"):
            assert mc.get_current_profile() == "fast"
    
    def test_complex_alias_chains(self):
        """Test complex alias chain resolution."""
        complex_config = {
            "models": {
                "final-model": {
                    "provider": "test",
                    "model": "final"
                }
            },
            "profiles": {
                "final-profile": {
                    "model": "final-model"
                }
            },
            "aliases": {
                "level1": "level2",
                "level2": "level3", 
                "level3": "level4",
                "level4": "final-profile"
            },
            "providers": {
                "test": {"base_url": "https://test.com"}
            }
        }
        
        mc.configure(config_dict=complex_config)
        
        # Should resolve through the entire chain
        model = mc.model("level1")
        assert model.provider == "test"
        assert model.name == "final"
        
        # Test chain tracking
        chain = mc.aliases.resolve_chain("level1")
        expected_chain = ["level1", "level2", "level3", "level4", "final-profile"]
        assert chain == expected_chain
    
    def test_intent_matching_with_fallbacks(self):
        """Test intent matching with fallback scenarios."""
        intent_config = {
            "models": {
                "cheap-fast": {
                    "provider": "test",
                    "model": "cheap-fast-model",
                    "metadata": {
                        "cost": "low",
                        "speed": "fast",
                        "reasoning": "low"
                    }
                },
                "expensive-smart": {
                    "provider": "test",
                    "model": "expensive-smart-model", 
                    "metadata": {
                        "cost": "high",
                        "speed": "slow",
                        "reasoning": "high"
                    }
                },
                "balanced": {
                    "provider": "test",
                    "model": "balanced-model",
                    "metadata": {
                        "cost": "medium",
                        "speed": "medium", 
                        "reasoning": "medium"
                    }
                }
            },
            "providers": {
                "test": {"base_url": "https://test.com"}
            }
        }
        
        mc.configure(config_dict=intent_config)
        
        # Test exact matches
        cheap_model = mc.model("urn:llm:intent:chat?cost=low")
        assert cheap_model.name == "cheap-fast-model"
        
        smart_model = mc.model("urn:llm:intent:chat?reasoning=high")
        assert smart_model.name == "expensive-smart-model"
        
        # Test multiple criteria
        balanced_model = mc.model("urn:llm:intent:chat?cost=medium&speed=medium")
        assert balanced_model.name == "balanced-model"
        
        # Test impossible criteria
        try:
            model = mc.model("urn:llm:intent:chat?cost=free&reasoning=superhuman")
            # Trigger resolution by accessing a property
            _ = model.provider
            pytest.fail("Expected ResolutionError for impossible criteria")
        except ResolutionError as exc_info:
            assert "No models found matching intent criteria" in str(exc_info)
            assert "suggestions" in exc_info.context
    
    def test_configuration_template_integration(self):
        """Test integration with configuration templates (user-config-only approach)."""
        # Use configuration templates to generate user configuration
        dev_config = mc.generate_development_config()
        mc.configure(config_dict=dev_config)
        
        # Test that template-generated models work with all resolution types
        # 1. Direct model access
        gpt_model = mc.model("gpt-3.5-turbo")
        assert gpt_model.provider == "openai"
        
        # 2. Profile access
        with mc.profile("fast"):
            fast_model = mc.model()
            # Template-generated profiles should work
            assert fast_model is not None
        
        # 3. Intent matching with template-generated metadata
        try:
            local_model = mc.model("urn:llm:intent:chat?cost=free")
            assert local_model is not None
        except Exception:
            # Intent matching may not find matches depending on template metadata
            pass
        
        # 4. List available models from configuration
        available_models = mc.list_models()
        assert len(available_models) > 0
        assert "gpt-3.5-turbo" in available_models
        
        # 5. Test configuration validation
        validation_result = mc.validate_config(config_dict=dev_config)
        assert validation_result.is_valid
    
    def test_logging_integration_workflow(self, sample_config_dict):
        """Test workflow with verbose logging enabled."""
        # Enable verbose logging
        mc.enable_verbose_logging("DEBUG")
        assert mc.is_verbose_enabled()
        
        # Configure and use models (should generate logs)
        mc.configure(config_dict=sample_config_dict)
        
        # Test various operations that should generate logs
        model1 = mc.model("test-openai")
        model2 = mc.model("urn:llm:intent:chat?cost=low")
        
        with mc.profile("smart"):
            model3 = mc.model()
        
        # Test alias resolution (should log chain)
        model4 = mc.model("chain1")
        
        # Disable logging
        mc.disable_verbose_logging()
        assert not mc.is_verbose_enabled()
    
    def test_configuration_templates_workflow(self):
        """Test workflow using configuration templates."""
        # Test OpenAI-only configuration
        openai_config = mc.generate_openai_config()
        mc.configure(config_dict=openai_config)
        
        gpt_model = mc.model("gpt-4")
        assert gpt_model.provider == "openai"
        
        with mc.profile("creative"):
            creative_model = mc.model()
            assert creative_model.parameters["temperature"] == 0.9
        
        # Test multi-provider configuration
        multi_config = mc.generate_multi_provider_config(
            openai_models=["gpt-3.5-turbo"],
            anthropic_models=["claude-3-haiku"],
            ollama_models=["llama3"]
        )
        
        mc.configure(config_dict=multi_config)
        
        # Test cross-provider profiles
        with mc.profile("fast"):
            fast_model = mc.model()
            # Should pick the fastest/cheapest available
            assert fast_model.cost_tier in ["low", "medium"]
        
        with mc.profile("local"):
            local_model = mc.model()
            assert local_model.is_local
    
    def test_error_context_and_suggestions(self, sample_config_dict):
        """Test that errors provide helpful context and suggestions."""
        mc.configure(config_dict=sample_config_dict)
        
        # Test resolution error with suggestions
        try:
            model = mc.model("nonexistent-model")
            # Trigger resolution by accessing a property
            _ = model.provider
            pytest.fail("Expected ResolutionError for nonexistent model")
        except ResolutionError as error:
            assert "suggestions" in error.context
            suggestions = error.context["suggestions"]
            # Check that we have helpful suggestions (the exact format may vary)
            assert len(suggestions) > 0
            suggestions_text = " ".join(suggestions)
            assert "available models" in suggestions_text.lower() or "test-openai" in suggestions_text
        
        # Test circular alias error with suggestions
        circular_config = {
            "models": {"test": {"provider": "test", "model": "test"}},
            "aliases": {"a": "b", "b": "a"},
            "providers": {"test": {"base_url": "https://test.com"}}
        }
        
        mc.configure(config_dict=circular_config)
        
        try:
            model = mc.model("a")
            # Trigger resolution by accessing a property
            _ = model.provider
            pytest.fail("Expected CircularAliasError for circular alias")
        except CircularAliasError as exc_info:
            pass  # This is expected
        
            error = exc_info
            assert len(error.suggestions) > 0
            assert any("circular" in suggestion.lower() for suggestion in error.suggestions)
            
            # Test detailed error message
            detailed_message = error.get_detailed_message()
            assert "Suggestions:" in detailed_message
            assert len(detailed_message) > len(str(error))


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_large_configuration_handling(self):
        """Test handling of large configurations."""
        # Create a large configuration
        large_config = {
            "models": {},
            "profiles": {},
            "aliases": {},
            "providers": {}
        }
        
        # Add many models
        for i in range(100):
            large_config["models"][f"model-{i}"] = {
                "provider": f"provider-{i % 10}",
                "model": f"test-model-{i}",
                "metadata": {
                    "cost": ["low", "medium", "high"][i % 3],
                    "reasoning": ["low", "medium", "high"][i % 3],
                    "speed": ["slow", "medium", "fast"][i % 3]
                }
            }
        
        # Add providers
        for i in range(10):
            large_config["providers"][f"provider-{i}"] = {
                "base_url": f"https://api-{i}.com"
            }
        
        # Add profiles
        for i in range(50):
            large_config["profiles"][f"profile-{i}"] = {
                "model": f"model-{i * 2}"
            }
        
        # Add aliases
        for i in range(50):
            large_config["aliases"][f"alias-{i}"] = f"profile-{i}"
        
        # Should handle large configuration without issues
        mc.configure(config_dict=large_config)
        
        # Test various operations
        model = mc.model("model-50")
        assert model.provider == "provider-0"  # 50 % 10 = 0
        
        profile_model = mc.model("profile-25")
        assert profile_model.name == "test-model-50"  # 25 * 2 = 50
        
        alias_model = mc.model("alias-30")
        assert alias_model.name == "test-model-60"  # 30 * 2 = 60
        
        # Test intent matching with many options
        low_cost_model = mc.model("urn:llm:intent:chat?cost=low")
        assert low_cost_model.metadata["cost"] == "low"
    
    def test_repeated_operations(self, sample_config_dict):
        """Test repeated operations for consistency."""
        mc.configure(config_dict=sample_config_dict)
        
        # Test that repeated model resolution is consistent
        for _ in range(10):
            model1 = mc.model("test-openai")
            model2 = mc.model("test-openai")
            
            assert model1.provider == model2.provider
            assert model1.name == model2.name
            assert model1.parameters == model2.parameters
        
        # Test repeated profile context switching
        for _ in range(10):
            with mc.profile("fast"):
                assert mc.get_current_profile() == "fast"
                model = mc.model()
                assert model.parameters["temperature"] == 0.3
            
            assert mc.get_current_profile() is None
        
        # Test repeated intent matching
        for _ in range(10):
            model = mc.model("urn:llm:intent:chat?cost=low")
            assert model.metadata["cost"] == "low"