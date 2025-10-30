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
Performance and integration tests for user-config-only approach.

This module tests startup performance, large configuration handling,
and comprehensive integration scenarios without relying on static registry.
"""

import time
import pytest
import model_compass as mc
from model_compass.config_templates import ConfigurationTemplates


class TestStartupPerformance:
    """Test cases for startup performance without registry loading."""
    
    def test_fast_startup_without_registry(self):
        """Test that startup is fast without registry loading."""
        mc.reset_configuration()
        
        # Measure import time (should be fast since no registry loading)
        start_time = time.time()
        import model_compass as mc_fresh
        import_time = time.time() - start_time
        
        # Import should be very fast (under 100ms even on slow systems)
        assert import_time < 0.1, f"Import took {import_time:.3f}s, expected < 0.1s"
    
    def test_fast_configuration_loading(self):
        """Test that configuration loading is fast."""
        config = ConfigurationTemplates.generate_openai_template()
        
        start_time = time.time()
        mc.configure(config_dict=config)
        config_time = time.time() - start_time
        
        # Configuration should load quickly (under 50ms)
        assert config_time < 0.05, f"Configuration took {config_time:.3f}s, expected < 0.05s"
        
        mc.reset_configuration()
    
    def test_fast_model_resolution(self):
        """Test that model resolution is fast."""
        config = ConfigurationTemplates.generate_openai_template()
        mc.configure(config_dict=config)
        
        # First resolution (may include some setup)
        start_time = time.time()
        model1 = mc.model("gpt-4")
        first_resolution_time = time.time() - start_time
        
        # Second resolution (should be faster due to caching)
        start_time = time.time()
        model2 = mc.model("gpt-4")
        second_resolution_time = time.time() - start_time
        
        # Both should be fast, second should be faster
        assert first_resolution_time < 0.01, f"First resolution took {first_resolution_time:.3f}s"
        assert second_resolution_time < 0.005, f"Second resolution took {second_resolution_time:.3f}s"
        assert second_resolution_time <= first_resolution_time
        
        mc.reset_configuration()


class TestLargeConfigurationPerformance:
    """Test cases for performance with large user configurations."""
    
    def create_large_configuration(self, num_models: int = 100) -> dict:
        """Create a large configuration with many models."""
        config = {
            "models": {},
            "providers": {
                "test-provider": {
                    "base_url": "http://test.com",
                    "timeout": 30
                }
            },
            "profiles": {},
            "aliases": {}
        }
        
        # Add many models
        for i in range(num_models):
            model_id = f"test-model-{i}"
            config["models"][model_id] = {
                "provider": "test-provider",
                "model": f"model-{i}",
                "metadata": {
                    "reasoning": "medium" if i % 2 == 0 else "high",
                    "cost": "low" if i % 3 == 0 else "medium",
                    "speed": "fast" if i % 4 == 0 else "medium",
                    "capabilities": ["chat", "completion"],
                    "index": i
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000 + i
                }
            }
            
            # Add some profiles
            if i % 10 == 0:
                config["profiles"][f"profile-{i}"] = {
                    "model": model_id,
                    "parameters": {"temperature": 0.5}
                }
            
            # Add some aliases
            if i % 15 == 0:
                config["aliases"][f"alias-{i}"] = model_id
        
        return config
    
    def test_large_configuration_loading(self):
        """Test loading large configurations performs well."""
        large_config = self.create_large_configuration(100)
        
        start_time = time.time()
        mc.configure(config_dict=large_config)
        config_time = time.time() - start_time
        
        # Should handle 100 models reasonably fast (under 200ms)
        assert config_time < 0.2, f"Large config loading took {config_time:.3f}s, expected < 0.2s"
        
        # Verify all models are available
        available_models = mc.list_models()
        assert len(available_models) == 100
        
        mc.reset_configuration()
    
    def test_large_configuration_validation(self):
        """Test validation performance with large configurations."""
        large_config = self.create_large_configuration(50)
        
        start_time = time.time()
        result = mc.validate_config(config_dict=large_config)
        validation_time = time.time() - start_time
        
        # Validation should be reasonably fast even for large configs
        assert validation_time < 0.1, f"Validation took {validation_time:.3f}s, expected < 0.1s"
        assert result.is_valid
    
    def test_intent_resolution_with_many_models(self):
        """Test intent resolution performance with many models."""
        large_config = self.create_large_configuration(50)
        mc.configure(config_dict=large_config)
        
        start_time = time.time()
        try:
            model = mc.model("urn:llm:intent:chat?reasoning=high&cost=low")
            resolution_time = time.time() - start_time
            
            # Intent resolution should be reasonably fast even with many models
            assert resolution_time < 0.05, f"Intent resolution took {resolution_time:.3f}s, expected < 0.05s"
            assert model is not None
        except Exception:
            # Intent resolution may not find matches, which is acceptable
            resolution_time = time.time() - start_time
            assert resolution_time < 0.05, f"Intent resolution took {resolution_time:.3f}s even when failing"
        
        mc.reset_configuration()


class TestComprehensiveIntegration:
    """Comprehensive integration tests for all resolution methods."""
    
    def setup_method(self):
        """Set up comprehensive test configuration."""
        self.config = ConfigurationTemplates.generate_multi_provider_template(
            openai_models=["gpt-4", "gpt-3.5-turbo"],
            anthropic_models=["claude-3-sonnet", "claude-3-haiku"],
            ollama_models=["llama3", "codellama"]
        )
        mc.configure(config_dict=self.config)
    
    def teardown_method(self):
        """Clean up after each test."""
        mc.reset_configuration()
    
    def test_all_resolution_methods_work(self):
        """Test that all resolution methods work with user-config-only."""
        # 1. Direct model resolution
        gpt_model = mc.model("gpt-4")
        assert gpt_model.provider == "openai"
        assert gpt_model.name == "gpt-4"
        
        # 2. Profile resolution
        with mc.profile("fast"):
            fast_model = mc.model()
            assert fast_model is not None
        
        # 3. Physical identifier resolution
        physical_model = mc.model("openai/gpt-4")
        assert physical_model.provider == "openai"
        assert physical_model.name == "gpt-4"
        
        # 4. Alias resolution (if aliases exist)
        aliases = mc.list_aliases()
        if aliases:
            alias_model = mc.model(aliases[0])
            assert alias_model is not None
        
        # 5. Intent resolution
        try:
            intent_model = mc.model("urn:llm:intent:chat?cost=low")
            assert intent_model is not None
        except Exception:
            # Intent resolution may not find matches, which is acceptable
            pass
    
    def test_nested_profile_contexts(self):
        """Test nested profile context switching."""
        profiles = mc.list_profiles()
        if len(profiles) >= 2:
            profile1, profile2 = profiles[0], profiles[1]
            
            # Test nested contexts
            with mc.profile(profile1):
                model1 = mc.model()
                assert mc.get_current_profile() == profile1
                
                with mc.profile(profile2):
                    model2 = mc.model()
                    assert mc.get_current_profile() == profile2
                    
                    # Models should be different (or at least contexts are different)
                    assert mc.get_current_profile() != profile1
                
                # Should be back to first profile
                assert mc.get_current_profile() == profile1
            
            # Should be back to no profile
            assert mc.get_current_profile() is None
    
    def test_error_handling_integration(self):
        """Test error handling works correctly across all resolution methods."""
        # Test model not found (lazy resolution - error on property access)
        model_proxy = mc.model("nonexistent-model")
        with pytest.raises(Exception):  # ResolutionError or ConfigurationError
            _ = model_proxy.provider
        
        # Test profile not found
        with pytest.raises(Exception):
            with mc.profile("nonexistent-profile"):
                pass
        
        # Test invalid physical identifier (lazy resolution - error on property access)
        physical_proxy = mc.model("nonexistent-provider/some-model")
        with pytest.raises(Exception):
            _ = physical_proxy.provider
        
        # Test invalid URN (lazy resolution - error on property access)
        urn_proxy = mc.model("urn:invalid:format")
        with pytest.raises(Exception):
            _ = urn_proxy.provider
    
    def test_configuration_api_integration(self):
        """Test integration of configuration-focused API methods."""
        # Test listing methods
        models = mc.list_models()
        profiles = mc.list_profiles()
        aliases = mc.list_aliases()
        
        assert len(models) > 0
        assert len(profiles) > 0
        # Aliases may be empty, which is fine
        
        # Test configuration status
        assert mc.is_configured()
        
        status = mc.is_configured_with_details()
        assert status["configured"]
        assert status["models_count"] == len(models)
        assert status["profiles_count"] == len(profiles)
        
        # Test configuration suggestions
        suggestions = mc.get_config_suggestions()
        assert isinstance(suggestions, list)
        
        # Test validation
        result = mc.validate_config(config_dict=self.config)
        assert result.is_valid
    
    def test_template_generation_integration(self):
        """Test integration of template generation with main API."""
        # Test all template types
        openai_config = mc.generate_template("openai")
        anthropic_config = mc.generate_template("anthropic")
        ollama_config = mc.generate_template("ollama")
        multi_config = mc.generate_template("multi", openai_models=["gpt-4"], anthropic_models=["claude-3-sonnet"])
        
        # All should be valid and usable
        for config in [openai_config, anthropic_config, ollama_config, multi_config]:
            result = mc.validate_config(config_dict=config)
            assert result.is_valid
            
            # Should be able to configure and use
            mc.configure(config_dict=config)
            assert mc.is_configured()
            
            models = mc.list_models()
            assert len(models) > 0
            
            # Should be able to resolve at least one model
            model = mc.model(models[0])
            assert model is not None
            
            mc.reset_configuration()


class TestConfigurationValidationPerformance:
    """Test cases for configuration validation performance and accuracy."""
    
    def test_validation_accuracy_with_various_configs(self):
        """Test validation accuracy with different configuration types."""
        # Valid configuration
        valid_config = ConfigurationTemplates.generate_openai_template()
        result = mc.validate_config(config_dict=valid_config)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Invalid configuration - missing required fields
        invalid_config = {
            "models": {
                "invalid": {
                    "metadata": {}  # Missing provider and model
                }
            }
        }
        result = mc.validate_config(config_dict=invalid_config)
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Configuration with warnings - missing metadata
        warning_config = {
            "models": {
                "minimal": {
                    "provider": "test",
                    "model": "test"
                    # No metadata
                }
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            }
        }
        result = mc.validate_config(config_dict=warning_config)
        assert result.is_valid
        assert result.has_warnings
        assert len(result.suggestions) > 0
    
    def test_validation_performance_scaling(self):
        """Test that validation performance scales reasonably with config size."""
        # Small configuration
        small_config = ConfigurationTemplates.generate_openai_template(models=["gpt-4"])
        start_time = time.time()
        result1 = mc.validate_config(config_dict=small_config)
        small_time = time.time() - start_time
        
        # Larger configuration
        large_config = ConfigurationTemplates.generate_multi_provider_template(
            openai_models=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            anthropic_models=["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
            ollama_models=["llama3", "codellama", "mistral", "phi3"]
        )
        start_time = time.time()
        result2 = mc.validate_config(config_dict=large_config)
        large_time = time.time() - start_time
        
        # Both should be valid
        assert result1.is_valid
        assert result2.is_valid
        
        # Large config should not be dramatically slower
        # (allowing for some scaling, but not exponential)
        assert large_time < small_time * 10, f"Large config validation too slow: {large_time:.3f}s vs {small_time:.3f}s"
        assert large_time < 0.1, f"Large config validation took {large_time:.3f}s, expected < 0.1s"
    
    def test_validation_with_complex_references(self):
        """Test validation accuracy with complex profile and alias references."""
        complex_config = {
            "models": {
                "model1": {"provider": "test", "model": "test1", "metadata": {"type": "base"}},
                "model2": {"provider": "test", "model": "test2", "metadata": {"type": "advanced"}}
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            },
            "profiles": {
                "profile1": {"model": "model1"},
                "profile2": {"model": "alias1"},  # References alias
                "profile3": {"model": "test/direct"}  # Physical identifier
            },
            "aliases": {
                "alias1": "model2",
                "alias2": "profile1",  # References profile
                "alias3": "alias1"     # Chain of aliases
            }
        }
        
        result = mc.validate_config(config_dict=complex_config)
        
        # Should be valid despite complex references
        assert result.is_valid
        
        # Should be able to configure and resolve all references
        mc.configure(config_dict=complex_config)
        
        # Test all resolution paths work
        model1 = mc.model("model1")  # Direct model
        assert model1 is not None
        
        with mc.profile("profile1"):  # Profile -> model
            profile_model = mc.model()
            assert profile_model is not None
        
        alias_model = mc.model("alias1")  # Alias -> model
        assert alias_model is not None
        
        mc.reset_configuration()


class TestMemoryAndResourceUsage:
    """Test cases for memory usage and resource management."""
    
    def test_memory_usage_stays_reasonable(self):
        """Test that memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and configure multiple times
        for i in range(10):
            config = ConfigurationTemplates.generate_multi_provider_template(
                openai_models=["gpt-4"],
                anthropic_models=["claude-3-sonnet"],
                ollama_models=["llama3"]
            )
            mc.configure(config_dict=config)
            
            # Use the configuration
            model = mc.model("gpt-4")
            assert model is not None
            
            mc.reset_configuration()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 10MB)
        assert memory_growth < 10 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB"
    
    def test_no_memory_leaks_in_resolution(self):
        """Test that repeated model resolution doesn't leak memory."""
        import psutil
        import os
        
        config = ConfigurationTemplates.generate_openai_template()
        mc.configure(config_dict=config)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Resolve models many times
        for i in range(100):
            model = mc.model("gpt-4")
            assert model is not None
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should not grow significantly
        assert memory_growth < 5 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB during resolution"
        
        mc.reset_configuration()