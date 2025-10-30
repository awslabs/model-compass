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
Unit tests for core API functions and profile context switching.
"""

import pytest
import tempfile
from pathlib import Path

import model_compass as mc
from model_compass.exceptions import ConfigurationError, ResolutionError
from model_compass.proxy import ModelProxy


class TestCoreAPI:
    """Test cases for core API functions."""
    
    def test_configure_with_dict(self, sample_config_dict):
        """Test configuring with dictionary."""
        mc.configure(config_dict=sample_config_dict)
        
        assert mc.is_configured()
        
        # Test that models are accessible
        model = mc.model("test-openai")
        assert isinstance(model, ModelProxy)
        assert model.provider == "openai"
    
    def test_configure_with_yaml_file(self, temp_yaml_config):
        """Test configuring with YAML file."""
        mc.configure(config_path=temp_yaml_config)
        
        assert mc.is_configured()
        
        # Test that configuration loaded correctly
        model = mc.model("test-anthropic")
        assert model.provider == "anthropic"
    
    def test_configure_with_json_file(self, temp_json_config):
        """Test configuring with JSON file."""
        mc.configure(config_path=temp_json_config)
        
        assert mc.is_configured()
        
        # Test that configuration loaded correctly
        model = mc.model("test-ollama")
        assert model.provider == "ollama"
        assert model.is_local
    
    def test_configure_validation_errors(self):
        """Test configure function input validation."""
        # No arguments
        with pytest.raises(ValueError) as exc_info:
            mc.configure()
        assert "Either config_path or config_dict must be provided" in str(exc_info.value)
        
        # Both arguments
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_path="test.yaml", config_dict={})
        assert "Cannot provide both config_path and config_dict" in str(exc_info.value)
        
        # Invalid config_path type
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_path=123)
        assert "config_path must be a string or Path object" in str(exc_info.value)
        
        # Empty config_path
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_path="")
        assert "config_path cannot be empty" in str(exc_info.value)
        
        # Invalid config_dict type
        with pytest.raises(ValueError) as exc_info:
            mc.configure(config_dict="not a dict")
        assert "config_dict must be a dictionary" in str(exc_info.value)
    
    def test_model_function_validation(self, configured_mc):
        """Test model function input validation."""
        # Invalid identifier type
        with pytest.raises(ValueError) as exc_info:
            mc.model(123)
        assert "identifier must be a string or None" in str(exc_info.value)
        
        # Empty identifier
        with pytest.raises(ValueError) as exc_info:
            mc.model("")
        assert "identifier cannot be an empty string" in str(exc_info.value)
        
        # Valid None identifier
        model = mc.model(None)
        assert isinstance(model, ModelProxy)
        
        # Valid string identifier
        model = mc.model("test-openai")
        assert isinstance(model, ModelProxy)
    
    def test_model_function_not_configured(self):
        """Test model function when not configured."""
        with pytest.raises(ConfigurationError) as exc_info:
            mc.model("test")
        assert "Library not configured" in str(exc_info.value)
    
    def test_is_configured_function(self, sample_config_dict):
        """Test is_configured function."""
        # Initially not configured
        assert not mc.is_configured()
        
        # After configuration
        mc.configure(config_dict=sample_config_dict)
        assert mc.is_configured()
    
    def test_global_collections(self, configured_mc):
        """Test global collection objects."""
        # Test models collection
        assert "test-openai" in mc.models
        model_list = mc.models.list()
        assert len(model_list) > 0
        
        # Test profiles collection
        assert "default" in mc.profiles
        profile_list = mc.profiles.list()
        assert "default" in profile_list
        
        # Test aliases collection
        assert "main" in mc.aliases
        alias_list = mc.aliases.list()
        assert "main" in alias_list
    
    def test_configuration_state_cleanup(self, sample_config_dict):
        """Test that configuration state is properly cleaned up on failure."""
        # Configure successfully first
        mc.configure(config_dict=sample_config_dict)
        assert mc.is_configured()
        
        # Try to configure with invalid data
        with pytest.raises(ConfigurationError):
            mc.configure(config_dict={
                "models": {
                    "invalid-model": {
                        "provider": "nonexistent-provider",  # This should cause validation error
                        "model": "test"
                    }
                },
                "providers": {}  # Empty providers but model references nonexistent-provider
            })
        
        # Configuration should be cleaned up after failure
        assert not mc.is_configured()
        
        # Reconfigure with valid config
        mc.configure(config_dict=sample_config_dict)
        assert mc.is_configured()
        model = mc.model("test-openai")
        assert model.provider == "openai"


class TestProfileContextSwitching:
    """Test cases for profile context switching."""
    
    def test_basic_profile_context(self, configured_mc):
        """Test basic profile context switching."""
        # Initially no profile
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
        
        # Enter profile context
        with mc.profile("fast") as active_profile:
            assert active_profile == "fast"
            assert mc.get_current_profile() == "fast"
            assert mc.get_profile_stack() == [None]
            
            # Model should use the profile
            model = mc.model()
            assert model.parameters["temperature"] == 0.3  # fast profile temperature
        
        # After context, should be restored
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
    
    def test_nested_profile_contexts(self, configured_mc):
        """Test nested profile context switching."""
        with mc.profile("fast"):
            assert mc.get_current_profile() == "fast"
            assert mc.get_profile_stack() == [None]
            
            with mc.profile("smart"):
                assert mc.get_current_profile() == "smart"
                assert mc.get_profile_stack() == [None, "fast"]
                
                # Model should use inner profile
                model = mc.model()
                assert model.provider == "anthropic"  # smart profile uses anthropic
                
                with mc.profile("local"):
                    assert mc.get_current_profile() == "local"
                    assert mc.get_profile_stack() == [None, "fast", "smart"]
                    
                    # Model should use innermost profile
                    model = mc.model()
                    assert model.provider == "ollama"  # local profile uses ollama
                
                # Back to smart profile
                assert mc.get_current_profile() == "smart"
                assert mc.get_profile_stack() == [None, "fast"]
            
            # Back to fast profile
            assert mc.get_current_profile() == "fast"
            assert mc.get_profile_stack() == [None]
        
        # Back to default
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
    
    def test_profile_context_with_exceptions(self, configured_mc):
        """Test profile context restoration when exceptions occur."""
        assert mc.get_current_profile() is None
        
        try:
            with mc.profile("fast"):
                assert mc.get_current_profile() == "fast"
                
                # Raise exception inside context
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Profile should be restored even after exception
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
    
    def test_nested_profile_context_with_exceptions(self, configured_mc):
        """Test nested profile context restoration with exceptions."""
        with mc.profile("fast"):
            assert mc.get_current_profile() == "fast"
            
            try:
                with mc.profile("smart"):
                    assert mc.get_current_profile() == "smart"
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Should be back to outer profile
            assert mc.get_current_profile() == "fast"
            assert mc.get_profile_stack() == [None]
        
        # Should be back to default
        assert mc.get_current_profile() is None
        assert mc.get_profile_stack() == []
    
    def test_profile_validation(self, configured_mc):
        """Test profile context validation."""
        # Invalid profile name type
        with pytest.raises(ValueError) as exc_info:
            with mc.profile(123):
                pass
        assert "Profile name must be a non-empty string" in str(exc_info.value)
        
        # Empty profile name
        with pytest.raises(ValueError) as exc_info:
            with mc.profile(""):
                pass
        assert "Profile name must be a non-empty string" in str(exc_info.value)
        
        # None profile name
        with pytest.raises(ValueError) as exc_info:
            with mc.profile(None):
                pass
        assert "Profile name must be a non-empty string" in str(exc_info.value)
        
        # Nonexistent profile
        with pytest.raises(ResolutionError) as exc_info:
            with mc.profile("nonexistent"):
                pass
        assert "Profile 'nonexistent' not found" in str(exc_info.value)
        assert "available_profiles" in exc_info.value.context
    
    def test_profile_context_not_configured(self):
        """Test profile context when not configured."""
        with pytest.raises(ConfigurationError) as exc_info:
            with mc.profile("test"):
                pass
        assert "Library not configured" in str(exc_info.value)
    
    def test_reset_profile_context(self, configured_mc):
        """Test resetting profile context."""
        # Set up nested contexts
        with mc.profile("fast"):
            with mc.profile("smart"):
                assert mc.get_current_profile() == "smart"
                assert len(mc.get_profile_stack()) == 2
                
                # Reset context
                mc.reset_profile_context()
                
                assert mc.get_current_profile() is None
                assert mc.get_profile_stack() == []
    
    def test_profile_context_with_model_resolution(self, configured_mc):
        """Test that profile context affects model resolution."""
        # Default model
        default_model = mc.model()
        assert default_model.provider == "openai"  # default profile
        
        # Model in profile context
        with mc.profile("smart"):
            smart_model = mc.model()
            assert smart_model.provider == "anthropic"  # smart profile
            
            # Explicit identifier should override context
            explicit_model = mc.model("test-ollama")
            assert explicit_model.provider == "ollama"
        
        # Back to default
        restored_model = mc.model()
        assert restored_model.provider == "openai"
    
    def test_profile_context_stack_integrity(self, configured_mc):
        """Test profile context stack integrity under various conditions."""
        # Test multiple rapid context switches
        for i in range(5):
            with mc.profile("fast"):
                assert mc.get_current_profile() == "fast"
                with mc.profile("smart"):
                    assert mc.get_current_profile() == "smart"
                assert mc.get_current_profile() == "fast"
            assert mc.get_current_profile() is None
        
        # Stack should be clean
        assert mc.get_profile_stack() == []
    
    def test_profile_context_whitespace_handling(self, configured_mc):
        """Test profile context with whitespace in names."""
        # Whitespace should be stripped
        with mc.profile("  fast  ") as active_profile:
            assert active_profile == "fast"
            assert mc.get_current_profile() == "fast"