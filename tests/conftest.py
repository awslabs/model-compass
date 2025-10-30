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
Pytest configuration and shared fixtures for Model Compass tests.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any

import model_compass as mc
from model_compass.data_models import ModelConfig, Profile, ProviderConfig


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "default_profile": "default",
        "resolution_timeout": 30,
        "models": {
            "test-openai": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "context_length": 4096,
                    "capabilities": ["chat", "completion"],
                    "deployment": "cloud"
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            },
            "test-anthropic": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "medium",
                    "context_length": 200000,
                    "capabilities": ["chat", "analysis"],
                    "deployment": "cloud"
                },
                "parameters": {
                    "temperature": 0.5,
                    "max_tokens": 2048
                }
            },
            "test-ollama": {
                "provider": "ollama",
                "model": "llama3",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "medium",
                    "context_length": 8192,
                    "capabilities": ["chat", "completion"],
                    "deployment": "local"
                },
                "parameters": {
                    "temperature": 0.7,
                    "num_predict": 1024
                }
            }
        },
        "profiles": {
            "default": {
                "model": "test-openai",
                "parameters": {"temperature": 0.1}
            },
            "fast": {
                "model": "test-openai",
                "parameters": {"temperature": 0.3, "max_tokens": 512}
            },
            "smart": {
                "model": "test-anthropic",
                "parameters": {"temperature": 0.0}
            },
            "local": {
                "model": "test-ollama",
                "parameters": {"temperature": 0.8}
            }
        },
        "aliases": {
            "main": "default",
            "backup": "fast",
            "dev": "local",
            "chain1": "chain2",
            "chain2": "main"
        },
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "headers": {
                    "Authorization": "Bearer ${OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "timeout": 45,
                "headers": {
                    "x-api-key": "${ANTHROPIC_API_KEY}",
                    "Content-Type": "application/json"
                }
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 60,
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }
    }


@pytest.fixture
def circular_alias_config():
    """Configuration with circular alias references for testing."""
    return {
        "models": {
            "test-model": {
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            }
        },
        "aliases": {
            "alias1": "alias2",
            "alias2": "alias3",
            "alias3": "alias1"  # Creates circular reference
        },
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1"
            }
        }
    }


@pytest.fixture
def temp_yaml_config(sample_config_dict):
    """Create a temporary YAML configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        f.flush()
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_json_config(sample_config_dict):
    """Create a temporary JSON configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config_dict, f, indent=2)
        f.flush()
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def invalid_yaml_config():
    """Create a temporary invalid YAML configuration file."""
    invalid_yaml = """
models:
  test-model:
    provider: openai
    model: gpt-3.5-turbo
  invalid_indentation:
provider: anthropic  # Wrong indentation
    model: claude-3
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(invalid_yaml)
        f.flush()
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def invalid_json_config():
    """Create a temporary invalid JSON configuration file."""
    invalid_json = """
{
  "models": {
    "test-model": {
      "provider": "openai",
      "model": "gpt-3.5-turbo"
    }
  },  // Invalid comment in JSON
  "trailing_comma": true,
}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(invalid_json)
        f.flush()
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def reset_model_compass():
    """Reset Model Compass state before each test."""
    # Reset entire configuration state
    mc.reset_configuration()
    
    # Model registry no longer exists - using user-config-only approach
    
    # Disable verbose logging for tests (unless specifically enabled)
    mc.disable_verbose_logging()
    
    yield
    
    # Cleanup after test
    mc.reset_configuration()


@pytest.fixture
def configured_mc(sample_config_dict):
    """Model Compass instance configured with sample data."""
    mc.configure(config_dict=sample_config_dict)
    return mc


@pytest.fixture
def sample_model_config():
    """Sample ModelConfig instance for testing."""
    return ModelConfig(
        provider="test-provider",
        model="test-model",
        metadata={
            "reasoning": "high",
            "cost": "medium",
            "capabilities": ["chat", "completion"]
        },
        parameters={
            "temperature": 0.7,
            "max_tokens": 1024
        }
    )


@pytest.fixture
def sample_provider_config():
    """Sample ProviderConfig instance for testing."""
    return ProviderConfig(
        base_url="https://api.test.com/v1",
        timeout=30,
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }
    )


@pytest.fixture
def sample_profile():
    """Sample Profile instance for testing."""
    return Profile(
        model="test-model",
        parameters={
            "temperature": 0.5,
            "max_tokens": 2048
        }
    )