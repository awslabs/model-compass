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
Multi-Environment Setup Example

This example shows how to set up Model Compass for different environments
(development, staging, production) with appropriate model selections and fallbacks.
"""

import os
import model_compass as mc


def get_environment_config():
    """Get configuration based on current environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return get_production_config()
    elif env == "staging":
        return get_staging_config()
    else:
        return get_development_config()


def get_development_config():
    """Development configuration with local models and cloud fallbacks."""
    return {
        "default_profile": "dev",
        "models": {
            # Local models for development
            "llama3": {
                "provider": "ollama",
                "model": "llama3",
                "metadata": {
                    "reasoning": "high",
                    "cost": "free",
                    "speed": "medium",
                    "deployment": "local"
                },
                "parameters": {"temperature": 0.8, "num_predict": 1024}
            },
            "codellama": {
                "provider": "ollama", 
                "model": "codellama",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["code"],
                    "deployment": "local"
                },
                "parameters": {"temperature": 0.1, "num_predict": 2048}
            },
            # Cloud fallback for when local isn't available
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            }
        },
        "providers": {
            "ollama": {
                "base_url": "http://localhost:11434",
                "timeout": 120
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "headers": {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            }
        },
        "profiles": {
            "dev": {
                "model": "llama3",
                "parameters": {"temperature": 0.9}
            },
            "code": {
                "model": "codellama",
                "parameters": {"temperature": 0.1}
            },
            "fallback": {
                "model": "gpt-3.5-turbo",
                "parameters": {"temperature": 0.7}
            }
        },
        "aliases": {
            "main": "llama3",
            "backup": "gpt-3.5-turbo",
            "coder": "codellama"
        }
    }


def get_staging_config():
    """Staging configuration with fast cloud models."""
    return {
        "default_profile": "staging",
        "models": {
            "claude-3-haiku": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.5, "max_tokens": 2048}
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low", 
                    "speed": "fast",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.5, "max_tokens": 2048}
            }
        },
        "providers": {
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "headers": {"x-api-key": os.getenv("ANTHROPIC_API_KEY")}
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "headers": {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            }
        },
        "profiles": {
            "staging": {
                "model": "claude-3-haiku",
                "parameters": {"temperature": 0.5}
            },
            "testing": {
                "model": "gpt-3.5-turbo",
                "parameters": {"temperature": 0.3}
            }
        },
        "aliases": {
            "main": "claude-3-haiku",
            "backup": "gpt-3.5-turbo"
        }
    }


def get_production_config():
    """Production configuration with high-quality models and fallbacks."""
    return {
        "default_profile": "production",
        "models": {
            "gpt-4": {
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {
                    "reasoning": "high",
                    "cost": "high",
                    "speed": "medium",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.1, "max_tokens": 2048}
            },
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "medium",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.1, "max_tokens": 4096}
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "deployment": "cloud"
                },
                "parameters": {"temperature": 0.1, "max_tokens": 2048}
            }
        },
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "timeout": 30,
                "headers": {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "timeout": 60,
                "headers": {"x-api-key": os.getenv("ANTHROPIC_API_KEY")}
            }
        },
        "profiles": {
            "production": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.1}
            },
            "analysis": {
                "model": "claude-3-sonnet",
                "parameters": {"temperature": 0.3}
            },
            "fallback": {
                "model": "gpt-3.5-turbo",
                "parameters": {"temperature": 0.1}
            }
        },
        "aliases": {
            "primary": "gpt-4",
            "secondary": "claude-3-sonnet",
            "emergency": "gpt-3.5-turbo",
            "main": "primary"
        }
    }


def setup_model_compass():
    """Set up Model Compass with environment-appropriate configuration."""
    config = get_environment_config()
    
    # Validate configuration before using
    result = mc.validate_config(config_dict=config)
    
    if not result.is_valid:
        print("❌ Configuration validation failed:")
        for error in result.errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")
    
    if result.has_warnings:
        print("⚠️  Configuration warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Configure Model Compass
    mc.configure(config_dict=config)
    
    # Enable performance optimizations
    mc.enable_performance_optimizations()
    
    env = os.getenv("ENVIRONMENT", "development")
    print(f"✅ Model Compass configured for {env} environment")
    print(f"📊 Available models: {', '.join(mc.list_models())}")
    print(f"🎯 Available profiles: {', '.join(mc.list_profiles())}")


def demonstrate_usage():
    """Demonstrate usage patterns for different environments."""
    env = os.getenv("ENVIRONMENT", "development")
    print(f"\n🚀 Demonstrating usage in {env} environment:")
    
    # Use default model
    default_model = mc.model()
    print(f"Default model: {default_model.provider}/{default_model.model_name}")
    
    # Use intent-based resolution
    try:
        fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")
        print(f"Fast model: {fast_model.provider}/{fast_model.model_name}")
    except Exception as e:
        print(f"Intent resolution failed: {e}")
    
    # Use aliases
    aliases = mc.list_aliases()
    if aliases:
        alias_model = mc.model(aliases[0])
        print(f"Alias '{aliases[0]}' resolves to: {alias_model.provider}/{alias_model.model_name}")
    
    # Profile switching example
    profiles = mc.list_profiles()
    if len(profiles) > 1:
        with mc.profile(profiles[1]):
            profile_model = mc.model()
            print(f"Profile '{profiles[1]}' uses: {profile_model.provider}/{profile_model.model_name}")


if __name__ == "__main__":
    # Set up Model Compass based on environment
    setup_model_compass()
    
    # Demonstrate usage
    demonstrate_usage()
    
    # Show configuration status
    status = mc.is_configured_with_details()
    print(f"\n📈 Configuration status:")
    print(f"  Models: {status['models_count']}")
    print(f"  Profiles: {status['profiles_count']}")
    print(f"  Aliases: {status['aliases_count']}")
    print(f"  Providers: {status['providers_count']}")
    
    # Show performance stats if optimizations are enabled
    try:
        stats = mc.get_performance_stats()
        print(f"\n⚡ Performance stats:")
        for cache_name, cache_stats in stats.items():
            if isinstance(cache_stats, dict) and 'hit_rate' in cache_stats:
                print(f"  {cache_name}: {cache_stats['hit_rate']:.1f}% hit rate")
    except Exception:
        print("Performance stats not available")