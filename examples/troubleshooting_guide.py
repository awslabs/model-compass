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
Troubleshooting Guide for Common Configuration Issues

This example demonstrates how to diagnose and fix common configuration problems
with Model Compass, including helpful error messages and debugging techniques.
"""

import model_compass as mc
from model_compass.exceptions import ConfigurationError, ResolutionError


def demonstrate_common_errors():
    """Demonstrate common configuration errors and their solutions."""
    
    print("🔧 Common Configuration Issues and Solutions")
    print("=" * 60)
    
    # Error 1: No configuration loaded
    print("\n1. No Configuration Loaded Error")
    print("-" * 40)
    
    try:
        # Reset to ensure no configuration
        mc.reset_configuration()
        model = mc.model("gpt-4")
    except ConfigurationError as e:
        print(f"❌ Error: {e}")
        print("✅ Solution: Configure Model Compass first")
        print("   Example:")
        print("   config = mc.generate_template('openai')")
        print("   mc.configure(config_dict=config)")
    
    # Error 2: Model not found
    print("\n2. Model Not Found Error")
    print("-" * 40)
    
    # Set up minimal config for demonstration
    minimal_config = {
        "models": {
            "test-model": {
                "provider": "test",
                "model": "test-model",
                "metadata": {"reasoning": "medium"}
            }
        },
        "providers": {
            "test": {"base_url": "http://test.com"}
        }
    }
    mc.configure(config_dict=minimal_config)
    
    try:
        model = mc.model("nonexistent-model")
    except ResolutionError as e:
        print(f"❌ Error: {e}")
        print("✅ Solutions:")
        print("   - Check available models:", mc.list_models())
        print("   - Add the model to your configuration")
        print("   - Use a configuration template")
        
        # Show detailed error message
        detailed = e.get_detailed_message()
        if "Configuration examples:" in detailed:
            print("   - Configuration example provided in error message")
    
    # Error 3: Invalid configuration structure
    print("\n3. Invalid Configuration Structure")
    print("-" * 40)
    
    invalid_configs = [
        # Missing required fields
        {
            "models": {
                "invalid-model": {
                    "metadata": {}  # Missing provider and model
                }
            }
        },
        # Wrong data types
        {
            "models": "not a dictionary",
            "providers": {}
        },
        # Empty configuration
        {}
    ]
    
    for i, invalid_config in enumerate(invalid_configs, 1):
        print(f"\n   Invalid Config {i}:")
        result = mc.validate_config(config_dict=invalid_config)
        
        if not result.is_valid:
            print(f"   ❌ Validation failed:")
            for error in result.errors[:2]:  # Show first 2 errors
                print(f"      - {error}")
        
        if result.suggestions:
            print(f"   ✅ Suggestions:")
            for suggestion in result.suggestions[:2]:  # Show first 2 suggestions
                print(f"      - {suggestion}")
    
    # Error 4: Intent resolution failures
    print("\n4. Intent Resolution Failures")
    print("-" * 40)
    
    # Configure with limited metadata
    limited_config = {
        "models": {
            "basic-model": {
                "provider": "test",
                "model": "basic-model"
                # No metadata for intent resolution
            }
        },
        "providers": {
            "test": {"base_url": "http://test.com"}
        }
    }
    mc.configure(config_dict=limited_config)
    
    try:
        model = mc.model("urn:llm:intent:chat?reasoning=high&cost=low")
    except ResolutionError as e:
        print(f"❌ Error: {e}")
        print("✅ Solutions:")
        print("   - Add metadata to your models for intent resolution")
        print("   - Use configuration templates with rich metadata")
        print("   - Check available metadata fields in your models")
        
        # Show configuration example from error
        if hasattr(e, 'configuration_examples') and e.configuration_examples:
            print("   - Example configuration provided in error message")


def demonstrate_debugging_techniques():
    """Demonstrate debugging techniques for configuration issues."""
    
    print("\n\n🔍 Debugging Techniques")
    print("=" * 40)
    
    # Set up a good configuration for debugging examples
    config = mc.generate_template("multi",
        openai_models=["gpt-4", "gpt-3.5-turbo"],
        anthropic_models=["claude-3-sonnet"],
        ollama_models=["llama3"]
    )
    mc.configure(config_dict=config)
    
    # Technique 1: Check configuration status
    print("\n1. Check Configuration Status")
    print("-" * 30)
    
    status = mc.is_configured_with_details()
    print(f"Configured: {status['configured']}")
    print(f"Models: {status['models_count']}")
    print(f"Profiles: {status['profiles_count']}")
    print(f"Aliases: {status['aliases_count']}")
    print(f"Providers: {status['providers_count']}")
    print(f"Has intent metadata: {status.get('has_intent_metadata', False)}")
    
    # Technique 2: List available resources
    print("\n2. List Available Resources")
    print("-" * 30)
    
    print(f"Available models: {mc.list_models()}")
    print(f"Available profiles: {mc.list_profiles()}")
    print(f"Available aliases: {mc.list_aliases()}")
    
    # Technique 3: Validate configuration
    print("\n3. Validate Configuration")
    print("-" * 30)
    
    result = mc.validate_config(config_dict=config)
    print(f"Valid: {result.is_valid}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Suggestions: {len(result.suggestions)}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:2]:
            print(f"  - {warning}")
    
    if result.suggestions:
        print("Suggestions:")
        for suggestion in result.suggestions[:2]:
            print(f"  - {suggestion}")
    
    # Technique 4: Test model resolution
    print("\n4. Test Model Resolution")
    print("-" * 30)
    
    test_identifiers = [
        "gpt-4",  # Direct model
        "openai/gpt-4",  # Physical identifier
        "urn:llm:intent:chat?reasoning=high",  # Intent
        "nonexistent"  # Should fail
    ]
    
    for identifier in test_identifiers:
        try:
            model = mc.model(identifier)
            print(f"✅ '{identifier}' → {model.provider}/{model.model_name}")
        except Exception as e:
            print(f"❌ '{identifier}' → {type(e).__name__}: {str(e)[:50]}...")
    
    # Technique 5: Check model properties
    print("\n5. Inspect Model Properties")
    print("-" * 30)
    
    model = mc.model("gpt-4")
    print(f"Provider: {model.provider}")
    print(f"Model name: {model.model_name}")
    print(f"Base URL: {model.base_url}")
    print(f"Parameters: {list(model.parameters.keys())}")
    print(f"Metadata: {list(model.metadata.keys())}")


def demonstrate_configuration_validation():
    """Demonstrate comprehensive configuration validation."""
    
    print("\n\n✅ Configuration Validation Best Practices")
    print("=" * 50)
    
    # Example configurations with different issues
    test_configs = {
        "minimal_valid": {
            "models": {
                "test-model": {
                    "provider": "test",
                    "model": "test-model"
                }
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            }
        },
        
        "rich_metadata": {
            "models": {
                "rich-model": {
                    "provider": "test",
                    "model": "rich-model",
                    "metadata": {
                        "reasoning": "high",
                        "cost": "medium",
                        "speed": "fast",
                        "capabilities": ["chat", "completion"]
                    },
                    "parameters": {"temperature": 0.7}
                }
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            },
            "profiles": {
                "default": {"model": "rich-model"}
            }
        },
        
        "missing_provider": {
            "models": {
                "orphan-model": {
                    "provider": "missing-provider",
                    "model": "orphan-model"
                }
            },
            "providers": {}
        }
    }
    
    for config_name, config in test_configs.items():
        print(f"\n{config_name.replace('_', ' ').title()}:")
        print("-" * 30)
        
        result = mc.validate_config(config_dict=config)
        
        print(f"Valid: {result.is_valid}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        if result.suggestions:
            print("Suggestions:")
            for suggestion in result.suggestions[:3]:  # Limit to 3
                print(f"  💡 {suggestion}")


def demonstrate_error_recovery():
    """Demonstrate error recovery techniques."""
    
    print("\n\n🔄 Error Recovery Techniques")
    print("=" * 40)
    
    print("\n1. Graceful Fallbacks")
    print("-" * 20)
    
    # Set up config with fallback chain
    fallback_config = {
        "models": {
            "primary": {
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {"reasoning": "high", "cost": "high"}
            },
            "secondary": {
                "provider": "anthropic", 
                "model": "claude-3-sonnet",
                "metadata": {"reasoning": "high", "cost": "medium"}
            },
            "fallback": {
                "provider": "test",
                "model": "test-model",
                "metadata": {"reasoning": "medium", "cost": "low"}
            }
        },
        "providers": {
            "openai": {"base_url": "https://api.openai.com/v1"},
            "anthropic": {"base_url": "https://api.anthropic.com"},
            "test": {"base_url": "http://test.com"}
        },
        "aliases": {
            "main": "primary",
            "backup": "secondary",
            "emergency": "fallback"
        }
    }
    
    mc.configure(config_dict=fallback_config)
    
    # Demonstrate fallback usage
    fallback_chain = ["main", "backup", "emergency"]
    
    for alias in fallback_chain:
        try:
            model = mc.model(alias)
            print(f"✅ Using {alias}: {model.provider}/{model.model_name}")
            break  # Use first successful model
        except Exception as e:
            print(f"❌ {alias} failed: {str(e)[:30]}...")
            continue
    
    print("\n2. Configuration Repair")
    print("-" * 20)
    
    # Show how to fix common issues
    broken_config = {
        "models": {
            "broken-model": {
                "provider": "missing-provider",
                "model": "broken-model"
            }
        }
    }
    
    result = mc.validate_config(config_dict=broken_config)
    
    if not result.is_valid:
        print("Broken configuration detected:")
        for error in result.errors:
            print(f"  ❌ {error}")
        
        print("\nAuto-repair suggestions:")
        for suggestion in result.suggestions:
            print(f"  🔧 {suggestion}")
        
        # Show repaired version
        repaired_config = {
            "models": {
                "broken-model": {
                    "provider": "test",
                    "model": "broken-model",
                    "metadata": {"reasoning": "medium"}
                }
            },
            "providers": {
                "test": {"base_url": "http://test.com"}
            }
        }
        
        repaired_result = mc.validate_config(config_dict=repaired_config)
        print(f"\nAfter repair - Valid: {repaired_result.is_valid}")


if __name__ == "__main__":
    print("🚨 Model Compass Troubleshooting Guide")
    print("=" * 60)
    
    # Demonstrate common errors
    demonstrate_common_errors()
    
    # Show debugging techniques
    demonstrate_debugging_techniques()
    
    # Show validation best practices
    demonstrate_configuration_validation()
    
    # Show error recovery
    demonstrate_error_recovery()
    
    print("\n\n📚 Additional Resources:")
    print("- Use mc.validate_config() before mc.configure()")
    print("- Check mc.is_configured_with_details() for status")
    print("- Use mc.get_config_suggestions() for improvements")
    print("- Generate templates with mc.generate_template()")
    print("- Enable verbose logging for detailed error info")
    print("- Check the documentation for configuration examples")