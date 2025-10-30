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
Intent Resolution Best Practices Example

This example demonstrates best practices for setting up model metadata
to enable effective intent-based resolution with URN queries.
"""

import model_compass as mc


def create_intent_optimized_config():
    """Create a configuration optimized for intent resolution."""
    return {
        "default_profile": "balanced",
        "models": {
            # High-reasoning models
            "gpt-4": {
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {
                    # Core intent fields
                    "reasoning": "high",
                    "cost": "high",
                    "speed": "medium",
                    
                    # Capability tags
                    "capabilities": ["chat", "completion", "function_calling", "analysis"],
                    
                    # Context and constraints
                    "context_length": 8192,
                    "multimodal": False,
                    
                    # Deployment info
                    "deployment": "cloud",
                    "provider_tier": "premium",
                    
                    # Use case tags
                    "use_cases": ["research", "analysis", "complex_reasoning", "writing"],
                    
                    # Quality indicators
                    "accuracy": "high",
                    "reliability": "high",
                    
                    # Custom business metadata
                    "compliance": "enterprise",
                    "data_residency": "us",
                    "sla_tier": "premium"
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "metadata": {
                    "reasoning": "high",
                    "cost": "medium",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "vision", "analysis"],
                    "context_length": 200000,
                    "multimodal": True,
                    "deployment": "cloud",
                    "provider_tier": "premium",
                    "use_cases": ["analysis", "writing", "vision", "long_context"],
                    "accuracy": "high",
                    "reliability": "high",
                    "compliance": "enterprise",
                    "data_residency": "us",
                    "sla_tier": "standard"
                },
                "parameters": {"temperature": 0.7, "max_tokens": 4096}
            },
            
            # Fast, cost-effective models
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "function_calling"],
                    "context_length": 16385,
                    "multimodal": False,
                    "deployment": "cloud",
                    "provider_tier": "standard",
                    "use_cases": ["chat", "simple_tasks", "prototyping"],
                    "accuracy": "medium",
                    "reliability": "high",
                    "compliance": "standard",
                    "data_residency": "us",
                    "sla_tier": "standard"
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            
            "claude-3-haiku": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "low",
                    "speed": "fast",
                    "capabilities": ["chat", "completion", "vision"],
                    "context_length": 200000,
                    "multimodal": True,
                    "deployment": "cloud",
                    "provider_tier": "standard",
                    "use_cases": ["chat", "quick_analysis", "vision"],
                    "accuracy": "medium",
                    "reliability": "high",
                    "compliance": "standard",
                    "data_residency": "us",
                    "sla_tier": "basic"
                },
                "parameters": {"temperature": 0.7, "max_tokens": 2048}
            },
            
            # Local models
            "llama3": {
                "provider": "ollama",
                "model": "llama3",
                "metadata": {
                    "reasoning": "high",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["chat", "completion"],
                    "context_length": 8192,
                    "multimodal": False,
                    "deployment": "local",
                    "provider_tier": "self_hosted",
                    "use_cases": ["development", "privacy_sensitive", "offline"],
                    "accuracy": "high",
                    "reliability": "medium",
                    "compliance": "private",
                    "data_residency": "local",
                    "sla_tier": "self_managed",
                    "local": True,
                    "offline_capable": True
                },
                "parameters": {"temperature": 0.7, "num_predict": 2048}
            },
            
            # Specialized models
            "codellama": {
                "provider": "ollama",
                "model": "codellama",
                "metadata": {
                    "reasoning": "medium",
                    "cost": "free",
                    "speed": "medium",
                    "capabilities": ["chat", "completion", "code"],
                    "context_length": 16384,
                    "multimodal": False,
                    "deployment": "local",
                    "provider_tier": "self_hosted",
                    "use_cases": ["coding", "development", "code_review"],
                    "accuracy": "high",
                    "reliability": "medium",
                    "compliance": "private",
                    "data_residency": "local",
                    "sla_tier": "self_managed",
                    "local": True,
                    "offline_capable": True,
                    "specialization": "code"
                },
                "parameters": {"temperature": 0.1, "num_predict": 2048}
            }
        },
        
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "headers": {"Authorization": "Bearer ${OPENAI_API_KEY}"}
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "headers": {"x-api-key": "${ANTHROPIC_API_KEY}"}
            },
            "ollama": {
                "base_url": "http://localhost:11434"
            }
        },
        
        "profiles": {
            "balanced": {"model": "claude-3-sonnet"},
            "fast": {"model": "gpt-3.5-turbo"},
            "powerful": {"model": "gpt-4"},
            "local": {"model": "llama3"},
            "coding": {"model": "codellama"}
        }
    }


def demonstrate_intent_resolution():
    """Demonstrate various intent resolution patterns."""
    
    print("🎯 Intent Resolution Examples:")
    print("=" * 50)
    
    # Basic intent resolution
    examples = [
        # Performance-based selection
        ("urn:llm:intent:chat?reasoning=high&cost=low", "High reasoning, low cost"),
        ("urn:llm:intent:chat?speed=fast&cost=low", "Fast and cheap"),
        ("urn:llm:intent:completion?reasoning=high", "Best reasoning"),
        
        # Capability-based selection
        ("urn:llm:intent:chat?capabilities=vision", "Vision capabilities"),
        ("urn:llm:intent:completion?capabilities=code", "Code capabilities"),
        ("urn:llm:intent:chat?multimodal=true", "Multimodal support"),
        
        # Deployment-based selection
        ("urn:llm:intent:chat?deployment=local", "Local deployment"),
        ("urn:llm:intent:chat?deployment=cloud&cost=low", "Cloud but cheap"),
        ("urn:llm:intent:completion?local=true", "Local models only"),
        
        # Use case-based selection
        ("urn:llm:intent:chat?use_cases=analysis", "Analysis tasks"),
        ("urn:llm:intent:completion?use_cases=coding", "Coding tasks"),
        ("urn:llm:intent:chat?use_cases=writing", "Writing tasks"),
        
        # Business requirements
        ("urn:llm:intent:chat?compliance=enterprise", "Enterprise compliance"),
        ("urn:llm:intent:completion?data_residency=local", "Local data residency"),
        ("urn:llm:intent:chat?sla_tier=premium", "Premium SLA"),
        
        # Complex multi-criteria
        ("urn:llm:intent:chat?reasoning=high&deployment=cloud&compliance=enterprise", "High reasoning, cloud, enterprise"),
        ("urn:llm:intent:completion?cost=free&capabilities=code", "Free coding model"),
        ("urn:llm:intent:chat?speed=fast&multimodal=true&cost=low", "Fast, multimodal, cheap")
    ]
    
    for urn, description in examples:
        try:
            model = mc.model(urn)
            print(f"✅ {description}")
            print(f"   URN: {urn}")
            print(f"   → {model.provider}/{model.model_name}")
            print()
        except Exception as e:
            print(f"❌ {description}")
            print(f"   URN: {urn}")
            print(f"   → Error: {str(e)}")
            print()


def demonstrate_custom_metadata():
    """Demonstrate how to add custom metadata fields for specific business needs."""
    
    print("🏢 Custom Metadata Examples:")
    print("=" * 50)
    
    # Example with custom business metadata
    custom_config = {
        "models": {
            "business-model": {
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {
                    # Standard fields
                    "reasoning": "high",
                    "cost": "high",
                    "speed": "medium",
                    
                    # Custom business fields
                    "department": "finance",
                    "security_level": "confidential",
                    "region": "us-east",
                    "business_unit": "analytics",
                    "cost_center": "CC-1001",
                    "approved_for": ["financial_analysis", "risk_assessment"],
                    "data_classification": "sensitive",
                    "audit_required": True,
                    "max_concurrent_users": 10
                }
            }
        },
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "headers": {"Authorization": "Bearer ${OPENAI_API_KEY}"}
            }
        }
    }
    
    # Configure with custom metadata
    mc.configure(config_dict=custom_config)
    
    # Custom intent resolution examples
    custom_examples = [
        ("urn:llm:intent:chat?department=finance", "Finance department models"),
        ("urn:llm:intent:completion?security_level=confidential", "Confidential security level"),
        ("urn:llm:intent:chat?region=us-east", "US East region"),
        ("urn:llm:intent:completion?approved_for=financial_analysis", "Approved for financial analysis")
    ]
    
    for urn, description in custom_examples:
        try:
            model = mc.model(urn)
            print(f"✅ {description}")
            print(f"   URN: {urn}")
            print(f"   → {model.provider}/{model.model_name}")
            print()
        except Exception as e:
            print(f"❌ {description}")
            print(f"   URN: {urn}")
            print(f"   → Error: {str(e)}")
            print()


def demonstrate_metadata_best_practices():
    """Demonstrate best practices for metadata design."""
    
    print("📋 Metadata Best Practices:")
    print("=" * 50)
    
    best_practices = [
        "1. Use consistent vocabulary across models",
        "2. Include both technical and business metadata",
        "3. Use standardized values (high/medium/low, not excellent/good/poor)",
        "4. Include capability tags as lists for multi-matching",
        "5. Add use-case tags to help with intent resolution",
        "6. Include deployment and compliance information",
        "7. Use boolean flags for binary properties (local, multimodal, etc.)",
        "8. Add custom fields for business-specific requirements",
        "9. Keep metadata up-to-date as models change",
        "10. Test intent resolution with your metadata schema"
    ]
    
    for practice in best_practices:
        print(f"  {practice}")
    
    print("\n🔍 Metadata Schema Example:")
    print("-" * 30)
    
    schema_example = {
        "reasoning": "high|medium|low",
        "cost": "high|medium|low|free", 
        "speed": "fast|medium|slow",
        "capabilities": ["chat", "completion", "function_calling", "vision", "code"],
        "context_length": "integer",
        "multimodal": "boolean",
        "deployment": "cloud|local|hybrid",
        "use_cases": ["analysis", "writing", "coding", "chat", "vision"],
        "accuracy": "high|medium|low",
        "reliability": "high|medium|low",
        "compliance": "enterprise|standard|basic",
        "data_residency": "us|eu|local|global",
        "local": "boolean",
        "offline_capable": "boolean"
    }
    
    for field, values in schema_example.items():
        print(f"  {field}: {values}")


if __name__ == "__main__":
    # Set up intent-optimized configuration
    config = create_intent_optimized_config()
    mc.configure(config_dict=config)
    
    # Demonstrate intent resolution
    demonstrate_intent_resolution()
    
    # Show custom metadata examples
    demonstrate_custom_metadata()
    
    # Show best practices
    demonstrate_metadata_best_practices()
    
    print("\n📊 Configuration Summary:")
    status = mc.is_configured_with_details()
    print(f"  Models: {status['models_count']}")
    print(f"  Profiles: {status['profiles_count']}")
    print(f"  Intent metadata coverage: {'Good' if status.get('has_intent_metadata') else 'Limited'}")
    
    # Show suggestions for improvement
    suggestions = mc.get_config_suggestions()
    if suggestions:
        print(f"\n💡 Configuration Suggestions:")
        for suggestion in suggestions[:3]:  # Show first 3 suggestions
            print(f"  - {suggestion}")