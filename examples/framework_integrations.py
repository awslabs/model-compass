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
Framework Integration Examples

This example demonstrates how to integrate Model Compass with popular
AI frameworks using the user-config-only approach with explicit configuration.
"""

import model_compass as mc


def setup_model_compass():
    """Set up Model Compass with a comprehensive configuration."""
    # Generate a multi-provider configuration
    config = mc.generate_template("multi",
        openai_models=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        anthropic_models=["claude-3-sonnet", "claude-3-haiku"],
        ollama_models=["llama3", "codellama"]
    )
    
    # Configure Model Compass
    mc.configure(config_dict=config)
    print("✅ Model Compass configured with multi-provider setup")
    print(f"📊 Available models: {', '.join(mc.list_models())}")


def langchain_integration():
    """Demonstrate LangChain integration with Model Compass."""
    print("\n🦜 LangChain Integration")
    print("=" * 40)
    
    try:
        from langchain.llms import OpenAI
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage
        
        # Get model configuration from Model Compass
        model = mc.model("gpt-4")
        
        # Use with LangChain LLM
        llm = OpenAI(
            model_name=model.model_name,
            openai_api_base=model.base_url,
            openai_api_key=model.get_header_value("Authorization").replace("Bearer ", ""),
            **model.parameters
        )
        
        print(f"✅ LangChain LLM configured with {model.provider}/{model.model_name}")
        print(f"   Base URL: {model.base_url}")
        print(f"   Parameters: {model.parameters}")
        
        # Use with LangChain ChatModel
        chat_model = ChatOpenAI(
            model_name=model.model_name,
            openai_api_base=model.base_url,
            openai_api_key=model.get_header_value("Authorization").replace("Bearer ", ""),
            **model.parameters
        )
        
        print(f"✅ LangChain ChatModel configured")
        
        # Example with different models for different tasks
        print("\n📋 Task-Specific Model Selection:")
        
        # Fast model for simple tasks
        try:
            fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")
            print(f"   Fast tasks: {fast_model.provider}/{fast_model.model_name}")
        except:
            fast_model = mc.model("gpt-3.5-turbo")
            print(f"   Fast tasks: {fast_model.provider}/{fast_model.model_name}")
        
        # Powerful model for complex reasoning
        try:
            reasoning_model = mc.model("urn:llm:intent:chat?reasoning=high")
            print(f"   Complex reasoning: {reasoning_model.provider}/{reasoning_model.model_name}")
        except:
            reasoning_model = mc.model("gpt-4")
            print(f"   Complex reasoning: {reasoning_model.provider}/{reasoning_model.model_name}")
        
    except ImportError:
        print("❌ LangChain not installed. Install with: pip install langchain")
    except Exception as e:
        print(f"❌ LangChain integration error: {e}")


def dspy_integration():
    """Demonstrate DSPy integration with Model Compass."""
    print("\n🔧 DSPy Integration")
    print("=" * 40)
    
    try:
        import dspy
        
        # Configure different models for different environments
        environments = {
            "development": "fast",
            "staging": "balanced", 
            "production": "powerful"
        }
        
        for env, profile in environments.items():
            print(f"\n{env.title()} Environment:")
            
            try:
                # Use profile-based selection
                with mc.profile(profile):
                    model = mc.model()
                    
                    if model.provider == "openai":
                        # Configure DSPy with OpenAI
                        lm = dspy.OpenAI(
                            model=model.model_name,
                            api_base=model.base_url,
                            api_key=model.get_header_value("Authorization").replace("Bearer ", ""),
                            **model.parameters
                        )
                        dspy.settings.configure(lm=lm)
                        print(f"   ✅ DSPy configured with OpenAI {model.model_name}")
                        
                    elif model.provider == "anthropic":
                        # Note: DSPy Anthropic integration may require specific setup
                        print(f"   ℹ️  Anthropic model available: {model.model_name}")
                        print(f"      Base URL: {model.base_url}")
                        print(f"      API Key header: x-api-key")
                        
                    elif model.provider == "ollama":
                        # Configure DSPy with local Ollama
                        print(f"   ℹ️  Local Ollama model: {model.model_name}")
                        print(f"      Base URL: {model.base_url}")
                        print(f"      Use with custom DSPy LM implementation")
                        
            except Exception as e:
                print(f"   ❌ Profile '{profile}' not available: {e}")
        
        # Example: Intent-based model selection for DSPy
        print(f"\n🎯 Intent-Based Selection:")
        
        try:
            # Get best model for reasoning tasks
            reasoning_model = mc.model("urn:llm:intent:completion?reasoning=high")
            print(f"   Best reasoning model: {reasoning_model.provider}/{reasoning_model.model_name}")
            
            # Get cost-effective model for simple tasks
            cheap_model = mc.model("urn:llm:intent:chat?cost=low")
            print(f"   Cost-effective model: {cheap_model.provider}/{cheap_model.model_name}")
            
        except Exception as e:
            print(f"   ❌ Intent resolution failed: {e}")
        
    except ImportError:
        print("❌ DSPy not installed. Install with: pip install dspy-ai")
    except Exception as e:
        print(f"❌ DSPy integration error: {e}")


def autogen_integration():
    """Demonstrate Autogen integration with Model Compass."""
    print("\n🤖 Autogen Integration")
    print("=" * 40)
    
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        
        # Create different agents with different models
        agents = []
        
        # Reasoning agent with powerful model
        try:
            reasoning_model = mc.model("urn:llm:intent:chat?reasoning=high")
        except:
            reasoning_model = mc.model("gpt-4")
        
        reasoning_agent = AssistantAgent(
            name="reasoning_agent",
            system_message="You are an expert at complex reasoning and analysis.",
            llm_config={
                "model": reasoning_model.model_name,
                "api_base": reasoning_model.base_url,
                "api_key": reasoning_model.get_header_value("Authorization").replace("Bearer ", ""),
                **reasoning_model.parameters
            }
        )
        agents.append(reasoning_agent)
        print(f"✅ Reasoning agent: {reasoning_model.provider}/{reasoning_model.model_name}")
        
        # Fast agent with cost-effective model
        try:
            fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")
        except:
            fast_model = mc.model("gpt-3.5-turbo")
        
        fast_agent = AssistantAgent(
            name="fast_agent",
            system_message="You provide quick responses and handle simple tasks efficiently.",
            llm_config={
                "model": fast_model.model_name,
                "api_base": fast_model.base_url,
                "api_key": fast_model.get_header_value("Authorization").replace("Bearer ", ""),
                **fast_model.parameters
            }
        )
        agents.append(fast_agent)
        print(f"✅ Fast agent: {fast_model.provider}/{fast_model.model_name}")
        
        # Code agent with code-specialized model
        try:
            code_model = mc.model("urn:llm:intent:completion?capabilities=code")
        except:
            try:
                code_model = mc.model("codellama")
            except:
                code_model = mc.model("gpt-4")
        
        if code_model.provider == "ollama":
            print(f"ℹ️  Code agent would use local model: {code_model.model_name}")
            print(f"   Configure Autogen with Ollama endpoint: {code_model.base_url}")
        else:
            code_agent = AssistantAgent(
                name="code_agent",
                system_message="You are an expert programmer and code reviewer.",
                llm_config={
                    "model": code_model.model_name,
                    "api_base": code_model.base_url,
                    "api_key": code_model.get_header_value("Authorization").replace("Bearer ", ""),
                    **code_model.parameters
                }
            )
            agents.append(code_agent)
            print(f"✅ Code agent: {code_model.provider}/{code_model.model_name}")
        
        # User proxy agent
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )
        
        print(f"✅ Created {len(agents)} specialized agents")
        print("   Each agent uses the most appropriate model for its role")
        
        # Example: Environment-specific agent configuration
        print(f"\n🌍 Environment-Specific Configuration:")
        
        environments = ["development", "staging", "production"]
        for env in environments:
            try:
                with mc.profile(env):
                    env_model = mc.model()
                    print(f"   {env.title()}: {env_model.provider}/{env_model.model_name}")
            except:
                print(f"   {env.title()}: Profile not available")
        
    except ImportError:
        print("❌ Autogen not installed. Install with: pip install pyautogen")
    except Exception as e:
        print(f"❌ Autogen integration error: {e}")


def custom_framework_integration():
    """Demonstrate integration with custom frameworks."""
    print("\n⚙️  Custom Framework Integration")
    print("=" * 40)
    
    # Example: Generic framework integration pattern
    class CustomFramework:
        def __init__(self, model_config):
            self.model_config = model_config
            self.provider = model_config.provider
            self.model_name = model_config.model_name
            self.base_url = model_config.base_url
            self.parameters = model_config.parameters
            self.headers = model_config.headers
        
        def make_request(self, prompt):
            # This would contain actual API call logic
            return f"Response from {self.provider}/{self.model_name} for: {prompt[:50]}..."
    
    # Use Model Compass for configuration
    models_to_test = ["gpt-4", "claude-3-sonnet", "llama3"]
    
    for model_id in models_to_test:
        try:
            model = mc.model(model_id)
            framework = CustomFramework(model)
            
            print(f"✅ Custom framework configured with {model.provider}/{model.model_name}")
            print(f"   Base URL: {model.base_url}")
            print(f"   Headers: {list(model.headers.keys())}")
            print(f"   Parameters: {list(model.parameters.keys())}")
            
            # Simulate usage
            response = framework.make_request("Hello, world!")
            print(f"   Test response: {response}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to configure {model_id}: {e}")
    
    # Example: Dynamic model selection
    print("🔄 Dynamic Model Selection:")
    
    def get_best_model_for_task(task_type):
        """Get the best model for a specific task type."""
        intent_queries = {
            "reasoning": "urn:llm:intent:completion?reasoning=high",
            "speed": "urn:llm:intent:chat?speed=fast",
            "cost": "urn:llm:intent:chat?cost=low",
            "local": "urn:llm:intent:chat?deployment=local"
        }
        
        try:
            return mc.model(intent_queries.get(task_type, "gpt-4"))
        except:
            # Fallback to available models
            available = mc.list_models()
            return mc.model(available[0]) if available else None
    
    for task in ["reasoning", "speed", "cost", "local"]:
        model = get_best_model_for_task(task)
        if model:
            print(f"   {task.title()} tasks: {model.provider}/{model.model_name}")
        else:
            print(f"   {task.title()} tasks: No suitable model found")


def framework_best_practices():
    """Demonstrate best practices for framework integration."""
    print("\n📋 Framework Integration Best Practices")
    print("=" * 50)
    
    practices = [
        "1. Use Model Compass for centralized model configuration",
        "2. Leverage intent-based resolution for dynamic model selection",
        "3. Use profiles for environment-specific configurations",
        "4. Implement fallback chains for reliability",
        "5. Cache model configurations to avoid repeated resolution",
        "6. Use aliases for easy model switching",
        "7. Validate configurations before framework initialization",
        "8. Handle resolution errors gracefully",
        "9. Use metadata for intelligent model selection",
        "10. Enable performance optimizations for large setups"
    ]
    
    for practice in practices:
        print(f"  {practice}")
    
    print(f"\n💡 Example Implementation Pattern:")
    print("-" * 30)
    
    example_code = '''
class FrameworkAdapter:
    def __init__(self, model_identifier=None):
        # Use Model Compass for model resolution
        self.model = mc.model(model_identifier)
        self.framework_client = self._create_client()
    
    def _create_client(self):
        # Create framework-specific client
        if self.model.provider == "openai":
            return OpenAIClient(
                api_key=self.model.get_api_key(),
                base_url=self.model.base_url,
                **self.model.parameters
            )
        elif self.model.provider == "anthropic":
            return AnthropicClient(
                api_key=self.model.get_api_key(),
                base_url=self.model.base_url,
                **self.model.parameters
            )
        # Add more providers as needed
    
    def generate(self, prompt):
        return self.framework_client.generate(prompt)

# Usage
adapter = FrameworkAdapter("urn:llm:intent:chat?reasoning=high")
response = adapter.generate("Explain quantum computing")
'''
    
    print(example_code)


if __name__ == "__main__":
    print("🔗 Model Compass Framework Integration Examples")
    print("=" * 60)
    
    # Set up Model Compass
    setup_model_compass()
    
    # Demonstrate integrations
    langchain_integration()
    dspy_integration()
    autogen_integration()
    custom_framework_integration()
    framework_best_practices()
    
    print(f"\n📊 Configuration Summary:")
    status = mc.is_configured_with_details()
    print(f"  Models: {status['models_count']}")
    print(f"  Profiles: {status['profiles_count']}")
    print(f"  Providers: {status['providers_count']}")
    
    print(f"\n🎯 Available for Framework Integration:")
    print(f"  Models: {', '.join(mc.list_models())}")
    print(f"  Profiles: {', '.join(mc.list_profiles())}")
    
    print(f"\n✨ Framework integration complete!")
    print("   Use these patterns to integrate Model Compass with your preferred AI framework.")