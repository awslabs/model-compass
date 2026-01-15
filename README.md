# Model Compass

A lightweight, provider-agnostic Python library for LLM configuration and access. Model Compass provides a unified interface for configuring LLM providers and accessing models through intent-based URNs, logical profiles, physical identifiers, and flexible alias chains.

## Features

- **🎯 Intent-Based Resolution**: Use URN strings to specify model requirements and let Model Compass find the best match
- **📋 Profile Management**: Define logical profiles for different environments (dev, staging, prod)
- **🔗 Flexible Aliases**: Create alias chains for easy model switching and deployment patterns
- **⚡ Performance Optimized**: Built-in caching and lazy loading for optimal performance
- **🔧 Provider Agnostic**: Works with OpenAI, Anthropic, Ollama, and custom providers
- **🛡️ Type Safe**: Full type hints and comprehensive error handling
- **🧪 Well Tested**: 90+ tests ensuring reliability

## Quick Start

### Installation

```bash
# Install from source (recommended)
pip install git+https://github.com/awslabs/model-compass.git

# Or clone and install locally
git clone https://github.com/awslabs/model-compass.git
cd model-compass
pip install -e .
```

### Basic Usage

```python
import model_compass as mc

# Option 1: Generate a configuration template
config = mc.generate_template("openai")  # or "anthropic", "ollama", "multi"
mc.configure(config_dict=config)

# Option 2: Create your own configuration
mc.configure(config_dict={
    "models": {
        "gpt-4": {
            "provider": "openai",
            "model": "gpt-4",
            "metadata": {"reasoning": "high", "cost": "high", "speed": "medium"},
            "parameters": {"temperature": 0.7, "max_tokens": 2048}
        },
        "claude": {
            "provider": "anthropic", 
            "model": "claude-3-sonnet-20240229",
            "metadata": {"reasoning": "high", "cost": "medium", "speed": "fast"},
            "parameters": {"temperature": 0.7, "max_tokens": 4096}
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
        }
    }
})

# Access models through different methods
model = mc.model("gpt-4")                                    # Direct model ID
model = mc.model("openai/gpt-4")                            # Physical identifier
model = mc.model("urn:llm:intent:chat?reasoning=high")      # Intent-based URN

# Use the model
print(f"Using {model.provider}/{model.name}")
print(f"Base URL: {model.base_url}")
print(f"Cost tier: {model.cost_tier}")
```

## Configuration Templates

Model Compass provides built-in configuration templates for popular providers to get you started quickly:

### Generate Templates

```python
import model_compass as mc

# Generate OpenAI configuration
openai_config = mc.generate_template("openai")
mc.configure(config_dict=openai_config)

# Generate Anthropic configuration  
anthropic_config = mc.generate_template("anthropic")

# Generate Ollama configuration for local models
ollama_config = mc.generate_template("ollama")

# Generate multi-provider configuration
multi_config = mc.generate_template("multi", 
    openai_models=["gpt-4", "gpt-3.5-turbo"],
    anthropic_models=["claude-3-sonnet", "claude-3-haiku"],
    ollama_models=["llama3", "codellama"]
)
```

### Template Customization

```python
# Customize which models to include
config = mc.generate_template("openai", 
    models=["gpt-4", "gpt-4-turbo"],
    include_profiles=True,
    default_profile="balanced"
)

# Generate development vs production configurations
dev_config = mc.generate_development_config()    # Includes local + cloud options
prod_config = mc.generate_production_config()    # Reliable cloud models only
```

### Configuration Validation

```python
# Validate configuration before using it
result = mc.validate_config(config_dict=config)
if result.is_valid:
    mc.configure(config_dict=config)
else:
    print("Configuration errors:")
    for error in result.errors:
        print(f"  - {error}")
    
    print("Suggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")
```

## Configuration

Model Compass supports both YAML and JSON configuration files:

### YAML Configuration

```yaml
# config.yaml
default_profile: "production"

models:
  gpt-4:
    provider: openai
    model: gpt-4
    metadata:
      reasoning: high
      cost: high
      speed: medium
    parameters:
      temperature: 0.7
      max_tokens: 2048

  claude-sonnet:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      speed: fast

profiles:
  development:
    model: claude-sonnet
    parameters:
      temperature: 0.9
  
  production:
    model: gpt-4
    parameters:
      temperature: 0.1

aliases:
  main: production
  backup: development
  smart: gpt-4

providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
  
  anthropic:
    base_url: https://api.anthropic.com
    timeout: 45
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"
```

### Load Configuration

```python
import model_compass as mc

# From file
mc.configure(config_path="config.yaml")

# From dictionary
mc.configure(config_dict=config_data)
```

## Model Resolution Methods

### 1. Direct Model ID

```python
model = mc.model("gpt-4")
```

### 2. Physical Identifier (Provider/Model)

```python
model = mc.model("openai/gpt-4")
model = mc.model("anthropic/claude-3-sonnet")
```

### 3. Profile-Based Resolution

```python
# Use default profile
model = mc.model()

# Use specific profile
model = mc.model("production")

# Profile context switching
with mc.profile("development"):
    dev_model = mc.model()  # Uses development profile
```

### 4. Intent-Based URN Resolution

```python
# Find model by capabilities
model = mc.model("urn:llm:intent:chat?reasoning=high&cost=low")

# Multiple criteria
model = mc.model("urn:llm:intent:completion?speed=fast&provider=openai")

# Deployment preference
model = mc.model("urn:llm:intent:chat?deployment=local")
```

### 5. Alias Resolution

```python
model = mc.model("main")      # Resolves through alias chain
model = mc.model("backup")    # Fallback model
```

## Model Proxy API

The `ModelProxy` class provides convenient access to model information:

```python
model = mc.model("gpt-4")

# Basic properties
print(model.provider)        # "openai"
print(model.name)           # "gpt-4"
print(model.base_url)       # "https://api.openai.com/v1"

# Convenience properties
print(model.is_local)       # False
print(model.is_cloud)       # True
print(model.cost_tier)      # "high"
print(model.reasoning_level) # "high"
print(model.speed_tier)     # "medium"

# Dictionary-like access
print(model["temperature"])  # 0.7
print("max_tokens" in model) # True

# Get all parameters
params = model.parameters
metadata = model.metadata

# Framework compatibility
model_dict = model.to_dict()
```

## Collections API

Access models, profiles, and aliases through collection interfaces:

```python
# Iterate over available models
for model_id in mc.models:
    model = mc.models[model_id]
    print(f"{model_id}: {model.provider}/{model.name}")

# Find models by metadata
fast_models = mc.models.find(speed="fast")
local_models = mc.models.find(deployment="local")

# List available items
print("Available profiles:", list(mc.profiles))
print("Available aliases:", list(mc.aliases))

# Check availability
if "production" in mc.profiles:
    prod_model = mc.profiles["production"]
```

## Performance Optimizations

Enable performance optimizations for better performance with large configurations:

```python
import model_compass as mc

# Configure your models
mc.configure(config_path="large_config.yaml")

# Enable all performance optimizations
mc.enable_performance_optimizations()

# Get performance statistics
stats = mc.get_performance_stats()
print(f"Config cache hit rate: {stats['configuration_cache']['hit_rate']}%")
print(f"Resolution cache size: {stats['resolution_cache']['size']}")
```

## Error Handling

Model Compass provides detailed error messages with helpful suggestions:

```python
from model_compass.exceptions import ResolutionError, ConfigurationError

try:
    model = mc.model("nonexistent-model")
    provider = model.provider  # Triggers resolution
except ResolutionError as e:
    print(f"Resolution failed: {e}")
    print(f"Suggestions: {e.context['suggestions']}")
```

## Framework Integration

Model Compass integrates seamlessly with popular AI frameworks using explicit configuration:

### LangChain Integration

```python
import model_compass as mc
from langchain.llms import OpenAI

# Configure Model Compass with templates
config = mc.generate_template("openai")
mc.configure(config_dict=config)

# Get model configuration with intent-based selection
model = mc.model("urn:llm:intent:chat?reasoning=high")

# Use with LangChain
llm = OpenAI(
    model_name=model.model_name,
    openai_api_base=model.base_url,
    openai_api_key=model.get_header_value("Authorization").replace("Bearer ", ""),
    **model.parameters
)
```

### DSPy Integration

```python
import model_compass as mc
import dspy

# Configure Model Compass with multi-provider setup
config = mc.generate_template("multi", 
    openai_models=["gpt-4", "gpt-3.5-turbo"],
    anthropic_models=["claude-3-sonnet"]
)
mc.configure(config_dict=config)

# Environment-specific model selection
with mc.profile("development"):
    dev_model = mc.model()
    
with mc.profile("production"):
    prod_model = mc.model()

# Configure DSPy with the selected model
if prod_model.provider == "openai":
    lm = dspy.OpenAI(
        model=prod_model.model_name,
        api_base=prod_model.base_url,
        api_key=prod_model.get_header_value("Authorization").replace("Bearer ", ""),
        **prod_model.parameters
    )
    dspy.settings.configure(lm=lm)
```

### Autogen Integration

```python
import model_compass as mc
from autogen import AssistantAgent

# Configure Model Compass
config = mc.generate_template("multi")
mc.configure(config_dict=config)

# Get specialized models for different agents
reasoning_model = mc.model("urn:llm:intent:chat?reasoning=high")
fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")

# Create agents with appropriate models
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

fast_agent = AssistantAgent(
    name="fast_agent",
    system_message="You provide quick responses for simple tasks.",
    llm_config={
        "model": fast_model.model_name,
        "api_base": fast_model.base_url,
        "api_key": fast_model.get_header_value("Authorization").replace("Bearer ", ""),
        **fast_model.parameters
    }
)
```

### Custom Framework Integration

```python
import model_compass as mc

# Configure Model Compass
config = mc.generate_template("your-provider")
mc.configure(config_dict=config)

class CustomFrameworkAdapter:
    def __init__(self, model_identifier=None):
        self.model = mc.model(model_identifier)
        self.client = self._create_client()
    
    def _create_client(self):
        if self.model.provider == "openai":
            return OpenAIClient(
                api_key=self.model.get_api_key(),
                base_url=self.model.base_url,
                **self.model.parameters
            )
        # Add more providers as needed
    
    def generate(self, prompt):
        return self.client.generate(prompt)

# Usage with intent-based selection
adapter = CustomFrameworkAdapter("urn:llm:intent:chat?reasoning=high")
response = adapter.generate("Your prompt here")
```

For complete framework integration examples, see [examples/framework_integrations.py](examples/framework_integrations.py).

## Best Practices

### 1. Environment-Based Profiles

```python
# config.yaml
profiles:
  development:
    model: local-llama
    parameters:
      temperature: 0.9
      
  staging:
    model: claude-sonnet
    parameters:
      temperature: 0.5
      
  production:
    model: gpt-4
    parameters:
      temperature: 0.1

aliases:
  current: ${ENVIRONMENT:-development}
```

### 2. Fallback Chains

```python
aliases:
  primary: gpt-4
  secondary: claude-sonnet
  fallback: local-llama
  
  main: primary
  backup: secondary
```

### 3. Intent-Based Selection

```python
# Let Model Compass choose the best model
reasoning_model = mc.model("urn:llm:intent:analysis?reasoning=high")
fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")
local_model = mc.model("urn:llm:intent:completion?deployment=local")
```

### 4. Profile Context Management

```python
# Nested contexts
with mc.profile("production"):
    prod_model = mc.model()
    
    with mc.profile("development"):
        dev_model = mc.model()  # Temporary switch
        
    # Back to production context
    prod_model_again = mc.model()
```

## Troubleshooting

### Common Configuration Issues

#### "No configuration loaded" Error

```python
# ❌ This will fail
import model_compass as mc
model = mc.model("gpt-4")  # ConfigurationError: No configuration loaded

# ✅ Configure first
mc.configure(config_dict=mc.generate_template("openai"))
model = mc.model("gpt-4")  # Works!
```

#### Model Not Found

```python
# ❌ Model doesn't exist in configuration
model = mc.model("nonexistent-model")  # ResolutionError with suggestions

# ✅ Check available models
print("Available models:", mc.list_models())

# ✅ Add model to configuration or use existing one
model = mc.model("gpt-4")  # Use existing model
```

#### Intent Resolution No Matches

```python
# ❌ No models match criteria
model = mc.model("urn:llm:intent:chat?reasoning=impossible")

# ✅ Check what metadata is available
config = mc.generate_template("multi")
result = mc.validate_config(config_dict=config)
print("Suggestions:", result.suggestions)

# ✅ Use realistic criteria
model = mc.model("urn:llm:intent:chat?reasoning=high&cost=medium")
```

#### Configuration Validation

```python
# Validate your configuration
result = mc.validate_config(config_dict=your_config)

if not result.is_valid:
    print("Errors:")
    for error in result.errors:
        print(f"  - {error}")

if result.has_warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

print("Suggestions:")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")
```

### Getting Help

```python
# Check configuration status
status = mc.is_configured_with_details()
if not status["configured"]:
    print(status["quick_start"])

# Get configuration suggestions
if mc.is_configured():
    suggestions = mc.get_config_suggestions()
    for suggestion in suggestions:
        print(f"💡 {suggestion}")

# Generate examples for missing models
from model_compass import ConfigurationExamples
example = ConfigurationExamples.get_model_example("your-model", "your-provider")
print("Add this to your configuration:")
print(example)
```

## API Reference

### Core Functions

- `configure(config_path=None, config_dict=None)` - Load configuration
- `model(identifier=None)` - Get model proxy
- `profile(profile_name)` - Profile context manager
- `is_configured()` - Check if library is configured
- `enable_performance_optimizations()` - Enable caching and optimizations
- `get_performance_stats()` - Get performance statistics

### Configuration-Focused API

- `generate_template(provider, **kwargs)` - Generate configuration templates
- `validate_config(config_dict=None, config_path=None)` - Validate configuration
- `get_config_suggestions()` - Get improvement recommendations
- `is_configured_with_details()` - Detailed configuration status
- `list_models()` - List available model identifiers
- `list_profiles()` - List available profile names
- `list_aliases()` - List available alias names

### Template Generators

- `generate_development_config()` - Development-friendly configuration
- `generate_production_config()` - Production-ready configuration
- `ConfigurationTemplates.generate_openai_template()` - OpenAI models
- `ConfigurationTemplates.generate_anthropic_template()` - Anthropic models
- `ConfigurationTemplates.generate_ollama_template()` - Ollama local models
- `ConfigurationTemplates.generate_multi_provider_template()` - Multi-provider setup

### Configuration Utilities

- `ConfigurationValidator.validate(config)` - Validate configuration structure
- `ConfigurationExamples.get_model_example(model_id)` - Generate model examples
- `ConfigurationExamples.get_quick_start_instructions()` - Quick-start help

### Collections

- `models` - ModelsCollection for accessing models
- `profiles` - ProfilesCollection for accessing profiles  
- `aliases` - AliasesCollection for accessing aliases

### Exceptions

- `LLMConfigError` - Base exception class
- `ConfigurationError` - Configuration-related errors
- `ResolutionError` - Model resolution errors (with helpful suggestions)
- `CircularAliasError` - Circular alias reference errors
- `NetworkError` - Network connectivity errors
- `ProviderError` - Provider API errors

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.