# Model Compass Configuration Examples

This directory contains comprehensive examples for configuring and using Model Compass with different providers, environments, and use cases.

## Configuration Files

### Single Provider Examples

- **[openai_complete.yaml](openai_complete.yaml)** - Complete OpenAI configuration with multiple GPT models, profiles for different environments, and rich metadata for intent resolution.

- **[anthropic_complete.yaml](anthropic_complete.yaml)** - Complete Anthropic Claude configuration with all Claude 3 models, task-specific profiles, and comprehensive metadata.

- **[ollama_complete.yaml](ollama_complete.yaml)** - Complete Ollama configuration for local models including Llama 3, CodeLlama, Phi-3, and other open-source models.

### Multi-Provider Examples

- **[multi_provider_complete.yaml](multi_provider_complete.yaml)** - Comprehensive multi-provider setup combining OpenAI, Anthropic, and Ollama with intelligent fallbacks and environment-specific profiles.

## Python Examples

### Environment Setup

- **[multi_environment_setup.py](multi_environment_setup.py)** - Demonstrates how to configure Model Compass for different environments (development, staging, production) with appropriate model selections and fallbacks.

### Intent Resolution

- **[intent_resolution_best_practices.py](intent_resolution_best_practices.py)** - Best practices for setting up model metadata to enable effective intent-based resolution with URN queries. Includes examples of custom metadata fields and business-specific requirements.

### Troubleshooting

- **[troubleshooting_guide.py](troubleshooting_guide.py)** - Comprehensive guide for diagnosing and fixing common configuration problems, including error handling, debugging techniques, and recovery strategies.

## Quick Start Examples

### Using Configuration Templates

```python
import model_compass as mc

# Generate and use OpenAI template
config = mc.generate_template("openai")
mc.configure(config_dict=config)

# Generate multi-provider template
config = mc.generate_template("multi",
    openai_models=["gpt-4", "gpt-3.5-turbo"],
    anthropic_models=["claude-3-sonnet"],
    ollama_models=["llama3"]
)
mc.configure(config_dict=config)
```

### Loading from Files

```python
import model_compass as mc

# Load from YAML file
mc.configure(config_path="examples/openai_complete.yaml")

# Validate before loading
result = mc.validate_config(config_path="examples/multi_provider_complete.yaml")
if result.is_valid:
    mc.configure(config_path="examples/multi_provider_complete.yaml")
```

### Environment-Based Configuration

```python
import os
import model_compass as mc

# Set environment
os.environ["ENVIRONMENT"] = "production"

# Load environment-specific configuration
from examples.multi_environment_setup import setup_model_compass
setup_model_compass()
```

## Configuration Patterns

### 1. Environment-Based Profiles

Use different models for different environments:

```yaml
profiles:
  development:
    model: llama3  # Local model for dev
  staging:
    model: claude-3-haiku  # Fast cloud model for staging  
  production:
    model: gpt-4  # Best quality for production
```

### 2. Task-Specific Profiles

Optimize models for specific tasks:

```yaml
profiles:
  coding:
    model: codellama
    parameters:
      temperature: 0.1
  creative:
    model: gpt-4
    parameters:
      temperature: 0.9
  analysis:
    model: claude-3-sonnet
    parameters:
      temperature: 0.3
```

### 3. Fallback Chains

Set up intelligent fallbacks:

```yaml
aliases:
  primary: gpt-4
  secondary: claude-3-sonnet
  emergency: llama3
  main: primary  # Points to primary by default
```

### 4. Intent Resolution Metadata

Rich metadata for intent-based selection:

```yaml
models:
  gpt-4:
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, completion, function_calling]
      use_cases: [analysis, writing, complex_reasoning]
      deployment: cloud
      compliance: enterprise
```

## Best Practices

### Configuration Validation

Always validate configurations before using them:

```python
# Validate configuration
result = mc.validate_config(config_dict=config)

if not result.is_valid:
    print("Configuration errors:")
    for error in result.errors:
        print(f"  - {error}")

if result.has_warnings:
    print("Configuration warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

print("Suggestions for improvement:")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")
```

### Error Handling

Handle configuration and resolution errors gracefully:

```python
from model_compass.exceptions import ConfigurationError, ResolutionError

try:
    mc.configure(config_path="config.yaml")
    model = mc.model("my-model")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Show quick-start instructions
    status = mc.is_configured_with_details()
    if not status["configured"]:
        print(status["quick_start"])
except ResolutionError as e:
    print(f"Resolution error: {e}")
    # Show detailed error with suggestions
    print(e.get_detailed_message())
```

### Performance Optimization

Enable performance optimizations for large configurations:

```python
# Configure Model Compass
mc.configure(config_path="large_config.yaml")

# Enable performance optimizations
mc.enable_performance_optimizations()

# Monitor performance
stats = mc.get_performance_stats()
print(f"Configuration cache hit rate: {stats.get('configuration_cache', {}).get('hit_rate', 0)}%")
```

## Running the Examples

### Prerequisites

1. Install Model Compass:
   ```bash
   pip install model-compass
   ```

2. Set up API keys (for cloud providers):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. Install Ollama (for local models):
   ```bash
   # Follow instructions at https://ollama.ai
   ollama pull llama3
   ollama pull codellama
   ```

### Running Examples

```bash
# Run environment setup example
python examples/multi_environment_setup.py

# Run intent resolution examples
python examples/intent_resolution_best_practices.py

# Run troubleshooting guide
python examples/troubleshooting_guide.py

# Load configuration files
python -c "
import model_compass as mc
mc.configure(config_path='examples/openai_complete.yaml')
print('Available models:', mc.list_models())
"
```

## Customization

### Adding Custom Metadata

Extend models with custom business metadata:

```yaml
models:
  business-model:
    provider: openai
    model: gpt-4
    metadata:
      # Standard fields
      reasoning: high
      cost: high
      # Custom business fields
      department: finance
      security_level: confidential
      compliance: sox
      approved_for: [financial_analysis, risk_assessment]
```

### Custom Profiles

Create profiles for specific use cases:

```yaml
profiles:
  financial_analysis:
    model: business-model
    parameters:
      temperature: 0.1
      max_tokens: 4096
  
  customer_support:
    model: claude-3-haiku
    parameters:
      temperature: 0.7
      max_tokens: 2048
```

### Environment Variables

Use environment variables for dynamic configuration:

```yaml
providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
  
profiles:
  current:
    model: "${DEFAULT_MODEL:-gpt-3.5-turbo}"
```

## Troubleshooting

If you encounter issues:

1. **Validate your configuration**:
   ```python
   result = mc.validate_config(config_path="your-config.yaml")
   print("Valid:", result.is_valid)
   ```

2. **Check configuration status**:
   ```python
   status = mc.is_configured_with_details()
   print("Status:", status)
   ```

3. **Get suggestions**:
   ```python
   suggestions = mc.get_config_suggestions()
   for suggestion in suggestions:
       print(f"💡 {suggestion}")
   ```

4. **Run the troubleshooting guide**:
   ```bash
   python examples/troubleshooting_guide.py
   ```

For more help, see the [main documentation](../README.md) or the troubleshooting guide example.