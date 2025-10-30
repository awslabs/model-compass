# API Reference

This document provides detailed API reference for Model Compass.

## Core Functions

### `configure(config_path=None, config_dict=None)`

Load configuration from file or dictionary.

**Parameters:**
- `config_path` (str, optional): Path to YAML or JSON configuration file
- `config_dict` (dict, optional): Configuration dictionary

**Raises:**
- `ConfigurationError`: If configuration is invalid or cannot be loaded
- `ValueError`: If both or neither arguments are provided

**Example:**
```python
import model_compass as mc

# From file
mc.configure(config_path="config.yaml")

# From dictionary
mc.configure(config_dict={
    "models": {"gpt-4": {"provider": "openai", "model": "gpt-4"}},
    "providers": {"openai": {"base_url": "https://api.openai.com/v1"}}
})
```

### `model(identifier=None)`

Get a model proxy object with property-based access.

**Parameters:**
- `identifier` (str, optional): Model identifier (URN, profile, alias, or physical ID). If None, uses the current profile or default profile.

**Returns:**
- `ModelProxy`: Model proxy instance for the resolved model

**Raises:**
- `ConfigurationError`: If library is not configured
- `ResolutionError`: If identifier cannot be resolved
- `ValueError`: If identifier is not a string or None

**Example:**
```python
# Get default model
default_model = mc.model()

# Get specific model by profile
prod_model = mc.model("prod")

# Get model by intent
smart_model = mc.model("urn:llm:intent:chat?reasoning=high")

# Get model by physical identifier
openai_model = mc.model("openai/gpt-4")
```

### `profile(profile_name)`

Context manager for temporarily switching to a different profile.

**Parameters:**
- `profile_name` (str): Name of the profile to switch to

**Yields:**
- `str`: The profile name that was activated

**Raises:**
- `ConfigurationError`: If library is not configured
- `ResolutionError`: If profile does not exist
- `ValueError`: If profile_name is None or empty

**Example:**
```python
# Default profile
default_model = mc.model()

# Temporarily switch to local profile
with mc.profile("local") as active_profile:
    print(f"Active profile: {active_profile}")
    local_model = mc.model()
    
    # Nested context switching
    with mc.profile("prod"):
        prod_model = mc.model()

# Back to default
default_again = mc.model()
```

### `is_configured()`

Check if the library has been configured.

**Returns:**
- `bool`: True if library is configured, False otherwise

**Example:**
```python
if not mc.is_configured():
    mc.configure(config_path="config.yaml")
```

### `enable_performance_optimizations()`

Enable all performance optimizations for the library.

This function enables:
- Configuration file caching to avoid re-parsing files
- Resolution result caching for frequently accessed models
- Lazy loading of model definitions and provider configurations
- Optimized intent matching algorithm for large model sets

**Raises:**
- `ConfigurationError`: If library is not configured

**Example:**
```python
mc.configure(config_path="config.yaml")
mc.enable_performance_optimizations()
```

### `get_performance_stats()`

Get performance statistics from all caches and optimizations.

**Returns:**
- `dict`: Dictionary with comprehensive performance statistics

**Example:**
```python
stats = mc.get_performance_stats()
print(f"Config cache hit rate: {stats['configuration_cache']['hit_rate']}%")
print(f"Resolution cache size: {stats['resolution_cache']['size']}")
```

### `get_current_profile()`

Get the currently active profile.

**Returns:**
- `str` or `None`: Current profile name or None if using default

**Example:**
```python
print(mc.get_current_profile())  # None (default)

with mc.profile("local"):
    print(mc.get_current_profile())  # "local"
```

### `get_profile_stack()`

Get the current profile context stack.

**Returns:**
- `list`: List of profile names in the current stack (outermost first)

**Example:**
```python
with mc.profile("local"):
    with mc.profile("dev"):
        print(mc.get_profile_stack())  # [None, "local"]
```

## ModelProxy Class

The `ModelProxy` class provides convenient access to model configuration through properties and dictionary-like interface.

### Properties

#### Basic Properties

- `provider` (str): Provider name (e.g., "openai", "anthropic")
- `name` (str): Model name (e.g., "gpt-4", "claude-3-sonnet")
- `base_url` (str): Provider base URL
- `parameters` (dict): Model parameters (temperature, max_tokens, etc.)
- `metadata` (dict): Model metadata (reasoning, cost, speed, etc.)

#### Convenience Properties

- `is_local` (bool): True if model runs locally
- `is_cloud` (bool): True if model runs in the cloud
- `cost_tier` (str): Cost tier ("low", "medium", "high")
- `reasoning_level` (str): Reasoning capability ("low", "medium", "high")
- `speed_tier` (str): Speed tier ("slow", "medium", "fast")

### Dictionary-like Interface

```python
model = mc.model("gpt-4")

# Access parameters
temperature = model["temperature"]
max_tokens = model["max_tokens"]

# Check if parameter exists
if "top_p" in model:
    top_p = model["top_p"]

# Get all keys
param_names = list(model.keys())

# Get all items
all_params = dict(model.items())
```

### Methods

#### `to_dict()`

Convert model configuration to dictionary format.

**Returns:**
- `dict`: Complete model configuration as dictionary

**Example:**
```python
model = mc.model("gpt-4")
config_dict = model.to_dict()
print(config_dict)
# {
#     "provider": "openai",
#     "name": "gpt-4", 
#     "base_url": "https://api.openai.com/v1",
#     "parameters": {"temperature": 0.7, "max_tokens": 2048},
#     "metadata": {"reasoning": "high", "cost": "high"}
# }
```

## Collections

### ModelsCollection

Access and iterate over available models.

```python
# Iterate over models
for model_id in mc.models:
    model = mc.models[model_id]
    print(f"{model_id}: {model.provider}/{model.name}")

# Access specific model
gpt4_model = mc.models["gpt-4"]

# Check if model exists
if "claude-sonnet" in mc.models:
    claude = mc.models["claude-sonnet"]

# Find models by metadata
fast_models = mc.models.find(speed="fast")
local_models = mc.models.find(deployment="local")
high_reasoning = mc.models.find(reasoning="high", cost="low")
```

#### `find(**criteria)`

Find models matching the given metadata criteria.

**Parameters:**
- `**criteria`: Keyword arguments for metadata criteria

**Returns:**
- `list`: List of model IDs matching the criteria

### ProfilesCollection

Access and iterate over available profiles.

```python
# Iterate over profiles
for profile_name in mc.profiles:
    profile = mc.profiles[profile_name]
    print(f"{profile_name}: {profile.model}")

# Access specific profile
prod_profile = mc.profiles["production"]

# Check if profile exists
if "development" in mc.profiles:
    dev_profile = mc.profiles["development"]
```

### AliasesCollection

Access and iterate over available aliases.

```python
# Iterate over aliases
for alias_name in mc.aliases:
    target = mc.aliases[alias_name]
    print(f"{alias_name} -> {target}")

# Access specific alias
main_target = mc.aliases["main"]

# Check if alias exists
if "backup" in mc.aliases:
    backup_target = mc.aliases["backup"]
```

## Data Models

### ModelConfig

Represents a model configuration.

**Attributes:**
- `provider` (str): Provider name
- `model` (str): Model name
- `metadata` (dict): Model metadata
- `parameters` (dict): Default parameters

### Profile

Represents a profile configuration.

**Attributes:**
- `model` (str): Model reference
- `parameters` (dict): Profile-specific parameters

### ProviderConfig

Represents a provider configuration.

**Attributes:**
- `base_url` (str): Provider base URL
- `timeout` (int): Request timeout in seconds
- `headers` (dict): HTTP headers

### ResolvedModel

Represents a fully resolved model with provider configuration.

**Attributes:**
- `config` (ModelConfig): Model configuration
- `provider_config` (ProviderConfig): Provider configuration
- `effective_parameters` (dict): Merged parameters

## Exceptions

### LLMConfigError

Base exception class for all Model Compass errors.

**Attributes:**
- `message` (str): Error message
- `context` (dict): Additional error context

### ConfigurationError

Raised when configuration is invalid or cannot be loaded.

**Inherits from:** `LLMConfigError`

**Example:**
```python
from model_compass.exceptions import ConfigurationError

try:
    mc.configure(config_dict={"invalid": "config"})
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Context: {e.context}")
```

### ResolutionError

Raised when a model identifier cannot be resolved.

**Inherits from:** `LLMConfigError`

**Additional Attributes:**
- `identifier` (str): The identifier that failed to resolve
- `resolution_path` (list): Resolution path for debugging
- `suggestions` (list): Helpful suggestions for fixing the error

**Example:**
```python
from model_compass.exceptions import ResolutionError

try:
    model = mc.model("nonexistent-model")
    _ = model.provider  # Triggers resolution
except ResolutionError as e:
    print(f"Resolution failed: {e}")
    print(f"Identifier: {e.identifier}")
    print(f"Suggestions: {e.context['suggestions']}")
```

### CircularAliasError

Raised when circular alias references are detected.

**Inherits from:** `LLMConfigError`

**Additional Attributes:**
- `alias_chain` (list): The circular alias chain
- `suggestions` (list): Suggestions for fixing the circular reference

**Example:**
```python
from model_compass.exceptions import CircularAliasError

try:
    model = mc.model("circular-alias")
    _ = model.provider
except CircularAliasError as e:
    print(f"Circular alias detected: {e}")
    print(f"Chain: {' -> '.join(e.alias_chain)}")
    print(f"Suggestions: {e.suggestions}")
```

## URN Format

Model Compass supports URN-based intent resolution with the following format:

```
urn:llm:intent:<service>?<param1>=<value1>&<param2>=<value2>
```

### Supported Services

- `chat` - Conversational models
- `completion` - Text completion models
- `analysis` - Analysis and reasoning models

### Supported Parameters

- `reasoning` - Reasoning capability (low, medium, high)
- `speed` - Response speed (slow, medium, fast)
- `cost` - Cost tier (low, medium, high)
- `provider` - Specific provider (openai, anthropic, ollama)
- `deployment` - Deployment type (local, cloud)
- `context_length` - Minimum context length (numeric)
- `capabilities` - Required capabilities (chat, completion, vision, etc.)

### Examples

```python
# High reasoning capability
model = mc.model("urn:llm:intent:chat?reasoning=high")

# Fast and low cost
model = mc.model("urn:llm:intent:completion?speed=fast&cost=low")

# Local deployment
model = mc.model("urn:llm:intent:chat?deployment=local")

# Specific provider with high reasoning
model = mc.model("urn:llm:intent:analysis?provider=anthropic&reasoning=high")

# Multiple capabilities
model = mc.model("urn:llm:intent:chat?capabilities=vision&reasoning=high")
```

## Configuration Schema

### Complete Configuration Example

```yaml
# Global settings
default_profile: "production"
resolution_timeout: 30

# Model definitions
models:
  gpt-4:
    provider: openai
    model: gpt-4
    metadata:
      reasoning: high
      cost: high
      speed: medium
      context_length: 8192
      capabilities: [chat, completion, function_calling]
      deployment: cloud
    parameters:
      temperature: 0.7
      max_tokens: 2048
      top_p: 1.0

  claude-sonnet:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      context_length: 200000
      capabilities: [chat, analysis, vision]
      deployment: cloud
    parameters:
      temperature: 0.5
      max_tokens: 4096

# Profile definitions
profiles:
  development:
    model: claude-sonnet
    parameters:
      temperature: 0.9
      
  production:
    model: gpt-4
    parameters:
      temperature: 0.1
      max_tokens: 1024

# Alias definitions
aliases:
  main: production
  backup: development
  smart: gpt-4
  fast: claude-sonnet

# Provider definitions
providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
      Content-Type: "application/json"
      
  anthropic:
    base_url: https://api.anthropic.com
    timeout: 45
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"
      Content-Type: "application/json"
      anthropic-version: "2023-06-01"
```

### Required Fields

#### Models
- `provider` (str): Provider name
- `model` (str): Model name

#### Providers  
- `base_url` (str): Provider base URL

#### Profiles
- `model` (str): Model reference

### Optional Fields

#### Models
- `metadata` (dict): Model metadata for intent matching
- `parameters` (dict): Default model parameters

#### Providers
- `timeout` (int): Request timeout (default: 30)
- `headers` (dict): HTTP headers

#### Profiles
- `parameters` (dict): Profile-specific parameters

#### Global
- `default_profile` (str): Default profile name
- `resolution_timeout` (int): Resolution timeout (default: 30)