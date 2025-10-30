# Configuration Examples

This document provides comprehensive configuration examples for different use cases and deployment patterns.

## Basic Configuration

### Simple Single-Provider Setup

```yaml
# config.yaml - Basic OpenAI setup
models:
  gpt-4:
    provider: openai
    model: gpt-4
    parameters:
      temperature: 0.7
      max_tokens: 2048

  gpt-3.5-turbo:
    provider: openai
    model: gpt-3.5-turbo
    parameters:
      temperature: 0.8
      max_tokens: 1024

providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
      Content-Type: "application/json"

profiles:
  default:
    model: gpt-3.5-turbo
  
  premium:
    model: gpt-4
    parameters:
      temperature: 0.1

aliases:
  main: default
  smart: premium
```

### Multi-Provider Configuration

```yaml
# config.yaml - Multiple providers
models:
  # OpenAI models
  gpt-4:
    provider: openai
    model: gpt-4
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, completion, function_calling]
      deployment: cloud
    parameters:
      temperature: 0.7
      max_tokens: 2048

  gpt-3.5-turbo:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
    parameters:
      temperature: 0.8
      max_tokens: 1024

  # Anthropic models
  claude-opus:
    provider: anthropic
    model: claude-3-opus-20240229
    metadata:
      reasoning: high
      cost: high
      speed: slow
      capabilities: [chat, analysis, vision]
      deployment: cloud
    parameters:
      temperature: 0.5
      max_tokens: 4096

  claude-sonnet:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      capabilities: [chat, analysis]
      deployment: cloud
    parameters:
      temperature: 0.6
      max_tokens: 2048

  # Local models
  llama3:
    provider: ollama
    model: llama3
    metadata:
      reasoning: medium
      cost: low
      speed: medium
      capabilities: [chat, completion]
      deployment: local
    parameters:
      temperature: 0.7
      num_predict: 1024

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

  ollama:
    base_url: http://localhost:11434
    timeout: 60
    headers:
      Content-Type: "application/json"

profiles:
  development:
    model: llama3
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
      max_tokens: 1024

aliases:
  current: ${ENVIRONMENT:-development}
  main: production
  backup: staging
  local: llama3
  cloud: gpt-4
  fast: gpt-3.5-turbo
  smart: claude-opus
```

## Environment-Specific Configurations

### Development Environment

```yaml
# configs/development.yaml
default_profile: development

models:
  local-llama:
    provider: ollama
    model: llama3
    metadata:
      reasoning: medium
      cost: low
      speed: medium
      deployment: local
    parameters:
      temperature: 0.9
      num_predict: 512

  dev-gpt:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
    parameters:
      temperature: 0.8
      max_tokens: 1000

providers:
  ollama:
    base_url: http://localhost:11434
    timeout: 120
    headers:
      Content-Type: "application/json"

  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

profiles:
  development:
    model: local-llama
    parameters:
      temperature: 0.9

  cloud-dev:
    model: dev-gpt
    parameters:
      temperature: 0.8

aliases:
  main: development
  fallback: cloud-dev
  local: local-llama
```

### Staging Environment

```yaml
# configs/staging.yaml
default_profile: staging

models:
  staging-claude:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      deployment: cloud
    parameters:
      temperature: 0.5
      max_tokens: 2048

  staging-gpt:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
    parameters:
      temperature: 0.6
      max_tokens: 1500

providers:
  anthropic:
    base_url: https://api.anthropic.com
    timeout: 45
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"
      Content-Type: "application/json"

  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

profiles:
  staging:
    model: staging-claude
    parameters:
      temperature: 0.5

  staging-fast:
    model: staging-gpt
    parameters:
      temperature: 0.6

aliases:
  main: staging
  fast: staging-fast
  primary: staging-claude
```

### Production Environment

```yaml
# configs/production.yaml
default_profile: production
resolution_timeout: 60

models:
  prod-gpt4:
    provider: openai
    model: gpt-4
    metadata:
      reasoning: high
      cost: high
      speed: medium
      deployment: cloud
      reliability: high
    parameters:
      temperature: 0.1
      max_tokens: 2048
      top_p: 0.95

  prod-claude:
    provider: anthropic
    model: claude-3-opus-20240229
    metadata:
      reasoning: high
      cost: high
      speed: slow
      deployment: cloud
      reliability: high
    parameters:
      temperature: 0.0
      max_tokens: 4096

  prod-fallback:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      reliability: medium
    parameters:
      temperature: 0.3
      max_tokens: 1024

providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 60
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
      Content-Type: "application/json"

  anthropic:
    base_url: https://api.anthropic.com
    timeout: 90
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"
      Content-Type: "application/json"
      anthropic-version: "2023-06-01"

profiles:
  production:
    model: prod-gpt4
    parameters:
      temperature: 0.1

  production-reasoning:
    model: prod-claude
    parameters:
      temperature: 0.0

  production-fast:
    model: prod-fallback
    parameters:
      temperature: 0.3

aliases:
  main: production
  primary: prod-gpt4
  reasoning: production-reasoning
  fallback: production-fast
  backup: prod-fallback
```

## Use Case-Specific Configurations

### Content Generation Pipeline

```yaml
# configs/content_generation.yaml
models:
  creative-writer:
    provider: openai
    model: gpt-4
    metadata:
      creativity: high
      reasoning: high
      cost: high
      use_case: creative_writing
    parameters:
      temperature: 0.9
      max_tokens: 3000
      top_p: 0.95

  editor:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      use_case: editing
    parameters:
      temperature: 0.3
      max_tokens: 2000

  fact-checker:
    provider: openai
    model: gpt-4
    metadata:
      reasoning: high
      accuracy: high
      cost: high
      use_case: fact_checking
    parameters:
      temperature: 0.1
      max_tokens: 1500

  summarizer:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      speed: fast
      cost: low
      use_case: summarization
    parameters:
      temperature: 0.4
      max_tokens: 800

providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  content-creation:
    model: creative-writer
    parameters:
      temperature: 0.9

  content-editing:
    model: editor
    parameters:
      temperature: 0.3

  content-review:
    model: fact-checker
    parameters:
      temperature: 0.1

aliases:
  writer: creative-writer
  editor: editor
  checker: fact-checker
  summarizer: summarizer
```

### Data Analysis Pipeline

```yaml
# configs/data_analysis.yaml
models:
  data-analyst:
    provider: anthropic
    model: claude-3-opus-20240229
    metadata:
      reasoning: high
      analysis: high
      cost: high
      use_case: data_analysis
    parameters:
      temperature: 0.2
      max_tokens: 4000

  code-generator:
    provider: openai
    model: gpt-4
    metadata:
      coding: high
      reasoning: high
      cost: high
      use_case: code_generation
    parameters:
      temperature: 0.1
      max_tokens: 2000

  report-writer:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      writing: high
      speed: fast
      cost: low
      use_case: report_writing
    parameters:
      temperature: 0.5
      max_tokens: 2500

  validator:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      validation: high
      cost: medium
      use_case: validation
    parameters:
      temperature: 0.0
      max_tokens: 1500

providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  analysis:
    model: data-analyst
    parameters:
      temperature: 0.2

  coding:
    model: code-generator
    parameters:
      temperature: 0.1

  reporting:
    model: report-writer
    parameters:
      temperature: 0.5

  validation:
    model: validator
    parameters:
      temperature: 0.0

aliases:
  analyst: data-analyst
  coder: code-generator
  writer: report-writer
  validator: validator
```

### Customer Support System

```yaml
# configs/customer_support.yaml
models:
  support-agent:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      empathy: high
      speed: fast
      cost: low
      use_case: customer_support
    parameters:
      temperature: 0.7
      max_tokens: 1000

  escalation-handler:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      empathy: high
      cost: medium
      use_case: escalation_handling
    parameters:
      temperature: 0.5
      max_tokens: 2000

  knowledge-base:
    provider: openai
    model: gpt-4
    metadata:
      accuracy: high
      reasoning: high
      cost: high
      use_case: knowledge_retrieval
    parameters:
      temperature: 0.1
      max_tokens: 1500

  sentiment-analyzer:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      analysis: high
      speed: fast
      cost: low
      use_case: sentiment_analysis
    parameters:
      temperature: 0.3
      max_tokens: 500

providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  tier1-support:
    model: support-agent
    parameters:
      temperature: 0.7

  tier2-support:
    model: escalation-handler
    parameters:
      temperature: 0.5

  knowledge-lookup:
    model: knowledge-base
    parameters:
      temperature: 0.1

  sentiment-check:
    model: sentiment-analyzer
    parameters:
      temperature: 0.3

aliases:
  support: tier1-support
  escalation: tier2-support
  knowledge: knowledge-lookup
  sentiment: sentiment-check
```

## Advanced Configuration Patterns

### Cost-Optimized Configuration

```yaml
# configs/cost_optimized.yaml
models:
  # Tier 1: Cheapest models for simple tasks
  cheap-fast:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      cost: low
      speed: fast
      reasoning: medium
      tier: 1
    parameters:
      temperature: 0.7
      max_tokens: 500

  # Tier 2: Balanced cost/performance
  balanced:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      cost: medium
      speed: fast
      reasoning: high
      tier: 2
    parameters:
      temperature: 0.5
      max_tokens: 1500

  # Tier 3: Premium models for complex tasks
  premium:
    provider: openai
    model: gpt-4
    metadata:
      cost: high
      speed: medium
      reasoning: high
      tier: 3
    parameters:
      temperature: 0.3
      max_tokens: 2000

providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  cost-conscious:
    model: cheap-fast
    parameters:
      max_tokens: 300

  balanced-performance:
    model: balanced
    parameters:
      max_tokens: 1000

  premium-quality:
    model: premium
    parameters:
      max_tokens: 1500

aliases:
  # Cost-based routing
  cheap: cost-conscious
  medium: balanced-performance
  expensive: premium-quality
  
  # Task-based routing with cost consideration
  simple: cheap-fast
  complex: balanced
  critical: premium
```

### High-Availability Configuration

```yaml
# configs/high_availability.yaml
models:
  primary-gpt4:
    provider: openai-primary
    model: gpt-4
    metadata:
      reliability: high
      priority: 1
    parameters:
      temperature: 0.3
      max_tokens: 2000

  backup-gpt4:
    provider: openai-backup
    model: gpt-4
    metadata:
      reliability: high
      priority: 2
    parameters:
      temperature: 0.3
      max_tokens: 2000

  fallback-claude:
    provider: anthropic
    model: claude-3-opus-20240229
    metadata:
      reliability: high
      priority: 3
    parameters:
      temperature: 0.2
      max_tokens: 2000

  emergency-gpt35:
    provider: openai-primary
    model: gpt-3.5-turbo
    metadata:
      reliability: medium
      priority: 4
    parameters:
      temperature: 0.5
      max_tokens: 1500

providers:
  openai-primary:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY_PRIMARY}"

  openai-backup:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY_BACKUP}"

  anthropic:
    base_url: https://api.anthropic.com
    timeout: 45
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  high-availability:
    model: primary-gpt4
    parameters:
      temperature: 0.3

aliases:
  # Failover chain
  primary: primary-gpt4
  backup: backup-gpt4
  fallback: fallback-claude
  emergency: emergency-gpt35
  
  # Main entry point
  main: primary
```

### Multi-Region Configuration

```yaml
# configs/multi_region.yaml
models:
  us-east-gpt4:
    provider: openai-us-east
    model: gpt-4
    metadata:
      region: us-east
      latency: low
    parameters:
      temperature: 0.3

  us-west-gpt4:
    provider: openai-us-west
    model: gpt-4
    metadata:
      region: us-west
      latency: medium
    parameters:
      temperature: 0.3

  eu-gpt4:
    provider: openai-eu
    model: gpt-4
    metadata:
      region: europe
      latency: high
    parameters:
      temperature: 0.3

  asia-claude:
    provider: anthropic-asia
    model: claude-3-opus-20240229
    metadata:
      region: asia
      latency: medium
    parameters:
      temperature: 0.2

providers:
  openai-us-east:
    base_url: https://api.openai.com/v1
    timeout: 20
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  openai-us-west:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  openai-eu:
    base_url: https://api.openai.com/v1
    timeout: 40
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"

  anthropic-asia:
    base_url: https://api.anthropic.com
    timeout: 50
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

profiles:
  us-east:
    model: us-east-gpt4

  us-west:
    model: us-west-gpt4

  europe:
    model: eu-gpt4

  asia:
    model: asia-claude

aliases:
  # Region-based routing
  local: ${REGION:-us-east}
  nearest: us-east-gpt4
  
  # Latency-optimized
  fastest: us-east-gpt4
  balanced: us-west-gpt4
```

## Configuration Best Practices

### 1. Environment Variable Usage

```yaml
# Use environment variables for sensitive data
providers:
  openai:
    base_url: https://api.openai.com/v1
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
  
  anthropic:
    base_url: https://api.anthropic.com
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"

# Use environment variables for environment-specific settings
default_profile: ${ENVIRONMENT:-development}

aliases:
  current: ${ENVIRONMENT:-development}
  region: ${REGION:-us-east}
```

### 2. Metadata-Driven Configuration

```yaml
# Rich metadata for intent matching
models:
  gpt-4:
    provider: openai
    model: gpt-4
    metadata:
      # Capability metadata
      reasoning: high
      creativity: high
      analysis: high
      coding: high
      
      # Performance metadata
      speed: medium
      cost: high
      reliability: high
      
      # Feature metadata
      capabilities: [chat, completion, function_calling, vision]
      context_length: 8192
      
      # Deployment metadata
      deployment: cloud
      region: global
      
      # Use case metadata
      use_cases: [analysis, creative_writing, coding, reasoning]
      
      # Quality metadata
      accuracy: high
      consistency: high
```

### 3. Hierarchical Profiles

```yaml
# Base profiles
profiles:
  base-development:
    parameters:
      temperature: 0.8
      max_tokens: 1000

  base-production:
    parameters:
      temperature: 0.2
      max_tokens: 2000

# Specific profiles inheriting base settings
  development-fast:
    model: gpt-3.5-turbo
    parameters:
      temperature: 0.8  # Inherits from base-development
      max_tokens: 500   # Override

  production-reasoning:
    model: gpt-4
    parameters:
      temperature: 0.1  # Override base-production
      max_tokens: 3000  # Override
```

### 4. Alias Chains for Flexibility

```yaml
aliases:
  # Environment chains
  development: dev-fast
  staging: staging-balanced
  production: prod-premium
  
  # Feature chains
  reasoning: high-reasoning
  creative: high-creativity
  fast: low-latency
  
  # Fallback chains
  primary: prod-premium
  secondary: prod-balanced
  tertiary: prod-fast
  
  # Current environment
  current: ${ENVIRONMENT:-development}
  main: primary
  backup: secondary
```

### 5. Validation and Documentation

```yaml
# Document your configuration
# This configuration supports:
# - Multi-environment deployment (dev/staging/prod)
# - Cost optimization through tiered models
# - High availability with fallback chains
# - Intent-based model selection
# - Regional deployment support

# Validation metadata
_validation:
  required_env_vars:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - ENVIRONMENT
  
  supported_environments:
    - development
    - staging
    - production
  
  model_tiers:
    tier1: [gpt-3.5-turbo]
    tier2: [claude-3-sonnet]
    tier3: [gpt-4, claude-3-opus]
```

## AWS Bedrock Configurations

### Basic Bedrock Provider Configuration

```yaml
# Basic Bedrock setup with single region
providers:
  bedrock-us-east:
    type: bedrock
    region: us-east-1
    aws_profile: default
    timeout: 60

  bedrock-us-west:
    type: bedrock
    region: us-west-2
    aws_profile: default
    timeout: 60

models:
  claude-3-opus-bedrock:
    provider: bedrock-us-east
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: slow
      capabilities: [chat, analysis, vision]
      deployment: cloud
      foundation_model: claude
      context_length: 200000
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95

  titan-express-bedrock:
    provider: bedrock-us-east
    model_id: amazon.titan-text-express-v1
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: titan
      context_length: 8000
    parameters:
      max_tokens: 2048
      temperature: 0.8
      top_p: 0.9

profiles:
  bedrock-default:
    model: claude-3-opus-bedrock
    parameters:
      temperature: 0.7

  bedrock-fast:
    model: titan-express-bedrock
    parameters:
      temperature: 0.8

aliases:
  bedrock: bedrock-default
  bedrock-quick: bedrock-fast
```

### Multi-Model Bedrock Configuration

```yaml
# Comprehensive Bedrock model configuration with multiple foundation models
providers:
  bedrock-primary:
    type: bedrock
    region: us-east-1
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 90

  bedrock-secondary:
    type: bedrock
    region: us-west-2
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 90

models:
  # Anthropic Claude Models
  claude-3-opus:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: slow
      capabilities: [chat, analysis, vision, coding]
      deployment: cloud
      foundation_model: claude
      context_length: 200000
      use_cases: [complex_reasoning, analysis, creative_writing]
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95

  claude-3-sonnet:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      capabilities: [chat, analysis, coding]
      deployment: cloud
      foundation_model: claude
      context_length: 200000
      use_cases: [balanced_performance, coding, analysis]
    parameters:
      max_tokens: 2048
      temperature: 0.6
      top_p: 0.9

  claude-3-haiku:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: claude
      context_length: 200000
      use_cases: [quick_responses, simple_tasks]
    parameters:
      max_tokens: 1024
      temperature: 0.8
      top_p: 0.95

  # Amazon Titan Models
  titan-text-express:
    provider: bedrock-primary
    model_id: amazon.titan-text-express-v1
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, completion, summarization]
      deployment: cloud
      foundation_model: titan
      context_length: 8000
      use_cases: [content_generation, summarization]
    parameters:
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9

  titan-text-lite:
    provider: bedrock-primary
    model_id: amazon.titan-text-lite-v1
    metadata:
      reasoning: low
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: titan
      context_length: 4000
      use_cases: [simple_chat, basic_completion]
    parameters:
      max_tokens: 1024
      temperature: 0.8
      top_p: 0.95

  # Meta Llama Models
  llama3-70b-instruct:
    provider: bedrock-primary
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: medium
      capabilities: [chat, completion, coding]
      deployment: cloud
      foundation_model: llama
      context_length: 8000
      use_cases: [coding, reasoning, instruction_following]
    parameters:
      max_tokens: 2048
      temperature: 0.6
      top_p: 0.9

  llama3-8b-instruct:
    provider: bedrock-primary
    model_id: meta.llama3-8b-instruct-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: llama
      context_length: 8000
      use_cases: [general_chat, simple_tasks]
    parameters:
      max_tokens: 1024
      temperature: 0.7
      top_p: 0.95

  # Cohere Models
  cohere-command-text:
    provider: bedrock-primary
    model_id: cohere.command-text-v14
    metadata:
      reasoning: medium
      cost: medium
      speed: fast
      capabilities: [chat, completion, summarization]
      deployment: cloud
      foundation_model: cohere
      context_length: 4000
      use_cases: [content_generation, summarization]
    parameters:
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9

  cohere-command-light:
    provider: bedrock-primary
    model_id: cohere.command-light-text-v14
    metadata:
      reasoning: low
      cost: low
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: cohere
      context_length: 4000
      use_cases: [simple_chat, quick_responses]
    parameters:
      max_tokens: 1024
      temperature: 0.8
      top_p: 0.95

  # AI21 Labs Models
  ai21-j2-ultra:
    provider: bedrock-primary
    model_id: ai21.j2-ultra-v1
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, completion, analysis]
      deployment: cloud
      foundation_model: jurassic
      context_length: 8000
      use_cases: [complex_reasoning, analysis]
    parameters:
      max_tokens: 2048
      temperature: 0.6
      top_p: 0.9

  ai21-j2-mid:
    provider: bedrock-primary
    model_id: ai21.j2-mid-v1
    metadata:
      reasoning: medium
      cost: medium
      speed: fast
      capabilities: [chat, completion]
      deployment: cloud
      foundation_model: jurassic
      context_length: 8000
      use_cases: [general_purpose, content_generation]
    parameters:
      max_tokens: 1024
      temperature: 0.7
      top_p: 0.95

profiles:
  # Performance-based profiles
  bedrock-premium:
    model: claude-3-opus
    parameters:
      temperature: 0.3
      max_tokens: 4096

  bedrock-balanced:
    model: claude-3-sonnet
    parameters:
      temperature: 0.5
      max_tokens: 2048

  bedrock-fast:
    model: claude-3-haiku
    parameters:
      temperature: 0.7
      max_tokens: 1024

  # Use case-based profiles
  bedrock-reasoning:
    model: claude-3-opus
    parameters:
      temperature: 0.2
      max_tokens: 3000

  bedrock-coding:
    model: llama3-70b-instruct
    parameters:
      temperature: 0.1
      max_tokens: 2048

  bedrock-content:
    model: titan-text-express
    parameters:
      temperature: 0.8
      max_tokens: 2048

  bedrock-chat:
    model: claude-3-haiku
    parameters:
      temperature: 0.9
      max_tokens: 1024

aliases:
  # Foundation model aliases
  claude: claude-3-sonnet
  claude-best: claude-3-opus
  claude-fast: claude-3-haiku
  titan: titan-text-express
  llama: llama3-70b-instruct
  cohere: cohere-command-text
  ai21: ai21-j2-ultra
  
  # Performance aliases
  bedrock-best: bedrock-premium
  bedrock-quick: bedrock-fast
  bedrock-cheap: claude-3-haiku
  
  # Task aliases
  reasoning: bedrock-reasoning
  coding: bedrock-coding
  content: bedrock-content
  chat: bedrock-chat
```

### Environment-Specific Bedrock Configurations

#### Development Environment

```yaml
# configs/bedrock_development.yaml
default_profile: bedrock-dev

providers:
  bedrock-dev:
    type: bedrock
    region: us-east-1
    aws_profile: development
    timeout: 120

models:
  dev-claude-haiku:
    provider: bedrock-dev
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      environment: development
    parameters:
      max_tokens: 1024
      temperature: 0.9
      top_p: 0.95

  dev-titan-lite:
    provider: bedrock-dev
    model_id: amazon.titan-text-lite-v1
    metadata:
      reasoning: low
      cost: low
      speed: fast
      deployment: cloud
      environment: development
    parameters:
      max_tokens: 512
      temperature: 0.8
      top_p: 0.9

  dev-llama-8b:
    provider: bedrock-dev
    model_id: meta.llama3-8b-instruct-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      environment: development
    parameters:
      max_tokens: 1024
      temperature: 0.8
      top_p: 0.95

providers:
  bedrock-dev:
    type: bedrock
    region: us-east-1
    aws_profile: development
    timeout: 120

profiles:
  bedrock-dev:
    model: dev-claude-haiku
    parameters:
      temperature: 0.9
      max_tokens: 1024

  bedrock-dev-fast:
    model: dev-titan-lite
    parameters:
      temperature: 0.8
      max_tokens: 512

  bedrock-dev-coding:
    model: dev-llama-8b
    parameters:
      temperature: 0.6
      max_tokens: 1024

aliases:
  main: bedrock-dev
  fast: bedrock-dev-fast
  coding: bedrock-dev-coding
  cheap: dev-titan-lite
```

#### Staging Environment

```yaml
# configs/bedrock_staging.yaml
default_profile: bedrock-staging

providers:
  bedrock-staging:
    type: bedrock
    region: us-west-2
    aws_profile: staging
    timeout: 90

models:
  staging-claude-sonnet:
    provider: bedrock-staging
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      deployment: cloud
      environment: staging
    parameters:
      max_tokens: 2048
      temperature: 0.5
      top_p: 0.9

  staging-titan-express:
    provider: bedrock-staging
    model_id: amazon.titan-text-express-v1
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      environment: staging
    parameters:
      max_tokens: 1536
      temperature: 0.6
      top_p: 0.9

  staging-llama-70b:
    provider: bedrock-staging
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: medium
      deployment: cloud
      environment: staging
    parameters:
      max_tokens: 2048
      temperature: 0.4
      top_p: 0.9

profiles:
  bedrock-staging:
    model: staging-claude-sonnet
    parameters:
      temperature: 0.5
      max_tokens: 2048

  bedrock-staging-fast:
    model: staging-titan-express
    parameters:
      temperature: 0.6
      max_tokens: 1536

  bedrock-staging-reasoning:
    model: staging-llama-70b
    parameters:
      temperature: 0.3
      max_tokens: 2048

aliases:
  main: bedrock-staging
  fast: bedrock-staging-fast
  reasoning: bedrock-staging-reasoning
  primary: staging-claude-sonnet
```

#### Production Environment

```yaml
# configs/bedrock_production.yaml
default_profile: bedrock-production
resolution_timeout: 60

providers:
  bedrock-prod-primary:
    type: bedrock
    region: us-east-1
    aws_profile: production
    timeout: 60

  bedrock-prod-backup:
    type: bedrock
    region: us-west-2
    aws_profile: production
    timeout: 60

models:
  prod-claude-opus:
    provider: bedrock-prod-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: slow
      deployment: cloud
      environment: production
      reliability: high
    parameters:
      max_tokens: 4096
      temperature: 0.1
      top_p: 0.95

  prod-claude-sonnet:
    provider: bedrock-prod-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      deployment: cloud
      environment: production
      reliability: high
    parameters:
      max_tokens: 2048
      temperature: 0.2
      top_p: 0.9

  prod-claude-haiku-backup:
    provider: bedrock-prod-backup
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      environment: production
      reliability: medium
    parameters:
      max_tokens: 1024
      temperature: 0.4
      top_p: 0.95

  prod-ai21-ultra:
    provider: bedrock-prod-primary
    model_id: ai21.j2-ultra-v1
    metadata:
      reasoning: high
      cost: high
      speed: medium
      deployment: cloud
      environment: production
      reliability: high
    parameters:
      max_tokens: 2048
      temperature: 0.0
      top_p: 0.9

profiles:
  bedrock-production:
    model: prod-claude-opus
    parameters:
      temperature: 0.1
      max_tokens: 4096

  bedrock-production-fast:
    model: prod-claude-sonnet
    parameters:
      temperature: 0.2
      max_tokens: 2048

  bedrock-production-backup:
    model: prod-claude-haiku-backup
    parameters:
      temperature: 0.4
      max_tokens: 1024

  bedrock-production-reasoning:
    model: prod-ai21-ultra
    parameters:
      temperature: 0.0
      max_tokens: 2048

aliases:
  main: bedrock-production
  primary: prod-claude-opus
  fast: bedrock-production-fast
  backup: bedrock-production-backup
  reasoning: bedrock-production-reasoning
  fallback: prod-claude-haiku-backup
```

### Region-Specific Bedrock Configurations

```yaml
# Multi-region Bedrock deployment configuration
providers:
  bedrock-us-east-1:
    type: bedrock
    region: us-east-1
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 30

  bedrock-us-west-2:
    type: bedrock
    region: us-west-2
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 40

  bedrock-eu-west-1:
    type: bedrock
    region: eu-west-1
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 50

  bedrock-ap-southeast-1:
    type: bedrock
    region: ap-southeast-1
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 60

models:
  # US East (N. Virginia) - Primary region
  claude-us-east:
    provider: bedrock-us-east-1
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      region: us-east-1
      latency: low
      availability: high
      cost: standard
    parameters:
      max_tokens: 2048
      temperature: 0.6

  titan-us-east:
    provider: bedrock-us-east-1
    model_id: amazon.titan-text-express-v1
    metadata:
      region: us-east-1
      latency: low
      availability: high
      cost: low
    parameters:
      max_tokens: 1536
      temperature: 0.7

  # US West (Oregon) - Secondary region
  claude-us-west:
    provider: bedrock-us-west-2
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      region: us-west-2
      latency: medium
      availability: high
      cost: standard
    parameters:
      max_tokens: 2048
      temperature: 0.6

  llama-us-west:
    provider: bedrock-us-west-2
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      region: us-west-2
      latency: medium
      availability: high
      cost: medium
    parameters:
      max_tokens: 2048
      temperature: 0.5

  # Europe (Ireland)
  claude-eu-west:
    provider: bedrock-eu-west-1
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      region: eu-west-1
      latency: high
      availability: medium
      cost: standard
      compliance: gdpr
    parameters:
      max_tokens: 2048
      temperature: 0.6

  titan-eu-west:
    provider: bedrock-eu-west-1
    model_id: amazon.titan-text-express-v1
    metadata:
      region: eu-west-1
      latency: high
      availability: medium
      cost: low
      compliance: gdpr
    parameters:
      max_tokens: 1536
      temperature: 0.7

  # Asia Pacific (Singapore)
  claude-ap-southeast:
    provider: bedrock-ap-southeast-1
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      region: ap-southeast-1
      latency: high
      availability: medium
      cost: standard
    parameters:
      max_tokens: 2048
      temperature: 0.6

  cohere-ap-southeast:
    provider: bedrock-ap-southeast-1
    model_id: cohere.command-text-v14
    metadata:
      region: ap-southeast-1
      latency: high
      availability: medium
      cost: medium
    parameters:
      max_tokens: 1536
      temperature: 0.7

profiles:
  # Regional profiles
  us-east:
    model: claude-us-east
    parameters:
      temperature: 0.6

  us-west:
    model: claude-us-west
    parameters:
      temperature: 0.6

  europe:
    model: claude-eu-west
    parameters:
      temperature: 0.6

  asia-pacific:
    model: claude-ap-southeast
    parameters:
      temperature: 0.6

  # Latency-optimized profiles
  lowest-latency:
    model: claude-us-east
    parameters:
      temperature: 0.6
      max_tokens: 1024

  balanced-latency:
    model: claude-us-west
    parameters:
      temperature: 0.6
      max_tokens: 1536

aliases:
  # Geographic aliases
  us: us-east
  usa: us-east
  america: us-east
  eu: europe
  asia: asia-pacific
  
  # Latency aliases
  fastest: lowest-latency
  balanced: balanced-latency
  
  # Regional fallbacks
  primary: us-east
  secondary: us-west
  tertiary: europe
  
  # Current region based on environment
  local: ${AWS_REGION:-us-east}
  nearest: primary
```

### Bedrock Model Metadata for Intent Matching

```yaml
# Optimized metadata configuration for Model Compass intent matching
models:
  # High reasoning capability models
  claude-3-opus-reasoning:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      # Core capabilities
      reasoning: high
      analysis: high
      creativity: high
      coding: high
      
      # Performance characteristics
      speed: slow
      cost: high
      accuracy: high
      consistency: high
      
      # Technical specifications
      context_length: 200000
      max_output_tokens: 4096
      
      # Use case optimization
      use_cases: [complex_reasoning, deep_analysis, creative_writing, advanced_coding]
      complexity_handling: high
      
      # Intent matching tags
      intent_tags: [reasoning, analysis, creativity, coding, complex]
      
      # Quality metrics
      quality_tier: premium
      reliability: high
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: claude
      model_family: anthropic
    parameters:
      max_tokens: 4096
      temperature: 0.3
      top_p: 0.95

  # Balanced performance models
  claude-3-sonnet-balanced:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      # Core capabilities
      reasoning: high
      analysis: high
      creativity: medium
      coding: high
      
      # Performance characteristics
      speed: fast
      cost: medium
      accuracy: high
      consistency: high
      
      # Technical specifications
      context_length: 200000
      max_output_tokens: 2048
      
      # Use case optimization
      use_cases: [general_reasoning, analysis, coding, content_generation]
      complexity_handling: medium
      
      # Intent matching tags
      intent_tags: [reasoning, analysis, coding, balanced, general]
      
      # Quality metrics
      quality_tier: standard
      reliability: high
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: claude
      model_family: anthropic
    parameters:
      max_tokens: 2048
      temperature: 0.5
      top_p: 0.9

  # Fast response models
  claude-3-haiku-fast:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      # Core capabilities
      reasoning: medium
      analysis: medium
      creativity: medium
      coding: medium
      
      # Performance characteristics
      speed: fast
      cost: low
      accuracy: medium
      consistency: medium
      
      # Technical specifications
      context_length: 200000
      max_output_tokens: 1024
      
      # Use case optimization
      use_cases: [quick_responses, simple_tasks, chat, basic_analysis]
      complexity_handling: low
      
      # Intent matching tags
      intent_tags: [fast, quick, simple, chat, basic]
      
      # Quality metrics
      quality_tier: basic
      reliability: medium
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: claude
      model_family: anthropic
    parameters:
      max_tokens: 1024
      temperature: 0.7
      top_p: 0.95

  # Cost-optimized models
  titan-express-cost-optimized:
    provider: bedrock-primary
    model_id: amazon.titan-text-express-v1
    metadata:
      # Core capabilities
      reasoning: medium
      analysis: medium
      creativity: medium
      coding: low
      
      # Performance characteristics
      speed: fast
      cost: low
      accuracy: medium
      consistency: medium
      
      # Technical specifications
      context_length: 8000
      max_output_tokens: 2048
      
      # Use
 case optimization
      use_cases: [content_generation, summarization, simple_chat]
      complexity_handling: low
      
      # Intent matching tags
      intent_tags: [cost_effective, content, summarization, simple]
      
      # Quality metrics
      quality_tier: basic
      reliability: medium
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: titan
      model_family: amazon
    parameters:
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9

  # Coding-specialized models
  llama3-70b-coding:
    provider: bedrock-primary
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      # Core capabilities
      reasoning: high
      analysis: medium
      creativity: low
      coding: high
      
      # Performance characteristics
      speed: medium
      cost: medium
      accuracy: high
      consistency: high
      
      # Technical specifications
      context_length: 8000
      max_output_tokens: 2048
      
      # Use case optimization
      use_cases: [coding, code_review, technical_documentation, debugging]
      complexity_handling: medium
      
      # Intent matching tags
      intent_tags: [coding, programming, technical, development]
      
      # Quality metrics
      quality_tier: standard
      reliability: high
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: llama
      model_family: meta
    parameters:
      max_tokens: 2048
      temperature: 0.1
      top_p: 0.9

  # Content generation models
  cohere-command-content:
    provider: bedrock-primary
    model_id: cohere.command-text-v14
    metadata:
      # Core capabilities
      reasoning: medium
      analysis: medium
      creativity: high
      coding: low
      
      # Performance characteristics
      speed: fast
      cost: medium
      accuracy: medium
      consistency: medium
      
      # Technical specifications
      context_length: 4000
      max_output_tokens: 2048
      
      # Use case optimization
      use_cases: [content_creation, marketing_copy, creative_writing, summarization]
      complexity_handling: low
      
      # Intent matching tags
      intent_tags: [content, creative, marketing, writing]
      
      # Quality metrics
      quality_tier: standard
      reliability: medium
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: command
      model_family: cohere
    parameters:
      max_tokens: 2048
      temperature: 0.8
      top_p: 0.9

  # Analysis-specialized models
  ai21-j2-ultra-analysis:
    provider: bedrock-primary
    model_id: ai21.j2-ultra-v1
    metadata:
      # Core capabilities
      reasoning: high
      analysis: high
      creativity: medium
      coding: medium
      
      # Performance characteristics
      speed: medium
      cost: high
      accuracy: high
      consistency: high
      
      # Technical specifications
      context_length: 8000
      max_output_tokens: 2048
      
      # Use case optimization
      use_cases: [data_analysis, research, complex_reasoning, business_intelligence]
      complexity_handling: high
      
      # Intent matching tags
      intent_tags: [analysis, research, business, intelligence]
      
      # Quality metrics
      quality_tier: premium
      reliability: high
      
      # Deployment characteristics
      deployment: cloud
      foundation_model: jurassic
      model_family: ai21
    parameters:
      max_tokens: 2048
      temperature: 0.2
      top_p: 0.9

profiles:
  # Intent-based profiles
  high-reasoning:
    model: claude-3-opus-reasoning
    parameters:
      temperature: 0.2
      max_tokens: 4096

  balanced-performance:
    model: claude-3-sonnet-balanced
    parameters:
      temperature: 0.5
      max_tokens: 2048

  fast-response:
    model: claude-3-haiku-fast
    parameters:
      temperature: 0.7
      max_tokens: 1024

  cost-optimized:
    model: titan-express-cost-optimized
    parameters:
      temperature: 0.7
      max_tokens: 1536

  coding-focused:
    model: llama3-70b-coding
    parameters:
      temperature: 0.1
      max_tokens: 2048

  content-creation:
    model: cohere-command-content
    parameters:
      temperature: 0.8
      max_tokens: 2048

  analysis-focused:
    model: ai21-j2-ultra-analysis
    parameters:
      temperature: 0.2
      max_tokens: 2048

aliases:
  # Capability-based aliases
  reasoning: high-reasoning
  balanced: balanced-performance
  fast: fast-response
  cheap: cost-optimized
  coding: coding-focused
  content: content-creation
  analysis: analysis-focused
  
  # Quality-based aliases
  premium: high-reasoning
  standard: balanced-performance
  basic: fast-response
  
  # Use case aliases
  complex: high-reasoning
  general: balanced-performance
  simple: fast-response
  creative: content-creation
  technical: coding-focused
  research: analysis-focused
```

### Bedrock Configuration Best Practices

#### 1. AWS Profile and Credential Management

```yaml
# Secure credential configuration
providers:
  bedrock-prod:
    type: bedrock
    region: ${AWS_REGION:-us-east-1}
    aws_profile: ${AWS_PROFILE:-default}
    timeout: 60
    # Never embed credentials directly in configuration

  bedrock-dev:
    type: bedrock
    region: us-west-2
    aws_profile: development
    timeout: 120
    # Use different profiles for different environments
```

#### 2. Environment Variable Usage

```yaml
# Environment-driven configuration
default_profile: bedrock-${ENVIRONMENT:-development}

providers:
  bedrock-current:
    type: bedrock
    region: ${AWS_REGION:-us-east-1}
    aws_profile: ${AWS_PROFILE:-default}
    timeout: ${BEDROCK_TIMEOUT:-60}

aliases:
  current: bedrock-${ENVIRONMENT:-development}
  region: ${AWS_REGION:-us-east-1}
```

#### 3. Model Selection Optimization

```yaml
# Hierarchical model selection based on requirements
models:
  # Tier 1: Premium models for complex tasks
  bedrock-tier1:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      tier: 1
      cost: high
      quality: premium
      use_for: [complex_reasoning, critical_analysis]

  # Tier 2: Balanced models for general use
  bedrock-tier2:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      tier: 2
      cost: medium
      quality: standard
      use_for: [general_tasks, balanced_performance]

  # Tier 3: Fast models for simple tasks
  bedrock-tier3:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      tier: 3
      cost: low
      quality: basic
      use_for: [simple_tasks, quick_responses]

profiles:
  # Automatic tier selection based on complexity
  auto-select:
    model: bedrock-tier2  # Default to balanced
    parameters:
      temperature: 0.6

aliases:
  # Complexity-based routing
  complex: bedrock-tier1
  normal: bedrock-tier2
  simple: bedrock-tier3
  
  # Cost-based routing
  expensive: bedrock-tier1
  moderate: bedrock-tier2
  cheap: bedrock-tier3
```

#### 4. Fallback and High Availability

```yaml
# Multi-region fallback configuration
providers:
  bedrock-primary:
    type: bedrock
    region: us-east-1
    aws_profile: production
    timeout: 30

  bedrock-fallback:
    type: bedrock
    region: us-west-2
    aws_profile: production
    timeout: 45

models:
  primary-claude:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      priority: 1
      reliability: high

  fallback-claude:
    provider: bedrock-fallback
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      priority: 2
      reliability: high

  emergency-haiku:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      priority: 3
      reliability: medium

profiles:
  high-availability:
    model: primary-claude
    parameters:
      temperature: 0.5

aliases:
  primary: primary-claude
  fallback: fallback-claude
  emergency: emergency-haiku
  main: primary
```

#### 5. Cost Monitoring and Optimization

```yaml
# Cost-aware configuration with usage tracking metadata
models:
  cost-tier-1:
    provider: bedrock-primary
    model_id: amazon.titan-text-lite-v1
    metadata:
      cost_per_1k_tokens: 0.0003
      cost_tier: low
      recommended_for: [simple_tasks, high_volume]
    parameters:
      max_tokens: 1024

  cost-tier-2:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      cost_per_1k_tokens: 0.00025
      cost_tier: low
      recommended_for: [general_tasks, balanced_usage]
    parameters:
      max_tokens: 1024

  cost-tier-3:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      cost_per_1k_tokens: 0.003
      cost_tier: medium
      recommended_for: [complex_tasks, quality_focused]
    parameters:
      max_tokens: 2048

  cost-tier-4:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      cost_per_1k_tokens: 0.015
      cost_tier: high
      recommended_for: [critical_tasks, premium_quality]
    parameters:
      max_tokens: 4096

profiles:
  budget-conscious:
    model: cost-tier-1
    parameters:
      max_tokens: 512  # Further cost optimization

  cost-balanced:
    model: cost-tier-2
    parameters:
      max_tokens: 1024

  quality-focused:
    model: cost-tier-3
    parameters:
      max_tokens: 2048

  premium-quality:
    model: cost-tier-4
    parameters:
      max_tokens: 4096

aliases:
  # Cost-based selection
  cheapest: budget-conscious
  affordable: cost-balanced
  standard: quality-focused
  premium: premium-quality
  
  # Budget aliases
  low-cost: cost-tier-1
  medium-cost: cost-tier-2
  high-cost: cost-tier-3
  premium-cost: cost-tier-4
```

These configuration examples provide a solid foundation for various deployment scenarios and use cases. Choose the patterns that best fit your specific requirements and customize them as needed.