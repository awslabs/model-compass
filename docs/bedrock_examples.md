# Bedrock Configuration Examples

This document provides ready-to-use configuration templates for integrating AWS Bedrock models with Model Compass. These examples demonstrate how to configure Bedrock foundation models with rich metadata for intelligent intent-based resolution.

## Quick Start Templates

### Basic Single Model Configuration

```yaml
# config/bedrock_basic.yaml
providers:
  bedrock-default:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  claude-sonnet:
    provider: bedrock-default
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      capabilities: [chat, analysis, coding]
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95
```

### Multi-Model Foundation Configuration

```yaml
# config/bedrock_multi_model.yaml
providers:
  bedrock-us-east:
    type: bedrock
    region: us-east-1
    aws_profile: default
    
  bedrock-us-west:
    type: bedrock
    region: us-west-2
    aws_profile: default

models:
  # Anthropic Claude Models
  claude-opus:
    provider: bedrock-us-east
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: slow
      capabilities: [chat, analysis, vision, creative_writing]
      context_length: 200000
      foundation_model: claude
      model_size: large
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95

  claude-sonnet:
    provider: bedrock-us-east
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      capabilities: [chat, analysis, coding, vision]
      context_length: 200000
      foundation_model: claude
      model_size: medium
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95

  claude-haiku:
    provider: bedrock-us-east
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: very_fast
      capabilities: [chat, classification, summarization]
      context_length: 200000
      foundation_model: claude
      model_size: small
    parameters:
      max_tokens: 2048
      temperature: 0.5
      top_p: 0.9

  # Amazon Titan Models
  titan-express:
    provider: bedrock-us-east
    model_id: amazon.titan-text-express-v1
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      capabilities: [chat, summarization, text_generation]
      context_length: 8000
      foundation_model: titan
      model_size: medium
    parameters:
      maxTokenCount: 4096
      temperature: 0.7
      topP: 0.9

  titan-lite:
    provider: bedrock-us-east
    model_id: amazon.titan-text-lite-v1
    metadata:
      reasoning: low
      cost: very_low
      speed: very_fast
      capabilities: [chat, classification, simple_tasks]
      context_length: 4000
      foundation_model: titan
      model_size: small
    parameters:
      maxTokenCount: 2048
      temperature: 0.5
      topP: 0.8

  # Meta Llama Models
  llama-70b:
    provider: bedrock-us-east
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, analysis, coding, reasoning]
      context_length: 8000
      foundation_model: llama
      model_size: large
    parameters:
      max_gen_len: 2048
      temperature: 0.6
      top_p: 0.9

  llama-8b:
    provider: bedrock-us-east
    model_id: meta.llama3-8b-instruct-v1:0
    metadata:
      reasoning: medium
      cost: medium
      speed: fast
      capabilities: [chat, coding, simple_analysis]
      context_length: 8000
      foundation_model: llama
      model_size: small
    parameters:
      max_gen_len: 2048
      temperature: 0.6
      top_p: 0.9

  # Cohere Models
  cohere-command:
    provider: bedrock-us-east
    model_id: cohere.command-text-v14
    metadata:
      reasoning: medium
      cost: medium
      speed: fast
      capabilities: [chat, summarization, classification]
      context_length: 4000
      foundation_model: cohere
      model_size: medium
    parameters:
      max_tokens: 4000
      temperature: 0.75
      p: 0.01
      k: 0

  # AI21 Labs Models
  ai21-ultra:
    provider: bedrock-us-east
    model_id: ai21.j2-ultra-v1
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, analysis, creative_writing]
      context_length: 8000
      foundation_model: jurassic
      model_size: large
    parameters:
      maxTokens: 4000
      temperature: 0.7
      topP: 1
```

## Reasoning Capabilities Configuration

Configure models with detailed reasoning metadata for intelligent selection:

```yaml
# config/bedrock_reasoning.yaml
providers:
  bedrock-reasoning:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  # High Reasoning Models
  claude-opus-reasoning:
    provider: bedrock-reasoning
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      logical_reasoning: excellent
      mathematical_reasoning: excellent
      code_reasoning: excellent
      analytical_thinking: excellent
      problem_solving: excellent
      cost: high
      speed: slow
      use_cases: [complex_analysis, research, mathematical_problems, code_review]
    parameters:
      max_tokens: 4096
      temperature: 0.3  # Lower temperature for reasoning tasks

  llama-70b-reasoning:
    provider: bedrock-reasoning
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      reasoning: high
      logical_reasoning: excellent
      mathematical_reasoning: good
      code_reasoning: excellent
      analytical_thinking: good
      problem_solving: good
      cost: high
      speed: medium
      use_cases: [coding, logical_problems, step_by_step_analysis]
    parameters:
      max_gen_len: 2048
      temperature: 0.2

  # Medium Reasoning Models
  claude-sonnet-reasoning:
    provider: bedrock-reasoning
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      logical_reasoning: good
      mathematical_reasoning: good
      code_reasoning: excellent
      analytical_thinking: excellent
      problem_solving: good
      cost: medium
      speed: fast
      use_cases: [general_analysis, coding_assistance, structured_thinking]
    parameters:
      max_tokens: 4096
      temperature: 0.4

  # Fast Reasoning Models
  claude-haiku-reasoning:
    provider: bedrock-reasoning
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      logical_reasoning: fair
      mathematical_reasoning: fair
      code_reasoning: good
      analytical_thinking: good
      problem_solving: fair
      cost: low
      speed: very_fast
      use_cases: [quick_analysis, simple_reasoning, classification]
    parameters:
      max_tokens: 2048
      temperature: 0.5
```

## Cost and Performance Optimization

Configure models with detailed cost and performance metadata:

```yaml
# config/bedrock_cost_performance.yaml
providers:
  bedrock-optimized:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  # High Performance, High Cost
  claude-opus-premium:
    provider: bedrock-optimized
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      cost: high
      cost_tier: premium
      cost_per_1k_input_tokens: 0.015
      cost_per_1k_output_tokens: 0.075
      performance: excellent
      speed: slow
      latency: high
      throughput: low
      quality: excellent
      use_cases: [critical_analysis, high_quality_content, complex_reasoning]
      billing_model: pay_per_token
    parameters:
      max_tokens: 4096
      temperature: 0.7

  # Balanced Cost and Performance
  claude-sonnet-balanced:
    provider: bedrock-optimized
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      cost: medium
      cost_tier: standard
      cost_per_1k_input_tokens: 0.003
      cost_per_1k_output_tokens: 0.015
      performance: excellent
      speed: fast
      latency: medium
      throughput: high
      quality: excellent
      use_cases: [general_purpose, production_workloads, balanced_requirements]
      billing_model: pay_per_token
    parameters:
      max_tokens: 4096
      temperature: 0.7

  # Low Cost, High Speed
  claude-haiku-efficient:
    provider: bedrock-optimized
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      cost: low
      cost_tier: economy
      cost_per_1k_input_tokens: 0.00025
      cost_per_1k_output_tokens: 0.00125
      performance: good
      speed: very_fast
      latency: low
      throughput: very_high
      quality: good
      use_cases: [high_volume_processing, real_time_applications, cost_sensitive]
      billing_model: pay_per_token
    parameters:
      max_tokens: 2048
      temperature: 0.5

  # Ultra Low Cost
  titan-lite-budget:
    provider: bedrock-optimized
    model_id: amazon.titan-text-lite-v1
    metadata:
      cost: very_low
      cost_tier: budget
      cost_per_1k_input_tokens: 0.0001
      cost_per_1k_output_tokens: 0.0002
      performance: fair
      speed: fast
      latency: low
      throughput: high
      quality: fair
      use_cases: [batch_processing, simple_tasks, development_testing]
      billing_model: pay_per_token
    parameters:
      maxTokenCount: 2048
      temperature: 0.5
```

## Capability-Specific Model Configuration

Configure models based on specific capabilities:

```yaml
# config/bedrock_capabilities.yaml
providers:
  bedrock-capabilities:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  # Vision and Multimodal
  claude-sonnet-vision:
    provider: bedrock-capabilities
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      capabilities: [vision, image_analysis, multimodal, text_generation]
      vision_support: true
      image_formats: [jpeg, png, gif, webp]
      max_image_size: 5MB
      use_cases: [image_analysis, document_processing, visual_qa, multimodal_chat]
      reasoning: high
      cost: medium
    parameters:
      max_tokens: 4096
      temperature: 0.7

  # Code Generation and Analysis
  claude-sonnet-coding:
    provider: bedrock-capabilities
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      capabilities: [coding, code_generation, code_review, debugging, refactoring]
      programming_languages: [python, javascript, typescript, java, cpp, rust, go]
      code_quality: excellent
      debugging_ability: excellent
      use_cases: [code_generation, code_review, debugging, refactoring, documentation]
      reasoning: high
      cost: medium
    parameters:
      max_tokens: 4096
      temperature: 0.3  # Lower temperature for code generation

  llama-70b-coding:
    provider: bedrock-capabilities
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      capabilities: [coding, code_generation, algorithm_design, system_design]
      programming_languages: [python, javascript, cpp, java, rust, go, sql]
      code_quality: excellent
      algorithm_design: excellent
      use_cases: [algorithm_implementation, system_design, competitive_programming]
      reasoning: high
      cost: high
    parameters:
      max_gen_len: 2048
      temperature: 0.2

  # Text Analysis and Classification
  claude-haiku-classification:
    provider: bedrock-capabilities
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      capabilities: [classification, sentiment_analysis, entity_extraction, categorization]
      classification_accuracy: excellent
      processing_speed: very_fast
      use_cases: [content_moderation, sentiment_analysis, document_classification, tagging]
      reasoning: medium
      cost: low
    parameters:
      max_tokens: 1024
      temperature: 0.1  # Very low temperature for classification

  # Creative Writing and Content Generation
  claude-opus-creative:
    provider: bedrock-capabilities
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      capabilities: [creative_writing, storytelling, content_generation, copywriting]
      creativity: excellent
      writing_quality: excellent
      style_adaptation: excellent
      use_cases: [creative_writing, marketing_copy, storytelling, content_creation]
      reasoning: high
      cost: high
    parameters:
      max_tokens: 4096
      temperature: 0.9  # Higher temperature for creativity

  # Summarization and Information Extraction
  titan-express-summarization:
    provider: bedrock-capabilities
    model_id: amazon.titan-text-express-v1
    metadata:
      capabilities: [summarization, information_extraction, key_point_extraction]
      summarization_quality: good
      information_extraction: good
      processing_speed: fast
      use_cases: [document_summarization, meeting_notes, article_summaries]
      reasoning: medium
      cost: low
    parameters:
      maxTokenCount: 2048
      temperature: 0.3

  # Conversational AI and Chat
  cohere-command-chat:
    provider: bedrock-capabilities
    model_id: cohere.command-text-v14
    metadata:
      capabilities: [conversational_ai, chat, dialogue, customer_support]
      conversation_quality: good
      context_retention: good
      personality: friendly
      use_cases: [chatbots, customer_support, conversational_interfaces]
      reasoning: medium
      cost: medium
    parameters:
      max_tokens: 2048
      temperature: 0.7
```

## Intent-Based Model Selection Examples

Configure models for specific intent patterns:

```yaml
# config/bedrock_intent_patterns.yaml
providers:
  bedrock-intents:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  # Analysis Intent Models
  claude-opus-analysis:
    provider: bedrock-intents
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      intent_types: [analysis, research, investigation]
      reasoning: high
      analytical_depth: excellent
      research_capability: excellent
      cost: high
      use_cases: [data_analysis, research_papers, complex_investigations]
    parameters:
      max_tokens: 4096
      temperature: 0.4

  claude-sonnet-analysis:
    provider: bedrock-intents
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      intent_types: [analysis, review, evaluation]
      reasoning: high
      analytical_depth: good
      speed: fast
      cost: medium
      use_cases: [code_review, document_analysis, quick_evaluations]
    parameters:
      max_tokens: 4096
      temperature: 0.5

  # Chat Intent Models
  claude-haiku-chat:
    provider: bedrock-intents
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      intent_types: [chat, conversation, support]
      conversational_ability: excellent
      response_speed: very_fast
      cost: low
      personality: helpful
      use_cases: [customer_support, general_chat, quick_questions]
    parameters:
      max_tokens: 2048
      temperature: 0.7

  cohere-command-chat:
    provider: bedrock-intents
    model_id: cohere.command-text-v14
    metadata:
      intent_types: [chat, dialogue, assistance]
      conversational_ability: good
      response_speed: fast
      cost: medium
      personality: professional
      use_cases: [business_chat, professional_assistance, structured_dialogue]
    parameters:
      max_tokens: 2048
      temperature: 0.6

  # Classification Intent Models
  claude-haiku-classification:
    provider: bedrock-intents
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      intent_types: [classification, categorization, tagging]
      classification_accuracy: excellent
      processing_speed: very_fast
      cost: low
      consistency: high
      use_cases: [content_classification, sentiment_analysis, data_categorization]
    parameters:
      max_tokens: 512
      temperature: 0.1

  titan-lite-classification:
    provider: bedrock-intents
    model_id: amazon.titan-text-lite-v1
    metadata:
      intent_types: [classification, simple_categorization]
      classification_accuracy: good
      processing_speed: very_fast
      cost: very_low
      use_cases: [high_volume_classification, simple_tagging, batch_processing]
    parameters:
      maxTokenCount: 256
      temperature: 0.0

  # Generation Intent Models
  claude-opus-generation:
    provider: bedrock-intents
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      intent_types: [generation, creation, writing]
      creativity: excellent
      writing_quality: excellent
      content_depth: excellent
      cost: high
      use_cases: [content_creation, creative_writing, detailed_documentation]
    parameters:
      max_tokens: 4096
      temperature: 0.8

  llama-8b-generation:
    provider: bedrock-intents
    model_id: meta.llama3-8b-instruct-v1:0
    metadata:
      intent_types: [generation, coding, simple_writing]
      code_generation: excellent
      writing_quality: good
      speed: fast
      cost: medium
      use_cases: [code_generation, technical_writing, structured_content]
    parameters:
      max_gen_len: 2048
      temperature: 0.6
```

## Model Hierarchy Examples

Configure model hierarchies for different use cases:

```yaml
# config/bedrock_hierarchies.yaml
providers:
  bedrock-hierarchies:
    type: bedrock
    region: us-east-1
    aws_profile: default

models:
  # Enterprise Tier - High Quality, High Cost
  enterprise-claude-opus:
    provider: bedrock-hierarchies
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      tier: enterprise
      quality: premium
      reasoning: high
      cost: high
      speed: slow
      reliability: excellent
      use_cases: [critical_business_decisions, high_stakes_analysis, premium_content]
      sla: enterprise
    parameters:
      max_tokens: 4096
      temperature: 0.7

  # Production Tier - Balanced Quality and Performance
  production-claude-sonnet:
    provider: bedrock-hierarchies
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      tier: production
      quality: high
      reasoning: high
      cost: medium
      speed: fast
      reliability: excellent
      use_cases: [production_workloads, general_business_use, customer_facing]
      sla: production
    parameters:
      max_tokens: 4096
      temperature: 0.7

  production-llama-70b:
    provider: bedrock-hierarchies
    model_id: meta.llama3-70b-instruct-v1:0
    metadata:
      tier: production
      quality: high
      reasoning: high
      cost: high
      speed: medium
      reliability: good
      use_cases: [code_generation, technical_analysis, structured_reasoning]
      sla: production
    parameters:
      max_gen_len: 2048
      temperature: 0.6

  # Standard Tier - Good Quality, Moderate Cost
  standard-claude-haiku:
    provider: bedrock-hierarchies
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      tier: standard
      quality: good
      reasoning: medium
      cost: low
      speed: very_fast
      reliability: good
      use_cases: [general_purpose, internal_tools, development]
      sla: standard
    parameters:
      max_tokens: 2048
      temperature: 0.6

  standard-titan-express:
    provider: bedrock-hierarchies
    model_id: amazon.titan-text-express-v1
    metadata:
      tier: standard
      quality: good
      reasoning: medium
      cost: low
      speed: fast
      reliability: good
      use_cases: [content_generation, summarization, general_tasks]
      sla: standard
    parameters:
      maxTokenCount: 2048
      temperature: 0.7

  # Development Tier - Cost Optimized
  development-titan-lite:
    provider: bedrock-hierarchies
    model_id: amazon.titan-text-lite-v1
    metadata:
      tier: development
      quality: fair
      reasoning: low
      cost: very_low
      speed: fast
      reliability: fair
      use_cases: [testing, development, prototyping, batch_processing]
      sla: development
    parameters:
      maxTokenCount: 1024
      temperature: 0.5

  development-llama-8b:
    provider: bedrock-hierarchies
    model_id: meta.llama3-8b-instruct-v1:0
    metadata:
      tier: development
      quality: good
      reasoning: medium
      cost: medium
      speed: fast
      reliability: fair
      use_cases: [development, testing, code_experiments, learning]
      sla: development
    parameters:
      max_gen_len: 1024
      temperature: 0.6
```

## Usage Examples with Intent Resolution

Here's how to use these configurations with Model Compass intent resolution:

```python
import model_compass as mc

# Configure Model Compass with your chosen configuration
mc.configure(config_path="config/bedrock_capabilities.yaml")

# Example 1: Resolve model for high-reasoning analysis
analysis_config = mc.model("urn:llm:intent:analysis?reasoning=high&cost=medium")
# Resolves to: claude-sonnet-analysis

# Example 2: Resolve model for fast classification
classification_config = mc.model("urn:llm:intent:classification?speed=very_fast&cost=low")
# Resolves to: claude-haiku-classification

# Example 3: Resolve model for creative writing
creative_config = mc.model("urn:llm:intent:generation?creativity=excellent&quality=excellent")
# Resolves to: claude-opus-creative

# Example 4: Resolve model for code generation
coding_config = mc.model("urn:llm:intent:coding?programming_languages=python&quality=excellent")
# Resolves to: claude-sonnet-coding or llama-70b-coding

# Example 5: Resolve model for vision tasks
vision_config = mc.model("urn:llm:intent:analysis?capabilities=vision&image_analysis=true")
# Resolves to: claude-sonnet-vision

# Example 6: Resolve model by tier
enterprise_config = mc.model("urn:llm:intent:analysis?tier=enterprise&quality=premium")
# Resolves to: enterprise-claude-opus

# Example 7: Resolve cost-optimized model
budget_config = mc.model("urn:llm:intent:chat?cost=very_low&speed=fast")
# Resolves to: titan-lite-budget

# Example 8: Resolve model with fallback
try:
    config = mc.model("urn:llm:intent:analysis?reasoning=high&cost=low")
except mc.ResolutionError:
    # Fallback to medium cost if no low-cost high-reasoning model available
    config = mc.model("urn:llm:intent:analysis?reasoning=high&cost=medium")
```

## Configuration Validation

Validate your configurations before deployment:

```python
import model_compass as mc

def validate_bedrock_configuration(config_path):
    """Validate Bedrock configuration for common intent patterns."""
    try:
        mc.configure(config_path=config_path)
        
        # Test common intent patterns
        test_intents = [
            "urn:llm:intent:chat?cost=low",
            "urn:llm:intent:analysis?reasoning=high",
            "urn:llm:intent:classification?speed=fast",
            "urn:llm:intent:generation?creativity=high",
            "urn:llm:intent:coding?quality=good"
        ]
        
        results = {}
        for intent in test_intents:
            try:
                config = mc.model(intent)
                results[intent] = {
                    'status': 'success',
                    'model_id': config.model_id,
                    'cost': config.metadata.get('cost'),
                    'reasoning': config.metadata.get('reasoning'),
                    'speed': config.metadata.get('speed')
                }
            except mc.ResolutionError as e:
                results[intent] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        return {'configuration_error': str(e)}

# Validate configuration
validation_results = validate_bedrock_configuration("config/bedrock_capabilities.yaml")
for intent, result in validation_results.items():
    if result['status'] == 'success':
        print(f"✅ {intent} -> {result['model_id']}")
    else:
        print(f"❌ {intent} -> {result['error']}")
```

These configuration examples provide comprehensive templates for integrating AWS Bedrock models with Model Compass, enabling intelligent model selection based on your specific requirements for reasoning, cost, performance, and capabilities.