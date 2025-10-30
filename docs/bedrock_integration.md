# AWS Bedrock Integration Guide

## Overview

This guide demonstrates how to configure AWS Bedrock models with Model Compass for intelligent model selection and configuration management. AWS Bedrock provides access to foundation models from leading AI companies through a unified managed service.

Model Compass focuses on configuration resolution and intent-based model selection. It resolves Bedrock model configurations that you then use with the AWS SDK, LangChain, DSPy, or other frameworks. Model Compass does not handle the actual API calls or network configuration - it provides the resolved configuration parameters needed by your chosen framework.

## Prerequisites

Before configuring Bedrock models in Model Compass:

- AWS account with Bedrock access enabled
- AWS CLI configured with valid credentials or AWS profiles
- Model Compass installed and configured
- Python 3.8+ environment

Note: Model Compass only handles configuration resolution. You'll need the AWS SDK or a framework like LangChain to make actual Bedrock API calls.

## Basic Setup

### 1. AWS Bedrock Access Configuration

First, ensure your AWS account has access to Bedrock and the specific foundation models you want to use:

```bash
# Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Check available models in your region
aws bedrock list-foundation-models --region us-east-1 --query 'modelSummaries[*].[modelId,modelName]' --output table
```

### 2. Model Compass Configuration

Configure Bedrock models in Model Compass for intelligent resolution:

```yaml
# config/bedrock_basic.yaml
providers:
  bedrock-primary:
    type: bedrock
    region: us-east-1
    aws_profile: default
    
models:
  claude-3-opus:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: medium
      capabilities: [chat, analysis, vision]
      context_length: 200000
    parameters:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.95
```

Model Compass uses this configuration to resolve the appropriate model based on your intent, then provides the resolved parameters for use with your chosen framework.

### 3. Basic Usage Pattern

```python
import model_compass as mc

# Initialize Model Compass with Bedrock configuration
mc.configure(config_path="config/bedrock_basic.yaml")

# Model Compass resolves configuration based on intent
config = mc.model("urn:llm:intent:analysis?reasoning=high")

# Model Compass provides: config.model_id, config.region, config.parameters
# You use these with your chosen framework:

# Option 1: Direct AWS SDK usage
import boto3
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=config.region,
    profile_name=config.aws_profile
)

# Option 2: LangChain integration
from langchain_aws import BedrockLLM
llm = BedrockLLM(
    model_id=config.model_id,
    region_name=config.region,
    model_kwargs=config.parameters
)

# Option 3: Any other framework that accepts Bedrock parameters
```

## Configuration Management Best Practices

### Secure Configuration Patterns

Model Compass handles configuration resolution only. Security concerns like IAM permissions, VPC configuration, and data encryption are handled by the AWS SDK and your application infrastructure.

#### Environment-Based Configuration

Use environment variables and AWS profiles instead of embedding credentials:

```yaml
# config/bedrock_secure.yaml
providers:
  bedrock-secure:
    type: bedrock
    region: ${AWS_REGION:-us-east-1}
    aws_profile: ${AWS_PROFILE:-default}
    
models:
  claude-secure:
    provider: bedrock-secure
    model_id: ${BEDROCK_MODEL_ID:-anthropic.claude-3-sonnet-20240229-v1:0}
    metadata:
      security_level: high
```

#### Role-Based Configuration Management

Configure different AWS profiles for different environments to maintain separation:

```yaml
# config/bedrock_rbac.yaml
providers:
  bedrock-dev:
    type: bedrock
    region: us-east-1
    aws_profile: development
    
  bedrock-staging:
    type: bedrock
    region: us-east-1
    aws_profile: staging
    
  bedrock-prod:
    type: bedrock
    region: us-east-1
    aws_profile: production

models:
  claude-dev:
    provider: bedrock-dev
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      environment: development
      cost: low
      
  claude-prod:
    provider: bedrock-prod
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      environment: production
      cost: high
```

#### Configuration File Security

Protect Model Compass configuration files that reference AWS resources:

- Store configuration files outside web-accessible directories
- Use appropriate file permissions (600 or 640)
- Version control configurations without sensitive data
- Use configuration templates with environment variable substitution

#### Compliance Metadata Configuration

Use Model Compass metadata to tag models with compliance requirements for proper resolution:

```yaml
# config/bedrock_compliance.yaml
providers:
  bedrock-compliant:
    type: bedrock
    region: us-gov-east-1  # Use GovCloud for compliance
    aws_profile: compliance
    
models:
  claude-hipaa:
    provider: bedrock-compliant
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      compliance: [hipaa, sox, pci]
      data_residency: us
      security_level: high
    parameters:
      max_tokens: 2048
      temperature: 0.1
```

#### Resolution Audit Logging

Track which configurations Model Compass resolves for audit purposes:

```python
import model_compass as mc
import logging
import json
from datetime import datetime

# Configure audit logging for Model Compass resolutions
audit_logger = logging.getLogger('model_compass_audit')
audit_handler = logging.FileHandler('/var/log/model_compass_audit.log')
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

def audit_model_resolution(intent, config):
    """Log Model Compass resolution for audit purposes."""
    audit_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'intent': intent,
        'resolved_model': config.model_id,
        'region': config.region,
        'aws_profile': getattr(config, 'aws_profile', 'default'),
        'compliance_tags': config.metadata.get('compliance', [])
    }
    audit_logger.info(json.dumps(audit_data))

# Usage with audit logging
config = mc.model("urn:llm:intent:analysis?compliance=hipaa")
audit_model_resolution("urn:llm:intent:analysis?compliance=hipaa", config)

# The resolved config is then used by your application with the AWS SDK
# Model Compass's role ends at providing the configuration
```

## Cost Optimization Strategies

### Intent-Based Cost Management

Model Compass enables sophisticated cost optimization through intent-based model selection:

```yaml
# config/bedrock_cost_optimized.yaml
models:
  # High-cost, high-capability model
  claude-opus-premium:
    provider: bedrock-primary
    model_id: anthropic.claude-3-opus-20240229-v1:0
    metadata:
      reasoning: high
      cost: high
      speed: slow
      use_cases: [complex_analysis, research, creative_writing]
      cost_per_1k_tokens: 0.015
      
  # Medium-cost, balanced model  
  claude-sonnet-balanced:
    provider: bedrock-primary
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      use_cases: [general_chat, analysis, coding]
      cost_per_1k_tokens: 0.003
      
  # Low-cost, fast model
  claude-haiku-efficient:
    provider: bedrock-primary
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      use_cases: [simple_chat, classification, summarization]
      cost_per_1k_tokens: 0.00025
```

### Cost-Aware Model Selection

```python
import model_compass as mc

def get_cost_optimized_model(intent, max_cost_tier="medium"):
    """Get the most capable model within cost constraints."""
    try:
        # Try to get a model within cost constraints
        cost_constrained_intent = f"{intent}&cost={max_cost_tier}"
        return mc.model(cost_constrained_intent)
    except mc.ResolutionError:
        # Fallback to lowest cost option
        fallback_intent = f"{intent}&cost=low"
        return mc.model(fallback_intent)

# Usage examples
config = get_cost_optimized_model(
    "urn:llm:intent:chat?reasoning=medium", 
    max_cost_tier="low"
)

# For batch processing, always use cost-efficient models
batch_config = mc.model("urn:llm:intent:classification?cost=low&speed=fast")
```

### Usage Tracking and Budget Management

```python
import model_compass as mc
from datetime import datetime, timedelta
import json

class BedrockCostTracker:
    def __init__(self):
        self.usage_log = []
        
    def track_usage(self, intent, config, input_tokens, output_tokens):
        """Track model usage for cost analysis."""
        cost_per_1k = config.metadata.get('cost_per_1k_tokens', 0.001)
        total_tokens = input_tokens + output_tokens
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        usage_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'intent': intent,
            'model_id': config.model_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'estimated_cost': estimated_cost
        }
        
        self.usage_log.append(usage_record)
        
    def get_daily_cost_summary(self):
        """Get cost summary for the current day."""
        today = datetime.utcnow().date()
        daily_usage = [
            record for record in self.usage_log 
            if datetime.fromisoformat(record['timestamp']).date() == today
        ]
        
        total_cost = sum(record['estimated_cost'] for record in daily_usage)
        total_tokens = sum(record['total_tokens'] for record in daily_usage)
        
        return {
            'date': today.isoformat(),
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'request_count': len(daily_usage)
        }

# Usage
cost_tracker = BedrockCostTracker()

config = mc.model("urn:llm:intent:analysis?cost=medium")
# ... make Bedrock API call ...
cost_tracker.track_usage(
    intent="urn:llm:intent:analysis?cost=medium",
    config=config,
    input_tokens=150,
    output_tokens=300
)

# Get daily summary
summary = cost_tracker.get_daily_cost_summary()
print(f"Today's estimated cost: ${summary['total_cost']:.4f}")
```

### Multi-Region Cost Optimization

```yaml
# config/bedrock_multi_region_cost.yaml
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
  claude-us-east-cheap:
    provider: bedrock-us-east
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      cost: low
      region: us-east
      latency: medium
      
  claude-us-west-cheap:
    provider: bedrock-us-west
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    metadata:
      cost: low
      region: us-west
      latency: low  # Assuming closer to users
```

## Monitoring and Configuration Tracking

### Model Resolution Monitoring

```python
import model_compass as mc
import logging
from datetime import datetime
import json

class BedrockMonitor:
    def __init__(self):
        self.resolution_stats = {}
        self.error_log = []
        
        # Configure logging
        self.logger = logging.getLogger('bedrock_monitor')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def track_resolution(self, intent, config=None, error=None):
        """Track model resolution attempts and outcomes."""
        timestamp = datetime.utcnow().isoformat()
        
        if config:
            # Successful resolution
            model_id = config.model_id
            if model_id not in self.resolution_stats:
                self.resolution_stats[model_id] = {
                    'success_count': 0,
                    'error_count': 0,
                    'last_used': None
                }
            
            self.resolution_stats[model_id]['success_count'] += 1
            self.resolution_stats[model_id]['last_used'] = timestamp
            
            self.logger.info(f"Resolved intent '{intent}' to model '{model_id}'")
            
        else:
            # Failed resolution
            error_record = {
                'timestamp': timestamp,
                'intent': intent,
                'error': str(error)
            }
            self.error_log.append(error_record)
            
            self.logger.error(f"Failed to resolve intent '{intent}': {error}")
            
    def get_usage_statistics(self):
        """Get model usage statistics."""
        return {
            'model_usage': self.resolution_stats,
            'total_resolutions': sum(
                stats['success_count'] for stats in self.resolution_stats.values()
            ),
            'total_errors': len(self.error_log),
            'error_rate': len(self.error_log) / max(1, sum(
                stats['success_count'] + stats['error_count'] 
                for stats in self.resolution_stats.values()
            ))
        }

# Usage
monitor = BedrockMonitor()

def monitored_model_resolution(intent):
    """Resolve model with monitoring."""
    try:
        config = mc.model(intent)
        monitor.track_resolution(intent, config=config)
        return config
    except Exception as e:
        monitor.track_resolution(intent, error=e)
        raise

# Example usage
config = monitored_model_resolution("urn:llm:intent:analysis?reasoning=high")
```

### Configuration Validation and Health Checks

```python
import model_compass as mc
import boto3
from botocore.exceptions import ClientError

class BedrockHealthChecker:
    def __init__(self):
        self.health_status = {}
        
    def check_bedrock_access(self, region, profile=None):
        """Check if Bedrock is accessible in the specified region."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            bedrock_client = session.client('bedrock', region_name=region)
            
            # Test basic access
            response = bedrock_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            
            return {
                'status': 'healthy',
                'region': region,
                'available_models': available_models,
                'model_count': len(available_models)
            }
            
        except ClientError as e:
            return {
                'status': 'unhealthy',
                'region': region,
                'error': str(e)
            }
            
    def validate_model_compass_config(self, config_path):
        """Validate Model Compass configuration for Bedrock."""
        try:
            mc.configure(config_path=config_path)
            
            # Test resolution of common intents
            test_intents = [
                "urn:llm:intent:chat",
                "urn:llm:intent:analysis",
                "urn:llm:intent:classification"
            ]
            
            validation_results = {}
            for intent in test_intents:
                try:
                    config = mc.model(intent)
                    validation_results[intent] = {
                        'status': 'valid',
                        'resolved_model': config.model_id,
                        'region': config.region
                    }
                except Exception as e:
                    validation_results[intent] = {
                        'status': 'invalid',
                        'error': str(e)
                    }
                    
            return validation_results
            
        except Exception as e:
            return {'status': 'configuration_error', 'error': str(e)}
            
    def run_health_check(self, config_path):
        """Run comprehensive health check."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'config_validation': self.validate_model_compass_config(config_path),
            'bedrock_access': {}
        }
        
        # Check Bedrock access for common regions
        regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for region in regions:
            results['bedrock_access'][region] = self.check_bedrock_access(region)
            
        return results

# Usage
health_checker = BedrockHealthChecker()
health_report = health_checker.run_health_check('config/bedrock_config.yaml')

# Log health status
if all(
    result['status'] == 'valid' 
    for result in health_report['config_validation'].values()
    if isinstance(result, dict)
):
    print("✅ Model Compass configuration is valid")
else:
    print("❌ Model Compass configuration has issues")
    
for region, status in health_report['bedrock_access'].items():
    if status['status'] == 'healthy':
        print(f"✅ Bedrock access in {region}: {status['model_count']} models available")
    else:
        print(f"❌ Bedrock access in {region}: {status['error']}")
```

### Performance Monitoring

```python
import time
import statistics
from collections import defaultdict

class BedrockPerformanceMonitor:
    def __init__(self):
        self.resolution_times = defaultdict(list)
        self.api_response_times = defaultdict(list)
        
    def time_resolution(self, intent):
        """Time how long model resolution takes."""
        start_time = time.time()
        try:
            config = mc.model(intent)
            resolution_time = time.time() - start_time
            self.resolution_times[config.model_id].append(resolution_time)
            return config
        except Exception as e:
            resolution_time = time.time() - start_time
            self.resolution_times['error'].append(resolution_time)
            raise
            
    def time_api_call(self, model_id, api_call_func):
        """Time Bedrock API calls."""
        start_time = time.time()
        try:
            result = api_call_func()
            api_time = time.time() - start_time
            self.api_response_times[model_id].append(api_time)
            return result
        except Exception as e:
            api_time = time.time() - start_time
            self.api_response_times[f"{model_id}_error"].append(api_time)
            raise
            
    def get_performance_stats(self):
        """Get performance statistics."""
        stats = {
            'resolution_performance': {},
            'api_performance': {}
        }
        
        # Resolution performance
        for model_id, times in self.resolution_times.items():
            if times:
                stats['resolution_performance'][model_id] = {
                    'avg_time': statistics.mean(times),
                    'median_time': statistics.median(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'sample_count': len(times)
                }
                
        # API performance
        for model_id, times in self.api_response_times.items():
            if times:
                stats['api_performance'][model_id] = {
                    'avg_time': statistics.mean(times),
                    'median_time': statistics.median(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'sample_count': len(times)
                }
                
        return stats

# Usage
perf_monitor = BedrockPerformanceMonitor()

# Monitor resolution performance
config = perf_monitor.time_resolution("urn:llm:intent:analysis?reasoning=high")

# Monitor API performance
def make_bedrock_call():
    # Your Bedrock API call here
    return bedrock_client.invoke_model(...)

response = perf_monitor.time_api_call(config.model_id, make_bedrock_call)

# Get performance report
perf_stats = perf_monitor.get_performance_stats()
```

## Troubleshooting Common Issues

### Configuration Resolution Issues

```python
import model_compass as mc

def diagnose_resolution_issue(intent):
    """Diagnose why model resolution might be failing."""
    try:
        config = mc.model(intent)
        return {'status': 'success', 'config': config}
    except mc.ResolutionError as e:
        # Analyze the error and provide suggestions
        error_msg = str(e).lower()
        
        suggestions = []
        if 'no models found' in error_msg:
            suggestions.append("Check that models are configured with matching metadata")
            suggestions.append("Verify intent parameters match model metadata")
        elif 'provider not found' in error_msg:
            suggestions.append("Verify provider configuration exists")
            suggestions.append("Check AWS credentials and region settings")
        elif 'authentication' in error_msg:
            suggestions.append("Verify AWS credentials are configured")
            suggestions.append("Check IAM permissions for Bedrock access")
            
        return {
            'status': 'error',
            'error': str(e),
            'suggestions': suggestions
        }

# Usage
diagnosis = diagnose_resolution_issue("urn:llm:intent:analysis?reasoning=high")
if diagnosis['status'] == 'error':
    print(f"Resolution failed: {diagnosis['error']}")
    for suggestion in diagnosis['suggestions']:
        print(f"  - {suggestion}")
```

### Configuration Reference Issues

```python
import model_compass as mc

def verify_configuration_references(config_path):
    """Verify that Model Compass configuration references are valid."""
    try:
        # Test configuration loading
        mc.configure(config_path=config_path)
        
        # Test common resolution patterns
        test_cases = [
            "urn:llm:intent:chat",
            "urn:llm:intent:analysis", 
            "urn:llm:intent:classification"
        ]
        
        results = {}
        for intent in test_cases:
            try:
                config = mc.model(intent)
                results[intent] = {
                    'status': 'success',
                    'model_id': config.model_id,
                    'region': config.region,
                    'aws_profile': getattr(config, 'aws_profile', 'default')
                }
            except mc.ResolutionError as e:
                results[intent] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            'status': 'success',
            'config_path': config_path,
            'test_results': results
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'config_path': config_path,
            'error': str(e),
            'suggestions': [
                'Check configuration file syntax',
                'Verify all referenced AWS profiles exist',
                'Ensure environment variables are set',
                'Validate model metadata structure'
            ]
        }

# Usage
config_check = verify_configuration_references('config/bedrock_config.yaml')
if config_check['status'] == 'success':
    print("✅ Model Compass configuration is valid")
    for intent, result in config_check['test_results'].items():
        if result['status'] == 'success':
            print(f"✅ {intent} resolves to {result['model_id']}")
        else:
            print(f"❌ {intent} failed: {result['error']}")
else:
    print(f"❌ Configuration issue: {config_check['error']}")
    for suggestion in config_check.get('suggestions', []):
        print(f"  - {suggestion}")
```

## Best Practices Summary

### Model Compass Configuration
- Use environment variables for dynamic configuration values
- Version control your Model Compass configuration files
- Use separate configurations for different environments (dev/staging/prod)
- Implement configuration validation in your deployment pipeline

### Intent-Based Model Selection
- Configure models with rich metadata for accurate intent matching
- Use cost and performance metadata to enable intelligent routing
- Create fallback configurations for high availability
- Test intent resolution patterns before deployment

### AWS Integration
- Reference AWS profiles instead of embedding credentials
- Use environment variables for region and profile selection
- Validate that referenced AWS profiles exist and have proper permissions
- Keep Model Compass configurations separate from AWS infrastructure concerns

### Configuration Monitoring
- Track which models are resolved for different intents
- Monitor configuration resolution performance
- Validate configurations during deployment
- Set up alerts for resolution failures