# Deployment Guide

This guide covers best practices for deploying Model Compass in different environments and scenarios.

## Environment Setup

### Development Environment

#### Local Development Setup

```bash
# Install Model Compass from source
pip install git+https://github.com/awslabs/model-compass.git

# Create development configuration
mkdir -p configs
cat > configs/development.yaml << EOF
default_profile: development

models:
  local-llama:
    provider: ollama
    model: llama3
    metadata:
      deployment: local
      cost: low
      speed: medium
    parameters:
      temperature: 0.9

  dev-gpt:
    provider: openai
    model: gpt-3.5-turbo
    metadata:
      deployment: cloud
      cost: low
      speed: fast
    parameters:
      temperature: 0.8

providers:
  ollama:
    base_url: http://localhost:11434
    timeout: 120

  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer \${OPENAI_API_KEY}"

profiles:
  development:
    model: local-llama
    parameters:
      temperature: 0.9

  cloud-dev:
    model: dev-gpt

aliases:
  main: development
  fallback: cloud-dev
EOF
```

#### Development Environment Variables

```bash
# .env.development
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export ENVIRONMENT="development"
export MODEL_COMPASS_CONFIG="configs/development.yaml"
```

#### Development Usage

```python
# dev_setup.py
import os
import model_compass as mc

# Load environment-specific configuration
config_path = os.getenv("MODEL_COMPASS_CONFIG", "configs/development.yaml")
mc.configure(config_path=config_path)

# Enable performance optimizations for development
mc.enable_performance_optimizations()

# Test configuration
model = mc.model()
print(f"Development model: {model.provider}/{model.name}")

# Test intent resolution
fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")
print(f"Fast model: {fast_model.provider}/{fast_model.name}")
```

### Staging Environment

#### Staging Configuration

```yaml
# configs/staging.yaml
default_profile: staging
resolution_timeout: 45

models:
  staging-claude:
    provider: anthropic
    model: claude-3-sonnet-20240229
    metadata:
      reasoning: high
      cost: medium
      speed: fast
      deployment: cloud
      environment: staging
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
      environment: staging
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
      Content-Type: "application/json"

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
  backup: staging-gpt
```

#### Staging Deployment Script

```bash
#!/bin/bash
# deploy_staging.sh

set -e

echo "Deploying to staging environment..."

# Set environment variables
export ENVIRONMENT="staging"
export MODEL_COMPASS_CONFIG="configs/staging.yaml"

# Validate configuration
python -c "
import model_compass as mc
mc.configure(config_path='configs/staging.yaml')
print('✓ Configuration valid')

# Test model resolution
model = mc.model()
print(f'✓ Default model: {model.provider}/{model.name}')

# Test intent resolution
fast_model = mc.model('urn:llm:intent:chat?speed=fast')
print(f'✓ Fast model: {fast_model.provider}/{fast_model.name}')

print('✓ Staging deployment validation complete')
"

echo "Staging deployment complete!"
```

### Production Environment

#### Production Configuration

```yaml
# configs/production.yaml
default_profile: production
resolution_timeout: 60

models:
  prod-gpt4:
    provider: openai-primary
    model: gpt-4
    metadata:
      reasoning: high
      cost: high
      speed: medium
      deployment: cloud
      environment: production
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
      environment: production
      reliability: high
    parameters:
      temperature: 0.0
      max_tokens: 4096

  prod-fallback:
    provider: openai-backup
    model: gpt-3.5-turbo
    metadata:
      reasoning: medium
      cost: low
      speed: fast
      deployment: cloud
      environment: production
      reliability: medium
    parameters:
      temperature: 0.3
      max_tokens: 1024

providers:
  openai-primary:
    base_url: https://api.openai.com/v1
    timeout: 60
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY_PRIMARY}"
      Content-Type: "application/json"

  openai-backup:
    base_url: https://api.openai.com/v1
    timeout: 60
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY_BACKUP}"
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

#### Production Deployment

```python
# production_deploy.py
import os
import logging
import model_compass as mc
from model_compass.exceptions import ConfigurationError, ResolutionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_production_config():
    """Validate production configuration before deployment."""
    try:
        # Load production configuration
        mc.configure(config_path="configs/production.yaml")
        logger.info("✓ Configuration loaded successfully")
        
        # Enable performance optimizations
        mc.enable_performance_optimizations()
        logger.info("✓ Performance optimizations enabled")
        
        # Test primary model
        primary_model = mc.model("main")
        logger.info(f"✓ Primary model: {primary_model.provider}/{primary_model.name}")
        
        # Test fallback model
        fallback_model = mc.model("fallback")
        logger.info(f"✓ Fallback model: {fallback_model.provider}/{fallback_model.name}")
        
        # Test intent resolution
        reasoning_model = mc.model("urn:llm:intent:analysis?reasoning=high")
        logger.info(f"✓ Reasoning model: {reasoning_model.provider}/{reasoning_model.name}")
        
        # Test performance stats
        stats = mc.get_performance_stats()
        logger.info(f"✓ Performance stats available: {len(stats)} cache types")
        
        logger.info("✓ Production configuration validation complete")
        return True
        
    except (ConfigurationError, ResolutionError) as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during validation: {e}")
        return False

def deploy_production():
    """Deploy to production environment."""
    logger.info("Starting production deployment...")
    
    # Validate required environment variables
    required_vars = [
        "OPENAI_API_KEY_PRIMARY",
        "OPENAI_API_KEY_BACKUP", 
        "ANTHROPIC_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"✗ Missing required environment variables: {missing_vars}")
        return False
    
    # Validate configuration
    if not validate_production_config():
        logger.error("✗ Configuration validation failed")
        return False
    
    logger.info("✓ Production deployment successful")
    return True

if __name__ == "__main__":
    success = deploy_production()
    exit(0 if success else 1)
```

## Container Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy configuration files
COPY configs/ configs/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_COMPASS_CONFIG=configs/production.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import model_compass as mc; mc.configure(config_path='configs/production.yaml'); print('healthy')"

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY_PRIMARY=${OPENAI_API_KEY_PRIMARY}
      - OPENAI_API_KEY_BACKUP=${OPENAI_API_KEY_BACKUP}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - MODEL_COMPASS_CONFIG=configs/production.yaml
    volumes:
      - ./configs:/app/configs:ro
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import model_compass as mc; mc.configure(config_path='configs/production.yaml')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Kubernetes Deployment

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-compass-config
  namespace: default
data:
  production.yaml: |
    default_profile: production
    resolution_timeout: 60
    
    models:
      prod-gpt4:
        provider: openai
        model: gpt-4
        metadata:
          reasoning: high
          cost: high
          reliability: high
        parameters:
          temperature: 0.1
          max_tokens: 2048
    
    providers:
      openai:
        base_url: https://api.openai.com/v1
        timeout: 60
        headers:
          Authorization: "Bearer ${OPENAI_API_KEY}"
    
    profiles:
      production:
        model: prod-gpt4
        parameters:
          temperature: 0.1
    
    aliases:
      main: production
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-compass-secrets
  namespace: default
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-compass-app
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-compass-app
  template:
    metadata:
      labels:
        app: model-compass-app
    spec:
      containers:
      - name: app
        image: model-compass-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: model-compass-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: model-compass-secrets
              key: anthropic-api-key
        - name: MODEL_COMPASS_CONFIG
          value: "/app/configs/production.yaml"
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs
          readOnly: true
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import model_compass as mc; mc.configure(config_path='/app/configs/production.yaml')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import model_compass as mc; mc.configure(config_path='/app/configs/production.yaml'); mc.model('main')"
          initialDelaySeconds: 10
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: config-volume
        configMap:
          name: model-compass-config
```

## Monitoring and Observability

### Health Checks

```python
# health_check.py
import model_compass as mc
from model_compass.exceptions import ConfigurationError, ResolutionError

def health_check():
    """Comprehensive health check for Model Compass."""
    checks = {
        "configuration": False,
        "primary_model": False,
        "fallback_model": False,
        "intent_resolution": False,
        "performance_stats": False
    }
    
    try:
        # Check configuration loading
        mc.configure(config_path="configs/production.yaml")
        checks["configuration"] = True
        
        # Check primary model resolution
        primary_model = mc.model("main")
        _ = primary_model.provider  # Trigger resolution
        checks["primary_model"] = True
        
        # Check fallback model resolution
        fallback_model = mc.model("fallback")
        _ = fallback_model.provider  # Trigger resolution
        checks["fallback_model"] = True
        
        # Check intent resolution
        intent_model = mc.model("urn:llm:intent:chat?reasoning=high")
        _ = intent_model.provider  # Trigger resolution
        checks["intent_resolution"] = True
        
        # Check performance stats
        stats = mc.get_performance_stats()
        if len(stats) > 0:
            checks["performance_stats"] = True
        
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Calculate overall health
    passed = sum(checks.values())
    total = len(checks)
    health_score = passed / total
    
    return {
        "healthy": health_score >= 0.8,
        "score": health_score,
        "checks": checks
    }

def readiness_check():
    """Quick readiness check for load balancer."""
    try:
        mc.configure(config_path="configs/production.yaml")
        model = mc.model("main")
        _ = model.provider  # Trigger resolution
        return True
    except Exception:
        return False

if __name__ == "__main__":
    import json
    health = health_check()
    print(json.dumps(health, indent=2))
```

### Performance Monitoring

```python
# monitoring.py
import time
import logging
import model_compass as mc
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelCompassMonitor:
    """Monitor Model Compass performance and health."""
    
    def __init__(self):
        self.metrics = {
            "resolution_times": [],
            "cache_stats": {},
            "error_count": 0,
            "last_check": None
        }
    
    def collect_metrics(self):
        """Collect performance metrics."""
        try:
            # Measure resolution time
            start_time = time.time()
            model = mc.model("main")
            _ = model.provider  # Trigger resolution
            resolution_time = time.time() - start_time
            
            self.metrics["resolution_times"].append({
                "timestamp": datetime.now().isoformat(),
                "duration": resolution_time
            })
            
            # Keep only last 100 measurements
            if len(self.metrics["resolution_times"]) > 100:
                self.metrics["resolution_times"] = self.metrics["resolution_times"][-100:]
            
            # Collect cache statistics
            self.metrics["cache_stats"] = mc.get_performance_stats()
            self.metrics["last_check"] = datetime.now().isoformat()
            
            logger.info(f"Metrics collected - Resolution time: {resolution_time:.4f}s")
            
        except Exception as e:
            self.metrics["error_count"] += 1
            logger.error(f"Metrics collection failed: {e}")
    
    def get_summary(self):
        """Get performance summary."""
        if not self.metrics["resolution_times"]:
            return {"status": "no_data"}
        
        times = [m["duration"] for m in self.metrics["resolution_times"]]
        
        return {
            "status": "healthy",
            "resolution_time": {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            },
            "cache_stats": self.metrics["cache_stats"],
            "error_count": self.metrics["error_count"],
            "last_check": self.metrics["last_check"]
        }

# Usage
monitor = ModelCompassMonitor()

# Collect metrics periodically
import threading
import time

def collect_metrics_periodically():
    while True:
        monitor.collect_metrics()
        time.sleep(60)  # Collect every minute

# Start monitoring thread
monitoring_thread = threading.Thread(target=collect_metrics_periodically, daemon=True)
monitoring_thread.start()
```

### Logging Configuration

```python
# logging_config.py
import logging
import model_compass as mc

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_compass.log')
    ]
)

# Enable Model Compass verbose logging in development
if os.getenv("ENVIRONMENT") == "development":
    mc.enable_verbose_logging()

# Custom logger for Model Compass events
mc_logger = logging.getLogger("model_compass.app")

def log_model_usage(identifier, model_info):
    """Log model usage for analytics."""
    mc_logger.info(
        "Model accessed",
        extra={
            "identifier": identifier,
            "provider": model_info.get("provider"),
            "model": model_info.get("name"),
            "cost_tier": model_info.get("cost_tier"),
            "environment": os.getenv("ENVIRONMENT")
        }
    )

# Usage
model = mc.model("urn:llm:intent:chat?reasoning=high")
log_model_usage("reasoning_intent", model.to_dict())
```

## Security Best Practices

### API Key Management

```python
# secure_config.py
import os
from cryptography.fernet import Fernet

class SecureConfigManager:
    """Secure configuration manager with encrypted API keys."""
    
    def __init__(self, encryption_key=None):
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate key from environment or use default
            key = os.getenv("MODEL_COMPASS_ENCRYPTION_KEY")
            if key:
                self.cipher = Fernet(key.encode())
            else:
                self.cipher = None
    
    def decrypt_api_key(self, encrypted_key):
        """Decrypt API key."""
        if self.cipher and encrypted_key.startswith("encrypted:"):
            encrypted_data = encrypted_key[10:]  # Remove "encrypted:" prefix
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        return encrypted_key
    
    def load_secure_config(self, config_path):
        """Load configuration with encrypted API keys."""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Decrypt API keys in provider headers
        for provider_name, provider_config in config.get("providers", {}).items():
            headers = provider_config.get("headers", {})
            for header_name, header_value in headers.items():
                if isinstance(header_value, str) and "API_KEY" in header_name:
                    # Decrypt if encrypted, otherwise get from environment
                    if header_value.startswith("encrypted:"):
                        headers[header_name] = self.decrypt_api_key(header_value)
                    elif header_value.startswith("${") and header_value.endswith("}"):
                        env_var = header_value[2:-1]
                        headers[header_name] = f"Bearer {os.getenv(env_var)}"
        
        return config

# Usage
secure_manager = SecureConfigManager()
config = secure_manager.load_secure_config("configs/production.yaml")
mc.configure(config_dict=config)
```

### Network Security

```yaml
# configs/secure_production.yaml
providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 30
    headers:
      Authorization: "Bearer ${OPENAI_API_KEY}"
      User-Agent: "ModelCompass/1.0"
      X-Request-ID: "${REQUEST_ID}"
    # Add certificate verification
    verify_ssl: true
    # Add retry configuration
    max_retries: 3
    backoff_factor: 0.3

  anthropic:
    base_url: https://api.anthropic.com
    timeout: 45
    headers:
      x-api-key: "${ANTHROPIC_API_KEY}"
      User-Agent: "ModelCompass/1.0"
      X-Request-ID: "${REQUEST_ID}"
    verify_ssl: true
    max_retries: 3
    backoff_factor: 0.5
```

## Troubleshooting

### Common Issues

#### Configuration Loading Issues

```python
# troubleshoot_config.py
import model_compass as mc
from model_compass.exceptions import ConfigurationError

def diagnose_config_issues(config_path):
    """Diagnose configuration loading issues."""
    try:
        mc.configure(config_path=config_path)
        print("✓ Configuration loaded successfully")
        return True
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        print(f"Context: {e.context}")
        
        # Provide specific guidance
        if "not found" in str(e):
            print("Suggestion: Check if the configuration file exists")
        elif "Invalid YAML" in str(e):
            print("Suggestion: Validate YAML syntax")
        elif "provider" in str(e):
            print("Suggestion: Check provider configuration")
        
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

# Usage
diagnose_config_issues("configs/production.yaml")
```

#### Model Resolution Issues

```python
# troubleshoot_resolution.py
def diagnose_resolution_issues(identifier):
    """Diagnose model resolution issues."""
    try:
        model = mc.model(identifier)
        provider = model.provider  # Trigger resolution
        print(f"✓ Model resolved: {provider}/{model.name}")
        return True
    except ResolutionError as e:
        print(f"✗ Resolution error: {e}")
        print(f"Identifier: {e.identifier}")
        print(f"Suggestions: {e.context.get('suggestions', [])}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

# Usage
diagnose_resolution_issues("urn:llm:intent:chat?reasoning=high")
```

### Performance Troubleshooting

```python
# performance_troubleshoot.py
def diagnose_performance():
    """Diagnose performance issues."""
    stats = mc.get_performance_stats()
    
    print("Performance Diagnostics:")
    print("=" * 50)
    
    # Check cache hit rates
    config_cache = stats.get("configuration_cache", {})
    resolution_cache = stats.get("resolution_cache", {})
    
    config_hit_rate = config_cache.get("hit_rate", 0)
    resolution_hit_rate = resolution_cache.get("hit_rate", 0)
    
    print(f"Configuration cache hit rate: {config_hit_rate}%")
    if config_hit_rate < 50:
        print("⚠️  Low config cache hit rate - consider enabling performance optimizations")
    
    print(f"Resolution cache hit rate: {resolution_hit_rate}%")
    if resolution_hit_rate < 30:
        print("⚠️  Low resolution cache hit rate - check for frequent cache invalidation")
    
    # Check cache sizes
    config_size = config_cache.get("size", 0)
    resolution_size = resolution_cache.get("size", 0)
    
    print(f"Configuration cache size: {config_size}")
    print(f"Resolution cache size: {resolution_size}")
    
    if resolution_size == 0:
        print("⚠️  Empty resolution cache - performance optimizations may not be enabled")

# Usage
diagnose_performance()
```

This deployment guide provides comprehensive coverage of deploying Model Compass across different environments with proper security, monitoring, and troubleshooting practices.