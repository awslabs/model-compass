# Framework Integration Guide

This guide shows how to integrate Model Compass with popular LLM frameworks.

## LangChain Integration

### Basic Setup

```python
import model_compass as mc
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage

# Configure Model Compass
mc.configure(config_path="config.yaml")

# Enable performance optimizations
mc.enable_performance_optimizations()
```

### Using with LangChain LLMs

```python
# Get model configuration from Model Compass
model = mc.model("urn:llm:intent:completion?reasoning=high&cost=medium")

# Configure LangChain LLM based on provider
if model.provider == "openai":
    llm = OpenAI(
        model_name=model.name,
        openai_api_base=model.base_url,
        temperature=model.parameters.get("temperature", 0.7),
        max_tokens=model.parameters.get("max_tokens", 1000)
    )
elif model.provider == "anthropic":
    llm = ChatAnthropic(
        model=model.name,
        anthropic_api_url=model.base_url,
        temperature=model.parameters.get("temperature", 0.7),
        max_tokens=model.parameters.get("max_tokens", 1000)
    )

# Use the LLM
response = llm("What is the capital of France?")
print(response)
```

### Environment-Specific Models

```python
import os
from langchain.llms import OpenAI

# Configure Model Compass with environment-based profiles
mc.configure(config_dict={
    "profiles": {
        "development": {
            "model": "gpt-3.5-turbo",
            "parameters": {"temperature": 0.9}
        },
        "production": {
            "model": "gpt-4",
            "parameters": {"temperature": 0.1}
        }
    },
    "models": {
        "gpt-3.5-turbo": {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        },
        "gpt-4": {
            "provider": "openai", 
            "model": "gpt-4"
        }
    },
    "providers": {
        "openai": {
            "base_url": "https://api.openai.com/v1"
        }
    },
    "aliases": {
        "current": os.getenv("ENVIRONMENT", "development")
    }
})

# Get environment-appropriate model
model = mc.model("current")

# Create LangChain LLM
llm = OpenAI(
    model_name=model.name,
    openai_api_base=model.base_url,
    **model.parameters
)
```

### Chain with Model Switching

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Different models for different tasks
reasoning_model = mc.model("urn:llm:intent:analysis?reasoning=high")
fast_model = mc.model("urn:llm:intent:chat?speed=fast&cost=low")

# Create LLMs
reasoning_llm = OpenAI(
    model_name=reasoning_model.name,
    openai_api_base=reasoning_model.base_url,
    **reasoning_model.parameters
)

fast_llm = OpenAI(
    model_name=fast_model.name,
    openai_api_base=fast_model.base_url,
    **fast_model.parameters
)

# Use reasoning model for complex analysis
analysis_prompt = PromptTemplate(
    input_variables=["problem"],
    template="Analyze this problem step by step: {problem}"
)
analysis_chain = LLMChain(llm=reasoning_llm, prompt=analysis_prompt)

# Use fast model for simple responses
response_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Summarize this analysis in simple terms: {analysis}"
)
response_chain = LLMChain(llm=fast_llm, prompt=response_prompt)

# Execute chain
problem = "How can we reduce carbon emissions in transportation?"
analysis = analysis_chain.run(problem=problem)
summary = response_chain.run(analysis=analysis)

print("Analysis:", analysis)
print("Summary:", summary)
```

### AWS Bedrock Integration with LangChain

```python
import model_compass as mc
from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
import boto3

# Configure Model Compass with Bedrock models
mc.configure(config_dict={
    "providers": {
        "bedrock-us-east": {
            "type": "bedrock",
            "region": "us-east-1",
            "aws_profile": "default"
        },
        "bedrock-eu-west": {
            "type": "bedrock", 
            "region": "eu-west-1",
            "aws_profile": "production"
        }
    },
    "models": {
        "claude-opus": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-opus-20240229-v1:0",
            "metadata": {
                "reasoning": "high",
                "cost": "high",
                "speed": "slow",
                "capabilities": ["chat", "analysis", "vision"]
            },
            "parameters": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95
            }
        },
        "claude-sonnet": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0", 
            "metadata": {
                "reasoning": "high",
                "cost": "medium",
                "speed": "fast",
                "capabilities": ["chat", "analysis"]
            },
            "parameters": {
                "max_tokens": 2048,
                "temperature": 0.5,
                "top_p": 0.9
            }
        },
        "claude-haiku": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "metadata": {
                "reasoning": "medium", 
                "cost": "low",
                "speed": "fast",
                "capabilities": ["chat"]
            },
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0.3,
                "top_p": 0.8
            }
        }
    }
})

# Get Bedrock model configuration from Model Compass
def create_bedrock_llm(intent):
    """Create LangChain Bedrock LLM using Model Compass configuration."""
    config = mc.model(intent)
    
    # Create boto3 client with resolved configuration
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=config.region,
        profile_name=getattr(config, 'aws_profile', 'default')
    )
    
    # Create LangChain Bedrock LLM
    llm = Bedrock(
        client=bedrock_client,
        model_id=config.model_id,
        model_kwargs=config.parameters
    )
    
    return llm

# Use different Bedrock models for different tasks
reasoning_llm = create_bedrock_llm("urn:llm:intent:analysis?reasoning=high&cost=high")
balanced_llm = create_bedrock_llm("urn:llm:intent:analysis?reasoning=high&cost=medium")
fast_llm = create_bedrock_llm("urn:llm:intent:chat?speed=fast&cost=low")

# Create chains with different Bedrock models
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Complex analysis with Claude Opus
analysis_prompt = PromptTemplate(
    input_variables=["problem"],
    template="Provide a detailed analysis of this problem: {problem}"
)
analysis_chain = LLMChain(llm=reasoning_llm, prompt=analysis_prompt)

# Quick summary with Claude Haiku
summary_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Summarize this analysis in 2-3 sentences: {analysis}"
)
summary_chain = LLMChain(llm=fast_llm, prompt=summary_prompt)

# Execute Bedrock chain
problem = "How can AI improve healthcare outcomes?"
analysis = analysis_chain.run(problem=problem)
summary = summary_chain.run(analysis=analysis)

print("Detailed Analysis (Claude Opus):", analysis)
print("Quick Summary (Claude Haiku):", summary)
```

### Bedrock Multi-Region Fallback with LangChain

```python
from model_compass.exceptions import ResolutionError

def create_bedrock_llm_with_fallback(primary_intent, fallback_intent):
    """Create Bedrock LLM with regional fallback."""
    try:
        # Try primary region first
        config = mc.model(primary_intent)
        print(f"Using primary model: {config.model_id} in {config.region}")
    except ResolutionError:
        # Fall back to secondary region
        config = mc.model(fallback_intent)
        print(f"Falling back to: {config.model_id} in {config.region}")
    
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=config.region,
        profile_name=getattr(config, 'aws_profile', 'default')
    )
    
    return Bedrock(
        client=bedrock_client,
        model_id=config.model_id,
        model_kwargs=config.parameters
    )

# Configure multi-region Bedrock setup
mc.configure(config_dict={
    "providers": {
        "bedrock-us-east": {"region": "us-east-1", "aws_profile": "default"},
        "bedrock-us-west": {"region": "us-west-2", "aws_profile": "default"}
    },
    "models": {
        "claude-primary": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {"region": "us-east", "priority": "primary"}
        },
        "claude-fallback": {
            "provider": "bedrock-us-west", 
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {"region": "us-west", "priority": "fallback"}
        }
    }
})

# Create LLM with fallback
llm = create_bedrock_llm_with_fallback(
    primary_intent="urn:llm:intent:chat?region=us-east",
    fallback_intent="urn:llm:intent:chat?region=us-west"
)

# Use with LangChain
response = llm("What are the benefits of cloud computing?")
print(response)
```

### Bedrock Error Handling with LangChain

```python
import boto3
from botocore.exceptions import ClientError
from langchain.llms import Bedrock

def create_robust_bedrock_llm(intent, max_retries=3):
    """Create Bedrock LLM with comprehensive error handling."""
    config = mc.model(intent)
    
    for attempt in range(max_retries):
        try:
            bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=config.region,
                profile_name=getattr(config, 'aws_profile', 'default')
            )
            
            # Test the connection
            bedrock_client.list_foundation_models()
            
            llm = Bedrock(
                client=bedrock_client,
                model_id=config.model_id,
                model_kwargs=config.parameters
            )
            
            # Test model invocation
            test_response = llm("Hello")
            print(f"Successfully connected to {config.model_id} in {config.region}")
            return llm
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                print(f"Throttling detected, retrying in {2**attempt} seconds...")
                time.sleep(2**attempt)
            elif error_code == 'ValidationException':
                print(f"Model {config.model_id} not available in {config.region}")
                break
            else:
                print(f"AWS error: {error_code}")
                break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    # Fallback to different model
    print("Falling back to alternative model...")
    fallback_config = mc.model("urn:llm:intent:chat?cost=low")
    return create_bedrock_llm(fallback_config)

# Usage with error handling
llm = create_robust_bedrock_llm("urn:llm:intent:analysis?reasoning=high")
```

## DSPy Integration

### Basic Setup

```python
import model_compass as mc
import dspy

# Configure Model Compass
mc.configure(config_path="config.yaml")

# Get model configuration
model = mc.model("urn:llm:intent:chat?reasoning=high")

# Configure DSPy with Model Compass
if model.provider == "openai":
    lm = dspy.OpenAI(
        model=model.name,
        api_base=model.base_url,
        **model.parameters
    )
elif model.provider == "anthropic":
    lm = dspy.Claude(
        model=model.name,
        api_base=model.base_url,
        **model.parameters
    )

dspy.settings.configure(lm=lm)
```

### Multi-Model DSPy Pipeline

```python
import dspy

# Configure different models for different stages
mc.configure(config_dict={
    "models": {
        "fast-model": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "metadata": {"speed": "fast", "cost": "low"}
        },
        "reasoning-model": {
            "provider": "openai",
            "model": "gpt-4",
            "metadata": {"reasoning": "high", "cost": "high"}
        }
    },
    "providers": {
        "openai": {"base_url": "https://api.openai.com/v1"}
    }
})

# Get models for different tasks
fast_model = mc.model("fast-model")
reasoning_model = mc.model("reasoning-model")

# Create DSPy language models
fast_lm = dspy.OpenAI(
    model=fast_model.name,
    api_base=fast_model.base_url,
    **fast_model.parameters
)

reasoning_lm = dspy.OpenAI(
    model=reasoning_model.name,
    api_base=reasoning_model.base_url,
    **reasoning_model.parameters
)

# Define DSPy signatures
class GenerateQuery(dspy.Signature):
    """Generate a search query from a question."""
    question = dspy.InputField()
    query = dspy.OutputField()

class AnalyzeResults(dspy.Signature):
    """Analyze search results and provide an answer."""
    question = dspy.InputField()
    results = dspy.InputField()
    answer = dspy.OutputField()

# Create modules with different models
class RAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use fast model for query generation
        self.generate_query = dspy.ChainOfThought(GenerateQuery, lm=fast_lm)
        # Use reasoning model for analysis
        self.analyze = dspy.ChainOfThought(AnalyzeResults, lm=reasoning_lm)
    
    def forward(self, question):
        query = self.generate_query(question=question)
        # Simulate search results
        results = f"Search results for: {query.query}"
        answer = self.analyze(question=question, results=results)
        return answer

# Use the pipeline
rag = RAGPipeline()
result = rag("What are the benefits of renewable energy?")
print(result.answer)
```

### Profile-Based DSPy Configuration

```python
import dspy

# Configure with profiles for different environments
mc.configure(config_dict={
    "profiles": {
        "development": {
            "model": "gpt-3.5-turbo",
            "parameters": {"temperature": 0.9, "max_tokens": 500}
        },
        "testing": {
            "model": "gpt-3.5-turbo", 
            "parameters": {"temperature": 0.5, "max_tokens": 1000}
        },
        "production": {
            "model": "gpt-4",
            "parameters": {"temperature": 0.1, "max_tokens": 2000}
        }
    },
    "models": {
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4": {"provider": "openai", "model": "gpt-4"}
    },
    "providers": {
        "openai": {"base_url": "https://api.openai.com/v1"}
    }
})

def configure_dspy_for_environment(env="development"):
    """Configure DSPy for specific environment."""
    with mc.profile(env):
        model = mc.model()
        lm = dspy.OpenAI(
            model=model.name,
            api_base=model.base_url,
            **model.parameters
        )
        dspy.settings.configure(lm=lm)
        return lm

# Configure for different environments
dev_lm = configure_dspy_for_environment("development")
prod_lm = configure_dspy_for_environment("production")

# Use in DSPy modules
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)

# Test with development model
dspy.settings.configure(lm=dev_lm)
qa_dev = QAModule()
dev_answer = qa_dev("What is machine learning?")

# Deploy with production model
dspy.settings.configure(lm=prod_lm)
qa_prod = QAModule()
prod_answer = qa_prod("What is machine learning?")
```

### AWS Bedrock Integration with DSPy

```python
import model_compass as mc
import dspy
import boto3
from dspy.clients import Bedrock

# Configure Model Compass with Bedrock models for DSPy
mc.configure(config_dict={
    "providers": {
        "bedrock-us-east": {
            "type": "bedrock",
            "region": "us-east-1", 
            "aws_profile": "default"
        }
    },
    "models": {
        "claude-sonnet-dspy": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {
                "reasoning": "high",
                "cost": "medium",
                "speed": "fast",
                "framework": "dspy"
            },
            "parameters": {
                "max_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.9
            }
        },
        "titan-text-dspy": {
            "provider": "bedrock-us-east", 
            "model_id": "amazon.titan-text-express-v1",
            "metadata": {
                "reasoning": "medium",
                "cost": "low", 
                "speed": "fast",
                "framework": "dspy"
            },
            "parameters": {
                "maxTokenCount": 1024,
                "temperature": 0.5,
                "topP": 0.8
            }
        },
        "llama-dspy": {
            "provider": "bedrock-us-east",
            "model_id": "meta.llama3-70b-instruct-v1:0",
            "metadata": {
                "reasoning": "high",
                "cost": "medium",
                "speed": "medium", 
                "framework": "dspy"
            },
            "parameters": {
                "max_gen_len": 2048,
                "temperature": 0.4,
                "top_p": 0.9
            }
        }
    }
})

def create_bedrock_dspy_client(intent):
    """Create DSPy Bedrock client using Model Compass configuration."""
    config = mc.model(intent)
    
    # Create boto3 client
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=config.region,
        profile_name=getattr(config, 'aws_profile', 'default')
    )
    
    # Create DSPy Bedrock client
    lm = Bedrock(
        client=bedrock_client,
        model=config.model_id,
        **config.parameters
    )
    
    return lm

# Multi-Model DSPy Pipeline with Bedrock
class BedrockRAGPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # Use fast, cheap model for query generation
        query_model = create_bedrock_dspy_client("urn:llm:intent:chat?cost=low&speed=fast")
        
        # Use reasoning model for analysis
        analysis_model = create_bedrock_dspy_client("urn:llm:intent:analysis?reasoning=high")
        
        # Define signatures
        self.generate_query = dspy.ChainOfThought(
            "question -> search_query",
            lm=query_model
        )
        
        self.analyze_results = dspy.ChainOfThought(
            "question, search_results -> detailed_answer",
            lm=analysis_model
        )
    
    def forward(self, question):
        # Generate search query with Titan (fast/cheap)
        query_result = self.generate_query(question=question)
        
        # Simulate search results
        search_results = f"Search results for: {query_result.search_query}"
        
        # Analyze with Claude (high reasoning)
        analysis = self.analyze_results(
            question=question,
            search_results=search_results
        )
        
        return analysis

# Use the Bedrock RAG pipeline
rag = BedrockRAGPipeline()
result = rag("What are the latest developments in quantum computing?")
print(result.detailed_answer)
```

### Bedrock Model Optimization with DSPy

```python
import dspy

# Configure multiple Bedrock models for optimization
mc.configure(config_dict={
    "models": {
        "claude-opus-premium": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-opus-20240229-v1:0",
            "metadata": {"reasoning": "highest", "cost": "high", "quality": "premium"},
            "parameters": {"max_tokens": 4096, "temperature": 0.1}
        },
        "claude-sonnet-balanced": {
            "provider": "bedrock-us-east", 
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {"reasoning": "high", "cost": "medium", "quality": "good"},
            "parameters": {"max_tokens": 2048, "temperature": 0.3}
        },
        "claude-haiku-fast": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0", 
            "metadata": {"reasoning": "medium", "cost": "low", "quality": "basic"},
            "parameters": {"max_tokens": 1024, "temperature": 0.5}
        }
    }
})

class AdaptiveBedrockModule(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # Create different model clients
        self.premium_model = create_bedrock_dspy_client("urn:llm:intent:analysis?quality=premium")
        self.balanced_model = create_bedrock_dspy_client("urn:llm:intent:analysis?quality=good") 
        self.fast_model = create_bedrock_dspy_client("urn:llm:intent:chat?quality=basic")
        
        # Define adaptive signatures
        self.classify_complexity = dspy.Predict(
            "question -> complexity_level",
            lm=self.fast_model
        )
        
        self.answer_simple = dspy.ChainOfThought(
            "question -> answer",
            lm=self.fast_model
        )
        
        self.answer_complex = dspy.ChainOfThought(
            "question -> detailed_answer", 
            lm=self.balanced_model
        )
        
        self.answer_expert = dspy.ChainOfThought(
            "question -> expert_analysis",
            lm=self.premium_model
        )
    
    def forward(self, question):
        # First, classify question complexity with fast model
        complexity = self.classify_complexity(question=question)
        
        # Route to appropriate model based on complexity
        if "simple" in complexity.complexity_level.lower():
            return self.answer_simple(question=question)
        elif "complex" in complexity.complexity_level.lower():
            return self.answer_complex(question=question)
        else:
            return self.answer_expert(question=question)

# Use adaptive routing
adaptive_module = AdaptiveBedrockModule()

# Simple question -> Haiku
simple_result = adaptive_module("What is the capital of France?")

# Complex question -> Sonnet or Opus
complex_result = adaptive_module("Explain the implications of quantum entanglement for cryptography")

print("Simple:", simple_result)
print("Complex:", complex_result)
```

### Bedrock Error Handling and Fallbacks in DSPy

```python
import dspy
from model_compass.exceptions import ResolutionError
from botocore.exceptions import ClientError

class RobustBedrockModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.setup_models_with_fallbacks()
    
    def setup_models_with_fallbacks(self):
        """Setup Bedrock models with fallback chain."""
        try:
            # Primary: Claude Sonnet
            self.primary_model = create_bedrock_dspy_client(
                "urn:llm:intent:analysis?reasoning=high&cost=medium"
            )
            print("Primary model (Claude Sonnet) configured")
        except (ResolutionError, ClientError) as e:
            print(f"Primary model failed: {e}")
            self.primary_model = None
        
        try:
            # Fallback: Titan Text
            self.fallback_model = create_bedrock_dspy_client(
                "urn:llm:intent:chat?cost=low&speed=fast"
            )
            print("Fallback model (Titan) configured")
        except (ResolutionError, ClientError) as e:
            print(f"Fallback model failed: {e}")
            self.fallback_model = None
        
        # Emergency fallback: Use any available model
        if not self.primary_model and not self.fallback_model:
            try:
                emergency_config = mc.model("urn:llm:intent:chat")
                self.emergency_model = create_bedrock_dspy_client(emergency_config)
                print("Emergency model configured")
            except Exception as e:
                raise RuntimeError(f"No Bedrock models available: {e}")
    
    def get_available_model(self):
        """Get the best available model."""
        if self.primary_model:
            return self.primary_model
        elif self.fallback_model:
            return self.fallback_model
        else:
            return self.emergency_model
    
    def forward(self, question):
        model = self.get_available_model()
        
        # Create signature with available model
        answer_signature = dspy.ChainOfThought(
            "question -> answer",
            lm=model
        )
        
        try:
            return answer_signature(question=question)
        except Exception as e:
            print(f"Model invocation failed: {e}")
            # Try with different model if available
            if model == self.primary_model and self.fallback_model:
                fallback_signature = dspy.ChainOfThought(
                    "question -> answer",
                    lm=self.fallback_model
                )
                return fallback_signature(question=question)
            else:
                raise

# Usage with error handling
try:
    robust_module = RobustBedrockModule()
    result = robust_module("Explain machine learning in simple terms")
    print(result.answer)
except Exception as e:
    print(f"All models failed: {e}")
```

## Autogen Integration

### Basic Multi-Agent Setup

```python
import model_compass as mc
from autogen import AssistantAgent, UserProxyAgent

# Configure Model Compass
mc.configure(config_path="config.yaml")

# Get different models for different agent roles
reasoning_model = mc.model("urn:llm:intent:analysis?reasoning=high")
creative_model = mc.model("urn:llm:intent:chat?speed=fast&reasoning=medium")
critic_model = mc.model("urn:llm:intent:analysis?reasoning=high&cost=low")

# Create agents with different models
analyst = AssistantAgent(
    name="analyst",
    system_message="You are a data analyst. Provide detailed analysis with evidence.",
    llm_config={
        "model": reasoning_model.name,
        "api_base": reasoning_model.base_url,
        **reasoning_model.parameters
    }
)

writer = AssistantAgent(
    name="writer",
    system_message="You are a creative writer. Write engaging content.",
    llm_config={
        "model": creative_model.name,
        "api_base": creative_model.base_url,
        **creative_model.parameters
    }
)

critic = AssistantAgent(
    name="critic",
    system_message="You are a critic. Review and improve the work.",
    llm_config={
        "model": critic_model.name,
        "api_base": critic_model.base_url,
        **critic_model.parameters
    }
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3
)

# Multi-agent conversation
user_proxy.initiate_chat(
    analyst,
    message="Analyze the trend of remote work adoption post-2020."
)
```

### Environment-Aware Agent Configuration

```python
import os
from autogen import AssistantAgent, GroupChat, GroupChatManager

# Configure with environment-based model selection
mc.configure(config_dict={
    "profiles": {
        "development": {
            "model": "gpt-3.5-turbo",
            "parameters": {"temperature": 0.8}
        },
        "production": {
            "model": "gpt-4",
            "parameters": {"temperature": 0.3}
        }
    },
    "models": {
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "gpt-4": {"provider": "openai", "model": "gpt-4"}
    },
    "providers": {
        "openai": {"base_url": "https://api.openai.com/v1"}
    },
    "aliases": {
        "current": os.getenv("ENVIRONMENT", "development")
    }
})

def create_agent(name, system_message, profile="current"):
    """Create an agent with environment-appropriate model."""
    with mc.profile(profile) if profile != "current" else mc.profile("development"):
        model = mc.model("current")
        
        return AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "model": model.name,
                "api_base": model.base_url,
                **model.parameters
            }
        )

# Create agents
planner = create_agent(
    "planner",
    "You are a project planner. Break down tasks and create plans."
)

developer = create_agent(
    "developer", 
    "You are a software developer. Write code and technical solutions."
)

reviewer = create_agent(
    "reviewer",
    "You are a code reviewer. Review code for quality and best practices."
)

# Group chat
groupchat = GroupChat(
    agents=[planner, developer, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "model": mc.model("current").name,
        "api_base": mc.model("current").base_url,
        **mc.model("current").parameters
    }
)
```

### Cost-Optimized Agent Pipeline

```python
from autogen import AssistantAgent

# Configure models with cost considerations
mc.configure(config_dict={
    "models": {
        "cheap-fast": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "metadata": {"cost": "low", "speed": "fast", "reasoning": "medium"}
        },
        "expensive-smart": {
            "provider": "openai", 
            "model": "gpt-4",
            "metadata": {"cost": "high", "speed": "slow", "reasoning": "high"}
        }
    },
    "providers": {
        "openai": {"base_url": "https://api.openai.com/v1"}
    }
})

def create_cost_optimized_agents():
    """Create agents optimized for different cost/performance needs."""
    
    # Use cheap model for initial processing
    preprocessor = AssistantAgent(
        name="preprocessor",
        system_message="Clean and structure the input data.",
        llm_config={
            **mc.model("urn:llm:intent:chat?cost=low&speed=fast").to_dict()
        }
    )
    
    # Use expensive model only for complex reasoning
    analyzer = AssistantAgent(
        name="analyzer", 
        system_message="Perform deep analysis requiring complex reasoning.",
        llm_config={
            **mc.model("urn:llm:intent:analysis?reasoning=high").to_dict()
        }
    )
    
    # Use cheap model for final formatting
    formatter = AssistantAgent(
        name="formatter",
        system_message="Format the results for presentation.",
        llm_config={
            **mc.model("urn:llm:intent:chat?cost=low&speed=fast").to_dict()
        }
    )
    
    return preprocessor, analyzer, formatter

# Create cost-optimized pipeline
prep, analyze, format_agent = create_cost_optimized_agents()

# Sequential processing to minimize costs
def process_with_cost_optimization(data):
    # Step 1: Cheap preprocessing
    cleaned = prep.generate_reply([{"role": "user", "content": f"Clean this data: {data}"}])
    
    # Step 2: Expensive analysis (only when needed)
    analysis = analyze.generate_reply([{"role": "user", "content": f"Analyze: {cleaned}"}])
    
    # Step 3: Cheap formatting
    result = format_agent.generate_reply([{"role": "user", "content": f"Format: {analysis}"}])
    
    return result

# Usage
result = process_with_cost_optimization("Raw data that needs processing...")
```

### AWS Bedrock Integration with Autogen

```python
import model_compass as mc
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import boto3

# Configure Model Compass with Bedrock models for Autogen
mc.configure(config_dict={
    "providers": {
        "bedrock-us-east": {
            "type": "bedrock",
            "region": "us-east-1",
            "aws_profile": "default"
        },
        "bedrock-eu-west": {
            "type": "bedrock", 
            "region": "eu-west-1",
            "aws_profile": "production"
        }
    },
    "models": {
        "claude-opus-agent": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-opus-20240229-v1:0",
            "metadata": {
                "reasoning": "highest",
                "cost": "high",
                "role": "analyst",
                "capabilities": ["analysis", "research", "complex_reasoning"]
            },
            "parameters": {
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.9
            }
        },
        "claude-sonnet-agent": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {
                "reasoning": "high",
                "cost": "medium", 
                "role": "coordinator",
                "capabilities": ["planning", "coordination", "balanced_reasoning"]
            },
            "parameters": {
                "max_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.9
            }
        },
        "claude-haiku-agent": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "metadata": {
                "reasoning": "medium",
                "cost": "low",
                "role": "executor", 
                "capabilities": ["quick_tasks", "formatting", "simple_responses"]
            },
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0.5,
                "top_p": 0.8
            }
        },
        "llama-agent": {
            "provider": "bedrock-us-east",
            "model_id": "meta.llama3-70b-instruct-v1:0",
            "metadata": {
                "reasoning": "high",
                "cost": "medium",
                "role": "specialist",
                "capabilities": ["technical_analysis", "code_review"]
            },
            "parameters": {
                "max_gen_len": 2048,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
    }
})

def create_bedrock_agent_config(intent):
    """Create Autogen agent config using Model Compass Bedrock configuration."""
    config = mc.model(intent)
    
    # Convert Model Compass config to Autogen format
    llm_config = {
        "config_list": [{
            "model": config.model_id,
            "api_type": "bedrock",
            "aws_region": config.region,
            "aws_profile": getattr(config, 'aws_profile', 'default'),
            **config.parameters
        }]
    }
    
    return llm_config

# Create specialized Bedrock agents
def create_bedrock_research_team():
    """Create a research team using different Bedrock models."""
    
    # Senior analyst with Claude Opus (highest reasoning)
    senior_analyst = AssistantAgent(
        name="senior_analyst",
        system_message="""You are a senior research analyst with deep expertise. 
        Provide comprehensive analysis with detailed reasoning and evidence.
        Focus on identifying key insights and implications.""",
        llm_config=create_bedrock_agent_config("urn:llm:intent:analysis?reasoning=highest&role=analyst")
    )
    
    # Research coordinator with Claude Sonnet (balanced)
    coordinator = AssistantAgent(
        name="coordinator",
        system_message="""You are a research coordinator. 
        Organize research tasks, synthesize findings, and ensure comprehensive coverage.
        Coordinate between team members and maintain research quality.""",
        llm_config=create_bedrock_agent_config("urn:llm:intent:planning?reasoning=high&role=coordinator")
    )
    
    # Technical specialist with Llama
    tech_specialist = AssistantAgent(
        name="tech_specialist", 
        system_message="""You are a technical specialist.
        Focus on technical aspects, implementation details, and code analysis.
        Provide technical insights and validate technical claims.""",
        llm_config=create_bedrock_agent_config("urn:llm:intent:analysis?reasoning=high&role=specialist")
    )
    
    # Quick responder with Claude Haiku (fast/cheap)
    quick_responder = AssistantAgent(
        name="quick_responder",
        system_message="""You are a quick responder for simple tasks.
        Handle formatting, simple questions, and quick clarifications.
        Keep responses concise and focused.""",
        llm_config=create_bedrock_agent_config("urn:llm:intent:chat?cost=low&role=executor")
    )
    
    return senior_analyst, coordinator, tech_specialist, quick_responder

# Create the research team
analyst, coordinator, specialist, responder = create_bedrock_research_team()

# Create user proxy
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False
)

# Multi-agent research workflow
def conduct_bedrock_research(topic):
    """Conduct research using Bedrock-powered agents."""
    
    # Step 1: Coordinator plans the research
    user_proxy.initiate_chat(
        coordinator,
        message=f"Plan a comprehensive research approach for: {topic}"
    )
    
    # Step 2: Senior analyst provides deep analysis
    user_proxy.initiate_chat(
        analyst,
        message=f"Provide detailed analysis of: {topic}"
    )
    
    # Step 3: Technical specialist adds technical perspective
    user_proxy.initiate_chat(
        specialist,
        message=f"Analyze technical aspects of: {topic}"
    )
    
    # Step 4: Quick responder summarizes findings
    user_proxy.initiate_chat(
        responder,
        message="Summarize the key findings from our research"
    )

# Usage
conduct_bedrock_research("Impact of AI on software development practices")
```

### Bedrock Multi-Region Agent Setup

```python
from autogen import AssistantAgent, GroupChat, GroupChatManager

# Configure multi-region Bedrock setup for high availability
mc.configure(config_dict={
    "providers": {
        "bedrock-primary": {
            "type": "bedrock",
            "region": "us-east-1",
            "aws_profile": "default",
            "priority": "primary"
        },
        "bedrock-backup": {
            "type": "bedrock",
            "region": "us-west-2", 
            "aws_profile": "default",
            "priority": "backup"
        }
    },
    "models": {
        "claude-primary": {
            "provider": "bedrock-primary",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {"region": "primary", "availability": "high"}
        },
        "claude-backup": {
            "provider": "bedrock-backup",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0", 
            "metadata": {"region": "backup", "availability": "fallback"}
        }
    }
})

class ResilientBedrockAgent:
    """Agent wrapper with automatic regional fallback."""
    
    def __init__(self, name, system_message, primary_intent, backup_intent):
        self.name = name
        self.system_message = system_message
        self.primary_intent = primary_intent
        self.backup_intent = backup_intent
        self.current_agent = self._create_agent(primary_intent, is_primary=True)
    
    def _create_agent(self, intent, is_primary=True):
        """Create agent with specified intent."""
        try:
            config = mc.model(intent)
            region_type = "primary" if is_primary else "backup"
            print(f"Creating {self.name} agent in {region_type} region: {config.region}")
            
            return AssistantAgent(
                name=f"{self.name}_{region_type}",
                system_message=self.system_message,
                llm_config=create_bedrock_agent_config(intent)
            )
        except Exception as e:
            print(f"Failed to create {region_type} agent: {e}")
            if is_primary:
                # Try backup region
                return self._create_agent(self.backup_intent, is_primary=False)
            else:
                raise RuntimeError(f"Both regions failed for {self.name}")
    
    def generate_reply(self, messages):
        """Generate reply with automatic fallback."""
        try:
            return self.current_agent.generate_reply(messages)
        except Exception as e:
            print(f"Primary agent failed: {e}, switching to backup")
            self.current_agent = self._create_agent(self.backup_intent, is_primary=False)
            return self.current_agent.generate_reply(messages)

# Create resilient agents
resilient_planner = ResilientBedrockAgent(
    name="planner",
    system_message="You are a strategic planner. Create detailed project plans.",
    primary_intent="urn:llm:intent:planning?region=primary",
    backup_intent="urn:llm:intent:planning?region=backup"
)

resilient_analyst = ResilientBedrockAgent(
    name="analyst", 
    system_message="You are a data analyst. Analyze data and provide insights.",
    primary_intent="urn:llm:intent:analysis?region=primary",
    backup_intent="urn:llm:intent:analysis?region=backup"
)

# Use resilient agents
planning_result = resilient_planner.generate_reply([
    {"role": "user", "content": "Create a plan for implementing AI governance"}
])

analysis_result = resilient_analyst.generate_reply([
    {"role": "user", "content": "Analyze the risks and benefits of the proposed plan"}
])
```

### Bedrock Cost-Optimized Agent Workflow

```python
from autogen import AssistantAgent

# Configure Bedrock models with cost optimization
mc.configure(config_dict={
    "models": {
        "bedrock-premium": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-opus-20240229-v1:0",
            "metadata": {"cost": "high", "quality": "premium", "use_case": "critical_analysis"},
            "parameters": {"max_tokens": 4096, "temperature": 0.1}
        },
        "bedrock-balanced": {
            "provider": "bedrock-us-east",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {"cost": "medium", "quality": "high", "use_case": "general_purpose"},
            "parameters": {"max_tokens": 2048, "temperature": 0.3}
        },
        "bedrock-economy": {
            "provider": "bedrock-us-east", 
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "metadata": {"cost": "low", "quality": "good", "use_case": "simple_tasks"},
            "parameters": {"max_tokens": 1024, "temperature": 0.5}
        }
    }
})

class CostOptimizedBedrockWorkflow:
    """Workflow that optimizes costs by using appropriate models for each task."""
    
    def __init__(self):
        self.setup_agents()
        self.cost_tracker = {"premium": 0, "balanced": 0, "economy": 0}
    
    def setup_agents(self):
        """Setup agents with different cost tiers."""
        
        # Premium agent for critical analysis
        self.premium_agent = AssistantAgent(
            name="premium_analyst",
            system_message="You are a premium analyst. Provide the highest quality analysis.",
            llm_config=create_bedrock_agent_config("urn:llm:intent:analysis?quality=premium")
        )
        
        # Balanced agent for general tasks
        self.balanced_agent = AssistantAgent(
            name="balanced_worker",
            system_message="You are a general purpose agent. Handle most tasks efficiently.",
            llm_config=create_bedrock_agent_config("urn:llm:intent:chat?quality=high")
        )
        
        # Economy agent for simple tasks
        self.economy_agent = AssistantAgent(
            name="economy_helper",
            system_message="You are an economy agent. Handle simple tasks quickly.",
            llm_config=create_bedrock_agent_config("urn:llm:intent:chat?quality=good")
        )
    
    def classify_task_complexity(self, task):
        """Classify task complexity to determine appropriate agent."""
        # Simple heuristics for demo - in practice, use more sophisticated classification
        if any(word in task.lower() for word in ["critical", "complex", "detailed", "comprehensive"]):
            return "premium"
        elif any(word in task.lower() for word in ["analyze", "evaluate", "compare", "research"]):
            return "balanced"
        else:
            return "economy"
    
    def execute_task(self, task):
        """Execute task with cost-optimized agent selection."""
        complexity = self.classify_task_complexity(task)
        self.cost_tracker[complexity] += 1
        
        if complexity == "premium":
            agent = self.premium_agent
            print(f"Using premium agent (Claude Opus) for: {task[:50]}...")
        elif complexity == "balanced":
            agent = self.balanced_agent
            print(f"Using balanced agent (Claude Sonnet) for: {task[:50]}...")
        else:
            agent = self.economy_agent
            print(f"Using economy agent (Claude Haiku) for: {task[:50]}...")
        
        return agent.generate_reply([{"role": "user", "content": task}])
    
    def get_cost_summary(self):
        """Get summary of agent usage for cost tracking."""
        total_tasks = sum(self.cost_tracker.values())
        if total_tasks == 0:
            return "No tasks executed"
        
        summary = f"Cost Summary (Total tasks: {total_tasks}):\n"
        for tier, count in self.cost_tracker.items():
            percentage = (count / total_tasks) * 100
            summary += f"  {tier.capitalize()}: {count} tasks ({percentage:.1f}%)\n"
        
        return summary

# Usage example
workflow = CostOptimizedBedrockWorkflow()

# Execute different types of tasks
tasks = [
    "Format this list of items",  # Economy
    "Analyze market trends for Q4",  # Balanced  
    "Provide comprehensive analysis of AI safety risks",  # Premium
    "Summarize this document",  # Economy
    "Compare different cloud architectures"  # Balanced
]

results = []
for task in tasks:
    result = workflow.execute_task(task)
    results.append(result)

# Check cost optimization
print(workflow.get_cost_summary())
```

### Bedrock Error Handling and Monitoring for Autogen

```python
import time
import logging
from autogen import AssistantAgent
from botocore.exceptions import ClientError, BotoCoreError

# Setup logging for Bedrock operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredBedrockAgent:
    """Bedrock agent with comprehensive error handling and monitoring."""
    
    def __init__(self, name, system_message, intent, max_retries=3):
        self.name = name
        self.system_message = system_message
        self.intent = intent
        self.max_retries = max_retries
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "throttling_events": 0,
            "fallback_usage": 0,
            "average_response_time": 0
        }
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create Autogen agent with error handling."""
        try:
            config = mc.model(self.intent)
            logger.info(f"Creating {self.name} with model {config.model_id} in {config.region}")
            
            return AssistantAgent(
                name=self.name,
                system_message=self.system_message,
                llm_config=create_bedrock_agent_config(self.intent)
            )
        except Exception as e:
            logger.error(f"Failed to create agent {self.name}: {e}")
            # Create fallback agent
            fallback_config = mc.model("urn:llm:intent:chat?cost=low")
            self.metrics["fallback_usage"] += 1
            return AssistantAgent(
                name=f"{self.name}_fallback",
                system_message=self.system_message,
                llm_config=create_bedrock_agent_config(fallback_config)
            )
    
    def generate_reply_with_monitoring(self, messages):
        """Generate reply with comprehensive monitoring and error handling."""
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self.agent.generate_reply(messages)
                
                # Update success metrics
                end_time = time.time()
                response_time = end_time - start_time
                self.metrics["successful_requests"] += 1
                self._update_average_response_time(response_time)
                
                logger.info(f"{self.name} responded successfully in {response_time:.2f}s")
                return response
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                logger.warning(f"{self.name} AWS error on attempt {attempt + 1}: {error_code}")
                
                if error_code == 'ThrottlingException':
                    self.metrics["throttling_events"] += 1
                    wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                    logger.info(f"Throttling detected, waiting {wait_time:.1f}s before retry")
                    time.sleep(wait_time)
                    continue
                elif error_code in ['ValidationException', 'ResourceNotFoundException']:
                    logger.error(f"Model validation error: {e}")
                    break  # Don't retry validation errors
                else:
                    logger.error(f"Unexpected AWS error: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    break
                    
            except BotoCoreError as e:
                logger.error(f"{self.name} Boto core error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
                
            except Exception as e:
                logger.error(f"{self.name} unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                break
        
        # All retries failed
        self.metrics["failed_requests"] += 1
        logger.error(f"{self.name} failed after {self.max_retries} attempts")
        
        # Return error response
        return {
            "role": "assistant",
            "content": f"I apologize, but I'm experiencing technical difficulties. Please try again later."
        }
    
    def _update_average_response_time(self, new_time):
        """Update rolling average response time."""
        current_avg = self.metrics["average_response_time"]
        successful_requests = self.metrics["successful_requests"]
        
        if successful_requests == 1:
            self.metrics["average_response_time"] = new_time
        else:
            # Rolling average
            self.metrics["average_response_time"] = (
                (current_avg * (successful_requests - 1) + new_time) / successful_requests
            )
    
    def get_health_metrics(self):
        """Get agent health and performance metrics."""
        total = self.metrics["total_requests"]
        if total == 0:
            return "No requests processed yet"
        
        success_rate = (self.metrics["successful_requests"] / total) * 100
        
        return {
            "agent_name": self.name,
            "total_requests": total,
            "success_rate": f"{success_rate:.1f}%",
            "failed_requests": self.metrics["failed_requests"],
            "throttling_events": self.metrics["throttling_events"],
            "fallback_usage": self.metrics["fallback_usage"],
            "average_response_time": f"{self.metrics['average_response_time']:.2f}s"
        }

# Create monitored Bedrock agents
monitored_analyst = MonitoredBedrockAgent(
    name="monitored_analyst",
    system_message="You are a data analyst with comprehensive monitoring.",
    intent="urn:llm:intent:analysis?reasoning=high"
)

monitored_writer = MonitoredBedrockAgent(
    name="monitored_writer", 
    system_message="You are a content writer with error handling.",
    intent="urn:llm:intent:chat?speed=fast"
)

# Usage with monitoring
def run_monitored_workflow():
    """Run workflow with comprehensive monitoring."""
    
    # Test different scenarios
    test_messages = [
        [{"role": "user", "content": "Analyze the benefits of cloud computing"}],
        [{"role": "user", "content": "Write a brief summary of AI trends"}],
        [{"role": "user", "content": "What are the key considerations for data privacy?"}]
    ]
    
    for i, messages in enumerate(test_messages):
        print(f"\n--- Test {i+1} ---")
        
        # Analyst response
        analyst_response = monitored_analyst.generate_reply_with_monitoring(messages)
        print(f"Analyst: {analyst_response}")
        
        # Writer response  
        writer_response = monitored_writer.generate_reply_with_monitoring(messages)
        print(f"Writer: {writer_response}")
    
    # Print health metrics
    print("\n--- Health Metrics ---")
    print("Analyst Metrics:", monitored_analyst.get_health_metrics())
    print("Writer Metrics:", monitored_writer.get_health_metrics())

# Run the monitored workflow
run_monitored_workflow()
```

## Best Practices for Framework Integration

### 1. Model Selection Strategy

```python
# Define clear model selection criteria
def select_model_for_task(task_type, performance_req="medium", cost_limit="medium"):
    """Select optimal model based on task requirements."""
    
    if task_type == "reasoning":
        return mc.model(f"urn:llm:intent:analysis?reasoning=high&cost={cost_limit}")
    elif task_type == "creative":
        return mc.model(f"urn:llm:intent:chat?reasoning={performance_req}&speed=fast")
    elif task_type == "summarization":
        return mc.model(f"urn:llm:intent:completion?speed=fast&cost=low")
    else:
        return mc.model(f"urn:llm:intent:chat?reasoning={performance_req}")

# Use in framework
reasoning_model = select_model_for_task("reasoning", cost_limit="high")
creative_model = select_model_for_task("creative", performance_req="high")
```

### 2. Error Handling and Fallbacks

```python
from model_compass.exceptions import ResolutionError

def get_model_with_fallback(primary_intent, fallback_model="gpt-3.5-turbo"):
    """Get model with fallback strategy."""
    try:
        return mc.model(primary_intent)
    except ResolutionError:
        print(f"Primary model not available, falling back to {fallback_model}")
        return mc.model(fallback_model)

# Use with frameworks
model = get_model_with_fallback(
    "urn:llm:intent:chat?reasoning=high&deployment=local",
    fallback_model="gpt-3.5-turbo"
)
```

### Bedrock-Specific Error Handling and Fallback Patterns

```python
import boto3
import time
from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
from model_compass.exceptions import ResolutionError

class BedrockErrorHandler:
    """Comprehensive error handling for Bedrock integrations."""
    
    def __init__(self):
        self.retry_config = {
            'ThrottlingException': {'max_retries': 5, 'base_delay': 1, 'backoff_multiplier': 2},
            'ServiceQuotaExceededException': {'max_retries': 3, 'base_delay': 5, 'backoff_multiplier': 2},
            'InternalServerException': {'max_retries': 3, 'base_delay': 2, 'backoff_multiplier': 1.5},
            'ValidationException': {'max_retries': 0, 'base_delay': 0, 'backoff_multiplier': 1},
            'ResourceNotFoundException': {'max_retries': 0, 'base_delay': 0, 'backoff_multiplier': 1}
        }
    
    def handle_bedrock_error(self, error, attempt=0):
        """Handle specific Bedrock errors with appropriate retry logic."""
        if isinstance(error, ClientError):
            error_code = error.response['Error']['Code']
            error_message = error.response['Error']['Message']
            
            print(f"Bedrock error: {error_code} - {error_message}")
            
            if error_code in self.retry_config:
                config = self.retry_config[error_code]
                if attempt < config['max_retries']:
                    delay = config['base_delay'] * (config['backoff_multiplier'] ** attempt)
                    print(f"Retrying in {delay} seconds... (attempt {attempt + 1}/{config['max_retries']})")
                    time.sleep(delay)
                    return True  # Should retry
                else:
                    print(f"Max retries exceeded for {error_code}")
                    return False  # Should not retry
            else:
                print(f"Unknown error code: {error_code}")
                return False
        
        elif isinstance(error, NoCredentialsError):
            print("AWS credentials not found. Please configure your credentials.")
            return False
        
        elif isinstance(error, BotoCoreError):
            print(f"Boto core error: {error}")
            return attempt < 2  # Retry up to 2 times for boto core errors
        
        else:
            print(f"Unexpected error: {error}")
            return False

def create_resilient_bedrock_config(primary_intent, fallback_chain=None):
    """Create Bedrock configuration with comprehensive fallback strategy."""
    
    if fallback_chain is None:
        fallback_chain = [
            "urn:llm:intent:chat?provider=bedrock&region=us-east-1",
            "urn:llm:intent:chat?provider=bedrock&region=us-west-2", 
            "urn:llm:intent:chat?provider=bedrock&region=eu-west-1",
            "urn:llm:intent:chat?provider=openai"  # Final fallback to OpenAI
        ]
    
    error_handler = BedrockErrorHandler()
    
    # Try primary intent first
    for intent in [primary_intent] + fallback_chain:
        try:
            config = mc.model(intent)
            
            # Test Bedrock connection if it's a Bedrock model
            if hasattr(config, 'region') and config.provider == 'bedrock':
                test_client = boto3.client(
                    'bedrock-runtime',
                    region_name=config.region,
                    profile_name=getattr(config, 'aws_profile', 'default')
                )
                
                # Test with a simple list operation
                test_client.list_foundation_models()
                print(f"Successfully connected to Bedrock in {config.region}")
            
            return config
            
        except ClientError as e:
            if not error_handler.handle_bedrock_error(e):
                print(f"Skipping to next fallback option due to: {e}")
                continue
        except ResolutionError as e:
            print(f"Model resolution failed for {intent}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error with {intent}: {e}")
            continue
    
    raise RuntimeError("All fallback options exhausted")

def bedrock_multi_region_fallback():
    """Configure multi-region Bedrock fallback with Model Compass."""
    
    mc.configure(config_dict={
        "providers": {
            "bedrock-us-east": {
                "type": "bedrock",
                "region": "us-east-1",
                "aws_profile": "default",
                "priority": 1
            },
            "bedrock-us-west": {
                "type": "bedrock", 
                "region": "us-west-2",
                "aws_profile": "default",
                "priority": 2
            },
            "bedrock-eu-west": {
                "type": "bedrock",
                "region": "eu-west-1", 
                "aws_profile": "default",
                "priority": 3
            }
        },
        "models": {
            "claude-primary": {
                "provider": "bedrock-us-east",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "metadata": {"region_priority": 1}
            },
            "claude-west": {
                "provider": "bedrock-us-west",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0", 
                "metadata": {"region_priority": 2}
            },
            "claude-eu": {
                "provider": "bedrock-eu-west",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "metadata": {"region_priority": 3}
            }
        }
    })
    
    # Create fallback chain based on region priorities
    fallback_chain = [
        "urn:llm:intent:chat?region_priority=1",
        "urn:llm:intent:chat?region_priority=2", 
        "urn:llm:intent:chat?region_priority=3"
    ]
    
    return create_resilient_bedrock_config(
        "urn:llm:intent:chat?reasoning=high",
        fallback_chain
    )

def bedrock_cost_aware_fallback():
    """Implement cost-aware fallback strategy for Bedrock models."""
    
    mc.configure(config_dict={
        "models": {
            "claude-opus-premium": {
                "provider": "bedrock-us-east",
                "model_id": "anthropic.claude-3-opus-20240229-v1:0",
                "metadata": {"cost": "high", "quality": "premium", "fallback_priority": 3}
            },
            "claude-sonnet-balanced": {
                "provider": "bedrock-us-east",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "metadata": {"cost": "medium", "quality": "high", "fallback_priority": 2}
            },
            "claude-haiku-economy": {
                "provider": "bedrock-us-east", 
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "metadata": {"cost": "low", "quality": "good", "fallback_priority": 1}
            }
        }
    })
    
    # Cost-aware fallback: start with premium, fall back to cheaper options
    cost_fallback_chain = [
        "urn:llm:intent:analysis?quality=premium",
        "urn:llm:intent:analysis?quality=high",
        "urn:llm:intent:chat?quality=good"
    ]
    
    return create_resilient_bedrock_config(
        "urn:llm:intent:analysis?reasoning=high",
        cost_fallback_chain
    )

# Usage examples
try:
    # Multi-region fallback
    config = bedrock_multi_region_fallback()
    print(f"Using model: {config.model_id} in region: {config.region}")
    
    # Cost-aware fallback
    cost_config = bedrock_cost_aware_fallback()
    print(f"Cost-optimized model: {cost_config.model_id}")
    
except RuntimeError as e:
    print(f"All fallback strategies failed: {e}")

# Framework integration with Bedrock error handling
def create_framework_client_with_bedrock_fallback(framework, intent):
    """Create framework client with Bedrock error handling."""
    
    config = create_resilient_bedrock_config(intent)
    
    if framework == "langchain":
        from langchain.llms import Bedrock
        client = boto3.client(
            'bedrock-runtime',
            region_name=config.region,
            profile_name=getattr(config, 'aws_profile', 'default')
        )
        return Bedrock(client=client, model_id=config.model_id, model_kwargs=config.parameters)
    
    elif framework == "dspy":
        import dspy
        from dspy.clients import Bedrock
        client = boto3.client(
            'bedrock-runtime', 
            region_name=config.region,
            profile_name=getattr(config, 'aws_profile', 'default')
        )
        return Bedrock(client=client, model=config.model_id, **config.parameters)
    
    elif framework == "autogen":
        return {
            "config_list": [{
                "model": config.model_id,
                "api_type": "bedrock",
                "aws_region": config.region,
                "aws_profile": getattr(config, 'aws_profile', 'default'),
                **config.parameters
            }]
        }
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")

# Usage with different frameworks
langchain_llm = create_framework_client_with_bedrock_fallback(
    "langchain", 
    "urn:llm:intent:analysis?reasoning=high"
)

dspy_client = create_framework_client_with_bedrock_fallback(
    "dspy",
    "urn:llm:intent:chat?speed=fast"
)

autogen_config = create_framework_client_with_bedrock_fallback(
    "autogen",
    "urn:llm:intent:analysis?reasoning=high"
)
```

### 3. Performance Monitoring

```python
import time

def monitor_model_performance():
    """Monitor model performance and cache efficiency."""
    stats = mc.get_performance_stats()
    
    print("Model Compass Performance Stats:")
    print(f"Config cache hit rate: {stats['configuration_cache']['hit_rate']}%")
    print(f"Resolution cache hit rate: {stats['resolution_cache']['hit_rate']}%")
    print(f"Intent scoring cache size: {stats['intent_scoring_cache']['size']}")
    
    return stats

# Monitor performance periodically
def timed_model_access(identifier):
    """Time model access for performance monitoring."""
    start_time = time.time()
    model = mc.model(identifier)
    _ = model.provider  # Trigger resolution
    end_time = time.time()
    
    print(f"Model resolution took {end_time - start_time:.4f} seconds")
    return model

# Usage
model = timed_model_access("urn:llm:intent:chat?reasoning=high")
stats = monitor_model_performance()
```

### 4. Configuration Management

```python
import json
from pathlib import Path

def load_framework_config(framework_name, environment="development"):
    """Load framework-specific configuration."""
    
    config_file = Path(f"configs/{framework_name}_{environment}.yaml")
    if config_file.exists():
        mc.configure(config_path=str(config_file))
    else:
        # Fallback to default configuration
        mc.configure(config_path="config.yaml")
    
    # Enable optimizations for production
    if environment == "production":
        mc.enable_performance_optimizations()

# Framework-specific initialization
def initialize_langchain(environment="development"):
    load_framework_config("langchain", environment)
    return mc.model("urn:llm:intent:chat?reasoning=medium")

def initialize_dspy(environment="development"):
    load_framework_config("dspy", environment)
    model = mc.model("urn:llm:intent:completion?speed=fast")
    
    import dspy
    lm = dspy.OpenAI(
        model=model.name,
        api_base=model.base_url,
        **model.parameters
    )
    dspy.settings.configure(lm=lm)
    return lm

def initialize_autogen(environment="development"):
    load_framework_config("autogen", environment)
    return {
        "model": mc.model("urn:llm:intent:chat?reasoning=high").name,
        "api_base": mc.model("urn:llm:intent:chat?reasoning=high").base_url,
        **mc.model("urn:llm:intent:chat?reasoning=high").parameters
    }
```

This integration guide provides comprehensive examples for using Model Compass with popular LLM frameworks, showing how to leverage its features for optimal model selection, cost optimization, and performance monitoring.